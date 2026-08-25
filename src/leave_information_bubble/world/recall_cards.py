"""MemoryBundle rendering helpers for the recall facade (strict leaf)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from typing import Any

from .contracts import MemoryBundle, RecallClock
from .recall_common import _rows
from .store import WorldStore

_SOURCE_KIND_SUMMARY_CAP = 8
_SOURCE_KIND_MAX_CHARS = 80
_LIMITATIONS_CARD_MAX_ITEMS = 8
_LIMITATIONS_ITEM_MAX_CHARS = 160
_LIMITATIONS_CARD_MAX_CHARS = 640
_LIMITATIONS_TRUNCATED = "limitations_truncated"
_SOURCE_FLIP_URI_CAP = 8
_SOURCE_FLIP_URI_MAX_CHARS = 240


def _bundle(
    *,
    store: WorldStore,
    anchors: list[dict[str, Any]] | None = None,
    assertions: list[dict[str, Any]] | None = None,
    neighbors: list[dict[str, Any]] | None = None,
    inquiries: list[dict[str, Any]] | None = None,
    paths: list[list[str]] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    reasons: list[str],
    evidence_limit: int,
    moment: str | None = None,
    clock: RecallClock = RecallClock.KNOWLEDGE,
    truncated: bool = False,
) -> MemoryBundle:
    assertion_rows = assertions or []
    with store.read_connection() as connection:
        evidence, evidence_truncated = _evidence(
            connection,
            [row["id"] for row in assertion_rows],
            limit=evidence_limit,
            moment=moment,
            clock=clock,
        )
        basis_summaries = _basis_summaries(
            connection,
            [row["id"] for row in assertion_rows],
            moment=moment,
            clock=clock,
        )
        object_ids = [row["id"] for row in [*(anchors or []), *(neighbors or [])]]
        inquiry_ids = [row["id"] for row in (inquiries or [])]
        observation_candidates, candidates_truncated = _candidate_observations(
            connection,
            object_ids,
            inquiry_ids,
            limit=evidence_limit,
            moment=moment,
            clock=clock,
        )
        assertion_cards = _assertion_cards(
            connection, assertion_rows, basis_summaries, moment=moment, clock=clock
        )
    return MemoryBundle(
        anchor_objects=anchors or [],
        assertions=assertion_cards,
        neighboring_objects=neighbors or [],
        inquiries=inquiries or [],
        evidence_refs=evidence,
        candidate_observation_refs=observation_candidates,
        paths=paths or [],
        reasons=reasons,
        candidates=candidates or [],
        truncated=(
            truncated
            or evidence_truncated
            or candidates_truncated
            or any(bool(summary.get("refs_truncated")) for summary in basis_summaries.values())
        ),
    )


def _assertion_cards(
    connection: sqlite3.Connection,
    assertions: list[dict[str, Any]],
    basis_summaries: dict[str, dict[str, Any]],
    *,
    moment: str | None,
    clock: RecallClock,
) -> list[dict[str, Any]]:
    """Render storage assertions as compact Agent-facing cognition cards.

    Cards always carry the bounded ``qualifiers`` map (parsed from
    ``qualifiers_json``; pre-v10 rows with a NULL column render as ``{}``).
    Cards also carry two optional correction signals (spec: 查询时修正信号). When
    the assertion actively supersedes an older one (its row has
    ``supersedes_id``) the card adds ``supersedes`` with the retired id and
    the stamp on the superseding row (``superseded_at``, null for pre-v7
    data). And when any of the card's evidence sources has superseded
    assertions citing it, ``source_flip_flops`` maps each such source_uri to
    how many of its claims were retired — a flip-flop signal for source
    reliability. Both fields are absent when the signal does not apply.
    """
    object_ids = list(
        dict.fromkeys(
            identifier
            for row in assertions
            for identifier in (row["subject_id"], row.get("object_id"))
            if identifier
        )
    )
    placeholders = ", ".join("?" for _ in object_ids)
    names = (
        {
            row["id"]: row["canonical_name"]
            for row in _rows(
                connection.execute(
                    f"SELECT id, canonical_name FROM objects WHERE id IN ({placeholders})",
                    object_ids,
                ).fetchall()
            )
        }
        if object_ids
        else {}
    )
    card_sources, source_truncations = _card_source_uris(
        connection, assertions, moment=moment, clock=clock
    )
    flip_flops = _source_flip_flops(connection, card_sources, moment=moment, clock=clock)
    cards: list[dict[str, Any]] = []
    for row in assertions:
        literal = json.loads(row["literal_json"]) if row["literal_json"] else None
        card: dict[str, Any] = {
            "id": row["id"],
            "subject": {"id": row["subject_id"], "canonical_name": names.get(row["subject_id"], "")},
            "predicate": row["predicate"],
            "object": (
                {"id": row["object_id"], "canonical_name": names.get(row["object_id"], "")}
                if row["object_id"]
                else None
            ),
            "literal": literal,
            "epistemic_role": row["epistemic_role"],
            "confidence": row["confidence"],
            "qualifiers": json.loads(row["qualifiers_json"]) if row["qualifiers_json"] else {},
            "event_time_start": row["event_time_start"],
            "event_time_end": row["event_time_end"],
            "basis_summary": basis_summaries.get(row["id"], _empty_basis_summary()),
        }
        supersedes_id = row.get("supersedes_id")
        if supersedes_id:
            card["supersedes"] = {
                "supersedes_id": supersedes_id,
                "superseded_at": row.get("superseded_at"),
            }
        card_flip_flops = {
            _display_source_uri(uri)[0]: flip_flops[uri]
            for uri in card_sources.get(row["id"], [])
            if flip_flops.get(uri, 0) > 0
        }
        source_flip_flops_truncated = source_truncations.get(row["id"], False) or any(
            _display_source_uri(uri)[1]
            for uri in card_sources.get(row["id"], [])
            if flip_flops.get(uri, 0) > 0
        )
        if card_flip_flops:
            card["source_flip_flops"] = card_flip_flops
        card["source_flip_flops_truncated"] = source_flip_flops_truncated
        cards.append(card)
    return cards


def _basis_summaries(
    connection: sqlite3.Connection,
    assertion_ids: list[str],
    *,
    moment: str | None,
    clock: RecallClock,
) -> dict[str, dict[str, Any]]:
    """Build one bounded provenance summary and two representative refs per assertion.

    The aggregate statement returns at most one row per assertion. The windowed
    representative statement returns at most two rows per assertion, so card
    rendering never loads observation bodies or falls into assertion-by-
    assertion queries.
    """
    if not assertion_ids:
        return {}
    marks = ", ".join("?" for _ in assertion_ids)
    time_filter = ""
    values: list[Any] = list(assertion_ids)
    if moment is not None:
        if clock is RecallClock.KNOWLEDGE:
            time_filter = " AND ae.linked_at <= ?"
        else:
            time_filter = " AND COALESCE(o.source_published_at, o.observed_at) <= ?"
        values.append(moment)
    filtered = (
        "WITH filtered AS ("
        "SELECT ae.assertion_id, ae.observation_id, ae.role, o.depth, o.source_kind, "
        "o.source_published_at, o.observed_at, "
        "CASE WHEN json_valid(o.metadata_json) AND ("
        "json_extract(o.metadata_json, '$.material_reliability') IN ('best_effort', 'automatic') "
        "OR json_extract(o.metadata_json, '$.subtitle_reliability') = 'best_effort' "
        "OR lower(COALESCE(json_extract(o.metadata_json, '$.acquisition_method'), '')) LIKE '%asr%' "
        "OR lower(COALESCE(json_extract(o.metadata_json, '$.acquisition_method'), '')) "
        "LIKE 'faster_whisper:%') "
        "THEN 1 ELSE 0 END AS low_reliability, "
        "CASE WHEN json_valid(o.metadata_json) AND json_type(o.metadata_json, '$.limitations') = 'array' "
        "AND json_array_length(o.metadata_json, '$.limitations') > 0 THEN 1 ELSE 0 END AS has_limitations "
        "FROM assertion_evidence ae JOIN observations o ON o.id = ae.observation_id "
        f"WHERE ae.assertion_id IN ({marks}){time_filter}"
        "), "
    )
    aggregate_statement = (
        filtered
        + "role_counts AS (SELECT assertion_id, json_group_object(role, count) AS counts "
        "FROM (SELECT assertion_id, role, COUNT(*) AS count FROM filtered GROUP BY assertion_id, role) "
        "GROUP BY assertion_id), "
        "depth_counts AS (SELECT assertion_id, json_group_object(depth, count) AS counts "
        "FROM (SELECT assertion_id, depth, COUNT(*) AS count FROM filtered GROUP BY assertion_id, depth) "
        "GROUP BY assertion_id), "
        "kind_frequency AS (SELECT assertion_id, source_kind, COUNT(*) AS count FROM filtered "
        "GROUP BY assertion_id, source_kind), "
        "ranked_kinds AS (SELECT assertion_id, source_kind, count, "
        "ROW_NUMBER() OVER (PARTITION BY assertion_id ORDER BY count DESC, source_kind ASC) AS position "
        "FROM kind_frequency), "
        "bucketed_kinds AS (SELECT assertion_id, CASE WHEN position <= 8 THEN "
        "CASE WHEN length(source_kind) > 80 THEN substr(source_kind, 1, 80) ELSE source_kind END "
        "ELSE 'other' END AS source_kind_bucket, count FROM ranked_kinds), "
        "bounded_kinds AS (SELECT assertion_id, source_kind_bucket, SUM(count) AS count FROM bucketed_kinds "
        "GROUP BY assertion_id, source_kind_bucket), "
        "kind_counts AS (SELECT assertion_id, json_group_object(source_kind_bucket, count) AS counts "
        "FROM bounded_kinds GROUP BY assertion_id), "
        "kind_flags AS (SELECT assertion_id, MAX(position > 8) AS source_kinds_truncated, "
        "MAX(length(source_kind) > 80) AS source_kind_values_truncated FROM ranked_kinds "
        "GROUP BY assertion_id) "
        "SELECT f.assertion_id, COUNT(*) AS total_links, r.counts AS role_counts, d.counts AS depth_counts, "
        "k.counts AS kind_counts, flags.source_kinds_truncated, flags.source_kind_values_truncated, "
        "MIN(f.source_published_at) AS published_earliest, "
        "MAX(f.source_published_at) AS published_latest, MIN(f.observed_at) AS observed_earliest, "
        "MAX(f.observed_at) AS observed_latest, MAX(f.low_reliability) AS has_low_reliability, "
        "MAX(f.has_limitations) AS has_limitations FROM filtered f "
        "JOIN role_counts r ON r.assertion_id = f.assertion_id "
        "JOIN depth_counts d ON d.assertion_id = f.assertion_id "
        "JOIN kind_counts k ON k.assertion_id = f.assertion_id "
        "JOIN kind_flags flags ON flags.assertion_id = f.assertion_id "
        "GROUP BY f.assertion_id"
    )
    aggregate_rows = _rows(connection.execute(aggregate_statement, values).fetchall())
    representative_statement = (
        filtered
        + "ranked AS (SELECT assertion_id, observation_id, role, depth, "
        "CASE WHEN length(source_kind) > 80 THEN substr(source_kind, 1, 80) "
        "ELSE source_kind END AS source_kind, "
        "length(source_kind) > 80 AS source_kind_truncated, "
        "ROW_NUMBER() OVER (PARTITION BY assertion_id ORDER BY observation_id, role) AS position "
        "FROM filtered) SELECT assertion_id, observation_id, role, depth, source_kind, "
        "source_kind_truncated FROM ranked "
        "WHERE position <= 2 ORDER BY assertion_id, position"
    )
    representative_rows = _rows(connection.execute(representative_statement, values).fetchall())
    representatives: dict[str, list[dict[str, Any]]] = {}
    for row in representative_rows:
        representatives.setdefault(row["assertion_id"], []).append(
            {
                "observation_id": row["observation_id"],
                "role": row["role"],
                "depth": row["depth"],
                "source_kind": row["source_kind"],
                "source_kind_truncated": bool(row["source_kind_truncated"]),
            }
        )
    summaries: dict[str, dict[str, Any]] = {}
    for row in aggregate_rows:
        refs = representatives.get(row["assertion_id"], [])
        summaries[row["assertion_id"]] = {
            "total_links": int(row["total_links"]),
            "by_role": _count_map(row["role_counts"], ("supports", "context", "contradicts")),
            "by_depth": _count_map(row["depth_counts"], ("seen", "content", "discussion", "media")),
            "by_source_kind": _count_map(row["kind_counts"], ()),
            "source_kinds_truncated": bool(row["source_kinds_truncated"]),
            "source_kind_values_truncated": bool(row["source_kind_values_truncated"]),
            "time_range": {
                "published_at": {"earliest": row["published_earliest"], "latest": row["published_latest"]},
                "observed_at": {"earliest": row["observed_earliest"], "latest": row["observed_latest"]},
            },
            "has_low_reliability": bool(row["has_low_reliability"]),
            "has_limitations": bool(row["has_limitations"]),
            "representative_refs": refs,
            "refs_truncated": int(row["total_links"]) > len(refs),
        }
    return summaries


def _empty_basis_summary() -> dict[str, Any]:
    return {
        "total_links": 0,
        "by_role": {"supports": 0, "context": 0, "contradicts": 0},
        "by_depth": {"seen": 0, "content": 0, "discussion": 0, "media": 0},
        "by_source_kind": {},
        "source_kinds_truncated": False,
        "source_kind_values_truncated": False,
        "time_range": {
            "published_at": {"earliest": None, "latest": None},
            "observed_at": {"earliest": None, "latest": None},
        },
        "has_low_reliability": False,
        "has_limitations": False,
        "representative_refs": [],
        "refs_truncated": False,
    }


def _count_map(serialized: str | None, required: Sequence[str]) -> dict[str, int]:
    try:
        raw = json.loads(serialized) if serialized else {}
    except (TypeError, ValueError):
        raw = {}
    counts = {str(key): int(value) for key, value in raw.items()} if isinstance(raw, dict) else {}
    for key in required:
        counts.setdefault(key, 0)
    return counts


def _card_source_uris(
    connection: sqlite3.Connection,
    assertions: list[dict[str, Any]],
    *,
    moment: str | None,
    clock: RecallClock,
) -> tuple[dict[str, list[str]], dict[str, bool]]:
    """Map each assertion id to the source_uris its evidence observations cite.

    One batched query for the whole card set (no per-card round trips); an
    assertion with no evidence gets no entry.
    """
    if not assertions:
        return {}, {}
    marks = ", ".join("?" for _ in assertions)
    time_filter = ""
    values: list[Any] = [row["id"] for row in assertions]
    if moment is not None:
        if clock is RecallClock.KNOWLEDGE:
            time_filter = " AND ae.linked_at <= ?"
        else:
            time_filter = " AND COALESCE(o.source_published_at, o.observed_at) <= ?"
        values.append(moment)
    rows = _rows(
        connection.execute(
            "WITH distinct_sources AS (SELECT ae.assertion_id, o.source_uri "
            "FROM assertion_evidence ae JOIN observations o ON o.id = ae.observation_id "
            f"WHERE ae.assertion_id IN ({marks}){time_filter} GROUP BY ae.assertion_id, o.source_uri), "
            "ranked_sources AS (SELECT assertion_id, source_uri, "
            "ROW_NUMBER() OVER (PARTITION BY assertion_id ORDER BY source_uri) AS position, "
            "COUNT(*) OVER (PARTITION BY assertion_id) AS total_sources FROM distinct_sources) "
            "SELECT assertion_id, source_uri, total_sources FROM ranked_sources "
            "WHERE position <= 8 ORDER BY assertion_id, position",
            values,
        ).fetchall()
    )
    sources: dict[str, list[str]] = {}
    truncated: dict[str, bool] = {}
    for row in rows:
        sources.setdefault(row["assertion_id"], []).append(row["source_uri"])
        truncated[row["assertion_id"]] = int(row["total_sources"]) > _SOURCE_FLIP_URI_CAP
    return sources, truncated


def _display_source_uri(uri: str) -> tuple[str, bool]:
    if len(uri) <= _SOURCE_FLIP_URI_MAX_CHARS:
        return uri, False
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    prefix_chars = _SOURCE_FLIP_URI_MAX_CHARS - len(digest) - 1
    return f"{uri[:prefix_chars]}~{digest}", True


def _source_flip_flops(
    connection: sqlite3.Connection,
    card_sources: dict[str, list[str]],
    *,
    moment: str | None,
    clock: RecallClock,
) -> dict[str, int]:
    """Count superseded assertions citing each source_uri (source flip-flops).

    A source's flip-flop count is how many assertions citing it were later
    retired by a superseding one (any evidence role). Only sources cited by
    the cards under render are queried, in one batched statement; sources
    with no superseded assertions stay out of the result.
    """
    uris = list(dict.fromkeys(uri for sources in card_sources.values() for uri in sources))
    if not uris or (moment is not None and clock is RecallClock.WORLD):
        return {}
    marks = ", ".join("?" for _ in uris)
    visibility_filter = ""
    evidence_time_filter = ""
    values: list[Any] = list(uris)
    if moment is not None:
        evidence_time_filter = " AND ae.linked_at <= ?"
        visibility_filter = (
            " AND EXISTS (SELECT 1 FROM world_audit audit, "
            "json_each(audit.delta_json, '$.assertions') item "
            "WHERE json_extract(item.value, '$.id') = s.id AND audit.committed_at <= ?)"
        )
        values.extend((moment, moment))
    rows = _rows(
        connection.execute(
            "SELECT o.source_uri, COUNT(DISTINCT a.id) AS flip_count FROM assertions a "
            "JOIN assertion_evidence ae ON ae.assertion_id = a.id "
            "JOIN observations o ON o.id = ae.observation_id "
            f"WHERE o.source_uri IN ({marks}) "
            f"{evidence_time_filter}"
            "AND EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id"
            f"{visibility_filter}) "
            "GROUP BY o.source_uri",
            values,
        ).fetchall()
    )
    return {row["source_uri"]: int(row["flip_count"]) for row in rows}


def _evidence(
    connection: sqlite3.Connection,
    assertion_ids: list[str],
    *,
    limit: int,
    moment: str | None,
    clock: RecallClock,
) -> tuple[list[dict[str, Any]], bool]:
    if not assertion_ids:
        return [], False
    marks = ", ".join("?" for _ in assertion_ids)
    time_filter = ""
    values = list(assertion_ids)
    if moment is not None:
        if clock is RecallClock.KNOWLEDGE:
            time_filter = " AND ae.linked_at <= ?"
        else:
            time_filter = " AND COALESCE(o.source_published_at, o.observed_at) <= ?"
        values.append(moment)
    statement = (
        "SELECT ae.assertion_id, ae.observation_id, ae.role, ae.linked_at, "
        "o.source_uri, o.title, o.content_ref, o.depth, o.source_kind, "
        "o.source_published_at, o.observed_at, o.metadata_json FROM assertion_evidence ae "
        "JOIN observations o ON o.id = ae.observation_id "
        f"WHERE ae.assertion_id IN ({marks}){time_filter} "
        "ORDER BY ae.assertion_id, ae.observation_id, ae.role LIMIT ?"
    )
    values.append(limit + 1)
    refs = _rows(connection.execute(statement, values).fetchall())
    truncated = len(refs) > limit
    refs = refs[:limit]
    for ref in refs:
        try:
            metadata = json.loads(str(ref.pop("metadata_json") or "{}"))
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        ref["material_reliability"] = _material_reliability(metadata)
        ref["limitations"] = _material_limitations(metadata)
    return refs, truncated


def _material_reliability(metadata: dict[str, Any]) -> str:
    value = metadata.get("material_reliability")
    if value in {"source_direct", "confirmed", "best_effort", "automatic", "mixed", "unknown"}:
        return str(value)
    if metadata.get("subtitle_reliability") in {"confirmed", "best_effort"}:
        return str(metadata["subtitle_reliability"])
    method = str(metadata.get("acquisition_method", "")).casefold()
    if method.startswith("faster_whisper:") or "asr" in method:
        return "automatic"
    if metadata.get("modality") == "document_text":
        return "source_direct"
    return "unknown"


def _material_limitations(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("limitations")
    if not isinstance(raw, list):
        return []
    return _bounded_limitations(raw)


def _bounded_limitations(raw: list[Any]) -> list[str]:
    limitations: list[str] = []
    total_chars = 0
    truncated = False
    for item in raw:
        if not isinstance(item, str) or not item:
            truncated = True
            continue
        value = item[:_LIMITATIONS_ITEM_MAX_CHARS]
        truncated = truncated or value != item
        if (
            len(limitations) >= _LIMITATIONS_CARD_MAX_ITEMS
            or total_chars + len(value) > _LIMITATIONS_CARD_MAX_CHARS
        ):
            truncated = True
            continue
        limitations.append(value)
        total_chars += len(value)
    if not truncated:
        return limitations
    while limitations and (
        len(limitations) >= _LIMITATIONS_CARD_MAX_ITEMS
        or total_chars + len(_LIMITATIONS_TRUNCATED) > _LIMITATIONS_CARD_MAX_CHARS
    ):
        total_chars -= len(limitations.pop())
    limitations.append(_LIMITATIONS_TRUNCATED)
    return limitations


def _candidate_observations(
    connection: sqlite3.Connection,
    object_ids: list[str],
    inquiry_ids: list[str],
    *,
    limit: int,
    moment: str | None,
    clock: RecallClock,
) -> tuple[list[dict[str, Any]], bool]:
    output: list[dict[str, Any]] = []
    target = limit + 1
    time_column = (
        "linked_at" if clock is RecallClock.KNOWLEDGE else ("COALESCE(o.source_published_at, o.observed_at)")
    )
    for target_kind, table, column, identifiers in (
        ("object", "object_observations", "object_id", object_ids),
        ("inquiry", "inquiry_observations", "inquiry_id", inquiry_ids),
    ):
        remaining = target - len(output)
        if not identifiers or remaining <= 0:
            continue
        unique_ids = list(dict.fromkeys(identifiers))
        marks = ", ".join("?" for _ in unique_ids)
        time_filter = f" AND {time_column} <= ?" if moment is not None else ""
        values: list[object] = [target_kind, *unique_ids]
        if moment is not None:
            values.append(moment)
        values.append(remaining)
        statement = (
            f"SELECT ? AS target_kind, links.{column} AS target_id, "
            "links.observation_id, links.role, o.source_uri, o.title, o.depth, "
            "o.source_published_at, o.content_ref "
            f"FROM {table} links JOIN observations o ON o.id = links.observation_id "
            f"WHERE links.{column} IN ({marks}){time_filter} "
            f"ORDER BY links.{column}, links.observation_id, links.role LIMIT ?"
        )
        output.extend(_rows(connection.execute(statement, values).fetchall()))
    return output[:limit], len(output) > limit
