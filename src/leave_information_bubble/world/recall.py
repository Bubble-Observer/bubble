# ruff: noqa: D101, D102

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .contracts import (
    MemoryBundle,
    MemoryOverview,
    MemoryOverviewCounts,
    RecallClock,
)
from .recall_cards import _assertion_cards, _basis_summaries, _bundle
from .recall_common import (
    _MAX_RESULTS,
    _limit,
    _objects_by_id,
    _parse_iso,
    _rows,
)
from .recall_ego import _expand_ego, _parse_event_cursor
from .recall_overview import (
    _candidate_event_timestamp,
    _candidate_latest_timestamp,
    _coverage_rank,
    _inquiry_candidate,
    _object_candidate,
    _object_name_rows,
    _overview_active_events,
    _overview_cold_bridges,
    _overview_coverages,
    _overview_reactivated,
    _overview_subject_coverage,
)
from .recall_search import (
    _MATCH_SIMILARITY_THRESHOLD,  # noqa: F401  re-export (test_recall_search.py:607)
    _identity_terms,
    _match_candidates,
    _matching_assertions,
    _matching_assertions_count,
    _matching_inquiries,
    _matching_inquiries_count,
    _matching_name_usage,
    _matching_objects,
    _matching_objects_count,
    _merge_usage_subjects,
    _terms,
    identity_alias_forms,  # noqa: F401  re-export (console/inspection.py:15)
    legacy_alias_forms,  # noqa: F401  re-export (console/inspection.py:15)
)
from .store import WorldStore, alias_lookup_forms  # noqa: F401  re-export (test_recall_search.py:300)

_MAX_DEPTH = 5

_OVERVIEW_LIMIT = 3
_OVERVIEW_ACTIVE_DAYS = 14
_OVERVIEW_COLD_DAYS = 90
_OVERVIEW_INQUIRY_POOL_PER_STATUS = 48

_EGO_EVENT_LIMIT = 5

_READ_VIEWS = ("self_attributes", "out_edges", "in_edges", "correction_chain")
_READ_VIEW_CAP = 15  # per-view page; the read window (spec B3)
_READ_LIST_CAP = 12  # name_usages / participants bound inside the identity summary
_READ_CHAIN_CAP = 3  # correction chains shown per read
_READ_CHAIN_LENGTH_CAP = 5  # entries per chain (middle dropped with a flag)


def _latest_committed_object(connection: sqlite3.Connection) -> str | None:
    """Return the object of the most recently committed delta.

    An omitted ``object_id`` reads "what was just committed" — the
    memory_recent duty the design §4.3 default absorbs. Commit recency comes
    from ``world_audit`` deltas (like ``recent()``), newest commit first with
    the id as the deterministic tiebreak.
    """
    row = connection.execute(
        "SELECT json_extract(item.value, '$.id') AS id FROM world_audit,"
        " json_each(delta_json, '$.objects') AS item"
        " ORDER BY committed_at DESC, json_extract(item.value, '$.id') DESC LIMIT 1"
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _current_subject_assertions(
    connection: sqlite3.Connection,
    subject_id: str,
    *,
    predicate: str | None = None,
    cap: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Return current (non-superseded) subject-anchored assertions, capped."""
    where = (
        "subject_id = ? AND NOT EXISTS"
        " (SELECT 1 FROM assertions s WHERE s.supersedes_id = assertions.id)"
    )
    parameters: list[Any] = [subject_id]
    if predicate is not None:
        where += " AND predicate = ?"
        parameters.append(predicate)
    rows = _rows(
        connection.execute(
            f"SELECT * FROM assertions WHERE {where} ORDER BY id LIMIT ?",
            (*parameters, cap + 1),
        ).fetchall()
    )
    return rows[:cap], len(rows) > cap


def _identity_summary(connection: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    """Build the bounded identity portrait for one object (design §4.3).

    Every name surface is separated by origin: active identity aliases,
    removed identity aliases (history), legacy ``object_aliases`` forms the
    identity table never claimed, and current ``name_usage`` assertions
    (agent-submitted community usage with their assertion ids). The outbound
    participant list resolves roles from the ``has_participant`` qualifiers.
    ``view_counts`` always reports the true current totals so the agent can
    decide whether to expand a direction.
    """
    object_id = str(row["id"])
    active = _rows(
        connection.execute(
            "SELECT raw_alias, normalized_alias FROM identity_aliases"
            " WHERE object_id = ? AND status = 'active' ORDER BY normalized_alias",
            (object_id,),
        ).fetchall()
    )
    removed = _rows(
        connection.execute(
            "SELECT raw_alias, normalized_alias FROM identity_aliases"
            " WHERE object_id = ? AND status = 'removed' ORDER BY normalized_alias",
            (object_id,),
        ).fetchall()
    )
    identity_forms = {
        alias["normalized_alias"] for alias in [*active, *removed]
    }
    legacy = [
        alias["normalized_alias"]
        for alias in _rows(
            connection.execute(
                "SELECT normalized_alias FROM object_aliases"
                " WHERE object_id = ? ORDER BY normalized_alias",
                (object_id,),
            ).fetchall()
        )
        if alias["normalized_alias"] not in identity_forms
    ]
    usages, usages_truncated = _current_subject_assertions(
        connection, object_id, predicate="name_usage", cap=_READ_LIST_CAP
    )
    participants, participants_truncated = _current_subject_assertions(
        connection, object_id, predicate="has_participant", cap=_READ_LIST_CAP
    )
    participant_names = _object_name_map(connection, [p["object_id"] for p in participants])
    counts = _direction_counts(connection, object_id)
    identity: dict[str, Any] = {
        "object_id": object_id,
        "canonical_name": row["canonical_name"],
        "kind": row["kind"],
        "type_key": row.get("type_key"),
        "provisional": bool(row["provisional"]),
        "version": int(row["version"]),
        "active_aliases": active,
        "removed_aliases": removed,
        "legacy_aliases": legacy,
        "name_usages": [
            {
                "assertion_id": usage["id"],
                "literal": json.loads(usage["literal_json"]) if usage["literal_json"] else None,
            }
            for usage in usages
        ],
        "name_usages_truncated": usages_truncated,
        "participants": [
            {
                "object_id": edge["object_id"],
                "canonical_name": participant_names.get(edge["object_id"], ""),
                "role": _qualifiers(edge).get("role", ""),
            }
            for edge in participants
            if edge["object_id"]
        ],
        "participants_truncated": participants_truncated,
        "view_counts": counts,
    }
    start, end = row.get("event_time_start"), row.get("event_time_end")
    if start is not None or end is not None:
        identity["event_time"] = {"start": start, "end": end}
    return identity


def _object_name_map(
    connection: sqlite3.Connection, identifiers: Sequence[str | None]
) -> dict[str, str]:
    """Resolve canonical names for object ids (empty for missing ids)."""
    identifiers = [identifier for identifier in identifiers if identifier]
    if not identifiers:
        return {}
    marks = ", ".join("?" for _ in identifiers)
    return {
        str(row["id"]): str(row["canonical_name"])
        for row in _rows(
            connection.execute(
                f"SELECT id, canonical_name FROM objects WHERE id IN ({marks})", identifiers
            ).fetchall()
        )
    }


def _qualifiers(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["qualifiers_json"]) if row.get("qualifiers_json") else {}


def _direction_counts(
    connection: sqlite3.Connection, object_id: str
) -> dict[str, int]:
    """Count current assertions by direction (self literal / out edge / in edge)."""

    def _count(where: str, parameters: Sequence[Any]) -> int:
        row = connection.execute(
            "SELECT COUNT(*) FROM assertions WHERE"
            f" {where} AND NOT EXISTS"
            " (SELECT 1 FROM assertions s WHERE s.supersedes_id = assertions.id)",
            parameters,
        ).fetchone()
        return int(row[0])

    return {
        "self_attributes": _count("subject_id = ? AND literal_json IS NOT NULL", (object_id,)),
        "out_edges": _count("subject_id = ? AND object_id IS NOT NULL", (object_id,)),
        "in_edges": _count(
            "object_id = ? AND subject_id != ?", (object_id, object_id)
        ),
    }


def _staged_row(
    connection: sqlite3.Connection,
    table: str,
    wake_id: str,
    reference: str,
    status: str,
) -> dict[str, Any] | None:
    """One staged row of this wake+status matching a staged_id or target_ref.

    The working graph (Core §6.2): staged reads resolve by staged_id (a
    newly created item) or by target_ref (an update/drop covering a formal
    item). Other wakes and non-active statuses never surface here.
    """
    rows = _rows(
        connection.execute(
            f"SELECT * FROM {table} WHERE wake_id = ? AND status = ?"
            " AND (staged_id = ? OR target_ref = ?) LIMIT 1",
            (wake_id, status, reference, reference),
        ).fetchall()
    )
    if not rows:
        return None
    row = rows[0]
    if table == "staged_objects":
        row["aliases"] = json.loads(row.get("aliases_json") or "[]")
        row.pop("aliases_json", None)
    elif table == "staged_assertions":
        row["qualifiers"] = json.loads(row.get("qualifiers_json") or "{}")
        row["evidence"] = json.loads(row.get("evidence_json") or "[]")
        row["literal"] = (
            json.loads(row["literal_json"]) if row.get("literal_json") is not None else None
        )
        row.pop("qualifiers_json", None)
        row.pop("evidence_json", None)
        row.pop("literal_json", None)
    return row


def _staged_identity(staged: Mapping[str, Any]) -> dict[str, Any]:
    """Render the identity portrait of a newly created staged object.

    Mirrors ``_identity_summary``'s surface names; every formal-derived
    section is empty because nothing has been committed yet. Aliases are the
    raw strings the Agent submitted — normalization happens at finalize.
    """
    identity: dict[str, Any] = {
        "object_id": staged["staged_id"],
        "canonical_name": staged["canonical_name"],
        "kind": staged["kind"],
        "type_key": staged.get("type_key"),
        "provisional": bool(staged.get("provisional")),
        "version": int(staged["version"]),
        "active_aliases": [
            {"raw_alias": alias, "normalized_alias": alias}
            for alias in staged.get("aliases", [])
        ],
        "removed_aliases": [],
        "legacy_aliases": [],
        "name_usages": [],
        "name_usages_truncated": False,
        "participants": [],
        "participants_truncated": False,
        "view_counts": {"self_attributes": 0, "out_edges": 0, "in_edges": 0},
    }
    start, end = staged.get("event_time_start"), staged.get("event_time_end")
    if start is not None or end is not None:
        identity["event_time"] = {"start": start, "end": end}
    return identity


def _direction_rows(
    connection: sqlite3.Connection,
    object_id: str,
    direction: str,
    cap: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Return current assertions of one direction, capped with the cut flag."""
    if direction == "self_attributes":
        where, parameters = "subject_id = ? AND literal_json IS NOT NULL", (object_id,)
    elif direction == "out_edges":
        where, parameters = "subject_id = ? AND object_id IS NOT NULL", (object_id,)
    else:
        where, parameters = "object_id = ? AND subject_id != ?", (object_id, object_id)
    rows = _rows(
        connection.execute(
            f"SELECT * FROM assertions WHERE {where} AND NOT EXISTS"
            " (SELECT 1 FROM assertions s WHERE s.supersedes_id = assertions.id)"
            " ORDER BY id LIMIT ?",
            (*parameters, cap + 1),
        ).fetchall()
    )
    return rows[:cap], len(rows) > cap


def _render_cards(
    connection: sqlite3.Connection, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Render storage assertion rows as the same Agent-facing cards recall uses."""
    summaries = _basis_summaries(
        connection,
        [row["id"] for row in rows],
        moment=None,
        clock=RecallClock.KNOWLEDGE,
    )
    return _assertion_cards(
        connection,
        rows,
        summaries,
        moment=None,
        clock=RecallClock.KNOWLEDGE,
    )


def _correction_chains(
    connection: sqlite3.Connection,
    object_id: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Render subject-anchored supersede chains oldest-first, tails marked current.

    A chain exists only where a later assertion replaced an earlier one
    (``supersedes_id``): supersede never deletes, and the tail of each chain
    is the current cognition. Only the longest chains are shown; a chain
    longer than the entry cap keeps its head and tail and drops the middle
    with a per-chain flag. Returns the chains and whether more chains were
    cut.
    """
    rows = _rows(
        connection.execute(
            "SELECT id, predicate, literal_json, object_id, supersedes_id, superseded_at"
            " FROM assertions WHERE subject_id = ? ORDER BY id",
            (object_id,),
        ).fetchall()
    )
    by_id = {row["id"]: row for row in rows}
    superseded_targets = {row["supersedes_id"] for row in rows if row["supersedes_id"]}
    chain_id_lists: list[list[str]] = []
    for row in rows:
        if row["id"] in superseded_targets:
            continue  # not a tail: a later row replaced it
        ids: list[str] = []
        current = row
        while current is not None:
            ids.append(current["id"])
            current = by_id.get(current["supersedes_id"])
        if len(ids) > 1:
            chain_id_lists.append(ids)
    chain_id_lists.sort(key=len, reverse=True)
    chains: list[dict[str, Any]] = []
    for ids in chain_id_lists[:_READ_CHAIN_CAP]:
        # ids run current -> head; render head -> current so the chain reads
        # as correction history with the tail marked as the current cognition.
        tail_id = ids[0]
        items = [by_id[identifier] for identifier in reversed(ids)]
        middle_cut = len(items) > _READ_CHAIN_LENGTH_CAP
        if middle_cut:
            items = items[: _READ_CHAIN_LENGTH_CAP - 1] + [items[-1]]
        rendered: list[dict[str, Any]] = []
        for position, item in enumerate(items):
            rendered.append(
                {
                    "id": item["id"],
                    "predicate": item["predicate"],
                    "literal": json.loads(item["literal_json"]) if item["literal_json"] else None,
                    "object_id": item["object_id"],
                    "supersedes_id": item["supersedes_id"],
                    "superseded_at": item["superseded_at"],
                    "current": position == len(items) - 1,
                }
            )
        chains.append(
            {"tail_id": tail_id, "chain": rendered, "chain_truncated": middle_cut}
        )
    return chains, len(chain_id_lists) > _READ_CHAIN_CAP


def _assertion_chain(
    connection: sqlite3.Connection, row: dict[str, Any]
) -> dict[str, Any]:
    """Bound the evidence view's correction chain around one assertion.

    ``supersedes`` walks the ancestors oldest-first (immediate predecessor
    last); ``superseded_by`` lists the bounded assertions that replaced this
    one. Both lists carry id + predicate only — the evidence cards carry the
    full detail.
    """
    supersedes: list[dict[str, Any]] = []
    current = row
    seen: set[str] = set()
    while len(supersedes) < _READ_CHAIN_LENGTH_CAP and current.get("supersedes_id"):
        target = str(current["supersedes_id"])
        if target in seen:
            break
        seen.add(target)
        ancestor = connection.execute(
            "SELECT id, predicate FROM assertions WHERE id = ?", (target,)
        ).fetchone()
        if ancestor is None:
            break
        supersedes.append({"id": ancestor[0], "predicate": ancestor[1]})
        current = dict(ancestor)
    supersedes.reverse()
    superseded_by = [
        {"id": row["id"], "predicate": row["predicate"]}
        for row in _rows(
            connection.execute(
                "SELECT id, predicate FROM assertions WHERE supersedes_id = ? ORDER BY id LIMIT ?",
                (row["id"], _READ_CHAIN_CAP),
            ).fetchall()
        )
    ]
    return {"supersedes": supersedes, "superseded_by": superseded_by}


def _read_view_items(
    connection: sqlite3.Connection,
    object_id: str,
    requested: set[str],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Expand the requested views with per-view caps; nothing else expands."""
    view_items: dict[str, Any] = {}
    view_truncated: dict[str, bool] = {}
    for direction in ("self_attributes", "out_edges", "in_edges"):
        if direction not in requested:
            continue
        rows, truncated = _direction_rows(connection, object_id, direction, _READ_VIEW_CAP)
        view_items[direction] = _render_cards(connection, rows)
        view_truncated[direction] = truncated
    if "correction_chain" in requested:
        chains, chains_truncated = _correction_chains(connection, object_id)
        view_items["correction_chain"] = chains
        view_truncated["correction_chain"] = chains_truncated
    return view_items, view_truncated


def _scalar_entry(field: str, left: object, right: object) -> dict[str, Any]:
    """One side-by-side row for a scalar compare field."""
    return {"field": field, "left": left, "right": right, "equal": left == right}


def _set_entry(field: str, left: set[str], right: set[str]) -> dict[str, Any]:
    """One side-by-side row for a set field: shared plus per-side onlys.

    Every component is capped so a compare never expands without bound; a cut
    anywhere raises the field's truncated flag while the shared set — the
    decision-relevant part for identity work — stays intact up to its cap.
    """
    shared = sorted(left & right)
    left_only = sorted(left - right)
    right_only = sorted(right - left)
    truncated = False
    for component in (shared, left_only, right_only):
        if len(component) > _READ_VIEW_CAP:
            del component[_READ_VIEW_CAP:]
            truncated = True
    return {
        "field": field,
        "shared": shared,
        "left_only": left_only,
        "right_only": right_only,
        "equal": not left_only and not right_only,
        "truncated": truncated,
    }


def _compare_object_side(connection: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    """Collect one object's comparable identity surfaces, unbounded (design §4.4).

    Name surfaces follow the same origin split as the identity portrait
    (active / removed identity aliases, legacy object_aliases forms, current
    name_usage literals); participants resolve to ``id|role`` forms; key
    assertions are the current subject-anchored judgment signatures. The
    source domain has no object-level column yet, so it is not representable
    here (decision log D-010).
    """
    object_id = str(row["id"])
    active = {
        str(alias["normalized_alias"])
        for alias in _rows(
            connection.execute(
                "SELECT normalized_alias FROM identity_aliases"
                " WHERE object_id = ? AND status = 'active'",
                (object_id,),
            ).fetchall()
        )
    }
    removed = {
        str(alias["normalized_alias"])
        for alias in _rows(
            connection.execute(
                "SELECT normalized_alias FROM identity_aliases"
                " WHERE object_id = ? AND status = 'removed'",
                (object_id,),
            ).fetchall()
        )
    }
    identity_forms = active | removed
    legacy = {
        str(alias["normalized_alias"])
        for alias in _rows(
            connection.execute(
                "SELECT normalized_alias FROM object_aliases WHERE object_id = ?",
                (object_id,),
            ).fetchall()
        )
        if alias["normalized_alias"] not in identity_forms
    }
    current = _rows(
        connection.execute(
            "SELECT predicate, literal_json, object_id, qualifiers_json FROM assertions"
            " WHERE subject_id = ? AND NOT EXISTS"
            " (SELECT 1 FROM assertions s WHERE s.supersedes_id = assertions.id)",
            (object_id,),
        ).fetchall()
    )
    usages: set[str] = set()
    participants: set[str] = set()
    key: set[str] = set()
    for assertion in current:
        if assertion["predicate"] == "name_usage" and assertion["literal_json"]:
            usages.add(str(json.loads(assertion["literal_json"])))
        if assertion["predicate"] == "has_participant" and assertion["object_id"]:
            participants.add(
                f"{assertion['object_id']}|{_qualifiers(assertion).get('role', '')}"
            )
        if assertion["literal_json"] is not None:
            key.add(f"{assertion['predicate']}|{json.loads(assertion['literal_json'])}")
        elif assertion["object_id"]:
            key.add(f"{assertion['predicate']}|{assertion['object_id']}")
    start, end = row.get("event_time_start"), row.get("event_time_end")
    return {
        "canonical_name": row["canonical_name"],
        "kind": row["kind"],
        "type_key": row.get("type_key"),
        "event_time": {"start": start, "end": end} if start is not None or end is not None else None,
        "active_aliases": active,
        "removed_aliases": removed,
        "legacy_aliases": legacy,
        "name_usages": usages,
        "participants": participants,
        "key_assertions": key,
    }


def _compare_assertion_side(connection: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    """Collect one assertion's comparable signature fields, unbounded (§7.3)."""
    start, end = row.get("event_time_start"), row.get("event_time_end")
    return {
        "subject_id": str(row["subject_id"]),
        "predicate": str(row["predicate"]),
        "object_id": str(row["object_id"]) if row.get("object_id") else None,
        "literal": json.loads(row["literal_json"]) if row.get("literal_json") else None,
        "event_time": {"start": start, "end": end} if start is not None or end is not None else None,
        "epistemic_role": str(row["epistemic_role"]),
        "evidence": {
            str(item["observation_id"])
            for item in _rows(
                connection.execute(
                    "SELECT observation_id FROM assertion_evidence WHERE assertion_id = ?",
                    (str(row["id"]),),
                ).fetchall()
            )
        },
        "supersedes_id": row.get("supersedes_id"),
        "superseded_by": {
            str(item["id"])
            for item in _rows(
                connection.execute(
                    "SELECT id FROM assertions WHERE supersedes_id = ?", (str(row["id"]),)
                ).fetchall()
            )
        },
    }


class WorldRecall:
    def __init__(self, store: WorldStore) -> None:
        self._store = store

    def recent(self, limit: int = 12) -> MemoryBundle:
        """Return the most recently committed assertions, newest commit first.

        This is *commit recency* (最近提交): what the agent most recently
        wrote down, ordered by world_audit committed_at. It is deliberately
        distinct from the overview's active frontier (活跃), which ranks by
        event time so re-reading old content cannot masquerade as new
        activity (spec §7). recent() answers "what was just committed";
        overview.active_fronts answers "which events are currently moving".
        """
        cap = _limit(limit)
        with self._store.read_connection() as connection:
            statement = (
                "SELECT a.* FROM assertions a LEFT JOIN (SELECT json_extract(item.value, '$.id') AS id, "
                "MAX(committed_at) AS committed_at FROM world_audit, "
                "json_each(delta_json, '$.assertions') AS item GROUP BY id) audit ON audit.id = a.id "
                "WHERE NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id) "
                "ORDER BY audit.committed_at DESC, a.id DESC LIMIT ?"
            )
            assertions = _rows(connection.execute(statement, (cap + 1,)).fetchall())
        return _bundle(
            assertions=assertions[:cap],
            reasons=["recent assertions"],
            store=self._store,
            evidence_limit=cap,
            truncated=len(assertions) > cap,
        )

    def changes(self, since: datetime | None = None, limit: int = 15) -> dict[str, Any]:
        """Return what entered the world after *since*, newest commits first.

        Parses ``world_audit`` deltas committed after the boundary and collects
        new object/assertion/inquiry ids plus resolved inquiry ids. Deltas are
        read newest-first so the deduped id lists lead with the latest changes
        (spec B2 "how the world changed" reads newest-first; audit B-5 caught
        the old ASC order returning the oldest changes). Since is compared on
        the UTC ISO string (consistent with stored formats).
        """
        cap = _limit(limit)
        if since is None:
            with self._store.read_connection() as connection:
                row = connection.execute(
                    "SELECT MAX(attempted_at) FROM proposal_attempts"
                ).fetchone()
                since = _parse_iso(row[0]) if row and row[0] else None
        boundary = since.astimezone(UTC).isoformat() if since is not None else None
        new_objects: list[str] = []
        new_assertions: list[str] = []
        new_inquiries: list[str] = []
        resolved: list[str] = []
        with self._store.read_connection() as connection:
            rows = connection.execute(
                "SELECT delta_json FROM world_audit"
                + (" WHERE committed_at > ?" if boundary else "")
                + " ORDER BY committed_at DESC",
                (boundary,) if boundary else (),
            ).fetchall()
            for (delta_json,) in rows:
                try:
                    delta = json.loads(delta_json)
                except (TypeError, ValueError):
                    continue
                if not isinstance(delta, dict):
                    continue
                for key, sink in (
                    ("objects", new_objects),
                    ("assertions", new_assertions),
                    ("inquiries", new_inquiries),
                    ("resolve_inquiries", resolved),
                ):
                    for i in delta.get(key, []):
                        if not i or not i.get("id"):
                            continue
                        sink.append(str(i.get("id")))
        unique_objects = list(dict.fromkeys(new_objects))
        unique_assertions = list(dict.fromkeys(new_assertions))
        unique_inquiries = list(dict.fromkeys(new_inquiries))
        unique_resolved = list(dict.fromkeys(resolved))
        return {
            "since": since.isoformat() if since is not None else None,
            "new_objects": unique_objects[:cap],
            "new_assertions": unique_assertions[:cap],
            "new_inquiries": unique_inquiries[:cap],
            "resolved_inquiries": unique_resolved[:cap],
            "truncated": any(
                len(items) > cap
                for items in (unique_objects, unique_assertions, unique_inquiries, unique_resolved)
            ),
        }

    def read(
        self,
        object_id: str | None = None,
        assertion_id: str | None = None,
        *,
        views: Sequence[str] = (),
        include_working: bool = False,
        wake_id: str = "",
    ) -> MemoryBundle:
        """Return one object's complete current portrait, or one assertion's evidence view.

        Object mode (design §4.3) always returns the identity summary: every
        name surface (canonical, active identity aliases, removed identity
        aliases, legacy ``object_aliases`` forms, current ``name_usage``
        assertions), kind/type_key, the event span, the outbound participant
        list, and the three-direction view counts. The views themselves stay
        folded until requested through ``views`` — an output-driven read that
        never dumps the whole object by default. Each requested view is
        bounded and reports its cut; ``correction_chain`` renders the
        subject-anchored supersede chains oldest-first with the tail marked as
        the current cognition (supersede never deletes, the tail is current).
        ``object_id`` defaults to the most recently committed object, which
        absorbs memory_recent's "what was just committed" duty.

        Assertion mode (``assertion_id``) returns the judgment with its
        evidence refs and the bounded ``assertion_chain``: what the judgment
        supersedes and what superseded it.

        With ``include_working`` (and a ``wake_id`` to overlay) the read
        reflects the current wake's active staged graph (Core §6.5): a staged
        create resolves by its staged_id into a staged portrait, a staged
        update covering a formal id returns the merged portrait, and a staged
        drop of a formal id yields a ``reasons``-only bundle ("dropped in
        working graph") — the item no longer exists in the merged view.
        Staged rows never surface for other wakes, and ``views`` read the
        formal graph (staged edges in views arrive with graph_inspect).

        Raises:
            ValueError: If both ``object_id`` and ``assertion_id`` are given,
                or a ``views`` member is not one of ``self_attributes`` /
                ``out_edges`` / ``in_edges`` / ``correction_chain``.

        """
        if object_id is not None and assertion_id is not None:
            raise ValueError("memory_read takes object_id or assertion_id, not both")
        unknown_views = [view for view in views if view not in _READ_VIEWS]
        if unknown_views:
            raise ValueError(f"unknown view: {unknown_views[0]}")
        requested = set(views)
        with self._store.read_connection() as connection:
            if assertion_id is not None:
                rows = _rows(
                    connection.execute(
                        "SELECT * FROM assertions WHERE id = ?", (assertion_id,)
                    ).fetchall()
                )
                staged = dropped = None
                if include_working and wake_id:
                    staged = _staged_row(connection, "staged_assertions", wake_id, assertion_id, "active")
                    dropped = _staged_row(
                        connection, "staged_assertions", wake_id, assertion_id, "abandoned"
                    )
                if not rows and staged is None:
                    return MemoryBundle(reasons=["unknown assertion"])
                if dropped is not None and staged is None and not rows:
                    return MemoryBundle(reasons=["assertion dropped in working graph"])
                if staged is not None:
                    source = "merged" if rows else "staged"
                    merged_row: dict[str, Any] = {
                        "id": rows[0]["id"] if rows else staged["staged_id"],
                        "subject_id": staged["subject_ref"],
                        "predicate": staged["predicate"],
                        "object_id": staged["object_ref"],
                        "literal_json": (
                            json.dumps(staged["literal"]) if staged["literal"] is not None else None
                        ),
                        "epistemic_role": staged["epistemic_role"],
                        "confidence": staged["confidence"],
                        "event_time_start": staged["event_time_start"],
                        "event_time_end": staged["event_time_end"],
                        "supersedes_id": staged["supersedes_ref"],
                        "qualifiers_json": json.dumps(staged["qualifiers"]),
                        "source": source,
                    }
                    bundle = _bundle(
                        assertions=[merged_row],
                        reasons=["assertion evidence read (working graph)"],
                        store=self._store,
                        evidence_limit=_READ_VIEW_CAP,
                    )
                    if not rows:
                        bundle = bundle.model_copy(
                            update={
                                "evidence_refs": [
                                    {
                                        "assertion_id": staged["staged_id"],
                                        "observation_id": observation_id,
                                        "role": "evidence",
                                    }
                                    for observation_id in staged["evidence"]
                                ]
                            }
                        )
                    return bundle
                if include_working:
                    for row in rows:
                        row["source"] = "formal"
                chain = _assertion_chain(connection, rows[0])
                bundle = _bundle(
                    assertions=rows,
                    reasons=["assertion evidence read"],
                    store=self._store,
                    evidence_limit=_READ_VIEW_CAP,
                )
                return bundle.model_copy(update={"assertion_chain": chain})
            if object_id is None:
                object_id = _latest_committed_object(connection)
                if object_id is None:
                    return MemoryBundle(reasons=["no committed object to read"])
            staged = dropped = None
            if include_working and wake_id:
                staged = _staged_row(connection, "staged_objects", wake_id, object_id, "active")
                dropped = _staged_row(connection, "staged_objects", wake_id, object_id, "abandoned")
            rows = _objects_by_id(connection, [object_id])
            if staged is not None and staged["staged_id"] == object_id:
                # a newly created staged object: staged portrait, no formal row
                staged_row: dict[str, Any] = {
                    "id": staged["staged_id"],
                    "canonical_name": staged["canonical_name"],
                    "kind": staged["kind"],
                    "type_key": staged.get("type_key"),
                    "provisional": bool(staged.get("provisional")),
                    "version": int(staged["version"]),
                    "event_time_start": staged["event_time_start"],
                    "event_time_end": staged["event_time_end"],
                    "source": "staged",
                }
                return MemoryBundle(
                    anchor_objects=[staged_row],
                    reasons=["object identity read (working graph)"],
                ).model_copy(update={"identity": _staged_identity(staged)})
            if dropped is not None and not rows:
                return MemoryBundle(reasons=["object dropped in working graph"])
            if not rows:
                return MemoryBundle(reasons=["unknown object"])
            object_row = rows[0]
            if staged is None and include_working:
                object_row["source"] = "formal"
            if staged is not None:
                # a staged update covers this formal object: merged portrait
                for field in (
                    "canonical_name",
                    "kind",
                    "type_key",
                    "event_time_start",
                    "event_time_end",
                ):
                    object_row[field] = staged[field]
                object_row["version"] = int(staged["version"])
                object_row["source"] = "merged"
            identity = _identity_summary(connection, object_row)
            if staged is not None:
                identity["canonical_name"] = staged["canonical_name"]
                identity["kind"] = staged["kind"]
                identity["type_key"] = staged.get("type_key")
                identity["version"] = int(staged["version"])
                identity["active_aliases"] = [
                    {"raw_alias": alias, "normalized_alias": alias}
                    for alias in staged["aliases"]
                ]
                identity["removed_aliases"] = []
                identity["source"] = "merged"
                if staged["event_time_start"] or staged["event_time_end"]:
                    identity["event_time"] = {
                        "start": staged["event_time_start"],
                        "end": staged["event_time_end"],
                    }
                else:
                    identity.pop("event_time", None)
            view_items, view_truncated = _read_view_items(connection, object_id, requested)
        bundle = MemoryBundle(anchor_objects=[object_row], reasons=["object identity read"])
        return bundle.model_copy(
            update={"identity": identity, "views": view_items, "view_truncated": view_truncated}
        )

    def compare(self, left_id: str, right_id: str) -> MemoryBundle:
        """Compare two ids side by side without deciding (design §4.4, plan §7.3).

        Object mode lines up the identity fields (canonical name, kind,
        type_key, event span, active/removed/legacy alias forms, asserted
        usages, participants, key assertions); assertion mode lines up the
        signature fields (subject, predicate, object/literal, event window,
        epistemic role, evidence, supersede relations). Scalar fields render
        ``left/right/equal``; set fields render ``shared/left_only/right_only``
        with per-component caps and a truncated flag. The payload states facts
        and differences only — it never returns a merge verdict. Both ids must
        resolve to the same mode; a mixed pair is an error, an unknown id
        yields reasons only (the envelope resolves which side is missing).

        This reads the formal graph only; staged ids arrive with the finalize
        milestone.

        Raises:
            ValueError: If either id is blank, or one id is an object and the
                other an assertion.

        """
        if not left_id or not right_id:
            raise ValueError("memory_compare takes left_id and right_id")
        with self._store.read_connection() as connection:
            left_objects = _objects_by_id(connection, [left_id])
            right_objects = _objects_by_id(connection, [right_id])
            left_assertions = _rows(
                connection.execute(
                    "SELECT * FROM assertions WHERE id = ?", (left_id,)
                ).fetchall()
            )
            right_assertions = _rows(
                connection.execute(
                    "SELECT * FROM assertions WHERE id = ?", (right_id,)
                ).fetchall()
            )
            left_mode = "object" if left_objects else ("assertion" if left_assertions else None)
            right_mode = "object" if right_objects else ("assertion" if right_assertions else None)
            if left_mode is None or right_mode is None:
                missing = left_id if left_mode is None else right_id
                return MemoryBundle(reasons=[f"unknown id: {missing}"])
            if left_mode != right_mode:
                raise ValueError(
                    "memory_compare compares two objects or two assertions, not one of each"
                )
            if left_mode == "object":
                left_data = _compare_object_side(connection, left_objects[0])
                right_data = _compare_object_side(connection, right_objects[0])
                scalar_fields = (
                    "canonical_name",
                    "kind",
                    "type_key",
                    "event_time",
                )
                set_fields = (
                    "active_aliases",
                    "removed_aliases",
                    "legacy_aliases",
                    "name_usages",
                    "participants",
                    "key_assertions",
                )
                compare: dict[str, Any] = {
                    "mode": "object",
                    "left": {
                        "id": left_id,
                        "status": "ok",
                        "canonical_name": left_data["canonical_name"],
                    },
                    "right": {
                        "id": right_id,
                        "status": "ok",
                        "canonical_name": right_data["canonical_name"],
                    },
                    "fields": [
                        *[
                            _scalar_entry(field, left_data[field], right_data[field])
                            for field in scalar_fields
                        ],
                        *[
                            _set_entry(field, left_data[field], right_data[field])
                            for field in set_fields
                        ],
                    ],
                }
                reasons = ["object identity compare"]
            else:
                left_data = _compare_assertion_side(connection, left_assertions[0])
                right_data = _compare_assertion_side(connection, right_assertions[0])
                scalar_fields = (
                    "subject_id",
                    "predicate",
                    "object_id",
                    "literal",
                    "event_time",
                    "epistemic_role",
                    "supersedes_id",
                )
                compare = {
                    "mode": "assertion",
                    "left": {
                        "id": left_id,
                        "status": "ok",
                        "predicate": left_data["predicate"],
                    },
                    "right": {
                        "id": right_id,
                        "status": "ok",
                        "predicate": right_data["predicate"],
                    },
                    "fields": [
                        *[
                            _scalar_entry(field, left_data[field], right_data[field])
                            for field in scalar_fields
                        ],
                        _set_entry("evidence", left_data["evidence"], right_data["evidence"]),
                        _set_entry(
                            "superseded_by", left_data["superseded_by"], right_data["superseded_by"]
                        ),
                    ],
                }
                reasons = ["assertion signature compare"]
        bundle = MemoryBundle(reasons=reasons)
        return bundle.model_copy(update={"compare": compare})

    def search(
        self,
        query: str,
        *,
        limit: int = 12,
        as_of: datetime | None = None,
        clock: RecallClock = RecallClock.KNOWLEDGE,
        kind: str | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        predicate: str | None = None,
        has_participants: bool | None = None,
        assertion_count_min: int | None = None,
        assertion_count_max: int | None = None,
        offset: int = 0,
    ) -> MemoryBundle:
        """Match objects, assertions, and open inquiries by full-text query.

        Every matched object also gets one ``candidates`` entry labeled with
        its highest name-recall layer (contract §7.3 / task 4.1), in layer
        priority order: ``identity_alias_exact`` (an active identity alias),
        ``canonical_exact`` (canonical name; several objects may share one),
        ``name_usage`` (an asserted community usage, carrying the assertion
        id, qualifiers, time, and evidence counts), ``legacy_name`` (a
        historical ``object_aliases`` row, explicitly
        ``identity_authority=false``), ``possible_match`` (a fuzzy
        resemblance), or ``text_match`` (a full-text hit only). All layers
        compare through the identity normalizer; the legacy layer
        additionally serves the legacy ``alias_lookup_forms``. An exact
        identity hit is only the highest-priority candidate — it never
        triggers writes or merges. Objects that surface only through matched
        assertion or inquiry text (other than name_usage) get no candidate
        entry. No confidence float is computed anywhere.

        Structured filters (design §4.2): ``kind``, ``has_participants`` and
        ``assertion_count_min``/``max`` constrain the object bucket, and
        ``predicate`` the assertion bucket; the ``time_from``/``time_to``
        window is an overlap test on event spans applied to both. All three
        buckets share the same ``offset`` page position, so paging one
        structured query returns coherent pages (the facade owns cursors;
        this method takes the raw offset).
        """
        terms = _terms(query)
        if not terms:
            return MemoryBundle(reasons=["empty query"])
        cap = _limit(limit)
        moment = as_of.astimezone(UTC).isoformat() if as_of is not None else None
        from_iso = time_from.astimezone(UTC).isoformat() if time_from is not None else None
        to_iso = time_to.astimezone(UTC).isoformat() if time_to is not None else None
        identity_terms = _identity_terms(query)
        with self._store.read_connection() as connection:
            objects = _matching_objects(
                connection,
                terms,
                cap + 1,
                moment,
                clock,
                kind=kind,
                time_from=from_iso,
                time_to=to_iso,
                has_participants=has_participants,
                assertion_count_min=assertion_count_min,
                assertion_count_max=assertion_count_max,
                offset=offset,
            )
            assertions = _matching_assertions(
                connection,
                terms,
                cap + 1,
                moment,
                clock,
                predicate=predicate,
                time_from=from_iso,
                time_to=to_iso,
                offset=offset,
            )
            inquiries = _matching_inquiries(connection, terms, cap + 1, moment, clock)
            usage = _matching_name_usage(connection, identity_terms, cap + 1, moment, clock)
            if usage:
                objects = _merge_usage_subjects(connection, objects, usage, cap + 1)
            matched = objects[:cap]
            candidates = _match_candidates(connection, matched, identity_terms, usage)
        reasons = ["matched object aliases", "matched assertion text", "matched active inquiries"]
        if usage:
            reasons.append("matched name usage assertions")
        return _bundle(
            anchors=matched,
            assertions=assertions[:cap],
            inquiries=inquiries[:cap],
            candidates=candidates,
            reasons=reasons,
            store=self._store,
            evidence_limit=cap,
            moment=moment,
            clock=clock,
            truncated=any(len(items) > cap for items in (objects, assertions, inquiries)),
        )

    def search_count(
        self,
        query: str,
        *,
        as_of: datetime | None = None,
        clock: RecallClock = RecallClock.KNOWLEDGE,
        kind: str | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        predicate: str | None = None,
        has_participants: bool | None = None,
        assertion_count_min: int | None = None,
        assertion_count_max: int | None = None,
    ) -> dict[str, int]:
        """Return per-bucket totals for a structured search, without rows.

        The count mirrors exactly the filters ``search`` applies: ``kind``,
        ``has_participants`` and ``assertion_count_*`` constrain objects,
        ``predicate`` constrains assertions, and the time window constrains
        both. An empty query reports zeros (consistent with search's
        "empty query" bundle).
        """
        terms = _terms(query)
        if not terms:
            return {"objects": 0, "assertions": 0, "inquiries": 0}
        moment = as_of.astimezone(UTC).isoformat() if as_of is not None else None
        from_iso = time_from.astimezone(UTC).isoformat() if time_from is not None else None
        to_iso = time_to.astimezone(UTC).isoformat() if time_to is not None else None
        with self._store.read_connection() as connection:
            objects = _matching_objects_count(
                connection,
                terms,
                moment,
                clock,
                kind=kind,
                time_from=from_iso,
                time_to=to_iso,
                has_participants=has_participants,
                assertion_count_min=assertion_count_min,
                assertion_count_max=assertion_count_max,
            )
            assertions = _matching_assertions_count(
                connection,
                terms,
                moment,
                clock,
                predicate=predicate,
                time_from=from_iso,
                time_to=to_iso,
            )
            inquiries = _matching_inquiries_count(connection, terms, moment, clock)
        return {"objects": objects, "assertions": assertions, "inquiries": inquiries}

    def expand(
        self,
        object_ids: Sequence[str],
        *,
        depth: int = 1,
        limit: int = 30,
        event_limit: int = _EGO_EVENT_LIMIT,
        before: str | None = None,
        include_history: bool = False,
    ) -> MemoryBundle:
        """Return every assertion anchored on each frontier object, plus neighbors.

        Aggregates by subject: each frontier object returns ALL of its
        assertions — object-valued edges and literal assertions alike — so an
        object whose cognition is mostly literal is never empty (audit F4:
        47/49 objects expanded to nothing because the old SQL required
        ``object_id IS NOT NULL``). Assertions already superseded by a later
        one are retired from the expansion by default (B2-8), so the graph
        shows the current knowledge state, not every historical claim; pass
        *include_history* to surface retired claims beside current ones
        (design §4.5: optional historical state, default current). Only
        object-valued edges extend the frontier: literal rows are reported
        but spawn no neighbors. The ego sections stay current in both states
        — history rides the assertions list and the memory_read correction
        chain view.

        Args:
            object_ids: The root objects to expand from (deduplicated, capped).
            depth: How many object-edge hops to follow from the roots.
            limit: Maximum assertions to collect (also caps roots).
            event_limit: Independent cap for the entity-centric
                ``participated_events`` section (distinct from *limit*, which
                governs assertions/roots). Events beyond the cap are counted
                in ``omitted_counts``, never silently dropped.
            before: Optional event cursor (the ``event_next_cursor`` of a
                previous page, a ``{"t": <event_time_start>, "id": <event_id>}``
                JSON token) to page backward through participated events in
                ``event_time_start DESC, id ASC`` order.
            include_history: When True, retired (superseded) assertions are
                collected too, and their object edges still extend the
                frontier — the slice is the graph as recorded, not only the
                current surface. Defaults to False (current state).

        Returns:
            A bundle with the roots as anchors, the anchored assertions, the
            neighbor objects reached through object edges, the paths taken,
            and their evidence references. Under the ego view the bundle also
            carries the direct neighbor objects with edge predicates
            (``ego_neighbors``), the subject's current status assertions
            (``status_assertions``), the recent event timeline
            (``event_timeline``), the event-centric ``has_participant`` edges
            of an event root (``event_edges``, with role/qualifiers/evidence
            summaries) and the entity-centric recent participated events
            (``participated_events``, sorted ``event_time_start DESC, id
            ASC`` with the ``event_next_cursor`` for pagination). The
            ``omitted_counts`` map reports how many qualifying items each ego
            section cut over its cap; all sections are additive and capped.

        Raises:
            ValueError: If *depth*, *limit* or *event_limit* is negative or
                zero, or *before* is not a well-formed event cursor.

        """
        if depth < 0:
            raise ValueError("depth must be non-negative")
        cap = _limit(limit)
        event_cap = _limit(event_limit)
        cursor = _parse_event_cursor(before)
        roots = list(dict.fromkeys(object_ids))[:cap]
        paths = {identifier: [identifier] for identifier in roots}
        frontier = roots
        assertions: list[dict[str, Any]] = []
        neighbors: list[dict[str, Any]] = []
        output_paths: list[list[str]] = []
        collect_cap = cap + 1
        with self._store.read_connection() as connection:
            anchors = _objects_by_id(connection, roots)
            for _ in range(min(depth, _MAX_DEPTH)):
                next_frontier: list[str] = []
                for current in frontier:
                    current_only = (
                        " AND NOT EXISTS (SELECT 1 FROM assertions s"
                        " WHERE s.supersedes_id = assertions.id)"
                        if not include_history
                        else ""
                    )
                    rows = _rows(
                        connection.execute(
                            "SELECT * FROM assertions WHERE (subject_id = ? OR object_id = ?)"
                            f"{current_only} ORDER BY id LIMIT ?",
                            (current, current, collect_cap),
                        ).fetchall()
                    )
                    for row in rows:
                        if len(assertions) >= collect_cap:
                            break
                        if row["id"] in {item["id"] for item in assertions}:
                            continue
                        other = row["object_id"] if row["subject_id"] == current else row["subject_id"]
                        assertions.append(row)
                        if other is not None and other not in paths:
                            paths[other] = [*paths[current], other]
                            next_frontier.append(other)
                            neighbors.extend(_objects_by_id(connection, [other]))
                            output_paths.append(paths[other])
                    if len(assertions) >= collect_cap:
                        break
                frontier = next_frontier
                if not frontier or len(assertions) >= collect_cap:
                    break
        truncated = len(assertions) > cap
        assertions = assertions[:cap]
        retained_neighbor_ids = {
            identifier
            for row in assertions
            for identifier in (row.get("subject_id"), row.get("object_id"))
            if identifier and identifier not in roots
        }
        neighbors = [row for row in neighbors if row["id"] in retained_neighbor_ids]
        output_paths = [path for path in output_paths if path and path[-1] in retained_neighbor_ids]
        ego = _expand_ego(self._store, roots, event_limit=event_cap, before=cursor)
        return _bundle(
            anchors=anchors,
            assertions=assertions,
            neighbors=list({row["id"]: row for row in neighbors}.values()),
            paths=output_paths,
            reasons=["subject-anchored assertion expansion"],
            store=self._store,
            evidence_limit=cap,
            truncated=truncated or ego.truncated,
        ).model_copy(
            update={
                "ego_neighbors": ego.neighbors,
                "status_assertions": ego.statuses,
                "event_timeline": ego.timeline,
                "event_edges": ego.event_edges,
                "participated_events": ego.participated_events,
                "omitted_counts": ego.omitted_counts,
                "sort_basis": ego.sort_basis,
                "event_next_cursor": ego.event_next_cursor,
            }
        )

    def evidence(self, assertion_id: str) -> MemoryBundle:
        with self._store.read_connection() as connection:
            assertions = _rows(
                connection.execute("SELECT * FROM assertions WHERE id = ?", (assertion_id,)).fetchall()
            )  # noqa: E501
        return _bundle(
            assertions=assertions,
            reasons=["assertion evidence"],
            store=self._store,
            evidence_limit=_MAX_RESULTS,
        )

    def inquiries(
        self,
        object_id: str | None = None,
        limit: int = 20,
        inquiry_id: str | None = None,
    ) -> MemoryBundle:
        """Return open and dormant inquiries ordered open-first shallow-first, or one by id."""
        cap = _limit(limit)
        coverage_sql = (
            "COALESCE((SELECT MAX(CASE o.depth WHEN 'media' THEN 3 "
            "WHEN 'discussion' THEN 2 WHEN 'content' THEN 1 WHEN 'seen' THEN 0 END) "
            "FROM object_observations oo JOIN observations o ON o.id = oo.observation_id "
            "WHERE oo.object_id = inquiries.subject_id), -1)"
        )
        status_order = "CASE status WHEN 'open' THEN 0 ELSE 1 END"
        statement = (
            f"SELECT *, {coverage_sql} AS coverage_depth "
            "FROM inquiries WHERE status IN ('open','dormant')"
        )
        parameters: list[str] = []
        if object_id is not None:
            statement += " AND subject_id = ?"
            parameters.append(object_id)
        if inquiry_id is not None:
            statement = (
                f"SELECT *, {coverage_sql} AS coverage_depth "
                "FROM inquiries WHERE id = ?"
            )
            parameters = [inquiry_id]
        statement += f" ORDER BY {status_order}, coverage_depth, id LIMIT ?"
        with self._store.read_connection() as connection:
            rows = _rows(connection.execute(statement, (*parameters, cap + 1)).fetchall())
        truncated = len(rows) > cap
        rows = rows[:cap]
        _COVERAGE_NAMES = {-1: "none", 0: "seen", 1: "content", 2: "discussion", 3: "media"}
        for row in rows:
            row["coverage"] = _COVERAGE_NAMES.get(int(row.pop("coverage_depth", -1)), "none")
            row["dormant"] = row["status"] == "dormant"
        return MemoryBundle(
            inquiries=rows,
            reasons=["inquiry by id" if inquiry_id else "active inquiries"],
            truncated=truncated,
        )

    def overview(
        self,
        *,
        as_of: datetime | None = None,
        limit: int = _OVERVIEW_LIMIT,
        active_days: int = _OVERVIEW_ACTIVE_DAYS,
        cold_days: int = _OVERVIEW_COLD_DAYS,
    ) -> MemoryOverview:
        """Return a bounded, explainable and strictly read-only memory map.

        This deliberately does not call :meth:`inquiries`: that legacy helper
        demotes stale inquiries as a compatibility behavior, whereas an
        overview must never change durable inquiry status.  The two windows
        are arguments so callers can record them in offline fixtures instead
        of hiding product policy in a prompt.
        """
        if limit < 1 or active_days < 1 or cold_days < active_days:
            raise ValueError("overview limits must be positive and cold_days >= active_days")
        cap = min(limit, _OVERVIEW_LIMIT)
        moment = (as_of or datetime.now(UTC)).astimezone(UTC)
        active_since = moment - timedelta(days=active_days)
        cold_since = moment - timedelta(days=cold_days)
        moment_text = moment.isoformat()
        active_text = active_since.isoformat()
        cold_text = cold_since.isoformat()

        with self._store.read_connection() as connection:
            count_row = connection.execute(
                "SELECT (SELECT COUNT(*) FROM objects) AS objects, "
                "(SELECT COUNT(*) FROM assertions) AS assertions, "
                "(SELECT COUNT(*) FROM inquiries WHERE status = 'open') AS open_inquiries, "
                "(SELECT COUNT(*) FROM inquiries WHERE status = 'dormant') AS dormant_inquiries"
            ).fetchone()
            counts = MemoryOverviewCounts(
                **{
                key: int(value)
                for key, value in dict(count_row).items()
                }
            )
            inquiry_rows = []
            for status in ("open", "dormant"):
                inquiry_rows.extend(
                    _rows(
                        connection.execute(
                            "SELECT id, subject_id, prompt, status, attempt_count, last_attempted_at "
                            "FROM inquiries WHERE status = ? "
                            "ORDER BY attempt_count, COALESCE(last_attempted_at, ''), id LIMIT ?",
                            (status, _OVERVIEW_INQUIRY_POOL_PER_STATUS),
                        ).fetchall()
                    )
                )
            coverage = _overview_coverages(connection, inquiry_rows)
            active_events = _overview_active_events(connection, active_text, moment_text)
            initial_name_ids = set(active_events) | {str(row["subject_id"]) for row in inquiry_rows}
            names = {
                str(row["id"]): str(row["canonical_name"])
                for row in _object_name_rows(connection, initial_name_ids)
            }
            active_ids = set(active_events)
            active_inquiry_by_subject = _overview_subject_coverage(inquiry_rows, coverage, "open")

            active_candidates = [
                _object_candidate(
                    identifier,
                    names.get(identifier, identifier),
                    active_events[identifier],
                    active_inquiry_by_subject.get(identifier),
                )
                for identifier in sorted(active_ids)
            ]
            active_candidates.sort(
                key=lambda item: (
                    0 if item.inquiry_coverage is not None else 1,
                    _coverage_rank(item.inquiry_coverage.coverage if item.inquiry_coverage else None),
                    -_candidate_event_timestamp(item.facts).timestamp(),
                    -_candidate_latest_timestamp(item.facts).timestamp(),
                    item.id,
                )
            )

            reactivated_candidates = _overview_reactivated(
                connection,
                inquiry_rows,
                coverage,
                names,
                moment_text,
            )
            cold_candidates = _overview_cold_bridges(
                connection,
                active_ids,
                active_inquiry_by_subject,
                _overview_active_events(connection, cold_text, moment_text),
            )

        gap_candidates = [
            _inquiry_candidate(row, coverage[str(row["id"])])
            for row in inquiry_rows
            if row["status"] == "open"
        ]
        gap_candidates.sort(
            key=lambda item: (
                _coverage_rank(item.inquiry_coverage.coverage if item.inquiry_coverage else None),
                item.inquiry_coverage.answering_assertion_count if item.inquiry_coverage else 0,
                item.inquiry_coverage.attempt_count if item.inquiry_coverage else 0,
                -_candidate_latest_timestamp(item.facts).timestamp(),
                item.id,
            )
        )
        return MemoryOverview(
            as_of=moment,
            counts=counts,
            active_fronts=active_candidates[:cap],
            reactivated_fronts=reactivated_candidates[:cap],
            cold_bridges=cold_candidates[:cap],
            coverage_gaps=gap_candidates[:cap],
            truncated=any(
                len(items) > cap
                for items in (
                    active_candidates,
                    reactivated_candidates,
                    cold_candidates,
                    gap_candidates,
                )
            ),
        )
