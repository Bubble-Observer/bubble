"""Overview composition helpers for WorldRecall.overview (strict leaf)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from .contracts import InquiryCoverage, MemoryOverviewCandidate, MemoryOverviewFact
from .recall_common import _parse_iso

_OVERVIEW_AUDIT_POOL = 96
_OVERVIEW_EDGE_POOL = 120
_OVERVIEW_REACTIVATION_DAYS = 180
_OVERVIEW_DELTA_ITEM_POOL = 24
_OVERVIEW_NEIGHBOR_POOL = 24
_DEPTH_RANK = {None: -1, "seen": 0, "content": 1, "discussion": 2, "media": 3}
_DEPTH_NAME = {value: key for key, value in _DEPTH_RANK.items()}


def _overview_coverages(
    connection: sqlite3.Connection, inquiries: list[dict[str, Any]]
) -> dict[str, InquiryCoverage]:
    """Build inquiry-local coverage in three bounded aggregate reads.

    Object observation depth is returned only as context.  It never takes part
    in the coverage calculation or gap ordering, preventing a deeply read
    subject from falsely making an unresearched inquiry appear covered.
    """
    if not inquiries:
        return {}
    identifiers = [str(row["id"]) for row in inquiries]
    marks = ", ".join("?" for _ in identifiers)
    direct = {
        str(row["inquiry_id"]): dict(row)
        for row in connection.execute(
            "SELECT io.inquiry_id, COUNT(DISTINCT io.observation_id) AS direct_count, "
            "MAX(CASE o.depth WHEN 'media' THEN 3 WHEN 'discussion' THEN 2 "
            "WHEN 'content' THEN 1 WHEN 'seen' THEN 0 END) AS direct_depth, "
            "MAX(o.observed_at) AS last_direct_at "
            "FROM inquiry_observations io JOIN observations o ON o.id = io.observation_id "
            f"WHERE io.inquiry_id IN ({marks}) GROUP BY io.inquiry_id",
            identifiers,
        ).fetchall()
    }
    answers = {
        str(row["answers_inquiry_id"]): int(row["answer_count"])
        for row in connection.execute(
            "SELECT answers_inquiry_id, COUNT(*) AS answer_count FROM current_assertions "
            f"WHERE answers_inquiry_id IN ({marks}) GROUP BY answers_inquiry_id",
            identifiers,
        ).fetchall()
    }
    subject_ids = list(dict.fromkeys(str(row["subject_id"]) for row in inquiries))
    subject_marks = ", ".join("?" for _ in subject_ids)
    context_depth = {
        str(row["object_id"]): _DEPTH_NAME.get(int(row["depth"]))
        for row in connection.execute(
            "SELECT oo.object_id, MAX(CASE o.depth WHEN 'media' THEN 3 WHEN 'discussion' THEN 2 "
            "WHEN 'content' THEN 1 WHEN 'seen' THEN 0 END) AS depth "
            "FROM object_observations oo JOIN observations o ON o.id = oo.observation_id "
            f"WHERE oo.object_id IN ({subject_marks}) GROUP BY oo.object_id",
            subject_ids,
        ).fetchall()
    }
    output: dict[str, InquiryCoverage] = {}
    for row in inquiries:
        identifier = str(row["id"])
        direct_row = direct.get(identifier, {})
        direct_count = int(direct_row.get("direct_count", 0))
        direct_depth = _DEPTH_NAME.get(direct_row.get("direct_depth"))
        answer_count = answers.get(identifier, 0)
        attempts = int(row.get("attempt_count") or 0)
        if answer_count:
            label = "answered"
        elif direct_depth is not None:
            label = direct_depth
        elif attempts:
            label = "attempted"
        else:
            label = "none"
        output[identifier] = InquiryCoverage(
            direct_observation_count=direct_count,
            direct_max_depth=direct_depth,
            answering_assertion_count=answer_count,
            attempt_count=attempts,
            last_attempted_at=_parse_iso(row.get("last_attempted_at")),
            coverage=label,
            subject_context_depth=context_depth.get(str(row["subject_id"])),
        )
    return output


def _overview_active_events(
    connection: sqlite3.Connection, active_since: str, as_of: str
) -> dict[str, list[MemoryOverviewFact]]:
    """Collect event-time recent observation and cognition-change facts by object.

    Facts carry *event time* where the payload declares one: an object or an
    assertion uses its own event_time_start (commit time as fallback), and an
    observation uses the *stored* observed_at — the immutable first-observation
    time — instead of the re-read payload's fresh timestamp. Only facts at or
    after the window boundary survive, so a re-read of old content can neither
    refresh observed_at nor surface the object as active; observed_at only
    remains as the observation's own fallback clock (spec §7 "重读假装变热").
    Observation rows are introduced by the same durable delta that records
    their object/inquiry links.  Parsing the *bounded* audit window keeps
    this overview path on idx_world_audit_committed_at rather than scanning
    all observations or all object_observations; the stored observed_at lookup
    is a bounded point read on the collected window ids only.
    """
    boundary = _parse_iso(active_since) or datetime.min.replace(tzinfo=UTC)
    events: dict[str, list[MemoryOverviewFact]] = {}
    audit_rows = connection.execute(
        "SELECT committed_at, delta_json FROM world_audit WHERE committed_at >= ? AND committed_at <= ? "
        "ORDER BY committed_at DESC, commit_id DESC LIMIT ?",
        (active_since, as_of, _OVERVIEW_AUDIT_POOL),
    ).fetchall()
    assertion_ids: dict[str, datetime | None] = {}
    object_ids: dict[str, datetime | None] = {}
    observations: dict[str, datetime | None] = {}
    pending_links: list[dict[str, Any]] = []
    for row in audit_rows:
        try:
            delta = json.loads(row["delta_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(delta, dict):
            continue
        at = _parse_iso(row["committed_at"])
        for item in delta.get("objects", [])[:_OVERVIEW_DELTA_ITEM_POOL]:
            if isinstance(item, dict) and item.get("id"):
                object_ids[str(item["id"])] = _parse_iso(item.get("event_time_start")) or at
        for item in delta.get("assertions", [])[:_OVERVIEW_DELTA_ITEM_POOL]:
            if isinstance(item, dict) and item.get("id"):
                assertion_ids[str(item["id"])] = _parse_iso(item.get("event_time_start")) or at
        for item in delta.get("observations", [])[:_OVERVIEW_DELTA_ITEM_POOL]:
            if isinstance(item, dict) and item.get("id"):
                observations[str(item["id"])] = _parse_iso(item.get("observed_at")) or at
        for link in delta.get("observation_links", [])[:_OVERVIEW_DELTA_ITEM_POOL]:
            if isinstance(link, dict) and link.get("target_kind") == "object":
                pending_links.append(link)
    if observations:
        marks = ", ".join("?" for _ in observations)
        stored = {
            str(row["id"]): _parse_iso(row["observed_at"])
            for row in connection.execute(
                f"SELECT id, observed_at FROM observations WHERE id IN ({marks})", list(observations)
            ).fetchall()
        }
        for identifier, _payload_at in observations.items():
            stored_at = stored.get(identifier)
            if stored_at is not None:
                observations[identifier] = stored_at
    for link in pending_links:
        observation = str(link.get("observation_id", ""))
        target = str(link.get("target_id", ""))
        if observation and target and observation in observations:
            at = observations[observation]
            if at is not None and at >= boundary:
                events.setdefault(target, []).append(
                    MemoryOverviewFact(id=observation, kind="observation", at=at)
                )
    for identifier, at in object_ids.items():
        if at is not None and at >= boundary:
            events.setdefault(identifier, []).append(MemoryOverviewFact(id=identifier, kind="object", at=at))
    if assertion_ids:
        marks = ", ".join("?" for _ in assertion_ids)
        for row in connection.execute(
            f"SELECT id, subject_id, object_id FROM assertions WHERE id IN ({marks})", list(assertion_ids)
        ).fetchall():
            fact_at = assertion_ids[str(row["id"])]
            if fact_at is None or fact_at < boundary:
                continue
            fact = MemoryOverviewFact(
                id=str(row["id"]),
                kind="assertion",
                at=fact_at,
                related_id=str(row["subject_id"]),
            )
            events.setdefault(str(row["subject_id"]), []).append(fact)
            if row["object_id"]:
                events.setdefault(str(row["object_id"]), []).append(fact)
    return events


def _overview_subject_coverage(
    inquiries: list[dict[str, Any]], coverage: dict[str, InquiryCoverage], status: str
) -> dict[str, InquiryCoverage]:
    selected: dict[str, tuple[tuple[int, str], InquiryCoverage]] = {}
    for row in inquiries:
        if row["status"] != status:
            continue
        item = coverage[str(row["id"])]
        key = (_coverage_rank(item.coverage), str(row["id"]))
        subject = str(row["subject_id"])
        if subject not in selected or key < selected[subject][0]:
            selected[subject] = (key, item)
    return {subject: item for subject, (_, item) in selected.items()}


def _object_candidate(
    identifier: str,
    name: str,
    facts: list[MemoryOverviewFact],
    coverage: InquiryCoverage | None,
) -> MemoryOverviewCandidate:
    reasons: list[str] = []
    if any(fact.kind == "observation" for fact in facts):
        reasons.append("recent_observation")
    if any(fact.kind in {"assertion", "object"} for fact in facts):
        reasons.append("recent_cognition_change")
    return MemoryOverviewCandidate(
        id=identifier,
        kind="object",
        name=name,
        surfaced_because=reasons,
        facts=_unique_facts(facts),
        inquiry_coverage=coverage,
    )


def _inquiry_candidate(row: dict[str, Any], coverage: InquiryCoverage) -> MemoryOverviewCandidate:
    facts = [MemoryOverviewFact(id=str(row["id"]), kind="inquiry", at=coverage.last_attempted_at)]
    reasons: list[str] = ["underexplored"]
    if coverage.attempt_count >= 2:
        reasons.append("repeated_inquiry_point")
    return MemoryOverviewCandidate(
        id=str(row["id"]),
        kind="inquiry",
        name=str(row["prompt"])[:120],
        prompt=str(row["prompt"])[:240],
        surfaced_because=reasons,
        facts=facts,
        inquiry_coverage=coverage,
    )


def _overview_reactivated(
    connection: sqlite3.Connection,
    inquiries: list[dict[str, Any]],
    coverage: dict[str, InquiryCoverage],
    names: dict[str, str],
    as_of: str,
) -> list[MemoryOverviewCandidate]:
    candidates: list[MemoryOverviewCandidate] = []
    dormant_attempts = [
        str(row["last_attempted_at"])
        for row in inquiries
        if row["status"] == "dormant" and row.get("last_attempted_at")
    ]
    if not dormant_attempts:
        return candidates
    bounded_since = max(
        _parse_iso(min(dormant_attempts)) or datetime.min.replace(tzinfo=UTC),
        (_parse_iso(as_of) or datetime.now(UTC)) - timedelta(days=_OVERVIEW_REACTIVATION_DAYS),
    )
    activity = _overview_active_events(connection, bounded_since.isoformat(), as_of)
    for inquiry in inquiries:
        if inquiry["status"] != "dormant" or not inquiry.get("last_attempted_at"):
            continue
        since = str(inquiry["last_attempted_at"])
        subject = str(inquiry["subject_id"])
        neighbors = {subject}
        for edge in connection.execute(
            "SELECT subject_id, object_id FROM current_assertions WHERE subject_id = ? OR object_id = ? "
            "ORDER BY id LIMIT ?",
            (subject, subject, _OVERVIEW_NEIGHBOR_POOL),
        ).fetchall():
            neighbors.add(str(edge["subject_id"]))
            if edge["object_id"]:
                neighbors.add(str(edge["object_id"]))
        facts: list[MemoryOverviewFact] = []
        boundary = _parse_iso(since)
        for related in sorted(neighbors):
            for event in activity.get(related, []):
                if event.at is not None and boundary is not None and event.at > boundary:
                    facts.append(event.model_copy(update={"related_id": related}))
        facts = _unique_facts(facts)[:3]
        if not facts:
            continue
        reasons: list[str] = ["dormant_reactivated"]
        if any(fact.kind == "observation" for fact in facts):
            reasons.append("recent_observation")
        if any(fact.kind in {"assertion", "object"} for fact in facts):
            reasons.append("recent_cognition_change")
        if any(fact.related_id != subject for fact in facts):
            reasons.append("related_active_neighbor")
        candidates.append(
            MemoryOverviewCandidate(
                id=str(inquiry["id"]),
                kind="inquiry",
                name=str(inquiry["prompt"])[:120],
                prompt=str(inquiry["prompt"])[:240],
                surfaced_because=reasons,
                facts=facts,
                inquiry_coverage=coverage[str(inquiry["id"])],
            )
        )
    candidates.sort(key=lambda item: (-_candidate_latest_timestamp(item.facts).timestamp(), item.id))
    return candidates


def _overview_cold_bridges(
    connection: sqlite3.Connection,
    active_ids: set[str],
    coverage: dict[str, InquiryCoverage],
    recent_events: dict[str, list[MemoryOverviewFact]],
) -> list[MemoryOverviewCandidate]:
    """Find only direct, deterministic two-active-root bridges (no graph service)."""
    roots = sorted(active_ids)[:12]
    if len(roots) < 2:
        return []
    marks = ", ".join("?" for _ in roots)
    edges = connection.execute(
        f"SELECT id, subject_id, object_id FROM current_assertions WHERE subject_id IN ({marks}) "
        f"OR object_id IN ({marks}) ORDER BY id LIMIT {_OVERVIEW_EDGE_POOL}",
        (*roots, *roots),
    ).fetchall()
    bridges: dict[str, list[MemoryOverviewFact]] = {}
    root_sets: dict[str, set[str]] = {}
    for edge in edges:
        left, right = str(edge["subject_id"]), edge["object_id"]
        if right is None:
            # Literal assertions (no object target) can never connect two
            # active roots; their subject only matched the roots filter.
            continue
        if left in active_ids and right not in active_ids:
            bridge, root = right, left
        elif right in active_ids and left not in active_ids:
            bridge, root = left, right
        else:
            continue
        root_sets.setdefault(bridge, set()).add(root)
        bridges.setdefault(bridge, []).append(
            MemoryOverviewFact(id=str(edge["id"]), kind="assertion", related_id=root)
        )
    output: list[MemoryOverviewCandidate] = []
    for bridge, _roots_for_bridge in bridges.items():
        if len(root_sets[bridge]) < 2:
            continue
        if bridge in recent_events:
            continue
        active_facts = [
            fact
            for root in sorted(root_sets[bridge])[:2]
            for fact in recent_events.get(root, [])[:1]
        ]
        output.append(
            MemoryOverviewCandidate(
                id=bridge,
                kind="object",
                name=_object_name(connection, bridge),
                surfaced_because=["related_active_neighbor"],
                facts=_unique_facts([*bridges[bridge][:2], *active_facts])[:4],
                inquiry_coverage=coverage.get(bridge),
            )
        )
    output.sort(key=lambda item: item.id)
    return output


def _object_name_rows(connection: sqlite3.Connection, identifiers: set[str]) -> list[Any]:
    if not identifiers:
        return []
    ordered = sorted(identifiers)
    marks = ", ".join("?" for _ in ordered)
    return connection.execute(
        f"SELECT id, canonical_name FROM objects WHERE id IN ({marks}) ORDER BY id", ordered
    ).fetchall()


def _object_name(connection: sqlite3.Connection, identifier: str) -> str:
    row = connection.execute("SELECT canonical_name FROM objects WHERE id = ?", (identifier,)).fetchone()
    return str(row["canonical_name"]) if row is not None else identifier


def _coverage_rank(coverage: str | None) -> int:
    ranks = {
        "none": 0,
        "attempted": 1,
        "seen": 2,
        "content": 3,
        "discussion": 4,
        "media": 5,
        "answered": 6,
    }
    return ranks.get(coverage or "none", 0)


def _candidate_latest_timestamp(facts: list[MemoryOverviewFact]) -> datetime:
    times = [fact.at for fact in facts if fact.at is not None]
    return max(times) if times else datetime.min.replace(tzinfo=UTC)


def _candidate_event_timestamp(facts: list[MemoryOverviewFact]) -> datetime:
    """Latest *event time* among an object's facts, observed_at only as fallback.

    Objects and assertions declare their own event time; observations only
    carry observed_at. Ranking active fronts on event time first means a
    re-read of old material can never lift an object above genuinely recent
    events — observed_at stays as the tiebreak/fallback clock (spec §7).
    """
    times = [
        fact.at for fact in facts if fact.kind in {"object", "assertion"} and fact.at is not None
    ]
    if times:
        return max(times)
    return _candidate_latest_timestamp(facts)


def _unique_facts(facts: list[MemoryOverviewFact]) -> list[MemoryOverviewFact]:
    seen: set[tuple[str, str, str | None]] = set()
    output: list[MemoryOverviewFact] = []
    for fact in facts:
        key = (fact.id, fact.kind, fact.related_id)
        if key not in seen:
            seen.add(key)
            output.append(fact)
    return output
