"""Durable working graph: wake-isolated staging tables plus the patch protocol.

Plan §6.2 / §6.3, D-006 / D-012. Staging lives in the world SQLite database
(typed tables mirroring the formal shapes) so active staging survives process
restarts and resumes. Each patch item carries its own ``op_id`` idempotency
key; the ``staged_patch_receipts`` ledger separates the op key from the
host-issued ``staged_id`` (Core §6.2 hard requirement). Statuses live in
``active`` / ``finalized`` / ``abandoned``; drop is a status change, never a
physical delete. Only ``active`` rows of the current wake form the working
graph; every other wake's staging is invisible here.

This module is deliberately formal-store-agnostic on read side: the merged
view code in ``recall`` consumes ``read_active_staged`` output directly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .graph_contract import QUALIFIER_KEYS, normalize_identity_alias, normalize_qualifiers
from .graph_patch_contract import (
    INQUIRY_KINDS,
    OBJECT_KINDS,
    graph_patch_item_violation,
)
from .similarity import bigrams, jaccard, normalize
from .store import WorldStore

# Result statuses a patch item can land in.
OK = "ok"
REJECTED = "rejected"
CONFLICT = "conflict"
NEEDS_IDENTITY_RESOLUTION = "needs_identity_resolution"

_ACTIVE = "active"
_FINALIZED = "finalized"
_ABANDONED = "abandoned"
_STATUSES = {_ACTIVE, _FINALIZED, _ABANDONED}

# Object/inquiry kinds share the same source as the advertised Graph Patch
# contract; staging must never accept a value the Agent schema rejects.
_OBJECT_KINDS = set(OBJECT_KINDS)
_INQUIRY_KINDS = set(INQUIRY_KINDS)


def _bounded_target_text(object_ref: object, literal: object, limit: int = 48) -> str:
    """Render one assertion's target (object ref or bounded literal) for summaries."""
    if object_ref:
        return str(object_ref)
    text = str(literal)
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _payload_hash(item: Mapping[str, Any]) -> str:
    """Stable digest of one patch item for the idempotency ledger."""
    canonical = json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _rows(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _aliases(connection: sqlite3.Connection, row: Mapping[str, Any]) -> list[str]:
    return [str(alias) for alias in json.loads(row.get("aliases_json") or "[]")]


def _staged_seq(connection: sqlite3.Connection, wake_id: str) -> int:
    """Per-wake monotonically increasing staged id sequence (serialized writes)."""
    row = connection.execute(
        "SELECT COUNT(*) AS count FROM staged_objects WHERE wake_id = ?", (wake_id,)
    ).fetchone()
    return int(row["count"]) if row else 0


def _staged_object(connection: sqlite3.Connection, wake_id: str, reference: str) -> dict[str, Any] | None:
    """Resolve a reference to an active staged object (by staged_id or target_ref)."""
    rows = _rows(
        connection.execute(
            "SELECT * FROM staged_objects WHERE wake_id = ? AND status = 'active'"
            " AND (staged_id = ? OR target_ref = ?) LIMIT 1",
            (wake_id, reference, reference),
        ).fetchall()
    )
    return rows[0] if rows else None


def _staged_inquiry(connection: sqlite3.Connection, wake_id: str, reference: str) -> dict[str, Any] | None:
    """Resolve a reference to an active staged inquiry (by staged_id)."""
    rows = _rows(
        connection.execute(
            "SELECT * FROM staged_inquiries WHERE wake_id = ? AND status = 'active'"
            " AND staged_id = ? LIMIT 1",
            (wake_id, reference),
        ).fetchall()
    )
    return rows[0] if rows else None


def _dependent_refs(connection: sqlite3.Connection, wake_id: str, staged_id: str) -> list[dict[str, str]]:
    """Active staged rows of this wake that reference *staged_id*.

    Core §6.7: a dropped staged item must not stay referenced — drop is
    rejected while active dependents exist, so the Agent drops or updates
    them first (drop-time mechanical guarantee, slice 4 / D-013).
    """
    dependents: list[dict[str, str]] = []
    for row in _rows(
        connection.execute(
            "SELECT staged_id, subject_ref, object_ref, supersedes_ref FROM staged_assertions"
            " WHERE wake_id = ? AND status = 'active'",
            (wake_id,),
        ).fetchall()
    ):
        if staged_id in {row["subject_ref"], row["object_ref"], row["supersedes_ref"]}:
            dependents.append({"kind": "assertion", "staged_id": str(row["staged_id"])})
    for row in _rows(
        connection.execute(
            "SELECT staged_id, subject_ref, deepens_ref, target_ref FROM staged_inquiries"
            " WHERE wake_id = ? AND status = 'active'",
            (wake_id,),
        ).fetchall()
    ):
        # target_ref covers resolve rows: a staged resolve keeps its target
        # inquiry alive. Assertions answering a staged inquiry are deliberately
        # NOT dependents — dropping the answer assertion is legal and finalize
        # fails closed on the missing co-published answer (B4-8).
        if staged_id in {row["subject_ref"], row["deepens_ref"], row["target_ref"]}:
            dependents.append({"kind": "inquiry", "staged_id": str(row["staged_id"])})
    return dependents


def _formal_object_exists(connection: sqlite3.Connection, identifier: str) -> bool:
    return connection.execute("SELECT 1 FROM objects WHERE id = ?", (identifier,)).fetchone() is not None


def _formal_assertion_exists(connection: sqlite3.Connection, identifier: str) -> bool:
    return connection.execute("SELECT 1 FROM assertions WHERE id = ?", (identifier,)).fetchone() is not None


def _object_reference_resolves(connection: sqlite3.Connection, wake_id: str, reference: str) -> bool:
    """Resolve a subject/object ref against formal or active staged objects."""
    if _formal_object_exists(connection, reference):
        return True
    return _staged_object(connection, wake_id, reference) is not None


def _assertion_reference_resolves(connection: sqlite3.Connection, wake_id: str, reference: str) -> bool:
    """Resolve a supersedes/target ref against formal or active staged assertions."""
    if _formal_assertion_exists(connection, reference):
        return True
    rows = _rows(
        connection.execute(
            "SELECT 1 FROM staged_assertions WHERE wake_id = ? AND status = 'active'"
            " AND (staged_id = ? OR target_ref = ?) LIMIT 1",
            (wake_id, reference, reference),
        ).fetchall()
    )
    return bool(rows)


def _inquiry_reference_resolves(connection: sqlite3.Connection, wake_id: str, reference: str) -> bool:
    """Resolve a deepens ref against formal or active staged inquiries."""
    if _formal_inquiry_exists(connection, reference):
        return True
    return _staged_inquiry(connection, wake_id, reference) is not None


def _formal_inquiry_exists(connection: sqlite3.Connection, identifier: str) -> bool:
    return connection.execute("SELECT 1 FROM inquiries WHERE id = ?", (identifier,)).fetchone() is not None


def _identity_check(
    connection: sqlite3.Connection,
    wake_id: str,
    canonical: str,
    aliases: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (occupied, candidates) for one create/update identity pre-check.

    Plan §6.4: ACTIVE identity alias exact occupancy is a hard block
    (``occupied`` — no confirm_distinct override); exact canonical hits and
    staged-alias hits surface as ``candidates`` the Agent may confirm
    distinct or target instead. Formal aliases are compared in the shared
    normalized form; staged aliases stay raw (D-012 ⑤). ``exclude`` skips
    the item being examined (own staged_id / formal target_ref).
    """
    excluded = set(exclude)
    canonical_lower = canonical.casefold()

    occupied: list[dict[str, str]] = []
    normalized = [normalize_identity_alias(str(alias)) for alias in aliases if str(alias).strip()]
    if normalized:
        placeholders = ",".join("?" for _ in normalized)
        for row in _rows(
            connection.execute(
                "SELECT object_id, normalized_alias FROM identity_aliases"
                f" WHERE status = 'active' AND normalized_alias IN ({placeholders})",
                tuple(normalized),
            ).fetchall()
        ):
            object_id = str(row["object_id"])
            if object_id in excluded:
                continue
            name = connection.execute(
                "SELECT canonical_name FROM objects WHERE id = ?", (object_id,)
            ).fetchone()
            occupied.append(
                {
                    "id": object_id,
                    "alias": str(row["normalized_alias"]),
                    "canonical_name": str(name["canonical_name"]) if name else object_id,
                }
            )
    if occupied:
        return occupied, []

    candidates: list[dict[str, str]] = []
    for row in _rows(
        connection.execute(
            "SELECT id, canonical_name, kind FROM objects WHERE lower(canonical_name) = ?",
            (canonical_lower,),
        ).fetchall()
    ):
        if row["id"] not in excluded:
            candidates.append(
                {
                    "id": row["id"],
                    "canonical_name": row["canonical_name"],
                    "name": row["canonical_name"],
                    "kind": str(row["kind"]),
                    "basis": "canonical_exact",
                }
            )
    for row in _rows(
        connection.execute(
            "SELECT staged_id, target_ref, canonical_name, kind, aliases_json FROM staged_objects"
            " WHERE wake_id = ? AND status = 'active'",
            (wake_id,),
        ).fetchall()
    ):
        if row["staged_id"] in excluded or (row["target_ref"] and row["target_ref"] in excluded):
            continue
        if row["canonical_name"].casefold() == canonical_lower:
            candidates.append(
                {
                    "id": row["staged_id"],
                    "canonical_name": row["canonical_name"],
                    "name": row["canonical_name"],
                    "kind": str(row["kind"]),
                    "basis": "canonical_exact",
                }
            )
        for alias in _aliases(connection, row):
            if alias.casefold() in {item.casefold() for item in aliases}:
                if not any(entry["id"] == row["staged_id"] for entry in candidates):
                    candidates.append(
                        {
                            "id": row["staged_id"],
                            "canonical_name": row["canonical_name"],
                            "name": row["canonical_name"],
                            "kind": str(row["kind"]),
                            "basis": "staged_alias",
                        }
                    )
                break
    return occupied, candidates


_SIMILAR_NAME_THRESHOLD = 0.45
_CONTAINMENT_MIN_LEN = 2
_CONTAINMENT_FLOOR = 0.55
_HINT_LIMIT = 3
_IDENTITY_RESULT_CAP = 20


def _identity_feedback(key: str, values: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Bound model-visible identity matches without hiding that more exist."""
    window = [dict(value) for value in values[:_IDENTITY_RESULT_CAP]]
    output: dict[str, Any] = {key: window}
    if len(values) > len(window):
        output["omitted_counts"] = {key: len(values) - len(window)}
    return output


def _similar_name_hints(
    connection: sqlite3.Connection,
    wake_id: str,
    canonical: str,
    own_staged_id: str,
) -> list[dict[str, Any]]:
    """Advisory fuzzy-name hints for a successful object create (never blocking).

    The identity pre-check (``_identity_check``) only surfaces exact matches;
    an abbreviation or partial-name object (e.g. "TES" vs
    "滔搏电竞俱乐部 (TES)") gets no signal at all. This scan compares the new
    canonical against formal canonical names, active identity aliases, and
    this wake's active staged canonical names, and reports similar existing
    objects so the Agent can declare an identity alias, target an existing
    id, or confirm distinct. Purely advisory: results live in the success
    response, never in the verdict, and never trigger resolution.
    """
    hints: dict[str, tuple[float, str, str, str]] = {}
    norm = normalize(canonical)

    def consider(ref_id: str, name: str, kind: str, basis: str, score: float) -> None:
        if ref_id == own_staged_id:
            return
        if score > hints.get(ref_id, (0.0,))[0]:
            hints[ref_id] = (score, name, kind, basis)

    # Strongest signal: the new canonical is an active identity alias owner.
    if norm:
        for row in _rows(
            connection.execute(
                "SELECT a.object_id, o.canonical_name, o.kind FROM identity_aliases a"
                " JOIN objects o ON o.id = a.object_id"
                " WHERE a.status = 'active' AND a.normalized_alias = ?",
                (norm,),
            ).fetchall()
        ):
            consider(
                str(row["object_id"]),
                str(row["canonical_name"]),
                str(row["kind"] or ""),
                "canonical_matches_active_alias",
                1.0,
            )
    # Fuzzy: canonical-name containment or bigram similarity.
    for row in _rows(connection.execute("SELECT id, canonical_name, kind FROM objects").fetchall()):
        score, basis = _name_similarity(canonical, norm, str(row["canonical_name"]))
        if basis is not None:
            consider(str(row["id"]), str(row["canonical_name"]), str(row["kind"] or ""), basis, score)
    for row in _rows(
        connection.execute(
            "SELECT staged_id, canonical_name, kind FROM staged_objects"
            " WHERE wake_id = ? AND status = 'active'",
            (wake_id,),
        ).fetchall()
    ):
        score, basis = _name_similarity(canonical, norm, str(row["canonical_name"]))
        if basis is not None:
            consider(str(row["staged_id"]), str(row["canonical_name"]), str(row["kind"] or ""), basis, score)

    ranked = sorted(hints.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
    return [
        {
            "id": ref_id,
            "name": entry[1],
            "kind": entry[2],
            "basis": entry[3],
            "similarity": round(entry[0], 2),
        }
        for ref_id, entry in ranked[:_HINT_LIMIT]
    ]


def _name_similarity(canonical: str, norm: str, name: str) -> tuple[float, str | None]:
    """Score one existing name against the new canonical; None means no hint."""
    other = normalize(name)
    score = jaccard(bigrams(canonical), bigrams(name))
    if (
        len(norm) >= _CONTAINMENT_MIN_LEN
        and len(other) >= _CONTAINMENT_MIN_LEN
        and (norm in other or other in norm)
    ):
        return max(score, _CONTAINMENT_FLOOR), "name_containment"
    if score >= _SIMILAR_NAME_THRESHOLD:
        return score, "name_similar"
    return score, None


def _assertion_signature(payload: Mapping[str, Any]) -> str:
    """Block staged duplicates with an equivalent assertion signature.

    Same subject/predicate/object-or-literal/role/event window, plus
    non-empty qualifiers (mirrors the formal dedupe).
    """
    window = (
        f"{_normalized_event_time(payload.get('event_time_start'))}|"
        f"{_normalized_event_time(payload.get('event_time_end'))}"
        if payload.get("event_time_start") or payload.get("event_time_end")
        else ""
    )
    qualifiers = _comparison_qualifiers(payload.get("qualifiers"))
    qualifier_part = json.dumps(qualifiers, sort_keys=True) if qualifiers else ""
    target = payload.get("literal") if payload.get("literal") is not None else payload.get("object_ref")
    return json.dumps(
        [
            str(payload.get("subject_ref") or "").strip(),
            str(payload.get("predicate") or "").strip().casefold(),
            str(target).strip() if payload.get("object_ref") is not None else target,
            payload.get("epistemic_role") or "fact",
            window,
            qualifier_part,
        ],
        sort_keys=True,
    )


def _normalized_event_time(value: object) -> str:
    """Normalize one validated ISO instant so Z and +00:00 compare equally."""
    if value is None or not str(value).strip():
        return ""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC).isoformat()
    except ValueError:
        # Legacy/corrupt staging stays inspectable; finalize owns the typed
        # compile error and must remain the last defensive ring.
        return f"invalid:{value}"


def _comparison_qualifiers(value: object) -> dict[str, str]:
    """Mirror finalize's legacy-key tolerance for read-only comparisons."""
    if not isinstance(value, Mapping):
        return {}
    supported = {str(key): item for key, item in value.items() if str(key) in QUALIFIER_KEYS}
    try:
        return normalize_qualifiers(supported)
    except ValueError:
        # Invalid supported values still belong to finalize's compile-error
        # surface. A sentinel prevents a spurious collision without crashing
        # graph_inspect on durable historical staging.
        return {
            "__invalid_qualifiers__": json.dumps(
                supported, ensure_ascii=False, sort_keys=True, default=str
            )
        }


def _assertion_payload_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a formal or staged assertion row into the shared comparison shape."""
    return {
        "subject_ref": row.get("subject_id") or row.get("subject_ref"),
        "predicate": row["predicate"],
        "object_ref": row.get("object_id") or row.get("object_ref"),
        "literal": json.loads(row["literal_json"]) if row.get("literal_json") else None,
        "epistemic_role": row["epistemic_role"],
        "event_time_start": row.get("event_time_start"),
        "event_time_end": row.get("event_time_end"),
        "qualifiers": json.loads(row.get("qualifiers_json") or "{}"),
    }


def _exact_duplicate(
    connection: sqlite3.Connection,
    wake_id: str,
    payload: Mapping[str, Any],
    *,
    exclude_staged_id: str | None = None,
) -> str | None:
    """Return the id of an equivalent current formal assertion or active staged one.

    ``exclude_staged_id`` skips the assertion being examined so an update or
    re-check never self-reports as its own duplicate (slice 4 / D-013).
    """
    signature = _assertion_signature(payload)

    for row in _rows(
        connection.execute(
            "SELECT * FROM assertions a"
            " WHERE a.subject_id = ?"
            " AND NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)",
            (str(payload.get("subject_ref") or "").strip(),),
        ).fetchall()
    ):
        if _assertion_signature(_assertion_payload_from_row(row)) == signature:
            return str(row["id"])
    for row in _rows(
        connection.execute(
            "SELECT * FROM staged_assertions WHERE wake_id = ? AND status = 'active'"
            " AND subject_ref = ?",
            (wake_id, str(payload.get("subject_ref") or "").strip()),
        ).fetchall()
    ):
        if row["staged_id"] == exclude_staged_id:
            continue
        if _assertion_signature(_assertion_payload_from_row(row)) == signature:
            return str(row["staged_id"])
    return None


def _event_windows_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Treat missing bounds as open so unknown/overlapping relation spans collide."""
    left_start = _normalized_event_time(left.get("event_time_start")) or None
    left_end = _normalized_event_time(left.get("event_time_end")) or None
    right_start = _normalized_event_time(right.get("event_time_start")) or None
    right_end = _normalized_event_time(right.get("event_time_end")) or None
    bounds = (left_start, left_end, right_start, right_end)
    if any(value is not None and value.startswith("invalid:") for value in bounds):
        return False
    if left_start is not None and left_end is not None and left_start > left_end:
        return False
    if right_start is not None and right_end is not None and right_start > right_end:
        return False
    if left_end is not None and right_start is not None and left_end < right_start:
        return False
    return not (right_end is not None and left_start is not None and right_end < left_start)


def _relation_lane(payload: Mapping[str, Any]) -> tuple[str, str, str, str, str] | None:
    """Return relation identity without its revisable event window."""
    object_ref = payload.get("object_ref")
    if object_ref is None:
        return None
    qualifiers = _comparison_qualifiers(payload.get("qualifiers"))
    return (
        str(payload.get("subject_ref") or "").strip(),
        str(payload.get("predicate") or "").strip().casefold(),
        str(object_ref).strip(),
        str(payload.get("epistemic_role") or "fact"),
        json.dumps(qualifiers, sort_keys=True),
    )


def _supersedes_ref_lineage(
    connection: sqlite3.Connection,
    wake_id: str,
    supersedes_ref: object,
) -> set[str]:
    """Return a formal/staged supersede target plus its active staged ancestors."""
    current = str(supersedes_ref or "")
    lineage: set[str] = set()
    while current and current not in lineage:
        lineage.add(current)
        row = connection.execute(
            "SELECT supersedes_ref FROM staged_assertions "
            "WHERE wake_id = ? AND status = 'active' AND staged_id = ?",
            (wake_id, current),
        ).fetchone()
        current = str(row["supersedes_ref"] or "") if row is not None else ""
    return lineage


def _overlapping_relation(
    connection: sqlite3.Connection,
    wake_id: str,
    payload: Mapping[str, Any],
    *,
    exclude_staged_id: str | None = None,
    allowed_existing_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Find a parallel object relation whose semantic lane and time overlap.

    Exact signature twins are handled by ``_exact_duplicate``. This second
    guard catches the otherwise-silent case where only the event window
    changes (especially unknown -> known): that is a revision and must name
    the prior assertion with ``supersedes_ref``. Non-overlapping recurring
    episodes remain valid independent assertions.
    """
    lane = _relation_lane(payload)
    if lane is None:
        return None
    signature = _assertion_signature(payload)
    candidates = [
        *_rows(
            connection.execute(
                "SELECT * FROM assertions a WHERE a.subject_id = ? AND a.object_id = ?"
                " AND NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)",
                (lane[0], lane[2]),
            ).fetchall()
        ),
        *_rows(
            connection.execute(
                "SELECT * FROM staged_assertions WHERE wake_id = ? AND status = 'active'"
                " AND subject_ref = ? AND object_ref = ?",
                (wake_id, lane[0], lane[2]),
            ).fetchall()
        ),
    ]
    for row in candidates:
        existing_id = str(row.get("id") or row.get("staged_id"))
        if existing_id == exclude_staged_id or existing_id in (allowed_existing_ids or set()):
            continue
        candidate = _assertion_payload_from_row(row)
        if _assertion_signature(candidate) == signature:
            continue
        if _relation_lane(candidate) == lane and _event_windows_overlap(payload, candidate):
            return {
                "existing_id": existing_id,
                "existing_event_time_start": row.get("event_time_start"),
                "existing_event_time_end": row.get("event_time_end"),
            }
    return None


def apply_patch(
    store: WorldStore,
    wake_id: str,
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply one bounded batch of graph_patch items; each entry returns its own verdict.

    Batch semantics (Core §6.3): every item commits independently — an item
    already staged is never rolled back by a later item's failure; a failure
    returns its own rejected entry with a stable error code. The whole batch
    shares one write connection so the working graph stays internally
    consistent within the call.
    """
    results: list[dict[str, Any]] = []
    with store.write_connection() as connection:
        for index, item in enumerate(items):
            results.append(_apply_item(connection, wake_id, item, index=index))
    return results


def _apply_item(
    connection: sqlite3.Connection,
    wake_id: str,
    item: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    op_id = str(item.get("op_id") or "").strip()
    violation = graph_patch_item_violation(item, index=index)
    if violation is not None:
        # Contract rejection precedes both implementation dispatch and the
        # idempotency ledger: malformed semantic input has zero SQLite side
        # effects and cannot later be replayed as a successful patch.
        return {
            "op_id": op_id,
            "status": REJECTED,
            "error_code": violation.code,
            "field": violation.field,
            "message": violation.message,
            "action_hint": violation.action_hint,
        }
    if not op_id:
        return {"op_id": "", "status": REJECTED, "error_code": "missing_op_id"}
    kind = str(item.get("kind") or "")
    action = str(item.get("action") or "")
    # ── idempotency ledger: same wake + op_id + same payload replays the
    # original result; a different payload is an explicit conflict.
    payload_hash = _payload_hash(item)
    receipt = connection.execute(
        "SELECT payload_hash, result_json FROM staged_patch_receipts WHERE wake_id = ? AND op_id = ?",
        (wake_id, op_id),
    ).fetchone()
    if receipt is not None:
        if receipt["payload_hash"] != payload_hash:
            return {
                "op_id": op_id,
                "status": CONFLICT,
                "error_code": "op_id_reused",
                "message": f"op_id {op_id} already applied with a different payload",
            }
        return dict(json.loads(receipt["result_json"]))
    try:
        result = _execute_item(connection, wake_id, item)
    except (TypeError, ValueError, OverflowError) as error:
        # per-item exception isolation (Core §6.3.5): one malformed field
        # must not roll back items already staged in this batch
        result = {
            "op_id": op_id,
            "status": REJECTED,
            "error_code": "invalid_payload",
            "message": f"item could not be applied: {error}",
        }
    connection.execute(
        "INSERT INTO staged_patch_receipts (wake_id, op_id, payload_hash, result_json, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (wake_id, op_id, payload_hash, json.dumps(result, ensure_ascii=False), _now()),
    )
    return result


def _execute_item(
    connection: sqlite3.Connection,
    wake_id: str,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    kind = str(item.get("kind") or "")
    action = str(item.get("action") or "")
    op_id = str(item.get("op_id") or "")
    target_ref = str(item.get("target_ref") or "") or None
    payload = item.get("payload")
    if action == "drop" and payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return {"op_id": op_id, "status": REJECTED, "error_code": "invalid_payload"}
    if kind == "object":
        return _apply_object(connection, wake_id, op_id, action, target_ref, payload, item)
    if kind == "assertion":
        return _apply_assertion(connection, wake_id, op_id, action, target_ref, payload, item)
    if kind == "inquiry":
        return _apply_inquiry(connection, wake_id, op_id, action, target_ref, payload)
    return {"op_id": op_id, "status": REJECTED, "error_code": "unknown_kind"}


def _identity_verdict(
    connection: sqlite3.Connection,
    wake_id: str,
    canonical: str,
    aliases: Iterable[str],
    item: Mapping[str, Any],
    *,
    exclude: Iterable[str],
    prior_basis: Sequence[Mapping[str, str]] = (),
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]] | None]:
    """Run the identity pre-check; return (occupied, unresolved candidates, basis).

    ``occupied`` is a hard block (plan §6.4 ACTIVE alias exact occupancy)
    that no decision can override. A candidate already confirmed distinct
    (present in ``prior_basis``) stays resolved across updates — the item is
    not asked to re-confirm what it already grounded (D-013). A fresh
    confirm_distinct decision targeting one of the unresolved candidates
    produces the recorded basis entry.
    """
    occupied, candidates = _identity_check(connection, wake_id, canonical, aliases, exclude=exclude)
    if occupied:
        return occupied, [], None
    if not candidates:
        return [], [], None
    resolved = {str(entry.get("distinct_from")) for entry in prior_basis}
    unresolved = [entry for entry in candidates if entry["id"] not in resolved]
    decision = item.get("decision")
    confirmed = (
        isinstance(decision, dict)
        and decision.get("action") == "confirm_distinct"
        and decision.get("distinct_from") in {entry["id"] for entry in unresolved}
        and str(decision.get("basis") or "").strip()
    )
    if confirmed:
        basis: list[dict[str, str]] = [
            {"distinct_from": str(decision["distinct_from"]), "basis": str(decision["basis"]).strip()}
        ]
        return [], [], basis
    return [], unresolved, None


def _apply_object(
    connection: sqlite3.Connection,
    wake_id: str,
    op_id: str,
    action: str,
    target_ref: str | None,
    payload: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    if action == "drop":
        if target_ref is None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "missing_target_ref"}
        existing = _staged_object(connection, wake_id, target_ref)
        if existing is not None:
            dependents = _dependent_refs(connection, wake_id, existing["staged_id"])
            if dependents:
                return {
                    "op_id": op_id,
                    "status": REJECTED,
                    "error_code": "dependent_exists",
                    "dependents": dependents,
                    "message": (
                        f"object {existing['staged_id']} is referenced by active staged "
                        "items; drop or update them first (Core §6.7)"
                    ),
                }
            connection.execute(
                "UPDATE staged_objects SET status = 'abandoned', updated_at = ? WHERE staged_id = ?",
                (_now(), existing["staged_id"]),
            )
            return {
                "op_id": op_id,
                "status": OK,
                "staged_id": existing["staged_id"],
                "action": "dropped",
                "summary": f"dropped staged object {existing['staged_id']}",
            }
        # drop is a staging-status change: a formal-only target has no staged
        # item to retire in this slice (formal retirement arrives with finalize)
        return {
            "op_id": op_id,
            "status": REJECTED,
            "error_code": "target_unavailable",
            "message": f"no active staged object {target_ref} to drop",
        }
    canonical = str(payload.get("canonical_name") or "").strip()
    if not canonical:
        return {"op_id": op_id, "status": REJECTED, "error_code": "missing_canonical_name"}
    # kind is explicit on every object write (entity is the fallback, never
    # a silent default); update must not overwrite an existing kind with a
    # missing one either.
    raw_kind = payload.get("kind")
    if not raw_kind:
        return {"op_id": op_id, "status": REJECTED, "error_code": "missing_kind"}
    kind = str(raw_kind)
    if kind not in _OBJECT_KINDS:
        return {"op_id": op_id, "status": REJECTED, "error_code": "invalid_kind"}
    aliases = [str(alias) for alias in payload.get("aliases") or []]
    if action == "create":
        if target_ref is not None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "create_takes_no_target_ref"}
        occupied, candidates, basis = _identity_verdict(
            connection, wake_id, canonical, aliases, item, exclude=()
        )
        if occupied:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "alias_occupied",
                **_identity_feedback("occupied", occupied),
                "message": (
                    "Alias(es) are already held by active identity alias owners; "
                    "exact alias occupancy is a hard block (Core §6.4)"
                ),
                "hints": [
                    "Use a different alias, or reference/update the existing owner "
                    f"{occupied[0]['id']} instead."
                ],
            }
        if candidates:
            return {
                "op_id": op_id,
                "status": NEEDS_IDENTITY_RESOLUTION,
                "error_code": "identity_candidate_exists",
                **_identity_feedback("candidates", candidates),
                "hints": [
                    "Use memory_search / memory_read / memory_compare on the candidates, "
                    "then either target an existing id or retry with a "
                    "decision {action: confirm_distinct, distinct_from: <candidate id>, "
                    "basis: <why this is a different referent>}."
                ],
            }
        sequence = _staged_seq(connection, wake_id)
        staged_id = f"{wake_id}:s{sequence + 1}"
        connection.execute(
            "INSERT INTO staged_objects (staged_id, wake_id, status, target_ref, kind,"
            " canonical_name, type_key, domain_hints_json, provisional, identity_basis_json,"
            " event_time_start, event_time_end, aliases_json, created_at, updated_at, version)"
            " VALUES (?, ?, 'active', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (
                staged_id,
                wake_id,
                kind,
                canonical,
                payload.get("type_key"),
                json.dumps(payload.get("domain_hints") or []),
                1 if payload.get("provisional") else 0,
                json.dumps(basis or []),
                payload.get("event_time_start"),
                payload.get("event_time_end"),
                json.dumps(aliases),
                _now(),
                _now(),
            ),
        )
        result: dict[str, Any] = {
            "op_id": op_id,
            "status": OK,
            "staged_id": staged_id,
            "action": "created",
            "version": 1,
            "summary": f"object {canonical!r} staged as {staged_id}",
        }
        similar = _similar_name_hints(connection, wake_id, canonical, staged_id)
        if similar:
            result["similar_objects"] = similar
            result["hints"] = [
                "Similar existing objects are advisory hints, not blocks: declare an "
                "identity alias, target an existing id, or confirm_distinct."
            ]
        return result
    if action == "update":
        if target_ref is None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "missing_target_ref"}
        existing = _staged_object(connection, wake_id, target_ref)
        if existing is not None:
            expected = item.get("expected_version")
            if expected is not None and int(existing["version"]) != int(expected):
                return {
                    "op_id": op_id,
                    "status": REJECTED,
                    "error_code": "version_conflict",
                    "current_version": existing["version"],
                }
            exclude = {existing["staged_id"]}
            if existing["target_ref"]:
                exclude.add(str(existing["target_ref"]))
            prior_basis = json.loads(existing.get("identity_basis_json") or "[]")
            occupied, candidates, basis = _identity_verdict(
                connection, wake_id, canonical, aliases, item, exclude=exclude, prior_basis=prior_basis
            )
            if occupied:
                return {
                    "op_id": op_id,
                    "status": REJECTED,
                    "error_code": "alias_occupied",
                    **_identity_feedback("occupied", occupied),
                    "message": (
                        "Alias(es) are already held by active identity alias owners; "
                        "exact alias occupancy is a hard block (Core §6.4)"
                    ),
                }
            if candidates:
                return {
                    "op_id": op_id,
                    "status": NEEDS_IDENTITY_RESOLUTION,
                    "error_code": "identity_candidate_exists",
                    **_identity_feedback("candidates", candidates),
                    "hints": [
                        "The update collides with an existing identity. Confirm distinct "
                        "with a decision {action: confirm_distinct, distinct_from: "
                        "<candidate id>, basis: <why this is a different referent>} or "
                        "target an existing id instead."
                    ],
                }
            if basis:
                prior_basis = [
                    entry for entry in prior_basis if entry.get("distinct_from") != basis[0]["distinct_from"]
                ] + basis
            connection.execute(
                "UPDATE staged_objects SET canonical_name = ?, kind = ?, type_key = ?,"
                " domain_hints_json = ?, provisional = ?, identity_basis_json = ?,"
                " event_time_start = ?, event_time_end = ?, aliases_json = ?,"
                " updated_at = ?, version = version + 1 WHERE staged_id = ?",
                (
                    canonical,
                    kind,
                    payload.get("type_key"),
                    json.dumps(payload.get("domain_hints") or []),
                    1 if payload.get("provisional", existing["provisional"]) else 0,
                    json.dumps(prior_basis),
                    payload.get("event_time_start"),
                    payload.get("event_time_end"),
                    json.dumps(aliases),
                    _now(),
                    existing["staged_id"],
                ),
            )
            return {
                "op_id": op_id,
                "status": OK,
                "staged_id": existing["staged_id"],
                "action": "updated",
                "version": int(existing["version"]) + 1,
                "summary": (
                    f"object {canonical!r} updated as {existing['staged_id']} v{int(existing['version']) + 1}"
                ),
            }
        if not _formal_object_exists(connection, target_ref):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "target_unavailable",
                "message": f"no formal or active staged object {target_ref}",
            }
        expected = item.get("expected_version")
        formal_row = connection.execute("SELECT version FROM objects WHERE id = ?", (target_ref,)).fetchone()
        if expected is not None and int(formal_row["version"]) != int(expected):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "version_conflict",
                "current_version": formal_row["version"],
            }
        occupied, candidates, basis = _identity_verdict(
            connection, wake_id, canonical, aliases, item, exclude={target_ref}
        )
        if occupied:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "alias_occupied",
                **_identity_feedback("occupied", occupied),
                "message": (
                    "Alias(es) are already held by active identity alias owners; "
                    "exact alias occupancy is a hard block (Core §6.4)"
                ),
            }
        if candidates:
            return {
                "op_id": op_id,
                "status": NEEDS_IDENTITY_RESOLUTION,
                "error_code": "identity_candidate_exists",
                **_identity_feedback("candidates", candidates),
                "hints": [
                    "The update collides with an existing identity. Confirm distinct "
                    "with a decision {action: confirm_distinct, distinct_from: "
                    "<candidate id>, basis: <why this is a different referent>} or "
                    "target an existing id instead."
                ],
            }
        sequence = _staged_seq(connection, wake_id)
        staged_id = f"{wake_id}:s{sequence + 1}"
        connection.execute(
            "INSERT INTO staged_objects (staged_id, wake_id, status, target_ref, kind,"
            " canonical_name, type_key, domain_hints_json, provisional, identity_basis_json,"
            " event_time_start, event_time_end, aliases_json, created_at, updated_at, version,"
            " base_version)"
            " VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                staged_id,
                wake_id,
                target_ref,
                kind,
                canonical,
                payload.get("type_key"),
                json.dumps(payload.get("domain_hints") or []),
                1 if payload.get("provisional") else 0,
                json.dumps(basis or []),
                payload.get("event_time_start"),
                payload.get("event_time_end"),
                json.dumps(aliases),
                _now(),
                _now(),
                int(formal_row["version"]),
            ),
        )
        return {
            "op_id": op_id,
            "status": OK,
            "staged_id": staged_id,
            "action": "updated",
            "version": 1,
            "summary": f"object {canonical!r} overlay staged as {staged_id} for {target_ref}",
        }
    return {"op_id": op_id, "status": REJECTED, "error_code": "unknown_action"}


def _apply_assertion(
    connection: sqlite3.Connection,
    wake_id: str,
    op_id: str,
    action: str,
    target_ref: str | None,
    payload: Mapping[str, Any],
    item: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if action == "create":
        if target_ref is not None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "create_takes_no_target_ref"}
        subject_ref = str(payload.get("subject_ref") or "")
        predicate = str(payload.get("predicate") or "")
        if not subject_ref or not predicate:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "missing_assertion_fields",
            }
        if not _object_reference_resolves(connection, wake_id, subject_ref):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependency_unavailable",
                "message": f"subject {subject_ref} is neither a formal object nor an active staged object",
            }
        object_ref = payload.get("object_ref")
        if object_ref is not None and not _object_reference_resolves(connection, wake_id, str(object_ref)):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependency_unavailable",
                "message": f"object {object_ref} is neither a formal object nor an active staged object",
            }
        supersedes_ref = payload.get("supersedes_ref")
        if supersedes_ref is not None and not _assertion_reference_resolves(
            connection, wake_id, str(supersedes_ref)
        ):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependency_unavailable",
                "message": f"supersedes {supersedes_ref} is neither a formal nor a live staged assertion",
            }
        # Wave B (D-019): an answering assertion names the inquiry it answers.
        # answers_ref must resolve to a formal or live staged inquiry — the
        # official path for the Agent is to answer via an assertion and then
        # resolve the inquiry by naming this very assertion.
        answers_ref = payload.get("answers_ref")
        if answers_ref is not None and not _inquiry_reference_resolves(connection, wake_id, str(answers_ref)):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependency_unavailable",
                "message": (
                    f"answer target {answers_ref} is neither a formal inquiry nor an active staged inquiry"
                ),
            }
        # the host verifies evidence refs exist (plan §6.5): fabricated or
        # stale observation ids are a predictable Store-side loss
        evidence_ids = [str(ref) for ref in payload.get("evidence") or []]
        if evidence_ids:
            placeholders = ",".join("?" for _ in evidence_ids)
            rows = {
                str(row["id"])
                for row in connection.execute(
                    f"SELECT id FROM observations WHERE id IN ({placeholders})", tuple(evidence_ids)
                ).fetchall()
            }
            missing = [ref for ref in evidence_ids if ref not in rows]
            if missing:
                return {
                    "op_id": op_id,
                    "status": REJECTED,
                    "error_code": "evidence_unavailable",
                    "missing_evidence_ids": missing,
                    "message": (
                        "evidence refs do not match any stored observation; "
                        "link real observation ids or drop them"
                    ),
                }
        raw_qualifiers = payload.get("qualifiers")
        try:
            normalized_qualifiers = normalize_qualifiers(raw_qualifiers or {})
        except ValueError:
            # F-B contract drift: reject invented keys when the patch is
            # staged, not at finalize, so one bad key cannot fail a whole wake.
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "invalid_qualifiers",
                "message": (
                    "assertion qualifiers support only: role, language, "
                    "community, scope, granularity; move other notes "
                    "(e.g. source conflicts) into the literal"
                ),
            }
        comparison_payload = {**payload, "qualifiers": normalized_qualifiers}
        # A signature-identical correction may name its own collision target;
        # naming any other target must not bypass the duplicate gate.
        supersedes_lineage = _supersedes_ref_lineage(connection, wake_id, supersedes_ref)
        duplicate = _exact_duplicate(connection, wake_id, comparison_payload)
        if duplicate is not None and duplicate not in supersedes_lineage:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "exact_duplicate",
                "existing_id": duplicate,
                "hints": [
                    "The equivalent assertion already exists as current cognition. "
                    "Cancel, cite it, or create a correction with supersedes_ref."
                ],
            }
        overlap = _overlapping_relation(
            connection,
            wake_id,
            comparison_payload,
            allowed_existing_ids=supersedes_lineage,
        )
        if overlap is not None:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "overlapping_relation",
                **overlap,
                "hints": [
                    "A current relation already uses this subject, predicate, object, role, "
                    "and qualifiers over an overlapping or unknown time span. If this is a "
                    "refinement or correction, resend with supersedes_ref set to existing_id; "
                    "otherwise use a truthful distinct qualifier or a non-overlapping interval."
                ],
            }
        sequence = connection.execute(
            "SELECT COUNT(*) AS count FROM staged_assertions WHERE wake_id = ?", (wake_id,)
        ).fetchone()["count"]
        staged_id = f"{wake_id}:a{sequence + 1}"
        literal = payload.get("literal")
        connection.execute(
            "INSERT INTO staged_assertions (staged_id, wake_id, status, target_ref, subject_ref,"
            " predicate, object_ref, literal_json, epistemic_role, confidence, event_time_start,"
            " event_time_end, supersedes_ref, qualifiers_json, evidence_json, created_at,"
            " updated_at, version, answers_ref)"
            " VALUES (?, ?, 'active', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                staged_id,
                wake_id,
                subject_ref,
                predicate,
                object_ref,
                json.dumps(literal) if literal is not None else None,
                payload.get("epistemic_role") or "fact",
                float(payload.get("confidence", 0.8)),
                payload.get("event_time_start"),
                payload.get("event_time_end"),
                supersedes_ref,
                json.dumps(normalized_qualifiers),
                json.dumps(payload.get("evidence") or []),
                _now(),
                _now(),
                answers_ref,
            ),
        )
        return {
            "op_id": op_id,
            "status": OK,
            "staged_id": staged_id,
            "action": "created",
            "version": 1,
            "summary": (
                f"assertion {subject_ref} {predicate} "
                f"{_bounded_target_text(object_ref, literal)} staged as {staged_id}"
            ),
            **({"supersedes": str(supersedes_ref)} if supersedes_ref is not None else {}),
        }
    if action == "update":
        if target_ref is None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "missing_target_ref"}
        rows = _rows(
            connection.execute(
                "SELECT * FROM staged_assertions WHERE wake_id = ? AND status = 'active'"
                " AND (staged_id = ? OR target_ref = ?) LIMIT 1",
                (wake_id, target_ref, target_ref),
            ).fetchall()
        )
        if not rows:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "target_unavailable",
                "message": f"no active staged assertion {target_ref}",
            }
        existing = rows[0]
        expected = item.get("expected_version") if item else None
        if expected is not None and int(existing["version"]) != int(expected):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "version_conflict",
                "current_version": existing["version"],
            }
        raw_qualifiers = payload.get("qualifiers")
        try:
            normalized_qualifiers = normalize_qualifiers(raw_qualifiers or {})
        except ValueError:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "invalid_qualifiers",
                "message": (
                    "assertion qualifiers support only: role, language, "
                    "community, scope, granularity; move other notes "
                    "(e.g. source conflicts) into the literal"
                ),
            }
        comparison_payload = {**payload, "qualifiers": normalized_qualifiers}
        # the update may move the assertion onto an equivalent current
        # signature: mirror the create-time duplicate gate so a patch-accepted
        # item can never fail inspect/finalize on the same rule (slice 4 / D-013)
        duplicate = _exact_duplicate(
            connection,
            wake_id,
            comparison_payload,
            exclude_staged_id=existing["staged_id"],
        )
        if duplicate is not None:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "exact_duplicate",
                "existing_id": duplicate,
                "hints": [
                    "The updated assertion duplicates current cognition. Cancel, cite it, "
                    "or create a correction with supersedes_ref."
                ],
            }
        overlap = _overlapping_relation(
            connection,
            wake_id,
            comparison_payload,
            exclude_staged_id=str(existing["staged_id"]),
        )
        if overlap is not None:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "overlapping_relation",
                **overlap,
                "hints": [
                    "The update would create a parallel relation over an overlapping or "
                    "unknown time span; keep this item distinct or supersede the existing relation."
                ],
            }
        # Wave B: the update may (re)name the answered inquiry; when the field
        # is absent the existing answer link is kept — an update never silently
        # severs the answer a resolve is grounded in.
        if "answers_ref" in payload:
            answers_ref = payload.get("answers_ref")
            if answers_ref is not None and not _inquiry_reference_resolves(
                connection, wake_id, str(answers_ref)
            ):
                return {
                    "op_id": op_id,
                    "status": REJECTED,
                    "error_code": "dependency_unavailable",
                    "message": (
                        f"answer target {answers_ref} is neither a formal inquiry"
                        " nor an active staged inquiry"
                    ),
                }
        else:
            answers_ref = existing["answers_ref"]
        literal = payload.get("literal")
        connection.execute(
            "UPDATE staged_assertions SET literal_json = ?, epistemic_role = ?, confidence = ?,"
            " event_time_start = ?, event_time_end = ?, qualifiers_json = ?, evidence_json = ?,"
            " answers_ref = ?, updated_at = ?, version = version + 1 WHERE staged_id = ?",
            (
                json.dumps(literal) if literal is not None else None,
                payload.get("epistemic_role") or existing["epistemic_role"],
                float(payload.get("confidence", existing["confidence"])),
                payload.get("event_time_start"),
                payload.get("event_time_end"),
                json.dumps(normalized_qualifiers),
                json.dumps(payload.get("evidence") or []),
                answers_ref,
                _now(),
                existing["staged_id"],
            ),
        )
        return {
            "op_id": op_id,
            "status": OK,
            "staged_id": existing["staged_id"],
            "action": "updated",
            "version": int(existing["version"]) + 1,
            "summary": f"assertion {existing['staged_id']} updated to v{int(existing['version']) + 1}",
        }
    if action == "drop":
        if target_ref is None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "missing_target_ref"}
        rows = _rows(
            connection.execute(
                "SELECT * FROM staged_assertions WHERE wake_id = ? AND status = 'active'"
                " AND (staged_id = ? OR target_ref = ?) LIMIT 1",
                (wake_id, target_ref, target_ref),
            ).fetchall()
        )
        if not rows:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "target_unavailable",
                "message": f"no active staged assertion {target_ref} to drop",
            }
        dependents = _dependent_refs(connection, wake_id, str(rows[0]["staged_id"]))
        if dependents:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependent_exists",
                "dependents": dependents,
                "message": (
                    f"assertion {rows[0]['staged_id']} is still referenced by active "
                    "staged supersedes; drop or update them first (Core §6.7)"
                ),
            }
        connection.execute(
            "UPDATE staged_assertions SET status = 'abandoned', updated_at = ? WHERE staged_id = ?",
            (_now(), rows[0]["staged_id"]),
        )
        return {
            "op_id": op_id,
            "status": OK,
            "staged_id": rows[0]["staged_id"],
            "action": "dropped",
            "summary": f"dropped staged assertion {rows[0]['staged_id']}",
        }
    if action == "supersede":
        # supersede is create + supersedes_ref; require the target up front
        supersedes_ref = str(payload.get("supersedes_ref") or "")
        if not supersedes_ref:
            return {"op_id": op_id, "status": REJECTED, "error_code": "missing_supersedes_ref"}
        if not _assertion_reference_resolves(connection, wake_id, supersedes_ref):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependency_unavailable",
                "message": f"supersedes {supersedes_ref} is neither a formal nor a live staged assertion",
            }
        return _apply_assertion(
            connection,
            wake_id,
            op_id,
            "create",
            None,
            {**dict(payload), "supersedes_ref": supersedes_ref},
            item,
        )
    return {"op_id": op_id, "status": REJECTED, "error_code": "unknown_action"}


def _apply_inquiry(
    connection: sqlite3.Connection,
    wake_id: str,
    op_id: str,
    action: str,
    target_ref: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if action == "create":
        if target_ref is not None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "create_takes_no_target_ref"}
        subject_ref = str(payload.get("subject_ref") or "")
        prompt = str(payload.get("prompt") or "")
        rationale = str(payload.get("rationale") or "")
        if not subject_ref or not prompt or not rationale:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "missing_inquiry_fields",
            }
        if not _object_reference_resolves(connection, wake_id, subject_ref):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependency_unavailable",
                "message": f"subject {subject_ref} is neither a formal object nor an active staged object",
            }
        kind = str(payload.get("kind") or "factual")
        if kind not in _INQUIRY_KINDS:
            return {"op_id": op_id, "status": REJECTED, "error_code": "invalid_kind"}
        deepens_ref = payload.get("deepens_ref")
        if deepens_ref is not None and not _inquiry_reference_resolves(connection, wake_id, str(deepens_ref)):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependency_unavailable",
                "message": f"deepens {deepens_ref} is neither a formal nor a live staged inquiry",
            }
        # Wave B (D-019): the answer link lives on the answering assertion —
        # an inquiry create that carries answers_ref fails with a typed error
        # naming the official path, so a patched "inquiry as answer" can never
        # silently compile into an answer-less resolution.
        if payload.get("answers_ref") is not None:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "inquiry_create_takes_no_answers_ref",
                "message": (
                    "answers_ref belongs on the answering assertion (create it "
                    "with kind 'assertion' and answers_ref pointing at this "
                    "inquiry), then resolve the inquiry with action 'resolve' "
                    "naming that assertion"
                ),
            }
        sequence = connection.execute(
            "SELECT COUNT(*) AS count FROM staged_inquiries WHERE wake_id = ?", (wake_id,)
        ).fetchone()["count"]
        staged_id = f"{wake_id}:i{sequence + 1}"
        connection.execute(
            "INSERT INTO staged_inquiries (staged_id, wake_id, status, target_ref, subject_ref,"
            " prompt, rationale, kind, deepens_ref, answers_ref, created_at, updated_at, version)"
            " VALUES (?, ?, 'active', NULL, ?, ?, ?, ?, ?, NULL, ?, ?, 1)",
            (
                staged_id,
                wake_id,
                subject_ref,
                prompt,
                rationale,
                kind,
                deepens_ref,
                _now(),
                _now(),
            ),
        )
        return {
            "op_id": op_id,
            "status": OK,
            "staged_id": staged_id,
            "action": "created",
            "version": 1,
            "summary": f"inquiry staged as {staged_id}: {prompt[:48]}",
        }
    if action == "resolve":
        if target_ref is None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "missing_target_ref"}
        target = _staged_inquiry(connection, wake_id, target_ref)
        target_version: int
        if target is None:
            formal_target = connection.execute(
                "SELECT subject_id, prompt, rationale, version FROM inquiries WHERE id = ?",
                (target_ref,),
            ).fetchone()
            if formal_target is None:
                return {
                    "op_id": op_id,
                    "status": REJECTED,
                    "error_code": "target_unavailable",
                    "message": f"no active staged inquiry {target_ref}",
                }
            target_version = int(formal_target["version"])
            target_subject = str(formal_target["subject_id"])
            target_prompt = str(formal_target["prompt"])
            target_rationale = str(formal_target["rationale"])
        else:
            target_version = int(target["version"])
            target_subject = str(target["subject_ref"])
            target_prompt = str(target["prompt"])
            target_rationale = str(target["rationale"])
        answers_ref = payload.get("answers_ref")
        if answers_ref is None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "missing_answers_ref"}
        rows = _rows(
            connection.execute(
                "SELECT answers_ref FROM staged_assertions WHERE wake_id = ? AND status = 'active'"
                " AND (staged_id = ? OR target_ref = ?) LIMIT 1",
                (wake_id, answers_ref, answers_ref),
            ).fetchall()
        )
        if rows:
            declared_answer = rows[0]["answers_ref"]
        else:
            formal_answer = connection.execute(
                "SELECT answers_inquiry_id FROM assertions WHERE id = ?", (answers_ref,)
            ).fetchone()
            if formal_answer is None:
                return {
                    "op_id": op_id,
                    "status": REJECTED,
                    "error_code": "dependency_unavailable",
                    "message": (f"answer {answers_ref} is neither a formal nor an active staged assertion"),
                }
            declared_answer = formal_answer["answers_inquiry_id"]
        # the resolve must name an assertion that actually answers the target:
        # a staged assertion answering some other inquiry is not an answering
        # assertion for this one (Core §5, B2)
        target_identifier = target_ref if target is None else str(target["staged_id"])
        if declared_answer != target_identifier:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "not_an_answering_assertion",
                "message": (
                    f"assertion {answers_ref} answers {declared_answer}, not target {target_identifier}"
                ),
            }
        expected_version = payload.get("expected_version")
        try:
            expected_version = int(expected_version)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "invalid_expected_version",
            }
        if expected_version != target_version:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "version_conflict",
                "current_version": target_version,
            }
        sequence = connection.execute(
            "SELECT COUNT(*) AS count FROM staged_inquiries WHERE wake_id = ?", (wake_id,)
        ).fetchone()["count"]
        staged_id = f"{wake_id}:r{sequence + 1}"
        # The row mirrors its target's identity (subject/prompt/rationale) so
        # read surfaces show what is being resolved; kind 'resolution' is the
        # row-class marker the finalize layer uses to compile resolve rows.
        connection.execute(
            "INSERT INTO staged_inquiries (staged_id, wake_id, status, target_ref, subject_ref,"
            " prompt, rationale, kind, deepens_ref, answers_ref, created_at, updated_at, version)"
            " VALUES (?, ?, 'active', ?, ?, ?, ?, 'resolution', NULL, ?, ?, ?, ?)",
            (
                staged_id,
                wake_id,
                target_identifier,
                target_subject,
                target_prompt,
                target_rationale,
                answers_ref,
                _now(),
                _now(),
                expected_version,
            ),
        )
        return {
            "op_id": op_id,
            "status": OK,
            "staged_id": staged_id,
            "action": "resolved",
            "version": 1,
            "summary": f"resolved {target_identifier} via {answers_ref} as {staged_id}",
        }
    if action == "drop":
        if target_ref is None:
            return {"op_id": op_id, "status": REJECTED, "error_code": "missing_target_ref"}
        # Wave B: resolve rows carry target_ref, so a bare OR lookup could
        # abandon a resolve row when the operator meant its target inquiry.
        # Match the staged_id first, and only fall back to target_ref.
        rows = _rows(
            connection.execute(
                "SELECT * FROM staged_inquiries WHERE wake_id = ? AND status = 'active'"
                " AND staged_id = ? LIMIT 1",
                (wake_id, target_ref),
            ).fetchall()
        )
        if not rows:
            rows = _rows(
                connection.execute(
                    "SELECT * FROM staged_inquiries WHERE wake_id = ? AND status = 'active'"
                    " AND target_ref = ? LIMIT 1",
                    (wake_id, target_ref),
                ).fetchall()
            )
        if not rows:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "target_unavailable",
                "message": f"no active staged inquiry {target_ref}",
            }
        dependents = _dependent_refs(connection, wake_id, str(rows[0]["staged_id"]))
        if dependents:
            return {
                "op_id": op_id,
                "status": REJECTED,
                "error_code": "dependent_exists",
                "dependents": dependents,
                "message": (
                    f"inquiry {rows[0]['staged_id']} is still deepened by an active "
                    "staged inquiry; drop or update it first (Core §6.7)"
                ),
            }
        connection.execute(
            "UPDATE staged_inquiries SET status = 'abandoned', updated_at = ? WHERE staged_id = ?",
            (_now(), rows[0]["staged_id"]),
        )
        return {
            "op_id": op_id,
            "status": OK,
            "staged_id": rows[0]["staged_id"],
            "action": "dropped",
            "summary": f"dropped staged inquiry {rows[0]['staged_id']}",
        }
    return {"op_id": op_id, "status": REJECTED, "error_code": "unknown_action"}


def read_active_staged(
    store: WorldStore,
    wake_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Return the current wake's active staging: objects, assertions, inquiries.

    The merged view code overlays this over formal reads; it is the
    authoritative working-graph surface for the wake (Core §6.2). Non-active
    rows and every other wake's rows never appear.
    """
    objects: list[dict[str, Any]] = []
    assertions: list[dict[str, Any]] = []
    inquiries: list[dict[str, Any]] = []
    with store.read_connection() as connection:
        for row in _rows(
            connection.execute(
                "SELECT * FROM staged_objects WHERE wake_id = ? AND status = 'active' ORDER BY staged_id",
                (wake_id,),
            ).fetchall()
        ):
            row["aliases"] = _aliases(connection, row)
            row["identity_basis"] = json.loads(row.get("identity_basis_json") or "[]")
            row["provisional"] = bool(row.get("provisional"))
            row.pop("aliases_json", None)
            row.pop("identity_basis_json", None)
            objects.append(row)
        for row in _rows(
            connection.execute(
                "SELECT * FROM staged_assertions WHERE wake_id = ? AND status = 'active' ORDER BY staged_id",
                (wake_id,),
            ).fetchall()
        ):
            row["qualifiers"] = json.loads(row.get("qualifiers_json") or "{}")
            row["evidence"] = json.loads(row.get("evidence_json") or "[]")
            row["literal"] = json.loads(row["literal_json"]) if row.get("literal_json") is not None else None
            row.pop("qualifiers_json", None)
            row.pop("evidence_json", None)
            row.pop("literal_json", None)
            assertions.append(row)
        for row in _rows(
            connection.execute(
                "SELECT * FROM staged_inquiries WHERE wake_id = ? AND status = 'active' ORDER BY staged_id",
                (wake_id,),
            ).fetchall()
        ):
            inquiries.append(row)
    return {"objects": objects, "assertions": assertions, "inquiries": inquiries}


def staged_object_ids(store: WorldStore, wake_id: str) -> set[str]:
    """Staged ids (and covered formal target ids) of the wake's active objects."""
    ids: set[str] = set()
    with store.read_connection() as connection:
        for row in connection.execute(
            "SELECT staged_id, target_ref FROM staged_objects WHERE wake_id = ? AND status = 'active'",
            (wake_id,),
        ).fetchall():
            ids.add(str(row["staged_id"]))
            if row["target_ref"]:
                ids.add(str(row["target_ref"]))
    return ids
