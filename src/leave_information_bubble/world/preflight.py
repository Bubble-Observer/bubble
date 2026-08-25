"""Shared readiness rules for the working graph: graph_inspect, graph_diff.

Plan §6.9 / §7.6 / §7.7, slice 4. ``inspect_working_graph`` is the single
canonical rule set that graph_inspect reports and (in slice 5)
``finalize_graph`` will gate on: on the same unchanged snapshot, a patch
item the host accepted must never be rejected by inspect on the same rule —
blockers here mirror the patch-time gates (identity candidates, exact
signatures, reference resolution) plus the mechanical problems the Store
commit path will report (dangling evidence refs, isolated new objects,
unanchored events). Warnings are quality candidates only: they never block,
so an honest empty working graph stays publishable (§6.9).

``diff_working_graph`` answers "what did this wake change relative to the
formal graph": formal before / staged after per item, derived from final
staged rows (prior staged versions are not retained; the patch receipts
ledger keeps the op history). Everything is bounded and paged.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

from .graph_contract import normalize_identity_alias
from .staging import (
    _assertion_reference_resolves,
    _exact_duplicate,
    _identity_check,
    _object_reference_resolves,
    _overlapping_relation,
    _rows,
    _supersedes_ref_lineage,
)
from .store import WorldStore

# Bounds for inspect/diff responses.
_INSPECT_ITEMS_CAP = 200
_FINDINGS_CAP = 50
_DIFF_PAGE_CAP = 200
_ITEM_HISTORY_CAP = 50
_IDENTITY_DETAILS_CAP = 20

# Canonical blocker codes (plan §6.9): dangling_reference,
# abandoned_reference, identity_unresolved, alias_occupied, exact_duplicate,
# overlapping_relation, supersede_cycle, dangling_evidence_ref, zero_connection_object,
# event_identity_anchor, staged_alias_duplicate, wake_closed, plus the Wave B
# resolution gates (v18): target_unavailable, inquiry_already_resolved,
# inquiry_not_open, stale_expected_version, created_same_commit,
# invalid_expected_version, duplicate_resolve, no_answering_assertion.
# Warnings are quality candidates only and never block.


def _findings_key(code: str, ref: str) -> str:
    return f"{code} {ref}"


def _bounded_identity_details(
    key: str,
    values: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Bound one identity finding and state exactly how much was omitted."""
    window = [dict(value) for value in values[:_IDENTITY_DETAILS_CAP]]
    details: dict[str, object] = {key: window}
    if len(values) > len(window):
        details["omitted_counts"] = {key: len(values) - len(window)}
    return details


def _supersedes_chain_reaches(
    connection: sqlite3.Connection,
    wake_id: str,
    duplicate: str,
    checked_staged_id: str,
    checked_target_ref: str | None,
) -> bool:
    """Whether a staged row's supersede chain reaches the checked assertion.

    The exact-duplicate gate must not block a row that is being corrected: a
    correction (a staged row carrying ``supersedes_ref``) is the very pattern
    Core §6.7 says the equivalent-signature gate must allow, and metadata-only
    corrections are signature-identical to the row they supersede.
    """
    visited: set[str] = set()
    current = duplicate
    while current is not None and current not in visited:
        visited.add(current)
        row = connection.execute(
            "SELECT supersedes_ref, target_ref FROM staged_assertions"
            " WHERE wake_id = ? AND status = 'active' AND staged_id = ?",
            (wake_id, current),
        ).fetchone()
        if row is None or row["supersedes_ref"] is None:
            return False
        reference = str(row["supersedes_ref"])
        if reference in (checked_staged_id, checked_target_ref):
            return True
        current = reference
    return False


def inspect_working_graph(store: WorldStore, wake_id: str) -> dict[str, Any]:
    """Report the wake's working-graph state, blockers, and warnings.

    The report is read from the durable staging tables directly (never from
    tool memory), so on resume it shows the authoritative staged state
    (plan §7.6 recovery view). Returns a plain dict shaped for the
    graph_inspect tool payload.
    """
    with store.read_connection() as connection:
        objects = _rows(
            connection.execute(
                "SELECT * FROM staged_objects WHERE wake_id = ? ORDER BY staged_id", (wake_id,)
            ).fetchall()
        )
        assertions = _rows(
            connection.execute(
                "SELECT * FROM staged_assertions WHERE wake_id = ? ORDER BY staged_id", (wake_id,)
            ).fetchall()
        )
        inquiries = _rows(
            connection.execute(
                "SELECT * FROM staged_inquiries WHERE wake_id = ? ORDER BY staged_id", (wake_id,)
            ).fetchall()
        )
        active_objects = [row for row in objects if row["status"] == "active"]
        active_assertions = [row for row in assertions if row["status"] == "active"]
        active_inquiries = [row for row in inquiries if row["status"] == "active"]
        abandoned_ids = {
            str(row["staged_id"])
            for rows in (objects, assertions, inquiries)
            for row in rows
            if row["status"] == "abandoned"
        }

        blockers: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        seen: set[str] = set()
        cap_hit: dict[str, bool] = {"blockers": False, "warnings": False}

        def _add_blocker(code: str, ref: str, message: str, **extra: object) -> None:
            key = _findings_key(code, ref)
            if key in seen:
                return
            seen.add(key)
            if len(blockers) >= _FINDINGS_CAP:
                cap_hit["blockers"] = True
                return
            blockers.append({"code": code, "ref": ref, "message": message, **extra})

        def _add_warning(code: str, ref: str, message: str, **extra: object) -> None:
            key = _findings_key(code, ref)
            if key in seen:
                return
            seen.add(key)
            if len(warnings) >= _FINDINGS_CAP:
                cap_hit["warnings"] = True
                return
            warnings.append({"code": code, "ref": ref, "message": message, **extra})

        # ── terminal state: a finalized wake publishes no more staging ───────
        terminal = connection.execute(
            "SELECT commit_id, receipt_json FROM finalize_receipts WHERE wake_id = ?",
            (wake_id,),
        ).fetchone()
        if terminal is not None:
            active_total = len(active_objects) + len(active_assertions) + len(active_inquiries)
            if active_total:
                # active rows can only exist post-finalize (the finalizer
                # converges every active row in the same transaction), so
                # readiness must never bless them (G5a F-3)
                committed_at = json.loads(terminal["receipt_json"]).get("committed_at")
                _add_blocker(
                    "wake_closed",
                    wake_id,
                    f"wake {wake_id} already finalized at {committed_at}; {active_total} "
                    "active item(s) staged afterwards will never publish — start a new wake",
                    commit_id=str(terminal["commit_id"]),
                )

        # ── patch-time formal base: never silently rebase an overlay ───────
        for row in active_objects:
            target_ref = row["target_ref"]
            base_version = row.get("base_version")
            if target_ref is None:
                continue
            if base_version is None:
                _add_blocker(
                    "base_version_unknown",
                    str(row["staged_id"]),
                    f"formal object {target_ref} overlay has no trustworthy patch-time base version",
                    action_hint=(f"memory_read {target_ref}, then drop and re-patch this legacy overlay"),
                )
                continue
            formal = connection.execute(
                "SELECT version FROM objects WHERE id = ?", (str(target_ref),)
            ).fetchone()
            if formal is None:
                continue  # the existing dangling-reference rules diagnose absence
            current_version = int(formal["version"])
            if current_version != int(base_version):
                _add_blocker(
                    "stale_base",
                    str(row["staged_id"]),
                    f"formal object {target_ref} changed from version {base_version} "
                    f"to {current_version} after this overlay was patched",
                    base_version=int(base_version),
                    current_version=current_version,
                    action_hint=(f"memory_read {target_ref}, then drop and re-patch this overlay"),
                )

        # ── reference integrity: dangling / abandoned / evidence ─────────────
        for row in active_assertions:
            for ref, label in (
                (row["subject_ref"], "subject"),
                (row["object_ref"], "object"),
            ):
                if not ref:
                    continue
                if not _object_reference_resolves(connection, wake_id, ref):
                    if ref in abandoned_ids:
                        _add_blocker(
                            "abandoned_reference",
                            ref,
                            f"{label} {ref} of staged assertion {row['staged_id']} "
                            "refers to an abandoned staged item",
                        )
                    else:
                        _add_blocker(
                            "dangling_reference",
                            ref,
                            f"{label} {ref} of staged assertion {row['staged_id']} "
                            "is neither a formal object nor an active staged object",
                        )
            if row["supersedes_ref"]:
                target = row["supersedes_ref"]
                if not _assertion_reference_resolves(connection, wake_id, target):
                    if target in abandoned_ids:
                        _add_blocker(
                            "abandoned_reference",
                            target,
                            f"supersedes target {target} of staged assertion "
                            f"{row['staged_id']} is an abandoned staged assertion",
                        )
                    else:
                        _add_blocker(
                            "dangling_reference",
                            target,
                            f"supersedes target {target} of staged assertion "
                            f"{row['staged_id']} is neither a formal nor a live staged assertion",
                        )
            for evidence_id in json.loads(row["evidence_json"] or "[]"):
                exists = connection.execute(
                    "SELECT 1 FROM observations WHERE id = ?", (str(evidence_id),)
                ).fetchone()
                if exists is None:
                    _add_blocker(
                        "dangling_evidence_ref",
                        str(evidence_id),
                        f"evidence ref {evidence_id} of staged assertion {row['staged_id']} "
                        "matches no stored observation",
                    )
            if not json.loads(row["evidence_json"] or "[]"):
                _add_warning(
                    "no_evidence",
                    row["staged_id"],
                    f"staged assertion {row['staged_id']} carries no evidence links",
                )
        for row in active_inquiries:
            # Resolve rows (kind 'resolution') mirror their target's
            # subject/prompt; they are not inquiry creates and must not trip
            # the duplicate gates against their own target (Wave B, v18).
            if str(row["kind"]) == "resolution":
                continue
            ref = row["subject_ref"]
            if not _object_reference_resolves(connection, wake_id, ref):
                if ref in abandoned_ids:
                    _add_blocker(
                        "abandoned_reference",
                        ref,
                        f"subject {ref} of staged inquiry {row['staged_id']} "
                        "refers to an abandoned staged item",
                    )
                else:
                    _add_blocker(
                        "dangling_reference",
                        ref,
                        f"subject {ref} of staged inquiry {row['staged_id']} "
                        "is neither a formal object nor an active staged object",
                    )
            duplicate = connection.execute(
                "SELECT 1 FROM inquiries WHERE subject_id = ? AND prompt = ? AND resolved_at IS NULL",
                (ref, row["prompt"]),
            ).fetchone()
            if duplicate is not None:
                _add_warning(
                    "inquiry_duplicate",
                    row["staged_id"],
                    f"staged inquiry {row['staged_id']} duplicates an open formal inquiry "
                    f"on {ref}: {row['prompt'][:80]}",
                )
            for other in active_inquiries:
                # a resolution row mirrors its target by design
                if str(other["kind"]) == "resolution":
                    continue
                if other["staged_id"] != row["staged_id"] and (
                    other["subject_ref"] == ref and other["prompt"] == row["prompt"]
                ):
                    _add_warning(
                        "inquiry_duplicate",
                        row["staged_id"],
                        f"staged inquiry {row['staged_id']} duplicates staged inquiry "
                        f"{other['staged_id']}: {row['prompt'][:80]}",
                    )

        # ── Wave B: staged resolutions mirror the finalize resolve gates ────
        # The compile resolve pass gates on the same snapshot: the target must
        # still resolve at the frozen expected_version, the inquiry must not
        # already be resolved, one resolve per target per wake, and a
        # non-uncertainty answering assertion co-published in this wake
        # (B4-5 / D-019). Patch-time checks can drift (an answer dropped, a
        # formal version moved, a formal resolve landed), so inspect re-derives
        # the same gates the finalizer will run.
        staged_inquiry_versions = {
            str(row["staged_id"]): int(row["version"] or 0)
            for row in active_inquiries
            if str(row["kind"]) != "resolution"
        }
        delta_answers = {
            str(row["answers_ref"])
            for row in active_assertions
            if row.get("answers_ref") is not None
            and str(row.get("epistemic_role") or "fact") != "uncertainty"
        }
        seen_targets: dict[str, str] = {}
        for row in active_inquiries:
            if str(row["kind"]) != "resolution":
                continue
            staged_id = str(row["staged_id"])
            target_ref = row.get("target_ref")
            try:
                expected_version = int(row["version"] or 0)
            except (TypeError, ValueError):
                _add_blocker(
                    "invalid_expected_version",
                    staged_id,
                    f"resolve {staged_id} carries a non-integer expected_version",
                )
                continue
            formal_id = (
                str(target_ref) if target_ref is not None and target_ref in staged_inquiry_versions else None
            )
            if formal_id is None:
                formal_row = (
                    connection.execute(
                        "SELECT status, version, resolved_at FROM inquiries WHERE id = ?",
                        (target_ref,),
                    ).fetchone()
                    if target_ref is not None
                    else None
                )
                if formal_row is None:
                    _add_blocker(
                        "target_unavailable",
                        staged_id,
                        f"resolve {staged_id}: target {target_ref} compiles to "
                        "neither a staged inquiry of this wake nor a formal inquiry",
                    )
                    continue
                formal_id = str(target_ref)
                if formal_row["resolved_at"] is not None:
                    _add_blocker(
                        "inquiry_already_resolved",
                        staged_id,
                        f"resolve {staged_id}: inquiry {target_ref} is already resolved",
                    )
                    continue
                if str(formal_row["status"]) not in {"open", "dormant"}:
                    _add_blocker(
                        "inquiry_not_open",
                        staged_id,
                        f"resolve {staged_id}: inquiry {target_ref} is "
                        f"{formal_row['status']}, not open or dormant",
                    )
                    continue
                if int(formal_row["version"]) != expected_version:
                    _add_blocker(
                        "stale_expected_version",
                        staged_id,
                        f"resolve {staged_id}: stale expected_version "
                        f"{expected_version} for inquiry {target_ref} at version "
                        f"{formal_row['version']}",
                    )
                    continue
            else:
                if int(staged_inquiry_versions.get(str(target_ref), 0)) != expected_version:
                    _add_blocker(
                        "stale_expected_version",
                        staged_id,
                        f"resolve {staged_id}: stale expected_version "
                        f"{expected_version} for inquiry {target_ref} at version "
                        f"{staged_inquiry_versions.get(str(target_ref), '?')}",
                    )
                    continue
                if expected_version != 1:
                    _add_blocker(
                        "created_same_commit",
                        staged_id,
                        f"resolve {staged_id}: inquiry {target_ref} is created by "
                        "this same commit; expected_version must be 1",
                    )
                    continue
            if formal_id in seen_targets:
                _add_blocker(
                    "duplicate_resolve",
                    staged_id,
                    f"resolve {staged_id}: inquiry {formal_id} already has a "
                    f"resolve in this wake ({seen_targets[formal_id]})",
                )
                continue
            if formal_id not in delta_answers:
                _add_blocker(
                    "no_answering_assertion",
                    staged_id,
                    f"resolve {staged_id}: no non-uncertainty answering assertion "
                    f"in this wake for inquiry {formal_id}",
                )
                continue
            seen_targets[formal_id] = staged_id

        # ── identity resolution and exact duplicates ─────────────────────────
        for row in active_objects:
            exclude = {row["staged_id"]}
            if row["target_ref"]:
                exclude.add(str(row["target_ref"]))
            occupied, candidates = _identity_check(
                connection,
                wake_id,
                str(row["canonical_name"]),
                json.loads(row["aliases_json"] or "[]"),
                exclude=exclude,
            )
            if occupied:
                occupied_details = _bounded_identity_details("occupied", occupied)
                occupied_window = occupied[:_IDENTITY_DETAILS_CAP]
                occupied_suffix = (
                    f" (+{len(occupied) - len(occupied_window)} more)"
                    if len(occupied) > len(occupied_window)
                    else ""
                )
                _add_blocker(
                    "alias_occupied",
                    row["staged_id"],
                    f"staged object {row['staged_id']} uses identity alias(es) "
                    f"{[entry['alias'] for entry in occupied_window]}{occupied_suffix} "
                    "already occupied by "
                    "an active formal identity; no confirm override exists for "
                    "occupied aliases (plan §6.4) — update or drop the staged object",
                    **occupied_details,
                )
                continue
            resolved = {
                str(entry.get("distinct_from"))
                for entry in json.loads(row.get("identity_basis_json") or "[]")
            }
            unresolved = [entry for entry in candidates if entry["id"] not in resolved]
            if unresolved:
                candidate_details = _bounded_identity_details("candidates", unresolved)
                candidate_window = unresolved[:_IDENTITY_DETAILS_CAP]
                candidate_suffix = (
                    f" (+{len(unresolved) - len(candidate_window)} more)"
                    if len(unresolved) > len(candidate_window)
                    else ""
                )
                _add_blocker(
                    "identity_unresolved",
                    row["staged_id"],
                    f"staged object {row['staged_id']} collides with existing identity "
                    f"candidate(s) {[entry['id'] for entry in candidate_window]}"
                    f"{candidate_suffix}; confirm distinct "
                    "or target an existing id",
                    **candidate_details,
                )

        # ── same-wake staged alias duplication (G5a F-2) ─────────────────────
        # Two active staged objects of this wake compiling the same normalized
        # identity alias can never both commit (Store hard gate
        # _validate_object_aliases): even a both-sides confirm_distinct leaves
        # the state unpublishable, so inspect predicts the rejection instead
        # of blessing a snapshot finalize must refuse.
        staged_alias_owners: dict[str, set[str]] = {}
        for row in active_objects:
            for alias in json.loads(row["aliases_json"] or "[]"):
                if not str(alias).strip():
                    continue
                staged_alias_owners.setdefault(normalize_identity_alias(str(alias)), set()).add(
                    str(row["staged_id"])
                )
        for normalized_alias, owners in staged_alias_owners.items():
            if len(owners) < 2:
                continue
            owner_list = sorted(owners)
            _add_blocker(
                "staged_alias_duplicate",
                owner_list[0],
                f"identity alias {normalized_alias!r} is staged by "
                f"{', '.join(owner_list)}; two active staged objects can never "
                "both hold the alias formally — drop one or change its alias",
                owners=owner_list,
                alias=normalized_alias,
            )
        for row in active_assertions:
            payload = {
                "subject_ref": row["subject_ref"],
                "predicate": row["predicate"],
                "object_ref": row["object_ref"],
                "literal": json.loads(row["literal_json"]) if row["literal_json"] else None,
                "epistemic_role": row["epistemic_role"],
                "event_time_start": row["event_time_start"],
                "event_time_end": row["event_time_end"],
                "qualifiers": json.loads(row["qualifiers_json"] or "{}"),
            }
            duplicate = _exact_duplicate(
                connection,
                wake_id,
                payload,
                exclude_staged_id=row["staged_id"],
            )
            supersedes_ref = str(row["supersedes_ref"]) if row["supersedes_ref"] else None
            supersedes_lineage = _supersedes_ref_lineage(connection, wake_id, supersedes_ref)
            duplicate_is_target = duplicate is not None and duplicate in supersedes_lineage
            duplicate_is_overlay_source = duplicate is not None and _supersedes_chain_reaches(
                connection,
                wake_id,
                duplicate,
                str(row["staged_id"]),
                row["target_ref"],
            )
            if duplicate is not None and not duplicate_is_target and not duplicate_is_overlay_source:
                _add_blocker(
                    "exact_duplicate",
                    row["staged_id"],
                    f"staged assertion {row['staged_id']} duplicates current cognition "
                    f"assertion {duplicate}; create a correction with supersedes_ref instead",
                    existing_id=duplicate,
                )
            overlap = _overlapping_relation(
                connection,
                wake_id,
                payload,
                exclude_staged_id=str(row["staged_id"]),
                allowed_existing_ids=supersedes_lineage,
            )
            if overlap is not None:
                _add_blocker(
                    "overlapping_relation",
                    row["staged_id"],
                    f"staged assertion {row['staged_id']} creates a parallel relation over an "
                    "overlapping or unknown time span; supersede the existing assertion or "
                    "make the semantic lane/time span truthfully distinct",
                    **overlap,
                )

        # ── supersede cycle (defense: structurally impossible via graph_patch) ─
        staged_by_id = {str(row["staged_id"]): row for row in active_assertions}
        for row in active_assertions:
            if not row["supersedes_ref"]:
                continue
            visited: list[str] = []
            current: str | None = str(row["staged_id"])
            while current is not None:
                if current in visited:
                    _add_blocker(
                        "supersede_cycle",
                        row["staged_id"],
                        f"supersede chain of staged assertion {row['staged_id']} "
                        f"loops: {' -> '.join(visited + [current])}",
                    )
                    break
                visited.append(current)
                holder = staged_by_id.get(current)
                if holder is None or not holder["supersedes_ref"]:
                    current = None
                else:
                    current = str(holder["supersedes_ref"])
                    if current not in staged_by_id:
                        current = None  # formal target ends the chain

        # ── isolated new objects and unanchored events ───────────────────────
        subject_ids = {str(row["subject_ref"]) for row in active_assertions}
        relation_object_ids = {str(row["object_ref"]) for row in active_assertions if row["object_ref"]}
        for row in active_objects:
            if row["target_ref"] or int(row["provisional"] or 0):
                continue  # only newly created, non-provisional objects
            staged_id = str(row["staged_id"])
            if staged_id not in subject_ids and staged_id not in relation_object_ids:
                _add_blocker(
                    "zero_connection_object",
                    staged_id,
                    f"new non-provisional object {staged_id} "
                    f"({row['canonical_name']}) has no attribute or relation "
                    "connection in the active working graph",
                )
            if row["kind"] == "event" and staged_id not in relation_object_ids:
                anchored = any(
                    str(row2["subject_ref"]) == staged_id
                    and row2["object_ref"]
                    and str(row2["object_ref"]) != staged_id
                    for row2 in active_assertions
                )
                if not anchored:
                    _add_blocker(
                        "event_identity_anchor",
                        staged_id,
                        f"new non-provisional event {staged_id} "
                        f"({row['canonical_name']}) lacks any relation anchor "
                        "(participant, part_of, place, ...) explaining its identity",
                    )
            # Events are time anchors: a new non-provisional event without a
            # known time contradicts its own build condition (the contract
            # builds events only when the fact's time is known).
            if row["kind"] == "event" and not row["event_time_start"]:
                _add_blocker(
                    "event_time_missing",
                    staged_id,
                    f"new non-provisional event {staged_id} "
                    f"({row['canonical_name']}) lacks event_time_start; "
                    "events are time anchors and must carry their known time",
                )

        # ── stats and item summary ────────────────────────────────────────────
        def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
            counts = {"active": 0, "abandoned": 0, "finalized": 0}
            for row in rows:
                counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
            return counts

        stats = {
            "objects": _status_counts(objects),
            "assertions": _status_counts(assertions),
            "inquiries": _status_counts(inquiries),
        }
        active_total = sum(counts["active"] for counts in stats.values())

        items: list[dict[str, Any]] = []
        for row in active_objects:
            items.append(
                {
                    "staged_id": row["staged_id"],
                    "kind": "object",
                    "action": "created" if not row["target_ref"] else "updated",
                    "label": row["canonical_name"],
                    "target_ref": row["target_ref"],
                    "provisional": bool(row["provisional"]),
                    "version": row["version"],
                }
            )
        for row in active_assertions:
            predicate = row["predicate"]
            literal = json.loads(row["literal_json"]) if row["literal_json"] else None
            label = (
                f"{row['subject_ref']} {predicate} {row['object_ref']}"
                if row["object_ref"]
                else f"{row['subject_ref']} {predicate} {literal!r}"
            )
            action = (
                "supersede" if row["supersedes_ref"] else ("created" if not row["target_ref"] else "updated")
            )
            items.append(
                {
                    "staged_id": row["staged_id"],
                    "kind": "assertion",
                    "action": action,
                    "label": label[:160],
                    "target_ref": row["target_ref"],
                    "answers_ref": row["answers_ref"],
                    "version": row["version"],
                }
            )
        for row in active_inquiries:
            action = "resolved" if str(row["kind"]) == "resolution" else "created"
            items.append(
                {
                    "staged_id": row["staged_id"],
                    "kind": "inquiry",
                    "action": action,
                    "label": row["prompt"][:160],
                    "target_ref": row["target_ref"],
                    "deepens_ref": row["deepens_ref"],
                    "answers_ref": row["answers_ref"],
                    "version": row["version"],
                }
            )
        items.sort(key=lambda entry: (entry["version"], entry["staged_id"]))
        items_truncated = len(items) > _INSPECT_ITEMS_CAP
        items = items[:_INSPECT_ITEMS_CAP]

        return {
            "wake_id": wake_id,
            "readiness": "ready" if not blockers else "blocked",
            "stats": stats,
            "active_total": active_total,
            "blockers": blockers,
            "warnings": warnings,
            "blockers_truncated": cap_hit["blockers"],
            "warnings_truncated": cap_hit["warnings"],
            "items": items,
            "items_truncated": items_truncated,
            "note": "Authoritative staged state; survives restarts (resume view).",
        }


def staged_item_history(store: WorldStore, wake_id: str, staged_id: str) -> dict[str, Any] | None:
    """Return one staged item's current state plus its patch-receipt history.

    Receipts carry the original op_id and the result verdict per application
    (the ledger separates the op key from the host-issued staged id, Core
    §6.2), so this reconstructs the item's write history without a per-row
    version log.
    """
    with store.read_connection() as connection:
        row = None
        for table, select in (
            ("staged_objects", "staged_id, status, canonical_name, kind, version, target_ref"),
            (
                "staged_assertions",
                "staged_id, status, predicate, version, target_ref, supersedes_ref, answers_ref",
            ),
            (
                "staged_inquiries",
                "staged_id, status, prompt, kind, version, target_ref, deepens_ref, answers_ref",
            ),
        ):
            found = connection.execute(
                f"SELECT {select} FROM {table} WHERE wake_id = ? AND staged_id = ?",
                (wake_id, staged_id),
            ).fetchone()
            if found is not None:
                row = dict(found)
                row["kind"] = table.removeprefix("staged_")
                break
        if row is None:
            return None
        needle = f'"staged_id": "{staged_id}"'
        ops: list[dict[str, Any]] = []
        for receipt in connection.execute(
            "SELECT op_id, created_at, result_json FROM staged_patch_receipts"
            " WHERE wake_id = ? ORDER BY created_at",
            (wake_id,),
        ).fetchall():
            result = json.loads(receipt["result_json"])
            if needle in receipt["result_json"] and result.get("staged_id") == staged_id:
                ops.append(
                    {
                        "op_id": receipt["op_id"],
                        "action": result.get("action"),
                        "status": result.get("status"),
                        "applied_at": receipt["created_at"],
                    }
                )
        total = len(ops)
        window = ops[-_ITEM_HISTORY_CAP:]
        return {
            "item": row,
            "ops": window,
            "ops_total": total,
            "ops_truncated": total > len(window),
        }


def diff_working_graph(
    store: WorldStore,
    wake_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Page the current wake's changes relative to the formal graph.

    Each entry carries a formal ``before`` (when the target exists formally)
    and the staged ``after``; drops render as before-only with after null.
    Prior staged versions are not retained, so revised staged items show
    their current state with a history note. The response is explicitly
    marked unpublished — formal commit happens only at finalize.
    """
    page_size = min(max(int(limit), 1), _DIFF_PAGE_CAP)
    page_offset = max(int(offset), 0)
    with store.read_connection() as connection:
        active_objects = _rows(
            connection.execute(
                "SELECT * FROM staged_objects WHERE wake_id = ? AND status = 'active'"
                " ORDER BY created_at, staged_id",
                (wake_id,),
            ).fetchall()
        )
        active_assertions = _rows(
            connection.execute(
                "SELECT * FROM staged_assertions WHERE wake_id = ? AND status = 'active'"
                " ORDER BY created_at, staged_id",
                (wake_id,),
            ).fetchall()
        )
        active_inquiries = _rows(
            connection.execute(
                "SELECT * FROM staged_inquiries WHERE wake_id = ? AND status = 'active'"
                " ORDER BY created_at, staged_id",
                (wake_id,),
            ).fetchall()
        )
        abandoned_objects = _rows(
            connection.execute(
                "SELECT * FROM staged_objects WHERE wake_id = ? AND status = 'abandoned'"
                " ORDER BY created_at, staged_id",
                (wake_id,),
            ).fetchall()
        )
        abandoned_assertions = _rows(
            connection.execute(
                "SELECT * FROM staged_assertions WHERE wake_id = ? AND status = 'abandoned'"
                " ORDER BY created_at, staged_id",
                (wake_id,),
            ).fetchall()
        )
        abandoned_inquiries = _rows(
            connection.execute(
                "SELECT * FROM staged_inquiries WHERE wake_id = ? AND status = 'abandoned'"
                " ORDER BY created_at, staged_id",
                (wake_id,),
            ).fetchall()
        )

        entries: list[dict[str, Any]] = []
        by_action: dict[str, int] = {}
        by_kind: dict[str, int] = {}

        def _count(action: str, kind: str) -> None:
            by_action[action] = by_action.get(action, 0) + 1
            by_kind[kind] = by_kind.get(kind, 0) + 1

        def _object_after(row: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "canonical_name": row["canonical_name"],
                "kind": row["kind"],
                "provisional": bool(row["provisional"]),
                "type_key": row["type_key"],
                "aliases": json.loads(row["aliases_json"] or "[]"),
                "event_time_start": row["event_time_start"],
                "event_time_end": row["event_time_end"],
            }

        def _assertion_after(row: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "subject_ref": row["subject_ref"],
                "predicate": row["predicate"],
                "object_ref": row["object_ref"],
                "literal": json.loads(row["literal_json"]) if row["literal_json"] else None,
                "epistemic_role": row["epistemic_role"],
                "confidence": row["confidence"],
                "qualifiers": json.loads(row["qualifiers_json"] or "{}"),
                "evidence": json.loads(row["evidence_json"] or "[]"),
                "answers_ref": row.get("answers_ref"),
            }

        def _inquiry_after(row: Mapping[str, Any]) -> dict[str, Any]:
            is_resolution = str(row["kind"]) == "resolution"
            after: dict[str, Any] = {
                "subject_ref": row["subject_ref"],
                "prompt": row["prompt"],
                "rationale": row["rationale"],
                "kind": row["kind"],
                "status": "resolved" if is_resolution else "open",
                "version": int(row["version"]) + 1 if is_resolution else int(row["version"]),
                "deepens_ref": row.get("deepens_ref"),
                "answers_ref": row.get("answers_ref"),
            }
            if is_resolution:
                after["expected_version"] = int(row["version"])
            return after

        def _resolution_before(row: Mapping[str, Any]) -> dict[str, Any]:
            target_ref = str(row["target_ref"])
            staged_target = next(
                (
                    target
                    for target in active_inquiries
                    if str(target["staged_id"]) == target_ref and str(target["kind"]) != "resolution"
                ),
                None,
            )
            if staged_target is not None:
                return _inquiry_after(staged_target)
            formal_target = connection.execute(
                "SELECT id, subject_id, prompt, rationale, status, version, deepens_id "
                "FROM inquiries WHERE id = ?",
                (target_ref,),
            ).fetchone()
            if formal_target is None:
                return {"missing": True}
            return {
                "id": formal_target["id"],
                "subject_ref": formal_target["subject_id"],
                "prompt": formal_target["prompt"],
                "rationale": formal_target["rationale"],
                "status": formal_target["status"],
                "version": formal_target["version"],
                "deepens_ref": formal_target["deepens_id"],
                "answers_ref": None,
            }

        for row in active_objects:
            staged_id = str(row["staged_id"])
            if row["target_ref"]:
                before = connection.execute(
                    "SELECT id, canonical_name, kind, provisional, version FROM objects WHERE id = ?",
                    (row["target_ref"],),
                ).fetchone()
                entries.append(
                    {
                        "kind": "object",
                        "action": "update",
                        "staged_id": staged_id,
                        "target_ref": row["target_ref"],
                        "before": dict(before) if before else {"missing": True},
                        "after": _object_after(row),
                    }
                )
            else:
                entries.append(
                    {
                        "kind": "object",
                        "action": "create",
                        "staged_id": staged_id,
                        "target_ref": None,
                        "before": None,
                        "after": _object_after(row),
                    }
                )
            _count("update" if row["target_ref"] else "create", "object")
        for row in active_assertions:
            staged_id = str(row["staged_id"])
            if row["supersedes_ref"]:
                target = row["supersedes_ref"]
                formal_before = connection.execute(
                    "SELECT * FROM assertions WHERE id = ?", (target,)
                ).fetchone()
                if formal_before is not None:
                    # formal evidence lives in assertion_evidence, not a row column
                    evidence_ids = [
                        str(link["observation_id"])
                        for link in connection.execute(
                            "SELECT observation_id FROM assertion_evidence"
                            " WHERE assertion_id = ? ORDER BY linked_at, rowid",
                            (target,),
                        ).fetchall()
                    ]
                    before = _assertion_after(
                        {
                            **dict(formal_before),
                            "subject_ref": formal_before["subject_id"],
                            "object_ref": formal_before["object_id"],
                            "literal_json": formal_before["literal_json"],
                            "qualifiers_json": formal_before["qualifiers_json"],
                            "evidence_json": json.dumps(evidence_ids),
                            "answers_ref": formal_before["answers_inquiry_id"],
                        }
                    )
                    before["id"] = target
                else:
                    staged_before = next(
                        (entry for entry in active_assertions if entry["staged_id"] == target), None
                    )
                    before = _assertion_after(staged_before) if staged_before else {"missing": True}
                entries.append(
                    {
                        "kind": "assertion",
                        "action": "supersede",
                        "staged_id": staged_id,
                        "target_ref": target,
                        "before": before,
                        "after": _assertion_after(row),
                    }
                )
                _count("supersede", "assertion")
            else:
                entries.append(
                    {
                        "kind": "assertion",
                        "action": "create",
                        "staged_id": staged_id,
                        "target_ref": None,
                        "before": None,
                        "after": _assertion_after(row),
                        "history_note": (
                            "revised in this wake; prior staged versions are not retained"
                            if row["version"] > 1
                            else None
                        ),
                    }
                )
                _count("create", "assertion")
        for row in active_inquiries:
            is_resolution = str(row["kind"]) == "resolution"
            action = "resolved" if is_resolution else "create"
            entries.append(
                {
                    "kind": "inquiry",
                    "action": action,
                    "staged_id": row["staged_id"],
                    "target_ref": row["target_ref"],
                    "before": _resolution_before(row) if is_resolution else None,
                    "after": _inquiry_after(row),
                }
            )
            _count(action, "inquiry")
        for rows, kind, after_builder in (
            (abandoned_objects, "object", _object_after),
            (abandoned_assertions, "assertion", _assertion_after),
            (abandoned_inquiries, "inquiry", None),
        ):
            for row in rows:
                after = after_builder(row) if after_builder else _inquiry_after(row)
                entries.append(
                    {
                        "kind": kind,
                        "action": "drop",
                        "staged_id": row["staged_id"],
                        "target_ref": row["target_ref"],
                        "before": after,
                        "after": None,
                    }
                )
                _count("drop", kind)

        entries.sort(key=lambda entry: entry.get("staged_id", ""))
        total = len(entries)
        page = entries[page_offset : page_offset + page_size]
        return {
            "wake_id": wake_id,
            "published": False,
            "note": "Changes are staged for this wake only; formal commit happens at finalize_graph.",
            "summary": {"total": total, "by_action": by_action, "by_kind": by_kind},
            "entries": page,
            "offset": page_offset,
            "limit": page_size,
            "has_more": page_offset + page_size < total,
        }


__all__ = ["inspect_working_graph", "staged_item_history", "diff_working_graph"]
