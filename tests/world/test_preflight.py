"""Slice 4 acceptance: graph_inspect / graph_diff shared readiness rules (plan §6.9/§7.6/§7.7).

The working-graph readiness surface: inspect re-derives blockers from the
durable staging tables on every call, so candidates and alias occupancy that
grew *after* a patch was accepted (another wake finalizes a colliding object)
surface here even though patch-time checks passed. Blockers mirror the
patch-time gates plus the mechanical problems the Store commit path would
report; warnings are quality candidates only and never block.
"""

from __future__ import annotations

from leave_information_bubble.world import ObjectKind, WorldStore
from leave_information_bubble.world.contracts import (
    AssertionInput,
    CognitiveDelta,
    InquiryInput,
    InquiryResolution,
    ObjectInput,
)
from leave_information_bubble.world.finalize import finalize_graph
from leave_information_bubble.world.preflight import (
    diff_working_graph,
    inspect_working_graph,
    staged_item_history,
)
from leave_information_bubble.world.staging import OK, REJECTED, apply_patch

WAKE = "wake-inspect"
TIME = "2026-08-22T12:00:00+00:00"
OTHER_WAKE = "wake-inspect-2"

FORMAL_OBJECTS = [
    ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One", aliases=["T1"]),
    ObjectInput(id="s1", kind=ObjectKind.EVENT, canonical_name="Summer Cup"),
]


def _seed(store: WorldStore, commit_id: str = "seed") -> None:
    store.memory_commit(CognitiveDelta(objects=FORMAL_OBJECTS), commit_id)


def _create_object(op_id: str, canonical: str, **overrides: object) -> dict[str, object]:
    return {
        "op_id": op_id,
        "kind": "object",
        "action": "create",
        "payload": {"canonical_name": canonical, "kind": "entity", **overrides},
    }


def _create_assertion(op_id: str, subject: str, **overrides: object) -> dict[str, object]:
    payload = {
        "subject_ref": subject,
        "predicate": "related_to",
        "object_ref": "s1",
        "epistemic_role": "fact",
        **overrides,
    }
    if payload.get("object_ref") is None and "literal" in payload:
        payload.pop("object_ref")
    return {
        "op_id": op_id,
        "kind": "assertion",
        "action": "supersede" if "supersedes_ref" in payload else "create",
        "payload": payload,
    }


def _blockers_for(report: dict[str, object], code: str) -> list[dict[str, object]]:
    return [entry for entry in report["blockers"] if entry["code"] == code]


# ── zero-connection objects and unanchored events ────────────────────────────


def test_inspect_zero_connection_object_blocks_and_fix_unblocks(tmp_path) -> None:
    """A new non-provisional object with no connection blocks; adding one unblocks."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    created = apply_patch(store, WAKE, [_create_object("op-1", "Loner")])[0]
    assert created["status"] == OK

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    assert _blockers_for(report, "zero_connection_object") != []

    # connect it: the same snapshot now passes inspect on the same rule
    linked = apply_patch(store, WAKE, [_create_assertion("op-2", created["staged_id"])])[0]
    assert linked["status"] == OK
    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "ready"
    assert _blockers_for(report, "zero_connection_object") == []
    assert report["active_total"] == 2


def test_inspect_blocks_overlay_when_formal_base_changed_after_patch(tmp_path) -> None:
    """Patch-time formal bases fail closed and require a reread before revision."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    patched = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-1",
                "kind": "object",
                "action": "update",
                "target_ref": "t1",
                "payload": {"canonical_name": "Team One Draft", "kind": "entity"},
            }
        ],
    )[0]
    assert patched["status"] == OK

    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="t1",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Team One Concurrent",
                    aliases=["T1"],
                    expected_version=1,
                )
            ]
        ),
        "concurrent-update",
    )

    report = inspect_working_graph(store, WAKE)
    blocker = _blockers_for(report, "stale_base")
    assert report["readiness"] == "blocked"
    assert blocker == [
        {
            "code": "stale_base",
            "ref": patched["staged_id"],
            "message": "formal object t1 changed from version 1 to 2 after this overlay was patched",
            "base_version": 1,
            "current_version": 2,
            "action_hint": "memory_read t1, then drop and re-patch this overlay",
        }
    ]


def test_inspect_blocks_legacy_overlay_with_unknown_formal_base(tmp_path) -> None:
    """An upgraded v14 overlay cannot invent a historical patch-time version."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    patched = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-1",
                "kind": "object",
                "action": "update",
                "target_ref": "t1",
                "payload": {"canonical_name": "Legacy Draft", "kind": "entity"},
            }
        ],
    )[0]
    assert patched["status"] == OK
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE staged_objects SET base_version = NULL WHERE staged_id = ?",
            (patched["staged_id"],),
        )

    report = inspect_working_graph(store, WAKE)
    assert _blockers_for(report, "base_version_unknown") == [
        {
            "code": "base_version_unknown",
            "ref": patched["staged_id"],
            "message": "formal object t1 overlay has no trustworthy patch-time base version",
            "action_hint": ("memory_read t1, then drop and re-patch this legacy overlay"),
        }
    ]


def test_inspect_provisional_object_never_blocked(tmp_path) -> None:
    """An explicitly provisional creation skips the zero-connection blocker."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    created = apply_patch(store, WAKE, [_create_object("op-1", "Maybe", provisional=True)])[0]
    assert created["status"] == OK
    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "ready"
    assert _blockers_for(report, "zero_connection_object") == []


def test_inspect_event_anchor_does_not_miskill_anchored_events(tmp_path) -> None:
    """An event anchored by a relation passes; a bare event blocks (plan §6.9)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    # anchored event: appears as object of a relation
    anchored = apply_patch(
        store,
        WAKE,
        [_create_object("op-1", "Match Day", kind="event", event_time_start=TIME)],
    )[0]
    assert anchored["status"] == OK
    apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "assertion",
                "action": "create",
                "payload": {
                    "subject_ref": "t1",
                    "predicate": "participated_in",
                    "object_ref": anchored["staged_id"],
                    "epistemic_role": "fact",
                },
            }
        ],
    )
    # bare event: no relation anchor (but carries its known time, so the
    # time rule must not confound the anchor rule)
    bare = apply_patch(
        store,
        WAKE,
        [_create_object("op-3", "Solo Event", kind="event", event_time_start=TIME)],
    )[0]
    assert bare["status"] == OK

    report = inspect_working_graph(store, WAKE)
    event_blockers = _blockers_for(report, "event_identity_anchor")
    assert len(event_blockers) == 1
    assert event_blockers[0]["ref"] == bare["staged_id"]
    # the anchored event is never misflagged by the generic rule
    assert all(entry["ref"] != anchored["staged_id"] for entry in event_blockers)
    assert _blockers_for(report, "event_time_missing") == []


def test_inspect_event_requires_event_time(tmp_path) -> None:
    """Events are time anchors: a new non-provisional event without
    event_time_start blocks, and carrying the time clears the rule."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    untimed = apply_patch(store, WAKE, [_create_object("op-1", "Undated Scrim", kind="event")])[0]
    assert untimed["status"] == OK
    timed = apply_patch(
        store,
        WAKE,
        [_create_object("op-2", "Dated Match", kind="event", event_time_start=TIME)],
    )[0]
    assert timed["status"] == OK
    assert all(
        result["status"] == OK
        for result in apply_patch(
            store,
            WAKE,
            [
                {
                    "op_id": "op-3",
                    "kind": "assertion",
                    "action": "create",
                    "payload": {
                        "subject_ref": "t1",
                        "predicate": "participated_in",
                        "object_ref": untimed["staged_id"],
                        "epistemic_role": "fact",
                    },
                },
                {
                    "op_id": "op-4",
                    "kind": "assertion",
                    "action": "create",
                    "payload": {
                        "subject_ref": "t1",
                        "predicate": "participated_in",
                        "object_ref": timed["staged_id"],
                        "epistemic_role": "fact",
                    },
                },
            ],
        )
    )

    report = inspect_working_graph(store, WAKE)
    missing = _blockers_for(report, "event_time_missing")
    assert [entry["ref"] for entry in missing] == [untimed["staged_id"]]
    # the timed event passes both event rules when anchored
    assert _blockers_for(report, "event_identity_anchor") == []


# ── duplicates, identity blockers, and hard occupancy ────────────────────────


def test_inspect_duplicate_inquiry_is_warning_not_blocker(tmp_path) -> None:
    """Duplicate inquiries warn but never block (quality candidates, §6.9)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    def _inquiry(op_id: str) -> dict[str, object]:
        return {
            "op_id": op_id,
            "kind": "inquiry",
            "action": "create",
            "payload": {
                "subject_ref": "t1",
                "prompt": "repeat me",
                "rationale": "checking the duplicate warning",
                "kind": "factual",
            },
        }

    first = apply_patch(store, WAKE, [_inquiry("op-1")])[0]
    second = apply_patch(store, WAKE, [_inquiry("op-2")])[0]
    assert first["status"] == second["status"] == OK

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "ready"
    warning_codes = {entry["code"] for entry in report["warnings"]}
    assert "inquiry_duplicate" in warning_codes


def test_inspect_identity_blocker_surfaces_grown_candidate_and_decision_persists(tmp_path) -> None:
    """A candidate that appears after patch acceptance blocks until the decision persists."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    # patch-time: no collision, object accepted and connected
    created = apply_patch(store, WAKE, [_create_object("op-1", "Brand New")])[0]
    assert created["status"] == OK
    connected = apply_patch(store, WAKE, [_create_assertion("op-1b", created["staged_id"])])[0]
    assert connected["status"] == OK
    assert inspect_working_graph(store, WAKE)["readiness"] == "ready"

    # another wake finalizes the same referent name: the candidate now exists
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="t9", kind=ObjectKind.ENTITY, canonical_name="Brand New")]),
        "other-wake",
    )
    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    identity_blockers = _blockers_for(report, "identity_unresolved")
    assert len(identity_blockers) == 1
    candidate_ids = {entry["id"] for entry in identity_blockers[0]["candidates"]}
    assert "t9" in candidate_ids

    # the agent confirms distinct: the decision must persist and clear inspect
    resolved = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "object",
                "action": "update",
                "target_ref": created["staged_id"],
                "payload": {"canonical_name": "Brand New", "kind": "entity"},
                "decision": {
                    "action": "confirm_distinct",
                    "distinct_from": "t9",
                    "basis": "t9 is the sponsor; this is the player with the same surname",
                },
            }
        ],
    )[0]
    assert resolved["status"] == OK
    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "ready"
    assert _blockers_for(report, "identity_unresolved") == []


def test_inspect_identity_candidate_details_are_bounded_with_omission_count(tmp_path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(
        store,
        WAKE,
        [_create_object("op-crowded", "Crowded Name", provisional=True)],
    )[0]
    assert created["status"] == OK

    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id=f"crowded-{index:02d}",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Crowded Name",
                )
                for index in range(25)
            ]
        ),
        "crowded-formal",
    )

    report = inspect_working_graph(store, WAKE)
    blocker = _blockers_for(report, "identity_unresolved")[0]
    assert len(blocker["candidates"]) == 20
    assert blocker["omitted_counts"] == {"candidates": 5}
    assert "+5 more" in blocker["message"]


def test_inspect_staged_alias_duplicate_predicts_commit_rejection(tmp_path) -> None:
    """Two staged objects claiming one alias can never commit; inspect must say so (G5a F-2)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    alpha = apply_patch(
        store,
        WAKE,
        [_create_object("op-1", "Alpha", aliases=["Shared"], provisional=True)],
    )[0]
    beta = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "object",
                "action": "create",
                "payload": {
                    "canonical_name": "Beta",
                    "kind": "entity",
                    "aliases": ["Shared"],
                    "provisional": True,
                },
                "decision": {
                    "action": "confirm_distinct",
                    "distinct_from": alpha["staged_id"],
                    "basis": "two distinct referents with the same alias",
                },
            }
        ],
    )[0]
    assert alpha["status"] == beta["status"] == OK
    apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-3",
                "kind": "object",
                "action": "update",
                "target_ref": alpha["staged_id"],
                "payload": {"canonical_name": "Alpha", "kind": "entity", "aliases": ["Shared"]},
                "decision": {
                    "action": "confirm_distinct",
                    "distinct_from": beta["staged_id"],
                    "basis": "two distinct referents with the same alias",
                },
            }
        ],
    )
    # both sides confirmed distinct: the per-row identity checks pass, but the
    # commit gate can never accept the duplicate alias — inspect predicts it
    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    duplicates = _blockers_for(report, "staged_alias_duplicate")
    assert len(duplicates) == 1
    assert duplicates[0]["alias"] == "shared"
    assert set(duplicates[0]["owners"]) == {alpha["staged_id"], beta["staged_id"]}


def test_inspect_finalized_wake_with_new_staging_reports_terminal_blocker(tmp_path) -> None:
    """A finalized wake that gained new active items is never reported ready (G5a F-3)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    created = apply_patch(store, WAKE, [_create_object("op-1", "Settled")])[0]
    apply_patch(store, WAKE, [_create_assertion("op-2", created["staged_id"])])
    assert finalize_graph(store, WAKE).status == "published"

    # the agent stages new work in the already-finalized wake
    late = apply_patch(store, WAKE, [_create_object("op-3", "Latecomer")])[0]
    apply_patch(store, WAKE, [_create_assertion("op-4", late["staged_id"])])

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    terminal = _blockers_for(report, "wake_closed")
    assert len(terminal) == 1
    assert terminal[0]["commit_id"] == f"{WAKE}:finalize"
    assert "2 active item(s)" in terminal[0]["message"]

    # a finalized wake with no new staging reports ready (nothing to reject)
    assert finalize_graph(store, OTHER_WAKE).status == "published"
    assert inspect_working_graph(store, OTHER_WAKE)["readiness"] == "ready"


def test_inspect_alias_occupied_is_hard_block(tmp_path) -> None:
    """Exact active alias occupancy after patch acceptance blocks with no override."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    created = apply_patch(store, WAKE, [_create_object("op-1", "Team Two", aliases=["T2"])])[0]
    assert created["status"] == OK
    connected = apply_patch(store, WAKE, [_create_assertion("op-1b", created["staged_id"])])[0]
    assert connected["status"] == OK
    assert inspect_working_graph(store, WAKE)["readiness"] == "ready"

    # another wake's commit claims the alias the staged object uses
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="t8", kind=ObjectKind.ENTITY, canonical_name="Another Team", aliases=["T2"])
            ]
        ),
        "other-wake",
    )
    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    occupied = _blockers_for(report, "alias_occupied")
    assert len(occupied) == 1
    assert occupied[0]["ref"] == created["staged_id"]
    # the occupied alias is reported in the shared normalized form
    assert occupied[0]["occupied"][0]["alias"] == "t2"
    assert occupied[0]["occupied"][0]["id"] == "t8"


def test_inspect_exact_duplicate_is_patch_gate_and_dangling_evidence_is_blocker(tmp_path) -> None:
    """Equivalent assertions are rejected at patch time; evidence refs are verified."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    first = apply_patch(store, WAKE, [_create_assertion("op-1", "t1")])[0]
    assert first["status"] == OK
    # the exact-duplicate gate is the canonical first line (create-time reject)
    dup = apply_patch(store, WAKE, [_create_assertion("op-2", "t1")])[0]
    assert dup["status"] == REJECTED
    assert dup["error_code"] == "exact_duplicate"
    assert inspect_working_graph(store, WAKE)["readiness"] == "ready"

    # a stale evidence ref forced into the staging row is a blocker (defense)
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE staged_assertions SET evidence_json = ? WHERE wake_id = ? AND status = 'active'",
            ('["obs-missing"]', WAKE),
        )
    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    assert _blockers_for(report, "dangling_evidence_ref")[0]["ref"] == "obs-missing"


def test_inspect_twin_correction_pair_is_ready(tmp_path) -> None:
    """A metadata-only correction is not an exact duplicate of its own target (G5a F-6).

    The correction (supersedes_ref) is signature-identical to the row it
    supersedes: the equivalent-signature gate must not block it (Core §6.7),
    and the preflight duplicate check must see through the correction chain.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    first = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-1", "t1", predicate="ranked", object_ref=None, literal="twin")],
    )[0]
    assert first["status"] == OK
    correction = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-2",
                "t1",
                predicate="ranked",
                object_ref=None,
                literal="twin",
                confidence=0.9,
                supersedes_ref=f"{WAKE}:a1",
            )
        ],
    )[0]
    assert correction["status"] == OK

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "ready"
    assert not _blockers_for(report, "exact_duplicate")


# ── diff: supersede before/after, drops, paging ──────────────────────────────


def _seed_assertion(store: WorldStore) -> None:
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            assertions=[
                AssertionInput(
                    id="a-1",
                    subject_id="t1",
                    predicate="related_to",
                    object_id="s1",
                    epistemic_role="fact",
                    confidence=0.9,
                )
            ],
        ),
        "seed-assertion",
    )


def test_diff_supersede_shows_before_after_and_drop_shows_before_only(tmp_path) -> None:
    """graph_diff renders formal before / staged after per item, never published."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed_assertion(store)

    supersede = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-1",
                "kind": "assertion",
                "action": "supersede",
                "payload": {
                    "subject_ref": "t1",
                    "predicate": "related_to",
                    "object_ref": "s1",
                    "epistemic_role": "fact",
                    "confidence": 0.95,
                    "supersedes_ref": "a-1",
                },
            }
        ],
    )[0]
    assert supersede["status"] == OK
    created = apply_patch(store, WAKE, [_create_object("op-2", "Fresh")])[0]
    assert created["status"] == OK

    diff = diff_working_graph(store, WAKE)
    assert diff["published"] is False
    supersede_entry = next(entry for entry in diff["entries"] if entry["action"] == "supersede")
    assert supersede_entry["target_ref"] == "a-1"
    assert supersede_entry["before"]["confidence"] == 0.9
    assert supersede_entry["after"]["confidence"] == 0.95
    assert supersede_entry["before"]["id"] == "a-1"
    assert diff["summary"]["by_action"]["supersede"] == 1
    assert diff["summary"]["by_action"]["create"] == 1

    # drop renders as before-only with after null
    dropped = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-3", "kind": "object", "action": "drop", "target_ref": created["staged_id"]}],
    )[0]
    assert dropped["status"] == OK
    diff = diff_working_graph(store, WAKE)
    drop_entry = next(entry for entry in diff["entries"] if entry["action"] == "drop")
    assert drop_entry["after"] is None
    assert drop_entry["before"]["canonical_name"] == "Fresh"


def test_diff_pages_with_limit_offset(tmp_path) -> None:
    """graph_diff honors bounded limit/offset paging with a has_more signal."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    for i in range(5):
        result = apply_patch(store, WAKE, [_create_object(f"op-{i}", f"Item {i}")])[0]
        assert result["status"] == OK

    first = diff_working_graph(store, WAKE, limit=2)
    assert len(first["entries"]) == 2
    assert first["has_more"] is True
    assert first["summary"]["total"] == 5
    second = diff_working_graph(store, WAKE, limit=2, offset=2)
    assert len(second["entries"]) == 2
    assert second["has_more"] is True
    last = diff_working_graph(store, WAKE, limit=10, offset=4)
    assert len(last["entries"]) == 1
    assert last["has_more"] is False
    # pages are disjoint
    page_ids = {entry["staged_id"] for page in (first, second, last) for entry in page["entries"]}
    assert len(page_ids) == 5


# ── item history and the patch-ok invariance ─────────────────────────────────


def test_staged_item_history_reconstructs_ops(tmp_path) -> None:
    """item history shows the current row plus every patch op that touched it."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "Historic")])[0]
    staged_id = created["staged_id"]
    updated = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "object",
                "action": "update",
                "target_ref": staged_id,
                "payload": {"canonical_name": "Historic Revised", "kind": "entity"},
            }
        ],
    )[0]
    assert updated["status"] == OK

    history = inspect_working_graph(store, WAKE)
    assert history["items_truncated"] is False
    item = next(entry for entry in history["items"] if entry["staged_id"] == staged_id)
    assert item["version"] == 2

    detail = staged_item_history(store, WAKE, staged_id)
    assert detail is not None
    assert detail["item"]["version"] == 2
    actions = [op["action"] for op in detail["ops"]]
    assert actions == ["created", "updated"]


def test_staged_item_history_keeps_latest_fifty_with_explicit_cut(tmp_path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "Long History")])[0]
    staged_id = created["staged_id"]
    for index in range(2, 57):
        result = apply_patch(
            store,
            WAKE,
            [
                {
                    "op_id": f"op-{index}",
                    "kind": "object",
                    "action": "update",
                    "target_ref": staged_id,
                    "payload": {
                        "canonical_name": f"Long History {index}",
                        "kind": "entity",
                    },
                }
            ],
        )[0]
        assert result["status"] == OK

    detail = staged_item_history(store, WAKE, staged_id)
    assert detail is not None
    assert detail["ops_total"] == 56
    assert detail["ops_truncated"] is True
    assert len(detail["ops"]) == 50
    assert detail["ops"][0]["op_id"] == "op-7"
    assert detail["ops"][-1]["op_id"] == "op-56"


def test_patch_ok_items_never_rejected_by_inspect(tmp_path) -> None:
    """The invariance: a snapshot inspect accepts is exactly the patch result."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    objects = apply_patch(
        store,
        WAKE,
        [
            _create_object("op-1", "Star Player"),
            _create_object("op-2", "Match Day", kind="event", event_time_start=TIME),
        ],
    )
    assert all(result["status"] == OK for result in objects)
    star_id, match_id = (result["staged_id"] for result in objects)

    relations = apply_patch(
        store,
        WAKE,
        [
            _create_assertion("op-3", star_id),  # connects Star Player
            {
                "op_id": "op-4",
                "kind": "assertion",
                "action": "create",
                "payload": {
                    "subject_ref": "t1",
                    "predicate": "participated_in",
                    "object_ref": match_id,
                    "epistemic_role": "fact",
                },
            },
        ],
    )
    assert all(result["status"] == OK for result in relations)

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "ready"
    assert report["blockers"] == []
    assert report["stats"]["objects"]["active"] == 2
    assert report["stats"]["assertions"]["active"] == 2


# ── Wave B: staged resolutions mirror the finalize resolve gates (v18) ───────


def _create_inquiry(op_id: str, subject: str, **overrides: object) -> dict[str, object]:
    return {
        "op_id": op_id,
        "kind": "inquiry",
        "action": "create",
        "payload": {
            "subject_ref": subject,
            "prompt": "who is the champion?",
            "rationale": "seed rationale",
            "kind": "factual",
            **overrides,
        },
    }


def _resolve_op(op_id: str, target_ref: str, **overrides: object) -> dict[str, object]:
    return {
        "op_id": op_id,
        "kind": "inquiry",
        "action": "resolve",
        "target_ref": target_ref,
        "payload": {"expected_version": 1, **overrides},
    }


def test_inspect_resolution_not_misflagged_and_items_action_resolved(tmp_path) -> None:
    """A resolve row mirrors its target subject/prompt; inspect must not
    duplicate-warn it against its own target, and the items summary reports
    action 'resolved' for the resolution row (kind 'resolution')."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    inq = apply_patch(store, WAKE, [_create_inquiry("op-1", "t1")])[0]
    assert inq["status"] == OK
    answer = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-2", "t1", predicate="explains", answers_ref=inq["staged_id"])],
    )[0]
    assert answer["status"] == OK
    resolved = apply_patch(
        store, WAKE, [_resolve_op("op-3", inq["staged_id"], answers_ref=answer["staged_id"])]
    )[0]
    assert resolved["status"] == OK

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "ready"
    assert all(entry["code"] != "inquiry_duplicate" for entry in report["warnings"])
    by_id = {item["staged_id"]: item for item in report["items"] if item["kind"] == "inquiry"}
    assert by_id[inq["staged_id"]]["action"] == "created"
    assert by_id[resolved["staged_id"]]["action"] == "resolved"
    assert by_id[resolved["staged_id"]]["target_ref"] == inq["staged_id"]


def test_inquiry_relations_are_visible_in_inspect_history_and_diff(tmp_path) -> None:
    """Every staged inquiry relationship remains visible before publish.

    The graph shell must expose the same relation references that finalize will
    compile: an inquiry's ``deepens_ref``, an assertion's ``answers_ref``, and
    a resolution's answering assertion.  Dropping an item keeps those links in
    the diff's ``before`` image so review never loses the semantic reason for
    the change.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-formal",
                    subject_id="t1",
                    prompt="What happened?",
                    rationale="formal gap",
                )
            ],
        ),
        "seed-inquiry",
    )

    deepener = apply_patch(
        store,
        WAKE,
        [_create_inquiry("op-1", "t1", prompt="Why?", deepens_ref="inq-formal")],
    )[0]
    answer = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-2",
                "t1",
                predicate="explains",
                answers_ref=deepener["staged_id"],
            )
        ],
    )[0]
    resolution = apply_patch(
        store,
        WAKE,
        [
            _resolve_op(
                "op-3",
                deepener["staged_id"],
                answers_ref=answer["staged_id"],
            )
        ],
    )[0]
    assert {deepener["status"], answer["status"], resolution["status"]} == {OK}

    report = inspect_working_graph(store, WAKE)
    by_id = {item["staged_id"]: item for item in report["items"]}
    assert by_id[deepener["staged_id"]]["deepens_ref"] == "inq-formal"
    assert by_id[answer["staged_id"]]["answers_ref"] == deepener["staged_id"]
    assert by_id[resolution["staged_id"]]["answers_ref"] == answer["staged_id"]

    assert staged_item_history(store, WAKE, deepener["staged_id"])["item"]["deepens_ref"] == "inq-formal"
    assert (
        staged_item_history(store, WAKE, answer["staged_id"])["item"]["answers_ref"] == deepener["staged_id"]
    )
    assert (
        staged_item_history(store, WAKE, resolution["staged_id"])["item"]["answers_ref"]
        == answer["staged_id"]
    )

    diff = diff_working_graph(store, WAKE)
    by_id = {entry["staged_id"]: entry for entry in diff["entries"]}
    assert by_id[deepener["staged_id"]]["after"]["deepens_ref"] == "inq-formal"
    assert by_id[answer["staged_id"]]["after"]["answers_ref"] == deepener["staged_id"]
    assert by_id[resolution["staged_id"]]["after"]["answers_ref"] == answer["staged_id"]
    assert by_id[resolution["staged_id"]]["before"]["status"] == "open"
    assert by_id[resolution["staged_id"]]["after"]["status"] == "resolved"

    dropped = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-4",
                "kind": "assertion",
                "action": "drop",
                "target_ref": answer["staged_id"],
            }
        ],
    )[0]
    assert dropped["status"] == OK
    dropped_diff = diff_working_graph(store, WAKE)
    dropped_answer = next(
        entry for entry in dropped_diff["entries"] if entry["staged_id"] == answer["staged_id"]
    )
    assert dropped_answer["action"] == "drop"
    assert dropped_answer["before"]["answers_ref"] == deepener["staged_id"]
    assert dropped_answer["after"] is None


def test_inspect_resolution_uncertainty_answer_blocks(tmp_path) -> None:
    """B4-5 mirror: an UNCERTAINTY answering assertion passes staging (the
    patch gate checks only the declared target) but finalize requires a
    non-uncertainty answering assertion in the same delta — inspect blocks
    the same snapshot so a compile failure never surprises."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    inq = apply_patch(store, WAKE, [_create_inquiry("op-1", "t1")])[0]
    assert inq["status"] == OK
    answer = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-2", "t1", predicate="explains", answers_ref=inq["staged_id"], epistemic_role="uncertainty"
            )
        ],
    )[0]
    assert answer["status"] == OK
    resolved = apply_patch(
        store, WAKE, [_resolve_op("op-3", inq["staged_id"], answers_ref=answer["staged_id"])]
    )[0]
    assert resolved["status"] == OK

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    blocker = _blockers_for(report, "no_answering_assertion")
    assert blocker and blocker[0]["ref"] == resolved["staged_id"]


def test_inspect_resolution_after_answer_dropped_blocks(tmp_path) -> None:
    """Drift mirror: the answering assertion is dropped after the resolve was
    accepted; finalize would fail the delta-answer gate, so inspect blocks
    with no_answering_assertion."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    inq = apply_patch(store, WAKE, [_create_inquiry("op-1", "t1")])[0]
    assert inq["status"] == OK
    answer = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-2", "t1", predicate="explains", answers_ref=inq["staged_id"])],
    )[0]
    assert answer["status"] == OK
    resolved = apply_patch(
        store, WAKE, [_resolve_op("op-3", inq["staged_id"], answers_ref=answer["staged_id"])]
    )[0]
    assert resolved["status"] == OK
    dropped = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-4", "kind": "assertion", "action": "drop", "target_ref": answer["staged_id"]}],
    )[0]
    assert dropped["status"] == OK

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    assert _blockers_for(report, "no_answering_assertion") != []


def test_inspect_resolution_formal_target_requires_staged_answer(tmp_path) -> None:
    """B4-5 mirror: a resolve may target a formal inquiry and name its formal
    answering assertion — staging accepts it (every reference resolves) but
    finalize requires a co-published staged answer; inspect blocks first."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[InquiryInput(id="inq-f", subject_id="t1", prompt="Why?", rationale="gap")],
            assertions=[
                AssertionInput(
                    id="fa-1",
                    subject_id="t1",
                    predicate="explains",
                    object_id="s1",
                    epistemic_role="fact",
                    confidence=0.9,
                    answers_inquiry_id="inq-f",
                )
            ],
        ),
        "seed-answer",
    )
    resolved = apply_patch(store, WAKE, [_resolve_op("op-1", "inq-f", answers_ref="fa-1")])[0]
    assert resolved["status"] == OK

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    blocker = _blockers_for(report, "no_answering_assertion")
    assert blocker and blocker[0]["ref"] == resolved["staged_id"]


def test_inspect_resolution_already_resolved_blocks(tmp_path) -> None:
    """Formal-gate mirror: a formal resolve commit after staging blocks the
    staged resolution (inquiry_already_resolved)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[InquiryInput(id="inq-f", subject_id="t1", prompt="Why?", rationale="gap")],
        ),
        "seed-inquiry",
    )
    answer = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-1", "t1", predicate="expresses", answers_ref="inq-f")],
    )[0]
    assert answer["status"] == OK
    resolved = apply_patch(store, WAKE, [_resolve_op("op-2", "inq-f", answers_ref=answer["staged_id"])])[0]
    assert resolved["status"] == OK

    store.memory_commit(
        CognitiveDelta(
            assertions=[
                AssertionInput(
                    id="fa-now",
                    subject_id="t1",
                    predicate="explains",
                    object_id="s1",
                    epistemic_role="fact",
                    confidence=0.9,
                    answers_inquiry_id="inq-f",
                )
            ],
            resolve_inquiries=[InquiryResolution(id="inq-f", expected_version=1)],
        ),
        "resolve-now",
    )

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    blocker = _blockers_for(report, "inquiry_already_resolved")
    assert blocker and blocker[0]["ref"] == resolved["staged_id"]


def test_inspect_resolution_stale_expected_version_blocks(tmp_path) -> None:
    """Formal-gate mirror: a formal version bump after staging freezes
    expected_version at the version the resolve saw; inspect blocks with
    stale_expected_version once the formal row moves."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[InquiryInput(id="inq-move", subject_id="t1", prompt="Move?", rationale="gap")],
        ),
        "seed-move",
    )
    answer = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-1", "t1", predicate="expresses", answers_ref="inq-move")],
    )[0]
    assert answer["status"] == OK
    resolved = apply_patch(store, WAKE, [_resolve_op("op-2", "inq-move", answers_ref=answer["staged_id"])])[0]
    assert resolved["status"] == OK

    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-move",
                    subject_id="t1",
                    prompt="Move?",
                    rationale="gap",
                    expected_version=1,
                )
            ],
        ),
        "bump-move",
    )

    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "blocked"
    blocker = _blockers_for(report, "stale_expected_version")
    assert blocker and blocker[0]["ref"] == resolved["staged_id"]
