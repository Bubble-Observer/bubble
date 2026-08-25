"""Slice 3: durable working graph + graph_patch protocol (plan §6.2-§6.5, D-006/D-012)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio

from leave_information_bubble.world import ObjectKind, WorldStore
from leave_information_bubble.world.contracts import (
    AssertionInput,
    CognitiveDelta,
    InquiryInput,
    ObjectInput,
)
from leave_information_bubble.world.finalize import finalize_graph, inspect_working_graph
from leave_information_bubble.world.staging import (
    CONFLICT,
    NEEDS_IDENTITY_RESOLUTION,
    OK,
    REJECTED,
    apply_patch,
    read_active_staged,
)
from leave_information_bubble.world.tools import WorldTools
from tests.world._tools_helpers import _tools

WAKE = "wake-a"
OTHER = "wake-b"

FORMAL_OBJECTS = [
    ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One"),
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
        "confidence": 0.9,
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


# ── object lifecycle ──────────────────────────────────────────────────────


def test_staging_create_object_gets_host_issued_staged_id(tmp_path) -> None:
    """A staged create gets a stable staged id the next patch can reference."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    results = apply_patch(store, WAKE, [_create_object("op-1", "New Club", aliases=["NC"])])
    assert results[0]["status"] == OK
    staged_id = results[0]["staged_id"]
    assert staged_id.startswith(WAKE)

    # the working graph shows it for this wake
    working = read_active_staged(store, WAKE)
    assert [obj["staged_id"] for obj in working["objects"]] == [staged_id]
    assert working["objects"][0]["canonical_name"] == "New Club"
    assert working["objects"][0]["aliases"] == ["NC"]
    assert working["objects"][0]["status"] == "active"

    # a next patch can reference the staged id (create → reference round-trip)
    results = apply_patch(store, WAKE, [_create_assertion("op-2", staged_id)])
    assert results[0]["status"] == OK
    working = read_active_staged(store, WAKE)
    assert working["assertions"][0]["subject_ref"] == staged_id


def test_object_create_without_kind_is_rejected_not_defaulted(tmp_path) -> None:
    """Kind is explicit on every object write (entity is the fallback, never
    a silent default): a missing kind is rejected at the contract layer
    instead of silently staging an entity (six-kind contract, 2026-08-23)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    item = _create_object("op-1", "Mystery Thing")
    assert "kind" in item["payload"]
    del item["payload"]["kind"]
    result = apply_patch(store, WAKE, [item])[0]
    assert result["status"] == REJECTED
    assert result["error_code"] == "invalid_tool_arguments"
    assert result["field"] == "items[0].payload.kind"
    working = read_active_staged(store, WAKE)
    assert working["objects"] == []


def test_object_update_cannot_overwrite_kind_with_missing(tmp_path) -> None:
    """An update omitting kind is rejected too; a known kind is never
    overwritten by an empty payload (six-kind contract, 2026-08-23)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "New Club")])[0]
    assert created["status"] == OK
    item = {
        "op_id": "op-2",
        "kind": "object",
        "action": "update",
        "target_ref": created["staged_id"],
        "expected_version": 1,
        "payload": {"canonical_name": "Renamed Club"},
    }
    result = apply_patch(store, WAKE, [item])[0]
    assert result["status"] == REJECTED
    assert result["error_code"] == "invalid_tool_arguments"
    working = read_active_staged(store, WAKE)
    assert working["objects"][0]["canonical_name"] == "New Club"
    assert working["objects"][0]["kind"] == "entity"


def test_replayed_op_id_with_same_payload_replays_original_result(tmp_path) -> None:
    """Idempotency: wake + op_id + identical payload returns the original verdict."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    item = _create_object("op-1", "New Club")
    first = apply_patch(store, WAKE, [item])[0]
    second = apply_patch(store, WAKE, [item])[0]
    assert second["status"] == OK
    assert second == first
    assert second["staged_id"] == first["staged_id"]
    working = read_active_staged(store, WAKE)
    assert len(working["objects"]) == 1  # replayed op does not duplicate


def test_replayed_op_id_with_different_payload_conflicts(tmp_path) -> None:
    """Idempotency: the same op_id with a different payload is an explicit conflict."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    first = apply_patch(store, WAKE, [_create_object("op-1", "New Club")])[0]
    assert first["status"] == OK
    conflicting = apply_patch(store, WAKE, [_create_object("op-1", "Totally Different Name")])[0]
    assert conflicting["status"] == CONFLICT
    assert conflicting["error_code"] == "op_id_reused"
    # the original staged row is untouched
    working = read_active_staged(store, WAKE)
    assert working["objects"][0]["canonical_name"] == "New Club"


async def test_graph_shell_guard_replays_known_op_but_rejects_new_patch_after_finalize(
    tmp_path,
) -> None:
    """A durable finalize receipt is an atomic write barrier for the wake."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="thread-a",
        wake_id=WAKE,
        closed_wake_guard=True,
    )
    original = _create_object("op-before-close", "Published draft", provisional=True)
    first = (await tools.execute("graph_patch", {"items": [original]}, "first"))["payload"]["results"][0]
    assert first["status"] == OK
    assert finalize_graph(store, WAKE).status == "published"

    replay = (await tools.execute("graph_patch", {"items": [original]}, "replay"))["payload"]["results"][0]
    late = (
        await tools.execute(
            "graph_patch",
            {"items": [_create_object("op-after-close", "Stranded draft", provisional=True)]},
            "late",
        )
    )["payload"]["results"][0]

    assert replay == {**first, "replayed": True}
    assert late == {
        "op_id": "op-after-close",
        "status": REJECTED,
        "error_code": "wake_closed",
        "message": f"wake {WAKE} already has a durable finalize receipt",
    }
    assert read_active_staged(store, WAKE)["objects"] == []
    with store.read_connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?", (WAKE,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
            ).fetchone()[0]
            == 1
        )


def test_update_requires_resolvable_target_and_expected_version(tmp_path) -> None:
    """update validates the target and the optimistic lock."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "New Club")])[0]
    staged_id = created["staged_id"]

    missing = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "object",
                "action": "update",
                "target_ref": "no-such-id",
                "payload": {"canonical_name": "X", "kind": "entity"},
            }
        ],
    )[0]
    assert missing["status"] == REJECTED
    assert missing["error_code"] == "target_unavailable"

    stale = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-3",
                "kind": "object",
                "action": "update",
                "target_ref": staged_id,
                "expected_version": 9,
                "payload": {"canonical_name": "Renamed", "kind": "entity"},
            }
        ],
    )[0]
    assert stale["status"] == REJECTED
    assert stale["error_code"] == "version_conflict"
    assert stale["current_version"] == 1

    bumped = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-4",
                "kind": "object",
                "action": "update",
                "target_ref": staged_id,
                "expected_version": 1,
                "payload": {"canonical_name": "Renamed", "kind": "entity", "aliases": ["R"]},
            }
        ],
    )[0]
    assert bumped["status"] == OK
    assert bumped["staged_id"] == staged_id
    working = read_active_staged(store, WAKE)
    assert working["objects"][0]["canonical_name"] == "Renamed"
    assert working["objects"][0]["version"] == 2


def test_drop_blocks_active_references_and_is_a_status_change(tmp_path) -> None:
    """drop retires a staged item; nothing active may reference it afterwards."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "Doomed")])[0]
    staged_id = created["staged_id"]
    dropped = apply_patch(
        store, WAKE, [{"op_id": "op-2", "kind": "object", "action": "drop", "target_ref": staged_id}]
    )[0]
    assert dropped["status"] == OK
    assert dropped["action"] == "dropped"
    # no physical delete: the row still exists, retired
    working = read_active_staged(store, WAKE)
    assert working["objects"] == []

    blocked = apply_patch(store, WAKE, [_create_assertion("op-3", staged_id)])[0]
    assert blocked["status"] == REJECTED
    assert blocked["error_code"] == "dependency_unavailable"

    # dropping a formal-only target has no staged item to retire in this slice
    formal_drop = apply_patch(
        store, WAKE, [{"op_id": "op-4", "kind": "object", "action": "drop", "target_ref": "t1"}]
    )[0]
    assert formal_drop["status"] == REJECTED
    assert formal_drop["error_code"] == "target_unavailable"


# ── identity pre-check ─────────────────────────────────────────────────────


def test_identity_candidate_needs_resolution_and_confirm_distinct(tmp_path) -> None:
    """A canonical/alias collision returns candidates; confirm_distinct needs id + basis."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    colliding = apply_patch(store, WAKE, [_create_object("op-1", "Team One")])[0]
    assert colliding["status"] == NEEDS_IDENTITY_RESOLUTION
    assert colliding["error_code"] == "identity_candidate_exists"
    assert {"id": "t1"} in [{"id": c["id"]} for c in colliding["candidates"]]

    # confirm_distinct without a basis is not a valid decision
    no_basis = apply_patch(
        store,
        WAKE,
        [
            _create_object("op-2", "Team One")
            | {"decision": {"action": "confirm_distinct", "distinct_from": "t1"}}
        ],
    )[0]
    assert no_basis["status"] == REJECTED
    assert no_basis["error_code"] == "invalid_tool_arguments"
    assert no_basis["field"] == "items[0].decision.basis"

    # confirm_distinct against a candidate the pre-check returned → created
    confirmed = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-3",
                "kind": "object",
                "action": "create",
                "payload": {"canonical_name": "Team One", "kind": "entity"},
                "decision": {
                    "action": "confirm_distinct",
                    "distinct_from": "t1",
                    "basis": "a different organization that happens to share the name",
                },
            }
        ],
    )[0]
    assert confirmed["status"] == OK

    # confirm_distinct against an unrelated id is rejected (must match candidates)
    wrong_target = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-4",
                "kind": "object",
                "action": "create",
                "payload": {"canonical_name": "Team One", "kind": "entity"},
                "decision": {"action": "confirm_distinct", "distinct_from": "s1", "basis": "x"},
            }
        ],
    )[0]
    assert wrong_target["status"] == NEEDS_IDENTITY_RESOLUTION


def test_identity_candidate_feedback_is_bounded_with_omitted_count(tmp_path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id=f"same-{index}",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Shared Name",
                )
                for index in range(25)
            ]
        ),
        "many-identities",
    )

    result = apply_patch(store, WAKE, [_create_object("bounded-candidates", "Shared Name")])[0]

    assert result["status"] == NEEDS_IDENTITY_RESOLUTION
    assert len(result["candidates"]) == 20
    assert result["omitted_counts"] == {"candidates": 5}


# ── assertions & supersede ─────────────────────────────────────────────────


def test_assertion_exact_duplicate_blocked_against_formal_and_staged(tmp_path) -> None:
    """An equivalent current assertion — formal or staged — blocks a create."""
    store = WorldStore(tmp_path / "world.sqlite3")
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
        "seed",
    )
    dup_formal = apply_patch(store, WAKE, [_create_assertion("op-1", "t1")])[0]
    assert dup_formal["status"] == REJECTED
    assert dup_formal["error_code"] == "exact_duplicate"
    assert dup_formal["existing_id"] == "a-1"

    # a staged assertion is its own duplicate gate for the same wake
    store2 = WorldStore(tmp_path / "world2.sqlite3")
    _seed(store2)
    first = apply_patch(store2, WAKE, [_create_assertion("op-1", "t1")])[0]
    assert first["status"] == OK
    dup_staged = apply_patch(store2, WAKE, [_create_assertion("op-2", "t1")])[0]
    assert dup_staged["status"] == REJECTED
    assert dup_staged["error_code"] == "exact_duplicate"
    assert dup_staged["existing_id"] == first["staged_id"]


def test_overlapping_relation_requires_supersede_but_recurring_span_may_coexist(tmp_path) -> None:
    """Time refinement is a revision; a later disjoint episode is independent."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                *FORMAL_OBJECTS,
                ObjectInput(id="t2", kind=ObjectKind.ENTITY, canonical_name="Team Two"),
            ],
            assertions=[
                AssertionInput(
                    id="membership-old",
                    subject_id="t1",
                    predicate="member_of",
                    object_id="s1",
                    epistemic_role="fact",
                    confidence=0.8,
                    event_time_end=datetime(2020, 12, 31, tzinfo=UTC),
                ),
                AssertionInput(
                    id="membership-unknown",
                    subject_id="t2",
                    predicate="member_of",
                    object_id="s1",
                    epistemic_role="fact",
                    confidence=0.8,
                ),
            ],
        ),
        "seed",
    )

    overlap = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "overlap",
                "t2",
                predicate="member_of",
                event_time_start="2026-08-09T00:00:00Z",
            )
        ],
    )[0]
    assert overlap["status"] == REJECTED
    assert overlap["error_code"] == "overlapping_relation"
    assert overlap["existing_id"] == "membership-unknown"

    correction = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "correction",
                "t2",
                predicate="member_of",
                event_time_start="2026-08-09T00:00:00Z",
                supersedes_ref="membership-unknown",
            )
        ],
    )[0]
    assert correction["status"] == OK
    assert correction["supersedes"] == "membership-unknown"

    refined_correction = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "refined-correction",
                "t2",
                predicate="member_of",
                event_time_start="2026-08-10T00:00:00Z",
                supersedes_ref=correction["staged_id"],
            )
        ],
    )[0]
    assert refined_correction["status"] == OK
    assert refined_correction["supersedes"] == correction["staged_id"]

    recurring = apply_patch(
        store,
        "wake-recurring",
        [
            _create_assertion(
                "recurring",
                "t1",
                predicate="member_of",
                event_time_start="2021-01-01T00:00:00Z",
            )
        ],
    )[0]
    assert recurring["status"] == OK


def test_exact_duplicate_normalizes_predicate_time_and_qualifiers_before_staging(tmp_path) -> None:
    """Patch-time dedup mirrors the formal signature instead of deferring to finalize."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            assertions=[
                AssertionInput(
                    id="normalized",
                    subject_id="t1",
                    predicate="member_of",
                    object_id="s1",
                    epistemic_role="fact",
                    confidence=0.8,
                    event_time_start=datetime(2026, 8, 9, tzinfo=UTC),
                    qualifiers={"role": "starter"},
                )
            ],
        ),
        "seed",
    )

    duplicate = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "normalized-duplicate",
                "t1",
                predicate=" MEMBER_OF ",
                event_time_start="2026-08-09T00:00:00Z",
                qualifiers={"role": " starter "},
            )
        ],
    )[0]

    assert duplicate["status"] == REJECTED
    assert duplicate["error_code"] == "exact_duplicate"
    assert duplicate["existing_id"] == "normalized"


def test_assertion_invented_qualifier_key_rejected_at_write(tmp_path) -> None:
    """F-B: a qualifier key outside the contract fails at staging, not finalize.

    The advertised schema bounds qualifier keys (propertyNames enum); an
    invented key is rejected with an actionable hint listing the legal keys,
    so one bad key can never poison the whole wake at publish time.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    results = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-q1",
                "s1",
                qualifiers={"conflicting_source": "esports8 listed Jiejie on WBG"},
            )
        ],
    )
    assert results[0]["status"] == REJECTED
    assert results[0]["error_code"] == "invalid_tool_arguments"
    hint = str(results[0]["message"]) + " " + str(results[0].get("action_hint", ""))
    assert "role, language, community, scope, granularity" in hint
    # nothing was staged: the wake keeps no trace of the bad patch
    working = read_active_staged(store, WAKE)
    assert working["assertions"] == []


def test_assertion_legal_qualifiers_normalized_at_write(tmp_path) -> None:
    """Legal qualifier keys persist in the compile-normalized form (trimmed)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    results = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-q2", "s1", qualifiers={"scope": "  trading  "})],
    )
    assert results[0]["status"] == OK
    working = read_active_staged(store, WAKE)
    assert working["assertions"][0]["qualifiers"] == {"scope": "trading"}


def test_assertion_invented_qualifier_key_rejected_on_update(tmp_path) -> None:
    """The same key bound applies to assertion updates (F-B, update branch)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_assertion("op-q3", "s1")])[0]
    assert created["status"] == OK
    updated = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-q4",
                "kind": "assertion",
                "action": "update",
                "payload": {
                    "target_ref": created["staged_id"],
                    "qualifiers": {"granularity": "match", "invented": "x"},
                },
            }
        ],
    )[0]
    assert updated["status"] == REJECTED
    assert updated["error_code"] == "invalid_tool_arguments"


def test_assertion_supersede_requires_live_target(tmp_path) -> None:
    """supersede is create + supersedes_ref; the target must resolve to a live assertion."""
    store = WorldStore(tmp_path / "world.sqlite3")
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
        "seed",
    )
    item = {
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
    ok = apply_patch(store, WAKE, [item])[0]
    assert ok["status"] == OK
    assert ok["staged_id"].startswith(WAKE)
    working = read_active_staged(store, WAKE)
    assert working["assertions"][0]["supersedes_ref"] == "a-1"

    # a staged assertion can itself be superseded
    again_item = {
        **item,
        "op_id": "op-2",
        "payload": {**item["payload"], "supersedes_ref": ok["staged_id"]},
    }
    again = apply_patch(store, WAKE, [again_item])[0]
    assert again["status"] == OK

    # ... but the referenced assertion cannot be dropped while the superseder
    # is live (Core §6.7: a dropped staged item must not stay referenced)
    drop_before = apply_patch(
        store, WAKE, [{"op_id": "op-3", "kind": "assertion", "action": "drop", "target_ref": ok["staged_id"]}]
    )[0]
    assert drop_before["status"] == REJECTED
    assert drop_before["error_code"] == "dependent_exists"

    # once the supersede chain is dropped, the original target is no longer live
    drop_again = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-4", "kind": "assertion", "action": "drop", "target_ref": again["staged_id"]}],
    )[0]
    assert drop_again["status"] == OK
    drop = apply_patch(
        store, WAKE, [{"op_id": "op-5", "kind": "assertion", "action": "drop", "target_ref": ok["staged_id"]}]
    )[0]
    assert drop["status"] == OK
    broken = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-6",
                "kind": "assertion",
                "action": "supersede",
                "payload": {**item["payload"], "supersedes_ref": ok["staged_id"]},
            }
        ],
    )[0]
    assert broken["status"] == REJECTED
    assert broken["error_code"] == "dependency_unavailable"


# ── wake isolation & durability ────────────────────────────────────────────


def test_other_wake_sees_no_unfinalized_staging(tmp_path) -> None:
    """Another wake's reads and patch references never cross the wake boundary."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "Private")])[0]
    staged_id = created["staged_id"]

    assert read_active_staged(store, OTHER) == {"objects": [], "assertions": [], "inquiries": []}

    cross = apply_patch(store, OTHER, [_create_assertion("op-x", staged_id)])[0]
    assert cross["status"] == REJECTED
    assert cross["error_code"] == "dependency_unavailable"


def test_staging_survives_restart_and_replay(tmp_path) -> None:
    """Active staging persists across process restarts; receipts replay after reopen."""
    path = tmp_path / "world.sqlite3"
    store = WorldStore(path)
    _seed(store)
    item = _create_object("op-1", "Persistent")
    first = apply_patch(store, WAKE, [item])[0]

    reopened = WorldStore(path)
    working = read_active_staged(reopened, WAKE)
    assert working["objects"][0]["staged_id"] == first["staged_id"]
    replay = apply_patch(reopened, WAKE, [item])[0]
    assert replay["status"] == OK
    assert replay["staged_id"] == first["staged_id"]
    assert len(read_active_staged(reopened, WAKE)["objects"]) == 1


def test_batch_partial_failure_keeps_successful_items(tmp_path) -> None:
    """Core §6.3: a later item's failure never rolls back an already staged item."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    results = apply_patch(
        store,
        WAKE,
        [
            _create_object("op-1", "Kept"),
            _create_assertion("op-2", "no-such-subject"),
            _create_object("op-3", "Also Kept"),
        ],
    )
    assert results[0]["status"] == OK
    assert results[1]["status"] == REJECTED
    assert results[1]["error_code"] == "dependency_unavailable"
    assert results[2]["status"] == OK
    working = read_active_staged(store, WAKE)
    assert {obj["canonical_name"] for obj in working["objects"]} == {"Kept", "Also Kept"}


# ── inquiries (minimal) ────────────────────────────────────────────────────


def test_inquiry_create_and_drop(tmp_path) -> None:
    """Inquiry create stages a question; drop retires it."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-1",
                "kind": "inquiry",
                "action": "create",
                "payload": {
                    "subject_ref": "t1",
                    "prompt": "When did the club form?",
                    "rationale": "context for the transfer window",
                },
            }
        ],
    )[0]
    assert created["status"] == OK
    staged_id = created["staged_id"]
    assert staged_id.startswith(WAKE)
    working = read_active_staged(store, WAKE)
    assert working["inquiries"][0]["prompt"] == "When did the club form?"

    dropped = apply_patch(
        store, WAKE, [{"op_id": "op-2", "kind": "inquiry", "action": "drop", "target_ref": staged_id}]
    )[0]
    assert dropped["status"] == OK
    assert read_active_staged(store, WAKE)["inquiries"] == []


# ── tools-level wiring ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def _async_tools(tmp_path: pytest.TempPathFactory):
    tools, store = _tools(tmp_path, thread_id=WAKE)
    _seed(store)
    return tools, store


async def test_graph_patch_tool_envelope_and_validation(_async_tools) -> None:
    """graph_patch dispatches through the facade with a bounded batch envelope."""
    tools, store = _async_tools
    result = await tools.execute(
        "graph_patch",
        {"items": [_create_object("op-1", "Tool Club")]},
        "patch-call",
    )
    assert result["ok"] is True
    assert result["outcome"] == "success"
    assert result["counts"] == {"applied": 1, "replayed": 0, "problematic": 0}
    assert result["scope"] == {"wake_id": WAKE, "item_count": 1, "replay_count": 0}
    assert result["payload"]["results"][0]["status"] == OK
    assert result["payload"]["results"][0]["staged_id"].startswith(WAKE)

    mixed = await tools.execute(
        "graph_patch",
        {
            "items": [
                _create_object("op-mixed-good", "Mixed Good"),
                _create_assertion("op-mixed-bad", "missing-subject"),
            ]
        },
        "patch-mixed",
    )
    assert mixed["ok"] is True
    assert mixed["outcome"] == "partial"
    assert mixed["counts"] == {"applied": 1, "replayed": 0, "problematic": 1}

    not_a_list = await tools.execute("graph_patch", {"items": "nope"}, "patch-bad")
    assert not_a_list["ok"] is False
    assert not_a_list["limitations"] == ["invalid_arguments"]

    too_many = await tools.execute(
        "graph_patch", {"items": [_create_object(f"op-{i}", f"X{i}") for i in range(21)]}, "patch-cap"
    )
    assert too_many["ok"] is False


async def test_memory_read_include_working_sees_staged_creates(_async_tools) -> None:
    """A staged object is readable by its staged id only through the merged view."""
    tools, store = _async_tools
    staged_id = (await tools.execute("graph_patch", {"items": [_create_object("op-1", "Staged Star")]}, "p"))[
        "payload"
    ]["results"][0]["staged_id"]

    formal = await tools.execute("memory_read", {"object_id": staged_id}, "formal-read")
    assert formal["ok"] is True
    assert formal["memory"]["reasons"] == ["unknown object"]

    merged = await tools.execute(
        "memory_read", {"object_id": staged_id, "include_working": True}, "merged-read"
    )
    assert merged["ok"] is True
    assert merged["scope"]["include_working"] is True
    assert merged["memory"]["identity"]["object_id"] == staged_id
    assert merged["memory"]["identity"]["canonical_name"] == "Staged Star"
    # a plain create claims existence: provisional defaults to false (the
    # uncommitted-ness itself is carried by source=staged)
    assert merged["memory"]["identity"]["provisional"] is False
    assert merged["memory"]["anchor_objects"][0]["source"] == "staged"

    # an explicitly provisional create is reported as provisional
    tent_id = (
        await tools.execute(
            "graph_patch",
            {"items": [_create_object("op-2", "Maybe Entity", provisional=True)]},
            "p-tent",
        )
    )["payload"]["results"][0]["staged_id"]
    tent = await tools.execute("memory_read", {"object_id": tent_id, "include_working": True}, "tent-read")
    assert tent["memory"]["identity"]["provisional"] is True
    assert tent["memory"]["identity"]["object_id"] == tent_id


async def test_memory_read_include_working_merges_updates_and_hides_drops(_async_tools) -> None:
    """A staged update overlays the formal portrait; a staged drop hides it."""
    tools, store = _async_tools
    updated = await tools.execute(
        "graph_patch",
        {
            "items": [
                {
                    "op_id": "op-1",
                    "kind": "object",
                    "action": "update",
                    "target_ref": "t1",
                    "payload": {"canonical_name": "Team One Revised", "kind": "entity", "aliases": ["T1R"]},
                }
            ]
        },
        "p-update",
    )
    assert updated["payload"]["results"][0]["status"] == OK

    merged = await tools.execute("memory_read", {"object_id": "t1", "include_working": True}, "m-read")
    identity = merged["memory"]["identity"]
    assert identity["canonical_name"] == "Team One Revised"
    assert identity["active_aliases"] == [{"raw_alias": "T1R", "normalized_alias": "T1R"}]
    assert merged["memory"]["anchor_objects"][0]["source"] == "merged"

    # formal-only read stays formal
    formal = await tools.execute("memory_read", {"object_id": "t1"}, "f-read")
    assert formal["memory"]["identity"]["canonical_name"] == "Team One"

    # dropping the staged overlay retracts it: the merged view restores formal
    drop = await tools.execute(
        "graph_patch",
        {
            "items": [
                {
                    "op_id": "op-2",
                    "kind": "object",
                    "action": "drop",
                    "target_ref": updated["payload"]["results"][0]["staged_id"],
                }
            ]
        },
        "p-drop",
    )
    assert drop["payload"]["results"][0]["status"] == OK
    restored = await tools.execute("memory_read", {"object_id": "t1", "include_working": True}, "h-read")
    assert restored["memory"]["identity"]["canonical_name"] == "Team One"


async def test_memory_read_include_working_other_wake_invisible(_async_tools) -> None:
    """Staging of one wake never leaks into another wake's merged view."""
    tools, store = _async_tools
    staged_id = (await tools.execute("graph_patch", {"items": [_create_object("op-1", "Secret")]}, "p"))[
        "payload"
    ]["results"][0]["staged_id"]
    other_tools = WorldTools(store=store, adapters={}, thread_id=OTHER)
    read = await other_tools.execute(
        "memory_read", {"object_id": staged_id, "include_working": True}, "foreign-read"
    )
    assert read["memory"]["reasons"] == ["unknown object"]


async def test_world_tools_separates_thread_owner_from_working_wake(tmp_path) -> None:
    """Working graph state is wake-scoped even when several wakes share a thread."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="thread-shared",
        wake_id="wake-distinct",
    )

    result = await tools.execute(
        "graph_patch",
        {"items": [_create_object("op-1", "Wake Owned")]},
        "patch",
    )
    staged_id = result["payload"]["results"][0]["staged_id"]
    assert result["scope"]["wake_id"] == "wake-distinct"
    assert staged_id.startswith("wake-distinct:")
    assert read_active_staged(store, "thread-shared")["objects"] == []
    assert [row["staged_id"] for row in read_active_staged(store, "wake-distinct")["objects"]] == [staged_id]

    merged = await tools.execute(
        "memory_read",
        {"object_id": staged_id, "include_working": True},
        "read",
    )
    assert merged["memory"]["identity"]["canonical_name"] == "Wake Owned"


# ── slice 4 hardening: per-item isolation, literal round-trip, evidence ──────


def test_patch_exception_isolation_keeps_batch_siblings(tmp_path) -> None:
    """One malformed item rejects alone; already-staged siblings survive (Core §6.3.5)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    results = apply_patch(
        store,
        WAKE,
        [
            _create_object("op-1", "First"),
            {
                "op_id": "op-2",
                "kind": "assertion",
                "action": "create",
                "payload": {
                    "subject_ref": "t1",
                    "predicate": "related_to",
                    "object_ref": "s1",
                    "confidence": "not-a-number",
                },
            },
            _create_object("op-3", "Third"),
        ],
    )
    assert results[0]["status"] == OK
    assert results[1]["status"] == REJECTED
    assert results[1]["error_code"] == "invalid_tool_arguments"
    assert results[1]["field"] == "items[1].payload.confidence"
    assert results[2]["status"] == OK
    working = read_active_staged(store, WAKE)
    assert {row["canonical_name"] for row in working["objects"]} == {"First", "Third"}


def test_patch_literal_round_trip_and_duplicate_gate(tmp_path) -> None:
    """Literals store as JSON and the duplicate gate decodes them (DEV-2 fix)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    literal = {"zh": "冠军"}
    first = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-1",
                "kind": "assertion",
                "action": "create",
                "payload": {
                    "subject_ref": "t1",
                    "predicate": "has_title",
                    "literal": literal,
                    "epistemic_role": "fact",
                },
            }
        ],
    )[0]
    assert first["status"] == OK
    working = read_active_staged(store, WAKE)
    assert working["assertions"][0]["literal"] == literal

    # the exact-duplicate gate compares decoded literals, never raw JSON text
    dup = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "assertion",
                "action": "create",
                "payload": {
                    "subject_ref": "t1",
                    "predicate": "has_title",
                    "literal": literal,
                    "epistemic_role": "fact",
                },
            }
        ],
    )[0]
    assert dup["status"] == REJECTED
    assert dup["error_code"] == "exact_duplicate"


def test_patch_evidence_unavailable_rejects_stale_observation(tmp_path) -> None:
    """Evidence refs must match stored observations (plan §6.5, GAP-2 fix)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    result = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-1",
                "kind": "assertion",
                "action": "create",
                "payload": {
                    "subject_ref": "t1",
                    "predicate": "related_to",
                    "object_ref": "s1",
                    "epistemic_role": "fact",
                    "evidence": ["obs-missing"],
                },
            }
        ],
    )[0]
    assert result["status"] == REJECTED
    assert result["error_code"] == "evidence_unavailable"
    assert result["missing_evidence_ids"] == ["obs-missing"]


def test_patch_formal_target_version_conflict(tmp_path) -> None:
    """Updating a formal target with a stale expected_version conflicts (DEV-8 gate)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    stale = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-1",
                "kind": "object",
                "action": "update",
                "target_ref": "t1",
                "expected_version": 99,
                "payload": {"canonical_name": "Team One Revised", "kind": "entity"},
            }
        ],
    )[0]
    assert stale["status"] == REJECTED
    assert stale["error_code"] == "version_conflict"
    assert stale["current_version"] == 1

    fresh = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "object",
                "action": "update",
                "target_ref": "t1",
                "expected_version": stale["current_version"],
                "payload": {"canonical_name": "Team One Revised", "kind": "entity"},
            }
        ],
    )[0]
    assert fresh["status"] == OK


def test_formal_overlay_records_and_preserves_patch_time_base_version(tmp_path) -> None:
    """A formal overlay keeps the first observed formal base through later revisions."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    first = apply_patch(
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
    assert first["status"] == OK

    working = read_active_staged(store, WAKE)
    assert working["objects"][0]["base_version"] == 1

    revised = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "object",
                "action": "update",
                "target_ref": first["staged_id"],
                "expected_version": 1,
                "payload": {"canonical_name": "Team One Final Draft", "kind": "entity"},
            }
        ],
    )[0]
    assert revised["status"] == OK
    working = read_active_staged(store, WAKE)
    assert working["objects"][0]["base_version"] == 1


def test_inquiry_deepens_requires_live_target_and_drop_dependents(tmp_path) -> None:
    """deepens_ref must resolve; a deepened inquiry cannot be dropped first."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    def _inquiry(op_id: str, prompt: str, **overrides: object) -> dict[str, object]:
        return {
            "op_id": op_id,
            "kind": "inquiry",
            "action": "create",
            "payload": {
                "subject_ref": "t1",
                "prompt": prompt,
                "rationale": "test rationale",
                "kind": "factual",
                **overrides,
            },
        }

    broken = apply_patch(store, WAKE, [_inquiry("op-1", "base", deepens_ref="inq-ghost")])[0]
    assert broken["status"] == REJECTED
    assert broken["error_code"] == "dependency_unavailable"

    base = apply_patch(store, WAKE, [_inquiry("op-2", "base")])[0]
    assert base["status"] == OK
    assert base["version"] == 1
    deepened = apply_patch(store, WAKE, [_inquiry("op-3", "deeper", deepens_ref=base["staged_id"])])[0]
    assert deepened["status"] == OK

    # the deepened inquiry cannot be dropped while it is deepened
    drop_base = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-4", "kind": "inquiry", "action": "drop", "target_ref": base["staged_id"]}],
    )[0]
    assert drop_base["status"] == REJECTED
    assert drop_base["error_code"] == "dependent_exists"

    # dropping the deeper one first releases the chain
    drop_deep = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-5", "kind": "inquiry", "action": "drop", "target_ref": deepened["staged_id"]}],
    )[0]
    assert drop_deep["status"] == OK
    drop_base = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-6", "kind": "inquiry", "action": "drop", "target_ref": base["staged_id"]}],
    )[0]
    assert drop_base["status"] == OK


# ── Wave B: staged answer refs and inquiry resolve (v18) ────────────────────


def _resolve_op(op_id: str, target_ref: str | None, **overrides: object) -> dict[str, object]:
    return {
        "op_id": op_id,
        "kind": "inquiry",
        "action": "resolve",
        "target_ref": target_ref,
        "payload": {"expected_version": 1, **overrides},
    }


def _inquiry_op(op_id: str, prompt: str, **overrides: object) -> dict[str, object]:
    return {
        "op_id": op_id,
        "kind": "inquiry",
        "action": "create",
        "payload": {
            "subject_ref": "t1",
            "prompt": prompt,
            "rationale": "test rationale",
            "kind": "factual",
            **overrides,
        },
    }


def test_inquiry_create_rejects_answers_ref(tmp_path) -> None:
    """answers_ref belongs to the answering assertion, not to the inquiry: a
    create that carries it fails with a typed error naming the official path
    (assertion.answer), and nothing is staged."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    result = apply_patch(store, WAKE, [_inquiry_op("op-1", "Why?", answers_ref="a-ghost")])[0]
    assert result["status"] == REJECTED
    assert result["error_code"] == "invalid_tool_arguments"
    assert result["field"] == "items[0].payload.answers_ref"
    assert "kind 'assertion'" in result["action_hint"]
    assert read_active_staged(store, WAKE)["inquiries"] == []


def test_assertion_answers_ref_requires_live_inquiry(tmp_path) -> None:
    """A staged answering assertion must name a live inquiry (formal or active
    staged); a ghost target fails the op with a typed error."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    broken = apply_patch(store, WAKE, [_create_assertion("op-1", "t1", answers_ref="inq-ghost")])[0]
    assert broken["status"] == REJECTED
    assert broken["error_code"] == "dependency_unavailable"

    inq = apply_patch(store, WAKE, [_inquiry_op("op-2", "Why?")])[0]
    assert inq["status"] == OK
    answer = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-3", "t1", predicate="explains", answers_ref=inq["staged_id"])],
    )[0]
    assert answer["status"] == OK
    assert read_active_staged(store, WAKE)["assertions"]  # stored


def test_inquiry_resolve_requires_target_answer_and_version(tmp_path) -> None:
    """Resolve fail-closes on every bad input: missing/ghost target, missing/
    ghost answer, wrong expected version, and an answers_ref that does not
    declare the target inquiry."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    inq = apply_patch(store, WAKE, [_inquiry_op("op-1", "Why?")])[0]
    assert inq["status"] == OK
    answer = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-2", "t1", predicate="explains", answers_ref=inq["staged_id"])],
    )[0]
    assert answer["status"] == OK

    missing = apply_patch(store, WAKE, [_resolve_op("op-3", None)])[0]
    assert missing["status"] == REJECTED
    assert missing["error_code"] == "invalid_tool_arguments"
    assert missing["field"] == "items[0].target_ref"

    ghost_target = apply_patch(
        store,
        WAKE,
        [_resolve_op("op-4", "inq-ghost", answers_ref=answer["staged_id"])],
    )[0]
    assert ghost_target["status"] == REJECTED
    assert ghost_target["error_code"] == "target_unavailable"

    no_answer = apply_patch(store, WAKE, [_resolve_op("op-5", inq["staged_id"])])[0]
    assert no_answer["status"] == REJECTED
    assert no_answer["error_code"] == "invalid_tool_arguments"
    assert no_answer["field"] == "items[0].payload.answers_ref"

    ghost_answer = apply_patch(store, WAKE, [_resolve_op("op-6", inq["staged_id"], answers_ref="a-ghost")])[0]
    assert ghost_answer["status"] == REJECTED
    assert ghost_answer["error_code"] == "dependency_unavailable"

    # an assertion that exists but answers a different inquiry is not an
    # answering assertion for this target
    other = apply_patch(store, WAKE, [_inquiry_op("op-7", "Other?")])[0]
    assert other["status"] == OK
    other_answer = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-8", "t1", predicate="expresses", answers_ref=other["staged_id"])],
    )[0]
    assert other_answer["status"] == OK
    wrong = apply_patch(
        store, WAKE, [_resolve_op("op-9", inq["staged_id"], answers_ref=other_answer["staged_id"])]
    )[0]
    assert wrong["status"] == REJECTED
    assert wrong["error_code"] == "not_an_answering_assertion"

    bad_version = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-10",
                "kind": "inquiry",
                "action": "resolve",
                "target_ref": inq["staged_id"],
                "payload": {"expected_version": 99, "answers_ref": answer["staged_id"]},
            }
        ],
    )[0]
    assert bad_version["status"] == REJECTED
    assert bad_version["error_code"] == "version_conflict"


def test_inquiry_resolve_stages_and_blocks_drop_of_target(tmp_path) -> None:
    """A staged resolve is an active staged row referencing its target inquiry:
    dropping the target is rejected while the resolve lives; dropping the
    answer assertion or the resolve itself stays legal (Core §6.7)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    inq = apply_patch(store, WAKE, [_inquiry_op("op-1", "Why?")])[0]
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
    assert resolved["action"] == "resolved"

    # the resolve row keeps the target inquiry alive
    drop_target = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-4", "kind": "inquiry", "action": "drop", "target_ref": inq["staged_id"]}],
    )[0]
    assert drop_target["status"] == REJECTED
    assert drop_target["error_code"] == "dependent_exists"

    # the answer assertion is not a dependent of the resolve: dropping it is
    # allowed (finalize then fails closed on the missing co-published answer)
    drop_answer = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-5", "kind": "assertion", "action": "drop", "target_ref": answer["staged_id"]}],
    )[0]
    assert drop_answer["status"] == OK

    # the resolve row itself can be dropped, releasing the target
    drop_resolve = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-6", "kind": "inquiry", "action": "drop", "target_ref": resolved["staged_id"]}],
    )[0]
    assert drop_resolve["status"] == OK
    drop_target = apply_patch(
        store,
        WAKE,
        [{"op_id": "op-7", "kind": "inquiry", "action": "drop", "target_ref": inq["staged_id"]}],
    )[0]
    assert drop_target["status"] == OK


def test_inquiry_resolve_with_formal_answers_ref_passes_staging(tmp_path) -> None:
    """B4-5 staging shape: a resolve may target a FORMAL inquiry and name its
    FORMAL answering assertion — staging accepts it (every reference resolves);
    finalize must then fail the delta because no staged answer is co-published
    with the resolution."""
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
    assert resolved["action"] == "resolved"


def test_resolve_op_replay_is_idempotent(tmp_path) -> None:
    """B4-7: answer/deepen/resolve ops replay through the patch ledger — the
    same op_id with the same payload replays the original verdict and never
    duplicates the resolution row."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    inq = apply_patch(store, WAKE, [_inquiry_op("op-1", "who is the champion?")])[0]
    assert inq["status"] == OK
    answer = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-2", "t1", predicate="explains", answers_ref=inq["staged_id"])],
    )[0]
    assert answer["status"] == OK
    resolve_op = _resolve_op("op-3", inq["staged_id"], answers_ref=answer["staged_id"])
    first = apply_patch(store, WAKE, [resolve_op])[0]
    assert first["status"] == OK
    assert first["action"] == "resolved"

    replay = apply_patch(store, WAKE, [resolve_op])[0]
    assert replay["status"] == OK
    assert replay == first
    assert replay["staged_id"] == first["staged_id"]
    working = read_active_staged(store, WAKE)
    resolution_rows = [row for row in working["inquiries"] if row["kind"] == "resolution"]
    assert len(resolution_rows) == 1  # replayed resolve does not duplicate


# ── Wave C C3 contract shapes ────────────────────────────────────────────────


def test_identity_candidates_carry_name_kind_basis(tmp_path) -> None:
    """C3: every identity candidate carries id, name, kind and basis so the
    agent can decide reuse vs confirm_distinct without extra reads."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    colliding = apply_patch(store, WAKE, [_create_object("op-1", "Team One")])[0]
    assert colliding["status"] == NEEDS_IDENTITY_RESOLUTION
    assert colliding["error_code"] == "identity_candidate_exists"
    assert colliding["candidates"], "a canonical_exact candidate must be returned"
    for entry in colliding["candidates"]:
        assert {"id", "name", "kind", "basis"} <= set(entry)
        assert entry["basis"] in {"canonical_exact", "staged_alias", "alias_exact"}
        assert entry["name"]
        assert entry["kind"] in {"entity", "event", "concept", "organization", "person", "place"}
    formal = next(entry for entry in colliding["candidates"] if entry["id"] == "t1")
    assert formal["basis"] == "canonical_exact"
    assert formal["kind"] == "entity"

    # the enriched candidates also flow through the inspect blocker: stage a
    # second "Team One" whose confirm_distinct basis names only the formal t1,
    # leaving the same-wake staged s1 as an unresolved candidate
    confirmed_t1 = {
        **_create_object("op-2", "Team One"),
        "decision": {"action": "confirm_distinct", "distinct_from": "t1", "basis": "different team"},
    }
    first = apply_patch(store, WAKE, [confirmed_t1])[0]
    assert first["status"] == OK
    second = apply_patch(
        store,
        WAKE,
        [
            {
                **_create_object("op-3", "Team One"),
                "decision": {"action": "confirm_distinct", "distinct_from": "t1", "basis": "another team"},
            }
        ],
    )[0]
    assert second["status"] == OK
    inspect = inspect_working_graph(store, WAKE)
    blocker = next(entry for entry in inspect["blockers"] if entry["code"] == "identity_unresolved")
    assert blocker["candidates"], "the same-wake staged candidate stays unresolved"
    for entry in blocker["candidates"]:
        assert {"id", "name", "kind", "basis"} <= set(entry)


def test_supersede_result_echoes_old_to_new_chain(tmp_path) -> None:
    """C3: a supersede success returns the new staged id and the superseded
    target (old -> new), and a follow-up supersede chains onto the new id."""
    store = WorldStore(tmp_path / "world.sqlite3")
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
        "seed",
    )
    first = apply_patch(
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
    assert first["status"] == OK
    assert first["supersedes"] == "a-1"
    assert first["staged_id"] == "wake-a:a1"

    second = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "assertion",
                "action": "supersede",
                "payload": {
                    "subject_ref": "t1",
                    "predicate": "related_to",
                    "object_ref": "s1",
                    "epistemic_role": "fact",
                    "confidence": 0.99,
                    "supersedes_ref": "wake-a:a1",
                },
            }
        ],
    )[0]
    assert second["status"] == OK
    assert second["supersedes"] == "wake-a:a1"
    assert second["staged_id"] == "wake-a:a2"


def test_successful_patch_items_return_readable_summary(tmp_path) -> None:
    """C3: every successful patch item returns its staged id and a short
    human-readable summary of the structure it staged."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    results = apply_patch(
        store,
        WAKE,
        [
            _create_object("op-1", "New Concept"),
            _create_assertion("op-2", "t1", predicate="explains", object_ref="s1"),
        ],
    )
    for result in results:
        assert result["status"] == OK
        assert result["staged_id"]
        assert "summary" in result
        assert len(result["summary"]) >= 10
    assert "New Concept" in results[0]["summary"]
    assert results[0]["staged_id"] == "wake-a:s1"
    assert results[1]["staged_id"] == "wake-a:a1"
    assert "t1" in results[1]["summary"]


def test_object_create_similar_name_hint_containment(tmp_path) -> None:
    """A create whose canonical contains an existing formal canonical reports
    an advisory hint (name_containment), never a block."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)  # formal: Team One (t1), Summer Cup (s1)
    result = apply_patch(store, WAKE, [_create_object("op-1", "Team One Esports")])[0]
    assert result["status"] == OK
    assert result["staged_id"]  # still staged: hints are advisory
    similar = result["similar_objects"]
    assert similar[0]["id"] == "t1"
    assert similar[0]["name"] == "Team One"
    assert similar[0]["basis"] == "name_containment"
    assert result["hints"]


def test_object_create_similar_name_hint_bigrams(tmp_path) -> None:
    """Bigram similarity fires when containment does not (transposed words)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    # "One Team" vs "Team One": same bigrams, different order.
    result = apply_patch(store, WAKE, [_create_object("op-1", "One Team")])[0]
    assert result["status"] == OK
    similar = result["similar_objects"]
    assert similar[0]["id"] == "t1"
    assert similar[0]["basis"] == "name_similar"


def test_object_create_canonical_matches_active_alias_hint(tmp_path) -> None:
    """A new canonical equal to an active identity alias reports the owner as
    a hint (advisory), not the hard occupied block."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="t2", kind=ObjectKind.ENTITY, canonical_name="Team One NA", aliases=["T1"])
            ]
        ),
        "seed-alias",
    )
    result = apply_patch(store, WAKE, [_create_object("op-1", "T1")])[0]
    assert result["status"] == OK  # advisory only, never a block
    similar = result["similar_objects"]
    assert similar[0]["id"] == "t2"
    assert similar[0]["basis"] == "canonical_matches_active_alias"
    assert similar[0]["similarity"] == 1.0


def test_object_create_similar_staged_sibling_hint(tmp_path) -> None:
    """Similarity is also reported against this wake's own staged objects."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    first = apply_patch(store, WAKE, [_create_object("op-1", "Alpha Arena")])[0]
    assert "similar_objects" not in first
    second = apply_patch(store, WAKE, [_create_object("op-2", "Alpha Arena II")])[0]
    assert second["status"] == OK
    similar = second["similar_objects"]
    assert similar[0]["id"] == first["staged_id"]
    assert similar[0]["basis"] == "name_containment"


def test_object_create_no_similar_hint_when_names_distinct(tmp_path) -> None:
    """A clearly distinct canonical gets no similar_objects field at all."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    result = apply_patch(store, WAKE, [_create_object("op-1", "Random Esports Club")])[0]
    assert result["status"] == OK
    assert "similar_objects" not in result
    assert "hints" not in result
