"""Slice 5 Part 1 acceptance: finalize_graph bridge and crash-point semantics (plan §6.10/§6.11).

The deterministic publish path: readiness reuses inspect_working_graph
(blocked keeps staging), the compile is a pure function of staging + formal
tables, the formal commit and the staging finalization share one SQLite
transaction (D-007), and a stored finalize receipt wins every replay. The
six §6.11 crash points are covered behavior-equivalently:

- 1/2 (patch written / result not checkpointed): the patch ledger replays an
  op with the original result and one staged row (test_patch_replay...).
- 3/4 (crash before the formal transaction commits / receipt written but
  staging not finalized): the same-transaction finalizer makes the split
  states unobservable — a failure anywhere inside the transaction rolls the
  whole commit back and a retry converges (test_crash_inside_transaction...).
- 5 (success but result not returned): re-finalize replays the durable
  receipt (test_result_lost_replays...).
- 6 (same op replayed on resume): converges on the same staged rows and the
  same receipt (test_patch_replay... and test_idempotent_repeat...).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from leave_information_bubble.world import ObjectKind, WorldStore, WorldTools
from leave_information_bubble.world import finalize as finalize_module
from leave_information_bubble.world.contracts import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    InquiryInput,
    InquiryResolution,
    ObjectInput,
    ObservationDepth,
    ObservationInput,
)
from leave_information_bubble.world.finalize import FinalizeCompileError, finalize_graph
from leave_information_bubble.world.preflight import inspect_working_graph
from leave_information_bubble.world.staging import OK, REJECTED, apply_patch, read_active_staged

WAKE = "wake-finalize"
OTHER_WAKE = "wake-finalize-2"

FORMAL_OBJECTS = [
    ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One", aliases=["T1"]),
    ObjectInput(id="s1", kind=ObjectKind.EVENT, canonical_name="Summer Cup"),
]

FORMAL_ASSERTION = AssertionInput(
    id="a-1",
    subject_id="t1",
    predicate="related_to",
    object_id="s1",
    epistemic_role=EpistemicRole.FACT,
    confidence=0.8,
)

FORMAL_OBSERVATION = ObservationInput(
    id="obs-1",
    source_uri="https://example.test/obs-1",
    source_kind="public-web",
    title="a source",
    depth=ObservationDepth.CONTENT,
    observed_at="2026-08-01T00:00:00+00:00",
)


def _seed(store: WorldStore, commit_id: str = "seed") -> None:
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            observations=[FORMAL_OBSERVATION],
            assertions=[FORMAL_ASSERTION],
        ),
        commit_id,
    )


def _create_object(op_id: str, canonical: str, **overrides: object) -> dict[str, object]:
    return {
        "op_id": op_id,
        "kind": "object",
        "action": "create",
        "payload": {"canonical_name": canonical, "kind": "entity", **overrides},
    }


def _update_object(op_id: str, target_ref: str, **overrides: object) -> dict[str, object]:
    item = {
        "op_id": op_id,
        "kind": "object",
        "action": "update",
        "target_ref": target_ref,
        "payload": {"canonical_name": "Team One Updated", "kind": "entity"},
    }
    expected_version = overrides.pop("expected_version", None)
    if expected_version is not None:
        item["expected_version"] = expected_version  # item-level version protocol
    item["payload"].update(overrides)
    return item


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


def _drop(op_id: str, kind: str, target_ref: str) -> dict[str, object]:
    return {"op_id": op_id, "kind": kind, "action": "drop", "target_ref": target_ref}


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


def _staged(store: WorldStore, wake_id: str) -> dict[str, list[dict[str, object]]]:
    return read_active_staged(store, wake_id)


# ── happy path: the complete working graph publishes deterministically ───────


def test_finalize_publishes_complete_working_graph(tmp_path) -> None:
    """Creates, an overlay update, a person-route, an inquiry, and evidence publish."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    created = apply_patch(
        store,
        WAKE,
        [
            _create_object(
                "op-1",
                "Coach Zhang",
                kind="person",
                aliases=["CoachZ"],
                event_time_start="2020-01-01T00:00:00+00:00",
            ),
            _update_object("op-2", "t1", expected_version=1, aliases=["T1", "T1-X"]),
        ],
    )
    assert [entry["status"] for entry in created] == [OK, OK]
    coach = created[0]["staged_id"]

    applied = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-3",
                coach,
                predicate="coached_by",
                object_ref=None,
                literal="Coaching style: direct",
                evidence=["obs-1"],
            ),
            _create_inquiry("op-4", coach),
        ],
    )
    assert [entry["status"] for entry in applied] == [OK, OK]
    assert _staged(store, WAKE)["assertions"][0]["literal"] == "Coaching style: direct"

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.commit_id == f"{WAKE}:finalize"
    assert receipt.committed_at is not None
    assert receipt.stats.objects_created == 1
    assert receipt.stats.objects_updated == 1
    assert receipt.stats.assertions_created == 1
    assert receipt.stats.inquiries_created == 1
    assert receipt.stats.total_items == 4
    assert not receipt.all_work_abandoned
    # every staged id is mapped to its formal id in the receipt
    assert receipt.item_ids[coach] == coach
    assert receipt.item_ids[created[1]["staged_id"]] == "t1"

    with store.read_connection() as connection:
        formal_coach = connection.execute(
            "SELECT id, kind, type_key, canonical_name FROM objects WHERE id = ?", (coach,)
        ).fetchone()
        # the staged "person" publishes verbatim as a formal person kind
        assert formal_coach["kind"] == "person"
        assert formal_coach["type_key"] is None
        assert formal_coach["canonical_name"] == "Coach Zhang"
        formal_assertion = connection.execute(
            "SELECT id, subject_id, literal_json, confidence FROM assertions WHERE subject_id = ?",
            (coach,),
        ).fetchone()
        assert formal_assertion["literal_json"] == '"Coaching style: direct"'
        evidence_row = connection.execute(
            "SELECT assertion_id, observation_id FROM assertion_evidence"
            " WHERE assertion_id = ?",
            (formal_assertion["id"],),
        ).fetchone()
        assert evidence_row["observation_id"] == "obs-1"
        # the overlay update bumps the formal version instead of duplicating
        overlay = connection.execute(
            "SELECT version, canonical_name FROM objects WHERE id = 't1'"
        ).fetchone()
        assert overlay["version"] == 2
        assert overlay["canonical_name"] == "Team One Updated"
        alias_owner = connection.execute(
            "SELECT object_id FROM identity_aliases WHERE normalized_alias = 't1-x'"
        ).fetchone()
        assert alias_owner["object_id"] == "t1"
        formal_inquiry = connection.execute(
            "SELECT id, subject_id, kind FROM inquiries WHERE subject_id = ?", (coach,)
        ).fetchone()
        assert formal_inquiry["kind"] == "factual"

    # staging converged to the finalized terminal state
    assert _staged(store, WAKE) == {"objects": [], "assertions": [], "inquiries": []}
    with store.read_connection() as connection:
        finalized = connection.execute(
            "SELECT COUNT(*) AS count FROM staged_objects"
            " WHERE wake_id = ? AND status = 'finalized'",
            (WAKE,),
        ).fetchone()["count"]
    assert finalized == 2


def test_finalize_drops_invented_qualifier_keys_and_reports(tmp_path) -> None:
    """F-B publish tolerance: legacy staging with an invented qualifier key
    publishes with the key dropped and a problem entry, never compile_failed.

    The write path now rejects invented keys, so this simulates the historical
    state by editing the staged row directly.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    applied = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-1",
                "t1",
                predicate="related_to",
                object_ref="s1",
                qualifiers={"role": "observer"},
            )
        ],
    )
    assert applied[0]["status"] == OK
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE staged_assertions SET qualifiers_json = ?, updated_at = updated_at"
            " WHERE wake_id = ?",
            (
                '{"role": "observer", "conflicting_source": "esports8 lists Jiejie on WBG"}',
                WAKE,
            ),
        )

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.commit_id == f"{WAKE}:finalize"
    dropped = [
        w for w in receipt.warnings if w.get("code") == "qualifier_keys_dropped"
    ]
    assert len(dropped) == 1
    assert "conflicting_source" in dropped[0]["message"]
    assert receipt.problems == []
    with store.read_connection() as connection:
        formal = connection.execute(
            "SELECT qualifiers_json FROM assertions WHERE id LIKE '%a1'"
        ).fetchone()
        assert json.loads(formal["qualifiers_json"]) == {"role": "observer"}


def test_finalize_supersede_grows_chain_and_confidence_only_dedups(tmp_path) -> None:
    """A literal-change supersede extends the formal chain; a metadata-only one dedups.

    The formal assertion signature deliberately excludes confidence and
    evidence (committer._assertion_signature): a correction that changes no
    identity input collapses onto the superseded row — announced in the
    receipt's item_ids mapping, never silent.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    chain = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-1",
                "t1",
                predicate="ranked",
                object_ref=None,
                literal="1st",
                supersedes_ref="a-1",
                confidence=0.9,
                evidence=["obs-1"],
            ),
            _create_assertion(
                "op-2",
                "t1",
                predicate="related_to",
                object_ref="s1",
                supersedes_ref="a-1",
                confidence=0.95,
            ),
        ],
    )
    assert [entry["status"] for entry in chain] == [OK, OK]

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.assertions_created == 2
    # the confidence-only correction maps onto the superseded formal row
    assert receipt.item_ids[f"{WAKE}:a2"] == "a-1"

    with store.read_connection() as connection:
        replacement = connection.execute(
            "SELECT id, supersedes_id, literal_json FROM assertions WHERE supersedes_id = 'a-1'"
        ).fetchall()
        # only the identity-input change grew the chain: one new current row
        assert len(replacement) == 1
        assert replacement[0]["literal_json"] == '"1st"'
        assert replacement[0]["id"] == f"{WAKE}:a1"
        # the chain-row evidence is linked to the new row
        evidence = connection.execute(
            "SELECT observation_id FROM assertion_evidence WHERE assertion_id = ?",
            (f"{WAKE}:a1",),
        ).fetchone()
        assert evidence is not None and evidence["observation_id"] == "obs-1"
        # the metadata-only correction announced itself: no_evidence warning
        assert any(
            entry["code"] == "no_evidence" and entry["ref"] == f"{WAKE}:a2"
            for entry in receipt.warnings
        )


# ── staged→staged supersede (G5a F-1): a correction of this wake's own ────────
# ── staged assertion is a first-class patch operation and must compile ────────


def test_finalize_staged_supersede_publishes_chain(tmp_path) -> None:
    """A supersede targeting a live staged assertion publishes the full chain."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    target = apply_patch(
        store,
        WAKE,
        [_create_assertion("op-1", "t1", predicate="ranked", object_ref=None, literal="1st")],
    )[0]
    assert target["status"] == OK
    assert target["staged_id"] == f"{WAKE}:a1"
    superseder = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-2",
                "t1",
                predicate="ranked",
                object_ref=None,
                literal="2nd",
                supersedes_ref=f"{WAKE}:a1",
            )
        ],
    )[0]
    assert superseder["status"] == OK

    # patch and inspect agree the working graph is publishable (Core §6.9)
    report = inspect_working_graph(store, WAKE)
    assert report["readiness"] == "ready"
    assert not any(entry["code"] == "supersede_cycle" for entry in report["blockers"])

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.assertions_created == 2
    # both staged ids keep their host-issued ids and are announced truthfully
    assert receipt.item_ids[f"{WAKE}:a1"] == f"{WAKE}:a1"
    assert receipt.item_ids[f"{WAKE}:a2"] == f"{WAKE}:a2"

    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT id, supersedes_id, literal_json FROM assertions WHERE id IN (?, ?)",
            (f"{WAKE}:a1", f"{WAKE}:a2"),
        ).fetchall()
    by_id = {row["id"]: row for row in rows}
    # the staged target published under its staged id; the superseder chains it
    assert by_id[f"{WAKE}:a1"]["supersedes_id"] is None
    assert by_id[f"{WAKE}:a1"]["literal_json"] == '"1st"'
    assert by_id[f"{WAKE}:a2"]["supersedes_id"] == f"{WAKE}:a1"
    assert by_id[f"{WAKE}:a2"]["literal_json"] == '"2nd"'


def test_finalize_staged_supersede_two_digit_target_publishes(tmp_path) -> None:
    """A two-digit superseder that sorts before its single-digit target still publishes (G5a F-6).

    Staged ids are unpadded per-wake counters (``{wake_id}:a{n}``), so once the
    wake has ten or more staged rows the string order of ``read_active_staged``
    places ``:a11`` before ``:a2``. The compile resolver must not depend on row
    order to have seen the supersede target.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    for index in range(1, 11):
        applied = apply_patch(
            store,
            WAKE,
            [
                _create_assertion(
                    f"op-{index}",
                    "t1",
                    predicate="ranked",
                    object_ref=None,
                    literal=f"rank-{index}",
                )
            ],
        )[0]
        assert applied["status"] == OK
        assert applied["staged_id"] == f"{WAKE}:a{index}"
    superseder = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-11",
                "t1",
                predicate="ranked",
                object_ref=None,
                literal="rank-11",
                supersedes_ref=f"{WAKE}:a2",
            )
        ],
    )[0]
    assert superseder["status"] == OK
    assert superseder["staged_id"] == f"{WAKE}:a11"

    # the working graph is publishable and the chain survives compile order
    assert inspect_working_graph(store, WAKE)["readiness"] == "ready"
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.assertions_created == 11
    assert receipt.item_ids[f"{WAKE}:a11"] == f"{WAKE}:a11"
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT supersedes_id FROM assertions WHERE id = ?", (f"{WAKE}:a11",)
        ).fetchone()
    assert row["supersedes_id"] == f"{WAKE}:a2"


def test_finalize_compile_rejects_dangling_supersede_ref(tmp_path) -> None:
    """A supersedes ref that is neither staged nor formal fails closed at compile (G5a F-1)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    now = "2026-08-20T00:00:00+00:00"
    # the patch gate rejects dangling supersedes refs, so force the row
    # directly to prove the compile resolver itself never drops it silently
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO staged_assertions (staged_id, wake_id, status, target_ref,"
            " subject_ref, predicate, object_ref, literal_json, epistemic_role, confidence,"
            " event_time_start, event_time_end, supersedes_ref, qualifiers_json,"
            " evidence_json, created_at, updated_at, version)"
            " VALUES (?, ?, 'active', NULL, 't1', 'ranked', NULL, ?, 'fact', 0.9,"
            " NULL, NULL, 'no-such-assertion', '{}', '[]', ?, ?, 1)",
            (f"{WAKE}:a1", WAKE, '"1st"', now, now),
        )
    with pytest.raises(FinalizeCompileError) as error:
        finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert any("no-such-assertion" in problem for problem in error.value.problems)


def test_finalize_compile_rejects_supersede_cycle(tmp_path) -> None:
    """A supersede cycle fails closed at compile, before any store write (G5a F-6)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    now = "2026-08-20T00:00:00+00:00"
    # preflight blocks cycles, so force one directly to prove the compile's
    # write-order pass never turns it into a foreign-key crash
    with store.write_connection() as connection:
        for staged_id, supersedes, literal in (
            (f"{WAKE}:a1", f"{WAKE}:a2", '"x"'),
            (f"{WAKE}:a2", f"{WAKE}:a1", '"y"'),
        ):
            connection.execute(
                "INSERT INTO staged_assertions (staged_id, wake_id, status, target_ref,"
                " subject_ref, predicate, object_ref, literal_json, epistemic_role,"
                " confidence, event_time_start, event_time_end, supersedes_ref,"
                " qualifiers_json, evidence_json, created_at, updated_at, version)"
                " VALUES (?, ?, 'active', NULL, 't1', 'ranked', NULL, ?, 'fact', 0.9,"
                " NULL, NULL, ?, '{}', '[]', ?, ?, 1)",
                (staged_id, WAKE, literal, supersedes, now, now),
            )
    with pytest.raises(FinalizeCompileError) as error:
        finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert any("supersede cycle" in problem for problem in error.value.problems)


def test_finalize_supersede_of_signature_twin_collapses(tmp_path) -> None:
    """A metadata-only supersede crossing the ten-row boundary collapses onto the
    dedup winner instead of self-referencing a fresh row (G5a F-6).

    ``a10`` sorts before ``a9`` in string order, so the later-staged twin is the
    signature carrier: its supersedes edge would point at its own row, which is
    being inserted for the first time — the compile drops that edge (the store
    never materializes it either) and the receipt announces the collapse.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    for index in range(1, 9):
        applied = apply_patch(
            store,
            WAKE,
            [
                _create_assertion(
                    f"op-{index}",
                    "t1",
                    predicate="ranked",
                    object_ref=None,
                    literal=f"rank-{index}",
                )
            ],
        )[0]
        assert applied["status"] == OK
    original = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-9",
                "t1",
                predicate="ranked",
                object_ref=None,
                literal="twin",
                confidence=0.8,
                evidence=["obs-1"],
            )
        ],
    )[0]
    assert original["status"] == OK
    assert original["staged_id"] == f"{WAKE}:a9"
    correction = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-10",
                "t1",
                predicate="ranked",
                object_ref=None,
                literal="twin",
                confidence=0.9,
                supersedes_ref=f"{WAKE}:a9",
                evidence=["obs-1"],
            )
        ],
    )[0]
    assert correction["status"] == OK
    assert correction["staged_id"] == f"{WAKE}:a10"

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    # both twins collapse onto the string-order carrier, truthfully announced
    assert receipt.item_ids[f"{WAKE}:a9"] == f"{WAKE}:a10"
    assert receipt.item_ids[f"{WAKE}:a10"] == f"{WAKE}:a10"
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT id, supersedes_id FROM assertions WHERE id = ?", (f"{WAKE}:a10",)
        ).fetchone()
        assert row is not None and row["supersedes_id"] is None
        # both twins' evidence landed on the surviving row, deduplicated
        evidence = connection.execute(
            "SELECT observation_id FROM assertion_evidence WHERE assertion_id = ?",
            (f"{WAKE}:a10",),
        ).fetchall()
        assert [entry["observation_id"] for entry in evidence] == ["obs-1"]


def test_finalize_cross_wake_staged_supersede_rejected_at_patch_gate(tmp_path) -> None:
    """Another wake's staged assertion is never a valid supersede target (G5a F-1)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    other = apply_patch(
        store,
        OTHER_WAKE,
        [_create_assertion("op-1", "t1", predicate="ranked", object_ref=None, literal="1st")],
    )[0]
    assert other["status"] == OK

    rejected = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-2",
                "t1",
                predicate="ranked",
                object_ref=None,
                literal="2nd",
                supersedes_ref=other["staged_id"],
            )
        ],
    )[0]
    assert rejected["status"] == REJECTED
    assert rejected["error_code"] == "dependency_unavailable"


# ── post-finalize staging (G5a F-3): explicit closure, never a silent replay ─


def test_finalize_post_publish_staging_returns_wake_closed_and_new_wake_publishes(
    tmp_path,
) -> None:
    """Items staged after a publish are explicitly rejected, not silently lost."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    first = apply_patch(store, WAKE, [_create_object("op-1", "Settled")])[0]
    assert first["status"] == OK
    apply_patch(store, WAKE, [_create_assertion("op-2", first["staged_id"])])
    published = finalize_graph(store, WAKE)
    assert published.status == "published"

    # the agent keeps working in the same wake after the publish
    late = apply_patch(store, WAKE, [_create_object("op-3", "Latecomer")])[0]
    assert late["status"] == OK
    apply_patch(store, WAKE, [_create_assertion("op-4", late["staged_id"])])

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "wake_closed"
    assert receipt.commit_id == published.commit_id
    # the rejection describes the NEW items, never the stale published stats
    assert receipt.stats.total_items == 2
    assert receipt.item_ids == {}
    assert any(
        entry["code"] == "wake_closed" and late["staged_id"] in entry["stranded"]
        for entry in receipt.blockers
    )
    # staging stays active and readable; nothing new was committed
    assert len(_staged(store, WAKE)["objects"]) == 1
    with store.read_connection() as connection:
        commits = connection.execute(
            "SELECT COUNT(*) AS count FROM commit_receipts WHERE commit_id = ?",
            (f"{WAKE}:finalize",),
        ).fetchone()["count"]
        assert commits == 1

    # the documented escape hatch: a new wake publishes the same item
    moved = apply_patch(store, OTHER_WAKE, [_create_object("op-1", "Latecomer")])[0]
    assert moved["status"] == OK
    apply_patch(store, OTHER_WAKE, [_create_assertion("op-2", moved["staged_id"])])
    other = finalize_graph(store, OTHER_WAKE)
    assert other.status == "published"
    assert other.stats.objects_created == 1


# ── store hard-gate rejections (G5a F-2): structured, staging-preserving ─────


def test_finalize_store_alias_rejection_returns_commit_rejected_and_converges(
    tmp_path, monkeypatch
) -> None:
    """A Store hard-gate refusal is a structured receipt; staging stays fixable."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    alpha = apply_patch(
        store,
        WAKE,
        [_create_object("op-1", "Alpha", aliases=["Shared"], provisional=True)],
    )[0]
    assert alpha["status"] == OK
    assert inspect_working_graph(store, WAKE)["readiness"] == "ready"

    # a concurrent wake claims the alias after inspect blessed the snapshot:
    # the residual race surface the staged_alias_duplicate blocker cannot see
    real_commit = store.finalized_memory_commit

    def _sneaky_then_commit(delta, commit_id, finalizer):
        store.memory_commit(
            CognitiveDelta(
                objects=[
                    ObjectInput(
                        id="t9",
                        kind=ObjectKind.ENTITY,
                        canonical_name="Sneaky",
                        aliases=["Shared"],
                    )
                ]
            ),
            "sneaky-wake",
        )
        return real_commit(delta, commit_id, finalizer)

    monkeypatch.setattr(store, "finalized_memory_commit", _sneaky_then_commit)
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "commit_rejected"
    assert any("alias" in problem.lower() for problem in receipt.problems)

    # atomic rollback: nothing of this wake published; staging is preserved
    assert len(_staged(store, WAKE)["objects"]) == 1
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM commit_receipts WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None
        assert connection.execute(
            "SELECT COUNT(*) AS count FROM objects WHERE canonical_name = 'Alpha'"
        ).fetchone()["count"] == 0
        # the concurrent writer's commit survived untouched
        assert connection.execute(
            "SELECT COUNT(*) AS count FROM objects WHERE canonical_name = 'Sneaky'"
        ).fetchone()["count"] == 1

    # the escape hatch: withdraw the contested alias, then converge
    monkeypatch.undo()
    fixed = apply_patch(
        store,
        WAKE,
        [
            {
                "op_id": "op-2",
                "kind": "object",
                "action": "update",
                "target_ref": alpha["staged_id"],
                "payload": {"canonical_name": "Alpha", "kind": "entity", "aliases": []},
            }
        ],
    )[0]
    assert fixed["status"] == OK
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"


def test_finalize_stale_version_race_requires_explicit_repatch_before_convergence(
    tmp_path, monkeypatch
) -> None:
    """A concurrent writer landing between compile and commit is a structured rejection."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    overlay = apply_patch(
        store,
        WAKE,
        [_update_object("op-1", "t1", expected_version=1, aliases=["T1", "T1-X"])],
    )[0]
    assert overlay["status"] == OK

    real_commit = store.finalized_memory_commit

    def _bump_then_commit(delta, commit_id, finalizer):
        # a concurrent writer lands after the compile read t1's version
        with sqlite3.connect(str(store.path)) as connection:
            connection.execute("UPDATE objects SET version = version + 1 WHERE id = 't1'")
            connection.commit()
        return real_commit(delta, commit_id, finalizer)

    monkeypatch.setattr(store, "finalized_memory_commit", _bump_then_commit)
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "commit_rejected"
    assert any("stale" in problem for problem in receipt.problems)

    # the failed transaction rolled back atomically; staging is intact and
    # no finalize receipt / formal commit / audit row exists for this wake
    assert len(_staged(store, WAKE)["objects"]) == 1
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM commit_receipts WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM world_audit WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None

    # A retry must not silently compile the old overlay against v2. The
    # patch-time base guard now requires the Agent to reread and explicitly
    # withdraw/re-patch its draft before convergence.
    monkeypatch.undo()
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "blocked"
    assert [entry["code"] for entry in receipt.blockers] == ["stale_base"]

    stale_id = _staged(store, WAKE)["objects"][0]["staged_id"]
    dropped = apply_patch(store, WAKE, [_drop("op-2", "object", stale_id)])[0]
    assert dropped["status"] == OK
    fresh = apply_patch(
        store,
        WAKE,
        [_update_object("op-3", "t1", expected_version=2, aliases=["T1", "T1-X"])],
    )[0]
    assert fresh["status"] == OK

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.objects_updated == 1


def test_finalize_patch_time_stale_base_blocks_without_silent_rebase(tmp_path) -> None:
    """A formal change after patch remains visible and cannot be silently overwritten."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    patched = apply_patch(store, WAKE, [_update_object("op-1", "t1")])[0]
    assert patched["status"] == OK

    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(
                id="t1",
                kind=ObjectKind.ENTITY,
                canonical_name="Team One Concurrent",
                aliases=["T1"],
                expected_version=1,
            )]
        ),
        "concurrent-update",
    )

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "blocked"
    assert [entry["code"] for entry in receipt.blockers] == ["stale_base"]
    assert (
        receipt.blockers[0]["action_hint"]
        == "memory_read t1, then drop and re-patch this overlay"
    )
    assert len(_staged(store, WAKE)["objects"]) == 1
    with store.read_connection() as connection:
        formal = connection.execute(
            "SELECT canonical_name, version FROM objects WHERE id = 't1'"
        ).fetchone()
        assert tuple(formal) == ("Team One Concurrent", 2)
        assert connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM commit_receipts WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM world_audit WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None


def test_finalize_rejects_formal_change_after_ready_before_compile(tmp_path, monkeypatch) -> None:
    """A formal change inside the preflight->compile window must fail closed.

    The overlay records its durable patch-time base_version; preflight passed
    against v1, then a concurrent writer bumps the formal object to v2 before
    the compiler runs. The Store gate must reject the overlay against its
    durable base (v1) instead of the compiler silently re-reading v2 as the
    expected version and publishing the stale content as v3 (G5b-1 F-01).
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    patched = apply_patch(store, WAKE, [_update_object("op-1", "t1")])[0]
    assert patched["status"] == OK

    # preflight sees the patch-time base: ready with no stale blocker
    assert inspect_working_graph(store, WAKE)["readiness"] == "ready"

    real_compile = finalize_module.compile_final_delta

    def compile_after_concurrent_write(
        compile_store: WorldStore, wake_id: str, commit_id: str, **kwargs: object
    ) -> object:
        # the concurrent writer publishes formal v2 between preflight and compile
        compile_store.memory_commit(
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
        return real_compile(compile_store, wake_id, commit_id, **kwargs)

    monkeypatch.setattr(finalize_module, "compile_final_delta", compile_after_concurrent_write)
    receipt = finalize_graph(store, WAKE)

    assert receipt.status != "published"
    # the conflict is typed (commit_rejected from the Store gate, never a
    # silent overwrite) and tells the agent to reread and re-patch
    assert any("stale object version" in problem for problem in receipt.problems)
    assert any("drop" in problem for problem in receipt.problems)
    # concurrent writer's v2 survives verbatim, staging stays active
    with store.read_connection() as connection:
        formal = connection.execute(
            "SELECT canonical_name, version FROM objects WHERE id = 't1'"
        ).fetchone()
        assert tuple(formal) == ("Team One Concurrent", 2)
        assert connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM commit_receipts WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM world_audit WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None
    assert len(_staged(store, WAKE)["objects"]) == 1
    # the durable patch-time base stayed untouched by the failed finalize
    with store.read_connection() as connection:
        base = connection.execute(
            "SELECT base_version FROM staged_objects WHERE wake_id = ?", (WAKE,)
        ).fetchone()
        assert base["base_version"] == 1

    # the documented recovery: drop -> reread -> re-patch against v2
    stale_id = _staged(store, WAKE)["objects"][0]["staged_id"]
    dropped = apply_patch(store, WAKE, [_drop("op-2", "object", stale_id)])[0]
    assert dropped["status"] == OK
    fresh = apply_patch(
        store,
        WAKE,
        [_update_object("op-3", "t1", expected_version=2, aliases=["T1", "T1-X"])],
    )[0]
    assert fresh["status"] == OK
    monkeypatch.undo()
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.objects_updated == 1


def test_finalize_compile_rejects_overlay_without_durable_base(tmp_path) -> None:
    """A legacy overlay without a patch-time base_version fails closed at compile.

    Defense in depth behind the preflight base_version_unknown blocker: the
    compiler never falls back to the current formal version for a row that
    has no trustworthy patch-time base (G5b-1 F-01).
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    now = "2026-08-20T00:00:00+00:00"
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO staged_objects (staged_id, wake_id, status, target_ref, kind,"
            " canonical_name, type_key, domain_hints_json, provisional, identity_basis_json,"
            " event_time_start, event_time_end, aliases_json, created_at, updated_at,"
            " version, base_version)"
            " VALUES (?, ?, 'active', 't1', 'entity', 'Team One Updated', NULL, '[]', 0,"
            " '[]', NULL, NULL, '[\"T1\"]', ?, ?, 1, NULL)",
            (f"{WAKE}:s1", WAKE, now, now),
        )
    with pytest.raises(FinalizeCompileError) as error:
        finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert any("base version" in problem for problem in error.value.problems)


def test_same_wake_near_concurrent_finalize_converges_on_one_receipt(tmp_path) -> None:
    """Two connections reaching finalize together converge without split publication."""
    path = tmp_path / "world.sqlite3"
    store = WorldStore(path)
    _seed(store)
    created = apply_patch(
        store,
        WAKE,
        [_create_object("op-1", "Concurrent Draft", provisional=True)],
    )[0]
    assert created["status"] == OK
    barrier = threading.Barrier(2)

    def publish() -> object:
        barrier.wait(timeout=5)
        return finalize_graph(WorldStore(path), WAKE)

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = [future.result(timeout=10) for future in [executor.submit(publish) for _ in range(2)]]

    assert {receipt.status for receipt in receipts} <= {"published", "already_published"}
    assert "published" in {receipt.status for receipt in receipts}
    assert {receipt.commit_id for receipt in receipts} == {f"{WAKE}:finalize"}
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM world_audit WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = ? AND status = 'active'",
            (WAKE,),
        ).fetchone()[0] == 0


def test_graph_shell_patch_waits_across_finalize_compile_commit_window(
    tmp_path, monkeypatch
) -> None:
    """The local Graph Shell lock covers compile through the receipt transaction."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="thread-finalize",
        wake_id=WAKE,
        closed_wake_guard=True,
    )
    early = tools._graph_patch(  # noqa: SLF001 - deterministic facade race proof
        {"items": [_create_object("early-op", "Early", provisional=True)]}
    )["payload"]["results"][0]
    assert early["status"] == OK
    compiled = threading.Event()
    release_commit = threading.Event()
    late_started = threading.Event()
    real_compile = finalize_module.compile_final_delta

    def paused_compile(*args, **kwargs):
        result = real_compile(*args, **kwargs)
        compiled.set()
        assert release_commit.wait(timeout=5)
        return result

    def late_patch() -> dict:
        late_started.set()
        return tools._graph_patch(  # noqa: SLF001 - deterministic facade race proof
            {"items": [_create_object("late-op", "Late", provisional=True)]}
        )

    monkeypatch.setattr(finalize_module, "compile_final_delta", paused_compile)
    with ThreadPoolExecutor(max_workers=2) as executor:
        publish_future = executor.submit(finalize_graph, store, WAKE)
        assert compiled.wait(timeout=5)
        late_future = executor.submit(late_patch)
        assert late_started.wait(timeout=5)
        assert not late_future.done()
        release_commit.set()
        published = publish_future.result(timeout=10)
        late = late_future.result(timeout=10)["payload"]["results"][0]

    assert published.status == "published"
    assert published.stats.total_items == 1
    assert late["status"] == REJECTED
    assert late["error_code"] == "wake_closed"
    assert read_active_staged(store, WAKE)["objects"] == []
    with store.read_connection() as connection:
        assert [
            row["canonical_name"]
            for row in connection.execute(
                "SELECT canonical_name FROM objects WHERE canonical_name IN ('Early', 'Late')"
            ).fetchall()
        ] == ["Early"]
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = ?", (WAKE,)
        ).fetchone()[0] == 1


# ── readiness: blocked keeps staging for the next patch round ─────────────────


def test_finalize_blocked_keeps_staging_and_preserves_for_fix(tmp_path) -> None:
    """A readiness blocker stops the publish; patching the gap converges."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "Loner")])[0]
    assert created["status"] == OK

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "blocked"
    assert any(entry["code"] == "zero_connection_object" for entry in receipt.blockers)
    assert receipt.stats.total_items == 1

    # staging is preserved and no receipt or formal commit exists
    assert _staged(store, WAKE)["objects"]
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM commit_receipts WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None

    # the same snapshot converges after the gap is closed
    linked = apply_patch(
        store, WAKE, [_create_assertion("op-2", created["staged_id"])]
    )[0]
    assert linked["status"] == OK
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.total_items == 2


# ── empty and all-abandoned semantics ────────────────────────────────────────


def test_finalize_empty_work_graph_publishes_honestly(tmp_path) -> None:
    """A wake with no staging still gets a durable, clearly-stated empty publish."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.total_items == 0
    assert not receipt.all_work_abandoned
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM commit_receipts WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone()
        assert row is not None


def test_finalize_all_abandoned_publishes_empty_with_flag(tmp_path) -> None:
    """Dropped work publishes an empty formal commit flagged as abandoned."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "Ghost")])[0]
    assert created["status"] == OK
    dropped = apply_patch(store, WAKE, [_drop("op-2", "object", created["staged_id"])])[0]
    assert dropped["status"] == OK

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.abandoned == 1
    assert receipt.stats.total_items == 0
    assert receipt.all_work_abandoned
    with store.read_connection() as connection:
        # nothing of the abandoned item reached the formal tables
        assert connection.execute(
            "SELECT 1 FROM objects WHERE id = ?", (created["staged_id"],)
        ).fetchone() is None


# ── idempotency and the §6.11 crash points ───────────────────────────────────


def test_finalize_idempotent_repeat_returns_original_receipt(tmp_path) -> None:
    """Re-finalize (crash point 5: result lost) replays the durable receipt."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "Settled")])[0]
    assert created["status"] == OK
    apply_patch(store, WAKE, [_create_assertion("op-2", created["staged_id"])])

    first = finalize_graph(store, WAKE)
    assert first.status == "published"

    # the agent's tool result was lost; the resume path re-finalizes
    second = finalize_graph(store, WAKE)
    assert second.status == "already_published"
    assert second.replayed is True
    assert second.commit_id == first.commit_id
    assert second.committed_at == first.committed_at
    assert second.stats == first.stats

    with store.read_connection() as connection:
        commits = connection.execute(
            "SELECT COUNT(*) AS count FROM commit_receipts WHERE commit_id = ?",
            (f"{WAKE}:finalize",),
        ).fetchone()["count"]
        receipts = connection.execute(
            "SELECT COUNT(*) AS count FROM finalize_receipts WHERE wake_id = ?",
            (WAKE,),
        ).fetchone()["count"]
    assert commits == 1  # a second commit can never follow a completed finalize
    assert receipts == 1


def test_finalize_crash_inside_transaction_rolls_back_and_converges(tmp_path, monkeypatch) -> None:
    """Crash points 3/4: a failure inside the shared transaction rolls back atomically.

    The formal delta, the staging finalization, and the receipt insert are one
    SQLite transaction (D-007 preferred route): after the simulated crash there
    is no partial state — no formal rows, no receipt, staging still active —
    and the retry converges to exactly one commit and one receipt.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    created = apply_patch(store, WAKE, [_create_object("op-1", "Survivor")])[0]
    assert created["status"] == OK
    apply_patch(store, WAKE, [_create_assertion("op-2", created["staged_id"])])

    def _crash(connection, wake_id, now):
        raise RuntimeError("simulated crash inside the finalize transaction")

    monkeypatch.setattr(finalize_module, "_finalize_staging_rows", _crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        finalize_graph(store, WAKE)

    # atomic rollback: nothing was published, staging is untouched
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM commit_receipts WHERE commit_id = ?", (f"{WAKE}:finalize",)
        ).fetchone() is None
    assert _staged(store, WAKE)["objects"]

    # resume converges: one commit, one receipt, one published object
    monkeypatch.undo()
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    with store.read_connection() as connection:
        commits = connection.execute(
            "SELECT COUNT(*) AS count FROM commit_receipts WHERE commit_id = ?",
            (f"{WAKE}:finalize",),
        ).fetchone()["count"]
        created = connection.execute(
            "SELECT COUNT(*) AS count FROM objects WHERE canonical_name = 'Survivor'"
        ).fetchone()["count"]
    assert commits == 1
    assert created == 1


def test_finalize_patch_replay_reuses_ledger_and_single_staged_row(tmp_path) -> None:
    """Crash points 1/2/6: the patch ledger replays an op with one staged row."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    item = _create_object("op-1", "Ledgered")
    first = apply_patch(store, WAKE, [item])[0]
    assert first["status"] == OK
    apply_patch(store, WAKE, [_create_assertion("op-2", first["staged_id"])])
    # crash points 1/2: the agent re-issues the same op on resume
    replay = apply_patch(store, WAKE, [item])[0]
    assert replay["status"] == OK
    assert replay["staged_id"] == first["staged_id"]

    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT COUNT(*) AS count FROM staged_objects WHERE staged_id = ?",
            (first["staged_id"],),
        ).fetchone()["count"]
        receipts = connection.execute(
            "SELECT COUNT(*) AS count FROM staged_patch_receipts WHERE wake_id = ? AND op_id = ?",
            (WAKE, "op-1"),
        ).fetchone()["count"]
    assert rows == 1
    assert receipts == 1

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.objects_created == 1
    assert receipt.item_ids[first["staged_id"]] == first["staged_id"]


# ── cross-wake continuation: formal results are readable by later wakes ──────


def test_finalize_remaps_refs_and_next_wake_reads_formal(tmp_path) -> None:
    """A later wake resolves refs against the published formal rows."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    created = apply_patch(store, WAKE, [_create_object("op-1", "Evergreen")])[0]
    assert created["status"] == OK
    apply_patch(store, WAKE, [_create_assertion("op-2", created["staged_id"])])
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"

    # the next wake's patch resolves the published formal id directly
    continued = apply_patch(
        store,
        OTHER_WAKE,
        [_create_assertion("op-1", created["staged_id"], predicate="praised")],
    )[0]
    assert continued["status"] == OK
    other = finalize_graph(store, OTHER_WAKE)
    assert other.status == "published"
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT subject_id, predicate FROM assertions WHERE predicate = 'praised'"
        ).fetchone()
        assert row["subject_id"] == created["staged_id"]


# ── failure paths: compile problems and id collisions ────────────────────────


def test_finalize_compile_failure_preserves_staging_and_reports_problems(tmp_path) -> None:
    """An uncompilable item stops the publish with problems and keeps staging."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    # Bad instants are rejected at patch time (P1-4), so the compile-only
    # trigger is now the inverted interval: each instant is well-formed, so
    # staging tolerates the pair and only the compile-time ordering check
    # (the last ring that still can fail a wake) rejects it.
    bad = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-1",
                "t1",
                object_ref=None,
                literal="compile-fallback",
                event_time_start="2026-08-22T00:00:00Z",
                event_time_end="2026-08-23T00:00:00Z",
            )
        ],
    )[0]
    assert bad["status"] == OK
    # Patch-time validation now rejects an inverted interval. Simulate a
    # historical/corrupt staged row to keep the compiler's final defensive
    # ring covered without weakening the public write contract.
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE staged_assertions SET event_time_end = ? WHERE wake_id = ?",
            ("2026-08-21T00:00:00Z", WAKE),
        )

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "compile_failed"
    assert any(
        "time interval start must be before or equal to end" in problem
        for problem in receipt.problems
    )
    assert receipt.stats.total_items == 1
    # staging preserved for the agent to fix
    assert _staged(store, WAKE)["assertions"]
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (WAKE,)
        ).fetchone() is None

    # the escape hatch: drop and recreate the item, then converge
    dropped = apply_patch(store, WAKE, [_drop("op-2", "assertion", bad["staged_id"])])[0]
    assert dropped["status"] == OK
    fixed = apply_patch(
        store, WAKE, [_create_assertion("op-3", "t1", predicate="ranked")]
    )[0]
    assert fixed["status"] == OK
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.assertions_created == 1


def test_finalize_id_collision_falls_back_to_durable_id(tmp_path) -> None:
    """A staged id already occupied by a legacy row publishes under a durable id."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                *FORMAL_OBJECTS,
                ObjectInput(
                    id=f"{WAKE}:s1", kind=ObjectKind.ENTITY, canonical_name="Pre-Sown"
                ),
            ],
            observations=[FORMAL_OBSERVATION],
        ),
        "seed",
    )

    created = apply_patch(store, WAKE, [_create_object("op-1", "Fresh Creation")])[0]
    assert created["status"] == OK
    assert created["staged_id"] == f"{WAKE}:s1"  # host id collides with the legacy row
    apply_patch(store, WAKE, [_create_assertion("op-2", created["staged_id"])])

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    formal_id = receipt.item_ids[f"{WAKE}:s1"]
    assert formal_id != f"{WAKE}:s1"
    assert formal_id.startswith("object-")

    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT id, canonical_name FROM objects WHERE id IN (?, ?)",
            (f"{WAKE}:s1", formal_id),
        ).fetchall()
    by_id = {row["id"]: row["canonical_name"] for row in rows}
    assert by_id[f"{WAKE}:s1"] == "Pre-Sown"  # the legacy row is untouched
    assert by_id[formal_id] == "Fresh Creation"  # the new object lives on


def test_finalize_compile_error_surfaces_problems_directly(tmp_path) -> None:
    """compile_final_delta raises FinalizeCompileError with the problems."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-1",
                "t1",
                object_ref=None,
                literal="compile-fallback",
                event_time_start="2026-08-22T00:00:00Z",
                event_time_end="2026-08-23T00:00:00Z",
            )
        ],
    )
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE staged_assertions SET event_time_end = ? WHERE wake_id = ?",
            ("2026-08-21T00:00:00Z", WAKE),
        )
    with pytest.raises(FinalizeCompileError) as error:
        finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert any(
        "time interval start must be before or equal to end" in problem
        for problem in error.value.problems
    )


# ── Wave B: staged answer refs, deepen links and resolve compile (v18) ───────


def _resolve_op(op_id: str, target_ref: str, **overrides: object) -> dict[str, object]:
    return {
        "op_id": op_id,
        "kind": "inquiry",
        "action": "resolve",
        "target_ref": target_ref,
        "payload": {"expected_version": 1, **overrides},
    }


def test_finalize_same_wake_create_answer_resolve_publishes(tmp_path) -> None:
    """B4-4 end to end: one wake creates an inquiry, an answering assertion
    declaring it, and a resolve naming that assertion — finalize publishes the
    inquiry as resolved with the answer co-published in the same delta."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    inq = apply_patch(store, WAKE, [_create_inquiry("op-1", "t1")])[0]
    assert inq["status"] == OK
    answer = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-2", "t1", predicate="explains", answers_ref=inq["staged_id"]
            )
        ],
    )[0]
    assert answer["status"] == OK
    resolved = apply_patch(
        store, WAKE, [_resolve_op("op-3", inq["staged_id"], answers_ref=answer["staged_id"])]
    )[0]
    assert resolved["status"] == OK

    receipt = finalize_graph(store, WAKE)
    formal_inquiry_id = inq["staged_id"]
    assert receipt.commit_id == f"{WAKE}:finalize"
    assert receipt.item_ids[resolved["staged_id"]] == formal_inquiry_id
    with sqlite3.connect(str(tmp_path / "world.sqlite3")) as connection:
        inquiry_row = connection.execute(
            "SELECT status, version FROM inquiries WHERE id = ?", (formal_inquiry_id,)
        ).fetchone()
        assertion_row = connection.execute(
            "SELECT answers_inquiry_id FROM assertions WHERE id = ?", (answer["staged_id"],)
        ).fetchone()
        # the store receipt records the resolution itself
        commit_receipt = connection.execute(
            "SELECT receipt_json FROM commit_receipts WHERE commit_id = ?",
            (receipt.commit_id,),
        ).fetchone()[0]
    assert inquiry_row == ("resolved", 2)
    assert assertion_row == (formal_inquiry_id,)
    assert '"resolved_inquiry_ids":["' + formal_inquiry_id + '"]' in commit_receipt
    assert _staged(store, WAKE) == {"objects": [], "assertions": [], "inquiries": []}


def test_finalize_inquiries_created_counts_only_creates_not_resolutions(tmp_path) -> None:
    """M-EXT-01: resolution staging rows carry kind='resolution'; they must
    not inflate FinalizeStats.inquiries_created. One create + one resolve
    stages two rows but reports one created."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)

    inq = apply_patch(store, WAKE, [_create_inquiry("op-1", "t1")])[0]
    answer = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-2", "t1", predicate="explains", answers_ref=inq["staged_id"]
            )
        ],
    )[0]
    apply_patch(
        store, WAKE, [_resolve_op("op-3", inq["staged_id"], answers_ref=answer["staged_id"])]
    )[0]

    staged = _staged(store, WAKE)
    # two staged inquiry rows: one create, one resolution
    assert len(staged["inquiries"]) == 2
    assert sum(1 for row in staged["inquiries"] if row.get("kind") == "resolution") == 1

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert receipt.stats.inquiries_created == 1


def test_inquiry_relation_roundtrip_formal_and_staged_targets(tmp_path) -> None:
    """Inquiry relationship refs survive patch, review, finalize, and SQL.

    This exercises both formal and staged targets for ``deepens_ref`` and
    ``answers_ref`` in one publishable delta, including a staged resolution.
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

    first = apply_patch(
        store,
        WAKE,
        [_create_inquiry("op-1", "t1", prompt="Why?", deepens_ref="inq-formal")],
    )[0]
    second = apply_patch(
        store,
        WAKE,
        [
            _create_inquiry(
                "op-2",
                "t1",
                prompt="Why exactly?",
                deepens_ref=first["staged_id"],
            )
        ],
    )[0]
    formal_answer = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-3",
                "t1",
                predicate="partially_explains",
                answers_ref="inq-formal",
            )
        ],
    )[0]
    staged_answer = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-4",
                "t1",
                predicate="explains",
                answers_ref=second["staged_id"],
            )
        ],
    )[0]
    resolution = apply_patch(
        store,
        WAKE,
        [
            _resolve_op(
                "op-5",
                second["staged_id"],
                answers_ref=staged_answer["staged_id"],
            )
        ],
    )[0]
    assert {
        first["status"],
        second["status"],
        formal_answer["status"],
        staged_answer["status"],
        resolution["status"],
    } == {OK}

    receipt = finalize_graph(store, WAKE)
    assert receipt.item_ids[resolution["staged_id"]] == second["staged_id"]
    with sqlite3.connect(str(tmp_path / "world.sqlite3")) as connection:
        inquiries = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                "SELECT id, deepens_id, status FROM inquiries "
                "WHERE id IN (?, ?) ORDER BY id",
                (first["staged_id"], second["staged_id"]),
            ).fetchall()
        }
        answers = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT id, answers_inquiry_id FROM assertions WHERE id IN (?, ?)",
                (formal_answer["staged_id"], staged_answer["staged_id"]),
            ).fetchall()
        }
    assert inquiries == {
        first["staged_id"]: ("inq-formal", "open"),
        second["staged_id"]: (first["staged_id"], "resolved"),
    }
    assert answers == {
        formal_answer["staged_id"]: "inq-formal",
        staged_answer["staged_id"]: second["staged_id"],
    }


def test_finalize_deepen_maps_formal_deepens_id_across_string_order(tmp_path) -> None:
    """The staged->formal inquiry map must be complete before deepens refs are
    resolved: a deepener whose staged_id sorts BEFORE its target (i12 < i2)
    must still compile its deepens_id onto the right formal row."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    target: dict[str, object] | None = None
    for index in range(1, 12):
        result = apply_patch(
            store, WAKE, [_create_inquiry(f"op-{index}", "t1", prompt=f"p{index}")]
        )[0]
        assert result["status"] == OK
        if index == 2:
            target = result
    assert target is not None
    deepener = apply_patch(
        store,
        WAKE,
        [
            _create_inquiry(
                "op-12", "t1", prompt="deeper", deepens_ref=target["staged_id"]
            )
        ],
    )[0]
    assert deepener["status"] == OK
    # string order: wake-finalize:i12 < wake-finalize:i2, so the two-pass map
    # is exercised only if the compile resolves refs after building it
    assert str(deepener["staged_id"]) < str(target["staged_id"])

    receipt = finalize_graph(store, WAKE)
    assert receipt.commit_id == f"{WAKE}:finalize"
    with sqlite3.connect(str(tmp_path / "world.sqlite3")) as connection:
        deepens_id = connection.execute(
            "SELECT deepens_id FROM inquiries WHERE id = ?", (deepener["staged_id"],)
        ).fetchone()[0]
    assert deepens_id == target["staged_id"]


def test_finalize_resolve_without_staged_answer_fails_closed(tmp_path) -> None:
    """B4-5: a staged resolve may name a FORMAL answering assertion (staging
    accepts the reference), but finalize fails the delta because the answer
    must be co-published as a staged assertion — nothing is consumed."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(id="inq-f", subject_id="t1", prompt="Why?", rationale="gap")
            ],
            assertions=[
                AssertionInput(
                    id="fa-1",
                    subject_id="t1",
                    predicate="explains",
                    object_id="s1",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    answers_inquiry_id="inq-f",
                )
            ],
        ),
        "seed-answer",
    )
    resolved = apply_patch(
        store, WAKE, [_resolve_op("op-1", "inq-f", answers_ref="fa-1")]
    )[0]
    assert resolved["status"] == OK

    receipt = finalize_graph(store, WAKE)
    # D-013 single surface: the preflight resolve gate blocks first with the
    # typed blocker; the compile gate stays as the second line of defense
    assert receipt.status == "blocked"
    assert any(entry["code"] == "no_answering_assertion" for entry in receipt.blockers)
    with pytest.raises(FinalizeCompileError) as error:
        finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert any(
        "no non-uncertainty answering assertion" in problem
        for problem in error.value.problems
    )
    # the resolve row is untouched — the wake can drop it and recover
    staged = _staged(store, WAKE)
    assert len(staged["inquiries"]) == 1
    assert staged["inquiries"][0]["kind"] == "resolution"


def test_finalize_resolve_gates_already_resolved_stale_duplicate(tmp_path) -> None:
    """Compile gates fail closed: an already-resolved inquiry, a version moved
    between staging and finalize, and a second resolve of the same target."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed(store)
    # an already-resolved formal inquiry (resolved at version 2)
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(id="inq-done", subject_id="t1", prompt="Done?", rationale="gap")
            ],
            assertions=[
                AssertionInput(
                    id="fa-done",
                    subject_id="t1",
                    predicate="explains",
                    object_id="s1",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    answers_inquiry_id="inq-done",
                )
            ],
            resolve_inquiries=[InquiryResolution(id="inq-done", expected_version=1)],
        ),
        "seed-resolved",
    )
    resolved = apply_patch(
        store, WAKE, [_resolve_op("op-1", "inq-done", expected_version=2, answers_ref="fa-done")]
    )[0]
    assert resolved["status"] == OK
    receipt = finalize_graph(store, WAKE)
    # D-013 single surface: the preflight gate blocks first, the compile gate
    # stays the second line of defense
    assert receipt.status == "blocked"
    assert any(entry["code"] == "inquiry_already_resolved" for entry in receipt.blockers)
    with pytest.raises(FinalizeCompileError) as error:
        finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert any("already resolved" in problem for problem in error.value.problems)
    # the wake may drop the resolve and try again
    apply_patch(
        store,
        WAKE,
        [{"op_id": "op-2", "kind": "inquiry", "action": "drop", "target_ref": resolved["staged_id"]}],
    )

    # a formal inquiry whose version moves between resolve and finalize:
    # staging froze expected_version at the version it saw (3); the compile
    # compares against the formal row's CURRENT version (4)
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[InquiryInput(id="inq-move", subject_id="t1", prompt="Move?", rationale="gap")],
        ),
        "seed-move",
    )
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-move", subject_id="t1", prompt="Move?", rationale="gap", expected_version=1
                )
            ],
        ),
        "bump-move",
    )
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            assertions=[
                AssertionInput(
                    id="fa-move",
                    subject_id="t1",
                    # distinct signature: fa-done already claims explains/t1/s1
                    predicate="expresses",
                    object_id="s1",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    answers_inquiry_id="inq-move",
                )
            ],
        ),
        "answer-move",
    )
    resolved = apply_patch(
        store, WAKE, [_resolve_op("op-3", "inq-move", expected_version=2, answers_ref="fa-move")]
    )[0]
    assert resolved["status"] == OK
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-move", subject_id="t1", prompt="Move?", rationale="gap", expected_version=2
                )
            ],
        ),
        "bump-move-2",
    )
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "blocked"
    assert any(entry["code"] == "stale_expected_version" for entry in receipt.blockers)
    with pytest.raises(FinalizeCompileError) as error:
        finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert any("stale expected_version" in problem for problem in error.value.problems)
    apply_patch(
        store,
        WAKE,
        [{"op_id": "op-4", "kind": "inquiry", "action": "drop", "target_ref": resolved["staged_id"]}],
    )

    # two resolves of the same target in one wake, each grounded in a staged
    # answering assertion: the second fails closed at compile
    answer = apply_patch(
        store,
        WAKE,
        [
            _create_assertion(
                "op-5", "t1", predicate="participated_in", answers_ref="inq-move"
            )
        ],
    )[0]
    assert answer["status"] == OK
    first = apply_patch(
        store,
        WAKE,
        [_resolve_op("op-6", "inq-move", expected_version=3, answers_ref=answer["staged_id"])],
    )[0]
    assert first["status"] == OK
    second = apply_patch(
        store,
        WAKE,
        [_resolve_op("op-7", "inq-move", expected_version=3, answers_ref=answer["staged_id"])],
    )[0]
    assert second["status"] == OK
    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "blocked"
    assert any(entry["code"] == "duplicate_resolve" for entry in receipt.blockers)
    with pytest.raises(FinalizeCompileError) as error:
        finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert any("already has a resolve" in problem for problem in error.value.problems)


# ── T2-2: similar-open-inquiry report (observability, not omission) ───────────


def test_finalize_similar_open_inquiry_reports_without_omitting(tmp_path) -> None:
    """A staged inquiry that repeats an open/dormant line publishes anyway and
    reports the collision on the F-B warnings channel (T2-2)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-open",
                    subject_id="t1",
                    prompt="who is the champion?",
                    rationale="first line",
                )
            ],
        ),
        "seed-inquiry",
    )
    applied = apply_patch(
        store,
        WAKE,
        [_create_inquiry("op-1", "t1", prompt="who is the champion of the summer cup?")],
    )
    assert applied[0]["status"] == OK

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    similar = [w for w in receipt.warnings if w.get("code") == "similar_open_inquiry"]
    assert len(similar) == 1
    assert "inq-open" in similar[0]["message"]
    assert receipt.problems == []
    # reported, never omitted: the near-repeat (bigram jaccard 0.56, not the
    # exact (subject_id, prompt) pair the open-inquiry dedup index blocks)
    # publishes as a second open line
    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT id, status FROM inquiries WHERE subject_id = 't1'"
        ).fetchall()
    assert len(rows) == 2
    assert all(row["status"] == "open" for row in rows)


def test_finalize_similar_inquiry_intra_delta_reports_subject_scoped(tmp_path) -> None:
    """Intra-delta similarity reports the second sibling, skips the first, and
    never compares across different subjects."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=FORMAL_OBJECTS), "seed-objects")
    applied = apply_patch(
        store,
        WAKE,
        [
            _create_inquiry("op-1", "t1", prompt="who is the champion?"),
            _create_inquiry("op-2", "t1", prompt="who is the champion of the summer cup?"),
            _create_inquiry("op-3", "s1", prompt="who is the champion?"),
        ],
    )
    assert [entry["status"] for entry in applied] == [OK, OK, OK]

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    similar = [w for w in receipt.warnings if w.get("code") == "similar_open_inquiry"]
    # op-2 is a near-repeat of op-1 (same subject, jaccard 0.56); op-1 has no
    # earlier sibling and the s1 copy shares no subject — exactly one report
    assert len(similar) == 1
    assert similar[0]["ref"] == applied[1]["staged_id"]
    # none of the three is the exact (subject_id, prompt) pair the open-
    # inquiry dedup index blocks, so all three publish
    with store.read_connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM inquiries WHERE status = 'open'"
        ).fetchone()["count"]
    assert count == 3


def test_finalize_similar_inquiry_declared_deepens_link_is_exempt(tmp_path) -> None:
    """A declared, resolvable deepens link is the model's intent — matching
    the parent by construction — and is exempt from the report (E2's rule)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-open",
                    subject_id="t1",
                    prompt="why did the team change?",
                    rationale="formal gap",
                )
            ],
        ),
        "seed-inquiry",
    )
    applied = apply_patch(
        store,
        WAKE,
        [
            _create_inquiry(
                "op-1",
                "t1",
                prompt="why did the team change in the end?",
                deepens_ref="inq-open",
            )
        ],
    )
    assert applied[0]["status"] == OK

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    assert [
        w for w in receipt.warnings if w.get("code") == "similar_open_inquiry"
    ] == []
    with store.read_connection() as connection:
        deepens_id = connection.execute(
            "SELECT deepens_id FROM inquiries WHERE id = ?",
            (applied[0]["staged_id"],),
        ).fetchone()
    assert deepens_id["deepens_id"] == "inq-open"


def test_finalize_similar_inquiry_strict_compile_is_silent(tmp_path) -> None:
    """Without the tolerance channel (warnings=None) the report is silent and
    the compile never fails on a repeated line."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-open",
                    subject_id="t1",
                    prompt="who is the champion?",
                    rationale="first line",
                )
            ],
        ),
        "seed-inquiry",
    )
    applied = apply_patch(
        store,
        WAKE,
        [_create_inquiry("op-1", "t1", prompt="who is the champion?")],
    )
    assert applied[0]["status"] == OK

    delta, _ = finalize_module.compile_final_delta(store, WAKE, f"{WAKE}:finalize")
    assert [inquiry.id for inquiry in delta.inquiries] == [applied[0]["staged_id"]]


# ── T2-2 end-to-end: the three exact-duplicate layers really engage ──────────
# The formal chain's exact-repeat defense is three stacked layers: preflight
# reports `inquiry_duplicate` (subject_id + prompt, resolved_at IS NULL), the
# open-inquiry dedup index swallows the INSERT OR IGNORE (subject_id + prompt,
# status = 'open' only), and the compile-level similar report fires on any
# open/dormant repeat (jaccard >= 0.20, exact repeats included).  These tests
# verify the layers actually engage on real staging → finalize flows, and
# where the index's open-only scope leaves the dormant blind spot that the
# report exists to cover.


def test_finalize_exact_duplicate_of_open_inquiry_is_swallowed_end_to_end(
    tmp_path,
) -> None:
    """An exact (subject_id, prompt) repeat of an OPEN line is warned by
    preflight, reported by compile, and swallowed by the dedup index — the
    store never holds two open copies."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-open",
                    subject_id="t1",
                    prompt="who is the champion?",
                    rationale="first line",
                )
            ],
        ),
        "seed-inquiry",
    )
    applied = apply_patch(
        store,
        WAKE,
        [_create_inquiry("op-1", "t1", prompt="who is the champion?")],
    )
    assert applied[0]["status"] == OK

    report = inspect_working_graph(store, WAKE)
    duplicate = [
        entry
        for entry in report["warnings"]
        if entry["code"] == "inquiry_duplicate"
    ]
    assert len(duplicate) == 1
    assert duplicate[0]["ref"] == applied[0]["staged_id"]
    assert "on t1" in duplicate[0]["message"]

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    # the compile still reports the repeat (jaccard 1.0 counts) even though
    # the store swallows the row — observability is layered, never silent
    similar = [w for w in receipt.warnings if w.get("code") == "similar_open_inquiry"]
    assert len(similar) == 1
    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT id, status FROM inquiries WHERE subject_id = 't1'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["id"] == "inq-open"
    assert rows[0]["status"] == "open"


def test_finalize_exact_duplicate_within_delta_collapses_to_one_row(
    tmp_path,
) -> None:
    """Two identical staged inquiries in one wake are warned intra-delta by
    preflight and collapse onto one open row at commit time."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=FORMAL_OBJECTS), "seed-objects")
    applied = apply_patch(
        store,
        WAKE,
        [
            _create_inquiry("op-1", "t1", prompt="who is the champion?"),
            _create_inquiry("op-2", "t1", prompt="who is the champion?"),
        ],
    )
    assert [entry["status"] for entry in applied] == [OK, OK]

    report = inspect_working_graph(store, WAKE)
    duplicate = [
        entry
        for entry in report["warnings"]
        if entry["code"] == "inquiry_duplicate"
    ]
    # the intra-delta check is symmetric: each identical sibling flags the
    # other, so both staged rows are warned
    assert len(duplicate) == 2
    assert {entry["ref"] for entry in duplicate} == {
        applied[0]["staged_id"],
        applied[1]["staged_id"],
    }

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    # the compile maps both staged rows (it has no prompt-level dedup); the
    # store index is the layer that swallows the second row at write time
    assert receipt.item_ids[applied[1]["staged_id"]] == applied[1]["staged_id"]
    with store.read_connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM inquiries WHERE status = 'open'"
        ).fetchone()["count"]
    assert count == 1


def test_finalize_exact_duplicate_of_dormant_line_publishes_with_report(
    tmp_path,
) -> None:
    """The dedup index only guards status = 'open' rows, so an exact repeat of
    a DORMANT line publishes as a new open row — the exact gap the compile
    report covers (observability, never interception)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=FORMAL_OBJECTS,
            inquiries=[
                InquiryInput(
                    id="inq-dormant",
                    subject_id="t1",
                    prompt="who is the champion?",
                    rationale="first line",
                )
            ],
        ),
        "seed-inquiry",
    )
    store.demote_stale_inquiries(datetime.now(UTC), max_age_days=0)
    with store.read_connection() as connection:
        status = connection.execute(
            "SELECT status FROM inquiries WHERE id = 'inq-dormant'"
        ).fetchone()
    assert status["status"] == "dormant"

    applied = apply_patch(
        store,
        WAKE,
        [_create_inquiry("op-1", "t1", prompt="who is the champion?")],
    )
    assert applied[0]["status"] == OK

    receipt = finalize_graph(store, WAKE)
    assert receipt.status == "published"
    similar = [w for w in receipt.warnings if w.get("code") == "similar_open_inquiry"]
    assert len(similar) == 1
    assert "inq-dormant" in similar[0]["message"]
    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT id, status FROM inquiries WHERE subject_id = 't1' ORDER BY id"
        ).fetchall()
    assert [(row["id"], row["status"]) for row in rows] == [
        ("inq-dormant", "dormant"),
        (applied[0]["staged_id"], "open"),
    ]
