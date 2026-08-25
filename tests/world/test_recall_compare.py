# Slice 2c: memory_compare — two ids side by side, shared/differing markers,
# never a verdict (design §4.4, plan §7.3). Recall-layer tests: object
# identity fields, assertion signature fields, missing/mixed handling, caps,
# and the no-adjudication guarantee.

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    ObjectInput,
    ObjectKind,
    WorldStore,
)
from leave_information_bubble.world.recall import WorldRecall
from tests.world._recall_helpers import (
    _assertion,
    _event_object,
    _object,
    _observation,
    _participant_edge,
)

VIEW_CAP = 15


def _name_usage(identifier: str, subject_id: str, literal: str) -> AssertionInput:
    return AssertionInput(
        id=identifier,
        subject_id=subject_id,
        predicate="name_usage",
        literal=literal,
        epistemic_role=EpistemicRole.FACT,
        confidence=0.8,
        evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
    )


def _compare_fields(bundle: Any) -> dict[str, Any]:
    assert bundle.compare is not None
    return {entry["field"]: entry for entry in bundle.compare["fields"]}


def test_compare_object_identity_fields_side_by_side(tmp_path: pytest.TempPathFactory) -> None:
    """The two portraits line up per field with shared/only splits and equal flags."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                # identity aliases are globally unique, so candidate overlap
                # surfaces through legacy forms, never a shared active alias
                _object("t1", aliases=["Team One", "T1 Official"]),
                _object("t2", aliases=["Team 1"]),
            ]
        ),
        "seed",
    )
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO object_aliases (normalized_alias, object_id) VALUES ('t1 official', 't2')"
        )

    bundle = WorldRecall(store).compare("t1", "t2")

    assert bundle.compare["mode"] == "object"
    assert bundle.compare["left"] == {"id": "t1", "status": "ok", "canonical_name": "T1"}
    assert bundle.compare["right"] == {"id": "t2", "status": "ok", "canonical_name": "T2"}
    assert "object identity compare" in bundle.reasons
    fields = _compare_fields(bundle)
    assert fields["canonical_name"] == {
        "field": "canonical_name",
        "left": "T1",
        "right": "T2",
        "equal": False,
    }
    assert fields["kind"] == {"field": "kind", "left": "entity", "right": "entity", "equal": True}
    assert fields["active_aliases"]["shared"] == []
    assert fields["active_aliases"]["left_only"] == ["t1 official", "team one"]
    assert fields["active_aliases"]["right_only"] == ["team 1"]
    assert fields["active_aliases"]["equal"] is False
    assert fields["active_aliases"]["truncated"] is False
    # t2's legacy form matches t1's active identity alias: a shared name
    # surface that only one side ever claimed as identity
    assert fields["legacy_aliases"]["shared"] == []
    assert fields["legacy_aliases"]["left_only"] == []
    assert fields["legacy_aliases"]["right_only"] == ["t1 official"]
    assert fields["legacy_aliases"]["equal"] is False


def test_compare_object_identical_all_equal(tmp_path: pytest.TempPathFactory) -> None:
    """An object compared with itself marks every field equal with empty onlys."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(objects=[_object("t1", aliases=["Team One"])]),
        "seed",
    )

    fields = _compare_fields(WorldRecall(store).compare("t1", "t1"))

    assert all(entry["equal"] for entry in fields.values())
    for entry in fields.values():
        if "left_only" in entry:
            assert entry["left_only"] == []
            assert entry["right_only"] == []


def test_compare_includes_event_span_and_type_key(tmp_path: pytest.TempPathFactory) -> None:
    """Event objects compare their type_key and span; a span-less object reports None."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="match-1",
                    kind=ObjectKind.EVENT,
                    canonical_name="Match One",
                    type_key="tournament",
                    event_time_start=datetime(2026, 8, 1, 9, tzinfo=UTC),
                    event_time_end=datetime(2026, 8, 1, 11, tzinfo=UTC),
                ),
                _object("t1"),
            ]
        ),
        "seed",
    )

    fields = _compare_fields(WorldRecall(store).compare("match-1", "t1"))

    assert fields["type_key"] == {
        "field": "type_key",
        "left": "tournament",
        "right": None,
        "equal": False,
    }
    assert fields["event_time"]["left"] == {
        "start": "2026-08-01T09:00:00+00:00",
        "end": "2026-08-01T11:00:00+00:00",
    }
    assert fields["event_time"]["right"] is None


def test_compare_participants_and_name_usages(tmp_path: pytest.TempPathFactory) -> None:
    """Participants (id|role) and asserted usages compare as sets."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _event_object("match-1"),
                _object("player-a"),
                _object("player-b"),
                _object("t1"),
            ],
            observations=[_observation("observation-1")],
            assertions=[
                _participant_edge("p-a", "match-1", "player-a", role="player"),
                _participant_edge("p-b", "match-1", "player-b", role="coach"),
                _name_usage("usage-1", "t1", "darkside"),
            ],
        ),
        "seed",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[_name_usage("usage-2", "t1", "皮蛋")],
        ),
        "later",
    )
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1b")],
            observations=[_observation("observation-3")],
            assertions=[_name_usage("usage-3", "t1b", "darkside")],
        ),
        "seed-b",
    )

    fields = _compare_fields(WorldRecall(store).compare("t1", "t1b"))

    assert fields["name_usages"]["shared"] == ["darkside"]
    assert fields["name_usages"]["left_only"] == ["皮蛋"]
    assert fields["name_usages"]["right_only"] == []


def test_compare_key_assertions_shared_signatures(tmp_path: pytest.TempPathFactory) -> None:
    """A judgment both objects carry shows up under shared, not left_only/right_only."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1"), _object("t2")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("a-1", "t1", literal="an attribute"),
                _assertion("b-1", "t2", literal="an attribute"),
                _assertion("a-2", "t1", literal="only on t1"),
            ],
        ),
        "seed",
    )

    fields = _compare_fields(WorldRecall(store).compare("t1", "t2"))

    assert "related_to|an attribute" in fields["key_assertions"]["shared"]
    assert "related_to|only on t1" in fields["key_assertions"]["left_only"]
    assert fields["key_assertions"]["equal"] is False


def test_compare_assertion_signature_and_role(tmp_path: pytest.TempPathFactory) -> None:
    """Assertion mode lines up signature, window and role scalars."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1"), _object("t2")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("a-1", "t1", literal="first"),
                _assertion("a-2", "t1", object_id="t2").model_copy(
                    update={"epistemic_role": EpistemicRole.AGENT_SYNTHESIS}
                ),
            ],
        ),
        "seed",
    )

    bundle = WorldRecall(store).compare("a-1", "a-2")

    assert bundle.compare["mode"] == "assertion"
    assert bundle.compare["left"] == {"id": "a-1", "status": "ok", "predicate": "related_to"}
    assert "assertion signature compare" in bundle.reasons
    fields = _compare_fields(bundle)
    assert fields["subject_id"]["left"] == "t1"
    assert fields["predicate"] == {
        "field": "predicate",
        "left": "related_to",
        "right": "related_to",
        "equal": True,
    }
    assert fields["literal"]["left"] == "first"
    assert fields["literal"]["right"] is None
    assert fields["object_id"]["left"] is None
    assert fields["object_id"]["right"] == "t2"
    assert fields["epistemic_role"]["equal"] is False
    assert fields["event_time"]["left"] is None
    assert fields["supersedes_id"]["left"] is None


def test_compare_assertion_evidence_split(tmp_path: pytest.TempPathFactory) -> None:
    """Evidence refs split into shared and per-side onlys."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[
                _observation("observation-1"),
                _observation("observation-2"),
                _observation("observation-3"),
            ],
            assertions=[
                _assertion("a-1", "t1", literal="v1"),
                _assertion("a-2", "t1", literal="v1").model_copy(
                    update={"qualifiers": {"lane": "second"}}
                ),
            ],
        ),
        "seed",
    )
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO assertion_evidence (assertion_id, observation_id, role, linked_at)"
            " VALUES ('a-2', 'observation-2', 'supports', '2026-08-03T12:00:00+00:00')"
        )
        connection.execute(
            "INSERT INTO assertion_evidence (assertion_id, observation_id, role, linked_at)"
            " VALUES ('a-2', 'observation-3', 'supports', '2026-08-03T12:00:00+00:00')"
        )

    fields = _compare_fields(WorldRecall(store).compare("a-1", "a-2"))

    assert fields["evidence"]["shared"] == ["observation-1"]
    assert fields["evidence"]["left_only"] == []
    assert fields["evidence"]["right_only"] == ["observation-2", "observation-3"]


def test_compare_assertion_supersede_relations(tmp_path: pytest.TempPathFactory) -> None:
    """Supersede position compares as scalars plus the bounded superseded_by set."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-0", "t1", literal="v0")],
        ),
        "v0",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("a-1", "t1", literal="v1").model_copy(update={"supersedes_id": "a-0"}),
                _assertion("a-3", "t1", literal="v3").model_copy(
                    update={
                        "supersedes_id": "a-1",
                        "superseded_at": datetime(2026, 8, 3, 12, 30, tzinfo=UTC),
                    }
                ),
            ],
        ),
        "v1",
    )

    fields = _compare_fields(WorldRecall(store).compare("a-0", "a-1"))

    assert fields["supersedes_id"] == {
        "field": "supersedes_id",
        "left": None,
        "right": "a-0",
        "equal": False,
    }
    assert fields["superseded_by"]["right_only"] == ["a-3"]
    assert fields["superseded_by"]["left_only"] == ["a-1"]


def test_compare_rejects_object_vs_assertion(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-1", "t1", literal="x")],
        ),
        "seed",
    )

    with pytest.raises(ValueError, match="not one of each"):
        WorldRecall(store).compare("t1", "a-1")


def test_compare_unknown_left_reports_missing(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("t1")]), "seed")

    bundle = WorldRecall(store).compare("missing", "t1")

    assert bundle.compare is None
    assert "unknown id" in bundle.reasons[0]
    assert "missing" in bundle.reasons[0]


def test_compare_unknown_right_reports_missing(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("t1")]), "seed")

    bundle = WorldRecall(store).compare("t1", "missing")

    assert bundle.compare is None
    assert "missing" in bundle.reasons[0]


def test_compare_both_unknown_reports_missing(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")

    bundle = WorldRecall(store).compare("missing-a", "missing-b")

    assert bundle.compare is None
    assert "missing-a" in bundle.reasons[0]


def test_compare_requires_both_ids(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("t1")]), "seed")

    with pytest.raises(ValueError, match="left_id and right_id"):
        WorldRecall(store).compare("t1", "")


def test_compare_key_assertions_capped_with_truncated(tmp_path: pytest.TempPathFactory) -> None:
    """A side beyond its cap reports the cut; the shared set stays intact."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1"), _object("t2")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("a-shared", "t1", literal="common"),
                _assertion("b-shared", "t2", literal="common"),
            ],
        ),
        "seed",
    )
    extra = [
        _assertion(f"a-{index}", "t1", literal=f"attribute {index}") for index in range(VIEW_CAP + 5)
    ]
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=extra,
        ),
        "expand",
    )

    fields = _compare_fields(WorldRecall(store).compare("t1", "t2"))

    assert fields["key_assertions"]["shared"] == ["related_to|common"]
    assert len(fields["key_assertions"]["left_only"]) == VIEW_CAP
    assert fields["key_assertions"]["truncated"] is True
    assert fields["key_assertions"]["equal"] is False


def test_compare_carries_no_adjudication(tmp_path: pytest.TempPathFactory) -> None:
    """The payload states facts and differences only — never a merge verdict."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1"), _object("t2")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("a-1", "t1", literal="x"),
                _assertion("a-2", "t2", literal="y"),
            ],
        ),
        "seed",
    )

    bundle = WorldRecall(store).compare("t1", "t2")

    serialized = bundle.model_dump(mode="json")
    assert "same_referent" not in serialized["compare"]
    assert "merge" not in json.dumps(serialized, ensure_ascii=False).casefold()
    assert "adjudicat" not in json.dumps(serialized, ensure_ascii=False).casefold()
