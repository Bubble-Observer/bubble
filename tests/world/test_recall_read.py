# Slice 2b: memory_read — one object's complete current portrait, or one
# assertion's evidence view (design §4.3). Recall-layer tests: identity
# summary, three-direction views, correction chains, evidence mode, defaults.

from __future__ import annotations

from datetime import UTC, datetime

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


def test_read_object_identity_summary(tmp_path: pytest.TempPathFactory) -> None:
    """The default portrait names every identity surface and view counts."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1", aliases=["Team One", "T1 Official"])],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-1", "t1", literal="an attribute")],
        ),
        "seed",
    )

    bundle = WorldRecall(store).read("t1")

    assert [row["id"] for row in bundle.anchor_objects] == ["t1"]
    identity = bundle.identity
    assert identity["object_id"] == "t1"
    assert identity["canonical_name"] == "T1"
    assert identity["kind"] == "entity"
    assert identity["type_key"] is None
    assert identity["provisional"] is False
    # an object without an event span carries no event_time key at all
    assert "event_time" not in identity
    assert {alias["normalized_alias"] for alias in identity["active_aliases"]} == {
        "t1 official",
        "team one",
    }
    assert all(alias["raw_alias"] for alias in identity["active_aliases"])
    assert identity["removed_aliases"] == []
    assert identity["legacy_aliases"] == []
    assert identity["name_usages"] == []
    assert identity["participants"] == []
    assert identity["view_counts"] == {"self_attributes": 1, "out_edges": 0, "in_edges": 0}
    # details stay folded unless a view is requested
    assert bundle.views == {}
    assert bundle.view_truncated == {}


def test_read_identity_carries_event_span_and_type_key(tmp_path: pytest.TempPathFactory) -> None:
    """Event objects expose their span and domain-independent type_key."""
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
                )
            ]
        ),
        "seed",
    )

    identity = WorldRecall(store).read("match-1").identity

    assert identity["type_key"] == "tournament"
    assert identity["event_time"] == {
        "start": "2026-08-01T09:00:00+00:00",
        "end": "2026-08-01T11:00:00+00:00",
    }


def test_read_defaults_to_latest_committed_object(tmp_path: pytest.TempPathFactory) -> None:
    """Omitting object_id reads the most recently committed object (absorbing recent)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("older")]), "older-commit")
    store.memory_commit(CognitiveDelta(objects=[_object("newer")]), "newer-commit")

    bundle = WorldRecall(store).read()

    assert [row["id"] for row in bundle.anchor_objects] == ["newer"]


def test_read_default_empty_world_returns_empty_bundle(tmp_path: pytest.TempPathFactory) -> None:
    """No committed objects is an empty result, not an error."""
    store = WorldStore(tmp_path / "world.sqlite3")

    bundle = WorldRecall(store).read()

    assert bundle.anchor_objects == []
    assert "no committed object" in bundle.reasons[0]


def test_read_unknown_object_returns_empty_bundle(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("known")]), "seed")

    bundle = WorldRecall(store).read("missing")

    assert bundle.anchor_objects == []
    assert "unknown object" in bundle.reasons[0]


def test_read_self_attributes_view(tmp_path: pytest.TempPathFactory) -> None:
    """Literal assertions anchored on the object render as cards when requested."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1"), _object("t2")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("a-1", "t1", literal="first attribute"),
                _assertion("a-2", "t1", literal="second attribute"),
                _assertion("edge", "t1", object_id="t2"),
            ],
        ),
        "seed",
    )

    bundle = WorldRecall(store).read("t1", views=["self_attributes"])

    assert bundle.identity["view_counts"]["self_attributes"] == 2
    cards = {card["id"]: card for card in bundle.views["self_attributes"]}
    assert set(cards) == {"a-1", "a-2"}
    assert cards["a-1"]["predicate"] == "related_to"
    assert cards["a-1"]["literal"] == "first attribute"
    assert "out_edges" not in bundle.views


def test_read_out_edges_view(tmp_path: pytest.TempPathFactory) -> None:
    """Object-valued edges from the object render with resolved names."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1"), _object("t2")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("edge-a", "t1", object_id="t2"),
                _assertion("edge-b", "t1", object_id="t2").model_copy(
                    update={"qualifiers": {"lane": "second"}}
                ),
            ],
        ),
        "seed",
    )

    bundle = WorldRecall(store).read("t1", views=["out_edges"])

    assert bundle.identity["view_counts"]["out_edges"] == 2
    cards = {card["id"]: card for card in bundle.views["out_edges"]}
    assert set(cards) == {"edge-a", "edge-b"}
    assert cards["edge-a"]["object"] == {"id": "t2", "canonical_name": "T2"}
    assert cards["edge-a"]["subject"] == {"id": "t1", "canonical_name": "T1"}


def test_read_in_edges_view(tmp_path: pytest.TempPathFactory) -> None:
    """Inbound references from other subjects render separately, not flattened."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1"), _object("t2")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("a-1", "t1", literal="own attribute"),
                _assertion("ref-a", "t2", object_id="t1"),
                _assertion("ref-b", "t2", object_id="t1").model_copy(
                    update={"qualifiers": {"lane": "second"}}
                ),
            ],
        ),
        "seed",
    )

    bundle = WorldRecall(store).read("t1", views=["self_attributes", "in_edges"])

    assert bundle.identity["view_counts"]["in_edges"] == 2
    cards = {card["id"]: card for card in bundle.views["in_edges"]}
    assert set(cards) == {"ref-a", "ref-b"}
    assert cards["ref-a"]["subject"] == {"id": "t2", "canonical_name": "T2"}
    # the object's own literal stays in its own direction
    assert [card["id"] for card in bundle.views["self_attributes"]] == ["a-1"]


def test_read_views_exclude_superseded_assertions(tmp_path: pytest.TempPathFactory) -> None:
    """Views show the current state; the correction chain records the retired claim."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-old", "t1", literal="old title")],
        ),
        "old-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("a-new", "t1", literal="new title").model_copy(
                    update={"supersedes_id": "a-old"}
                )
            ],
        ),
        "supersede-claim",
    )

    bundle = WorldRecall(store).read("t1", views=["self_attributes", "correction_chain"])

    assert [card["id"] for card in bundle.views["self_attributes"]] == ["a-new"]
    assert bundle.identity["view_counts"]["self_attributes"] == 1
    chains = bundle.views["correction_chain"]
    assert [entry["tail_id"] for entry in chains] == ["a-new"]
    assert [item["id"] for item in chains[0]["chain"]] == ["a-old", "a-new"]


def test_read_correction_chain_renders_head_to_tail(tmp_path: pytest.TempPathFactory) -> None:
    """A multi-hop chain reads oldest -> newest with the tail marked current."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-1", "t1", literal="v1")],
        ),
        "v1",
    )
    for step, previous in (("a-2", "a-1"), ("a-3", "a-2")):
        store.memory_commit(
            CognitiveDelta(
                observations=[_observation(f"observation-{step}")],
                assertions=[
                    _assertion(step, "t1", literal=f"v{step[-1]}").model_copy(
                        update={
                            "supersedes_id": previous,
                            # the superseding row carries the correction stamp
                            "superseded_at": datetime(2026, 8, 3, 12, 30, tzinfo=UTC),
                        }
                    )
                ],
            ),
            step,
        )

    bundle = WorldRecall(store).read("t1", views=["correction_chain"])

    chain = bundle.views["correction_chain"][0]
    assert chain["tail_id"] == "a-3"
    assert [item["id"] for item in chain["chain"]] == ["a-1", "a-2", "a-3"]
    assert [item["current"] for item in chain["chain"]] == [False, False, True]
    assert chain["chain"][2]["supersedes_id"] == "a-2"
    assert chain["chain"][2]["superseded_at"] is not None
    assert chain["chain"][0]["supersedes_id"] is None


def test_read_views_absent_by_default(tmp_path: pytest.TempPathFactory) -> None:
    """Without a views argument nothing expands (output-driven, no default dump)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-1", "t1", literal="attribute")],
        ),
        "seed",
    )

    bundle = WorldRecall(store).read("t1")

    assert bundle.views == {}
    assert bundle.view_truncated == {}
    assert bundle.identity["view_counts"]["self_attributes"] == 1


def test_read_assertion_evidence_view(tmp_path: pytest.TempPathFactory) -> None:
    """assertion_id returns the judgment with its bounded evidence refs."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1"), _object("t2")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-1", "t1", object_id="t2")],
        ),
        "seed",
    )

    bundle = WorldRecall(store).read(assertion_id="a-1")

    assert [card["id"] for card in bundle.assertions] == ["a-1"]
    assert bundle.assertions[0]["subject"] == {"id": "t1", "canonical_name": "T1"}
    assert bundle.assertions[0]["object"] == {"id": "t2", "canonical_name": "T2"}
    assert any(ref["observation_id"] == "observation-1" for ref in bundle.evidence_refs)
    assert bundle.assertion_chain == {"supersedes": [], "superseded_by": []}


def test_read_evidence_view_carries_correction_chain(tmp_path: pytest.TempPathFactory) -> None:
    """Evidence mode names what the judgment corrected and what corrected it."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-1", "t1", literal="v1")],
        ),
        "v1",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("a-2", "t1", literal="v2").model_copy(update={"supersedes_id": "a-1"}),
                _assertion("a-3", "t1", literal="v3").model_copy(update={"supersedes_id": "a-2"}),
            ],
        ),
        "v2",
    )

    bundle = WorldRecall(store).read(assertion_id="a-2")

    assert [item["id"] for item in bundle.assertion_chain["supersedes"]] == ["a-1"]
    assert [item["id"] for item in bundle.assertion_chain["superseded_by"]] == ["a-3"]


def test_read_unknown_assertion_returns_empty_bundle(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("a-1", "t1", literal="x")],
        ),
        "seed",
    )

    bundle = WorldRecall(store).read(assertion_id="missing")

    assert bundle.assertions == []
    assert "unknown assertion" in bundle.reasons[0]


def test_read_removed_and_legacy_aliases(tmp_path: pytest.TempPathFactory) -> None:
    """Retired identity forms stay visible as history, separated by origin."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(objects=[_object("t1", aliases=["Former Name", "Keep"])]),
        "seed",
    )
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE identity_aliases SET status = 'removed', removed_commit_id = 'later'"
            " WHERE object_id = 't1' AND normalized_alias = 'former name'"
        )
        connection.execute(
            "INSERT INTO object_aliases (normalized_alias, object_id) VALUES ('old-form', 't1')"
        )

    identity = WorldRecall(store).read("t1").identity

    assert [alias["normalized_alias"] for alias in identity["active_aliases"]] == ["keep"]
    assert [alias["normalized_alias"] for alias in identity["removed_aliases"]] == ["former name"]
    assert identity["legacy_aliases"] == ["old-form"]


def test_read_participants_list_outbound_has_participant_edges(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """An event's participants render with role and resolved names."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_event_object("match-1"), _object("player-a"), _object("player-b")],
            observations=[_observation("observation-1")],
            assertions=[
                _participant_edge("p-a", "match-1", "player-a", role="player"),
                _participant_edge("p-b", "match-1", "player-b", role="coach"),
            ],
        ),
        "seed",
    )

    identity = WorldRecall(store).read("match-1").identity

    assert identity["participants"] == [
        {"object_id": "player-a", "canonical_name": "Player A", "role": "player"},
        {"object_id": "player-b", "canonical_name": "Player B", "role": "coach"},
    ]
    assert identity["view_counts"]["out_edges"] == 2


def test_read_name_usages_list_current_usage_assertions(tmp_path: pytest.TempPathFactory) -> None:
    """Asserted community usages render with their assertion ids."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=[
                _name_usage("usage-1", "t1", "darkside"),
                _name_usage("usage-2", "t1", "皮蛋"),
            ],
        ),
        "seed",
    )

    identity = WorldRecall(store).read("t1").identity

    assert identity["name_usages"] == [
        {"assertion_id": "usage-1", "literal": "darkside"},
        {"assertion_id": "usage-2", "literal": "皮蛋"},
    ]


def test_read_rejects_object_and_assertion_ids(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("t1")]), "seed")

    with pytest.raises(ValueError, match="not both"):
        WorldRecall(store).read("t1", assertion_id="a-1")


def test_read_rejects_unknown_view(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("t1")]), "seed")

    with pytest.raises(ValueError, match="unknown view"):
        WorldRecall(store).read("t1", views=["bogus"])


def test_read_views_capped_with_truncated_flags(tmp_path: pytest.TempPathFactory) -> None:
    """A view beyond its cap reports the cut; uncut views carry no flag."""
    store = WorldStore(tmp_path / "world.sqlite3")
    assertions = [
        _assertion(f"a-{index}", "t1", literal=f"attribute {index}") for index in range(VIEW_CAP + 5)
    ]
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("t1")],
            observations=[_observation("observation-1")],
            assertions=assertions,
        ),
        "seed",
    )

    bundle = WorldRecall(store).read("t1", views=["self_attributes"])

    assert len(bundle.views["self_attributes"]) == VIEW_CAP
    assert bundle.view_truncated == {"self_attributes": True}
    # the count reports the true total, not the capped page
    assert bundle.identity["view_counts"]["self_attributes"] == VIEW_CAP + 5
