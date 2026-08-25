# Split from tests/world/test_world_recall.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leave_information_bubble.world import (
    CognitiveDelta,
    WorldRecall,
    WorldStore,
)
from tests.world._recall_helpers import (
    _assertion,
    _event_object,
    _object,
    _observation,
    _participant_edge,
    _TracingWorldStore,
)


def test_recent_limits_high_cardinality_assertions_in_sql(tmp_path: pytest.TempPathFactory) -> None:
    """Catch recent recall that reads every assertion before applying its result cap."""
    store = _TracingWorldStore(str(tmp_path / "world.sqlite3"))
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("subject")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion(f"assertion-{index}", "subject", literal=f"fact {index}")
                for index in range(8)
            ],
        ),
        "many-assertions",
    )

    bundle = WorldRecall(store).recent(limit=3)

    assert len(bundle.assertions) == 3
    assert bundle.truncated is True
    assert any("FROM assertions" in statement and "LIMIT 4" in statement for statement in store.statements)
    evidence_statements = [statement for statement in store.statements if "assertion_evidence" in statement]
    assert len(evidence_statements) <= 5  # detail refs, two basis batches, sources, and flip-flops


def test_recent_and_expand_exclude_superseded_assertions(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch recent/expand recall that resurrects retired claims (B2-8)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("s16")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("assertion-old", "s16", literal="old title")],
        ),
        "old-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("assertion-new", "s16", literal="new title").model_copy(
                    update={"supersedes_id": "assertion-old"}
                )
            ],
        ),
        "supersede-claim",
    )
    recall = WorldRecall(store)

    assert [row["id"] for row in recall.recent().assertions] == ["assertion-new"]
    assert [row["id"] for row in recall.expand(["s16"]).assertions] == ["assertion-new"]


def test_expand_returns_bounded_deduplicated_paths_with_evidence_refs(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch graph expansion that duplicates edges or loses supporting observations."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("a"), _object("b"), _object("c")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("a-b", "a", object_id="b"),
                _assertion("b-c", "b", object_id="c"),
            ],
        ),
        "linked-world",
    )

    bundle = WorldRecall(store).expand(["a", "a"], depth=2, limit=2)

    assert [row["id"] for row in bundle.assertions] == ["a-b", "b-c"]
    assert [row["id"] for row in bundle.neighboring_objects] == ["b", "c"]
    assert bundle.paths == [["a", "b"], ["a", "b", "c"]]
    assert [row["observation_id"] for row in bundle.evidence_refs] == [
        "observation-1",
        "observation-1",
    ]


def test_expand_limits_high_cardinality_edges_in_sql(tmp_path: pytest.TempPathFactory) -> None:
    """Catch expansion that materializes every adjacent edge before honoring the graph budget."""
    store = _TracingWorldStore(str(tmp_path / "world.sqlite3"))
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("root"), *[_object(f"leaf-{index}") for index in range(8)]],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion(f"edge-{index}", "root", object_id=f"leaf-{index}") for index in range(8)
            ],
        ),
        "many-edges",
    )

    bundle = WorldRecall(store).expand(["root"], limit=2)

    assert len(bundle.assertions) == 2
    assert bundle.truncated is True
    # the edge query aggregates by subject (literal assertions included — the
    # old object_id IS NOT NULL filter is gone), retires superseded assertions
    # in SQL (B2-8), and still bounds each frontier row read before slicing in
    # Python (trace callback inlines the bound ids)
    assert any(
        "FROM assertions WHERE (subject_id = 'root' OR object_id = 'root')" in statement
        and "NOT EXISTS (SELECT 1 FROM assertions s" in statement
        and "LIMIT 3" in statement
        for statement in store.statements
    )
    assert "object_id IS NOT NULL" not in " ".join(store.statements)


def test_expand_caps_deduplicated_roots_before_anchor_query(tmp_path: pytest.TempPathFactory) -> None:
    """Catch oversized root input that leaks past the expansion budget into anchors or SQL parameters."""
    store = WorldStore(tmp_path / "world.sqlite3")
    root_ids = [f"root-{index}" for index in range(5)]
    store.memory_commit(
        CognitiveDelta(objects=[_object(identifier) for identifier in root_ids]), "many-roots"
    )

    bundle = WorldRecall(store).expand([*root_ids, root_ids[0]], limit=2)

    assert [row["id"] for row in bundle.anchor_objects] == root_ids[:2]


def test_expand_returns_literal_assertions_anchored_on_subject(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch expansion that returns nothing for literal-only subjects (audit F4).

    The old SQL required ``object_id IS NOT NULL``, so objects whose cognition
    is literal (27 assertions, only 1 object-valued, in the audited store)
    expanded to empty bundles. Literal assertions are now reported, and only
    object-valued edges spawn neighbor traversal.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("blg"), _object("bin-team")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("blg-suspense", "blg", literal="Bin 宣布休赛"),
                _assertion("blg-bin", "blg", object_id="bin-team"),
            ],
        ),
        "literal-world",
    )

    bundle = WorldRecall(store).expand(["blg"], depth=1, limit=10)

    # rows come back in assertion-id order (the SQL ORDER BY id)
    assert [row["id"] for row in bundle.assertions] == ["blg-bin", "blg-suspense"]
    assert [row["id"] for row in bundle.neighboring_objects] == ["bin-team"]
    assert bundle.paths == [["blg", "bin-team"]]
    assert [row["observation_id"] for row in bundle.evidence_refs] == ["observation-1", "observation-1"]


def test_expand_ego_view_returns_direct_neighbors_with_edge_predicates(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Ego view: direct object-edge neighbors aggregate their edge predicates.

    The ego zoom reports who the subject is directly connected to and via
    which predicates (spec §3.1: Bin--member_of-->BLG), in first-seen
    assertion order with predicates sorted and assertion ids first-seen.
    Literal claims never spawn neighbors; superseded edges stay retired.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("bin"), _object("blg"), _object("match")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("edge-a", "bin", object_id="blg").model_copy(
                    update={"predicate": "member_of"}
                ),
                _assertion("edge-b", "bin", object_id="blg").model_copy(
                    update={"predicate": "contract"}
                ),
                _assertion("edge-c", "match", object_id="bin").model_copy(
                    update={"predicate": "involved_team"}
                ),
                _assertion("edge-d", "bin", object_id="bin").model_copy(
                    update={"predicate": "self_reference"}
                ),
            ],
        ),
        "ego-edges",
    )

    bundle = WorldRecall(store).expand(["bin"], depth=1, limit=10)

    assert bundle.ego_neighbors == [
        {
            "id": "blg",
            "canonical_name": "Blg",
            "predicates": ["contract", "member_of"],
            "assertion_ids": ["edge-a", "edge-b"],
        },
        {
            "id": "match",
            "canonical_name": "Match",
            "predicates": ["involved_team"],
            "assertion_ids": ["edge-c"],
        },
    ]
    # the ego fields never replace the legacy depth/limit semantics
    assert [row["id"] for row in bundle.assertions] == ["edge-a", "edge-b", "edge-c", "edge-d"]
    assert [row["id"] for row in bundle.neighboring_objects] == ["blg", "match"]
    assert bundle.truncated is False


def test_expand_ego_view_status_and_timeline_caps_with_truncation(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Ego status (<=3 literal claims) and event timeline (<=5) cap and mark truncated."""
    store = WorldStore(tmp_path / "world.sqlite3")
    times = [datetime(2026, 8, index, tzinfo=UTC) for index in range(1, 5)]
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("bin"), _object("blg")],
            observations=[_observation("observation-1")],
            assertions=[
                # 4 current literal claims: the newest three survive, all current
                _assertion(f"status-{index}", "bin", literal=f"claim {index}")
                for index in range(1, 5)
            ]
            + [
                # 6 event-time assertions (4 literal + 2 object edges): newest five survive
                _assertion(
                    f"event-{index}",
                    "bin",
                    literal=f"event {index}",
                    event_time_start=times[index - 1],
                )
                for index in range(1, 5)
            ]
            + [
                _assertion("event-5", "bin", object_id="blg", event_time_start=times[3]),
                _assertion("event-6", "blg", object_id="bin", event_time_start=times[3]),
            ],
        ),
        "ego-caps",
    )

    bundle = WorldRecall(store).expand(["bin"], depth=1, limit=30)

    # status: current literal claims, newest first, capped at 3
    assert [row["id"] for row in bundle.status_assertions] == ["status-4", "status-3", "status-2"]
    assert bundle.status_assertions[0]["literal"] == "claim 4"
    assert bundle.status_assertions[0]["predicate"] == "related_to"
    # timeline: event-time assertions involving the subject (either side),
    # newest event time first, capped at 5; edges carry their object target
    assert [row["id"] for row in bundle.event_timeline] == [
        "event-6",
        "event-5",
        "event-4",
        "event-3",
        "event-2",
    ]
    assert bundle.event_timeline[0]["object"] == {"id": "bin", "canonical_name": "Bin"}
    assert bundle.event_timeline[4]["literal"] == "event 2"
    assert bundle.truncated is True


def test_expand_ego_view_multi_root_timeline_is_globally_sorted_and_capped(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Multi-root timeline merges across roots, then globally re-sorts and caps.

    Each root's events arrive newest-first, so a naive concatenation is
    root-grouped: beta's newer event could be cut while alpha's oldest is
    kept, and the drop would go un-truncated. The aggregate must be ordered
    by event time across ALL roots and any over-cap cut must mark truncated.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    times = [datetime(2026, 8, index, tzinfo=UTC) for index in range(1, 7)]
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("alpha"), _object("beta")],
            observations=[_observation("observation-1")],
            assertions=[
                # alpha: Aug 1 / 3 / 5; beta: Aug 2 / 4 / 6 (interleaved)
                _assertion("a1", "alpha", literal="a1", event_time_start=times[0]),
                _assertion("a2", "alpha", literal="a2", event_time_start=times[2]),
                _assertion("a3", "alpha", literal="a3", event_time_start=times[4]),
                _assertion("b1", "beta", literal="b1", event_time_start=times[1]),
                _assertion("b2", "beta", literal="b2", event_time_start=times[3]),
                _assertion("b3", "beta", literal="b3", event_time_start=times[5]),
            ],
        ),
        "ego-multi-root",
    )

    bundle = WorldRecall(store).expand(["alpha", "beta"], depth=1, limit=30)

    # globally event_time-sorted, not root-grouped: b3 (Aug 6) first, a1 (Aug
    # 1) cut — never a newer beta event dropped while alpha's oldest survives
    assert [row["id"] for row in bundle.event_timeline] == [
        "b3",
        "a3",
        "b2",
        "a2",
        "b1",
    ]
    assert bundle.truncated is True


def test_expand_ego_view_stays_direct_at_higher_depth(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Ego neighbors are the direct ego only; depth still extends the legacy frontier."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("a"), _object("b"), _object("c")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("a-b", "a", object_id="b"),
                _assertion("b-c", "b", object_id="c"),
            ],
        ),
        "ego-depth",
    )

    bundle = WorldRecall(store).expand(["a"], depth=2, limit=30)

    assert [row["id"] for row in bundle.neighboring_objects] == ["b", "c"]
    assert [item["id"] for item in bundle.ego_neighbors] == ["b"]
    assert bundle.status_assertions == []
    assert bundle.event_timeline == []


def test_expand_ego_view_empty_for_isolated_object(tmp_path: pytest.TempPathFactory) -> None:
    """An isolated object yields empty ego sections without a truncation flag."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("solo")]), "isolated")

    bundle = WorldRecall(store).expand(["solo"], depth=1, limit=30)

    assert bundle.ego_neighbors == []
    assert bundle.status_assertions == []
    assert bundle.event_timeline == []
    assert bundle.event_edges == []
    assert bundle.participated_events == []
    assert bundle.omitted_counts == {}
    assert bundle.sort_basis == ""
    assert bundle.event_next_cursor is None
    assert bundle.assertions == []
    assert bundle.truncated is False


def test_event_ego_contains_role_qualified_participant_edges(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """An event root exposes its has_participant edges with role, qualifiers, evidence.

    The ego zoom on an event node returns the compiled participant edges
    (role from qualifiers, bounded qualifier map, zero-filled evidence counts
    by role, the event's own time). No team<->team per-match direct edge is
    ever synthesized: the match stays an event node plus participant
    assertions, and the team's own ego shows the event node, not the rival.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _event_object("match-1", event_time_start=datetime(2026, 8, 3, tzinfo=UTC)),
                _object("blg"),
                _object("bin"),
            ],
            observations=[
                _observation("observation-1"),
                _observation("observation-2"),
            ],
            assertions=[
                _participant_edge(
                    "part-1",
                    "match-1",
                    "blg",
                    role="team",
                    qualifiers={"community": "lpl"},
                    evidence=[("observation-1", "supports"), ("observation-2", "context")],
                ),
                _participant_edge("part-2", "match-1", "bin", role="player"),
            ],
        ),
        "event-ego",
    )

    bundle = WorldRecall(store).expand(["match-1"], depth=1, limit=10)

    assert bundle.event_edges == [
        {
            "id": "part-1",
            "object": {"id": "blg", "canonical_name": "Blg"},
            "role": "team",
            "qualifiers": {"community": "lpl", "role": "team"},
            "event_time_start": "2026-08-03T00:00:00+00:00",
            "epistemic_role": "fact",
            "confidence": 0.8,
            "evidence_counts": {"supports": 1, "context": 1, "contradicts": 0},
        },
        {
            "id": "part-2",
            "object": {"id": "bin", "canonical_name": "Bin"},
            "role": "player",
            "qualifiers": {"role": "player"},
            "event_time_start": "2026-08-03T00:00:00+00:00",
            "epistemic_role": "fact",
            "confidence": 0.8,
            "evidence_counts": {"supports": 1, "context": 0, "contradicts": 0},
        },
    ]
    assert bundle.omitted_counts == {}
    assert bundle.truncated is False
    # the event node's own ego names participants via has_participant only —
    # no team<->team direct edge exists anywhere
    assert [item["id"] for item in bundle.ego_neighbors] == ["blg", "bin"]
    assert all(item["predicates"] == ["has_participant"] for item in bundle.ego_neighbors)
    team_view = WorldRecall(store).expand(["blg"], depth=1, limit=10)
    assert [item["id"] for item in team_view.ego_neighbors] == ["match-1"]
    assert team_view.event_edges == []


def test_entity_ego_returns_recent_participated_events_with_stable_cursor(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """An entity root reverses its participation edges into recent events.

    Participated events sort by ``event_time_start DESC, id ASC`` (stable:
    same-time events break ties by event id ascending), respect an
    independent event_limit, and page forward through an optional
    before-cursor returned as event_next_cursor. Over-cap cuts surface in
    omitted_counts instead of silently vanishing.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("team-a"),
                _event_object("match-1", event_time_start=datetime(2026, 8, 3, tzinfo=UTC)),
                _event_object("match-2", event_time_start=datetime(2026, 8, 2, tzinfo=UTC)),
                _event_object("match-3", event_time_start=datetime(2026, 8, 3, tzinfo=UTC)),
            ],
            observations=[_observation("observation-1")],
            assertions=[
                _participant_edge("m1p", "match-1", "team-a", role="team"),
                _participant_edge("m2p", "match-2", "team-a", role="team"),
                _participant_edge("m3p", "match-3", "team-a", role="team"),
            ],
        ),
        "entity-ego",
    )

    first = WorldRecall(store).expand(["team-a"], depth=1, limit=10, event_limit=2)

    # Aug 3 first (DESC); the two Aug 3 events tie-break by id ASC
    assert [item["id"] for item in first.participated_events] == ["match-1", "match-3"]
    assert first.participated_events[0] == {
        "id": "match-1",
        "canonical_name": "Match 1",
        "event_time_start": "2026-08-03T00:00:00+00:00",
        "role": "team",
        "assertion_id": "m1p",
        "qualifiers": {"role": "team"},
        "evidence_counts": {"supports": 1, "context": 0, "contradicts": 0},
    }
    assert first.omitted_counts == {"participated_events": 1}
    assert first.sort_basis == "event_time_start DESC, id ASC"
    assert first.event_next_cursor == (
        '{"t": "2026-08-03T00:00:00+00:00", "id": "match-3"}'
    )
    assert first.truncated is True

    second = WorldRecall(store).expand(
        ["team-a"], depth=1, limit=10, event_limit=2, before=first.event_next_cursor
    )

    assert [item["id"] for item in second.participated_events] == ["match-2"]
    assert second.omitted_counts == {}
    assert second.event_next_cursor is None
    assert second.truncated is False


def test_dense_event_history_is_capped_without_hiding_omitted_count(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A dense participation history pages by cap and reports the full omission.

    The COUNT-based omitted count tells the caller how many qualifying
    participated events exist beyond the returned page — a cap cut is never
    silent, and the next_cursor allows continuing the surface.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("team-x"),
                *[
                    _event_object(
                        f"match-{index}", event_time_start=datetime(2026, 8, index, tzinfo=UTC)
                    )
                    for index in range(1, 21)
                ],
            ],
            observations=[_observation("observation-1")],
            assertions=[
                _participant_edge(f"mp-{index}", f"match-{index}", "team-x", role="team")
                for index in range(1, 21)
            ],
        ),
        "dense-ego",
    )

    bundle = WorldRecall(store).expand(["team-x"], depth=1, limit=30, event_limit=3)

    assert [item["id"] for item in bundle.participated_events] == [
        "match-20",
        "match-19",
        "match-18",
    ]
    # the team's own ego also caps the event nodes as direct neighbors; both
    # cap cuts are reported, never silent
    assert bundle.omitted_counts == {
        "neighbors": 15,
        "participated_events": 17,
    }
    assert bundle.event_next_cursor == (
        '{"t": "2026-08-18T00:00:00+00:00", "id": "match-18"}'
    )
    assert bundle.truncated is True


def test_expand_include_history_surfaces_superseded_assertions(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """include_history=true returns retired claims beside current ones; default stays current-only.

    Design §4.5: the switch is optional with the current state as the
    default; the historical state surfaces claims the current-only gate
    retires (B2-8), so an agent reviewing a correction can see both sides of
    a supersede chain in one expand instead of reading the chain separately.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("s16")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("assertion-old", "s16", literal="old title")],
        ),
        "old-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("assertion-new", "s16", literal="new title").model_copy(
                    update={"supersedes_id": "assertion-old"}
                )
            ],
        ),
        "supersede-claim",
    )
    recall = WorldRecall(store)

    assert [row["id"] for row in recall.expand(["s16"]).assertions] == ["assertion-new"]
    # per-root id order: "assertion-new" sorts before "assertion-old" (n < o)
    assert [row["id"] for row in recall.expand(["s16"], include_history=True).assertions] == [
        "assertion-new",
        "assertion-old",
    ]


def test_expand_include_history_walks_retired_edges(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A superseded object edge still extends the frontier under include_history.

    The historical state is the graph as recorded: a retired a->b edge
    resurrects b and b's own claims into the slice, while the current state
    only walks the live a->c edge. Both states stay bounded by depth/limit.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("a"), _object("b"), _object("c")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("edge-old", "a", object_id="b"),
                _assertion("b-lit", "b", literal="b status"),
            ],
        ),
        "old-edge",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("edge-new", "a", object_id="c").model_copy(
                    update={"supersedes_id": "edge-old"}
                )
            ],
        ),
        "replacement",
    )
    recall = WorldRecall(store)

    current = recall.expand(["a"], depth=2, limit=30)
    assert [row["id"] for row in current.assertions] == ["edge-new"]
    assert [row["id"] for row in current.neighboring_objects] == ["c"]

    historical = recall.expand(["a"], depth=2, limit=30, include_history=True)
    # per-frontier id order: a's rows first (edge-new, edge-old), then b's row
    assert [row["id"] for row in historical.assertions] == ["edge-new", "edge-old", "b-lit"]
    assert [row["id"] for row in historical.neighboring_objects] == ["c", "b"]


def test_expand_include_history_keeps_ego_current_only(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Ego sections stay current under include_history; history rides the assertions list.

    The ego zoom is the current cognitive surface — statuses and
    participation now, not the full paper trail. Retired claims appear in
    the history-enriched assertions list, while status_assertions and
    participated_events report exactly the same rows in both states; the
    correction chain view of memory_read covers the historical story.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("team-a"),
                _event_object("match-1", event_time_start=datetime(2026, 8, 3, tzinfo=UTC)),
            ],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("status-old", "team-a", literal="old status"),
                _participant_edge("part-old", "match-1", "team-a", role="team"),
            ],
        ),
        "old-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("status-new", "team-a", literal="new status").model_copy(
                    update={"supersedes_id": "status-old"}
                ),
                # a distinct event window keeps part-new's signature different
                # from part-old's, so the supersede row is not deduplicated away
                _participant_edge("part-new", "match-1", "team-a", role="team").model_copy(
                    update={
                        "supersedes_id": "part-old",
                        "event_time_start": datetime(2026, 8, 3, 10, tzinfo=UTC),
                    }
                ),
            ],
        ),
        "supersede",
    )
    recall = WorldRecall(store)

    current = recall.expand(["team-a"], depth=1, limit=30)
    historical = recall.expand(["team-a"], depth=1, limit=30, include_history=True)

    assert [row["id"] for row in current.status_assertions] == ["status-new"]
    assert [row["id"] for row in historical.status_assertions] == ["status-new"]
    assert [item["id"] for item in current.participated_events] == ["match-1"]
    assert [item["id"] for item in historical.participated_events] == ["match-1"]
    assert {"status-old", "part-old"} <= {row["id"] for row in historical.assertions}
    assert {"status-old", "part-old"} & {row["id"] for row in current.assertions} == set()


def test_ego_remains_one_hop_at_higher_depth(tmp_path: pytest.TempPathFactory) -> None:
    """Higher depth never widens the ego event surface: it stays direct.

    Even at depth 2 the ego sections show only what the ROOT participates in
    or connects to directly (event ev-1); ev-2 is two hops away via team-b
    and must not leak into ego_neighbors or participated_events, while the
    legacy depth/limit frontier still walks the graph.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("team-a"),
                _object("team-b"),
                _object("team-c"),
                _event_object("ev-1", event_time_start=datetime(2026, 8, 3, tzinfo=UTC)),
                _event_object("ev-2", event_time_start=datetime(2026, 8, 4, tzinfo=UTC)),
            ],
            observations=[_observation("observation-1")],
            assertions=[
                _participant_edge("ev1a", "ev-1", "team-a", role="team"),
                _participant_edge("ev1b", "ev-1", "team-b", role="team"),
                _participant_edge("ev2b", "ev-2", "team-b", role="team"),
                _participant_edge("ev2c", "ev-2", "team-c", role="team"),
            ],
        ),
        "ego-depth-events",
    )

    bundle = WorldRecall(store).expand(["team-a"], depth=2, limit=30, event_limit=10)

    assert [item["id"] for item in bundle.ego_neighbors] == ["ev-1"]
    assert [item["id"] for item in bundle.participated_events] == ["ev-1"]
    # the legacy frontier walks exactly depth 2: ev-1 then team-b; ev-2 is
    # three hops away and appears in NO section
    assert {item["id"] for item in bundle.neighboring_objects} == {"ev-1", "team-b"}
    assert "ev-2" not in {item["id"] for item in bundle.neighboring_objects}


