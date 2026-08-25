"""Offline contracts for the console's read-only inspection layer."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from leave_information_bubble.console.inspection import ReadOnlyInspection
from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    InquiryInput,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    ObservationLinkInput,
    WorldStore,
)
from leave_information_bubble.world.schema import (
    MIGRATION_V4_SQL,
    MIGRATION_V5_SQL,
    MIGRATION_V6_SQL,
    SCHEMA_V1_SQL,
    _apply_migration_v2,
    _apply_migration_v3,
    _apply_migration_v7,
    _apply_migration_v7_attempt_ledger,
    _apply_migration_v8_inquiry_touch_ledger,
    _apply_migration_v9,
)


def _world(path: Path) -> WorldStore:
    store = WorldStore(path)
    now = datetime(2026, 8, 14, tzinfo=UTC)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="object-one",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="中文社区话题",
                    aliases=["Community Topic"],
                )
            ],
            observations=[
                ObservationInput(
                    id="observation-one",
                    source_uri="https://example.test/one",
                    source_kind="public-web",
                    title="一条材料",
                    excerpt="摘要",
                    content_ref="",
                    depth=ObservationDepth.CONTENT,
                    observed_at=now,
                    source_published_at=now,
                    metadata={
                        "material_reliability": "best_effort",
                        "limitations": ["automatic transcript", "partial capture"],
                    },
                )
            ],
            assertions=[
                AssertionInput(
                    id="assertion-one",
                    subject_id="object-one",
                    predicate="expresses",
                    literal="一种中文社区说法",
                    epistemic_role=EpistemicRole.COMMUNITY_VIEW,
                    confidence=0.7,
                    evidence=[EvidenceInput(observation_id="observation-one", role="supports")],
                )
            ],
            inquiries=[
                InquiryInput(
                    id="inquiry-one",
                    subject_id="object-one",
                    prompt="这种说法如何演化？",
                    rationale="保留语言变化线索",
                    created_at=now,
                    last_attempted_at=now,
                )
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object",
                    target_id="object-one",
                    observation_id="observation-one",
                    role="context",
                )
            ],
        ),
        "agent:thread-one",
    )
    return store


def _graph_world(path: Path) -> WorldStore:
    """Create a small durable object graph with one literal and one object edge."""
    store = _world(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="object-two",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="国际社区话题",
                )
            ],
            assertions=[
                AssertionInput(
                    id="assertion-related",
                    subject_id="object-one",
                    predicate="related_to",
                    object_id="object-two",
                    epistemic_role=EpistemicRole.AGENT_SYNTHESIS,
                    confidence=0.8,
                )
            ],
        ),
        "agent:thread-two",
    )
    return store


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    world_db = tmp_path / "missing" / "world.sqlite3"
    reader = ReadOnlyInspection(world_db)

    assert reader.summary()["initialized"] is False
    assert reader.search_objects("anything")["available"] is False
    assert reader.object_detail("missing") is None
    assert not world_db.exists()


def test_memory_queries_are_bounded_and_read_only(tmp_path: Path) -> None:
    world_db = tmp_path / "world.sqlite3"
    _world(world_db)
    before = world_db.read_bytes()
    reader = ReadOnlyInspection(world_db)

    summary = reader.summary()
    assert summary["counts"]["objects"] == 1
    assert summary["counts"]["current_assertions"] == 1
    search = reader.search_objects("中文")
    assert search["items"][0]["id"] == "object-one"
    alias_search = reader.search_objects("community")
    assert alias_search["items"][0]["id"] == "object-one"
    detail = reader.object_detail("object-one")
    assert detail is not None
    assertion = detail["assertions"][0]
    evidence = assertion["evidence"][0]
    assert evidence["observation_id"] == "observation-one"
    assert assertion["direction"] == "subject"
    assert assertion["evidence_truncated"] is False
    assert evidence["role"] == "supports"
    assert evidence["observation"]["source_kind"] == "public-web"
    assert evidence["observation"]["depth"] == "content"
    assert evidence["observation"]["source_published_at"]
    assert evidence["observation"]["observed_at"]
    assert evidence["observation"]["material_reliability"] == "best_effort"
    assert evidence["observation"]["limitations"] == ["automatic transcript", "partial capture"]
    assert detail["inquiries"][0]["id"] == "inquiry-one"
    assert detail["observations"][0]["id"] == "observation-one"
    assert reader.inquiries(statuses=("open",))["items"][0]["subject_name"] == "中文社区话题"
    assert reader.recent_commits()["items"][0]["commit_id"] == "agent:thread-one"
    assert world_db.read_bytes() == before

    with pytest.raises(ValueError, match="limit"):
        reader.search_objects(limit=101)
    with pytest.raises(ValueError, match="status"):
        reader.inquiries(statuses=("unknown",))


def test_graph_snapshot_returns_real_current_object_relationships_without_writes(
    tmp_path: Path,
) -> None:
    world_db = tmp_path / "graph.sqlite3"
    _graph_world(world_db)
    before = world_db.read_bytes()

    snapshot = ReadOnlyInspection(world_db).graph_snapshot(limit=8)

    assert [node["id"] for node in snapshot["nodes"]] == ["object-one", "object-two"]
    assert snapshot["edges"] == [
        {
            "id": "assertion-related",
            "source": "object-one",
            "target": "object-two",
            "predicate": "related_to",
            "epistemic_role": "agent_synthesis",
            "confidence": 0.8,
            "assertion_count": 1,
            "assertion_ids": ["assertion-related"],
        }
    ]
    assert snapshot["available"] is True
    assert snapshot["truncated"] is False
    assert snapshot["selection"] == "relationship_first"
    assert snapshot["window"] == {"index": 0, "count": 1, "size": 2, "rotating": False}
    assert snapshot["stats"] == {
        "total_nodes": 2,
        "total_current_assertions": 2,
        "total_object_relations": 1,
        "total_object_connections": 1,
        "total_literal_assertions": 1,
        "displayed_nodes": 2,
        "displayed_relations": 1,
        "displayed_connections": 1,
        "collapsed_relations": 0,
    }
    assert snapshot["latest_commit"]["commit_id"] == "agent:thread-two"
    assert world_db.read_bytes() == before


def test_graph_snapshot_aggregates_same_semantic_relation_with_distinct_windows(
    tmp_path: Path,
) -> None:
    """Historical parallel assertions render as one honest semantic connection."""
    world_db = tmp_path / "parallel-relations.sqlite3"
    store = _graph_world(world_db)
    store.memory_commit(
        CognitiveDelta(
            assertions=[
                AssertionInput(
                    id="assertion-related-dated",
                    subject_id="object-one",
                    predicate="related_to",
                    object_id="object-two",
                    epistemic_role=EpistemicRole.AGENT_SYNTHESIS,
                    confidence=0.9,
                    event_time_start=datetime(2026, 8, 9, tzinfo=UTC),
                )
            ]
        ),
        "agent:thread-three",
    )
    before = world_db.read_bytes()

    snapshot = ReadOnlyInspection(world_db).graph_snapshot(limit=8)

    assert len(snapshot["edges"]) == 1
    assert snapshot["edges"][0]["assertion_count"] == 2
    assert snapshot["edges"][0]["assertion_ids"] == [
        "assertion-related",
        "assertion-related-dated",
    ]
    assert snapshot["stats"]["total_object_relations"] == 2
    assert snapshot["stats"]["total_object_connections"] == 1
    assert snapshot["stats"]["displayed_relations"] == 2
    assert snapshot["stats"]["displayed_connections"] == 1
    assert snapshot["stats"]["collapsed_relations"] == 1
    assert snapshot["truncated"] is False
    assert world_db.read_bytes() == before


def test_graph_snapshot_keeps_relation_endpoints_ahead_of_literal_heavy_nodes(
    tmp_path: Path,
) -> None:
    world_db = tmp_path / "relationship-first.sqlite3"
    store = WorldStore(world_db)
    busy_objects = [
        ObjectInput(
            id=f"busy-{index}",
            kind=ObjectKind.CONCEPT,
            canonical_name=f"Busy object {index}",
        )
        for index in range(10)
    ]
    assertions = [
        AssertionInput(
            id=f"busy-{index}-fact-{fact}",
            subject_id=f"busy-{index}",
            predicate=f"attribute_{fact}",
            literal=f"value {fact}",
            epistemic_role=EpistemicRole.AGENT_SYNTHESIS,
            confidence=0.9,
        )
        for index in range(10)
        for fact in range(2)
    ]
    store.memory_commit(
        CognitiveDelta(
            objects=[
                *busy_objects,
                ObjectInput(
                    id="quiet-source",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Quiet source",
                ),
                ObjectInput(
                    id="quiet-target",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Quiet target",
                ),
            ],
            assertions=[
                *assertions,
                AssertionInput(
                    id="quiet-relation",
                    subject_id="quiet-source",
                    predicate="connects_to",
                    object_id="quiet-target",
                    epistemic_role=EpistemicRole.AGENT_SYNTHESIS,
                    confidence=0.7,
                ),
            ],
        ),
        "agent:relationship-first",
    )

    snapshot = ReadOnlyInspection(world_db).graph_snapshot(limit=8)
    next_snapshot = ReadOnlyInspection(world_db).graph_snapshot(limit=8, window=1)

    assert {"quiet-source", "quiet-target"}.issubset({node["id"] for node in snapshot["nodes"]})
    assert [edge["id"] for edge in snapshot["edges"]] == ["quiet-relation"]
    assert snapshot["stats"]["displayed_nodes"] == 8
    assert snapshot["stats"]["total_nodes"] == 12
    assert snapshot["truncated"] is True
    assert snapshot["window"] == {"index": 0, "count": 2, "size": 8, "rotating": True}
    assert next_snapshot["window"] == {"index": 1, "count": 2, "size": 8, "rotating": True}
    displayed_across_windows = {node["id"] for view in (snapshot, next_snapshot) for node in view["nodes"]}
    assert displayed_across_windows == {
        "quiet-source",
        "quiet-target",
        *(f"busy-{index}" for index in range(10)),
    }


def test_graph_snapshot_rejects_negative_window(tmp_path: Path) -> None:
    world_db = tmp_path / "graph.sqlite3"
    _graph_world(world_db)

    with pytest.raises(ValueError, match="window"):
        ReadOnlyInspection(world_db).graph_snapshot(limit=8, window=-1)


def test_graph_snapshot_degrades_when_tables_lack_required_columns(tmp_path: Path) -> None:
    world_db = tmp_path / "partial-graph.sqlite3"
    with sqlite3.connect(world_db) as connection:
        connection.execute("CREATE TABLE objects (id TEXT PRIMARY KEY, canonical_name TEXT)")
        connection.execute("CREATE TABLE assertions (id TEXT PRIMARY KEY, subject_id TEXT)")
    before = world_db.read_bytes()

    snapshot = ReadOnlyInspection(world_db).graph_snapshot(limit=8)

    assert snapshot == {
        "available": False,
        "nodes": [],
        "edges": [],
        "truncated": False,
    }
    assert world_db.read_bytes() == before


def test_old_or_partial_schema_degrades_without_writes(tmp_path: Path) -> None:
    world_db = tmp_path / "partial.sqlite3"
    with sqlite3.connect(world_db) as connection:
        connection.execute("CREATE TABLE objects (id TEXT PRIMARY KEY, canonical_name TEXT, kind TEXT)")
        connection.execute("INSERT INTO objects VALUES ('one', 'Only object', 'topic')")
    before = world_db.read_bytes()

    reader = ReadOnlyInspection(world_db)
    assert reader.summary()["counts"]["objects"] == 1
    assert reader.search_objects()["available"] is False
    assert reader.inquiries()["available"] is False
    assert reader.recent_commits()["available"] is False
    assert world_db.read_bytes() == before


def test_run_inspection_uses_explicit_run_commit_and_thread_ledgers(tmp_path: Path) -> None:
    world_db = tmp_path / "world.sqlite3"
    store = _world(world_db)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="unrelated",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Unrelated",
                )
            ]
        ),
        "agent:thread-other",
    )
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO proposal_attempts("
            "attempt_id, commit_id, run_commit_id, durable_commit_id, attempt_no, parent_attempt_id, "
            "thread_id, attempted_at, outcome, new_objects, assertions, inquiries, "
            "omitted_assertions, omitted_inquiries, omitted_resolutions, resolved_inquiries, "
            "evidence_missing_assertions, error, delta_json, issues_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "agent:thread-one:attempt:1",
                "agent:thread-one",
                "agent:thread-one",
                "agent:thread-one",
                1,
                None,
                "thread-one",
                datetime.now(UTC).isoformat(),
                "committed",
                1,
                1,
                1,
                2,
                0,
                0,
                0,
                1,
                None,
                "{}",
                '[{"code":"seen_support"}]',
            ),
        )

    runtime_db = tmp_path / "runtime.sqlite3"
    with sqlite3.connect(runtime_db) as connection:
        connection.execute(
            "CREATE TABLE model_calls (id INTEGER PRIMARY KEY, thread_id TEXT, purpose TEXT, "
            "phase TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, "
            "cached_input_tokens INTEGER, cost_usd REAL, latency_ms REAL)"
        )
        connection.executemany(
            "INSERT INTO model_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "thread-one", "explore", "exploration", 10, 3, 4, 0.1, 20.0),
                (2, "thread-other", "explore", "exploration", 99, 99, 0, 9.9, 99.0),
            ],
        )

    inspection = ReadOnlyInspection(world_db, runtime_db).run_inspection("thread-one")

    assert inspection["model"]["calls"] == 1
    assert inspection["model"]["prompt_tokens"] == 10
    assert inspection["proposal_review"]["omissions"] == {
        "assertions": 2,
        "inquiries": 0,
        "resolutions": 0,
        "evidence_missing": 1,
    }
    commit_ids = [item["commit_id"] for item in inspection["durable_diff"]["commits"]]
    assert commit_ids == ["agent:thread-one"]
    assert {item["id"] for item in inspection["durable_diff"]["objects"]} == {"object-one"}
    assert "unrelated" not in str(inspection)


def test_run_inspection_uses_explicit_graph_shell_wake_finalize_receipt(tmp_path: Path) -> None:
    world_db = tmp_path / "world.sqlite3"
    store = _world(world_db)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="graph-object",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Graph Shell 对象",
                )
            ],
            assertions=[
                AssertionInput(
                    id="graph-assertion",
                    subject_id="graph-object",
                    predicate="has_state",
                    literal="已正式发布",
                    epistemic_role=EpistemicRole.AGENT_SYNTHESIS,
                    confidence=0.9,
                )
            ],
            inquiries=[
                InquiryInput(
                    id="graph-inquiry",
                    subject_id="graph-object",
                    prompt="下一次观察什么？",
                    rationale="保留后续入口",
                )
            ],
        ),
        "wake-graph:finalize",
    )

    inspection = ReadOnlyInspection(world_db).run_inspection("thread-graph", wake_id="wake-graph")

    assert inspection["wake_id"] == "wake-graph"
    assert [item["commit_id"] for item in inspection["durable_diff"]["commits"]] == ["wake-graph:finalize"]
    assert [item["id"] for item in inspection["writes"]["objects"]] == ["graph-object"]
    assert [item["id"] for item in inspection["writes"]["assertions"]] == ["graph-assertion"]
    assert inspection["writes"]["assertions"][0]["subject_name"] == "Graph Shell 对象"
    assert [item["id"] for item in inspection["writes"]["inquiries"]] == ["graph-inquiry"]
    assert "object-one" not in str(inspection["writes"])


def test_run_inspection_does_not_time_join_tool_commits(tmp_path: Path) -> None:
    world_db = tmp_path / "world.sqlite3"
    store = _world(world_db)
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="tool-observation",
                    source_uri="https://example.test/tool",
                    source_kind="public-web",
                    title="Tool material",
                    excerpt="not terminal cognition",
                    content_ref="",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime.now(UTC),
                )
            ]
        ),
        "tool:call-nearby",
    )

    inspection = ReadOnlyInspection(world_db).run_inspection("thread-one")

    assert {item["id"] for item in inspection["durable_diff"]["observations"]} == {"observation-one"}
    assert "tool-observation" not in str(inspection)


def test_object_detail_keeps_zero_evidence_and_supersede_history(tmp_path: Path) -> None:
    world_db = tmp_path / "world.sqlite3"
    store = _world(world_db)
    now = datetime(2026, 8, 14, tzinfo=UTC)
    store.memory_commit(
        CognitiveDelta(
            assertions=[
                AssertionInput(
                    id="assertion-no-evidence",
                    subject_id="object-one",
                    predicate="notes",
                    literal="a remembered but unsupported judgement",
                    epistemic_role=EpistemicRole.UNCERTAINTY,
                    confidence=0.2,
                ),
                AssertionInput(
                    id="assertion-two",
                    subject_id="object-one",
                    predicate="expresses",
                    literal="a revised judgement",
                    epistemic_role=EpistemicRole.COMMUNITY_VIEW,
                    confidence=0.4,
                    supersedes_id="assertion-one",
                    superseded_at=now,
                ),
            ]
        ),
        "agent:thread-two",
    )

    detail = ReadOnlyInspection(world_db).object_detail("object-one")

    assert detail is not None
    current = {item["id"]: item for item in detail["assertions"]}
    assert "assertion-one" not in current
    assert current["assertion-no-evidence"]["evidence"] == []
    assert current["assertion-two"]["supersedes_id"] == "assertion-one"
    assert current["assertion-two"]["superseded_at"]


def test_object_detail_is_bounded_and_malformed_metadata_fails_closed(tmp_path: Path) -> None:
    world_db = tmp_path / "world.sqlite3"
    store = _world(world_db)
    assertions = [
        AssertionInput(
            id=f"assertion-{number:03}",
            subject_id="object-one",
            predicate="notes",
            literal=f"judgement {number}",
            epistemic_role=EpistemicRole.UNCERTAINTY,
            confidence=0.1,
            evidence=[EvidenceInput(observation_id="observation-one", role="context")],
        )
        for number in range(50)
    ]
    inquiries = [
        InquiryInput(
            id=f"inquiry-{number:03}",
            subject_id="object-one",
            prompt=f"question {number}",
            rationale="bounded inspection test",
            created_at=datetime(2026, 8, 14, tzinfo=UTC),
        )
        for number in range(50)
    ]
    observations = [
        ObservationInput(
            id=f"linked-observation-{number:03}",
            source_uri=f"https://example.test/{number}",
            source_kind="public-web",
            title=f"linked material {number}",
            excerpt="bounded observation test",
            content_ref="",
            depth=ObservationDepth.SEEN,
            observed_at=datetime(2026, 8, 14, tzinfo=UTC),
        )
        for number in range(101)
    ]
    links = [
        ObservationLinkInput(
            target_kind="object",
            target_id="object-one",
            observation_id=item.id,
            role="context",
        )
        for item in observations
    ]
    store.memory_commit(
        CognitiveDelta(
            assertions=assertions,
            inquiries=inquiries,
            observations=observations,
            observation_links=links,
        ),
        "agent:many",
    )
    with store.write_connection() as connection:
        connection.execute("UPDATE observations SET metadata_json = '{not json' WHERE id = 'observation-one'")
    before = world_db.read_bytes()

    detail = ReadOnlyInspection(world_db).object_detail("object-one")

    assert detail is not None
    assert len(detail["assertions"]) == 40
    assert detail["assertions_truncated"] is True
    assert len(detail["inquiries"]) == 40
    assert detail["inquiries_truncated"] is True
    assert len(detail["observations"]) == 100
    assert detail["observations_truncated"] is True
    observation = detail["assertions"][0]["evidence"][0]["observation"]
    assert observation["material_reliability"] == "unknown"
    assert observation["limitations"] == []
    assert world_db.read_bytes() == before


def test_public_and_detail_text_bounds_are_explicit(tmp_path: Path) -> None:
    world_db = tmp_path / "world.sqlite3"
    store = _world(world_db)
    huge = "x" * 600
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE objects SET canonical_name = ?, domain_hints_json = ? WHERE id = 'object-one'",
            (huge, f'["{huge}"]'),
        )
        connection.execute(
            "UPDATE assertions SET literal_json = ? WHERE id = 'assertion-one'",
            (f'"{huge}"',),
        )
        connection.execute("UPDATE inquiries SET prompt = ? WHERE id = 'inquiry-one'", (huge,))
        connection.execute(
            "UPDATE observations SET title = ?, source_uri = ?, excerpt = ? WHERE id = 'observation-one'",
            (huge, huge, huge),
        )
        connection.execute(
            "INSERT INTO object_aliases(normalized_alias, object_id) VALUES (?, 'object-one')",
            (huge + "-alias",),
        )

    reader = ReadOnlyInspection(world_db)
    search = reader.search_objects()
    detail = reader.object_detail("object-one")
    public_inquiry = reader.inquiries(statuses=("open",))["items"][0]

    assert search["items"][0]["text_truncated"] is True
    assert len(search["items"][0]["canonical_name"]) == 500
    assert detail is not None
    assert detail["object"]["text_truncated"] is True
    # the huge legacy row now lives in its own read-only section, never in
    # the active identity aliases
    assert detail["aliases"] == ["community topic"]
    assert detail["aliases_truncated"] is False
    assert detail["legacy_names_truncated"] is True
    assert any(len(name) == 500 for name in detail["legacy_names"])
    assert detail["assertions"][0]["text_truncated"] is True
    assert len(detail["assertions"][0]["literal"]) == 500
    assert detail["inquiries"][0]["text_truncated"] is True
    assert len(public_inquiry["prompt"]) == 500
    observation = detail["observations"][0]
    assert {"title", "source_uri", "excerpt"}.issubset(observation["text_truncated_fields"])


def test_nested_json_at_projection_depth_is_replaced_and_marked(tmp_path: Path) -> None:
    world_db = tmp_path / "world.sqlite3"
    store = _world(world_db)
    huge = "x" * 600
    nested_literal = f'{{"a":{{"b":{{"c":{{"huge":"{huge}"}}}}}}}}'
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE assertions SET literal_json = ? WHERE id = 'assertion-one'",
            (nested_literal,),
        )

    detail = ReadOnlyInspection(world_db).object_detail("object-one")

    assert detail is not None
    assertion = detail["assertions"][0]
    assert assertion["text_truncated"] is True
    assert assertion["literal"] == {"a": {"b": {"c": {}}}}


def test_console_separates_identity_aliases_legacy_names_and_name_usage(
    tmp_path: Path,
) -> None:
    """Detail separates active identity aliases, legacy names and usage assertions."""
    world_db = tmp_path / "world.sqlite3"
    store = WorldStore(world_db)
    now = datetime(2026, 8, 1, tzinfo=UTC)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="hero",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Faker Lee",
                    aliases=["darkside"],
                )
            ],
            observations=[
                ObservationInput(
                    id="observation-one",
                    source_uri="https://example.test/one",
                    source_kind="public-web",
                    title="社区称呼材料",
                    excerpt="摘要",
                    content_ref="",
                    depth=ObservationDepth.CONTENT,
                    observed_at=now,
                    source_published_at=now,
                    metadata={},
                ),
                ObservationInput(
                    id="observation-two",
                    source_uri="https://example.test/two",
                    source_kind="public-web",
                    title="另一份材料",
                    excerpt="摘要二",
                    content_ref="",
                    depth=ObservationDepth.CONTENT,
                    observed_at=now,
                    source_published_at=now,
                    metadata={},
                ),
            ],
            assertions=[
                AssertionInput(
                    id="usage-1",
                    subject_id="hero",
                    predicate="name_usage",
                    literal="皮蛋",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    qualifiers={"community": "lpl", "language": "zh"},
                    event_time_start=now,
                    evidence=[
                        EvidenceInput(observation_id="observation-one", role="supports"),
                        EvidenceInput(observation_id="observation-two", role="context"),
                    ],
                )
            ],
        ),
        "identity-seed",
    )
    # legacy history is read-only: a pre-v10 object_aliases row for the same object
    with sqlite3.connect(world_db) as connection:
        connection.execute(
            "INSERT INTO object_aliases (object_id, normalized_alias) VALUES ('hero', '老粉称呼')"
        )

    reader = ReadOnlyInspection(world_db)
    detail = reader.object_detail("hero")

    assert detail is not None
    assert detail["aliases"] == ["darkside"]
    assert detail["legacy_names"] == ["老粉称呼"]
    assert detail["name_usages"] == [
        {
            "assertion_id": "usage-1",
            "literal": "皮蛋",
            "qualifiers": {"community": "lpl", "language": "zh"},
            "time": "2026-08-01T00:00:00+00:00",
            "evidence_counts": {"supports": 1, "context": 1, "contradicts": 0},
        }
    ]
    assert detail["aliases_truncated"] is False
    assert detail["legacy_names_truncated"] is False
    assert detail["name_usages_truncated"] is False

    # search labels the graded basis over identity, canonical and legacy surfaces
    assert reader.search_objects("darkside")["items"][0]["basis"] == "identity_alias_exact"
    assert reader.search_objects("Faker Lee")["items"][0]["basis"] == "canonical_exact"
    legacy_hit = reader.search_objects("老粉称呼")["items"][0]
    assert legacy_hit["id"] == "hero"
    assert legacy_hit["basis"] == "legacy_name"
    # a substring-only hit is a possible match, never an exact claim
    assert reader.search_objects("dark")["items"][0]["basis"] == "possible"
    # name_usage is an assertion summary surface, not a console search surface
    assert reader.search_objects("皮蛋")["items"] == []


def test_console_same_canonical_name_returns_all_objects(tmp_path: Path) -> None:
    """A shared canonical name returns every object, never just the first."""
    world_db = tmp_path / "world.sqlite3"
    store = WorldStore(world_db)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="acme-1",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Acme",
                    aliases=["alpha"],
                ),
                ObjectInput(
                    id="acme-2",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Acme",
                    aliases=["beta"],
                ),
            ]
        ),
        "shared-name",
    )

    reader = ReadOnlyInspection(world_db)
    items = reader.search_objects("acme")["items"]

    assert [item["id"] for item in items] == ["acme-1", "acme-2"]
    assert [item["basis"] for item in items] == ["canonical_exact", "canonical_exact"]
    # object detail never bleeds one object's aliases into the other
    first = reader.object_detail("acme-1")
    second = reader.object_detail("acme-2")
    assert first is not None and second is not None
    assert first["aliases"] == ["alpha"]
    assert second["aliases"] == ["beta"]


def test_console_reads_unmigrated_v9_snapshot_without_writing(tmp_path: Path) -> None:
    """A v9 snapshot without identity_aliases degrades; inspection never migrates."""
    world_db = tmp_path / "v9.sqlite3"
    connection = sqlite3.connect(world_db)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_V1_SQL)
    _apply_migration_v2(connection)
    connection.execute("PRAGMA user_version = 1")
    _apply_migration_v3(connection)
    connection.execute("PRAGMA user_version = 2")
    connection.executescript(MIGRATION_V4_SQL)
    connection.execute("PRAGMA user_version = 3")
    connection.executescript(MIGRATION_V5_SQL)
    connection.execute("PRAGMA user_version = 4")
    connection.executescript(MIGRATION_V6_SQL)
    connection.execute("PRAGMA user_version = 5")
    _apply_migration_v7(connection)
    connection.execute("PRAGMA user_version = 6")
    _apply_migration_v7_attempt_ledger(connection)
    _apply_migration_v8_inquiry_touch_ledger(connection)
    _apply_migration_v9(connection)
    connection.execute("INSERT INTO predicates(name) VALUES ('name_usage')")
    connection.execute(
        "INSERT INTO objects(id, kind, canonical_name, domain_hints_json, provisional,"
        " event_time_start, event_time_end, version)"
        " VALUES ('old-obj', 'entity', 'Old Channel', '[]', 0, NULL, NULL, 1)"
    )
    connection.execute("INSERT INTO object_aliases(object_id, normalized_alias) VALUES ('old-obj', '老频道')")
    connection.execute(
        "INSERT INTO observations(id, source_uri, source_kind, title, excerpt, content_ref,"
        " depth, source_published_at, observed_at, metadata_json)"
        " VALUES ('obs-v9', 'https://example.test/v9', 'public-web', '材料', '摘要', '',"
        " 'content', '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00', '{}')"
    )
    connection.execute(
        "INSERT INTO assertions(id, signature, subject_id, predicate, object_id, literal_json,"
        " epistemic_role, confidence, event_time_start, event_time_end, supersedes_id)"
        " VALUES ('usage-v9', 'sig-usage-v9', 'old-obj', 'name_usage', NULL, '\"皮蛋\"',"
        " 'fact', 0.8, '2026-08-01T00:00:00+00:00', NULL, NULL)"
    )
    connection.execute(
        "INSERT INTO assertion_evidence(assertion_id, observation_id, role, linked_at)"
        " VALUES ('usage-v9', 'obs-v9', 'supports', '2026-08-01T00:00:00+00:00')"
    )
    connection.commit()
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 9
    connection.close()
    before = world_db.read_bytes()

    reader = ReadOnlyInspection(world_db)
    canonical = reader.search_objects("Old Channel")["items"]
    legacy = reader.search_objects("老频道")["items"]
    detail = reader.object_detail("old-obj")

    assert canonical[0]["id"] == "old-obj"
    assert canonical[0]["basis"] == "canonical_exact"
    assert legacy[0]["id"] == "old-obj"
    assert legacy[0]["basis"] == "legacy_name"
    # without the identity table no identity claim is ever labeled
    assert legacy[0]["basis"] != "identity_alias_exact"
    assert detail is not None
    assert detail["aliases"] == []
    assert detail["legacy_names"] == ["老频道"]
    assert detail["name_usages"] == [
        {
            "assertion_id": "usage-v9",
            "literal": "皮蛋",
            "qualifiers": {},
            "time": "2026-08-01T00:00:00+00:00",
            "evidence_counts": {"supports": 1, "context": 0, "contradicts": 0},
        }
    ]
    # inspection opened the snapshot read-only: bytes, schema and version are untouched
    assert world_db.read_bytes() == before
    with sqlite3.connect(world_db) as probe:
        assert probe.execute("PRAGMA user_version").fetchone()[0] == 9
        tables = {str(row[0]) for row in probe.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "identity_aliases" not in tables
