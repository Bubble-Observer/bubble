"""G5b-1 Graph Shell protocol and vertical routing tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from leave_information_bubble.gateway.client import (
    NativeToolCall,
    StructuredModelOutputError,
    ToolModelResponse,
)
from leave_information_bubble.gateway.deepseek_client import DeepSeekClient
from leave_information_bubble.runtime.errors import AgentError, ErrorCode
from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    InquiryInput,
    ObjectInput,
    ObjectKind,
    WorldStore,
    WorldTools,
)
from leave_information_bubble.world.finalize import FinalizeReceipt, finalize_graph
from leave_information_bubble.world.staging import apply_patch, read_active_staged
from leave_information_bubble.world.store import CommitReplayConflict
from leave_information_bubble.world_agent import graph as graph_module
from leave_information_bubble.world_agent.graph import build_world_agent_graph
from leave_information_bubble.world_agent.live_deadline import LiveDeadlineExceeded
from leave_information_bubble.world_agent.run_report import build_run_report
from tests.world_agent._graph_helpers import _response, _ScriptedModel


def _shell_graph(
    tmp_path: Path,
    model: _ScriptedModel,
    *,
    max_turns: int = 8,
    max_cost_usd: float | None = None,
    store: WorldStore | None = None,
    wake_id: str = "wake-shell",
):
    store = store or WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="thread-shell",
        wake_id=wake_id,
        closed_wake_guard=True,
    )
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="thread-shell",
        wake_id=wake_id,
        max_turns=max_turns,
        max_cost_usd=max_cost_usd,
    )
    return graph, store


def test_graph_shell_error_bounds_text_and_marks_every_cut() -> None:
    result = graph_module._graph_shell_error(
        "invalid_arguments",
        field="arguments",
        message="m" * 3_000,
        action_hint="h" * 3_000,
        violations=[
            {
                "code": "invalid_arguments",
                "field": f"arguments.f{index}",
                "message": "bad",
            }
            for index in range(40)
        ],
        total_violations=40,
        violations_truncated=True,
    )["error"]

    assert result["message_truncated"] is True
    assert result["action_hint_truncated"] is True
    assert len(result["violations"]) == 32
    assert result["total_violations"] == 40
    assert result["violations_truncated"] is True


def test_cross_field_argument_feedback_names_ordering_violations() -> None:
    search = graph_module._semantic_argument_violations(
        "memory_search",
        {
            "assertion_count_min": 5,
            "assertion_count_max": 2,
            "time_from": "2026-08-24T12:00:00Z",
            "time_to": "2026-08-24T11:00:00Z",
        },
    )
    scan = graph_module._semantic_argument_violations(
        "discover_sources",
        {
            "window_start": "2026-08-24T12:00:00Z",
            "window_end": "2026-08-24T11:00:00Z",
        },
    )

    assert {item["field"] for item in search} == {
        "arguments.assertion_count_min",
        "arguments.time_to",
    }
    assert [item["field"] for item in scan] == ["arguments.window_end"]


def test_cross_field_time_checks_defer_mixed_timezone_inputs_to_schema_feedback() -> None:
    """A schema-invalid naive endpoint must not crash semantic aggregation."""
    assert (
        graph_module._semantic_argument_violations(
            "memory_search",
            {
                "time_from": "2026-08-24T12:00:00Z",
                "time_to": "2026-08-24T13:00:00",
            },
        )
        == []
    )
    assert (
        graph_module._semantic_argument_violations(
            "discover_sources",
            {
                "window_start": "2026-08-24T12:00:00Z",
                "window_end": "2026-08-24T13:00:00",
            },
        )
        == []
    )


def test_long_tool_error_signatures_digest_the_complete_field_set() -> None:
    common = [f"arguments.{index:02d}.{'x' * 20}" for index in range(10)]

    def result(last_field: str) -> dict[str, object]:
        return {
            "ok": False,
            "error": {
                "code": "invalid_arguments",
                "violations": [
                    *(
                        {"field": field, "message": "bad"}
                        for field in common
                    ),
                    {"field": last_field, "message": "bad"},
                ],
            },
        }

    first = graph_module._tool_error_signature("graph_inspect", result("arguments.tail.a"))
    second = graph_module._tool_error_signature("graph_inspect", result("arguments.tail.b"))

    assert first != second
    assert first is not None and len(first) < 300


async def test_scenario_a_full_graph_shell_loop_and_next_wake_formal_read(tmp_path) -> None:
    first_model = _ScriptedModel(
        [
            _response(
                "Search formal memory.",
                NativeToolCall(
                    id="search",
                    name="memory_search",
                    arguments={"query": "Known Anchor", "limit": 5},
                ),
            ),
            _response(
                "Read the known anchor.",
                NativeToolCall(
                    id="read",
                    name="memory_read",
                    arguments={"object_id": "known-anchor"},
                ),
            ),
            _response(
                "Stage a new provisional concept.",
                NativeToolCall(
                    id="patch",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-a1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Scenario A Concept",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Connect the staged concept to formal memory.",
                NativeToolCall(
                    id="patch-assertion",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-a2",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "wake-a-first:s1",
                                    "predicate": "related_to",
                                    "object_ref": "known-anchor",
                                    "epistemic_role": "fact",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Inspect readiness.",
                NativeToolCall(id="inspect", name="graph_inspect", arguments={}),
            ),
            _response(
                "Review the staged delta.",
                NativeToolCall(id="diff", name="graph_diff", arguments={}),
            ),
            _response(
                "Publish explicitly.",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            ),
        ]
    )
    first_graph, store = _shell_graph(
        tmp_path,
        first_model,
        wake_id="wake-a-first",
    )
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="known-anchor",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Known Anchor",
                )
            ]
        ),
        "seed-known",
    )

    first = await first_graph.ainvoke({"messages": [{"role": "user", "content": "Run scenario A."}]})
    assert first["terminal_status"] == "published"
    assert first["finalize_receipt"]["commit_id"] == "wake-a-first:finalize"
    assert first["wake_id"] == "wake-a-first"

    # F-06 A: the tool results are load-bearing, not decorative — search/read
    # returned the existing formal object, inspect shows the staged id and
    # readiness, diff shows the formal-before / staged-after delta explicitly
    # marked unpublished, and the formal commit happens only at finalize.
    search_memory = json.loads(first_model.requests[1][-1]["content"])["memory"]
    assert any(
        card.get("id") == "known-anchor" and card.get("canonical_name") == "Known Anchor"
        for card in search_memory["anchor_objects"]
    )
    read_memory = json.loads(first_model.requests[2][-1]["content"])["memory"]
    assert read_memory["identity"]["object_id"] == "known-anchor"
    assert read_memory["identity"]["canonical_name"] == "Known Anchor"

    inspect_result = json.loads(first_model.requests[5][-1]["content"])
    inspect_payload = inspect_result["payload"]
    assert inspect_result["scope"]["readiness"] == "ready"
    assert inspect_payload["readiness"] == "ready"
    assert inspect_payload["active_total"] == 2
    assert inspect_payload["stats"]["objects"]["active"] == 1
    assert inspect_payload["stats"]["assertions"]["active"] == 1
    staged_item = next(item for item in inspect_payload["items"] if item["staged_id"] == "wake-a-first:s1")
    assert staged_item["kind"] == "object"
    assert staged_item["action"] == "created"
    assert staged_item["label"] == "Scenario A Concept"

    diff_result = json.loads(first_model.requests[6][-1]["content"])
    diff_payload = diff_result["payload"]
    assert diff_result["scope"]["published"] is False
    assert diff_payload["published"] is False
    assert diff_payload["summary"]["total"] == 2
    created_object = next(
        entry for entry in diff_payload["entries"] if entry["staged_id"] == "wake-a-first:s1"
    )
    assert created_object["action"] == "create"
    assert created_object["before"] is None
    assert created_object["after"]["canonical_name"] == "Scenario A Concept"
    created_assertion = next(entry for entry in diff_payload["entries"] if entry["kind"] == "assertion")
    assert created_assertion["after"]["subject_ref"] == "wake-a-first:s1"
    assert created_assertion["after"]["object_ref"] == "known-anchor"

    # Exactly one formal commit per wake: staged rows converge to finalized
    # and the receipt ledger carries a single finalize/commit pair.
    with store.read_connection() as connection:
        formal_row = connection.execute(
            "SELECT canonical_name FROM objects WHERE id = 'wake-a-first:s1'"
        ).fetchone()
        assert formal_row is not None and formal_row["canonical_name"] == "Scenario A Concept"
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = 'wake-a-first'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = 'wake-a-first:finalize'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM staged_objects WHERE wake_id = 'wake-a-first' AND status = 'finalized'"
            ).fetchone()[0]
            == 1
        )

    second_model = _ScriptedModel(
        [
            _response(
                "Read the prior wake's formal result.",
                NativeToolCall(
                    id="read-formal",
                    name="memory_read",
                    arguments={"object_id": "wake-a-first:s1"},
                ),
            ),
            _response(
                "Expand the prior wake's formal edge.",
                NativeToolCall(
                    id="expand-formal",
                    name="memory_expand",
                    arguments={"object_ids": ["wake-a-first:s1"], "depth": 1, "limit": 10},
                ),
            ),
            _response(
                "Close the new wake explicitly.",
                NativeToolCall(id="finalize-second", name="finalize_graph", arguments={}),
            ),
        ]
    )
    second_graph, _ = _shell_graph(
        tmp_path,
        second_model,
        store=store,
        wake_id="wake-a-second",
    )

    second = await second_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Read the previous wake."}]}
    )

    formal_read = json.loads(second_model.requests[1][-1]["content"])["memory"]
    assert formal_read["identity"]["canonical_name"] == "Scenario A Concept"
    assert formal_read["identity"]["object_id"] == "wake-a-first:s1"
    formal_expansion = json.loads(second_model.requests[2][-1]["content"])["memory"]
    assert [
        (item["subject"]["id"], item["predicate"], item["object"]["id"])
        for item in formal_expansion["assertions"]
    ] == [("wake-a-first:s1", "related_to", "known-anchor")]
    assert second["wake_id"] == "wake-a-second"
    assert second["terminal_status"] == "published"
    # two fresh wakes on the same thread mint distinct identities and commits
    assert second["finalize_receipt"]["commit_id"] == "wake-a-second:finalize"
    assert second["wake_id"] != first["wake_id"]
    assert second["finalize_receipt"]["commit_id"] != first["finalize_receipt"]["commit_id"]


async def test_graph_shell_topology_and_tool_surface_are_legacy_exclusive(tmp_path) -> None:
    model = _ScriptedModel([_response("pause"), _response("pause")])
    graph, _store = _shell_graph(tmp_path, model)

    node_names = set(graph.get_graph().nodes)
    assert {"bootstrap", "agent", "tools", "completion_reminder", "stage_unpublished"} <= node_names
    assert not node_names.intersection(
        {
            "proposal_validation",
            "proposal_finalization",
            "consolidation",
            "proposal_repair",
            "commit",
            "recovery",
        }
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Open a Graph Shell wake."}]})
    first_surface = model.tool_names[0]
    assert "graph_patch" in first_surface
    assert "graph_inspect" in first_surface
    assert "graph_diff" in first_surface
    assert "finalize_graph" in first_surface
    assert "submit_cognition" not in first_surface
    assert "propose_inquiry" not in first_surface
    assert result["terminal_status"] == "staged_unpublished"


async def test_graph_shell_execution_rejects_hidden_inquiry_tool_without_side_effect(
    tmp_path,
) -> None:
    model = _ScriptedModel(
        [
            _response(
                "hallucinate a hidden inquiry tool",
                NativeToolCall(
                    id="hidden-inquiry",
                    name="propose_inquiry",
                    arguments={
                        "prompt": "This must not enter an incomplete lifecycle.",
                        "reason": "Graph Shell does not expose inquiry lifecycle in G5b-1.",
                    },
                ),
            ),
            _response(
                "finish without inquiry mutation",
                NativeToolCall(id="finish", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Do not mutate."}]})

    hidden_event = next(event for event in result["tool_events"] if event["call_id"] == "hidden-inquiry")
    assert hidden_event["diagnostic"]["error"]["code"] == "tool_not_available_in_mode"
    assert result["terminal_status"] == "published"
    assert read_active_staged(store, "wake-shell") == {
        "objects": [],
        "assertions": [],
        "inquiries": [],
    }
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = 'wake-shell'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("recoverable", [False, True])
async def test_graph_shell_reports_original_agent_error_recoverability(tmp_path, recoverable: bool) -> None:
    model = _ScriptedModel(
        [
            AgentError(
                ErrorCode.MODEL_TRANSIENT,
                "provider boundary failed",
                recoverable=recoverable,
            )
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Run."}]})
    report = build_run_report(
        thread_id="thread-shell",
        wake_id="wake-shell",
        domain="lol_cn",
        mission="report provider failure",
        mode="broad",
        object_id=None,
        result=result,
        store=store,
        runtime_path=tmp_path / "runtime.sqlite3",
        wall_ms=1,
    )

    expected = {
        "type": "AgentError",
        "code": "MODEL_TRANSIENT",
        "recoverable": recoverable,
        "message": "provider boundary failed",
    }
    assert result["raw_exception"] == expected
    assert report["execution"]["raw_exception"] == expected
    assert result["terminal_status"] == "staged_unpublished"


async def test_default_surface_is_graph_shell_and_submit_is_unavailable(tmp_path) -> None:
    """D3: the default graph's tool surface is Graph Shell only.

    submit_cognition is gone, finalize_graph is the only publish tool, and a
    hallucinated submit call is rejected with the typed
    ``tool_not_available_in_mode`` error — no legacy fallback executes.
    """
    store = WorldStore(tmp_path / "shell-surface.sqlite3")
    model = _ScriptedModel(
        [
            _response(
                "Stage one concept.",
                NativeToolCall(
                    id="patch-surface",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-surface",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Surface Concept",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Hallucinated legacy submit.",
                NativeToolCall(
                    id="submit-surface",
                    name="submit_cognition",
                    arguments={"schema_version": "1"},
                ),
            ),
            _response(
                "Publish the staged concept.",
                NativeToolCall(id="finalize-surface", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph = build_world_agent_graph(
        model=model,
        tools=WorldTools(store=store, adapters={}, thread_id="shell-thread", wake_id="shell-surface"),
        store=store,
        thread_id="shell-thread",
        wake_id="shell-surface",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Run the shell."}]})

    assert "finalize_graph" in model.tool_names[0]
    assert {"graph_patch", "graph_inspect", "graph_diff"} <= set(model.tool_names[0])
    assert "submit_cognition" not in model.tool_names[0]
    submit_event = next(event for event in result["tool_events"] if event.get("name") == "submit_cognition")
    assert submit_event["ok"] is False
    assert submit_event["diagnostic"]["error"]["code"] == "tool_not_available_in_mode"
    assert result["terminal_status"] == "published"
    assert result["finalize_receipt"]["commit_id"] == "shell-surface:finalize"
    with store.read_connection() as connection:
        assert (
            connection.execute(
                "SELECT canonical_name FROM objects WHERE canonical_name = 'Surface Concept'"
            ).fetchone()
            is not None
        )


async def test_graph_shell_patch_inspect_finalize_publishes_and_stops_model(tmp_path) -> None:
    model = _ScriptedModel(
        [
            _response(
                "Stage one provisional concept.",
                NativeToolCall(
                    id="patch-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Graph Shell Concept",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Inspect before publication.",
                NativeToolCall(id="inspect-1", name="graph_inspect", arguments={}),
            ),
            _response(
                "Publish explicitly.",
                NativeToolCall(id="finalize-1", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Preserve one concept."}]})

    assert result["terminal_status"] == "published"
    assert result["finalize_receipt"]["commit_id"] == "wake-shell:finalize"
    assert len(model.requests) == 3
    with store.read_connection() as connection:
        assert (
            connection.execute(
                "SELECT canonical_name FROM objects WHERE canonical_name = 'Graph Shell Concept'"
            ).fetchone()
            is not None
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = 'wake-shell'"
            ).fetchone()[0]
            == 1
        )


async def test_graph_shell_blocked_feedback_can_be_repaired_then_published(tmp_path) -> None:
    model = _ScriptedModel(
        [
            _response(
                "Stage an unconnected object.",
                NativeToolCall(
                    id="patch-object",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {"canonical_name": "Loner", "kind": "entity"},
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Try publication and read the blocker.",
                NativeToolCall(id="finalize-blocked", name="finalize_graph", arguments={}),
            ),
            _response(
                "Connect the staged object.",
                NativeToolCall(
                    id="patch-link",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-2",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "wake-shell:s1",
                                    "predicate": "related_to",
                                    "object_ref": "formal-anchor",
                                    "epistemic_role": "fact",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Inspect the repaired graph.",
                NativeToolCall(id="inspect-ready", name="graph_inspect", arguments={}),
            ),
            _response(
                "Publish the repaired graph.",
                NativeToolCall(id="finalize-ready", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="formal-anchor",
                    kind=ObjectKind.EVENT,
                    canonical_name="Formal Anchor",
                )
            ]
        ),
        "seed-anchor",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Build."}]})

    blocked = json.loads(model.requests[2][-1]["content"])["payload"]
    assert blocked["status"] == "blocked"
    assert [entry["code"] for entry in blocked["blockers"]] == ["zero_connection_object"]

    # F-06 B: after the real repair patch the host explicitly inspects ready
    # before finalizing, and the whole blocked-then-repaired process produces
    # exactly one formal commit/receipt — the failed attempt leaves nothing.
    repaired = json.loads(model.requests[4][-1]["content"])
    assert repaired["payload"]["readiness"] == "ready"
    assert repaired["payload"]["blockers"] == []
    assert repaired["scope"]["readiness"] == "ready"

    assert result["terminal_status"] == "published"
    assert len(model.requests) == 5
    with store.read_connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = 'wake-shell'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = 'wake-shell:finalize'"
            ).fetchone()[0]
            == 1
        )
        # the blocker's item was revised through the real patch path (op-2 in
        # the ledger), converged to finalized, and the link committed formally
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = 'wake-shell' AND op_id = 'op-2'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT status FROM staged_objects"
                " WHERE wake_id = 'wake-shell' AND staged_id = 'wake-shell:s1'"
            ).fetchone()[0]
            == "finalized"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM assertions"
                " WHERE subject_id = 'wake-shell:s1' AND predicate = 'related_to'"
                " AND object_id = 'formal-anchor'"
            ).fetchone()[0]
            == 1
        )


async def test_graph_shell_staged_delta_formally_invisible_until_finalize(tmp_path) -> None:
    """F-06 A: staged work is durable and inspectable but never formal early.

    The wake stops before finalize_graph: the staged rows exist with status
    'active' and graph_inspect reports the delta ready, while the formal
    tables hold nothing and no finalize receipt exists — a fresh wake reading
    the staged id gets an unknown-object bundle (formal visibility starts at
    finalize, exactly once per wake).
    """
    model = _ScriptedModel(
        [
            _response(
                "Stage a new provisional concept.",
                NativeToolCall(
                    id="patch-object",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Draft Concept",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Connect it to formal memory.",
                NativeToolCall(
                    id="patch-assertion",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-2",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "wake-shell:s1",
                                    "predicate": "related_to",
                                    "object_ref": "known-anchor",
                                    "epistemic_role": "fact",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Inspect the working graph.",
                NativeToolCall(id="inspect", name="graph_inspect", arguments={}),
            ),
            _response("Pause, keep the draft."),
            _response("Pause again."),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="known-anchor",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Known Anchor",
                )
            ]
        ),
        "seed-known",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Draft without publishing."}]})
    assert result["terminal_status"] == "staged_unpublished"

    inspect_payload = json.loads(model.requests[3][-1]["content"])["payload"]
    assert inspect_payload["readiness"] == "ready"
    staged_item = next(item for item in inspect_payload["items"] if item["staged_id"] == "wake-shell:s1")
    assert staged_item["action"] == "created"
    assert staged_item["label"] == "Draft Concept"

    with store.read_connection() as connection:
        staged = connection.execute(
            "SELECT status FROM staged_objects WHERE wake_id = 'wake-shell' AND staged_id = 'wake-shell:s1'"
        ).fetchone()
        assert staged is not None and staged["status"] == "active"
        assert connection.execute("SELECT 1 FROM objects WHERE id = 'wake-shell:s1'").fetchone() is None
        assert (
            connection.execute("SELECT 1 FROM assertions WHERE subject_id = 'wake-shell:s1'").fetchone()
            is None
        )
        assert connection.execute("SELECT COUNT(*) FROM finalize_receipts").fetchone()[0] == 0

    fresh_model = _ScriptedModel(
        [
            _response(
                "Read the other wake's draft.",
                NativeToolCall(
                    id="read-draft",
                    name="memory_read",
                    arguments={"object_id": "wake-shell:s1"},
                ),
            ),
            _response("Pause."),
            _response("Pause again."),
        ]
    )
    fresh_graph, _ = _shell_graph(tmp_path, fresh_model, store=store, wake_id="wake-shell-2")
    await fresh_graph.ainvoke({"messages": [{"role": "user", "content": "Read the draft."}]})
    read_memory = json.loads(fresh_model.requests[1][-1]["content"])["memory"]
    assert read_memory["reasons"] == ["unknown object"]
    assert read_memory["identity"] is None


@pytest.mark.parametrize("status", ["compile_failed", "commit_rejected"])
async def test_graph_shell_recoverable_finalize_status_returns_to_model(
    tmp_path, monkeypatch, status: str
) -> None:
    real_finalize = graph_module.finalize_graph
    attempts = 0

    def scripted_finalize(store: WorldStore, wake_id: str) -> FinalizeReceipt:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return FinalizeReceipt(
                wake_id=wake_id,
                status=status,
                problems=[f"simulated {status}"],
            )
        return real_finalize(store, wake_id)

    monkeypatch.setattr(graph_module, "finalize_graph", scripted_finalize)
    model = _ScriptedModel(
        [
            _response("First finalize.", NativeToolCall(id="f-1", name="finalize_graph", arguments={})),
            _response("Retry after feedback.", NativeToolCall(id="f-2", name="finalize_graph", arguments={})),
        ]
    )
    graph, _store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Finalize."}]})

    feedback = json.loads(model.requests[1][-1]["content"])["payload"]
    assert feedback["status"] == status
    assert result["terminal_status"] == "published"
    assert attempts == 2


async def test_graph_shell_rejects_inverted_interval_then_publishes_corrected_patch(tmp_path) -> None:
    # A cross-field error that can be identified from one patch must be
    # rejected before staging.  The model receives the exact field and can
    # submit a corrected item without a cleanup/drop round trip.
    model = _ScriptedModel(
        [
            _response(
                "stage an assertion with an inverted event_time interval",
                NativeToolCall(
                    id="bad-patch",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "bad-op",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "compile-subject",
                                    "predicate": "related_to",
                                    "object_ref": "compile-object",
                                    "epistemic_role": "fact",
                                    "event_time_start": "2026-08-23T00:00:00Z",
                                    "event_time_end": "2026-08-22T00:00:00Z",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "stage the corrected assertion",
                NativeToolCall(
                    id="corrected-patch",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "corrected-op",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "compile-subject",
                                    "predicate": "related_to",
                                    "object_ref": "compile-object",
                                    "epistemic_role": "fact",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "publish corrected graph",
                NativeToolCall(id="finish", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="compile-subject",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Compile subject",
                ),
                ObjectInput(
                    id="compile-object",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Compile object",
                ),
            ]
        ),
        "seed-compile",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Repair compile."}]})

    rejected = json.loads(model.requests[1][-1]["content"])["error"]
    assert rejected["code"] == "invalid_tool_arguments"
    assert rejected["field"] == "items[0].payload.event_time_end"
    assert rejected["total_violations"] == 1
    assert "must not precede event_time_start" in rejected["message"]
    assert result["terminal_status"] == "published"
    assert read_active_staged(store, "wake-shell")["assertions"] == []
    with store.read_connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM assertions WHERE predicate = 'related_to'").fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = 'wake-shell'"
            ).fetchone()[0]
            == 1
        )


async def test_graph_shell_actual_commit_rejection_is_revised_and_published(tmp_path, monkeypatch) -> None:
    model = _ScriptedModel(
        [
            _response(
                "stage a candidate alias",
                NativeToolCall(
                    id="alias-patch",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "alias-op",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Alpha",
                                    "kind": "entity",
                                    "aliases": ["Shared"],
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "observe store refusal",
                NativeToolCall(id="alias-finalize", name="finalize_graph", arguments={}),
            ),
            _response(
                "withdraw the contested alias",
                NativeToolCall(
                    id="alias-revise",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "alias-revise-op",
                                "kind": "object",
                                "action": "update",
                                "target_ref": "wake-shell:s1",
                                "payload": {
                                    "canonical_name": "Alpha",
                                    "kind": "entity",
                                    "aliases": [],
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "publish revised graph",
                NativeToolCall(id="finish", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    real_commit = store.finalized_memory_commit
    injected = False

    def conflict_once(delta, commit_id, finalizer):
        nonlocal injected
        if not injected:
            injected = True
            store.memory_commit(
                CognitiveDelta(
                    objects=[
                        ObjectInput(
                            id="concurrent-owner",
                            kind=ObjectKind.ENTITY,
                            canonical_name="Concurrent owner",
                            aliases=["Shared"],
                        )
                    ]
                ),
                "concurrent-alias-owner",
            )
        return real_commit(delta, commit_id, finalizer)

    monkeypatch.setattr(store, "finalized_memory_commit", conflict_once)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Repair store."}]})

    rejected = json.loads(model.requests[2][-1]["content"])["payload"]
    assert rejected["status"] == "commit_rejected"
    assert rejected["stats"]["total_items"] == 1
    assert any("alias" in problem.lower() for problem in rejected["problems"])
    assert result["terminal_status"] == "published"
    assert read_active_staged(store, "wake-shell")["objects"] == []
    with store.read_connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM objects WHERE canonical_name = 'Alpha'").fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = 'wake-shell:finalize'"
            ).fetchone()[0]
            == 1
        )


async def test_graph_shell_malformed_json_and_bad_finalize_arguments_are_repairable(tmp_path) -> None:
    malformed = ToolModelResponse(
        content="bad json",
        message={
            "role": "assistant",
            "content": "bad json",
            "tool_calls": [
                {
                    "id": "bad-json",
                    "type": "function",
                    "function": {"name": "graph_patch", "arguments": "{not-json"},
                }
            ],
        },
        tool_calls=[],
    )
    model = _ScriptedModel(
        [
            malformed,
            _response(
                "bad finalize args",
                NativeToolCall(id="bad-finalize", name="finalize_graph", arguments={"force": True}),
            ),
            _response("fixed", NativeToolCall(id="good-finalize", name="finalize_graph", arguments={})),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Repair calls."}]})

    malformed_error = json.loads(model.requests[1][-1]["content"])["error"]
    argument_error = json.loads(model.requests[2][-1]["content"])["error"]
    assert malformed_error == {
        "code": "invalid_tool_arguments_json",
        "message": "Malformed JSON arguments for tool 'graph_patch'",
        "candidates": [],
        "action_hint": "Return one valid JSON object matching this tool schema",
        "field": "arguments",
    }
    assert argument_error["code"] == "invalid_arguments"
    assert argument_error["field"] == "arguments.force"
    assert result["terminal_status"] == "published"
    assert read_active_staged(store, "wake-shell")["objects"] == []


async def test_graph_shell_schema_rejects_unknown_fields_before_dispatch(tmp_path) -> None:
    """F-03: arguments are validated against the advertised schema pre-dispatch.

    graph_inspect({"bogus": true}) must be rejected as invalid_arguments
    before any implementation runs; a corrected call and finalize continue.
    """
    model = _ScriptedModel(
        [
            _response(
                "inspect with an unknown field",
                NativeToolCall(
                    id="inspect-bogus",
                    name="graph_inspect",
                    arguments={"bogus": True},
                ),
            ),
            _response(
                "inspect with an unknown field on a read tool too",
                NativeToolCall(
                    id="read-bogus",
                    name="memory_read",
                    arguments={"object_id": "x", "bogus": True},
                ),
            ),
            _response(
                "inspect correctly",
                NativeToolCall(id="inspect-clean", name="graph_inspect", arguments={}),
            ),
            _response(
                "finalize",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Fix calls."}]})

    inspect_error = json.loads(model.requests[1][-1]["content"])["error"]
    assert inspect_error["code"] == "invalid_arguments"
    assert inspect_error["field"] == "arguments.bogus"
    assert "bogus" in inspect_error["message"]
    assert "advertised schema" in inspect_error["action_hint"]
    read_error = json.loads(model.requests[2][-1]["content"])["error"]
    assert read_error["code"] == "invalid_arguments"
    assert "bogus" in read_error["message"]
    assert result["terminal_status"] == "published"
    assert read_active_staged(store, "wake-shell")["objects"] == []
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM finalize_receipts").fetchone()[0] == 1


async def test_graph_shell_schema_feedback_reports_all_independent_fields(tmp_path) -> None:
    model = _ScriptedModel(
        [
            _response(
                "two invalid fields",
                NativeToolCall(
                    id="inspect-multi",
                    name="graph_inspect",
                    arguments={"item_id": 7, "bogus": True},
                ),
            ),
            _response("stop"),
        ]
    )
    graph, _ = _shell_graph(tmp_path, model, max_turns=2)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Validate once."}]})

    error = json.loads(model.requests[1][-1]["content"])["error"]
    assert error["total_violations"] == 2
    assert error.get("violations_truncated") is not True
    assert {item["field"] for item in error["violations"]} == {
        "arguments.bogus",
        "arguments.item_id",
    }
    assert result["terminal_status"] == "staged_unpublished"


async def test_schema_feedback_splits_every_unknown_property(tmp_path) -> None:
    model = _ScriptedModel(
        [
            _response(
                "many unknown fields",
                NativeToolCall(
                    id="inspect-many-unknown",
                    name="graph_inspect",
                    arguments={f"bad{index}": True for index in range(5)},
                ),
            ),
            _response("stop"),
        ]
    )
    graph, _ = _shell_graph(tmp_path, model, max_turns=2)

    await graph.ainvoke({"messages": [{"role": "user", "content": "Validate once."}]})

    error = json.loads(model.requests[1][-1]["content"])["error"]
    assert error["total_violations"] == 5
    assert {item["field"] for item in error["violations"]} == {
        f"arguments.bad{index}" for index in range(5)
    }
    assert error.get("violations_truncated") is not True


async def test_error_streak_distinguishes_corrected_argument_paths(tmp_path) -> None:
    model = _ScriptedModel(
        [
            _response(
                "bad one",
                NativeToolCall(id="bad-1", name="graph_inspect", arguments={"bogus": True}),
            ),
            _response(
                "different bad field",
                NativeToolCall(id="bad-2", name="graph_diff", arguments={"limit": "many"}),
            ),
            _response("stop"),
        ]
    )
    graph, _ = _shell_graph(tmp_path, model, max_turns=3)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Repair paths differ."}]})

    assert len(model.requests) == 3
    assert list(result["tool_error_streaks"].values()) == [1]
    assert "Tool error loop limit reached" not in result["terminal_summary"]


async def test_empty_working_graph_stops_without_completion_reminder(tmp_path) -> None:
    model = _ScriptedModel([_response("nothing worth preserving")])
    graph, _ = _shell_graph(tmp_path, model, max_turns=3)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Observe."}]})

    assert len(model.requests) == 1
    assert result["completion_reminder_used"] is False
    assert result["terminal_status"] == "staged_unpublished"


async def test_graph_shell_schema_rejects_missing_required_type_and_enum(tmp_path) -> None:
    """F-03: required, type and enum violations are typed, fixable feedback."""
    model = _ScriptedModel(
        [
            _response(
                "patch item missing required action",
                NativeToolCall(
                    id="missing-action",
                    name="graph_patch",
                    arguments={"items": [{"op_id": "op-1", "kind": "object", "payload": {}}]},
                ),
            ),
            _response(
                "patch correctly",
                NativeToolCall(
                    id="good-patch-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-3",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Schema Repaired",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "diff with a wrong scalar type",
                NativeToolCall(
                    id="bad-limit",
                    name="graph_diff",
                    arguments={"limit": "many"},
                ),
            ),
            _response(
                "inspect to reset the streak",
                NativeToolCall(id="inspect-clean", name="graph_inspect", arguments={}),
            ),
            _response(
                "patch with an invalid enum value",
                NativeToolCall(
                    id="bad-action",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-2",
                                "kind": "object",
                                "action": "explode",
                                "payload": {},
                            }
                        ]
                    },
                ),
            ),
            _response(
                "finalize",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Repair calls."}]})

    missing_error = json.loads(model.requests[1][-1]["content"])["error"]
    assert missing_error["code"] == "invalid_tool_arguments"
    assert missing_error["field"] == "items[0].action"
    assert "missing required field 'action'" in missing_error["message"]
    type_error = json.loads(model.requests[3][-1]["content"])["error"]
    assert type_error["code"] == "invalid_arguments"
    assert "not of type 'integer'" in type_error["message"]
    enum_error = json.loads(model.requests[5][-1]["content"])["error"]
    assert enum_error["code"] == "invalid_tool_arguments"
    assert enum_error["field"] == "items[0].action"
    assert "Unsupported Graph Patch variant 'object'.'explode'" in enum_error["message"]
    assert result["terminal_status"] == "published"
    assert read_active_staged(store, "wake-shell")["objects"] == []
    with store.read_connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM objects WHERE canonical_name = 'Schema Repaired'"
            ).fetchone()[0]
            == 1
        )


async def test_graph_shell_schema_error_streak_halts_with_zero_side_effects(tmp_path) -> None:
    """F-03: repeated schema violations count into the typed streak and halt.

    The failing calls must never reach a tool implementation: no staging,
    no receipts, no formal writes.
    """
    model = _ScriptedModel(
        [
            _response(
                "three unknown-field inspect calls",
                NativeToolCall(
                    id="inspect-1",
                    name="graph_inspect",
                    arguments={"bogus": True},
                ),
            ),
            _response(
                "second unknown-field inspect call",
                NativeToolCall(
                    id="inspect-2",
                    name="graph_inspect",
                    arguments={"bogus": True},
                ),
            ),
            _response(
                "third unknown-field inspect call",
                NativeToolCall(
                    id="inspect-3",
                    name="graph_inspect",
                    arguments={"bogus": True},
                ),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Call inspect."}]})

    assert result["terminal_status"] == "staged_unpublished"
    assert result["resume_allowed"] is True
    assert result["tool_error_streaks"] == {"graph_inspect|error:invalid_arguments|arguments.bogus": 3}
    assert len(model.requests) == 3
    assert read_active_staged(store, "wake-shell")["objects"] == []
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM staged_objects").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM staged_patch_receipts").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM finalize_receipts").fetchone()[0] == 0


async def test_graph_shell_recovers_provider_malformed_arguments_as_tool_feedback(tmp_path) -> None:
    malformed = ToolModelResponse(
        content="",
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "provider-bad-json",
                    "type": "function",
                    "function": {"name": "graph_patch", "arguments": "{not-json"},
                }
            ],
        },
        tool_calls=[],
        prompt_tokens=11,
        completion_tokens=2,
        cost_usd=0.001,
    )
    model = _ScriptedModel(
        [
            StructuredModelOutputError(
                "invalid native tool arguments for provider-bad-json",
                malformed,
            ),
            _response(
                "retry with explicit op id",
                NativeToolCall(
                    id="provider-fixed",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "provider-op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Provider repaired concept",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response("finish", NativeToolCall(id="provider-finish", name="finalize_graph", arguments={})),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Repair provider JSON."}]})

    feedback = json.loads(model.requests[1][-1]["content"])
    assert feedback["error"]["code"] == "invalid_tool_arguments_json"
    assert feedback["error"]["field"] == "arguments"
    assert result["terminal_status"] == "published"
    assert result["patch_success_count"] == 1
    assert result["raw_exception"] == {
        "type": "StructuredModelOutputError",
        "code": "MODEL_OUTPUT_INVALID",
        "recoverable": True,
        "message": "invalid native tool arguments for provider-bad-json",
    }
    report = build_run_report(
        thread_id="thread-shell",
        wake_id="wake-shell",
        domain="lol_cn",
        mission="recover malformed provider arguments",
        mode="broad",
        object_id=None,
        result=result,
        store=store,
        runtime_path=tmp_path / "runtime.sqlite3",
        wall_ms=1,
    )
    assert report["execution"]["raw_exception"] == result["raw_exception"]
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM commit_receipts").fetchone()[0] == 1
    assert read_active_staged(store, "wake-shell")["objects"] == []


async def test_graph_shell_fails_closed_for_malformed_provider_identity_and_args(tmp_path) -> None:
    provider_calls = 0

    async def create(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(
            model="deepseek-v4-flash",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id=7,
                                type="function",
                                function=SimpleNamespace(name="graph_patch", arguments="{not-json"),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )

    model = object.__new__(DeepSeekClient)
    model._model = "deepseek-v4-flash"
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Reject identity."}]})

    assert result["terminal_status"] == "staged_unpublished"
    assert provider_calls == 1
    assert result["tool_events"] == []
    assert read_active_staged(store, "wake-shell")["objects"] == []


async def test_graph_shell_terminal_finalize_skips_later_calls_in_same_message(tmp_path) -> None:
    model = _ScriptedModel(
        [
            _response(
                "publish, then an invalid late mutation",
                NativeToolCall(id="finish-first", name="finalize_graph", arguments={}),
                NativeToolCall(
                    id="late-patch",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "late-op",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Must never be staged",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            )
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Finalize."}]})

    assert result["terminal_status"] == "published"
    assert result["finalize_status"] == "published"
    assert result["resume_allowed"] is False
    assert [event["name"] for event in result["tool_events"]] == ["finalize_graph"]
    assert read_active_staged(store, "wake-shell")["objects"] == []
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 0


async def test_graph_shell_late_malformed_calls_cannot_override_published_terminal(tmp_path) -> None:
    calls = [
        {
            "id": "finish-first",
            "type": "function",
            "function": {"name": "finalize_graph", "arguments": "{}"},
        },
        *[
            {
                "id": f"late-bad-{index}",
                "type": "function",
                "function": {"name": "graph_patch", "arguments": "{not-json"},
            }
            for index in range(3)
        ],
    ]
    model = _ScriptedModel(
        [
            ToolModelResponse(
                content="terminal must remain authoritative",
                message={"role": "assistant", "content": None, "tool_calls": calls},
                tool_calls=[],
            )
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Finalize."}]})

    assert result["terminal_status"] == "published"
    assert result["finalize_status"] == "published"
    assert result["finalize_receipt"]["status"] == "published"
    assert result["resume_allowed"] is False
    assert result["tool_error_streaks"] == {}
    assert [event["name"] for event in result["tool_events"]] == ["finalize_graph"]
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM commit_receipts").fetchone()[0] == 1


def _patch_named(op_id: str, name: str) -> NativeToolCall:
    return NativeToolCall(
        id=f"call-{op_id}-{name}",
        name="graph_patch",
        arguments={
            "items": [
                {
                    "op_id": op_id,
                    "kind": "object",
                    "action": "create",
                    "payload": {
                        "canonical_name": name,
                        "kind": "concept",
                        "provisional": True,
                    },
                }
            ]
        },
    )


async def test_graph_shell_all_rejected_patch_batches_halt_on_same_error_code(tmp_path) -> None:
    model = _ScriptedModel(
        [
            _response("stage once", _patch_named("same-op", "Original")),
            _response("conflict one", _patch_named("same-op", "Revision One")),
            _response("conflict two", _patch_named("same-op", "Revision Two")),
            _response("conflict three", _patch_named("same-op", "Revision Three")),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Patch safely."}]})

    first_feedback = json.loads(model.requests[2][-1]["content"])
    assert first_feedback["ok"] is False
    assert first_feedback["error"] == {
        "code": "op_id_reused",
        "message": "op_id was already used with a different payload",
        "candidates": [],
        "action_hint": ("Use a new op_id for a revised payload; reuse an op_id only for an exact replay"),
        "field": "items[0].op_id",
    }
    assert result["terminal_status"] == "staged_unpublished"
    assert result["resume_allowed"] is True
    assert result["tool_error_streaks"] == {"graph_patch|error:op_id_reused|items[0].op_id": 3}
    assert result["patch_success_count"] == 1
    assert len(model.requests) == 4
    assert [row["canonical_name"] for row in read_active_staged(store, "wake-shell")["objects"]] == [
        "Original"
    ]


async def test_graph_shell_error_limit_skips_late_patch_and_finalize_in_same_message(tmp_path) -> None:
    calls = [
        *[
            {
                "id": f"bad-{index}",
                "type": "function",
                "function": {"name": "graph_patch", "arguments": "{not-json"},
            }
            for index in range(3)
        ],
        {
            "id": "late-patch",
            "type": "function",
            "function": {
                "name": "graph_patch",
                "arguments": json.dumps(_patch_named("late-op", "Must not stage").arguments),
            },
        },
        {
            "id": "late-finalize",
            "type": "function",
            "function": {"name": "finalize_graph", "arguments": "{}"},
        },
    ]
    model = _ScriptedModel(
        [
            ToolModelResponse(
                content="stop at the error boundary",
                message={"role": "assistant", "content": None, "tool_calls": calls},
                tool_calls=[],
            )
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Stop safely."}]})

    assert result["terminal_status"] == "staged_unpublished"
    assert result["resume_allowed"] is True
    assert result["patch_success_count"] == 0
    assert result["tool_error_streaks"] == {"graph_patch|error:invalid_tool_arguments_json|arguments": 3}
    assert len(result["tool_events"]) == 3
    replies = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in result["messages"]
        if message.get("role") == "tool"
    }
    assert set(replies) == {str(call["id"]) for call in calls}
    assert replies["late-patch"]["error"]["code"] == "tool_call_skipped_after_terminal"
    assert replies["late-finalize"]["error"]["code"] == "tool_call_skipped_after_terminal"
    assert read_active_staged(store, "wake-shell")["objects"] == []
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM finalize_receipts").fetchone()[0] == 0


async def test_graph_shell_partial_patch_batch_keeps_success_and_can_publish(tmp_path) -> None:
    partial = NativeToolCall(
        id="partial-batch",
        name="graph_patch",
        arguments={
            "items": [
                _patch_named("same-op", "Conflicting revision").arguments["items"][0],
                _patch_named("new-op", "New sibling").arguments["items"][0],
            ]
        },
    )
    model = _ScriptedModel(
        [
            _response("stage once", _patch_named("same-op", "Original")),
            _response("partial batch", partial),
            _response("publish", NativeToolCall(id="finish", name="finalize_graph", arguments={})),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Patch safely."}]})

    partial_feedback = json.loads(model.requests[2][-1]["content"])
    assert partial_feedback["ok"] is True
    assert [item["status"] for item in partial_feedback["payload"]["results"]] == [
        "conflict",
        "ok",
    ]
    assert result["terminal_status"] == "published"
    assert result["patch_success_count"] == 2
    assert result["tool_error_streaks"] == {}
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 2


async def test_graph_shell_deepen_ref_create_stages_and_finalizes(tmp_path) -> None:
    """B2/B4-2 composition root: an inquiry create carrying deepens_ref stages
    (Wave B) and finalizes with the formal deepens_id mapped — the Graph Shell
    no longer rejects inquiry relations as unimplemented."""
    model = _ScriptedModel(
        [
            _response(
                "stage a deepening inquiry",
                NativeToolCall(
                    id="deepen",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "inq-1",
                                "kind": "inquiry",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "formal",
                                    "prompt": "Why did it happen?",
                                    "rationale": "test",
                                    "deepens_ref": "inq-old",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response("finish", NativeToolCall(id="finish", name="finalize_graph", arguments={})),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="formal",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Formal Anchor",
                )
            ],
            inquiries=[InquiryInput(id="inq-old", subject_id="formal", prompt="Why?", rationale="gap")],
        ),
        "seed-formal",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Inquiry."}]})

    assert result["terminal_status"] == "published"
    assert result["finalize_receipt"]["status"] == "published"
    # the staged inquiry is a deepen overlay: it was staged, then finalized
    # with deepens_id pointing at the formal inquiry it deepens
    with sqlite3.connect(str(tmp_path / "world.sqlite3")) as connection:
        row = connection.execute("SELECT deepens_id FROM inquiries WHERE id = 'wake-shell:i1'").fetchone()
    assert row == ("inq-old",)


async def test_graph_patch_rejects_inquiry_deepens_id_before_dispatch(tmp_path, monkeypatch) -> None:
    """C-EXT-01: an Agent-visible near-miss relation field must fail before staging.

    The historical counterexample advertised ``deepens_id`` as valid because
    graph_patch payloads were open objects, staged the inquiry successfully,
    finalized it, and left the formal ``deepens_id`` NULL.  This regression
    exercises that complete SQLite path while also proving the corrected
    rejection never invokes the graph_patch implementation.
    """
    wrong_item = {
        "op_id": "op-silent-deepen",
        "kind": "inquiry",
        "action": "create",
        "payload": {
            "subject_ref": "formal",
            "prompt": "What is the deeper question?",
            "rationale": "contract regression",
            "deepens_id": "root-inquiry",
        },
    }
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="formal",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Formal Anchor",
                )
            ],
            inquiries=[
                InquiryInput(
                    id="root-inquiry",
                    subject_id="formal",
                    prompt="What happened?",
                    rationale="root gap",
                )
            ],
        ),
        "seed-silent-deepen",
    )
    schema_tools = WorldTools(
        store=store,
        adapters={},
        thread_id="thread-shell",
        wake_id="wake-shell",
    )
    patch_schema = next(
        item["function"]["parameters"]
        for item in schema_tools.schemas()
        if item["function"]["name"] == "graph_patch"
    )
    schema_errors = list(jsonschema.Draft202012Validator(patch_schema).iter_errors({"items": [wrong_item]}))

    implementation_calls: list[str] = []
    original_execute = WorldTools.execute

    async def recording_execute(self, name, arguments, call_id):
        implementation_calls.append(name)
        return await original_execute(self, name, arguments, call_id)

    monkeypatch.setattr(WorldTools, "execute", recording_execute)
    model = _ScriptedModel(
        [
            _response(
                "stage a near-miss deepening inquiry",
                NativeToolCall(
                    id="silent-deepen",
                    name="graph_patch",
                    arguments={"items": [wrong_item]},
                ),
            ),
            _response(
                "finish",
                NativeToolCall(id="finish", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, _ = _shell_graph(tmp_path, model, store=store)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Deepen the inquiry."}]})
    feedback = json.loads(model.requests[1][-1]["content"])
    with store.read_connection() as connection:
        formal_row = connection.execute(
            "SELECT deepens_id FROM inquiries WHERE id = 'wake-shell:i1'"
        ).fetchone()
        staged_count = connection.execute(
            "SELECT COUNT(*) FROM staged_inquiries WHERE wake_id = 'wake-shell'"
        ).fetchone()[0]
        patch_receipt_count = connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts "
            "WHERE wake_id = 'wake-shell' AND op_id = 'op-silent-deepen'"
        ).fetchone()[0]

    reproduced = {
        "schema_error_count": len(schema_errors),
        "feedback_ok": feedback.get("ok"),
        "terminal_status": result["terminal_status"],
        "formal_deepens_id": formal_row["deepens_id"] if formal_row else None,
        "implementation_calls": implementation_calls,
        "staged_count": staged_count,
        "patch_receipt_count": patch_receipt_count,
    }
    assert schema_errors, f"C-EXT-01 counterexample reproduced: {reproduced}"
    error = feedback["error"]
    assert error["code"] == "invalid_tool_arguments"
    assert error["field"] == "items[0].payload.deepens_id"
    assert "deepens_ref" in error["action_hint"]
    assert implementation_calls == []
    assert staged_count == 0
    assert patch_receipt_count == 0
    assert formal_row is None


async def test_graph_shell_answer_resolve_end_to_end(tmp_path) -> None:
    """B2/B4-3/4 composition root: a staged answering assertion answers a
    formal inquiry, a staged resolve names it, and finalize publishes the
    resolution with the answer co-published in the same delta."""
    model = _ScriptedModel(
        [
            _response(
                "stage the answering assertion",
                NativeToolCall(
                    id="answer",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-a1",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "t1",
                                    "predicate": "expresses",
                                    "object_ref": "s1",
                                    "epistemic_role": "fact",
                                    "confidence": 0.9,
                                    "answers_ref": "inq-f",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "resolve the inquiry naming that assertion",
                NativeToolCall(
                    id="resolve",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-r1",
                                "kind": "inquiry",
                                "action": "resolve",
                                "target_ref": "inq-f",
                                "payload": {
                                    "expected_version": 1,
                                    "answers_ref": "wake-shell:a1",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response("publish", NativeToolCall(id="finish", name="finalize_graph", arguments={})),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One"),
                ObjectInput(id="s1", kind=ObjectKind.EVENT, canonical_name="Summer Cup"),
            ],
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

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Inquiry."}]})

    assert result["terminal_status"] == "published"
    assert result["finalize_receipt"]["status"] == "published"
    answer_result = json.loads(model.requests[1][-1]["content"])
    resolve_result = json.loads(model.requests[2][-1]["content"])
    assert answer_result["payload"]["results"][0]["status"] == "ok"
    assert resolve_result["payload"]["results"][0]["status"] == "ok"
    with sqlite3.connect(str(tmp_path / "world.sqlite3")) as connection:
        inquiry_row = connection.execute(
            "SELECT status, version FROM inquiries WHERE id = 'inq-f'"
        ).fetchone()
        assertion_row = connection.execute(
            "SELECT answers_inquiry_id FROM assertions WHERE id = 'wake-shell:a1'"
        ).fetchone()
        commit_receipt = connection.execute(
            "SELECT receipt_json FROM commit_receipts WHERE commit_id = 'wake-shell:finalize'"
        ).fetchone()[0]
    assert inquiry_row == ("resolved", 2)
    assert assertion_row == ("inq-f",)
    assert '"resolved_inquiry_ids":["inq-f"]' in commit_receipt
    assert read_active_staged(store, "wake-shell") == {
        "objects": [],
        "assertions": [],
        "inquiries": [],
    }


async def test_graph_shell_inquiry_typed_errors_without_side_effect(tmp_path) -> None:
    """B2/B4-9: unsupported inquiry shapes fail with typed errors naming the
    fix — a resolve without target_ref, a resolve without answers_ref, and an
    answers_ref smuggled onto an inquiry create — and never stage anything."""
    model = _ScriptedModel(
        [
            _response(
                "resolve missing target",
                NativeToolCall(
                    id="bad-resolve-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "inquiry",
                                "action": "resolve",
                                "payload": {
                                    "expected_version": 1,
                                    "answers_ref": "wake-shell:a1",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "reset error streak",
                NativeToolCall(id="inspect-1", name="graph_inspect", arguments={}),
            ),
            _response(
                "resolve missing answering assertion",
                NativeToolCall(
                    id="bad-resolve-2",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-2",
                                "kind": "inquiry",
                                "action": "resolve",
                                "target_ref": "inq-f",
                                "payload": {"expected_version": 1},
                            }
                        ]
                    },
                ),
            ),
            _response(
                "reset error streak",
                NativeToolCall(id="inspect-2", name="graph_inspect", arguments={}),
            ),
            _response(
                "smuggle answers_ref onto inquiry create",
                NativeToolCall(
                    id="bad-create",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-3",
                                "kind": "inquiry",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "t1",
                                    "prompt": "Why?",
                                    "rationale": "test",
                                    "answers_ref": "wake-shell:a1",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response("finish", NativeToolCall(id="finish", name="finalize_graph", arguments={})),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Inquiry."}]})

    errors = [json.loads(model.requests[index][-1]["content"])["error"] for index in (1, 3, 5)]
    assert [error["code"] for error in errors] == ["invalid_tool_arguments"] * 3
    assert [error["field"] for error in errors] == [
        "items[0].target_ref",
        "items[0].payload.answers_ref",
        "items[0].payload.answers_ref",
    ]
    assert "kind 'assertion'" in errors[2]["action_hint"]
    assert read_active_staged(store, "wake-shell")["inquiries"] == []
    assert result["terminal_status"] == "published"


async def test_graph_shell_turn_boundary_keeps_active_staging_unpublished(tmp_path) -> None:
    model = _ScriptedModel(
        [
            _response(
                "stage then hit the turn boundary",
                NativeToolCall(
                    id="patch-only",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Retained Draft",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            )
        ]
    )
    graph, store = _shell_graph(tmp_path, model, max_turns=1)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Draft."}]})

    assert result["terminal_status"] == "staged_unpublished"
    assert result["resume_allowed"] is True
    assert "1 active staged item" in result["terminal_summary"]
    assert len(read_active_staged(store, "wake-shell")["objects"]) == 1
    with store.read_connection() as connection:
        assert (
            connection.execute("SELECT 1 FROM finalize_receipts WHERE wake_id = 'wake-shell'").fetchone()
            is None
        )


async def test_graph_shell_bootstrap_never_mutates_formal_inquiry_lifecycle(tmp_path) -> None:
    model = _ScriptedModel([_response("pause once"), _response("pause twice")])
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="inquiry-subject",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Inquiry subject",
                )
            ],
            inquiries=[
                InquiryInput(
                    id="old-open-inquiry",
                    subject_id="inquiry-subject",
                    prompt="Still open?",
                    rationale="Formal lifecycle must use the publish boundary.",
                )
            ],
        ),
        "seed-old-inquiry",
    )
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE inquiries SET last_attempted_at = '2020-01-01T00:00:00+00:00' "
            "WHERE id = 'old-open-inquiry'"
        )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Observe only."}]})

    assert result["terminal_status"] == "staged_unpublished"
    with store.read_connection() as connection:
        assert (
            connection.execute("SELECT status FROM inquiries WHERE id = 'old-open-inquiry'").fetchone()[0]
            == "open"
        )
        assert connection.execute("SELECT COUNT(*) FROM world_audit").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM finalize_receipts").fetchone()[0] == 0


@pytest.mark.parametrize("boundary", ["cost", "deadline", "natural"])
async def test_graph_shell_other_stop_boundaries_never_auto_finalize_active_staging(
    tmp_path, boundary: str
) -> None:
    patch_response = _response(
        "stage draft",
        NativeToolCall(
            id="patch-boundary",
            name="graph_patch",
            arguments={
                "items": [
                    {
                        "op_id": "op-boundary",
                        "kind": "object",
                        "action": "create",
                        "payload": {
                            "canonical_name": f"{boundary} Draft",
                            "kind": "concept",
                            "provisional": True,
                        },
                    }
                ]
            },
        ),
    )
    if boundary == "cost":
        patch_response.cost_usd = 0.02
        responses = [patch_response]
        max_cost = 0.01
    elif boundary == "deadline":
        responses = [patch_response, LiveDeadlineExceeded()]
        max_cost = None
    else:
        responses = [patch_response, _response("pause once"), _response("pause twice")]
        max_cost = None
    model = _ScriptedModel(responses)
    graph, store = _shell_graph(tmp_path, model, max_cost_usd=max_cost)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Draft."}]})

    assert result["terminal_status"] == "staged_unpublished"
    assert len(read_active_staged(store, "wake-shell")["objects"]) == 1
    with store.read_connection() as connection:
        assert (
            connection.execute("SELECT 1 FROM finalize_receipts WHERE wake_id = 'wake-shell'").fetchone()
            is None
        )


@pytest.mark.parametrize("terminal", ["already_published", "wake_closed"])
async def test_graph_shell_durable_finalize_terminals_stop_without_another_model(
    tmp_path, terminal: str
) -> None:
    model = _ScriptedModel(
        [_response("replay finalize", NativeToolCall(id="finalize", name="finalize_graph", arguments={}))]
    )
    graph, store = _shell_graph(tmp_path, model)
    first = finalize_graph(store, "wake-shell")
    assert first.status == "published"
    if terminal == "wake_closed":
        late = apply_patch(
            store,
            "wake-shell",
            [
                {
                    "op_id": "late",
                    "kind": "object",
                    "action": "create",
                    "payload": {
                        "canonical_name": "Late Draft",
                        "kind": "concept",
                        "provisional": True,
                    },
                }
            ],
        )[0]
        assert late["status"] == "ok"

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Replay."}]})

    assert result["terminal_status"] == terminal
    assert len(model.requests) == 1


async def test_graph_shell_patch_after_concurrent_finalize_stops_as_wake_closed(tmp_path) -> None:
    model = _ScriptedModel([_response("late patch", _patch_named("late-op", "Must stay closed"))])
    graph, store = _shell_graph(tmp_path, model)
    assert finalize_graph(store, "wake-shell").status == "published"

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Patch late."}]})

    assert result["terminal_status"] == "wake_closed"
    assert result["finalize_status"] == "wake_closed"
    assert result["resume_allowed"] is False
    assert len(model.requests) == 1
    assert read_active_staged(store, "wake-shell")["objects"] == []


async def test_graph_shell_closed_mixed_replay_batch_stops_as_wake_closed(tmp_path) -> None:
    original_patch = _patch_named("old-op", "Published Draft")
    first_model = _ScriptedModel(
        [
            _response("stage", original_patch),
            _response(
                "publish",
                NativeToolCall(id="publish", name="finalize_graph", arguments={}),
            ),
        ]
    )
    first_graph, store = _shell_graph(tmp_path, first_model)
    first = await first_graph.ainvoke({"messages": [{"role": "user", "content": "Publish once."}]})
    assert first["terminal_status"] == "published"

    mixed_call = NativeToolCall(
        id="mixed-after-close",
        name="graph_patch",
        arguments={
            "items": [
                original_patch.arguments["items"][0],
                _patch_named("new-op", "Must stay closed").arguments["items"][0],
            ]
        },
    )
    second_model = _ScriptedModel(
        [
            _response("mixed replay", mixed_call),
            _response(
                "must not run",
                NativeToolCall(id="late-finalize", name="finalize_graph", arguments={}),
            ),
        ]
    )
    second_graph, _ = _shell_graph(tmp_path, second_model)

    result = await second_graph.ainvoke({"messages": [{"role": "user", "content": "Do not reopen."}]})

    assert result["terminal_status"] == "wake_closed"
    assert result["finalize_status"] == "wake_closed"
    assert len(second_model.requests) == 1
    assert read_active_staged(store, "wake-shell")["objects"] == []


async def test_graph_shell_commit_replay_conflict_stays_loud(tmp_path, monkeypatch) -> None:
    def conflict(_store: WorldStore, _wake_id: str) -> FinalizeReceipt:
        raise CommitReplayConflict("durable receipt divergence")

    monkeypatch.setattr(graph_module, "finalize_graph", conflict)
    model = _ScriptedModel(
        [_response("finalize", NativeToolCall(id="finalize", name="finalize_graph", arguments={}))]
    )
    graph, _store = _shell_graph(tmp_path, model)

    with pytest.raises(CommitReplayConflict, match="durable receipt divergence"):
        await graph.ainvoke({"messages": [{"role": "user", "content": "Finalize."}]})


async def test_graph_shell_stale_base_requires_reread_drop_and_repatch(tmp_path) -> None:
    class ConcurrentUpdateModel(_ScriptedModel):
        store: WorldStore | None = None

        async def invoke_tools(self, messages, *, tools, **kwargs):
            if len(self.requests) == 1:
                assert self.store is not None
                self.store.memory_commit(
                    CognitiveDelta(
                        objects=[
                            ObjectInput(
                                id="formal-target",
                                kind=ObjectKind.ENTITY,
                                canonical_name="Concurrent Formal",
                                expected_version=1,
                            )
                        ]
                    ),
                    "concurrent-formal-update",
                )
            return await super().invoke_tools(messages, tools=tools, **kwargs)

    model = ConcurrentUpdateModel(
        [
            _response(
                "Patch v1.",
                NativeToolCall(
                    id="patch-v1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "update",
                                "target_ref": "formal-target",
                                "payload": {"canonical_name": "Old Draft", "kind": "entity"},
                            }
                        ]
                    },
                ),
            ),
            _response("Finalize old base.", NativeToolCall(id="stale", name="finalize_graph", arguments={})),
            _response(
                "Reread formal v2.",
                NativeToolCall(
                    id="reread",
                    name="memory_read",
                    arguments={"object_id": "formal-target"},
                ),
            ),
            _response(
                "Drop stale overlay.",
                NativeToolCall(
                    id="drop-stale",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-2",
                                "kind": "object",
                                "action": "drop",
                                "target_ref": "wake-shell:s1",
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Patch against v2.",
                NativeToolCall(
                    id="patch-v2",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-3",
                                "kind": "object",
                                "action": "update",
                                "target_ref": "formal-target",
                                "expected_version": 2,
                                "payload": {"canonical_name": "Reviewed Draft", "kind": "entity"},
                            }
                        ]
                    },
                ),
            ),
            _response("Publish v2 draft.", NativeToolCall(id="publish", name="finalize_graph", arguments={})),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    model.store = store
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="formal-target",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Original Formal",
                )
            ]
        ),
        "seed-formal",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Revise."}]})

    stale_receipt = json.loads(model.requests[2][-1]["content"])["payload"]
    assert stale_receipt["status"] == "blocked"
    assert stale_receipt["blockers"][0]["code"] == "stale_base"
    assert (
        stale_receipt["blockers"][0]["action_hint"]
        == "memory_read formal-target, then drop and re-patch this overlay"
    )
    reread = json.loads(model.requests[3][-1]["content"])["memory"]
    assert reread["identity"]["canonical_name"] == "Concurrent Formal"
    assert result["terminal_status"] == "published"
    with store.read_connection() as connection:
        formal = connection.execute(
            "SELECT canonical_name, version FROM objects WHERE id = 'formal-target'"
        ).fetchone()
        assert tuple(formal) == ("Reviewed Draft", 3)


# Wave C C4 scripted-model behavior proofs + C2/C3 contract locks.


def test_graph_shell_mechanics_prompt_covers_persistent_concepts() -> None:
    """C1: the Graph Shell prompt explains the persistent conceptual model
    compactly and domain-neutrally, without prescribing a fixed step order."""
    from leave_information_bubble.world.domain_config import resolve_domain_focus
    from leave_information_bubble.world_agent.prompt import graph_shell_prompt

    rendered = graph_shell_prompt("deep", None, resolve_domain_focus("lol_cn"))
    concepts = (
        "formal graph",
        "working graph",
        "object",
        "assertion",
        "inquiry",
        "supersede",
        "canonical name",
        "identity",
        "verbatim",
        "object_ref",
        "literal",
        "empty working graph",
        "staging",
        "resume",
    )
    for concept in concepts:
        assert concept in rendered, f"Graph Shell prompt must explain: {concept!r}"
    # the honest-empty rule and explicit supersede are requirements, not steps
    assert "honest" in rendered
    # not a fixed SOP: the prompt never orders the memory tools as a step list
    assert "step" not in rendered.casefold() or "not a required" in rendered.casefold()


async def test_graph_shell_tool_surface_fingerprint_stable_and_no_submit_cognition(tmp_path) -> None:
    """C2/C4: the main-wake tool list and order are stable across graph builds,
    and the Graph Shell schema never contains submit_cognition."""
    expected = [
        "memory_recent",
        "memory_search",
        "memory_read",
        "memory_compare",
        "memory_expand",
        "memory_evidence",
        "memory_inquiries",
        "memory_changes",
        "memory_overview",
        "discover_sources",
        "search_sources",
        "open_source",
        "sample_discussion",
        "inspect_media",
        "follow_related",
        "graph_patch",
        "graph_inspect",
        "graph_diff",
        "finalize_graph",
    ]
    surfaces = []
    for _ in range(2):
        model = _ScriptedModel([_response("pause"), _response("pause")])
        graph, _ = _shell_graph(tmp_path, model)
        await graph.ainvoke({"messages": [{"role": "user", "content": "Open a Graph Shell wake."}]})
        surfaces.append(model.tool_names[0])
    assert surfaces[0] == expected
    assert surfaces[1] == expected
    assert "submit_cognition" not in surfaces[0]
    # G5b-2 D2: the retired digest-flow observation tool is off the surface
    assert "digest_observation" not in surfaces[0]


async def test_graph_shell_core_tool_descriptions_are_differentiated(tmp_path) -> None:
    """C2: each of the eight core descriptions answers what/when/difference
    with substantive text, and finalize_graph stays argument-free."""
    model = _ScriptedModel([_response("pause"), _response("pause")])
    graph, _ = _shell_graph(tmp_path, model)
    await graph.ainvoke({"messages": [{"role": "user", "content": "Open a Graph Shell wake."}]})
    schemas = {entry["function"]["name"]: entry["function"] for entry in model.tool_schemas[0]}
    core = [
        "memory_search",
        "memory_read",
        "memory_compare",
        "memory_expand",
        "graph_patch",
        "graph_inspect",
        "graph_diff",
        "finalize_graph",
    ]
    for name in core:
        description = schemas[name]["description"]
        assert len(description) >= 120, f"{name} description is too thin"
        assert any(word in description for word in ("Unlike", "unlike", "when")), (
            f"{name} description must state when to use it / how it differs"
        )
    finalize_params = schemas["finalize_graph"]["parameters"]
    assert finalize_params == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    # the committing tool never claims a read capability it does not have
    assert "publish" in schemas["finalize_graph"]["description"]


async def test_graph_shell_identity_candidates_drive_reads_then_reuse(tmp_path) -> None:
    """C4: after identity candidates, the model reads/searchs/compares instead
    of fabricating ids, then reuses the existing id and publishes."""
    from leave_information_bubble.world import CognitiveDelta, ObjectInput, ObjectKind

    model = _ScriptedModel(
        [
            _response(
                "Create a concept whose name already exists.",
                NativeToolCall(
                    id="patch-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Formal Anchor",
                                    "kind": "entity",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Candidates returned — compare the existing referent first.",
                NativeToolCall(
                    id="search-1",
                    name="memory_search",
                    arguments={"query": "Formal Anchor"},
                ),
            ),
            _response(
                "Reuse the existing id instead of inventing a new one.",
                NativeToolCall(
                    id="patch-2",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-2",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "formal-anchor",
                                    "predicate": "related_to",
                                    "object_ref": "formal-anchor",
                                    "epistemic_role": "fact",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Publish the revised delta.",
                NativeToolCall(id="finalize-1", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="formal-anchor",
                    kind=ObjectKind.EVENT,
                    canonical_name="Formal Anchor",
                )
            ]
        ),
        "seed-anchor",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Build and publish."}]})

    collision = json.loads(model.requests[1][-1]["content"])
    assert collision["ok"] is False
    assert collision["error"]["code"] == "identity_candidate_exists"
    candidates = collision["error"]["candidates"]
    assert any(entry["id"] == "formal-anchor" for entry in candidates)
    # C3: candidates carry id + name + kind + basis
    assert all({"id", "name", "kind", "basis"} <= set(entry) for entry in candidates)
    assert result["terminal_status"] == "published"
    with store.read_connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM objects WHERE id = 'formal-anchor'").fetchone()[0] == 1
        )


async def test_graph_shell_identity_unresolved_confirm_distinct_then_publish(tmp_path) -> None:
    """C4: an unresolved identity may be confirmed distinct with a basis and
    then the graph publishes — the decision is honored end to end."""
    from leave_information_bubble.world import CognitiveDelta, ObjectInput, ObjectKind

    model = _ScriptedModel(
        [
            _response(
                "Create an entity sharing a name with an existing object.",
                NativeToolCall(
                    id="patch-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Same Name",
                                    "kind": "entity",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "A different referent despite the name — confirm distinct with basis.",
                NativeToolCall(
                    id="patch-2",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-2",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Same Name",
                                    "kind": "entity",
                                    "provisional": True,
                                },
                                "decision": {
                                    "action": "confirm_distinct",
                                    "distinct_from": "t1",
                                    "basis": "a second organization that happens to share the name",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Publish.",
                NativeToolCall(id="finalize-1", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Same Name")]),
        "seed-same-name",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Build and publish."}]})

    confirmed = json.loads(model.requests[2][-1]["content"])
    assert confirmed["ok"] is True
    assert confirmed["payload"]["results"][0]["status"] == "ok"
    assert result["terminal_status"] == "published"
    with store.read_connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM objects WHERE canonical_name = 'Same Name'").fetchone()[
                0
            ]
            == 2
        )


async def test_graph_shell_patch_success_reviewed_with_diff_then_finalized(tmp_path) -> None:
    """C4: after a successful patch the model can review via diff and inspect
    before the explicit finalize — write then review is the demonstrated path."""
    model = _ScriptedModel(
        [
            _response(
                "Stage one concept.",
                NativeToolCall(
                    id="patch-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Diff Reviewable",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Review what this wake changed.",
                NativeToolCall(id="diff-1", name="graph_diff", arguments={}),
            ),
            _response(
                "Inspect readiness.",
                NativeToolCall(id="inspect-1", name="graph_inspect", arguments={}),
            ),
            _response(
                "Publish explicitly.",
                NativeToolCall(id="finalize-1", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Build."}]})

    patch_result = json.loads(model.requests[1][-1]["content"])
    assert patch_result["ok"] is True
    assert patch_result["payload"]["results"][0]["staged_id"] == "wake-shell:s1"
    # C3: a successful patch returns the staged id plus a readable summary
    assert "summary" in patch_result["payload"]["results"][0]
    diff_result = json.loads(model.requests[2][-1]["content"])
    assert diff_result["scope"]["published"] is False
    assert diff_result["payload"]["summary"]["total"] == 1
    inspect_result = json.loads(model.requests[3][-1]["content"])
    assert inspect_result["payload"]["readiness"] == "ready"
    assert result["terminal_status"] == "published"


async def test_graph_shell_unsupported_inquiry_action_not_repeated(tmp_path) -> None:
    """C4: an unsupported action returns a typed error exactly once and the
    model moves to a supported action instead of retrying the same call."""
    model = _ScriptedModel(
        [
            _response(
                "Stage one open inquiry.",
                NativeToolCall(
                    id="patch-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "inquiry",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "formal-anchor",
                                    "prompt": "who is the champion?",
                                    "rationale": "gap",
                                    "kind": "factual",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Try to update an inquiry lifecycle state.",
                NativeToolCall(
                    id="patch-2",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-2",
                                "kind": "inquiry",
                                "action": "update",
                                "target_ref": "wake-shell:i1",
                                "payload": {"prompt": "changed"},
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Lifecycle mutation is unsupported — withdraw instead.",
                NativeToolCall(
                    id="patch-3",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-3",
                                "kind": "inquiry",
                                "action": "drop",
                                "target_ref": "wake-shell:i1",
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Publish the honest empty result.",
                NativeToolCall(id="finalize-1", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="formal-anchor",
                    kind=ObjectKind.EVENT,
                    canonical_name="Formal Anchor",
                )
            ]
        ),
        "seed-anchor",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Build."}]})

    unsupported = json.loads(model.requests[2][-1]["content"])
    assert unsupported["ok"] is False
    assert unsupported["error"]["code"] == "invalid_tool_arguments"
    assert unsupported["error"]["field"] == "items[0].action"
    # the model never repeated the unsupported action: turn 3 is a supported drop
    second_patch = json.loads(model.requests[3][-1]["content"])
    assert second_patch["payload"]["results"][0]["action"] == "dropped"
    # exactly one unsupported error across the whole transcript (same tool
    # result is re-presented in later message snapshots — count per call id)
    call_errors: dict[str, str] = {}
    for message in model.requests:
        for entry in message:
            if entry.get("role") == "tool":
                call_errors[str(entry["tool_call_id"])] = (
                    json.loads(entry["content"]).get("error", {}).get("code") or ""
                )
    assert list(call_errors.values()).count("invalid_tool_arguments") == 1
    assert result["terminal_status"] == "published"


async def test_graph_shell_supersede_result_echoes_old_to_new_chain(tmp_path) -> None:
    """C3: a successful supersede echoes the old -> new chain and both sides
    land in the formal graph after finalize."""
    model = _ScriptedModel(
        [
            _response(
                "Correct the assertion via supersede.",
                NativeToolCall(
                    id="patch-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "assertion",
                                "action": "supersede",
                                "payload": {
                                    "subject_ref": "t1",
                                    "predicate": "explains",
                                    "literal": "newer understanding",
                                    "epistemic_role": "fact",
                                    "supersedes_ref": "old-a1",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response(
                "Publish the correction.",
                NativeToolCall(id="finalize-1", name="finalize_graph", arguments={}),
            ),
        ]
    )
    graph, store = _shell_graph(tmp_path, model)
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")],
            assertions=[
                AssertionInput(
                    id="old-a1",
                    subject_id="t1",
                    predicate="explains",
                    literal="older understanding",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                )
            ],
        ),
        "seed-old",
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Correct."}]})

    supersede = json.loads(model.requests[1][-1]["content"])
    item = supersede["payload"]["results"][0]
    assert item["status"] == "ok"
    assert item["staged_id"] == "wake-shell:a1"
    assert item["supersedes"] == "old-a1"
    assert result["terminal_status"] == "published"
    with store.read_connection() as connection:
        chain = connection.execute(
            "SELECT id, supersedes_id, literal_json FROM assertions WHERE predicate = 'explains' ORDER BY id"
        ).fetchall()
        assert chain[0][0] == "old-a1"
        assert chain[1][0] == "wake-shell:a1"
        assert chain[1][1] == "old-a1"
        assert json.loads(chain[1][2]) == "newer understanding"
