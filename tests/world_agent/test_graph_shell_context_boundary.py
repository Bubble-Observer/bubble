"""Graph Shell physical-context boundary tests (design 2026-08-23).

Two layers: graph-level isolation proving the warn-then-hard-cut semantics
(warn once + dedupe, hard cut with zero model calls, precedence under the
turn/cost boundaries, resume fail-closed) in isolation, and the
composition-root CLI contract (parser defaults/validation and a scripted
run that hard-cuts at the context boundary).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from leave_information_bubble.gateway.client import NativeToolCall
from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world_agent.graph import (
    CONTEXT_WIND_DOWN_NOTICE,
    _estimate_request_tokens,
    build_world_agent_graph,
    graph_shell_initial_state,
)
from tests.world_agent._graph_helpers import _response, _ScriptedModel
from tests.world_agent.test_graph_shell_recovery import _patch_call
from tests.world_agent.test_vertical_slice import (
    _load_runner,
    _required_runner_args,
    _write_model,
    _write_replay_fixture,
)

THREAD, WAKE = "thread-ctx", "wake-ctx"


def test_context_estimate_counts_tool_schemas_sent_on_every_call() -> None:
    messages = [{"role": "user", "content": "hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "large_contract",
                "description": "x" * 3_000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    message_only = _estimate_request_tokens(messages, 100)
    complete = _estimate_request_tokens(messages, 100, tools=tools)

    assert complete > message_only + 1_500


def _build(
    store,
    saver,
    model,
    tools,
    *,
    max_turns: int = 8,
    max_cost_usd: float | None = None,
    context_warning_tokens: int | None = None,
    context_hard_cut_tokens: int | None = None,
):
    return build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        checkpointer=saver,
        thread_id=THREAD,
        wake_id=WAKE,
        max_turns=max_turns,
        max_cost_usd=max_cost_usd,
        context_warning_tokens=context_warning_tokens,
        context_hard_cut_tokens=context_hard_cut_tokens,
    )


def _tools(store: WorldStore) -> WorldTools:
    return WorldTools(store=store, adapters={}, thread_id=THREAD, wake_id=WAKE)


def _initial_input(store: WorldStore, content: str) -> dict:
    return graph_shell_initial_state(
        messages=[{"role": "user", "content": content}],
        store=store,
        thread_id=THREAD,
        wake_id=WAKE,
        domain_key="lol_cn",
        mode="broad",
        object_id=None,
    )


def _warnings(model: _ScriptedModel) -> list[str]:
    """Freshly-injected wind-down notices: the last message of a request.

    The injected notice is part of the transcript and therefore reappears as
    history in later requests; counting it there again would mistake history
    for a re-injection. A fresh injection is always the final message of the
    request it was appended to.
    """
    return [
        str(last["content"])
        for request in model.requests
        if request
        and isinstance((last := request[-1]).get("content"), str)
        and CONTEXT_WIND_DOWN_NOTICE in last["content"]
    ]


async def test_context_boundary_disabled_by_default(tmp_path: Path) -> None:
    """No context parameters -> the boundary checks stay inert (regression).

    A huge transcript must not inject anything, must not hard-cut, and must
    record ``context_usage_estimate == 0`` exactly like pre-design runs.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": THREAD}}
    model = _ScriptedModel(
        [
            _response(
                "publish",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            )
        ]
    )
    tools = _tools(store)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(store, saver, model, tools)
        result = await graph.ainvoke(_initial_input(store, "x" * 30_000), config)

    assert result["terminal_status"] == "published"
    assert result["context_usage_estimate"] == 0
    assert result["context_warning_injected"] is False
    assert result["context_cut"] is False
    assert len(model.requests) == 1
    assert _warnings(model) == []


async def test_context_warning_injects_once_then_dedupes(tmp_path: Path) -> None:
    """The wind-down notice is injected exactly once across the wake.

    Turn 1 crosses the warning line (estimate >= 20000); turn 2 stays over it
    but the persisted flag suppresses a second injection. The wake continues
    normally (patch then publish) — the warning never blocks work.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": THREAD}}
    model = _ScriptedModel(
        [
            _response("patch", _patch_call()),
            _response(
                "publish",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            ),
        ]
    )
    tools = _tools(store)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(
            store,
            saver,
            model,
            tools,
            context_warning_tokens=20_000,
            context_hard_cut_tokens=100_000,
        )
        result = await graph.ainvoke(_initial_input(store, "x" * 20_000), config)

    assert result["terminal_status"] == "published"
    assert result["context_warning_injected"] is True
    assert result["context_cut"] is False
    assert result["context_usage_estimate"] >= 20_000
    assert len(model.requests) == 2
    injections = _warnings(model)
    assert len(injections) == 1  # fresh injection only on turn 1
    assert injections[0] == CONTEXT_WIND_DOWN_NOTICE
    # the notice reads as a user directive, not a resource report — no
    # token counts, no "context almost full", no technical vocabulary
    assert "tokens" not in injections[0]
    assert "context" not in injections[0].lower()
    # the notice survives in the transcript exactly once — never duplicated
    assert (
        sum(
            isinstance(message.get("content"), str) and CONTEXT_WIND_DOWN_NOTICE in message["content"]
            for message in result["messages"]
        )
        == 1
    )


async def test_context_hard_cut_stops_without_model_call(tmp_path: Path) -> None:
    """At the hard cut the wake fails closed with zero model invocations.

    The pre-call estimate already exceeds the limit, so the agent returns the
    staged_unpublished terminal without ever touching the model — the scripted
    model must remain completely unconsumed.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": THREAD}}
    model = _ScriptedModel(
        [
            _response(
                "publish",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            )
        ]
    )
    tools = _tools(store)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(
            store,
            saver,
            model,
            tools,
            context_warning_tokens=None,
            context_hard_cut_tokens=25_000,
        )
        result = await graph.ainvoke(_initial_input(store, "x" * 30_000), config)

    assert result["terminal_status"] == "staged_unpublished"
    assert result["halted"] is True
    assert result["resume_allowed"] is True
    assert result["context_cut"] is True
    assert result["context_warning_injected"] is False
    assert result["context_usage_estimate"] >= 25_000
    assert "context boundary" in result["terminal_summary"]
    assert model.requests == []


async def test_context_hard_cut_holds_across_resume(tmp_path: Path) -> None:
    """A full wake keeps failing closed on every resume attempt.

    Resume replays the whole transcript — the estimate can only grow — so the
    same pre-call gate must stop the resumed run with zero model calls, not
    let it leak a call that would overflow the provider window.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": THREAD}}
    model = _ScriptedModel(
        [
            _response(
                "publish",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            )
        ]
    )
    tools = _tools(store)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(
            store,
            saver,
            model,
            tools,
            context_warning_tokens=None,
            context_hard_cut_tokens=25_000,
        )
        first = await graph.ainvoke(_initial_input(store, "x" * 30_000), config)
        assert first["terminal_status"] == "staged_unpublished"
        assert model.requests == []

        # CLI resume clears the terminal before re-invoking (composition root
        # does the same via aupdate_state); the gate must still stop the run.
        await graph.aupdate_state(config, {"terminal_status": "", "resume_count": 1})
        resumed = await graph.ainvoke(None, config)

    assert resumed["terminal_status"] == "staged_unpublished"
    assert resumed["context_cut"] is True
    assert resumed["context_usage_estimate"] >= 25_000
    assert model.requests == []


async def test_turn_boundary_precedes_context(tmp_path: Path) -> None:
    """The existing turn hard stop wins over the context hard stop.

    Turn 1 crosses only the warning line (est ~23.5k vs warning 20k / cut
    25k); the scripted turn-1 response alone adds 30k chars, so turn 2's
    estimate would exceed the context cut if it were checked — but the turn
    boundary (turns == max_turns) fires first, leaving ``context_cut`` unset.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": THREAD}}
    model = _ScriptedModel([_response("x" * 30_000, _patch_call())])
    tools = _tools(store)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(
            store,
            saver,
            model,
            tools,
            max_turns=1,
            context_warning_tokens=20_000,
            context_hard_cut_tokens=100_000,
        )
        result = await graph.ainvoke(_initial_input(store, "x" * 15_000), config)

    assert result["terminal_status"] == "staged_unpublished"
    assert "turn boundary" in result["terminal_summary"]
    assert result["context_cut"] is False
    assert result["context_warning_injected"] is True


async def test_cost_boundary_precedes_context(tmp_path: Path) -> None:
    """The existing cost hard stop wins over the context hard stop.

    ``max_cost_usd=0`` trips the cost gate on the very first agent entry even
    though the context estimate far exceeds the (deliberately tiny) cut.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": THREAD}}
    model = _ScriptedModel(
        [
            _response(
                "publish",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            )
        ]
    )
    tools = _tools(store)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(
            store,
            saver,
            model,
            tools,
            max_cost_usd=0.0,
            context_warning_tokens=None,
            context_hard_cut_tokens=1,
        )
        result = await graph.ainvoke(_initial_input(store, "x" * 30_000), config)

    assert result["terminal_status"] == "staged_unpublished"
    assert "cost boundary" in result["terminal_summary"]
    assert result["context_cut"] is False
    assert model.requests == []


def test_build_rejects_invalid_context_thresholds(tmp_path: Path) -> None:
    """Non-positive or inverted context thresholds fail at composition time."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = _tools(store)
    model = _ScriptedModel([])

    with pytest.raises(ValueError, match="context_hard_cut_tokens must be positive"):
        build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            thread_id=THREAD,
            wake_id=WAKE,
            max_turns=8,
            context_hard_cut_tokens=0,
        )
    with pytest.raises(ValueError, match="context_warning_tokens must not exceed"):
        build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            thread_id=THREAD,
            wake_id=WAKE,
            max_turns=8,
            context_warning_tokens=20_000,
            context_hard_cut_tokens=10_000,
        )


def test_cli_parser_context_threshold_contract() -> None:
    """CLI defaults (96 turns / 700k warn / 800k cut) and validation errors."""
    runner = _load_runner()
    args = runner.parse_args(_required_runner_args(THREAD) + ["--graph-shell"])
    assert args.max_turns == 96
    assert args.context_warning_tokens == 700_000
    assert args.context_hard_cut_tokens == 800_000

    with pytest.raises(SystemExit):
        runner.parse_args(
            _required_runner_args(THREAD)
            + ["--graph-shell", "--context-warning-tokens", "20000", "--context-hard-cut-tokens", "10000"]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(_required_runner_args(THREAD) + ["--graph-shell", "--context-hard-cut-tokens", "0"])


async def test_cli_scripted_run_hard_cuts_at_context_boundary(tmp_path: Path) -> None:
    """Composition root: a real CLI run hard-cuts at the context boundary.

    Turn 1 stays under the warning line and lands a patch; the huge response
    pushes turn 2's estimate past the hard cut, so the run ends
    staged_unpublished with zero additional model calls and the boundary
    facts recorded in the returned state.
    """
    thread, wake = THREAD, WAKE
    replay_path = tmp_path / f"{thread}-replay.json"
    model_path = tmp_path / f"{thread}-model.json"
    _write_replay_fixture(replay_path)
    _write_model(
        model_path,
        [
            {
                "content": "x" * 140_000,
                "tool_calls": [
                    {
                        "id": "ctx-patch",
                        "name": "graph_patch",
                        "arguments": _patch_call().arguments,
                    }
                ],
            }
        ],
    )
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--perspective",
            "Context boundary probe.",
            "--world-db",
            str(tmp_path / f"{thread}-world.sqlite3"),
            "--runtime-db",
            str(tmp_path / f"{thread}-runtime.sqlite3"),
            "--thread-id",
            thread,
            "--graph-shell",
            "--wake-id",
            wake,
            "--replay-fixture",
            str(replay_path),
            "--scripted-model-fixture",
            str(model_path),
            "--context-warning-tokens",
            "60000",
            "--context-hard-cut-tokens",
            "100000",
        ]
    )
    result = await runner.run(args, print_report=False)

    assert result["terminal_status"] == "staged_unpublished"
    assert result["context_cut"] is True
    assert result["context_usage_estimate"] >= 100_000
    assert "context boundary" in result["terminal_summary"]
