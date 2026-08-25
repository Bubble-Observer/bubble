"""G5b-1 Graph Shell crash-window recovery tests (CP1, CP2, CP6).

Every CP has two layers: a graph-level isolation test (this file, below) that
proves the replay/receipt semantics in isolation, and a composition-root test
through the real CLI resume path whose recovery summary is rebuilt entirely
from durable state — the graph-level tests label which layer they prove and
must never be cited alone as full-chain evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from leave_information_bubble.gateway.client import NativeToolCall
from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world_agent import graph as graph_module
from leave_information_bubble.world_agent.graph import (
    build_world_agent_graph,
    graph_shell_initial_state,
)
from tests.world_agent._graph_helpers import _response, _ScriptedModel
from tests.world_agent.test_vertical_slice import (
    _load_runner,
    _tool_turn,
    _write_model,
    _write_replay_fixture,
)


class CountingTools(WorldTools):
    def __init__(self, **kwargs):
        kwargs.setdefault("closed_wake_guard", True)
        super().__init__(**kwargs)
        self.calls: list[tuple[str, str]] = []

    async def execute(self, name, arguments, call_id):
        self.calls.append((name, call_id))
        return await super().execute(name, arguments, call_id)


class FailOnceAfterPatchTools(CountingTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.failed = False

    async def execute(self, name, arguments, call_id):
        result = await super().execute(name, arguments, call_id)
        if name == "graph_patch" and not self.failed:
            self.failed = True
            raise RuntimeError("CP1 after patch commit before tool-result checkpoint")
        return result


def _build(store, saver, model, tools):
    return build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        checkpointer=saver,
        thread_id="thread-recovery",
        wake_id="wake-recovery",
        max_turns=8,
    )


def _patch_call() -> NativeToolCall:
    return NativeToolCall(
        id="patch-call",
        name="graph_patch",
        arguments={
            "items": [
                {
                    "op_id": "stable-op",
                    "kind": "object",
                    "action": "create",
                    "payload": {
                        "canonical_name": "Crash Safe Draft",
                        "kind": "concept",
                        "provisional": True,
                    },
                }
            ]
        },
    )


def _initial_input(store: WorldStore, content: str) -> dict:
    return graph_shell_initial_state(
        messages=[{"role": "user", "content": content}],
        store=store,
        thread_id="thread-recovery",
        wake_id="wake-recovery",
        domain_key="lol_cn",
        mode="broad",
        object_id=None,
    )


async def test_prebootstrap_checkpoint_already_owns_graph_shell_identity(tmp_path: Path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": "thread-recovery"}}
    model = _ScriptedModel(
        [_response("publish", NativeToolCall(id="finalize", name="finalize_graph", arguments={}))]
    )
    tools = CountingTools(
        store=store,
        adapters={},
        thread_id="thread-recovery",
        wake_id="wake-recovery",
    )
    initial = graph_shell_initial_state(
        messages=[{"role": "user", "content": "Checkpoint identity first."}],
        store=store,
        thread_id="thread-recovery",
        wake_id="wake-recovery",
        domain_key="lol_cn",
        mode="broad",
        object_id=None,
    )

    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(store, saver, model, tools)
        interrupted = await graph.ainvoke(initial, config, interrupt_before=["bootstrap"])
        snapshot = await graph.aget_state(config)

        assert interrupted["wake_id"] == "wake-recovery"
        assert snapshot.values["wake_id"] == "wake-recovery"
        assert snapshot.values["thread_id"] == "thread-recovery"
        assert snapshot.values["execution_mode"] == "graph_shell"
        assert snapshot.values["world_store_identity"] == str(store.path.resolve())
        assert snapshot.next == ("bootstrap",)

        result = await graph.ainvoke(None, config)

    assert result["terminal_status"] == "published"
    assert len(model.requests) == 1


async def test_prebootstrap_resume_rejects_a_different_world_store(tmp_path: Path) -> None:
    first_store = WorldStore(tmp_path / "world-a.sqlite3")
    other_store = WorldStore(tmp_path / "world-b.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": "thread-recovery"}}
    first_model = _ScriptedModel([])
    other_model = _ScriptedModel(
        [_response("must not run", NativeToolCall(id="finalize", name="finalize_graph", arguments={}))]
    )
    first_tools = CountingTools(
        store=first_store,
        adapters={},
        thread_id="thread-recovery",
        wake_id="wake-recovery",
    )
    other_tools = CountingTools(
        store=other_store,
        adapters={},
        thread_id="thread-recovery",
        wake_id="wake-recovery",
    )

    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        first_graph = _build(first_store, saver, first_model, first_tools)
        await first_graph.ainvoke(
            _initial_input(first_store, "Checkpoint store A."),
            config,
            interrupt_before=["bootstrap"],
        )
        other_graph = _build(other_store, saver, other_model, other_tools)

        with pytest.raises(ValueError, match="world_store_identity mismatch"):
            await other_graph.ainvoke(None, config)

    assert len(other_model.requests) == 0
    with first_store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM finalize_receipts").fetchone()[0] == 0
    with other_store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM finalize_receipts").fetchone()[0] == 0


async def test_cp1_patch_side_effect_replays_one_ledger_and_staged_row(tmp_path: Path) -> None:
    """[graph-level isolation] CP1 replay semantics: a durable patch side
    effect without a ToolMessage checkpoint replays exactly once, idempotently.

    This test proves the replay mechanics in isolation. It deliberately
    fabricates ``resume_count``/``pending_recovery_summary`` via
    ``aupdate_state`` — that injection is NOT the production recovery path;
    the composition-root close of this loop is
    ``test_cp1_composition_root_resume_rebuilds_summary_after_patch_crash``,
    where the CLI rebuilds the same summary from durable state.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": "thread-recovery"}}
    model = _ScriptedModel(
        [
            _response("patch", _patch_call()),
            _response(
                "inspect replayed work",
                NativeToolCall(id="inspect-after-replay", name="graph_inspect", arguments={}),
            ),
            _response("publish", NativeToolCall(id="finalize", name="finalize_graph", arguments={})),
        ]
    )
    tools = FailOnceAfterPatchTools(
        store=store,
        adapters={},
        thread_id="thread-recovery",
        wake_id="wake-recovery",
    )

    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(store, saver, model, tools)
        with pytest.raises(RuntimeError, match="CP1 after patch commit"):
            await graph.ainvoke(
                _initial_input(store, "Patch safely."),
                config,
            )
        await graph.aupdate_state(
            config,
            {
                "resume_count": 1,
                "pending_recovery_summary": (
                    'Graph Shell durable recovery:\n{"active_total":1,"patch_ledger_count":1}'
                ),
            },
        )
        result = await graph.ainvoke(None, config)

    assert result["terminal_status"] == "published"
    assert result["patch_replay_count"] == 1
    assert result["resume_count"] == 1
    assert tools.calls.count(("graph_patch", "patch-call")) == 2
    assert tools.calls.count(("graph_inspect", "inspect-after-replay")) == 1
    assert "Graph Shell durable recovery:" in model.requests[1][-1]["content"]
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = 'wake-recovery'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts "
            "WHERE wake_id = 'wake-recovery' AND op_id = 'stable-op'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = 'wake-recovery:finalize'"
        ).fetchone()[0] == 1


async def test_cp2_checkpointed_tool_result_resumes_at_next_model_round(tmp_path: Path) -> None:
    """[graph-level isolation] CP2: a checkpointed ToolMessage skips replay.

    Same caveat as CP1 above: the recovery summary here is fabricated by
    ``aupdate_state`` for isolation. The composition-root close of this loop
    is ``test_cp2_composition_root_resume_skips_replay_when_tool_checkpointed``.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": "thread-recovery"}}
    model = _ScriptedModel(
        [
            _response("patch", _patch_call()),
            _response("publish", NativeToolCall(id="finalize", name="finalize_graph", arguments={})),
        ]
    )
    tools = CountingTools(
        store=store,
        adapters={},
        thread_id="thread-recovery",
        wake_id="wake-recovery",
    )

    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(store, saver, model, tools)
        interrupted = await graph.ainvoke(
            _initial_input(store, "Patch then pause."),
            config,
            interrupt_after=["tools"],
        )
        assert interrupted["patch_success_count"] == 1
        assert tools.calls.count(("graph_patch", "patch-call")) == 1

        await graph.aupdate_state(
            config,
            {
                "resume_count": 1,
                "pending_recovery_summary": (
                    'Graph Shell durable recovery:\n{"active_total":1,"patch_ledger_count":1}'
                ),
            },
        )
        result = await graph.ainvoke(None, config)

    assert result["terminal_status"] == "published"
    assert result.get("patch_replay_count", 0) == 0
    assert result["resume_count"] == 1
    assert tools.calls.count(("graph_patch", "patch-call")) == 1
    assert len(model.requests) == 2
    assert "Graph Shell durable recovery:" in model.requests[1][-1]["content"]


async def test_cp6_durable_finalize_receipt_wins_before_tool_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    config = {"configurable": {"thread_id": "thread-recovery"}}
    model = _ScriptedModel(
        [_response("publish", NativeToolCall(id="finalize", name="finalize_graph", arguments={}))]
    )
    tools = CountingTools(
        store=store,
        adapters={},
        thread_id="thread-recovery",
        wake_id="wake-recovery",
    )
    real_finalize = graph_module.finalize_graph
    calls = 0

    def fail_after_first_receipt(world_store, wake_id):
        nonlocal calls
        calls += 1
        receipt = real_finalize(world_store, wake_id)
        if calls == 1:
            assert receipt.status == "published"
            raise RuntimeError("CP6 after receipt before tool-result checkpoint")
        return receipt

    monkeypatch.setattr(graph_module, "finalize_graph", fail_after_first_receipt)

    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(store, saver, model, tools)
        with pytest.raises(RuntimeError, match="CP6 after receipt"):
            await graph.ainvoke(
                _initial_input(store, "Publish safely."),
                config,
            )
        result = await graph.ainvoke(None, config)

    assert result["terminal_status"] == "already_published"
    assert result["finalize_receipt"]["replayed"] is True
    assert calls == 2
    assert len(model.requests) == 1
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = 'wake-recovery:finalize'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = 'wake-recovery'"
        ).fetchone()[0] == 1
        assert sum(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE wake_id = 'wake-recovery' AND status = 'active'"
            ).fetchone()[0]
            for table in ("staged_objects", "staged_assertions", "staged_inquiries")
        ) == 0


def _cli_args(
    runner, tmp_path: Path, thread_id: str, wake_id: str, turns, *, resume: bool
) -> object:
    """Composition-root resume arguments over the crash-construction files."""
    replay_path = tmp_path / f"{thread_id}-replay.json"
    model_path = tmp_path / f"{thread_id}-resume-model.json"
    _write_replay_fixture(replay_path)
    _write_model(model_path, turns)
    args = [
        "--perspective",
        "Composition-root recovery probe.",
        "--world-db",
        str(tmp_path / f"{thread_id}-world.sqlite3"),
        "--runtime-db",
        str(tmp_path / f"{thread_id}-runtime.sqlite3"),
        "--thread-id",
        thread_id,
        "--graph-shell",
        "--wake-id",
        wake_id,
        "--replay-fixture",
        str(replay_path),
        "--scripted-model-fixture",
        str(model_path),
    ]
    if resume:
        args.append("--resume")
    return runner.parse_args(args)


def _recovery_summary_messages(result: dict) -> list[dict]:
    """Every durable recovery summary the resume run added to the transcript."""
    return [
        message
        for message in result["messages"]
        if message.get("role") == "user"
        and str(message.get("content", "")).startswith("Graph Shell durable recovery:\n")
    ]


def _world_after_publish(store: WorldStore, wake_id: str) -> None:
    """The published final state shared by CP1/CP2 composition-root resumes."""
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = ? AND status = 'finalized'",
            (wake_id,),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = 'wake-recovery:finalize'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM world_audit WHERE commit_id = 'wake-recovery:finalize'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM objects WHERE canonical_name = 'Crash Safe Draft'"
        ).fetchone()[0] == 1
        # published releases the CLI-reacquired lease (crash window recovery)
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_shell_writer_leases WHERE singleton_id = 1"
        ).fetchone()[0] == 0


async def test_cp1_composition_root_resume_rebuilds_summary_after_patch_crash(
    tmp_path: Path,
) -> None:
    """CP1 through the real composition root: durable patch, no ToolMessage
    checkpoint, CLI resume rebuilds the summary from durable state.

    The crash state is exactly what a hard process death leaves behind: a
    real graph_patch side effect committed to the world DB (ledger + staging),
    and a runtime checkpoint from before the tool-result write. Resume runs
    through ``cli.run --resume`` — no manual ``aupdate_state`` anywhere — and
    the summary the model sees is the one the CLI builds from the world's
    durable state, not an injected string.
    """
    thread, wake = "thread-recovery", "wake-recovery"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    store = WorldStore(world)
    config = {"configurable": {"thread_id": thread}}
    harness_model = _ScriptedModel([_response("patch", _patch_call())])
    tools = FailOnceAfterPatchTools(store=store, adapters={}, thread_id=thread, wake_id=wake)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(store, saver, harness_model, tools)
        with pytest.raises(RuntimeError, match="CP1 after patch commit"):
            await graph.ainvoke(_initial_input(store, "Patch safely."), config)
    assert len(harness_model.requests) == 1
    assert tools.calls.count(("graph_patch", "patch-call")) == 1
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_shell_writer_leases WHERE singleton_id = 1"
        ).fetchone()[0] == 0  # the crash window holds no lease at all

    runner = _load_runner()
    result = await runner.run(
        _cli_args(
            runner,
            tmp_path,
            thread,
            wake,
            [_tool_turn("finalize", "finalize_graph", {})],
            resume=True,
        ),
        print_report=False,
    )

    assert result["wake_id"] == wake
    assert result["resume_count"] == 1
    assert result["terminal_status"] == "published"
    assert result["patch_replay_count"] == 1
    summaries = _recovery_summary_messages(result)
    assert len(summaries) == 1
    recovery = json.loads(summaries[0]["content"].split("\n", 1)[1])
    assert recovery["wake_id"] == wake
    assert recovery["active_total"] == 1
    assert recovery["patch_ledger_count"] == 1
    # the summary honestly reflects the pre-tools checkpoint: the durable
    # ledger is the world truth, the checkpoint does not know the patch ran
    assert recovery["patch_success_count"] == 0
    assert recovery["patch_replay_count"] == 0
    assert recovery["finalize_receipt"] is False
    _world_after_publish(store, wake)


async def test_cp2_composition_root_resume_skips_replay_when_tool_checkpointed(
    tmp_path: Path,
) -> None:
    """CP2 through the real composition root: ToolMessage checkpointed, crash
    before the next model round, CLI resume continues without re-executing.

    The crash state is the interrupt_after=["tools"] snapshot — the same
    durable shape as a death between the tool-result checkpoint and the next
    model call. The CLI resume must not replay the patch, must rebuild the
    recovery summary from durable state, and must publish with one finalize.
    """
    thread, wake = "thread-recovery", "wake-recovery"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    store = WorldStore(world)
    config = {"configurable": {"thread_id": thread}}
    harness_model = _ScriptedModel([_response("patch", _patch_call())])
    tools = CountingTools(store=store, adapters={}, thread_id=thread, wake_id=wake)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(store, saver, harness_model, tools)
        interrupted = await graph.ainvoke(
            _initial_input(store, "Patch then pause."),
            config,
            interrupt_after=["tools"],
        )
        assert interrupted["patch_success_count"] == 1
    assert tools.calls.count(("graph_patch", "patch-call")) == 1

    runner = _load_runner()
    result = await runner.run(
        _cli_args(
            runner,
            tmp_path,
            thread,
            wake,
            [_tool_turn("finalize", "finalize_graph", {})],
            resume=True,
        ),
        print_report=False,
    )

    assert result["wake_id"] == wake
    assert result["resume_count"] == 1
    assert result["terminal_status"] == "published"
    assert result.get("patch_replay_count", 0) == 0
    summaries = _recovery_summary_messages(result)
    assert len(summaries) == 1
    recovery = json.loads(summaries[0]["content"].split("\n", 1)[1])
    assert recovery["active_total"] == 1
    assert recovery["patch_ledger_count"] == 1
    assert recovery["patch_success_count"] == 1
    assert recovery["patch_replay_count"] == 0
    assert recovery["finalize_receipt"] is False
    _world_after_publish(store, wake)


async def test_cp6_composition_root_resume_receipt_wins_without_model_call(
    tmp_path: Path, monkeypatch
) -> None:
    """CP6 through the real composition root: durable finalize receipt, crash
    before the result checkpoint, CLI resume replays the receipt with zero
    model rounds and zero re-finalization.

    The resume model fixture is empty: any model round would raise "fixture
    exhausted" and fail the run — success is the proof that the receipt path
    calls no model. ``finalize_graph`` is called exactly once in total (the
    crashed run's real finalize); the receipt replay never re-finalizes.
    """
    thread, wake = "thread-recovery", "wake-recovery"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    store = WorldStore(world)
    config = {"configurable": {"thread_id": thread}}
    harness_model = _ScriptedModel(
        [
            _response(
                "publish",
                NativeToolCall(id="finalize", name="finalize_graph", arguments={}),
            )
        ]
    )
    tools = CountingTools(store=store, adapters={}, thread_id=thread, wake_id=wake)
    real_finalize = graph_module.finalize_graph
    calls = 0

    def fail_after_first_receipt(world_store, wake_id):
        nonlocal calls
        calls += 1
        receipt = real_finalize(world_store, wake_id)
        if calls == 1:
            assert receipt.status == "published"
            raise RuntimeError("CP6 after receipt before tool-result checkpoint")
        return receipt

    monkeypatch.setattr(graph_module, "finalize_graph", fail_after_first_receipt)
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        graph = _build(store, saver, harness_model, tools)
        with pytest.raises(RuntimeError, match="CP6 after receipt"):
            await graph.ainvoke(_initial_input(store, "Publish safely."), config)
    assert calls == 1
    assert len(harness_model.requests) == 1

    runner = _load_runner()
    result = await runner.run(
        _cli_args(runner, tmp_path, thread, wake, [], resume=True),
        print_report=False,
    )

    assert result["wake_id"] == wake
    assert result["terminal_status"] == "already_published"
    assert result["finalize_receipt"]["replayed"] is True
    assert calls == 1
    assert _recovery_summary_messages(result) == []
    assert "already published wake" in result["terminal_summary"]
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (wake,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = 'wake-recovery:finalize'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM world_audit WHERE commit_id = 'wake-recovery:finalize'"
        ).fetchone()[0] == 1
        assert sum(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE wake_id = ? AND status = 'active'",
                (wake,),
            ).fetchone()[0]
            for table in ("staged_objects", "staged_assertions", "staged_inquiries")
        ) == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_shell_writer_leases WHERE singleton_id = 1"
        ).fetchone()[0] == 0
