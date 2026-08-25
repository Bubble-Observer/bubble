"""T1: storage faults under the wake are typed, never crashes.

A ``sqlite3.OperationalError`` (most commonly ``database is locked`` while
another process holds the world writer) must surface as a typed limitation at
the tool boundary and as a recoverable unpublished update at the graph
boundary — the wake keeps its staged work and resume is the retry.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world_agent.graph import build_world_agent_graph
from tests.world_agent._graph_helpers import _ScriptedModel


async def test_tools_dispatch_storage_fault_is_typed_limitation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="storage-thread",
        wake_id="storage-wake",
    )

    def broken_patch(self, arguments) -> dict:  # type: ignore[no-untyped-def]
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(WorldTools, "_graph_patch", broken_patch)
    result = await tools.execute(
        "graph_patch", {"items": []}, "call-storage"
    )
    assert result["ok"] is False
    assert result["limitations"] == ["storage_unavailable: OperationalError"]


async def test_graph_agent_storage_fault_is_recoverable_unpublished_update(
    tmp_path: Path,
) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="storage-thread",
        wake_id="storage-wake",
        closed_wake_guard=True,
    )
    model = _ScriptedModel([sqlite3.OperationalError("database is locked")])
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="storage-thread",
        wake_id="storage-wake",
        max_turns=4,
    )
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "begin"}]})
    assert result["terminal_status"] == "staged_unpublished"
    assert result["halted"] is True
    assert result["resume_allowed"] is True
    raw = result["raw_exception"]
    assert raw["type"] == "OperationalError"
    assert raw["recoverable"] is True
    assert "database is locked" in raw["message"]


class _ResumeStorageFaultGraph:
    """Proxy graph whose checkpoint write always hits ``database is locked``.

    Only the resume-path calls are needed: state reads succeed, the
    checkpoint update raises a typed sqlite error, and any accidental fresh
    run (which the fix must prevent after a failed resume) stays real.
    """

    def __init__(self, real) -> None:  # type: ignore[no-untyped-def]
        self._real = real

    async def aget_state(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return await self._real.aget_state(*args, **kwargs)

    async def aupdate_state(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise sqlite3.OperationalError("database is locked")

    async def ainvoke(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return await self._real.ainvoke(*args, **kwargs)


async def test_tools_dispatch_integrity_error_is_constraint_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constraint violation is a contract/program error, never disguised as
    a storage fault (sqlite3.Error surface narrowed)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="storage-thread",
        wake_id="storage-wake",
    )

    def broken_patch(self, arguments) -> dict:  # type: ignore[no-untyped-def]
        raise sqlite3.IntegrityError("UNIQUE constraint failed: inquiries.id")

    monkeypatch.setattr(WorldTools, "_graph_patch", broken_patch)
    result = await tools.execute(
        "graph_patch", {"items": []}, "call-constraint"
    )
    assert result["ok"] is False
    assert result["limitations"] == ["constraint_violation: IntegrityError"]


async def test_cli_resume_storage_fault_is_typed_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint write fault during resume is typed, never a crash: the
    wake keeps its staged work and resume stays the retry (reviewer
    condition: resume path inside the sqlite3.Error boundary)."""
    import leave_information_bubble.world_agent.cli as cli_module
    from tests.world_agent.test_vertical_slice import (
        _load_runner,
        _runner_args,
        _tool_turn,
    )

    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "patch-storage",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "resume-storage-op",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Resume storage draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            )
        ],
        "resume-storage-thread",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"

    real_build = cli_module.build_world_agent_graph
    monkeypatch.setattr(
        cli_module,
        "build_world_agent_graph",
        lambda *args, **kwargs: _ResumeStorageFaultGraph(
            real_build(*args, **kwargs)
        ),
    )
    resume_args = _runner_args(runner, tmp_path, [], "resume-storage-thread")
    resume_args.graph_shell = True
    resume_args.resume = True

    resumed = await runner.run(resume_args, print_report=False)

    assert resumed["terminal_status"] == "staged_unpublished"
    assert resumed["halted"] is True
    assert resumed["resume_allowed"] is True
    assert "storage error" in resumed["terminal_summary"].lower()


async def test_graph_node_keeps_storage_limitation_unrewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storage limitation is not rewritten into "rejected its arguments":
    the Agent must read storage as retry/wind-down, not as a fixable schema
    error (typed codes skip the argument rewrite)."""
    from leave_information_bubble.gateway.client import NativeToolCall
    from tests.world_agent._graph_helpers import _response

    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="storage-thread",
        wake_id="storage-wake",
        closed_wake_guard=True,
    )
    model = _ScriptedModel(
        [
            _response(
                "patch",
                NativeToolCall(
                    id="call-graph_patch",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-storage",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Draft",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response("wind down"),
        ]
    )
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="storage-thread",
        wake_id="storage-wake",
        max_turns=2,
    )

    def broken_patch(self, arguments) -> dict:  # type: ignore[no-untyped-def]
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(WorldTools, "_graph_patch", broken_patch)
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "begin"}]})

    assert result["terminal_status"] == "staged_unpublished"
    tool_messages = " ".join(str(m) for m in model.requests[1])
    assert "storage_unavailable" in tool_messages
    assert "rejected its arguments" not in tool_messages
