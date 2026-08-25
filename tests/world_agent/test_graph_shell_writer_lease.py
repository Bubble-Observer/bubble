"""Graph Shell singleton writer-lease lifecycle tests (C-1 / F-05)."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from leave_information_bubble.gateway.client import NativeToolCall
from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world.finalize import finalize_graph
from leave_information_bubble.world.initialization import init_blank_database_pair
from leave_information_bubble.world.staging import apply_patch as staging_apply_patch
from leave_information_bubble.world.writer_lease import (
    RELEASE_TERMINALS,
    abandon_writer_lease,
    release_writer_lease,
    writer_lease_action,
)
from leave_information_bubble.world_agent.cli import WorldWriteInProgressError
from leave_information_bubble.world_agent.graph import (
    build_world_agent_graph,
    graph_shell_initial_state,
)
from tests.world_agent._graph_helpers import _response, _ScriptedModel
from tests.world_agent.test_vertical_slice import _load_runner, _write_replay_fixture


def _write_model(path: Path, turns: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")


def _lease_row_count(world_path: Path) -> int:
    connection = sqlite3.connect(world_path)
    try:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM graph_shell_writer_leases WHERE singleton_id = 1"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def _lease_owner(world_path: Path) -> str | None:
    connection = sqlite3.connect(world_path)
    try:
        row = connection.execute(
            "SELECT owner_wake_id FROM graph_shell_writer_leases WHERE singleton_id = 1"
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        connection.close()


def _status_args(
    runner, world_path: Path, runtime_path: Path, thread_id: str
) -> object:
    """Parse a --graph-shell-status management entry (no fixtures needed)."""
    return runner.parse_args(
        [
            "--perspective",
            "Writer-lease status probe.",
            "--world-db",
            str(world_path),
            "--runtime-db",
            str(runtime_path),
            "--thread-id",
            thread_id,
            "--graph-shell",
            "--graph-shell-status",
        ]
    )


def _world_fingerprint(world_path: Path) -> dict[str, int]:
    """Row counts of every user table in *world_path* — the read-only
    contract fingerprint: a status entry must not change any of them."""
    connection = sqlite3.connect(world_path)
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            if not row[0].startswith("sqlite_")
            and not row[0].endswith("_fts")
            and "_fts_" not in row[0]
        ]
        return {
            name: int(connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in sorted(tables)
        }
    finally:
        connection.close()


def _abandon_args(
    runner, world_path: Path, runtime_path: Path, thread_id: str, wake_id: str
) -> object:
    """Parse a --graph-shell-abandon management entry (no fixtures needed)."""
    return runner.parse_args(
        [
            "--perspective",
            "Writer-lease abandon probe.",
            "--world-db",
            str(world_path),
            "--runtime-db",
            str(runtime_path),
            "--thread-id",
            thread_id,
            "--graph-shell",
            "--graph-shell-abandon",
            wake_id,
        ]
    )


def _runner_args(
    runner,
    tmp_path: Path,
    thread_id: str,
    wake_id: str,
    turns: list[dict[str, object]],
    *,
    resume: bool = False,
) -> object:
    replay_path = tmp_path / f"{thread_id}-replay.json"
    model_path = tmp_path / f"{thread_id}-model.json"
    _write_replay_fixture(replay_path)
    _write_model(model_path, turns)
    args = [
        "--perspective",
        "Writer-lease lifecycle probe.",
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


@pytest.mark.parametrize(
    ("terminal", "staging", "expected"),
    [
        ("published", False, "release"),
        ("published", True, "release"),
        ("already_published", False, "release"),
        ("already_published", True, "release"),
        ("wake_closed", False, "release"),
        ("wake_closed", True, "keep_recovery_action"),
        ("blocked", True, "keep"),
        ("compile_failed", True, "keep"),
        ("commit_rejected", True, "keep"),
        ("staged_unpublished", True, "keep"),
        ("staged_unpublished", False, "keep"),
        ("", False, "keep"),
    ],
)
def test_writer_lease_frozen_state_decision_table(
    terminal: str, staging: bool, expected: str
) -> None:
    """F-05 frozen lease lifecycle: every terminal has one deterministic action."""
    assert writer_lease_action(terminal, staging) == expected
    if terminal in RELEASE_TERMINALS:
        assert expected == "release"


async def test_crash_window_blocks_fresh_wake_with_recoverable_message(tmp_path) -> None:
    """C-1 reproduce: a crash after lease acquire and before the first
    checkpoint leaves the gate held with no programmatic unlock — the typed
    busy message must name the real recovery entries, not a nonexistent
    'explicitly abandoned' action.
    """
    runner = _load_runner()
    thread = "lease-crash"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)

    # wake-a acquires the lease, then dies before its first checkpoint (empty
    # scripted fixture raises on the first model invoke, like a hard crash).
    crash_args = _runner_args(runner, tmp_path, thread, "wake-a", [])
    with pytest.raises(RuntimeError, match="fixture exhausted"):
        await runner.run(crash_args, print_report=False)
    assert _lease_owner(world) == "wake-a"

    # fresh wake-b is refused by the atomic lease arbitration
    busy_args = _runner_args(
        runner,
        tmp_path,
        thread,
        "wake-b",
        [
            {
                "content": "must not run",
                "tool_calls": [],
            }
        ],
    )
    with pytest.raises(WorldWriteInProgressError) as error:
        await runner.run(busy_args, print_report=False)
    message = str(error.value)
    assert "world_write_in_progress" in message
    assert "explicitly abandoned" not in message
    assert "--graph-shell-status" in message
    assert "--graph-shell-abandon" in message
    assert _lease_owner(world) == "wake-a"


async def test_resume_foreign_owner_message_names_recovery_entries(tmp_path) -> None:
    """C-1: the resume-side foreign-lease error points at status/abandon.

    The setup deliberately rewrites the lease owner to a foreign wake id —
    the state an operator lands in after abandoning the original owner while
    another wake took the gate — to lock the message contract of the
    fail-closed path.
    """
    runner = _load_runner()
    thread = "lease-resume"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)

    patch_turn = {
        "content": "Stage a provisional concept.",
        "tool_calls": [
            {
                "id": "c1",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Lease Resume Probe",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            }
        ],
    }
    # patch, then two no-call turns: the reminder fires once and the wake
    # stops as staged_unpublished (lease kept by the frozen state machine)
    first_args = _runner_args(
        runner,
        tmp_path,
        thread,
        "wake-a",
        [patch_turn, {"content": "stop", "tool_calls": []}, {"content": "stop", "tool_calls": []}],
    )
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-a"

    # rewrite the gate owner to a foreign wake (post-abandon takeover state)
    connection = sqlite3.connect(world)
    try:
        connection.execute(
            "UPDATE graph_shell_writer_leases SET owner_wake_id = 'wake-foreign' "
            "WHERE singleton_id = 1"
        )
        connection.commit()
    finally:
        connection.close()

    resume_args = _runner_args(
        runner,
        tmp_path,
        thread,
        "wake-a",
        [{"content": "must not run", "tool_calls": []}],
        resume=True,
    )
    with pytest.raises(WorldWriteInProgressError) as error:
        await runner.run(resume_args, print_report=False)
    message = str(error.value)
    assert "world_write_in_progress" in message
    assert "wake-foreign" in message
    assert "explicitly abandoned" not in message
    assert "--graph-shell-status" in message
    assert "--graph-shell-abandon wake-foreign" in message


def test_status_parser_rules(tmp_path) -> None:
    """--graph-shell-status is a management entry: standalone since G5b-2
    (the Graph Shell is the only runtime), never combines with --resume,
    and skips the fixture-pair requirement."""
    runner = _load_runner()
    base = [
        "--world-db",
        str(tmp_path / "status-world.sqlite3"),
        "--runtime-db",
        str(tmp_path / "status-runtime.sqlite3"),
        "--thread-id",
        "status-thread",
    ]
    # G5b-2: --graph-shell is a no-op, so the management entry works without it
    args = runner.parse_args([*base, "--graph-shell-status"])
    assert args.graph_shell_status is True
    with pytest.raises(SystemExit):
        runner.parse_args(
            [*base, "--graph-shell-status", "--resume"]
        )
    # no fixtures required for the management entry
    args = runner.parse_args([*base, "--graph-shell", "--graph-shell-status"])
    assert args.graph_shell_status is True


async def test_status_no_world_is_side_effect_free(tmp_path) -> None:
    """A missing world is reported, never created: the status entry must not
    open WorldStore on a nonexistent path (WorldStore auto-creates)."""
    runner = _load_runner()
    world = tmp_path / "missing-world.sqlite3"
    runtime = tmp_path / "missing-runtime.sqlite3"
    result = await runner.run(
        _status_args(runner, world, runtime, "status-thread"),
        print_report=False,
    )
    assert result["status"] == "no_world"
    assert result["lease"] is None
    assert "nothing to recover" in result["recovery_action"]
    assert not world.exists()
    assert not runtime.exists()


async def test_status_no_lease_reports_no_active_writer_and_mutates_nothing(
    tmp_path,
) -> None:
    """An unleased world reports no_active_writer and the entry mutates
    nothing: world fingerprint before and after are identical."""
    runner = _load_runner()
    world, runtime = await init_blank_database_pair(
        tmp_path / "status-world.sqlite3", tmp_path / "status-runtime.sqlite3"
    )
    before = _world_fingerprint(world)
    result = await runner.run(
        _status_args(runner, world, runtime, "status-thread"),
        print_report=False,
    )
    assert result["status"] == "no_active_writer"
    assert result["lease"] is None
    assert "no active writer lease" in result["recovery_action"]
    assert _world_fingerprint(world) == before
    assert _lease_row_count(world) == 0


async def test_status_crash_before_first_checkpoint_reports_abandon_action(
    tmp_path,
) -> None:
    """Lease committed but ainvoke never started (no checkpoint, no receipt,
    no staging): the deterministic operator action is the explicit abandon
    entry for the exact owner.

    The durable rows are the contract, not how the crash landed: the state is
    constructed the same way the resume-foreign-owner test does — lease
    INSERTed into a fresh world with an empty runtime, exactly what a process
    kill between the lease commit and the first checkpoint write leaves
    behind.
    """
    runner = _load_runner()
    thread = "status-crash"
    world, runtime = await init_blank_database_pair(
        tmp_path / f"{thread}-world.sqlite3", tmp_path / f"{thread}-runtime.sqlite3"
    )
    connection = sqlite3.connect(world)
    try:
        connection.execute(
            "INSERT INTO graph_shell_writer_leases "
            "VALUES (1, 'wake-a', ?, 123456.0)",
            (thread,),
        )
        connection.commit()
    finally:
        connection.close()

    result = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert result["status"] == "lease_held"
    assert result["lease"] == {
        "owner_wake_id": "wake-a",
        "owner_thread_id": thread,
        "claimed_at": 123456.0,
    }
    owner = result["owner_wake"]
    assert owner["wake_claim"] is None
    assert owner["checkpoint_exists"] is False
    assert owner["checkpoint_terminal_status"] == ""
    assert owner["active_staging_count"] == 0
    assert owner["finalize_receipt"] is False
    assert "crash before first checkpoint" in result["recovery_action"]
    assert "--graph-shell-abandon wake-a" in result["recovery_action"]


async def test_status_inprocess_crash_leaves_resumable_checkpoint(tmp_path) -> None:
    """The in-process empty-fixture crash reproduces the lease-hold but lands
    AFTER the first checkpoint write: the status entry must report the
    checkpoint and point at resume, not abandon — evidence that the durable
    projection reads the actual runtime state."""
    runner = _load_runner()
    thread = "status-inprocess-crash"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)

    crash_args = _runner_args(runner, tmp_path, thread, "wake-a", [])
    with pytest.raises(RuntimeError, match="fixture exhausted"):
        await runner.run(crash_args, print_report=False)
    assert _lease_owner(world) == "wake-a"

    result = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert result["status"] == "lease_held"
    owner = result["owner_wake"]
    assert owner["checkpoint_exists"] is True
    assert owner["active_staging_count"] == 0
    assert owner["finalize_receipt"] is False
    assert "--resume --wake-id wake-a" in result["recovery_action"]


async def test_status_staged_unpublished_reports_resume_action(tmp_path) -> None:
    """Lease held with a checkpoint and active staging: resume continues."""
    runner = _load_runner()
    thread = "status-staged"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    patch_turn = {
        "content": "Stage a provisional concept.",
        "tool_calls": [
            {
                "id": "c1",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Status Probe",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            }
        ],
    }
    first = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [patch_turn, {"content": "stop", "tool_calls": []}, {"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-a"

    result = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert result["status"] == "lease_held"
    owner = result["owner_wake"]
    assert owner["checkpoint_exists"] is True
    assert owner["checkpoint_terminal_status"] == "staged_unpublished"
    assert owner["active_staging_count"] == 1
    assert owner["finalize_receipt"] is False
    assert "--resume --wake-id wake-a" in result["recovery_action"]
    # identity observation (F-B follow-up): the staged object declared no
    # aliases, so it must surface as a future twin-object risk.
    assert owner["active_objects_without_aliases"] == ["wake-a:s1"]


async def test_status_staged_object_with_aliases_not_reported_without_aliases(
    tmp_path,
) -> None:
    """A staged object with a declared alias is not flagged as an alias-less
    object; an assertion never appears in the alias-less list either."""
    runner = _load_runner()
    thread = "status-aliased"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    patch_turn = {
        "content": "Stage two objects.",
        "tool_calls": [
            {
                "id": "c1",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Aliased Probe",
                                "kind": "entity",
                                "aliases": ["AP"],
                            },
                        },
                        {
                            "op_id": "op-2",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Bare Probe",
                                "kind": "entity",
                            },
                        },
                    ]
                },
            }
        ],
    }
    first = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [patch_turn, {"content": "stop", "tool_calls": []}, {"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"

    result = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    owner = result["owner_wake"]
    assert owner["active_objects_without_aliases"] == ["wake-a:s2"]


async def test_status_published_releases_lease_then_typed_receipt_replay(
    tmp_path,
) -> None:
    """A real publish cycle releases the lease (no_active_writer), and the
    crash-between-receipt-and-release state — lease row still present next to
    the durable receipt — is typed as the already_published replay path."""
    runner = _load_runner()
    thread = "status-published"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    patch_turn = {
        "content": "Stage a provisional concept.",
        "tool_calls": [
            {
                "id": "c1",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Status Publish Probe",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            }
        ],
    }
    publish_turn = {
        "content": "Publish the draft.",
        "tool_calls": [{"id": "c2", "name": "finalize_graph", "arguments": {}}],
    }
    first = await runner.run(
        _runner_args(runner, tmp_path, thread, "wake-a", [patch_turn, publish_turn]),
        print_report=False,
    )
    assert first["terminal_status"] == "published"
    assert _lease_row_count(world) == 0

    result = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert result["status"] == "no_active_writer"

    # crash between the finalize transaction and the lease release: the
    # durable receipt now sits next to a still-held lease
    connection = sqlite3.connect(world)
    try:
        connection.execute(
            "INSERT INTO graph_shell_writer_leases "
            "VALUES (1, 'wake-a', ?, 123456.0)",
            (thread,),
        )
        connection.commit()
    finally:
        connection.close()

    result = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert result["status"] == "lease_held"
    owner = result["owner_wake"]
    assert owner["checkpoint_exists"] is True
    assert owner["checkpoint_terminal_status"] == "published"
    assert owner["active_staging_count"] == 0
    assert owner["finalize_receipt"] is True
    assert "--resume --wake-id wake-a" in result["recovery_action"]
    assert "already_published" in result["recovery_action"]


async def test_status_lease_held_is_repeatably_read_only(tmp_path) -> None:
    """Two status calls on the same held lease return the same projection and
    leave every world table's row count untouched."""
    runner = _load_runner()
    thread = "status-readonly"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    patch_turn = {
        "content": "Stage a provisional concept.",
        "tool_calls": [
            {
                "id": "c1",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Status Readonly Probe",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            }
        ],
    }
    await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [patch_turn, {"content": "stop", "tool_calls": []}, {"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    before = _world_fingerprint(world)
    first = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    second = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert first == second
    assert first["status"] == "lease_held"
    assert _world_fingerprint(world) == before
    assert _lease_owner(world) == "wake-a"


def test_abandon_parser_rules(tmp_path) -> None:
    """--graph-shell-abandon is a management entry: standalone since G5b-2
    (the Graph Shell is the only runtime), never combines with --resume or
    --graph-shell-status, needs the wake id."""
    runner = _load_runner()
    base = [
        "--world-db",
        str(tmp_path / "abandon-world.sqlite3"),
        "--runtime-db",
        str(tmp_path / "abandon-runtime.sqlite3"),
        "--thread-id",
        "abandon-thread",
    ]
    # G5b-2: --graph-shell is a no-op, so the management entry works without it
    args = runner.parse_args([*base, "--graph-shell-abandon", "wake-a"])
    assert args.graph_shell_abandon == "wake-a"
    with pytest.raises(SystemExit):
        runner.parse_args(
            [*base, "--graph-shell-abandon", "wake-a", "--resume"]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                *base,
                "--graph-shell-status",
                "--graph-shell-abandon",
                "wake-a",
            ]
        )
    with pytest.raises(SystemExit):
        runner.parse_args([*base, "--graph-shell-abandon"])
    args = runner.parse_args([*base, "--graph-shell", "--graph-shell-abandon", "wake-a"])
    assert args.graph_shell_abandon == "wake-a"


async def test_abandon_no_world_is_side_effect_free(tmp_path) -> None:
    """A missing world is reported, never created."""
    runner = _load_runner()
    world = tmp_path / "missing-abandon-world.sqlite3"
    runtime = tmp_path / "missing-abandon-runtime.sqlite3"
    result = await runner.run(
        _abandon_args(runner, world, runtime, "abandon-thread", "wake-a"),
        print_report=False,
    )
    assert result["status"] == "no_world"
    assert not world.exists()
    assert not runtime.exists()


async def test_abandon_no_lease_reports_already_abandoned(tmp_path) -> None:
    """An unleased world answers already_abandoned and mutates nothing —
    repeated runs are harmless (idempotent by construction)."""
    runner = _load_runner()
    world, runtime = await init_blank_database_pair(
        tmp_path / "abandon-world.sqlite3", tmp_path / "abandon-runtime.sqlite3"
    )
    before = _world_fingerprint(world)
    result = await runner.run(
        _abandon_args(runner, world, runtime, "abandon-thread", "wake-a"),
        print_report=False,
    )
    assert result["status"] == "already_abandoned"
    assert result["lease_released"] is False
    result = await runner.run(
        _abandon_args(runner, world, runtime, "abandon-thread", "wake-a"),
        print_report=False,
    )
    assert result["status"] == "already_abandoned"
    assert _world_fingerprint(world) == before


async def test_abandon_owner_mismatch_fails_closed(tmp_path) -> None:
    """Abandoning a different owner than the lease holder is refused with the
    actual owner named, and the durable state is untouched."""
    runner = _load_runner()
    thread = "abandon-mismatch"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    patch_turn = {
        "content": "Stage a provisional concept.",
        "tool_calls": [
            {
                "id": "c1",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Abandon Mismatch Probe",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            }
        ],
    }
    await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [patch_turn, {"content": "stop", "tool_calls": []}, {"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    assert _lease_owner(world) == "wake-a"
    before = _world_fingerprint(world)

    with pytest.raises(ValueError) as error:
        await runner.run(
            _abandon_args(runner, world, runtime, thread, "wake-b"),
            print_report=False,
        )
    message = str(error.value)
    assert "graph_shell_abandon_owner_mismatch" in message
    assert "wake-a" in message
    assert "wake-b" in message
    assert _lease_owner(world) == "wake-a"
    assert _world_fingerprint(world) == before


async def test_abandon_releases_gate_and_frees_world_for_fresh_wake(tmp_path) -> None:
    """The full recovery loop: a stuck staged_unpublished wake owns the gate
    and a runtime wake claim; the explicit abandon releases both in one world
    transaction, and a fresh wake can then take the gate and keep working."""
    runner = _load_runner()
    thread = "abandon-recover"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    patch_turn = {
        "content": "Stage a provisional concept.",
        "tool_calls": [
            {
                "id": "c1",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Abandon Recovery Probe",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            }
        ],
    }
    first = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [patch_turn, {"content": "stop", "tool_calls": []}, {"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-a"
    connection = sqlite3.connect(runtime)
    try:
        claim = connection.execute(
            "SELECT 1 FROM graph_shell_wake_claims WHERE wake_id = 'wake-a'"
        ).fetchone()
    finally:
        connection.close()
    assert claim is not None

    result = await runner.run(
        _abandon_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert result["status"] == "abandoned"
    assert result["abandoned_items"] == {
        "staged_objects": 1,
        "staged_assertions": 0,
        "staged_inquiries": 0,
    }
    assert result["lease_released"] is True
    assert result["runtime_claims_cleaned"] is True
    assert _lease_row_count(world) == 0
    connection = sqlite3.connect(runtime)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM graph_shell_wake_claims WHERE wake_id = 'wake-a'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()
    # the staging rows are marked abandoned, not deleted
    connection = sqlite3.connect(world)
    try:
        row = connection.execute(
            "SELECT status FROM staged_objects WHERE wake_id = 'wake-a'"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None and row[0] == "abandoned"

    # the gate is free: status agrees, and a fresh wake takes it and works
    status = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert status["status"] == "no_active_writer"
    fresh = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-b",
            [patch_turn, {"content": "stop", "tool_calls": []}, {"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    assert fresh["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-b"


async def test_abandon_repeat_is_already_abandoned(tmp_path) -> None:
    """Abandoning an already-abandoned wake reports already_abandoned and
    leaves the fresh gate untouched."""
    runner = _load_runner()
    thread = "abandon-repeat"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    patch_turn = {
        "content": "Stage a provisional concept.",
        "tool_calls": [
            {
                "id": "c1",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Abandon Repeat Probe",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            }
        ],
    }
    await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [patch_turn, {"content": "stop", "tool_calls": []}, {"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    first = await runner.run(
        _abandon_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert first["status"] == "abandoned"
    repeat = await runner.run(
        _abandon_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert repeat["status"] == "already_abandoned"
    assert repeat["lease_released"] is False
    assert repeat["abandoned_items"] == 0
    assert _lease_row_count(world) == 0


# --- C-1e: mid-run lease re-validation (writer_lease_lost) ----------------


class _PausingModel(_ScriptedModel):
    """Scripted model that parks the run before a chosen model turn so the
    test can release the writer lease mid-run (the C-1e interleave)."""

    def __init__(self, responses, *, pause_before_turn: int) -> None:
        super().__init__(responses)
        self.pause_before_turn = pause_before_turn
        self.reached_pause = asyncio.Event()
        self.resume = asyncio.Event()
        self._invoked = 0

    async def invoke_tools(self, messages, *, tools, **kwargs):
        self._invoked += 1
        if self._invoked == self.pause_before_turn:
            self.reached_pause.set()
            await self.resume.wait()
        return await super().invoke_tools(messages, tools=tools, **kwargs)


def _lost_patch(op_id: str, call_id: str) -> NativeToolCall:
    return NativeToolCall(
        id=call_id,
        name="graph_patch",
        arguments={
            "items": [
                {
                    "op_id": op_id,
                    "kind": "object",
                    "action": "create",
                    "payload": {
                        "canonical_name": f"Lost Lease Probe {op_id}",
                        "kind": "concept",
                        "provisional": True,
                    },
                }
            ]
        },
    )


def _lost_initial(store: WorldStore) -> dict:
    return graph_shell_initial_state(
        messages=[{"role": "user", "content": "Edit the working graph."}],
        store=store,
        thread_id="thread-lost",
        wake_id="wake-lost",
        domain_key="lol_cn",
        mode="broad",
        object_id=None,
    )


def _lost_graph(
    store: WorldStore,
    saver,
    model,
    *,
    enforce_writer_lease: bool,
) -> object:
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="thread-lost",
        wake_id="wake-lost",
        closed_wake_guard=True,
    )
    return (
        tools,
        build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            checkpointer=saver,
            thread_id="thread-lost",
            wake_id="wake-lost",
            max_turns=8,
            enforce_writer_lease=enforce_writer_lease,
        ),
    )


def _seed_lease(world_path: Path) -> None:
    connection = sqlite3.connect(world_path)
    try:
        connection.execute(
            "INSERT INTO graph_shell_writer_leases "
            "VALUES (1, 'wake-lost', 'thread-lost', 123456.0)"
        )
        connection.commit()
    finally:
        connection.close()


def _staged_rows(store: WorldStore) -> list[tuple[str, str]]:
    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT staged_id, status FROM staged_objects WHERE wake_id = 'wake-lost'"
        ).fetchall()
    return [(str(row["staged_id"]), str(row["status"])) for row in rows]


def _tool_transcript(result: dict) -> list[str]:
    return [
        str(message.get("content"))
        for message in result.get("messages", [])
        if message.get("role") == "tool"
    ]


async def test_writer_lease_lost_blocks_midrun_patch_with_typed_error(tmp_path) -> None:
    """C-1e: every graph_patch dispatch re-validates the singleton lease; an
    operator abandon mid-run fails the wake closed with the typed
    writer_lease_lost and no further staged mutation lands."""
    store = WorldStore(tmp_path / "lost-world.sqlite3")
    runtime = tmp_path / "lost-runtime.sqlite3"
    _seed_lease(store.path)
    model = _PausingModel(
        [
            _response("edit", _lost_patch("op-1", "patch-1")),
            _response("edit", _lost_patch("op-2", "patch-2")),
        ],
        pause_before_turn=2,
    )
    config = {"configurable": {"thread_id": "thread-lost"}}
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        tools, graph = _lost_graph(store, saver, model, enforce_writer_lease=True)
        run = asyncio.create_task(graph.ainvoke(_lost_initial(store), config))
        await model.reached_pause.wait()
        # patch 1 is durable; the operator releases the gate mid-run
        assert _staged_rows(store) == [("wake-lost:s1", "active")]
        abandoned, released = abandon_writer_lease(store, "wake-lost")
        assert sum(abandoned.values()) == 1
        assert released is True
        model.resume.set()
        result = await run

    assert result["terminal_status"] == "staged_unpublished"
    assert "Writer lease lost" in result["terminal_summary"]
    assert any('"code":"writer_lease_lost"' in text for text in _tool_transcript(result))
    # patch 2 never executed: exactly one staged row, marked abandoned
    assert _staged_rows(store) == [("wake-lost:s1", "abandoned")]


async def test_writer_lease_lost_blocks_midrun_finalize_with_typed_error(tmp_path) -> None:
    """C-1e: finalize_graph re-validates the lease too; after a mid-run
    abandon it is refused with writer_lease_lost and no durable receipt is
    ever written."""
    store = WorldStore(tmp_path / "lost-finalize-world.sqlite3")
    runtime = tmp_path / "lost-finalize-runtime.sqlite3"
    _seed_lease(store.path)
    model = _PausingModel(
        [
            _response("edit", _lost_patch("op-1", "patch-1")),
            _response(
                "publish",
                NativeToolCall(id="finalize-2", name="finalize_graph", arguments={}),
            ),
        ],
        pause_before_turn=2,
    )
    config = {"configurable": {"thread_id": "thread-lost"}}
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        tools, graph = _lost_graph(store, saver, model, enforce_writer_lease=True)
        run = asyncio.create_task(graph.ainvoke(_lost_initial(store), config))
        await model.reached_pause.wait()
        abandon_writer_lease(store, "wake-lost")
        model.resume.set()
        result = await run

    assert result["terminal_status"] == "staged_unpublished"
    assert any('"code":"writer_lease_lost"' in text for text in _tool_transcript(result))
    with store.read_connection() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM finalize_receipts WHERE wake_id = 'wake-lost'"
            ).fetchone()
            is None
        )


async def test_writer_lease_guard_defaults_off_for_direct_graph_harnesses(
    tmp_path,
) -> None:
    """Direct graph harnesses drive the shell graph without the CLI's lease
    machinery: with the guard off (default), a missing lease row is not
    mistaken for a lost lease and patches still execute."""
    store = WorldStore(tmp_path / "guard-off-world.sqlite3")
    runtime = tmp_path / "guard-off-runtime.sqlite3"
    model = _ScriptedModel(
        [
            _response("edit", _lost_patch("op-1", "patch-1")),
            _response("stop"),
            _response("stop"),
        ]
    )
    config = {"configurable": {"thread_id": "thread-lost"}}
    async with AsyncSqliteSaver.from_conn_string(str(runtime)) as saver:
        tools, graph = _lost_graph(store, saver, model, enforce_writer_lease=False)
        result = await graph.ainvoke(_lost_initial(store), config)

    assert result["terminal_status"] == "staged_unpublished"
    assert not any('"code":"writer_lease_lost"' in text for text in _tool_transcript(result))
    assert _staged_rows(store) == [("wake-lost:s1", "active")]
    assert _lease_row_count(store.path) == 0


async def test_cli_resume_reopens_abandoned_wake(tmp_path) -> None:
    """Resume after abandon re-acquires the freed gate: the operator's
    deliberate re-entry with the same wake identity continues to work — the
    resume-reopen contract behind writer_lease_lost."""
    runner = _load_runner()
    thread = "resume-reopen"
    world, runtime = await init_blank_database_pair(
        tmp_path / f"{thread}-world.sqlite3", tmp_path / f"{thread}-runtime.sqlite3"
    )

    def patch_turn(op_id: str) -> dict:
        return {
            "content": "Stage a provisional concept.",
            "tool_calls": [
                {
                    "id": f"c-{op_id}",
                    "name": "graph_patch",
                    "arguments": {
                        "items": [
                            {
                                "op_id": op_id,
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": f"Resume Reopen Probe {op_id}",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                }
            ],
        }

    stop = {"content": "stop", "tool_calls": []}
    first = await runner.run(
        _runner_args(
            runner, tmp_path, thread, "wake-a", [patch_turn("op-1"), stop, stop]
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"

    abandon = await runner.run(
        _abandon_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert abandon["status"] == "abandoned"

    resumed = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [patch_turn("op-2"), stop],
            resume=True,
        ),
        print_report=False,
    )
    assert resumed["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-a"
    # the pre-abandon item stays abandoned; the post-resume item is active
    connection = sqlite3.connect(world)
    try:
        rows = connection.execute(
            "SELECT staged_id, status FROM staged_objects WHERE wake_id = 'wake-a'"
        ).fetchall()
    finally:
        connection.close()
    assert [(str(row[0]), str(row[1])) for row in rows] == [
        ("wake-a:s1", "abandoned"),
        ("wake-a:s2", "active"),
    ]


# ── M-3 (G5b-1): --graph-shell-status must never migrate a pre-v17 world ───

def _strip_writer_lease_migration(world: Path) -> None:
    """Downgrade a fresh world to the exact pre-v17 shape: the v17 schema
    migration creates only the singleton writer-lease table, so dropping it
    and rewinding user_version reproduces a genuine pre-v17 world."""
    connection = sqlite3.connect(world)
    try:
        connection.execute("DROP TABLE graph_shell_writer_leases")
        connection.execute("PRAGMA user_version = 16")
        connection.commit()
    finally:
        connection.close()


async def test_status_pre_v17_world_is_typed_and_never_migrated(tmp_path) -> None:
    """M-3: a pre-v17 world (no writer-lease table) reports a typed
    no_active_writer status and the entry leaves user_version, the table set
    and the file untouched — a status check never migrates a historical
    world; the schema upgrade stays with real fresh/resume runs."""
    runner = _load_runner()
    thread = "pre17"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    _strip_writer_lease_migration(world)
    before = _world_fingerprint(world)
    before_version = int(
        sqlite3.connect(world).execute("PRAGMA user_version").fetchone()[0]
    )
    before_mtime_ns = world.stat().st_mtime_ns
    assert before_version == 16

    result = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert result["status"] == "no_active_writer"
    assert result["schema"] == "legacy_schema_no_writer_lease"
    assert result["lease"] is None
    assert "predates" in result["recovery_action"]

    assert _world_fingerprint(world) == before
    assert "graph_shell_writer_leases" not in before
    connection = sqlite3.connect(world)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 16
    finally:
        connection.close()
    assert world.stat().st_mtime_ns == before_mtime_ns

    # contrast: a real fresh wake still upgrades the world under the normal
    # migration discipline (WorldStore initializes the schema on open)
    wake = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [{"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    assert wake["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-a"
    connection = sqlite3.connect(world)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 18
        assert _lease_row_count(world) == 1
    finally:
        connection.close()


async def test_status_pre_v17_readonly_world_is_typed_without_traceback(
    tmp_path,
) -> None:
    """M-3: a pre-v17 world that rejects writes (Windows read-only
    attribute) must yield the typed no_active_writer status — never a raw
    sqlite3.OperationalError from an attempted schema-migration write."""
    runner = _load_runner()
    thread = "pre17-ro"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    _strip_writer_lease_migration(world)
    os.chmod(world, stat.S_IREAD)
    try:
        result = await runner.run(
            _status_args(runner, world, runtime, thread), print_report=False
        )
    finally:
        os.chmod(world, stat.S_IREAD | stat.S_IWRITE)
    assert result["status"] == "no_active_writer"
    assert result["schema"] == "legacy_schema_no_writer_lease"
    connection = sqlite3.connect(world)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 16
        assert "graph_shell_writer_leases" not in _world_fingerprint(world)
    finally:
        connection.close()


# ── M-4 (G5b-1): writer-lease decision branches at the composition root ────

async def test_composition_root_post_publish_patch_is_skipped_and_lease_released(
    tmp_path,
) -> None:
    """M-4/guard: after publish the tool loop halts at the terminal and a
    later patch in the same message is skipped as
    tool_call_skipped_after_terminal — the composition-root mechanism that
    makes stranded post-receipt staging unreachable for a CLI wake (so the
    keep_recovery_action branch stays defense-in-depth)."""
    runner = _load_runner()
    thread = "lease-guard"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)

    def patch_turn(op_id: str) -> dict:
        return {
            "content": "Stage a provisional concept.",
            "tool_calls": [
                {
                    "id": f"c-{op_id}",
                    "name": "graph_patch",
                    "arguments": {
                        "items": [
                            {
                                "op_id": op_id,
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": f"Guard Probe {op_id}",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                }
            ],
        }

    publish_and_late_patch = {
        "content": "Publish and keep editing.",
        "tool_calls": [
            {"id": "c2", "name": "finalize_graph", "arguments": {}},
            {
                "id": "c3",
                "name": "graph_patch",
                "arguments": {
                    "items": [
                        {
                            "op_id": "op-2",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Late Patch After Publish",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            },
        ],
    }
    result = await runner.run(
        _runner_args(
            runner, tmp_path, thread, "wake-a", [patch_turn("op-1"), publish_and_late_patch]
        ),
        print_report=False,
    )
    assert result["terminal_status"] == "published"
    assert result["finalize_status"] == "published"
    # the late patch was never executed: it is skipped with a typed marker
    assert "tool_call_skipped_after_terminal" in json.dumps(
        result.get("messages", []), ensure_ascii=False
    )
    # nothing was staged after publish; the gate is released
    assert _lease_owner(world) is None
    status = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert status["status"] == "no_active_writer"


async def test_composition_root_stranded_wake_operator_contract(tmp_path) -> None:
    """M-4 scenario 1: the stranded world state the keep_recovery_action
    branch defends — durable receipt plus post-receipt active staging plus a
    still-held lease — has the full operator contract at the composition
    root: status names the abandon action, resume fails closed, fresh wakes
    are refused, and the named abandon entry recovers the world."""
    runner = _load_runner()
    thread = "lease-stranded"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)

    def patch_turn(op_id: str) -> dict:
        return {
            "content": "Stage a provisional concept.",
            "tool_calls": [
                {
                    "id": f"c-{op_id}",
                    "name": "graph_patch",
                    "arguments": {
                        "items": [
                            {
                                "op_id": op_id,
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": f"Stranded Probe {op_id}",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                }
            ],
        }

    first = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [
                patch_turn("op-1"),
                {
                    "content": "Publish the draft.",
                    "tool_calls": [
                        {"id": "c2", "name": "finalize_graph", "arguments": {}}
                    ],
                },
            ],
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "published"
    assert _lease_owner(world) is None

    # the stranded window (F-3): a late patch that raced the receipt lands
    # at the store level (the closed-wake guard lives in the tool facade,
    # not the staging engine), and the process crashed before the lease
    # release — the exact state keep_recovery_action defends
    store = WorldStore(world)
    staging_apply_patch(
        store,
        "wake-a",
        [
            {
                "op_id": "op-late",
                "kind": "object",
                "action": "create",
                "payload": {
                    "canonical_name": "Stranded Late Item",
                    "kind": "concept",
                    "provisional": True,
                },
            }
        ],
    )
    connection = sqlite3.connect(world)
    try:
        connection.execute(
            "INSERT INTO graph_shell_writer_leases VALUES (1, 'wake-a', ?, 123456.0)",
            (thread,),
        )
        connection.commit()
    finally:
        connection.close()

    status = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert status["status"] == "lease_held"
    assert status["owner_wake"]["finalize_receipt"] is True
    assert status["owner_wake"]["active_staging_count"] == 1
    assert "--graph-shell-abandon wake-a" in status["recovery_action"]

    # resume fails closed: a durable receipt plus active staging can never
    # converge under the same wake identity
    with pytest.raises(ValueError, match="resume fails closed"):
        await runner.run(
            _runner_args(runner, tmp_path, thread, "wake-a", [], resume=True),
            print_report=False,
        )
    # fresh wakes are refused while the stranded gate is held
    with pytest.raises(WorldWriteInProgressError):
        await runner.run(
            _runner_args(
                runner,
                tmp_path,
                thread,
                "wake-b",
                [{"content": "must not run", "tool_calls": []}],
            ),
            print_report=False,
        )
    assert _lease_owner(world) == "wake-a"

    # the named recovery completes: abandon marks the stranded items and
    # releases the gate, then a fresh wake acquires
    abandoned = await runner.run(
        _abandon_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert abandoned["status"] == "abandoned"
    assert abandoned["abandoned_items"] == {
        "staged_objects": 1,
        "staged_assertions": 0,
        "staged_inquiries": 0,
    }
    assert abandoned["lease_released"] is True
    assert _lease_owner(world) is None
    fresh = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-c",
            [{"content": "stop", "tool_calls": []}],
        ),
        print_report=False,
    )
    assert fresh["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-c"


async def test_composition_root_concurrent_finalize_releases_lease_for_fresh_wake(
    tmp_path,
) -> None:
    """M-4 scenario 2: a wake whose staging was finalized concurrently
    (durable receipt, nothing active) releases the writer gate through the
    composition root — same-wake resume replays the durable receipt as
    already_published and a fresh wake immediately acquires."""
    runner = _load_runner()
    thread = "lease-concurrent"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)

    def patch_turn(op_id: str) -> dict:
        return {
            "content": "Stage a provisional concept.",
            "tool_calls": [
                {
                    "id": f"c-{op_id}",
                    "name": "graph_patch",
                    "arguments": {
                        "items": [
                            {
                                "op_id": op_id,
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": f"Concurrent Probe {op_id}",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                }
            ],
        }

    stop = {"content": "stop", "tool_calls": []}
    first = await runner.run(
        _runner_args(runner, tmp_path, thread, "wake-a", [patch_turn("op-1"), stop, stop]),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-a"

    # the concurrent process finalizes the wake's staging (F-3): the
    # durable receipt lands and the staging is consumed; this process
    # "crashed" before its own lease release
    receipt = finalize_graph(WorldStore(world), "wake-a")
    assert receipt.status == "published"
    assert _lease_owner(world) == "wake-a"

    # same-wake resume replays the durable receipt and releases the gate
    resumed = await runner.run(
        _runner_args(runner, tmp_path, thread, "wake-a", [], resume=True),
        print_report=False,
    )
    assert resumed["terminal_status"] == "already_published"
    assert _lease_owner(world) is None

    # fresh wake-b acquires the released gate immediately
    fresh = await runner.run(
        _runner_args(runner, tmp_path, thread, "wake-b", [stop]),
        print_report=False,
    )
    assert fresh["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-b"


# ── --graph-shell-restore: re-activate an abandoned working graph ─────────────


def _restore_args(
    runner, world_path: Path, runtime_path: Path, thread_id: str, wake_id: str
) -> object:
    """Parse a --graph-shell-restore management entry (no fixtures needed)."""
    return runner.parse_args(
        [
            "--perspective",
            "Writer-lease restore probe.",
            "--world-db",
            str(world_path),
            "--runtime-db",
            str(runtime_path),
            "--thread-id",
            thread_id,
            "--graph-shell",
            "--graph-shell-restore",
            wake_id,
        ]
    )


def _patch_turn(op_id: str, name: str) -> dict[str, object]:
    """A scripted graph_patch turn that stages one provisional concept."""
    return {
        "content": "Stage a provisional concept.",
        "tool_calls": [
            {
                "id": op_id,
                "name": "graph_patch",
                "arguments": {
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
            }
        ],
    }


async def _staged_then_abandoned(
    runner, tmp_path: Path, thread: str, wake_id: str
) -> tuple[Path, Path]:
    """Script one wake that stages a single concept and stops (staged_unpublished),
    then release it via the abandon entry. Returns (world, runtime)."""
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    stop = {"content": "stop", "tool_calls": []}
    first = await runner.run(
        _runner_args(
            runner, tmp_path, thread, wake_id, [_patch_turn(f"op-{wake_id}", "Restore Probe"), stop, stop]
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"
    result = await runner.run(
        _abandon_args(runner, world, runtime, thread, wake_id),
        print_report=False,
    )
    assert result["status"] == "abandoned"
    return world, runtime


def test_restore_parser_rules(tmp_path) -> None:
    """--graph-shell-restore is a management entry: standalone, never combines
    with --resume / --graph-shell-status / --graph-shell-abandon, and needs
    the wake id."""
    runner = _load_runner()
    base = [
        "--world-db",
        str(tmp_path / "restore-world.sqlite3"),
        "--runtime-db",
        str(tmp_path / "restore-runtime.sqlite3"),
        "--thread-id",
        "restore-thread",
    ]
    args = runner.parse_args([*base, "--graph-shell-restore", "wake-a"])
    assert args.graph_shell_restore == "wake-a"
    with pytest.raises(SystemExit):
        runner.parse_args([*base, "--graph-shell-restore", "wake-a", "--resume"])
    with pytest.raises(SystemExit):
        runner.parse_args(
            [*base, "--graph-shell-status", "--graph-shell-restore", "wake-a"]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [*base, "--graph-shell-abandon", "wake-a", "--graph-shell-restore", "wake-a"]
        )
    with pytest.raises(SystemExit):
        runner.parse_args([*base, "--graph-shell-restore"])
    args = runner.parse_args([*base, "--graph-shell", "--graph-shell-restore", "wake-a"])
    assert args.graph_shell_restore == "wake-a"


async def test_restore_no_world_is_side_effect_free(tmp_path) -> None:
    """A missing world is reported, never created."""
    runner = _load_runner()
    world = tmp_path / "missing-restore-world.sqlite3"
    runtime = tmp_path / "missing-restore-runtime.sqlite3"
    result = await runner.run(
        _restore_args(runner, world, runtime, "restore-thread", "wake-a"),
        print_report=False,
    )
    assert result["status"] == "no_world"
    assert not world.exists()
    assert not runtime.exists()


async def test_restore_nothing_to_restore_on_fresh_world(tmp_path) -> None:
    """A world with no abandoned staging and no receipt answers
    nothing_to_restore and mutates nothing."""
    runner = _load_runner()
    world, runtime = await init_blank_database_pair(
        tmp_path / "restore-world.sqlite3", tmp_path / "restore-runtime.sqlite3"
    )
    before = _world_fingerprint(world)
    result = await runner.run(
        _restore_args(runner, world, runtime, "restore-thread", "wake-a"),
        print_report=False,
    )
    assert result["status"] == "nothing_to_restore"
    assert result["reason"] == "no_abandoned_staging"
    assert result["restored_items"] == 0
    assert _world_fingerprint(world) == before


async def test_restore_published_wake_is_nothing_to_restore(tmp_path) -> None:
    """A published wake's dropped rows are history, not working graph:
    restore refuses with already_published and mutates nothing."""
    runner = _load_runner()
    thread = "restore-published"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    stop = {"content": "stop", "tool_calls": []}
    first = await runner.run(
        _runner_args(
            runner, tmp_path, thread, "wake-a", [_patch_turn("op-pub", "Published Probe"), stop, stop]
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"
    receipt = finalize_graph(WorldStore(world), "wake-a")
    assert receipt.status == "published"
    # a normal published run releases the lease; the direct finalize_graph
    # below is the "crashed before lease release" shortcut, so release it
    # explicitly to model the normal published state
    release_writer_lease(WorldStore(world), "wake-a")
    assert _lease_owner(world) is None
    before = _world_fingerprint(world)
    result = await runner.run(
        _restore_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert result["status"] == "nothing_to_restore"
    assert result["reason"] == "already_published"
    assert _world_fingerprint(world) == before


async def test_restore_owner_mismatch_fails_closed(tmp_path) -> None:
    """A lease held by another wake refuses restore with the real owner named;
    the still-active owner itself reports wake_still_active (its abandoned
    rows are mid-run drops, not a whole-wake abandon)."""
    runner = _load_runner()
    thread = "restore-mismatch"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    stop = {"content": "stop", "tool_calls": []}
    first = await runner.run(
        _runner_args(
            runner, tmp_path, thread, "wake-a", [_patch_turn("op-a", "Mismatch Probe"), stop, stop]
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"
    assert _lease_owner(world) == "wake-a"
    before = _world_fingerprint(world)

    with pytest.raises(ValueError) as error:
        await runner.run(
            _restore_args(runner, world, runtime, thread, "wake-b"),
            print_report=False,
        )
    message = str(error.value)
    assert "graph_shell_restore_owner_mismatch" in message
    assert "wake-a" in message
    assert "wake-b" in message

    self_restore = await runner.run(
        _restore_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert self_restore["status"] == "nothing_to_restore"
    assert self_restore["reason"] == "wake_still_active"
    assert _lease_owner(world) == "wake-a"
    assert _world_fingerprint(world) == before


async def test_restore_full_recovery_loop_publishes(tmp_path) -> None:
    """The full restore journey: abandon a staged_unpublished wake, restore
    its working graph, resume the same wake and publish — the abandoned work
    is recoverable end to end."""
    runner = _load_runner()
    thread = "restore-loop"
    world, runtime = await _staged_then_abandoned(runner, tmp_path, thread, "wake-a")
    assert _lease_owner(world) is None

    restored = await runner.run(
        _restore_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert restored["status"] == "restored"
    assert restored["restored_items"] == {
        "staged_objects": 1,
        "staged_assertions": 0,
        "staged_inquiries": 0,
    }
    assert restored["lease_acquired"] is True
    assert _lease_owner(world) == "wake-a"
    connection = sqlite3.connect(world)
    try:
        status = connection.execute(
            "SELECT status FROM staged_objects WHERE wake_id = 'wake-a'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert status == "active"

    status = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert status["status"] == "lease_held"
    assert status["lease"]["owner_wake_id"] == "wake-a"
    assert "resume" in status["recovery_action"]

    finalize_turn = {
        "content": "",
        "tool_calls": [{"id": "final", "name": "finalize_graph", "arguments": {}}],
    }
    resumed = await runner.run(
        _runner_args(runner, tmp_path, thread, "wake-a", [finalize_turn], resume=True),
        print_report=False,
    )
    assert resumed["terminal_status"] == "published"
    assert _lease_owner(world) is None
    connection = sqlite3.connect(world)
    try:
        objects = int(connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0])
        receipts = int(
            connection.execute(
                "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = 'wake-a'"
            ).fetchone()[0]
        )
        active = int(
            connection.execute(
                "SELECT COUNT(*) FROM staged_objects WHERE wake_id = 'wake-a' AND status = 'active'"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    assert objects == 1
    assert receipts == 1
    assert active == 0


async def test_restore_repeat_is_nothing_to_restore(tmp_path) -> None:
    """Restoring twice is harmless: the second call reports the wake already
    owns the gate with active staging."""
    runner = _load_runner()
    thread = "restore-repeat"
    world, runtime = await _staged_then_abandoned(runner, tmp_path, thread, "wake-a")
    first = await runner.run(
        _restore_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert first["status"] == "restored"
    repeat = await runner.run(
        _restore_args(runner, world, runtime, thread, "wake-a"),
        print_report=False,
    )
    assert repeat["status"] == "nothing_to_restore"
    assert repeat["reason"] == "wake_still_active"
    assert _lease_owner(world) == "wake-a"


async def test_restore_untraceable_fails_closed(tmp_path) -> None:
    """An abandoned row without a successful patch receipt is untraceable
    working graph: restore refuses and mutates nothing."""
    runner = _load_runner()
    thread = "restore-untraceable"
    world, runtime = await _staged_then_abandoned(runner, tmp_path, thread, "wake-a")
    connection = sqlite3.connect(world)
    try:
        connection.execute("DELETE FROM staged_patch_receipts WHERE wake_id = 'wake-a'")
        connection.commit()
    finally:
        connection.close()
    before = _world_fingerprint(world)

    with pytest.raises(ValueError) as error:
        await runner.run(
            _restore_args(runner, world, runtime, thread, "wake-a"),
            print_report=False,
        )
    message = str(error.value)
    assert "no successful durable patch receipt" in message
    assert _lease_owner(world) is None
    assert _world_fingerprint(world) == before


async def test_restore_runtime_mismatch_fails_closed(tmp_path) -> None:
    """Restore against a different runtime than the wake claimed fails closed
    (resume would fail the same way)."""
    runner = _load_runner()
    thread = "restore-runtime"
    world, runtime = await _staged_then_abandoned(runner, tmp_path, thread, "wake-a")
    other_runtime = tmp_path / "other-runtime.sqlite3"
    before = _world_fingerprint(world)

    with pytest.raises(ValueError) as error:
        await runner.run(
            _restore_args(runner, world, other_runtime, thread, "wake-a"),
            print_report=False,
        )
    message = str(error.value)
    assert "graph_shell_restore_runtime_mismatch" in message
    assert _lease_owner(world) is None
    assert _world_fingerprint(world) == before


async def test_restore_claim_missing_fails_closed(tmp_path) -> None:
    """Abandoned staging without the world wake claim refuses restore."""
    runner = _load_runner()
    thread = "restore-claim"
    world, runtime = await _staged_then_abandoned(runner, tmp_path, thread, "wake-a")
    connection = sqlite3.connect(world)
    try:
        connection.execute("DELETE FROM graph_shell_wake_claims WHERE wake_id = 'wake-a'")
        connection.commit()
    finally:
        connection.close()
    before = _world_fingerprint(world)

    with pytest.raises(ValueError) as error:
        await runner.run(
            _restore_args(runner, world, runtime, thread, "wake-a"),
            print_report=False,
        )
    assert "graph_shell_restore_claim_missing" in str(error.value)
    assert _lease_owner(world) is None
    assert _world_fingerprint(world) == before


async def test_restore_checkpoint_missing_fails_closed(tmp_path) -> None:
    """Abandoned staging without a runtime checkpoint refuses restore: it
    would wedge the world with active staging no resume could continue."""
    runner = _load_runner()
    thread = "restore-checkpoint"
    world, runtime = await _staged_then_abandoned(runner, tmp_path, thread, "wake-a")
    connection = sqlite3.connect(runtime)
    try:
        connection.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread,))
        connection.commit()
    finally:
        connection.close()
    before = _world_fingerprint(world)

    with pytest.raises(ValueError) as error:
        await runner.run(
            _restore_args(runner, world, runtime, thread, "wake-a"),
            print_report=False,
        )
    assert "graph_shell_restore_checkpoint_missing" in str(error.value)
    assert _lease_owner(world) is None
    assert _world_fingerprint(world) == before


async def test_status_reports_abandoned_staging_with_restore_action(tmp_path) -> None:
    """After an abandon, --graph-shell-status names the abandoned staging and
    points the operator at --graph-shell-restore — read-only."""
    runner = _load_runner()
    thread = "restore-status"
    world, runtime = await _staged_then_abandoned(runner, tmp_path, thread, "wake-a")
    before = _world_fingerprint(world)
    status = await runner.run(
        _status_args(runner, world, runtime, thread), print_report=False
    )
    assert status["status"] == "no_active_writer"
    assert status["abandoned_staging"] == {
        "wake-a": {"staged_objects": 1, "staged_assertions": 0, "staged_inquiries": 0}
    }
    assert "--graph-shell-restore wake-a" in status["recovery_action"]
    assert _world_fingerprint(world) == before


def _finalize_wake_args(
    runner, world_path: Path, runtime_path: Path, wake_id: str
) -> object:
    """Parse a --graph-shell-finalize-wake management entry (no fixtures needed)."""
    return runner.parse_args(
        [
            "--world-db",
            str(world_path),
            "--runtime-db",
            str(runtime_path),
            "--thread-id",
            "finalize-wake-thread",
            "--graph-shell",
            "--graph-shell-finalize-wake",
            wake_id,
        ]
    )


def test_finalize_wake_parser_rules(tmp_path) -> None:
    """--graph-shell-finalize-wake is a management entry: standalone, never
    combines with --resume / --graph-shell-status / --graph-shell-abandon /
    --graph-shell-restore."""
    runner = _load_runner()
    base = [
        "--world-db",
        str(tmp_path / "finalize-world.sqlite3"),
        "--runtime-db",
        str(tmp_path / "finalize-runtime.sqlite3"),
        "--thread-id",
        "finalize-thread",
    ]
    args = runner.parse_args([*base, "--graph-shell-finalize-wake", "wake-a"])
    assert args.graph_shell_finalize_wake == "wake-a"
    with pytest.raises(SystemExit):
        runner.parse_args(
            [*base, "--graph-shell-finalize-wake", "wake-a", "--resume"]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [*base, "--graph-shell-status", "--graph-shell-finalize-wake", "wake-a"]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [*base, "--graph-shell-abandon", "wake-a", "--graph-shell-finalize-wake", "wake-a"]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [*base, "--graph-shell-restore", "wake-a", "--graph-shell-finalize-wake", "wake-a"]
        )
    args = runner.parse_args([*base, "--graph-shell", "--graph-shell-finalize-wake", "wake-a"])
    assert args.graph_shell_finalize_wake == "wake-a"


async def test_finalize_wake_no_world_is_side_effect_free(tmp_path) -> None:
    """A missing world is reported, never created."""
    runner = _load_runner()
    world = tmp_path / "missing-finalize-world.sqlite3"
    runtime = tmp_path / "missing-finalize-runtime.sqlite3"
    result = await runner.run(
        _finalize_wake_args(runner, world, runtime, "wake-a"),
        print_report=False,
    )
    assert result["status"] == "no_world"
    assert not world.exists()
    assert not runtime.exists()


async def test_finalize_wake_wake_unknown_fails_closed(tmp_path) -> None:
    """A world with no wake claim answers wake_unknown and mutates nothing."""
    runner = _load_runner()
    world, runtime = await init_blank_database_pair(
        tmp_path / "finalize-unknown-world.sqlite3",
        tmp_path / "finalize-unknown-runtime.sqlite3",
    )
    before = _world_fingerprint(world)
    result = await runner.run(
        _finalize_wake_args(runner, world, runtime, "wake-a"),
        print_report=False,
    )
    assert result["status"] == "wake_unknown"
    assert result["wake_id"] == "wake-a"
    assert _world_fingerprint(world) == before


async def test_finalize_wake_nothing_to_finalize_without_active_staging(
    tmp_path,
) -> None:
    """A wake claim with no active staging answers nothing_to_finalize and
    mutates nothing."""
    runner = _load_runner()
    world, runtime = await init_blank_database_pair(
        tmp_path / "finalize-empty-world.sqlite3",
        tmp_path / "finalize-empty-runtime.sqlite3",
    )
    with sqlite3.connect(world) as connection:
        connection.execute(
            "INSERT INTO graph_shell_wake_claims (wake_id, thread_id, "
            "runtime_store_identity, claimed_at) VALUES (?, ?, ?, ?)",
            ("wake-a", "finalize-thread", str(runtime), 1.0),
        )
        connection.commit()
    before = _world_fingerprint(world)
    result = await runner.run(
        _finalize_wake_args(runner, world, runtime, "wake-a"),
        print_report=False,
    )
    assert result["status"] == "nothing_to_finalize"
    assert _world_fingerprint(world) == before


async def test_finalize_wake_publishes_active_staging_and_replays(tmp_path) -> None:
    """A halted wake's active staged graph is published deterministically; a
    repeat call replays the same receipt (already_published)."""
    runner = _load_runner()
    thread = "finalize-publish"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    stop = {"content": "stop", "tool_calls": []}
    first = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [_patch_turn("op-pub", "Finalize Probe"), stop, stop],
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"
    with sqlite3.connect(world) as connection:
        staged = connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = 'wake-a' "
            "AND status = 'active'"
        ).fetchone()[0]
    assert staged == 1

    result = await runner.run(
        _finalize_wake_args(runner, world, runtime, "wake-a"),
        print_report=False,
    )
    assert result["status"] == "published"
    assert result["commit_id"] == "wake-a:finalize"
    assert result["stats"]["objects_created"] == 1
    assert result["stats"]["total_items"] == 1
    assert "wake-a:s1" in result["item_ids"]
    # the CLI report path serializes FinalizeReceipt datetimes to ISO strings
    report = json.loads(json.dumps(result, default=str))
    assert report["status"] == "published"
    assert report["committed_at"].endswith("+00:00")
    with sqlite3.connect(world) as connection:
        formal = connection.execute(
            "SELECT id, canonical_name FROM objects"
        ).fetchall()
    assert formal == [("wake-a:s1", "Finalize Probe")]
    # the management entry needs no lease: the halted wake may still hold it
    assert _lease_owner(world) == "wake-a"

    # idempotent replay: same receipt, nothing written twice
    replayed = await runner.run(
        _finalize_wake_args(runner, world, runtime, "wake-a"),
        print_report=False,
    )
    assert replayed["status"] == "already_published"
    assert replayed["commit_id"] == "wake-a:finalize"
    with sqlite3.connect(world) as connection:
        assert connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 1


async def test_finalize_wake_wake_closed_after_second_staging(tmp_path) -> None:
    """Items staged after the wake published can never publish under it: the
    management entry reports wake_closed with the stranded ids (I1)."""
    runner = _load_runner()
    thread = "finalize-closed"
    world = tmp_path / f"{thread}-world.sqlite3"
    runtime = tmp_path / f"{thread}-runtime.sqlite3"
    await init_blank_database_pair(world, runtime)
    stop = {"content": "stop", "tool_calls": []}
    first = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            thread,
            "wake-a",
            [_patch_turn("op-1", "First Probe"), stop, stop],
        ),
        print_report=False,
    )
    assert first["terminal_status"] == "staged_unpublished"
    published = await runner.run(
        _finalize_wake_args(runner, world, runtime, "wake-a"),
        print_report=False,
    )
    assert published["status"] == "published"
    # stage one more item after publishing (models an out-of-band patch the
    # agent could not have made under the released lease)
    with sqlite3.connect(world) as connection:
        connection.execute(
            "INSERT INTO staged_objects (staged_id, wake_id, status, kind, "
            "canonical_name, type_key, domain_hints_json, aliases_json, "
            "created_at, updated_at, version, provisional, identity_basis_json) "
            "VALUES (?, ?, 'active', 'concept', ?, 'concept', '[]', '[]', "
            "?, ?, 1, 1, '[]')",
            (
                "wake-a:s2",
                "wake-a",
                "Late Probe",
                "2026-08-23T00:00:00+00:00",
                "2026-08-23T00:00:00+00:00",
            ),
        )
        connection.commit()
    closed = await runner.run(
        _finalize_wake_args(runner, world, runtime, "wake-a"),
        print_report=False,
    )
    assert closed["status"] == "wake_closed"
    assert "wake-a:s2" in str(closed["blockers"])
    with sqlite3.connect(world) as connection:
        # the late item stayed staged (nothing was written, nothing dropped)
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE staged_id = 'wake-a:s2' "
            "AND status = 'active'"
        ).fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 1
