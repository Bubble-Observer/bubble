from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from leave_information_bubble.console.runs import (
    RunAlreadyActiveError,
    RunManager,
    RunRecord,
    RunSpec,
    read_runtime_metrics,
)


async def test_manager_allows_only_one_active_run(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(_: RunSpec) -> dict[str, str]:
        started.set()
        await release.wait()
        return {"terminal_summary": "done"}

    manager = RunManager(runner)
    first = await manager.start(RunSpec("first", tmp_path / "runtime.sqlite3"))
    await started.wait()
    with pytest.raises(RunAlreadyActiveError):
        await manager.start(RunSpec("second", tmp_path / "runtime.sqlite3"))
    release.set()
    await _wait_for_terminal(manager, first.run_id)


async def test_manager_records_success_and_sanitizes_failure(tmp_path: Path) -> None:
    async def succeeds(_: RunSpec) -> dict[str, object]:
        return {"terminal_summary": "ready", "nested": {"count": 2}}

    success_manager = RunManager(succeeds)
    success = await success_manager.start(RunSpec("ok", tmp_path / "runtime.sqlite3", {"mode": "broad"}))
    completed = await _wait_for_terminal(success_manager, success.run_id)
    assert completed.status == "succeeded"
    assert completed.config_snapshot == {"mode": "broad"}
    assert completed.result_summary == {"terminal_summary": "ready", "nested": {"count": 2}}

    async def fails(_: RunSpec) -> None:
        raise RuntimeError("api_key=should-not-leak")

    failure_manager = RunManager(fails)
    failed = await failure_manager.start(RunSpec("bad", tmp_path / "runtime.sqlite3"))
    failed_record = await _wait_for_terminal(failure_manager, failed.run_id)
    assert failed_record.status == "failed"
    assert failed_record.error == "Runner failed (RuntimeError)."
    assert "should-not-leak" not in " ".join(event.message for event in failure_manager.list_events())


def test_read_runtime_metrics_aggregates_one_thread(tmp_path: Path) -> None:
    runtime_db = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(runtime_db)
    connection.execute(
        "CREATE TABLE model_calls (id INTEGER PRIMARY KEY, thread_id TEXT, turn INTEGER, "
        "phase TEXT, model TEXT, prompt_tokens INTEGER, cached_input_tokens INTEGER, "
        "completion_tokens INTEGER, cost_usd REAL, latency_ms REAL)"
    )
    connection.executemany(
        "INSERT INTO model_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "one", 1, "exploration", "deepseek-a", 10, 4, 3, 0.1, 42.0),
            (2, "one", 2, "consolidation", "deepseek-a", 20, 8, 7, 0.2, 75.0),
            (3, "two", 1, "exploration", "other", 100, 0, 20, 1.0, 90.0),
        ],
    )
    connection.commit()
    connection.close()

    metrics = read_runtime_metrics(runtime_db, "one")
    assert metrics.calls == 2
    assert metrics.prompt_tokens == 30
    assert metrics.completion_tokens == 10
    assert metrics.cached_input_tokens == 12
    assert metrics.cost_usd == pytest.approx(0.3)
    assert (metrics.latest_turn, metrics.latest_phase, metrics.latest_model) == (
        2,
        "consolidation",
        "deepseek-a",
    )
    assert metrics.latest_latency_ms == 75.0
    assert metrics.latest_prompt_tokens == 20
    assert metrics.latest_cached_input_tokens == 8
    assert metrics.latest_completion_tokens == 7


def test_events_are_bounded_and_redacted() -> None:
    async def runner(_: RunSpec) -> None:
        return None

    manager = RunManager(runner, event_limit=2)
    manager.publish("tool.progress", "Authorization: Bearer abc123")
    manager.publish("tool.progress", "second")
    manager.publish("tool.progress", "third")
    events = manager.list_events()
    assert [event.message for event in events] == ["third", "second"]


async def test_events_are_isolated_by_run(tmp_path: Path) -> None:
    async def runner(spec: RunSpec, publish) -> None:
        publish("tool.progress", f"event for {spec.thread_id}")

    manager = RunManager(runner)
    first = await manager.start(RunSpec("first", tmp_path / "runtime.sqlite3"))
    await _wait_for_terminal(manager, first.run_id)
    second = await manager.start(RunSpec("second", tmp_path / "runtime.sqlite3"))
    await _wait_for_terminal(manager, second.run_id)

    first_messages = [event.message for event in manager.list_events(run_id=first.run_id)]
    second_messages = [event.message for event in manager.list_events(run_id=second.run_id)]
    assert any("event for first" in message for message in first_messages)
    assert all("second" not in message for message in first_messages)
    assert any("event for second" in message for message in second_messages)
    assert all("first" not in message for message in second_messages)


async def test_manager_restores_terminal_history_without_running_it(tmp_path: Path) -> None:
    called = False

    async def runner(_: RunSpec) -> None:
        nonlocal called
        called = True

    manager = RunManager(runner)
    now = datetime.now(UTC)
    restored = await manager.restore(
        RunRecord(
            run_id="restored-one",
            thread_id="console-agent-old",
            config_snapshot={"profile_id": "agent"},
            runtime_db=tmp_path / "runtime.sqlite3",
            status="succeeded",
            queued_at=now,
            started_at=now,
            ended_at=now,
            result_summary={"terminal_summary": "done"},
        )
    )

    assert restored.run_id == "restored-one"
    assert (await manager.list_runs())[0].status == "succeeded"
    assert called is False


async def _wait_for_terminal(manager: RunManager, run_id: str):
    for _ in range(100):
        record = await manager.get(run_id)
        assert record is not None
        if record.status in {"succeeded", "failed"}:
            return record
        await asyncio.sleep(0)
    raise AssertionError("run did not finish")
