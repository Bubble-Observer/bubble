"""Offline tests for the opt-in P5-G1 live pre-call cost guard."""

from __future__ import annotations

import importlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leave_information_bubble.gateway.client import StructuredModelOutputError, ToolModelResponse
from leave_information_bubble.runtime.errors import AgentError, ErrorCode
from leave_information_bubble.world_agent.live_cost_guard import LiveCostGuard, _upper_bound_usd


class _FakeModel:
    """Deterministic provider boundary that records calls without network access."""

    def __init__(self, outcomes: list[ToolModelResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def invoke_tools(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]], **kwargs: Any
    ) -> ToolModelResponse:
        self.calls.append({"messages": messages, "tools": tools, "options": kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _request() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a request whose content must never appear in the audit file."""
    return (
        [{"role": "user", "content": "secret mission body"}],
        [{"type": "function", "function": {"name": "lookup", "parameters": {"secret": "value"}}}],
    )


def _response(cost_usd: float = 0.001) -> ToolModelResponse:
    """Build a billable normalized tool response."""
    return ToolModelResponse(
        content="ok",
        message={"role": "assistant", "content": "ok"},
        model="deepseek-v4-flash",
        prompt_tokens=1,
        cost_usd=cost_usd,
    )


def _audit_rows(path: Path) -> list[dict[str, Any]]:
    """Read append-only audit rows for assertions."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_guard_reserves_actual_max_tokens_settles_and_omits_request_body(tmp_path: Path) -> None:
    """A successful call is reserved by bytes plus reserve and settled by actual cost."""
    audit_path = tmp_path / "audit.jsonl"
    fake = _FakeModel([_response()])
    guard = LiveCostGuard(fake, "deepseek-v4-flash", Decimal("0.05"), audit_path)
    messages, tools = _request()

    response = await guard.invoke_tools(messages, tools=tools, max_tokens=8192, thinking=False)

    assert response.cost_usd == 0.001
    assert fake.calls[0]["options"]["max_tokens"] == 8192
    rows = _audit_rows(audit_path)
    assert [row["event"] for row in rows] == ["reserved", "settled"]
    assert rows[0]["estimated_input_tokens"] == rows[0]["request_bytes"] + 2048
    assert rows[0]["max_output_tokens"] == 8192
    assert rows[1]["settled_usd"] == "0.001"
    assert rows[0]["reservation_id"] == rows[1]["reservation_id"] == "g1-1"
    assert rows[0]["cumulative_outstanding_usd"] == rows[0]["reservation_usd"]
    assert rows[1]["cumulative_outstanding_usd"] == "0"
    assert "secret mission body" not in audit_path.read_text(encoding="utf-8")
    assert '"secret":"value"' not in audit_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_guard_rejects_exact_cap_before_model_call(tmp_path: Path) -> None:
    """The strict inequality rejects a request whose reservation exactly reaches the cap."""
    messages, tools = _request()
    request_bytes = len(
        json.dumps(
            {"messages": messages, "options": {"max_tokens": 4096}, "tools": tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    cap = _upper_bound_usd(request_bytes + 2048, 4096)
    fake = _FakeModel([])
    audit_path = tmp_path / "audit.jsonl"
    guard = LiveCostGuard(fake, "deepseek-v4-flash", cap, audit_path)

    with pytest.raises(AgentError) as raised:
        await guard.invoke_tools(messages, tools=tools)

    assert raised.value.code is ErrorCode.BUDGET_EXCEEDED
    assert fake.calls == []
    assert _audit_rows(audit_path)[0]["status"] == "budget_exceeded"


@pytest.mark.asyncio
async def test_guard_keeps_unknown_usage_reserved_for_later_pre_call_stop(tmp_path: Path) -> None:
    """An exception without a billable response cannot make a later call appear free."""
    messages, tools = _request()
    request_bytes = len(
        json.dumps(
            {"messages": messages, "options": {"max_tokens": 4096}, "tools": tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    reservation = _upper_bound_usd(request_bytes + 2048, 4096)
    fake = _FakeModel([RuntimeError("connection dropped")])
    audit_path = tmp_path / "audit.jsonl"
    guard = LiveCostGuard(fake, "deepseek-v4-flash", reservation * 2, audit_path)

    with pytest.raises(RuntimeError, match="connection dropped"):
        await guard.invoke_tools(messages, tools=tools)
    with pytest.raises(AgentError) as raised:
        await guard.invoke_tools(messages, tools=tools)

    assert raised.value.code is ErrorCode.BUDGET_EXCEEDED
    assert len(fake.calls) == 1
    assert [row["event"] for row in _audit_rows(audit_path)] == ["reserved", "unknown", "rejected"]


@pytest.mark.asyncio
async def test_structured_error_with_response_settles_then_reraises(tmp_path: Path) -> None:
    """Billable invalid output replaces its reservation with the reported actual cost."""
    billable = _response(0.0015)
    fake = _FakeModel([StructuredModelOutputError("invalid output", billable)])
    audit_path = tmp_path / "audit.jsonl"
    guard = LiveCostGuard(fake, "deepseek-v4-flash", Decimal("0.01"), audit_path)
    messages, tools = _request()

    with pytest.raises(StructuredModelOutputError, match="invalid output"):
        await guard.invoke_tools(messages, tools=tools, max_tokens=4096)

    rows = _audit_rows(audit_path)
    assert [row["event"] for row in rows] == ["reserved", "settled"]
    assert rows[-1]["settled_usd"] == "0.0015"


@pytest.mark.asyncio
async def test_success_without_usage_keeps_reservation_and_stops_later_call(tmp_path: Path) -> None:
    """A nominally successful response with zero usage cannot be settled as free."""
    messages, tools = _request()
    request_bytes = len(
        json.dumps(
            {"messages": messages, "options": {"max_tokens": 4096}, "tools": tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    reservation = _upper_bound_usd(request_bytes + 2048, 4096)
    fake = _FakeModel([ToolModelResponse(content="ok", message={}, cost_usd=0.0)])
    audit_path = tmp_path / "audit.jsonl"
    guard = LiveCostGuard(fake, "deepseek-v4-flash", reservation * 2, audit_path)

    with pytest.raises(AgentError) as first:
        await guard.invoke_tools(messages, tools=tools)
    with pytest.raises(AgentError) as second:
        await guard.invoke_tools(messages, tools=tools)

    assert first.value.code is ErrorCode.BUDGET_EXCEEDED
    assert second.value.code is ErrorCode.BUDGET_EXCEEDED
    assert len(fake.calls) == 1
    assert [row["event"] for row in _audit_rows(audit_path)] == ["reserved", "unknown", "rejected"]


@pytest.mark.asyncio
async def test_structured_error_without_usage_keeps_reservation_and_reraises(tmp_path: Path) -> None:
    """Invalid output with missing usage preserves the reservation and original error."""
    billable_unknown = ToolModelResponse(content="", message={}, cost_usd=0.0)
    fake = _FakeModel([StructuredModelOutputError("invalid output", billable_unknown)])
    audit_path = tmp_path / "audit.jsonl"
    guard = LiveCostGuard(fake, "deepseek-v4-flash", Decimal("0.01"), audit_path)
    messages, tools = _request()

    with pytest.raises(StructuredModelOutputError, match="invalid output"):
        await guard.invoke_tools(messages, tools=tools)

    assert [row["event"] for row in _audit_rows(audit_path)] == ["reserved", "unknown"]


@pytest.mark.parametrize("cap", [Decimal("Infinity"), Decimal("NaN"), Decimal("0")])
def test_guard_rejects_non_finite_or_non_positive_cap(tmp_path: Path, cap: Decimal) -> None:
    """The constructor rejects cap values that cannot support strict comparison."""
    with pytest.raises(ValueError, match="finite value greater than zero"):
        LiveCostGuard(_FakeModel([]), "deepseek-v4-flash", cap, tmp_path / "audit.jsonl")


def test_guard_requires_an_existing_audit_parent_directory(tmp_path: Path) -> None:
    """Audit writability is checked before the first guarded request is reserved."""
    with pytest.raises(ValueError, match="parent directory"):
        LiveCostGuard(
            _FakeModel([]),
            "deepseek-v4-flash",
            Decimal("0.01"),
            tmp_path / "missing" / "audit.jsonl",
        )


def test_guard_rejects_a_nonempty_audit_file_before_any_model_call(tmp_path: Path) -> None:
    """A fresh guard cannot silently discard a prior process's reservations."""
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text('{"event":"reserved"}\n', encoding="utf-8")
    fake = _FakeModel([])

    with pytest.raises(ValueError, match="absent or empty"):
        LiveCostGuard(fake, "deepseek-v4-flash", Decimal("0.01"), audit_path)

    assert fake.calls == []


def _runner_module() -> Any:
    """Load the installed composition root for offline CLI parser tests."""
    return importlib.import_module("leave_information_bubble.world_agent.cli")


def test_cli_leaves_default_behavior_disabled_and_rejects_invalid_live_guard_combinations(
    tmp_path: Path,
) -> None:
    """The optional live guard cannot silently apply to replay or incomplete configuration."""
    runner = _runner_module()
    base = [
        "--perspective",
        "m",
        "--thread-id",
        "thread-1",
        "--world-db",
        str(tmp_path / "world.sqlite3"),
        "--runtime-db",
        str(tmp_path / "runtime.sqlite3"),
    ]

    assert runner.parse_args(base).live_hard_cap_usd is None
    assert runner.parse_args(base).live_deadline_seconds is None
    with pytest.raises(SystemExit):
        runner.parse_args([*base, "--live-hard-cap-usd", "0.01"])
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                *base,
                "--live-hard-cap-usd",
                "0.01",
                "--live-cost-audit-path",
                str(tmp_path / "audit.jsonl"),
                "--replay-fixture",
                "fixture.json",
                "--scripted-model-fixture",
                "model.json",
            ]
        )
    for invalid_cap in ("Infinity", "NaN"):
        with pytest.raises(SystemExit):
            runner.parse_args(
                [
                    *base,
                    "--live-hard-cap-usd",
                    invalid_cap,
                    "--live-cost-audit-path",
                    str(tmp_path / "audit.jsonl"),
                ]
            )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                *base,
                "--resume",
                "--live-hard-cap-usd",
                "0.01",
                "--live-cost-audit-path",
                str(tmp_path / "audit.jsonl"),
            ]
        )
    for invalid_deadline in ("0", "-1", "Infinity", "NaN"):
        with pytest.raises(SystemExit):
            runner.parse_args(
                [
                    *base,
                    "--live-deadline-seconds",
                    invalid_deadline,
                ]
            )
    # the deadline is a pure wall-clock safety net: valid on fresh runs with no
    # cost cap, and valid on --resume runs (which cannot pair a hard cap)
    for extra in (
        [],
        [
            "--live-hard-cap-usd",
            "0.01",
            "--live-cost-audit-path",
            str(tmp_path / "audit.jsonl"),
        ],
        ["--resume"],
    ):
        assert (
            runner.parse_args(
                [
                    *base,
                    *extra,
                    "--live-deadline-seconds",
                    "30",
                ]
            ).live_deadline_seconds
            == 30
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                *base,
                "--replay-fixture",
                "fixture.json",
                "--scripted-model-fixture",
                "model.json",
                "--live-deadline-seconds",
                "30",
            ]
        )


def test_cli_describes_live_deadline_as_opt_in_abnormal_safety_horizon(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner_module()

    with pytest.raises(SystemExit) as raised:
        runner.parse_args(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    normalized = " ".join(help_text.split())
    assert "Opt-in wall-clock safety net for a live wake" in normalized
    assert "independent of cost caps" in normalized
    assert "fresh or --resume" in normalized
    assert "dispatched model call settle" in normalized
    assert "blocks new acquisition dispatch" in normalized
    assert "lets an already-dispatched acquisition settle under its own timeout" in normalized
    assert "bounds acquisition to remaining wake time" not in normalized
    assert "not a cognitive time limit, normal exit signal, or default" in normalized


@pytest.mark.asyncio
async def test_wrong_configured_model_fails_before_runner_creates_any_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A frozen-rate guard cannot accidentally wrap a differently priced live model."""
    runner = _runner_module()
    args = runner.parse_args(
        [
            "--perspective",
            "m",
            "--thread-id",
            "thread-1",
            "--world-db",
            str(tmp_path / "world.sqlite3"),
            "--runtime-db",
            str(tmp_path / "runtime.sqlite3"),
            "--live-hard-cap-usd",
            "0.01",
            "--live-cost-audit-path",
            str(tmp_path / "audit.jsonl"),
        ]
    )
    monkeypatch.setattr(runner, "get_settings", lambda: SimpleNamespace(deepseek_model="other-model"))
    monkeypatch.setattr(runner, "WorldStore", lambda _path: pytest.fail("store must not be created"))

    with pytest.raises(ValueError, match="requires configured model deepseek-v4-flash"):
        await runner.run(args)


@pytest.mark.asyncio
async def test_live_cap_resume_fails_before_runner_creates_any_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Manual Namespace construction cannot bypass the CLI's live-resume prohibition."""
    runner = _runner_module()
    args = runner.parse_args(
        [
            "--perspective",
            "m",
            "--thread-id",
            "thread-1",
            "--world-db",
            str(tmp_path / "world.sqlite3"),
            "--runtime-db",
            str(tmp_path / "runtime.sqlite3"),
            "--live-hard-cap-usd",
            "0.01",
            "--live-cost-audit-path",
            str(tmp_path / "audit.jsonl"),
        ]
    )
    args.resume = True
    monkeypatch.setattr(runner, "WorldStore", lambda _path: pytest.fail("store must not be created"))

    with pytest.raises(ValueError, match="unavailable with --resume"):
        await runner.run(args)


@pytest.mark.asyncio
async def test_nonempty_audit_fails_before_runner_creates_any_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A previous audit cannot be reused to bypass process-local reservations."""
    runner = _runner_module()
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text('{"event":"reserved"}\n', encoding="utf-8")
    args = runner.parse_args(
        [
            "--perspective",
            "m",
            "--thread-id",
            "thread-1",
            "--world-db",
            str(tmp_path / "world.sqlite3"),
            "--runtime-db",
            str(tmp_path / "runtime.sqlite3"),
            "--live-hard-cap-usd",
            "0.01",
            "--live-cost-audit-path",
            str(audit_path),
        ]
    )
    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: SimpleNamespace(deepseek_model="deepseek-v4-flash"),
    )
    monkeypatch.setattr(runner, "WorldStore", lambda _path: pytest.fail("store must not be created"))

    with pytest.raises(ValueError, match="absent or empty"):
        await runner.run(args)
