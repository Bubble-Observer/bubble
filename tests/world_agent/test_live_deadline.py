"""Offline coverage for the opt-in abnormal live safety horizon."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from leave_information_bubble.gateway.client import ToolModelResponse
from leave_information_bubble.runtime.errors import AgentError, ErrorCode
from leave_information_bubble.world_agent.live_cost_guard import LiveCostGuard
from leave_information_bubble.world_agent.live_deadline import LiveDeadlineGuard


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke_tools(self, _messages: list[dict[str, Any]], **_kwargs: object) -> ToolModelResponse:
        self.calls += 1
        return ToolModelResponse(
            content="ok",
            message={"role": "assistant", "content": "ok"},
            model="deepseek-v4-flash",
            prompt_tokens=1,
            cost_usd=0.001,
        )


def _request() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return [{"role": "user", "content": "m"}], []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_tokens": 8192, "tool_choice": {"type": "function", "function": {"name": "submit_cognition"}}},
        {"max_tokens": 4096, "tool_choice": {"type": "function", "function": {"name": "digest_observation"}}},
        {"max_tokens": 8192, "thinking": False},
    ],
    ids=["exploration", "proposal", "digest", "repair"],
)
async def test_expired_deadline_prevents_cost_reservation_and_provider_call(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    clock = _Clock()
    provider = _Model()
    audit = tmp_path / "cost.jsonl"
    guarded = LiveDeadlineGuard(
        LiveCostGuard(provider, "deepseek-v4-flash", Decimal("0.01"), audit), 1, clock
    )
    clock.value = 1
    messages, tools = _request()

    with pytest.raises(AgentError) as raised:
        await guarded.invoke_tools(messages, tools=tools, **options)

    assert raised.value.code is ErrorCode.POLICY_BLOCKED
    assert provider.calls == 0
    assert audit.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_inflight_call_settles_then_next_call_is_blocked(tmp_path: Path) -> None:
    """The safety horizon never cancels an already-dispatched billable call."""
    clock = _Clock()
    provider = _Model()
    audit = tmp_path / "cost.jsonl"
    guarded = LiveDeadlineGuard(
        LiveCostGuard(provider, "deepseek-v4-flash", Decimal("0.01"), audit), 1, clock
    )
    messages, tools = _request()

    await guarded.invoke_tools(messages, tools=tools)
    clock.value = 1
    with pytest.raises(AgentError, match="deadline"):
        await guarded.invoke_tools(messages, tools=tools)

    assert provider.calls == 1
    assert [line.split('"event":"')[1].split('"')[0] for line in audit.read_text().splitlines()] == [
        "reserved",
        "settled",
    ]


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_deadline_requires_finite_positive_seconds(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        LiveDeadlineGuard(_Model(), seconds, _Clock())


def test_deadline_exposes_same_monotonic_cutoff_for_world_tools() -> None:
    clock = _Clock(7.0)
    guarded = LiveDeadlineGuard(_Model(), 5.0, clock)

    assert guarded.deadline_at == 12.0
