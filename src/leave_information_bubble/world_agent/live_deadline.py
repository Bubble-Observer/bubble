"""Opt-in abnormal safety horizon for controlled live world-agent runs.

The model guard is not a cognitive time limit, normal wake-exit signal, or a
product default. It rejects a new provider dispatch after the horizon and lets
an in-flight model request settle. Acquisition tools receive the same wake
cutoff and may bound their own in-flight work to its remaining time.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from leave_information_bubble.gateway.client import ToolModelResponse
from leave_information_bubble.runtime.errors import AgentError, ErrorCode


class ToolModelInvoker(Protocol):
    """Provider boundary guarded before a new dispatch."""

    async def invoke_tools(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]], **kwargs: object
    ) -> ToolModelResponse:
        """Invoke one provider-native tool request."""


class LiveDeadlineExceeded(AgentError):
    """Non-retryable abnormal-stop signal preventing a new provider dispatch."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.POLICY_BLOCKED,
            "live safety deadline reached before model dispatch",
            recoverable=False,
        )


@dataclass
class LiveDeadlineGuard:
    """Reject model calls after a safety horizon without cancelling one in flight.

    Normal wake completion remains the graph and agent's responsibility.  An
    expired guard deliberately reports an incomplete run rather than trying to
    turn the elapsed safety horizon into a finalization request.
    """

    model: ToolModelInvoker
    deadline_seconds: float
    clock: Callable[[], float] = time.monotonic
    _deadline: float = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("live deadline must be finite and greater than zero")
        self._deadline = self.clock() + self.deadline_seconds

    @property
    def deadline_at(self) -> float:
        """Return the shared monotonic cutoff for tool coordination."""
        return self._deadline

    async def invoke_tools(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]], **kwargs: object
    ) -> ToolModelResponse:
        """Reject a new delegation when the safety horizon has elapsed."""
        if self.clock() >= self._deadline:
            raise LiveDeadlineExceeded()
        return await self.model.invoke_tools(messages, tools=tools, **kwargs)
