"""Unified error classification for all Agent and runtime operations.

Maps to the 11 error categories defined in docs/02-agent-reference-model.md §7.
Each error carries a stable code and a retry policy — only transient errors are retryable.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable error codes shared across all Agents and the runtime.

    Do NOT invent new codes without updating docs/02-agent-reference-model.md.
    """

    INPUT_INVALID = "INPUT_INVALID"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    TOOL_TRANSIENT = "TOOL_TRANSIENT"
    TOOL_PERMANENT = "TOOL_PERMANENT"
    MODEL_TRANSIENT = "MODEL_TRANSIENT"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    CONFLICT_UNRESOLVED = "CONFLICT_UNRESOLVED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Only these codes are permitted to auto-retry.
RETRYABLE_CODES: frozenset[ErrorCode] = frozenset(
    {ErrorCode.TOOL_TRANSIENT, ErrorCode.MODEL_TRANSIENT, ErrorCode.MODEL_OUTPUT_INVALID}
)

# MODEL_OUTPUT_INVALID should retry at most N times before escalating.
MAX_OUTPUT_REPAIR_RETRIES: int = 3


class AgentError(Exception):
    """Base exception for all Agent-produced errors.

    Always carries an ErrorCode so the runtime can decide retry/abort/escalate
    without inspecting the message string.
    """

    def __init__(self, code: ErrorCode, message: str, *, recoverable: bool | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable if recoverable is not None else code in RETRYABLE_CODES


def is_retryable(error: AgentError) -> bool:
    """Return True if this error's code permits automatic retry."""
    return error.code in RETRYABLE_CODES
