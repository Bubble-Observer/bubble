"""Tests for error classification — src/runtime/errors.py."""

from __future__ import annotations

from leave_information_bubble.runtime.errors import (
    RETRYABLE_CODES,
    AgentError,
    ErrorCode,
    is_retryable,
)

# ---------------------------------------------------------------------------
# Error code membership
# ---------------------------------------------------------------------------


def test_retryable_codes_are_correct() -> None:
    """Only transient and output-invalid errors should be retryable."""
    assert (
        frozenset(
            {
                ErrorCode.TOOL_TRANSIENT,
                ErrorCode.MODEL_TRANSIENT,
                ErrorCode.MODEL_OUTPUT_INVALID,
            }
        )
        == RETRYABLE_CODES
    )


def test_permanent_codes_not_in_retryable() -> None:
    """Permanent errors must not be retryable."""
    permanent = {
        ErrorCode.INPUT_INVALID,
        ErrorCode.SOURCE_UNAVAILABLE,
        ErrorCode.TOOL_PERMANENT,
        ErrorCode.EVIDENCE_INSUFFICIENT,
        ErrorCode.POLICY_BLOCKED,
        ErrorCode.BUDGET_EXCEEDED,
        ErrorCode.CONFLICT_UNRESOLVED,
        ErrorCode.INTERNAL_ERROR,
    }
    assert permanent.isdisjoint(RETRYABLE_CODES)


def test_all_error_codes_covered() -> None:
    """Every ErrorCode must be either retryable or permanent — no orphans."""
    all_codes = set(ErrorCode)
    covered = RETRYABLE_CODES | {
        ErrorCode.INPUT_INVALID,
        ErrorCode.SOURCE_UNAVAILABLE,
        ErrorCode.TOOL_PERMANENT,
        ErrorCode.EVIDENCE_INSUFFICIENT,
        ErrorCode.POLICY_BLOCKED,
        ErrorCode.BUDGET_EXCEEDED,
        ErrorCode.CONFLICT_UNRESOLVED,
        ErrorCode.INTERNAL_ERROR,
    }
    assert all_codes == covered


# ---------------------------------------------------------------------------
# AgentError
# ---------------------------------------------------------------------------


def test_agent_error_has_code() -> None:
    """AgentError must carry an ErrorCode."""
    err = AgentError(ErrorCode.MODEL_TRANSIENT, "timeout")
    assert err.code == ErrorCode.MODEL_TRANSIENT


def test_agent_error_message() -> None:
    """AgentError message must be accessible."""
    err = AgentError(ErrorCode.BUDGET_EXCEEDED, "Token budget exhausted")
    assert "Token budget exhausted" in str(err)


def test_agent_error_recoverable_auto_detect_true() -> None:
    """MODEL_TRANSIENT should auto-detect as recoverable=True."""
    err = AgentError(ErrorCode.MODEL_TRANSIENT, "timeout")
    assert err.recoverable is True


def test_agent_error_recoverable_auto_detect_false() -> None:
    """INPUT_INVALID should auto-detect as recoverable=False."""
    err = AgentError(ErrorCode.INPUT_INVALID, "bad input")
    assert err.recoverable is False


def test_agent_error_recoverable_explicit_override() -> None:
    """Explicit recoverable flag overrides auto-detection."""
    err = AgentError(ErrorCode.MODEL_TRANSIENT, "timeout", recoverable=False)
    assert err.recoverable is False


# ---------------------------------------------------------------------------
# is_retryable helper
# ---------------------------------------------------------------------------


def test_is_retryable_for_model_transient() -> None:
    """MODEL_TRANSIENT is retryable."""
    err = AgentError(ErrorCode.MODEL_TRANSIENT, "timeout")
    assert is_retryable(err)


def test_is_retryable_for_tool_transient() -> None:
    """TOOL_TRANSIENT is retryable."""
    err = AgentError(ErrorCode.TOOL_TRANSIENT, "network error")
    assert is_retryable(err)


def test_is_retryable_for_model_output_invalid() -> None:
    """MODEL_OUTPUT_INVALID is retryable (limited retries)."""
    err = AgentError(ErrorCode.MODEL_OUTPUT_INVALID, "bad json")
    assert is_retryable(err)


def test_is_retryable_for_evidence_insufficient() -> None:
    """EVIDENCE_INSUFFICIENT must NOT be retryable."""
    err = AgentError(ErrorCode.EVIDENCE_INSUFFICIENT, "no evidence")
    assert not is_retryable(err)
