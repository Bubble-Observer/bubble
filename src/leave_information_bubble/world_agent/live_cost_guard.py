"""Pre-call, opt-in cost guard for the isolated P5-G1 live probe."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from leave_information_bubble.gateway.client import StructuredModelOutputError, ToolModelResponse
from leave_information_bubble.runtime.errors import AgentError, ErrorCode

FROZEN_G1_MODEL = "deepseek-v4-flash"
# Safety reservation rates use DeepSeek's PEAK tier (effective 2026-08-17,
# ~7.1 CNY/USD): reservations are upper bounds, so the guard stays
# conservative even if a run slips into peak hours. The client ledger
# settles at the actual off-peak tier (deepseek_client._estimate_cost).
_INPUT_USD_PER_MILLION = Decimal("0.423")
_OUTPUT_USD_PER_MILLION = Decimal("1.268")
#: Audit-stance token estimate = UTF-8 request_bytes + this protocol reserve
#: (docs/token-estimation-accounting.md §1); ≈ checkpoint stance × 1.75 for
#: Chinese/JSON-dense requests, so cost-cap thresholds are NOT comparable to
#: the context-boundary thresholds without the conversion factor.
_PROTOCOL_TOKEN_RESERVE = 2048
_DEFAULT_MAX_TOKENS = 4096
_MILLION = Decimal("1000000")


class ToolModelInvoker(Protocol):
    """Minimal provider boundary required by the live cost guard."""

    async def invoke_tools(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]], **kwargs: object
    ) -> ToolModelResponse:
        """Invoke one provider-native tool call."""


@dataclass(frozen=True)
class _Reservation:
    """Metadata-only state needed to settle one guarded request."""

    request_hash: str
    request_bytes: int
    estimated_input_tokens: int
    max_tokens: int
    reservation_usd: Decimal


@dataclass
class LiveCostGuard:
    """Reserve a conservative per-call upper bound before an opt-in live request.

    The guard is deliberately process-local and narrowly bound to the frozen G1
    model.  Reservations remain outstanding when a provider call has unknown
    billing, which favors a safe stop over an optimistic retry.
    """

    model: ToolModelInvoker
    model_name: str
    hard_cap_usd: Decimal
    audit_path: Path
    _settled_usd: Decimal = field(default=Decimal("0"), init=False)
    _outstanding: dict[str, _Reservation] = field(default_factory=dict, init=False)
    _sequence: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        """Reject invalid G1-only configuration before any provider invocation."""
        if self.model_name != FROZEN_G1_MODEL:
            raise ValueError(f"live cost guard supports only {FROZEN_G1_MODEL}")
        if not self.hard_cap_usd.is_finite() or self.hard_cap_usd <= 0:
            raise ValueError("live hard cap must be a finite value greater than zero")
        validate_fresh_audit_path(self.audit_path)

    async def invoke_tools(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]], **options: object
    ) -> ToolModelResponse:
        """Reserve before dispatch, settle known usage, and preserve unknown usage."""
        call_options = dict(options)
        max_tokens = call_options.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        request_hash, request_bytes = _request_identity(messages, tools, call_options)
        estimated_input_tokens = request_bytes + _PROTOCOL_TOKEN_RESERVE
        reservation_usd = _upper_bound_usd(estimated_input_tokens, max_tokens)
        reservation_id = await self._reserve(
            request_hash=request_hash,
            request_bytes=request_bytes,
            estimated_input_tokens=estimated_input_tokens,
            max_tokens=max_tokens,
            reservation_usd=reservation_usd,
        )
        try:
            response = await self.model.invoke_tools(messages, tools=tools, **call_options)
        except StructuredModelOutputError as error:
            if _has_reliable_usage(error.response):
                try:
                    await self._settle_known(reservation_id, error.response.cost_usd)
                except ValueError:
                    await self._mark_unknown(reservation_id)
            else:
                await self._mark_unknown(reservation_id)
            raise
        except Exception:
            await self._mark_unknown(reservation_id)
            raise
        if not _has_reliable_usage(response):
            await self._mark_unknown(reservation_id)
            raise AgentError(
                ErrorCode.BUDGET_EXCEEDED,
                "live provider response has no reliable usage; reservation remains outstanding",
                recoverable=False,
            )
        try:
            await self._settle_known(reservation_id, response.cost_usd)
        except ValueError:
            await self._mark_unknown(reservation_id)
            raise
        return response

    async def _reserve(
        self,
        *,
        request_hash: str,
        request_bytes: int,
        estimated_input_tokens: int,
        max_tokens: int,
        reservation_usd: Decimal,
    ) -> str:
        """Record an upper-bound reservation before the underlying model is called."""
        async with self._lock:
            outstanding_usd = _outstanding_total(self._outstanding)
            projected = (
                self._settled_usd
                + outstanding_usd
                + reservation_usd
            )
            if projected >= self.hard_cap_usd:
                self._append_audit(
                    event="rejected",
                    reservation_id=None,
                    request_hash=request_hash,
                    request_bytes=request_bytes,
                    estimated_input_tokens=estimated_input_tokens,
                    max_tokens=max_tokens,
                    reservation_usd=reservation_usd,
                    settled_usd=None,
                    status="budget_exceeded",
                    cumulative_settled_usd=self._settled_usd,
                    cumulative_outstanding_usd=outstanding_usd,
                    cumulative_projected_usd=projected,
                )
                raise AgentError(
                    ErrorCode.BUDGET_EXCEEDED,
                    "live pre-call hard cap would be reached or exceeded",
                    recoverable=False,
                )
            reservation_id = f"g1-{self._sequence + 1}"
            reservation = _Reservation(
                request_hash=request_hash,
                request_bytes=request_bytes,
                estimated_input_tokens=estimated_input_tokens,
                max_tokens=max_tokens,
                reservation_usd=reservation_usd,
            )
            self._append_audit(
                event="reserved",
                reservation_id=reservation_id,
                request_hash=request_hash,
                request_bytes=request_bytes,
                estimated_input_tokens=estimated_input_tokens,
                max_tokens=max_tokens,
                reservation_usd=reservation_usd,
                settled_usd=None,
                status="reserved",
                cumulative_settled_usd=self._settled_usd,
                cumulative_outstanding_usd=outstanding_usd + reservation_usd,
                cumulative_projected_usd=projected,
            )
            self._sequence += 1
            self._outstanding[reservation_id] = reservation
            return reservation_id

    async def _settle_known(self, reservation_id: str, cost_usd: float) -> None:
        """Replace one reservation with reliable provider-reported actual cost."""
        settled_usd = _decimal_cost(cost_usd)
        async with self._lock:
            reservation = self._outstanding[reservation_id]
            next_settled_usd = self._settled_usd + settled_usd
            next_outstanding_usd = _outstanding_total(self._outstanding) - reservation.reservation_usd
            self._append_audit(
                event="settled",
                reservation_id=reservation_id,
                request_hash=reservation.request_hash,
                request_bytes=reservation.request_bytes,
                estimated_input_tokens=reservation.estimated_input_tokens,
                max_tokens=reservation.max_tokens,
                reservation_usd=reservation.reservation_usd,
                settled_usd=settled_usd,
                status="settled",
                cumulative_settled_usd=next_settled_usd,
                cumulative_outstanding_usd=next_outstanding_usd,
                cumulative_projected_usd=next_settled_usd + next_outstanding_usd,
            )
            self._outstanding.pop(reservation_id)
            self._settled_usd = next_settled_usd

    async def _mark_unknown(self, reservation_id: str) -> None:
        """Keep a reservation outstanding after an exception with unknown usage."""
        async with self._lock:
            reservation = self._outstanding[reservation_id]
            outstanding_usd = _outstanding_total(self._outstanding)
            self._append_audit(
                event="unknown",
                reservation_id=reservation_id,
                request_hash=reservation.request_hash,
                request_bytes=reservation.request_bytes,
                estimated_input_tokens=reservation.estimated_input_tokens,
                max_tokens=reservation.max_tokens,
                reservation_usd=reservation.reservation_usd,
                settled_usd=None,
                status="outstanding_unknown",
                cumulative_settled_usd=self._settled_usd,
                cumulative_outstanding_usd=outstanding_usd,
                cumulative_projected_usd=self._settled_usd + outstanding_usd,
            )

    def _append_audit(
        self,
        *,
        event: str,
        reservation_id: str | None,
        request_hash: str | None,
        request_bytes: int | None,
        estimated_input_tokens: int | None,
        max_tokens: int | None,
        reservation_usd: Decimal,
        settled_usd: Decimal | None,
        status: str,
        cumulative_settled_usd: Decimal,
        cumulative_outstanding_usd: Decimal,
        cumulative_projected_usd: Decimal,
    ) -> None:
        """Append a metadata-only JSONL row without retaining request content."""
        row = {
            "event": event,
            "reservation_id": reservation_id,
            "request_sha256": request_hash,
            "request_bytes": request_bytes,
            "estimated_input_tokens": estimated_input_tokens,
            "max_output_tokens": max_tokens,
            "reservation_usd": _decimal_text(reservation_usd),
            "settled_usd": None if settled_usd is None else _decimal_text(settled_usd),
            "status": status,
            "cumulative_settled_usd": _decimal_text(cumulative_settled_usd),
            "cumulative_outstanding_usd": _decimal_text(cumulative_outstanding_usd),
            "cumulative_projected_usd": _decimal_text(cumulative_projected_usd),
        }
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _request_identity(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]], options: dict[str, object]
) -> tuple[str, int]:
    """Return a stable hash and UTF-8 byte count for a provider request shape."""
    payload = json.dumps(
        {"messages": messages, "options": options, "tools": tools},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def validate_fresh_audit_path(audit_path: Path) -> None:
    """Create or verify an empty audit file before a fresh G1 process may run."""
    if not audit_path.parent.is_dir():
        raise ValueError("live cost audit parent directory must already exist")
    if audit_path.exists() and audit_path.stat().st_size != 0:
        raise ValueError("live cost audit file must be absent or empty for a fresh G1 run")
    with audit_path.open("a", encoding="utf-8", newline="\n"):
        pass


def _upper_bound_usd(estimated_input_tokens: int, max_tokens: int) -> Decimal:
    """Compute the frozen G1 per-call upper bound without rounding down."""
    return (
        Decimal(estimated_input_tokens) * _INPUT_USD_PER_MILLION
        + Decimal(max_tokens) * _OUTPUT_USD_PER_MILLION
    ) / _MILLION


def _outstanding_total(outstanding: dict[str, _Reservation]) -> Decimal:
    """Return the process-local total of reservations not safely settled yet."""
    return sum((item.reservation_usd for item in outstanding.values()), Decimal("0"))


def _has_reliable_usage(response: ToolModelResponse) -> bool:
    """Return whether the smallest required provider usage signal is present."""
    return response.prompt_tokens > 0


def _decimal_cost(cost_usd: float) -> Decimal:
    """Convert a non-negative reported cost without binary-float arithmetic."""
    try:
        value = Decimal(str(cost_usd))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("provider cost_usd is not a valid decimal") from error
    if not value.is_finite() or value < 0:
        raise ValueError("provider cost_usd must be a finite non-negative decimal")
    return value


def _decimal_text(value: Decimal) -> str:
    """Format audit amounts without scientific notation for exact zero."""
    return "0" if value == 0 else str(value)
