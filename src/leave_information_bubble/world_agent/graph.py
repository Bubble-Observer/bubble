"""Graph Shell: a small ReAct loop over the graph-editing tools.

Deterministic staging/publish path (graph_patch → graph_inspect/graph_diff →
finalize_graph) with an explicit recovery lane. No legacy proposal imports.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, TypedDict
from zoneinfo import ZoneInfo

import jsonschema
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from leave_information_bubble.gateway.client import (
    StructuredModelOutputError,
    ToolModelResponse,
)
from leave_information_bubble.runtime.errors import AgentError
from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world.domain_config import DomainFocus, resolve_domain_focus
from leave_information_bubble.world.finalize import finalize_graph
from leave_information_bubble.world.graph_patch_contract import (
    graph_patch_arguments_violation,
)
from leave_information_bubble.world.preflight import inspect_working_graph
from leave_information_bubble.world.writer_lease import lease_owner_is

from .live_deadline import LiveDeadlineExceeded
from .model_calls import ModelCallRecorder, ModelRequestEnvelope
from .prompt import AgentMode, graph_shell_prompt

# Stable tool error loop (Core §Slice 1 / design P9): the same error code
# repeated this many consecutive times halts the wake with a diagnostic
# instead of spinning forever.
TOOL_ERROR_STREAK_LIMIT = 3
_TOOL_EVENT_CAP = 128
_TOOL_EVENT_TEXT_CAP = 160
_TOOL_EVENT_LIMITATION_CAP = 8
_TOOL_EVENT_UNRESOLVED_CAP = 3

_GRAPH_FORMAT_CHECKER = jsonschema.FormatChecker()


@_GRAPH_FORMAT_CHECKER.checks("date-time")
def _is_tz_aware_iso_datetime(value: object) -> bool:
    """Validate the tz-aware ISO contract even without optional rfc3339 extras."""
    if not isinstance(value, str):
        return True  # the schema's type keyword reports non-strings
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


_TOOL_EVENT_ARGUMENTS = frozenset(
    {
        "adapter",
        "depth",
        "inquiry_id",
        "limit",
        "observation_id",
        "prompt",
        "query",
        "topic",
    }
)


def _tool_event(
    *,
    call_id: str,
    turn: int,
    name: str,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    """Return one bounded, checkpoint-safe tool fact for the end-of-wake report."""
    cards = result.get("cards")
    first_card = cards[0] if isinstance(cards, list) and cards and isinstance(cards[0], dict) else {}
    digest = result.get("digest")
    unresolved = digest.get("unresolved", []) if isinstance(digest, dict) else []

    def bounded(value: object) -> str:
        return str(value)[:_TOOL_EVENT_TEXT_CAP]

    unresolved_text = [item.get("what", "") if isinstance(item, Mapping) else item for item in unresolved]
    diagnostic: dict[str, Any] = {}
    if result.get("outcome") is not None:
        diagnostic["outcome"] = bounded(result["outcome"])
    for key in ("error", "scope", "completeness", "novelty", "returned"):
        value = result.get(key)
        if isinstance(value, Mapping):
            diagnostic[key] = _bounded_tool_diagnostic(value)
    payload = result.get("payload")
    if isinstance(payload, Mapping) and payload.get("status") is not None:
        diagnostic["status"] = bounded(payload["status"])
    for key in ("limit", "truncated", "found_ids", "missing_ids"):
        if key in result:
            diagnostic[key] = _bounded_tool_diagnostic(result[key])
    return {
        "call_id": call_id,
        "turn": turn,
        "name": name,
        "arguments": {
            key: bounded(value) for key, value in arguments.items() if key in _TOOL_EVENT_ARGUMENTS
        },
        "ok": bool(result.get("ok")),
        "limitations": [
            bounded(item) for item in result.get("limitations", [])[:_TOOL_EVENT_LIMITATION_CAP] if str(item)
        ],
        "elapsed_ms": max(0.0, round(elapsed_ms, 3)),
        "card": {
            key: bounded(first_card[key])
            for key in ("id", "observation_id", "title")
            if first_card.get(key) is not None
        },
        "unresolved": [bounded(item) for item in unresolved_text[:_TOOL_EVENT_UNRESOLVED_CAP] if str(item)],
        "diagnostic": diagnostic,
    }


def _tool_error_code(result: Mapping[str, Any]) -> str | None:
    """Extract a stable error code from a tool result, or None when it succeeded.

    The code feeds the consecutive-streak counter: the same code repeated
    ``TOOL_ERROR_STREAK_LIMIT`` times halts the wake. Codes group the varied
    failure shapes (limitations list, error string, bounded error dict) into
    one comparable key without changing any result payload.
    """
    if result.get("ok") is not False:
        return None
    limitations = result.get("limitations")
    if isinstance(limitations, list) and limitations:
        return f"limitation:{str(limitations[0])[:_TOOL_EVENT_TEXT_CAP]}"
    error = result.get("error")
    if isinstance(error, Mapping):
        code = error.get("code") or error.get("field") or "unknown"
        return f"error:{str(code)[:_TOOL_EVENT_TEXT_CAP]}"
    if isinstance(error, str) and error:
        return f"error:{error[:_TOOL_EVENT_TEXT_CAP]}"
    return "tool_failure"


def _tool_error_signature(name: str, result: Mapping[str, Any]) -> str | None:
    """Identify one repeated failure by tool, code, and actionable field set."""
    code = _tool_error_code(result)
    if code is None:
        return None
    fields: list[str] = []
    error = result.get("error")
    if isinstance(error, Mapping):
        violations = error.get("violations")
        if isinstance(violations, list):
            fields.extend(
                str(item.get("field"))
                for item in violations
                if isinstance(item, Mapping) and item.get("field")
            )
        if not fields and error.get("field"):
            fields.append(str(error["field"]))
    raw_field_key = ",".join(sorted(set(fields)))
    if len(raw_field_key) <= _TOOL_EVENT_TEXT_CAP:
        field_key = raw_field_key
    else:
        # Preserve a readable prefix without letting two long, independently
        # actionable field sets collapse merely because they share it.
        digest = hashlib.sha256(raw_field_key.encode("utf-8")).hexdigest()[:16]
        prefix_limit = _TOOL_EVENT_TEXT_CAP - len(digest) - 2
        field_key = f"{raw_field_key[:prefix_limit]}~{digest}"
    return f"{name[:80]}|{code}|{field_key}"


def _bounded_tool_diagnostic(value: object, depth: int = 0) -> object:
    """Project structured tool facts into a small checkpoint-safe JSON value."""
    if depth >= 3:
        return "[bounded]"
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 12:
                break
            output[str(key)[:80]] = _bounded_tool_diagnostic(item, depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        return [_bounded_tool_diagnostic(item, depth + 1) for item in value[:8]]
    if isinstance(value, str):
        return value[:_TOOL_EVENT_TEXT_CAP]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_TOOL_EVENT_TEXT_CAP]


# Soft evidence-citation discipline for terminal proposal feedback: observation
# ids must be reproduced exactly as tool results returned them; remembered
# content never justifies a constructed id. Hallucinated ids are sanitized at
# the proposal boundary, but the prompt works on the root cause.

# H1 grounding: terminal feedback (finalize/repair) names the durable
# memory_ids the agent actually touched through memory tools, turning the
# proposal step from recall into choice. The list is capped so the prompt
# never drowns; only memory tool results that carry object/inquiry cards are
# read (memory_changes and memory_evidence return bare ids / observation
# ids, which do not ground a memory_id choice).


# Task 6 (spec §5.4): the merged repair feedback consolidates every known
# rejection class into ONE message — at most 8 item-indexed entries with an
# executable action each, the rest counted and omitted. This applies the same
# 8-item discipline as the assertion-target repair cap (the ~8K-char repair
# message that itself drove model noncompliance), and the raw rejection text
# is bounded separately so the whole message stays far below it.
_FEEDBACK_ERROR_LIMIT = 2_400
_FEEDBACK_MESSAGE_LIMIT = 240
# Task 3.3: the structured issue diff caps its candidates, basis, and
# dependency lists with one small constant (mirroring the existing item cap)
# so a repair stays bounded however many candidates a rejection carried.
_ISSUE_KIND_SECTION = {
    "new_object": "new_objects",
    "object_update": "object_updates",
    "assertion": "assertions",
    "new_inquiry": "new_inquiries",
    "resolve_inquiry": "resolve_inquiries",
    "observation_link": "observation_links",
}


def _bounded_text(value: object, limit: int) -> str:
    """Render a feedback value as plain bounded text, never raw JSON."""
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 16]}... [truncated]"


def _render_omitted_dependency(dependency: object) -> str:
    """Render one omitted-dependency entry as a compact structured line.

    The contract's dependency field maps to the review issue's
    ``omitted_dependencies`` entries (each naming the referencing item kind,
    index, local reference, and why it was omitted), rendered as
    ``<kind>[<index>] <local_ref> (<reason>)`` so the model can reattach or
    drop each dependent item.
    """
    if not isinstance(dependency, Mapping):
        return _bounded_text(dependency, _FEEDBACK_MESSAGE_LIMIT)
    kind = _ISSUE_KIND_SECTION.get(
        str(dependency.get("item_kind") or ""), str(dependency.get("item_kind") or "item")
    )
    index = dependency.get("item_index")
    index_text = f"[{index}]" if isinstance(index, int) else ""
    rendered = f"{kind}{index_text}"
    local_ref = dependency.get("local_ref")
    if local_ref:
        rendered += f" {local_ref}"
    reason = dependency.get("reason")
    if reason:
        rendered += f" ({reason})"
    return rendered


# Spec §4.2 soft hint: a literal-target assertion whose text names an
# existing entity (exact alias match under the store's own normalization)
# earns a bounded nudge in the feedback — never a rejection; whether to
# rewrite the target as an object edge is the model's judgment. Bounds: at
# most _LITERAL_SCAN_CAP literal texts are scanned per proposal and at most
# _LITERAL_HIT_CAP hit entities are named (the rest counted), so the hint
# stays bounded however many literals and entities exist. The scan reads
# only the store (objects + object_aliases) and affects only the feedback
# message — never the committed delta — so a replay of the same proposal is
# byte-identical with or without the hint.

# Transcript role regime (cache-slice discipline, spec §6): 2 means the
# mid-stream role=system injections were converted to role=user (Task 1) and
# the proposal/digest transcripts are no longer flattened or rewritten after
# the fact (Task 2), so the provider's prefix cache survives across turns.
# Checkpoints recorded before this regime (role_epoch missing or < 2) still
# resume: the guard records a compatibility note instead of rejecting, and
# the transcript's legacy system messages are preserved as-is. The value
# lives in the context-layout rules module (imported above) so every
# reference — compatibility logic included — reads the same constant.

# Soft correction guidance appended to alias-collision feedback: the model
# decides whether the entity matches (guidance, not auto-reuse).

# Self-reflection hinting (spec §9): only fact-class roles raise cognitive
# conflict hints. Literal-different-but-compatible pairs are the inherent
# false-positive risk of exact matching; restricting the hint to FACT (the
# only unambiguous fact class) keeps it low.

# The A2 state injection caps the confirmed section at three items and the
# blind-spot section at five (inquiries injection experiment); conflict hints
# follow a separate bound.

# P4 keeps the current-protocol injection cadence and route, but replaces the
# fixed scan checklist with a non-commanding optional orientation. State fields
# themselves remain unchanged.


# Task 2.2 ghost-memory-id candidates: a proposal memory_id that names no
# stored object is never rewritten by name similarity — the committer
# rejects it with an invalid_reference issue (basis missing_memory_id). The
# repair feedback may instead carry bounded stored-object candidates
# recovered from the transcript's memory cards: the same id-hash suffix the
# removed H2 remap used (the b1 failure class — an inquiry card's hash
# reused as an object id) matched against stored canonical names and aliases
# at >= 0.20 bigram jaccard, the shared subtitle-reliability separation line
# the committer's inquiry dedup and the recall possible-match arm are
# calibrated to. Candidates only decorate the issue; the ghost id stays
# unresolved and the next proposal must name a real id or declare a local ref.


class ToolCallingModel(Protocol):
    """The provider-native model boundary needed by the world-agent graph."""

    def invoke_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        thinking: bool | None = None,
        max_tokens: int = 4096,
    ) -> Awaitable[ToolModelResponse]:
        """Return one assistant message, optionally containing native function calls."""


#: Frozen per-model context-window contract (official provider numbers,
#: verified from the DeepSeek API docs 2026-08-23). The Graph Shell hard cut
#: and warning thresholds stay BELOW the window: the chars-based estimate
#: under-measures provider-side overhead (system prompt, tool schemas,
#: protocol/caching), and a bare provider 400 would end the run without any
#: Graph Shell terminal semantics. See
#: ``docs/plans/2026-08-22-context-threshold-soft-reminder-design.md``.
CONTEXT_WINDOW_TOKENS: dict[str, int] = {"deepseek-v4-flash": 1_000_000}
DEFAULT_CONTEXT_WARNING_TOKENS = 700_000
DEFAULT_CONTEXT_HARD_CUT_TOKENS = 800_000
# The wind-down notice states a plain, objective wind-down plan the way an
# operator would put it — not a resource report ("context at N/M tokens")
# and not a command ("call finalize_graph now"). It tells the agent the
# exploration has enough material, to check what the working graph already
# holds, publish the settled parts, and stop; the remaining space is for
# winding down, not for new exploration.
CONTEXT_WIND_DOWN_NOTICE = (
    "这次探索的信息已经足够，收个尾吧：检查工作图里已有的内容，"
    "把确定的部分正式写入发布，然后结束本轮。"
    "剩下的空间留给收尾，不用再开新的探索了。"
)


@dataclass(frozen=True)
class WakeReasoningProfile:
    """The ONE frozen provider reasoning configuration for the whole wake.

    Task 5.3: exploration and proposal construct their request options from
    the same frozen instance, so the main-wake envelopes differ only in the
    forced submit tool_choice and their stable wake projection is one
    cacheable key per wake. The isolated digest round keeps its own envelope
    and is not governed by this profile.
    """

    thinking: bool = False
    reasoning_effort: str | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        # Thinking-mode reasoning consumes the output budget before any tool
        # call is written; the off-mode 8192 ceiling truncates the chain
        # mid-thought (observed live in canary w2: completion pinned at 8192,
        # zero tool calls, the wake starved at 11 turns with nothing staged).
        # Give thinking profiles the provider-default reasoning budget.
        if self.max_tokens is None:
            object.__setattr__(
                self,
                "max_tokens",
                32768 if self.thinking else 8192,
            )


class GraphShellState(TypedDict, total=False):
    """Checkpointed state for the opt-in G5b-1 Graph Shell topology."""

    messages: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    turn_count: int
    total_cost_usd: float
    tool_error_streaks: dict[str, int]
    terminal_status: str
    terminal_summary: str
    halted: bool
    resume_allowed: bool
    completion_reminder_used: bool
    context_warning_injected: bool
    context_cut: bool
    context_usage_estimate: int
    finalize_status: str
    finalize_receipt: dict[str, Any]
    patch_success_count: int
    patch_replay_count: int
    resume_count: int
    pending_recovery_summary: str
    raw_exception: dict[str, Any]
    wake_id: str
    thread_id: str
    execution_mode: Literal["graph_shell"]
    world_store_identity: str
    domain_key: str
    agent_mode: AgentMode
    object_seed: str


def graph_shell_initial_state(
    *,
    messages: list[dict[str, Any]],
    store: WorldStore,
    thread_id: str,
    wake_id: str,
    domain_key: str,
    mode: AgentMode,
    object_id: str | None,
) -> GraphShellState:
    """Build the complete identity-bearing state written by the first checkpoint."""
    if not thread_id:
        raise ValueError("thread_id is required for Graph Shell")
    if not wake_id:
        raise ValueError("wake_id is required for Graph Shell")
    return {
        "messages": list(messages),
        "tool_events": [],
        "turn_count": 0,
        "total_cost_usd": 0.0,
        "tool_error_streaks": {},
        "terminal_status": "",
        "terminal_summary": "",
        "halted": False,
        "resume_allowed": True,
        "completion_reminder_used": False,
        "context_warning_injected": False,
        "context_cut": False,
        "context_usage_estimate": 0,
        "finalize_status": "",
        "finalize_receipt": {},
        "patch_success_count": 0,
        "patch_replay_count": 0,
        "resume_count": 0,
        "pending_recovery_summary": "",
        "raw_exception": {},
        "wake_id": wake_id,
        "thread_id": thread_id,
        "execution_mode": "graph_shell",
        "world_store_identity": str(store.path.resolve()),
        "domain_key": domain_key,
        "agent_mode": mode,
        "object_seed": object_id or "",
    }


def _estimate_request_tokens(
    messages: list[dict[str, Any]],
    max_output_tokens: int | None,
    *,
    tools: Sequence[Mapping[str, Any]] = (),
) -> int:
    """Rough pre-call tokens: messages + schemas + reserved output budget.

    Matches the provider adapter's 1.5 chars/token approximation
    (``estimate_tokens``); the output allowance counts toward the physical
    window because the provider constrains input + requested output together
    (design O-1: input + output reservation). Only an estimate — the durable
    per-call authority is ``model_calls.prompt_tokens``, recorded after every
    successful invocation. This is the checkpoint stance of the frozen
    accounting (docs/token-estimation-accounting.md §1); verified vs the
    authoritative usage at ~2.15-2.3x for Chinese/JSON-dense requests.
    """
    text = json.dumps(
        {"messages": messages, "tools": list(tools)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return max(1, (len(text) * 2) // 3 + 1) + int(max_output_tokens or 0)


def _finalize_graph_schema() -> dict[str, Any]:
    """Return the explicit, argument-free Graph Shell publication tool."""
    return {
        "type": "function",
        "function": {
            "name": "finalize_graph",
            "description": (
                "Validate and publish this wake's durable working graph — the only "
                "tool that commits staging to the formal graph. Call it when you "
                "have decided the current delta should become formal; when readiness, "
                "blockers, or the delta are uncertain, inspect first. Blocked, "
                "compile-failed or rejected results "
                "keep staging available for revision. It takes no arguments: "
                "readiness, blockers and the unpublished delta belong to "
                "graph_inspect and graph_diff, which read without committing. The "
                "host never calls this automatically, and a wake publishes at most "
                "once — after publication, later work belongs to a new wake."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _graph_shell_error(
    code: str,
    *,
    field: str | None = None,
    message: str,
    candidates: Sequence[object] = (),
    action_hint: str,
    violations: Sequence[Mapping[str, object]] = (),
    total_violations: int | None = None,
    violations_truncated: bool = False,
) -> dict[str, Any]:
    """Return the bounded repair envelope every Graph Shell tool error uses.

    The candidate window is capped with an explicit ``omitted_counts`` marker
    so a cut is never silent (C3): the model sees that more candidates exist
    than the five returned.
    """
    window = list(candidates)[:5]
    bounded_message = _bounded_text(message, _FEEDBACK_ERROR_LIMIT)
    bounded_hint = _bounded_text(action_hint, _FEEDBACK_ERROR_LIMIT)
    error: dict[str, Any] = {
        "code": code,
        "message": bounded_message,
        "candidates": window,
        "action_hint": bounded_hint,
    }
    if bounded_message != str(message):
        error["message_truncated"] = True
    if bounded_hint != str(action_hint):
        error["action_hint_truncated"] = True
    if field is not None:
        error["field"] = field
    if len(candidates) > len(window):
        error["omitted_counts"] = {"candidates": len(candidates) - len(window)}
    if violations:
        error["violations"] = [
            {
                "code": _bounded_text(item.get("code", code), 120),
                "field": _bounded_text(item.get("field", "arguments"), 240),
                "message": _bounded_text(item.get("message", "invalid value"), 480),
            }
            for item in violations[:32]
        ]
        error["total_violations"] = int(total_violations or len(violations))
        if violations_truncated or error["total_violations"] > len(error["violations"]):
            error["violations_truncated"] = True
    return {"ok": False, "error": error}


_REQUIRED_PROPERTY_RE = re.compile(r"^'([^']+)' is a required property$")
_UNEXPECTED_PROPERTY_RE = re.compile(r"'([^']+)' (?:was|were) unexpected")


def _unexpected_property_names(error: jsonschema.ValidationError) -> list[str]:
    """Recover every property represented by one additionalProperties error."""
    if error.validator != "additionalProperties" or not isinstance(error.instance, Mapping):
        return []
    schema = error.schema if isinstance(error.schema, Mapping) else {}
    properties = schema.get("properties")
    known = set(properties) if isinstance(properties, Mapping) else set()
    pattern_properties = schema.get("patternProperties")
    patterns = list(pattern_properties) if isinstance(pattern_properties, Mapping) else []
    unexpected: list[str] = []
    for raw_name in error.instance:
        name = str(raw_name)
        if raw_name in known:
            continue
        if any(re.search(pattern, name) for pattern in patterns):
            continue
        unexpected.append(name)
    return sorted(set(unexpected))


def _schema_error_field(error: jsonschema.ValidationError) -> str:
    parts = ["arguments", *(str(part) for part in error.absolute_path)]
    field = ".".join(parts)
    required = _REQUIRED_PROPERTY_RE.match(error.message)
    if required:
        return f"{field}.{required.group(1)}"
    if error.validator == "additionalProperties":
        unexpected = _unexpected_property_names(error)
        if unexpected:
            return f"{field}.{unexpected[0]}"
        message_match = _UNEXPECTED_PROPERTY_RE.search(error.message)
        if message_match:
            return f"{field}.{message_match.group(1)}"
    return field


def _schema_violations(
    validator: jsonschema.Draft202012Validator,
    arguments: Mapping[str, Any],
) -> tuple[list[dict[str, str]], int]:
    """Collect independent schema errors with exact paths, bounded to 32."""
    errors: list[jsonschema.ValidationError] = []
    for outer in validator.iter_errors(arguments):
        if outer.validator in {"oneOf", "anyOf"} and outer.context:
            errors.append(jsonschema.exceptions.best_match(outer.context) or outer)
        else:
            errors.append(outer)
    ordered = sorted(errors, key=lambda error: (_schema_error_field(error), error.message))
    details: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for error in ordered:
        fields: list[str] = []
        if error.validator == "additionalProperties":
            base = ".".join(["arguments", *(str(part) for part in error.absolute_path)])
            fields = [f"{base}.{name}" for name in _unexpected_property_names(error)]
        if not fields:
            fields = [_schema_error_field(error)]
        for field in fields:
            message = (
                f"{field}: unexpected property is not allowed"
                if error.validator == "additionalProperties" and len(fields) > 1
                else f"{field}: {error.message}"
            )
            key = (field, message)
            if key in seen:
                continue
            seen.add(key)
            if len(details) < 32:
                details.append(
                    {
                        "code": "invalid_arguments",
                        "field": field,
                        "message": message,
                    }
                )
    return details, len(seen)


def _semantic_argument_violations(name: str, arguments: Mapping[str, Any]) -> list[dict[str, str]]:
    """Enforce cross-field invariants JSON Schema cannot compare directly."""
    details: list[dict[str, str]] = []
    if name == "memory_search":
        minimum = arguments.get("assertion_count_min")
        maximum = arguments.get("assertion_count_max")
        if (
            isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and minimum > maximum
        ):
            details.append(
                {
                    "code": "invalid_arguments",
                    "field": "arguments.assertion_count_min",
                    "message": ("arguments.assertion_count_min must not exceed assertion_count_max"),
                }
            )
        time_from = arguments.get("time_from")
        time_to = arguments.get("time_to")
        if isinstance(time_from, str) and isinstance(time_to, str):
            try:
                parsed_from = datetime.fromisoformat(time_from.replace("Z", "+00:00"))
                parsed_to = datetime.fromisoformat(time_to.replace("Z", "+00:00"))
            except ValueError:
                pass
            else:
                if (
                    parsed_from.tzinfo is not None
                    and parsed_to.tzinfo is not None
                    and parsed_to < parsed_from
                ):
                    details.append(
                        {
                            "code": "invalid_arguments",
                            "field": "arguments.time_to",
                            "message": "arguments.time_to must not precede time_from",
                        }
                    )
    if name in {"discover_sources", "search_sources"}:
        start = arguments.get("window_start")
        end = arguments.get("window_end")
        if isinstance(start, str) and isinstance(end, str):
            try:
                parsed_start = datetime.fromisoformat(start.replace("Z", "+00:00"))
                parsed_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
            except ValueError:
                pass
            else:
                if (
                    parsed_start.tzinfo is not None
                    and parsed_end.tzinfo is not None
                    and parsed_end < parsed_start
                ):
                    details.append(
                        {
                            "code": "invalid_arguments",
                            "field": "arguments.window_end",
                            "message": ("arguments.window_end must not precede window_start"),
                        }
                    )
    return details


# Staging-level rejection codes and their repair guidance. Staging is the
# second validation ring after the schema pre-validation: an item can pass
# the contract yet be refused on identity, version, dependency or duplicate
# grounds. The envelope must point the model at the exact field to fix
# (field map) and, where the staging verdict carries no message or hint, at
# the repair path (primary feedback) — never at a blanket retry.
_PATCH_ITEM_FIELDS: Mapping[str, str] = {
    "missing_op_id": "op_id",
    "op_id_reused": "op_id",
    "unknown_kind": "kind",
    "invalid_kind": "payload.kind",
    "unknown_action": "action",
    "missing_target_ref": "target_ref",
    "missing_canonical_name": "payload.canonical_name",
    "invalid_payload": "payload",
    "create_takes_no_target_ref": "target_ref",
    "missing_assertion_fields": "payload",
    "missing_inquiry_fields": "payload",
    "missing_answers_ref": "payload.answers_ref",
    "missing_supersedes_ref": "payload.supersedes_ref",
    "inquiry_create_takes_no_answers_ref": "payload.answers_ref",
    "invalid_expected_version": "payload.expected_version",
    "version_conflict": "payload.expected_version",
    "dependency_unavailable": "payload",
    "evidence_unavailable": "payload.evidence",
    "exact_duplicate": "payload",
    "overlapping_relation": "payload",
    "invalid_qualifiers": "payload",
    "not_an_answering_assertion": "payload.answers_ref",
    "dependent_exists": "target_ref",
    "target_unavailable": "target_ref",
    "alias_occupied": "payload.aliases",
    "identity_candidate_exists": "payload.canonical_name",
}


def _patch_item_primary_feedback(code: str, item: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Targeted feedback for staging codes without a verdict message.

    Returns (message, action_hint) for codes whose staging verdict carries
    neither; the envelope falls back to these before the generic text, so
    the model sees the actual repair path instead of a blanket "was
    rejected" retry.
    """
    if code == "version_conflict":
        current = item.get("current_version")
        return (
            f"conflicts with the target's current version {current}; resend with "
            "expected_version set to that value",
            "Keep the same op_id, add expected_version to the payload, and resend",
        )
    if code == "identity_candidate_exists":
        return (
            "needs identity resolution against existing candidates — this is not a "
            "rejection; the item can still be staged via a decision",
            "Confirm distinct with decision {action: confirm_distinct, "
            "distinct_from: <candidate id>, basis: <why this is a different "
            "referent>}, or target an existing id instead",
        )
    if code == "alias_occupied":
        occupied = item.get("occupied")
        owner = (
            str(occupied[0]["id"])
            if isinstance(occupied, list) and occupied and isinstance(occupied[0], Mapping)
            else "the existing owner"
        )
        return (
            "one or more aliases are already held by active identity alias owners (hard block)",
            f"Use a different alias, or reference/update the existing owner {owner} instead",
        )
    if code == "dependency_unavailable":
        ref = item.get("target_ref") or item.get("answers_ref") or item.get("supersedes_ref")
        suffix = f": {ref}" if ref else ""
        return (
            f"references an id that is neither formal nor staged in this wake{suffix}",
            "Stage the referenced object, assertion, or inquiry first, or point the "
            "reference at an existing formal/staged id",
        )
    if code == "exact_duplicate":
        return (
            "the payload exactly duplicates an already staged item",
            "Revise the payload so it differs from the existing item, or drop the duplicate",
        )
    if code == "overlapping_relation":
        return (
            "a current relation already uses the same subject, predicate, object, role, and "
            "qualifiers over an overlapping or unknown time span",
            "If this refines or corrects that relation, resend with supersedes_ref set to "
            "existing_id; otherwise use a truthful distinct qualifier or non-overlapping span",
        )
    if code == "invalid_expected_version":
        return (
            "expected_version must be an integer equal to the target's current version",
            "Query the target's current version (graph_inspect / memory_read) and resend",
        )
    if code == "not_an_answering_assertion":
        return (
            "the named assertion does not answer this inquiry",
            "Name an assertion whose answers_ref matches the target inquiry",
        )
    if code == "dependent_exists":
        return (
            "the target is referenced by active staged items; a drop is blocked",
            "Drop or update the dependent items first, or keep the target",
        )
    if code == "target_unavailable":
        return (
            "the target id is neither formal nor staged in this wake",
            "Query existing ids with graph_inspect / memory_read, then reference a valid one",
        )
    if code == "evidence_unavailable":
        return (
            "one or more evidence ids are not stored observation ids",
            "Reference stored observation IDs returned by source or memory tools, "
            "or remove the missing evidence references",
        )
    return None, None


def _patch_item_summary(index: int, item: Mapping[str, Any]) -> str:
    """One compact per-item entry for the aggregated rejection message."""
    code = str(item.get("error_code") or item.get("status") or "rejected")
    if code == "version_conflict":
        return f"items[{index}] version_conflict (current version {item.get('current_version')})"
    if code == "identity_candidate_exists":
        return f"items[{index}] identity_candidate_exists (needs identity resolution)"
    if code == "op_id_reused":
        return f"items[{index}] op_id_reused (op_id used with a different payload)"
    return f"items[{index}] {code}"


# Limitation codes whose generic wrap hint would send the model the wrong
# way: the arguments are fine, the *state* is not. The wrap falls back to
# these before the schema-correction hint.
_LIMITATION_ACTION_HINTS: Mapping[str, str] = {
    "limitation:invalid_cursor": (
        "The cursor is dead or belongs to another query; drop it and re-issue "
        "the original query without a cursor (or page from a fresh call)"
    ),
    "limitation:invalid_arguments": (
        "A parameter value is malformed; check the advertised parameter "
        "descriptions (e.g. tz-aware ISO-8601 datetimes) and retry"
    ),
}


def _limitation_action_hint(code: str) -> str | None:
    hint = _LIMITATION_ACTION_HINTS.get(code)
    if hint is not None:
        return hint
    if code.startswith("limitation:unknown_tool:"):
        return "Use only a tool advertised for this Graph Shell wake"
    return None


def _build_graph_shell_graph(
    *,
    model: ToolCallingModel,
    tools: WorldTools,
    store: WorldStore,
    checkpointer: BaseCheckpointSaver | None,
    max_turns: int,
    max_cost_usd: float | None,
    mode: AgentMode,
    object_id: str | None,
    focus: DomainFocus | None,
    reasoning_profile: WakeReasoningProfile,
    model_calls: ModelCallRecorder | None,
    thread_id: str | None,
    wake_id: str | None,
    operator_instructions: str,
    enforce_writer_lease: bool = False,
    context_warning_tokens: int | None = None,
    context_hard_cut_tokens: int | None = None,
) -> StateGraph:
    """Compile the mutually exclusive G5b-1 working-graph runtime.

    ``enforce_writer_lease`` arms the composition-root lease contract: every
    ``graph_patch`` and ``finalize_graph`` dispatch re-validates that this
    wake still owns the world's singleton writer lease and fails closed with
    the typed ``writer_lease_lost`` error otherwise. The CLI enables it for
    production runs; direct graph harnesses (tests, tooling) that drive the
    graph without the lease machinery keep the guard off by default.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if max_cost_usd is not None and max_cost_usd < 0:
        raise ValueError("max_cost_usd must be non-negative")
    if context_hard_cut_tokens is not None and context_hard_cut_tokens <= 0:
        raise ValueError("context_hard_cut_tokens must be positive")
    if (
        context_hard_cut_tokens is not None
        and context_warning_tokens is not None
        and context_warning_tokens > context_hard_cut_tokens
    ):
        raise ValueError("context_warning_tokens must not exceed context_hard_cut_tokens")
    effective_thread_id = str(thread_id or "")
    effective_wake_id = str(wake_id or "")
    if not effective_thread_id:
        raise ValueError("thread_id is required for Graph Shell")
    if not effective_wake_id:
        raise ValueError("wake_id is required for Graph Shell")
    effective_focus = focus or resolve_domain_focus("lol_cn")
    world_store_identity = str(store.path.resolve())
    object_seed = object_id or ""
    # Search experience is domain guidance for reconnaissance wakes: Broad
    # carries it, Deep stays on the pure universal core. Explicit here at the
    # composition root rather than inferred inside prompt.py, so the choice is
    # visible and a mode change can never silently alter it.
    prompt = graph_shell_prompt(
        mode,
        object_id,
        effective_focus,
        operator_instructions=operator_instructions,
        include_search_experience=(mode == "broad"),
    )
    unsafe_or_legacy = {
        "propose_inquiry",
        "claim_inquiry",
        "release_inquiry",
        "log_inquiry_point",
        "digest_observation",
    }
    wake_schemas = [
        schema for schema in tools.schemas() if schema["function"]["name"] not in unsafe_or_legacy
    ]
    wake_schemas.append(_finalize_graph_schema())
    allowed_graph_shell_tools = frozenset(schema["function"]["name"] for schema in wake_schemas)
    # F-03: one schema source — the exact dicts the Agent was shown. The
    # dispatch validates every call against these before any implementation
    # runs, so advertised contract and executed contract cannot drift.
    shell_schema_validators = {
        schema["function"]["name"]: jsonschema.Draft202012Validator(
            schema["function"]["parameters"],
            format_checker=_GRAPH_FORMAT_CHECKER,
        )
        for schema in wake_schemas
    }

    def _initial(messages: list[dict[str, Any]]) -> GraphShellState:
        return graph_shell_initial_state(
            messages=messages,
            store=store,
            thread_id=effective_thread_id,
            wake_id=effective_wake_id,
            domain_key=effective_focus.domain_key,
            mode=mode,
            object_id=object_id,
        )

    def _ensure_protocol(state: GraphShellState) -> None:
        expected: tuple[tuple[str, object], ...] = (
            ("wake_id", effective_wake_id),
            ("thread_id", effective_thread_id),
            ("execution_mode", "graph_shell"),
            ("world_store_identity", world_store_identity),
            ("domain_key", effective_focus.domain_key),
            ("agent_mode", mode),
            ("object_seed", object_seed),
        )
        for field, requested in expected:
            if state.get(field) != requested:
                raise ValueError(
                    f"checkpoint {field} mismatch: checkpoint={state.get(field)!r}, requested={requested!r}"
                )

    def _guarded(
        handler: Callable[[GraphShellState], Awaitable[GraphShellState]],
    ) -> Callable[[GraphShellState], Awaitable[GraphShellState]]:
        async def guarded(state: GraphShellState) -> GraphShellState:
            _ensure_protocol(state)
            return await handler(state)

        return guarded

    def _raw_exception_fact(error: BaseException, *, recoverable: bool | None = None) -> dict[str, Any]:
        error_code = getattr(error, "code", None)
        return {
            "type": type(error).__name__,
            "code": str(getattr(error_code, "value", error_code) or ""),
            "recoverable": (
                bool(getattr(error, "recoverable", False)) if recoverable is None else recoverable
            ),
            "message": _bounded_text(str(error), _FEEDBACK_ERROR_LIMIT),
        }

    def _unpublished_update(
        state: GraphShellState,
        reason: str,
        *,
        raw_exception: BaseException | None = None,
        recoverable: bool | None = None,
    ) -> GraphShellState:
        report = inspect_working_graph(store, effective_wake_id)
        update: GraphShellState = {
            "terminal_status": "staged_unpublished",
            "terminal_summary": (
                f"{reason}; wake {effective_wake_id} retains "
                f"{report['active_total']} active staged item(s). Resume this wake to continue."
            ),
            "halted": True,
            "resume_allowed": True,
            "finalize_status": state.get("finalize_status", ""),
        }
        if raw_exception is not None:
            update["raw_exception"] = _raw_exception_fact(raw_exception, recoverable=recoverable)
        return update

    def _recoverable_tool_argument_response(
        error: StructuredModelOutputError,
    ) -> ToolModelResponse | None:
        """Recover only intact native calls whose arguments need model repair."""
        response = error.response
        if not isinstance(response, ToolModelResponse):
            return None
        calls = response.message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return None
        needs_argument_repair = False
        for call in calls:
            if not isinstance(call, Mapping) or call.get("type") != "function":
                return None
            function = call.get("function")
            if (
                not isinstance(call.get("id"), str)
                or not call["id"]
                or not isinstance(function, Mapping)
                or not isinstance(function.get("name"), str)
                or not function["name"]
                or not isinstance(function.get("arguments"), str)
            ):
                return None
            try:
                parsed = json.loads(function["arguments"])
            except json.JSONDecodeError:
                needs_argument_repair = True
            else:
                if not isinstance(parsed, dict):
                    needs_argument_repair = True
        return response if needs_argument_repair else None

    async def bootstrap(state: GraphShellState) -> GraphShellState:
        if checkpointer is not None or state.get("execution_mode") == "graph_shell":
            # Checkpointed Graph Shell inputs must already carry the durable
            # protocol identity.  Validate before bootstrap can rebuild any
            # state, otherwise a pre-bootstrap resume could silently switch
            # world DB/domain/mode and overwrite the evidence of that switch.
            _ensure_protocol(state)
        local_now = datetime.now(ZoneInfo(effective_focus.timezone))
        with store.read_connection() as connection:
            counts = (
                int(connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0]),
                int(connection.execute("SELECT COUNT(*) FROM assertions").fetchone()[0]),
                int(connection.execute("SELECT COUNT(*) FROM inquiries WHERE status = 'open'").fetchone()[0]),
            )
        anchor = (
            f"Current knowledge store: {counts[0]} object(s) / {counts[1]} assertion(s) / "
            f"{counts[2]} active question(s).\n"
            f"Local time ({effective_focus.timezone}): {local_now.isoformat()}"
        )
        messages = [
            {"role": "system", "content": f"{prompt}\n\n{anchor}"},
            *list(state.get("messages", [])),
        ]
        return _initial(messages)

    async def agent(state: GraphShellState) -> GraphShellState:
        if state.get("terminal_status") in {"published", "already_published", "wake_closed"}:
            return {}
        turns = int(state.get("turn_count", 0))
        spent = float(state.get("total_cost_usd", 0.0))
        if turns >= max_turns:
            return _unpublished_update(state, "Graph Shell turn boundary reached")
        if max_cost_usd is not None and spent >= max_cost_usd:
            return _unpublished_update(state, "Graph Shell cost boundary reached")
        options: dict[str, Any] = {
            "thinking": reasoning_profile.thinking,
            "reasoning_effort": reasoning_profile.reasoning_effort,
            "max_tokens": reasoning_profile.max_tokens,
        }
        request_messages = list(state["messages"])
        raw_exception = dict(state.get("raw_exception", {}))
        pending_recovery_summary = str(state.get("pending_recovery_summary", ""))
        if pending_recovery_summary:
            request_messages.append({"role": "user", "content": pending_recovery_summary})
        # Physical context boundary (design 2026-08-23, absolute tokens below
        # the provider window): estimate the next request pre-call, hard-cut
        # at the configured limit and inject the wind-down notice once at the
        # warning threshold. The warning is orthogonal to completion_reminder
        # (behavior signal) — this is the resource signal; both dedupe on
        # their own persisted flag.
        context_estimate = 0
        context_warning_injected = False
        if context_hard_cut_tokens is not None:
            context_estimate = _estimate_request_tokens(
                request_messages,
                options["max_tokens"],
                tools=wake_schemas,
            )
            if context_estimate >= context_hard_cut_tokens:
                # The estimate never shrinks across resume (history replays in
                # full), so a full wake must keep failing closed on every
                # resume attempt — no model call is allowed once the physical
                # window is exhausted. ``context_cut`` stays set for report
                # and audit, but the gate does not depend on it.
                return {
                    **_unpublished_update(state, "Graph Shell context boundary reached"),
                    "context_cut": True,
                    "context_usage_estimate": context_estimate,
                }
            if (
                context_warning_tokens is not None
                and not state.get("context_warning_injected")
                and context_estimate >= context_warning_tokens
            ):
                # The notice is a user-voiced directive, not a resource report:
                # the operator would say "wind down and publish", never "your
                # context is at N/M tokens". The model treats user-role input
                # as human instruction, so the plain voice carries the intent.
                request_messages.append({"role": "user", "content": CONTEXT_WIND_DOWN_NOTICE})
                context_warning_injected = True
        try:
            response = await model.invoke_tools(request_messages, tools=wake_schemas, **options)
        except LiveDeadlineExceeded as error:
            return _unpublished_update(
                state,
                "Live deadline reached before a model call",
                raw_exception=error,
            )
        except StructuredModelOutputError as error:
            recovered = _recoverable_tool_argument_response(error)
            if recovered is None:
                return _unpublished_update(
                    state,
                    f"Model error stopped Graph Shell: {_bounded_text(str(error), _FEEDBACK_ERROR_LIMIT)}",
                    raw_exception=error,
                    recoverable=False,
                )
            response = recovered
            raw_exception = _raw_exception_fact(error, recoverable=True)
        except AgentError as error:
            return _unpublished_update(
                state,
                f"Model error stopped Graph Shell: {_bounded_text(str(error), _FEEDBACK_ERROR_LIMIT)}",
                raw_exception=error,
            )
        except sqlite3.Error as error:
            # Storage faults under the wake (most commonly ``database is
            # locked`` while another process holds the world writer) must not
            # crash the run: the wake retains its staged work and can be
            # resumed once the storage condition changes.
            return _unpublished_update(
                state,
                "World storage error stopped Graph Shell: "
                f"{_bounded_text(str(error), _FEEDBACK_ERROR_LIMIT)}",
                raw_exception=error,
                recoverable=True,
            )
        if model_calls is not None:
            if response.effective_request is not None:
                envelope = ModelRequestEnvelope(
                    model=response.effective_request.model,
                    thinking=bool(options.get("thinking")),
                    reasoning_effort=options.get("reasoning_effort"),
                    tool_schemas=list(response.effective_request.tool_schemas),
                    tool_choice=response.effective_request.tool_choice,
                    response_format=response.effective_request.response_format,
                    max_tokens=response.effective_request.max_tokens,
                    source="provider_effective",
                    provider_options=dict(response.effective_request.provider_options),
                )
            else:
                envelope = ModelRequestEnvelope(
                    model="",
                    thinking=bool(options.get("thinking")),
                    reasoning_effort=options.get("reasoning_effort"),
                    tool_schemas=list(wake_schemas),
                    tool_choice=None,
                    response_format=None,
                    max_tokens=options.get("max_tokens"),
                    source="caller_requested_fallback",
                )
            model_calls.append(
                thread_id=effective_thread_id,
                wake_id=effective_wake_id,
                turn=turns + 1,
                purpose="graph_shell",
                wake_protocol="current",
                phase="exploration",
                request_envelope=envelope,
                request_source=envelope.source,
                request_model=envelope.model,
                response=response,
            )
        update: GraphShellState = {
            "messages": [*request_messages, response.message],
            "turn_count": turns + 1,
            "total_cost_usd": spent + response.cost_usd,
            "terminal_summary": "",
            "halted": False,
            "pending_recovery_summary": "",
            "raw_exception": raw_exception,
            "context_usage_estimate": context_estimate,
        }
        if context_warning_injected:
            update["context_warning_injected"] = True
        return update

    def _normalize_patch_batch_result(result: dict[str, Any]) -> dict[str, Any]:
        """Promote an entirely refused patch batch into the stable error loop."""
        payload = result.get("payload")
        items = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(items, list) or not items:
            return result
        closed_item = next(
            (
                (index, item)
                for index, item in enumerate(items)
                if isinstance(item, Mapping) and item.get("error_code") == "wake_closed"
            ),
            None,
        )
        if closed_item is not None:
            index, item = closed_item
            return {
                **result,
                **_graph_shell_error(
                    "wake_closed",
                    field=f"items[{index}]",
                    message=str(item.get("message") or "wake already has a finalize receipt"),
                    action_hint="Stop editing this wake and start a fresh wake identity",
                ),
            }
        if any(isinstance(item, Mapping) and item.get("status") == "ok" for item in items):
            # Preserve per-item isolation: successful items in a mixed batch
            # remain durable and the complete verdict list remains visible.
            return result
        first_index, first = next(
            ((index, item) for index, item in enumerate(items) if isinstance(item, Mapping)),
            (0, {}),
        )
        code = str(first.get("error_code") or "patch_rejected")
        field = f"items[{first_index}].{_PATCH_ITEM_FIELDS.get(code, 'payload')}"
        if code == "op_id_reused":
            message = "op_id was already used with a different payload"
            action_hint = "Use a new op_id for a revised payload; reuse an op_id only for an exact replay"
        else:
            targeted_message, targeted_hint = _patch_item_primary_feedback(code, first)
            # str(None) would collapse the whole fallback chain to "None";
            # only a present verdict message outranks the targeted text
            message = (
                str(first["message"])
                if first.get("message") is not None
                else targeted_message or f"graph_patch item was rejected: {code}"
            )
            hints = first.get("hints")
            action_hint = (
                str(hints[0])
                if isinstance(hints, list) and hints
                else targeted_hint
                or "Correct the rejected item, keep a stable new op_id, and retry the batch"
            )
        # Name every problematic item (not just the first) so one follow-up
        # patch can repair the whole batch — the same aggregation the schema
        # pre-validation applies; full verdicts stay in payload.results.
        problem_items = [
            (index, item)
            for index, item in enumerate(items)
            if isinstance(item, Mapping)
            and item.get("status") in ("rejected", "conflict", "needs_identity_resolution")
        ]
        if len(problem_items) > 1:
            message = f"Graph Patch has {len(problem_items)} problematic item(s): " + "; ".join(
                _patch_item_summary(i, item) for i, item in problem_items
            )
        raw_candidates = first.get("candidates", first.get("occupied", []))
        candidates = raw_candidates if isinstance(raw_candidates, list) else []
        rejected_ids = [
            str(item.get("op_id") or "")
            for item in items
            if isinstance(item, Mapping) and item.get("status") == "rejected"
        ]
        error = _graph_shell_error(
            code,
            field=field,
            message=message,
            candidates=candidates,
            action_hint=action_hint,
        )
        if rejected_ids:
            error["error"]["rejected_ids"] = rejected_ids
        return {**result, **error}

    async def execute_tools(state: GraphShellState) -> GraphShellState:
        messages = list(state["messages"])
        calls = list(messages[-1].get("tool_calls", []))
        results: list[dict[str, Any]] = []
        events = [dict(event) for event in state.get("tool_events", [])]
        seen = {str(event.get("call_id", "")) for event in events}
        streaks = dict(state.get("tool_error_streaks", {}))
        terminal_status = ""
        halt_reason = "Tool error loop limit reached"
        finalize_status = state.get("finalize_status", "")
        finalize_receipt = dict(state.get("finalize_receipt", {}))
        patch_success_count = int(state.get("patch_success_count", 0))
        patch_replay_count = int(state.get("patch_replay_count", 0))
        for call_index, call in enumerate(calls):
            function = call.get("function", {})
            name = str(function.get("name", ""))
            started = time.perf_counter()
            try:
                arguments = json.loads(function.get("arguments", ""))
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
            except (TypeError, ValueError, json.JSONDecodeError):
                result = _graph_shell_error(
                    "invalid_tool_arguments_json",
                    field="arguments",
                    message=f"Malformed JSON arguments for tool {name!r}",
                    action_hint="Return one valid JSON object matching this tool schema",
                )
                arguments = {}
            else:
                if name not in allowed_graph_shell_tools:
                    result = _graph_shell_error(
                        "tool_not_available_in_mode",
                        field="name",
                        message=f"Tool {name!r} is not available in Graph Shell mode",
                        action_hint="Use only a tool advertised for this Graph Shell wake",
                    )
                else:
                    contract_violation = (
                        graph_patch_arguments_violation(
                            arguments,
                            schema=shell_schema_validators[name].schema,
                        )
                        if name == "graph_patch"
                        else None
                    )
                    schema_violations, schema_violation_total = (
                        ([], 0)
                        if name == "graph_patch"
                        else _schema_violations(shell_schema_validators[name], arguments)
                    )
                    semantic_violations = _semantic_argument_violations(name, arguments)
                    if semantic_violations:
                        schema_violation_total += len(semantic_violations)
                        schema_violations = [
                            *schema_violations,
                            *semantic_violations,
                        ][:32]
                    if contract_violation is not None:
                        result = _graph_shell_error(
                            contract_violation.code,
                            field=contract_violation.field,
                            message=contract_violation.message,
                            action_hint=contract_violation.action_hint,
                            violations=contract_violation.violations,
                            total_violations=contract_violation.total_violations,
                            violations_truncated=contract_violation.violations_truncated,
                        )
                    elif schema_violations:
                        # F-03: validate against the exact advertised schema
                        # before the tool implementation runs; the failing call
                        # has zero side effects and lands in the typed error
                        # loop (streak, then corrective feedback).
                        result = _graph_shell_error(
                            "invalid_arguments",
                            field=schema_violations[0]["field"],
                            message=(
                                f"Tool {name!r} has {schema_violation_total} argument "
                                "validation issue(s): "
                                + "; ".join(violation["message"] for violation in schema_violations[:8])
                            ),
                            action_hint=("Correct the arguments to match the advertised schema and retry"),
                            violations=schema_violations,
                            total_violations=schema_violation_total,
                            violations_truncated=(schema_violation_total > len(schema_violations)),
                        )
                    elif name in {"finalize_graph", "graph_patch"}:
                        if enforce_writer_lease and not lease_owner_is(store, effective_wake_id):
                            # the operator released this wake's writer lease
                            # mid-run (--graph-shell-abandon): no further world
                            # mutation may execute under a released gate
                            result = _graph_shell_error(
                                "writer_lease_lost",
                                field="name",
                                message=(
                                    f"Writer lease for wake {effective_wake_id} was "
                                    "released while this run was active; no further "
                                    "world mutations are allowed"
                                ),
                                action_hint=(
                                    "Stop this run; inspect --graph-shell-status "
                                    "before starting or resuming another wake"
                                ),
                            )
                            terminal_status = "staged_unpublished"
                            halt_reason = (
                                "Writer lease lost: the operator released this wake's write gate mid-run"
                            )
                        elif name == "finalize_graph":
                            receipt = finalize_graph(store, effective_wake_id)
                            finalize_receipt = receipt.model_dump(mode="json")
                            finalize_status = receipt.status
                            result = {
                                "ok": True,
                                "payload": finalize_receipt,
                                "scope": {"wake_id": effective_wake_id},
                                "summary": f"finalize_graph returned {receipt.status}",
                            }
                            if receipt.status in {
                                "published",
                                "already_published",
                                "wake_closed",
                            }:
                                terminal_status = receipt.status
                        else:
                            result = await tools.execute(name, arguments, str(call["id"]))
                    else:
                        result = await tools.execute(name, arguments, str(call["id"]))
                    if name == "graph_patch" and result.get("ok"):
                        result = _normalize_patch_batch_result(result)
                    if result.get("ok") is False and not isinstance(result.get("error"), Mapping):
                        code = _tool_error_code(result) or "invalid_arguments"
                        if "storage_unavailable" not in code and "constraint_violation" not in code:
                            # Typed storage/constraint limitations keep their
                            # own message: "rejected its arguments" would
                            # mislead the Agent into fixing arguments when it
                            # should retry or wind down (T1-2).
                            result = {
                                **result,
                                **_graph_shell_error(
                                    code,
                                    field="arguments",
                                    message=f"Tool {name} rejected its arguments ({code})",
                                    action_hint=(
                                        _limitation_action_hint(code)
                                        or "Correct the arguments using the advertised schema and retry"
                                    ),
                                ),
                            }
                    if name == "graph_patch" and result.get("ok"):
                        patch_success_count += sum(
                            1
                            for item in result.get("payload", {}).get("results", [])
                            if item.get("status") == "ok"
                        )
                        patch_replay_count += sum(
                            bool(item.get("replayed"))
                            for item in result.get("payload", {}).get("results", [])
                        )
            elapsed_ms = (time.perf_counter() - started) * 1_000
            if str(call.get("id", "")) not in seen:
                events.append(
                    _tool_event(
                        call_id=str(call.get("id", "")),
                        turn=int(state.get("turn_count", 0)),
                        name=name,
                        arguments=arguments,
                        result=result,
                        elapsed_ms=elapsed_ms,
                    )
                )
                seen.add(str(call.get("id", "")))
            if result.get("ok") is False:
                code = _tool_error_code(result) or "tool_failure"
                if code == "error:wake_closed":
                    terminal_status = "wake_closed"
                    finalize_status = "wake_closed"
                else:
                    signature = _tool_error_signature(name, result) or code
                    current = streaks.get(signature, 0) + 1
                    streaks = {signature: current}
                    if current >= TOOL_ERROR_STREAK_LIMIT:
                        terminal_status = "staged_unpublished"
            else:
                streaks = {}
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id", "")),
                    "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                }
            )
            if terminal_status in {
                "published",
                "already_published",
                "wake_closed",
                "staged_unpublished",
            }:
                # The durable finalize receipt is authoritative.  A model may
                # emit multiple calls in one assistant message, but no later
                # mutation or malformed call may run after publication or
                # overwrite the terminal checkpoint state.
                for skipped in calls[call_index + 1 :]:
                    skipped_function = skipped.get("function", {})
                    skipped_name = str(skipped_function.get("name", ""))
                    skipped_result = _graph_shell_error(
                        "tool_call_skipped_after_terminal",
                        field="name",
                        message=(
                            f"Tool {skipped_name!r} was not executed after Graph Shell "
                            f"entered terminal status {terminal_status!r}"
                        ),
                        action_hint="Start or resume only when the wake is non-terminal",
                    )
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(skipped.get("id", "")),
                            "content": json.dumps(
                                skipped_result,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    )
                break
        update: GraphShellState = {
            "messages": [*messages, *results],
            "tool_events": events[-_TOOL_EVENT_CAP:],
            "tool_error_streaks": streaks,
            "patch_success_count": patch_success_count,
            "patch_replay_count": patch_replay_count,
            "finalize_status": finalize_status,
            "finalize_receipt": finalize_receipt,
        }
        if terminal_status in {"published", "already_published", "wake_closed"}:
            update.update(
                terminal_status=terminal_status,
                terminal_summary=f"Graph Shell finalize terminal: {terminal_status}",
                halted=True,
                resume_allowed=False,
            )
        elif terminal_status == "staged_unpublished":
            update.update(_unpublished_update(state, halt_reason))
        return update

    async def completion_reminder(state: GraphShellState) -> GraphShellState:
        message = {
            "role": "user",
            "content": (
                "You made no tool call. If the working graph should be published, inspect it "
                "and explicitly call finalize_graph. Otherwise make one useful edit/tool call "
                "or stop again; stopping keeps staged work unpublished."
            ),
        }
        return {
            "messages": [*state["messages"], message],
            "completion_reminder_used": True,
        }

    async def stage_unpublished(state: GraphShellState) -> GraphShellState:
        return _unpublished_update(state, "Graph Shell stopped without explicit finalize_graph")

    def route_after_agent(
        state: GraphShellState,
    ) -> Literal["tools", "completion_reminder", "stage_unpublished", "__end__"]:
        if state.get("terminal_status"):
            return END
        calls = state["messages"][-1].get("tool_calls", [])
        if calls:
            return "tools"
        # An empty working graph needs no publication reminder. This avoids a
        # compulsory second model call when the Agent intentionally found
        # nothing worth preserving. Active staging still receives the one
        # wake-scoped reminder before ending unpublished.
        has_active_staging = True
        with suppress(sqlite3.Error):
            has_active_staging = bool(inspect_working_graph(store, effective_wake_id).get("active_total", 0))
        # A transient read fault leaves the conservative True default, so it
        # cannot suppress the reminder when unpublished work may exist.
        if has_active_staging and not state.get("completion_reminder_used", False):
            return "completion_reminder"
        return "stage_unpublished"

    def route_after_tools(state: GraphShellState) -> Literal["agent", "__end__"]:
        return END if state.get("terminal_status") else "agent"

    builder = StateGraph(GraphShellState)
    builder.add_node("bootstrap", bootstrap)
    builder.add_node("agent", _guarded(agent))
    builder.add_node("tools", _guarded(execute_tools))
    builder.add_node("completion_reminder", _guarded(completion_reminder))
    builder.add_node("stage_unpublished", _guarded(stage_unpublished))
    builder.add_edge(START, "bootstrap")
    builder.add_edge("bootstrap", "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "completion_reminder": "completion_reminder",
            "stage_unpublished": "stage_unpublished",
            END: END,
        },
    )
    builder.add_edge("completion_reminder", "agent")
    builder.add_edge("stage_unpublished", END)
    builder.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
    return builder.compile(checkpointer=checkpointer)


_DEFAULT_REASONING_PROFILE = WakeReasoningProfile()


def build_world_agent_graph(
    *,
    model: ToolCallingModel,
    tools: WorldTools,
    store: WorldStore,
    checkpointer: BaseCheckpointSaver | None = None,
    max_turns: int = 24,
    max_cost_usd: float | None = None,
    mode: AgentMode = "broad",
    object_id: str | None = None,
    focus: DomainFocus | None = None,
    reasoning_profile: WakeReasoningProfile = _DEFAULT_REASONING_PROFILE,
    model_calls: ModelCallRecorder | None = None,
    thread_id: str | None = None,
    wake_id: str | None = None,
    operator_instructions: str = "",
    enforce_writer_lease: bool = False,
    context_warning_tokens: int | None = None,
    context_hard_cut_tokens: int | None = None,
) -> StateGraph:
    """Compile the Graph Shell runtime — the one normal production graph.

    G5b-2: the legacy submit_cognition / proposal / repair / amendment graph
    is retired. This single entry point builds the working-graph runtime for
    the composition root and every test harness. ``enforce_writer_lease``
    arms the composition-root lease contract: every ``graph_patch`` and
    ``finalize_graph`` dispatch re-validates that this wake still owns the
    world's singleton writer lease and fails closed with the typed
    ``writer_lease_lost`` error otherwise. ``context_warning_tokens`` /
    ``context_hard_cut_tokens`` arm the physical-context boundary
    (design 2026-08-23): warn-then-wind-down injection and the hard stop;
    when both are None (the default) the checks stay inert.
    """
    return _build_graph_shell_graph(
        model=model,
        tools=tools,
        store=store,
        checkpointer=checkpointer,
        max_turns=max_turns,
        max_cost_usd=max_cost_usd,
        mode=mode,
        object_id=object_id,
        focus=focus,
        reasoning_profile=reasoning_profile,
        model_calls=model_calls,
        thread_id=thread_id,
        wake_id=wake_id,
        operator_instructions=operator_instructions,
        enforce_writer_lease=enforce_writer_lease,
        context_warning_tokens=context_warning_tokens,
        context_hard_cut_tokens=context_hard_cut_tokens,
    )
