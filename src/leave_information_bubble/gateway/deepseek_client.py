"""DeepSeek API client — concrete ModelClient using the openai SDK.

DeepSeek's API is OpenAI-compatible, so we use openai.AsyncOpenAI pointed at
https://api.deepseek.com. The default model is ``deepseek-v4-flash``.

Error mapping (OpenAI → ErrorCode):
- Timeout, APIConnectionError → MODEL_TRANSIENT (retryable)
- RateLimitError → MODEL_TRANSIENT (retryable)
- AuthenticationError → INTERNAL_ERROR (not retryable — config problem)
- Invalid JSON in structured output → MODEL_OUTPUT_INVALID (retryable, max 3)

References:
- docs/adr/0001-agent-orchestration-baseline.md §建议决定
- docs/02-agent-reference-model.md §7 (error classification)
- spikes/spike-06-model-swap/ (model independence pattern)

"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from openai import (
    APITimeoutError as OpenAITimeoutError,
)
from openai.types.chat import ChatCompletionMessageParam

from leave_information_bubble.gateway.client import (
    EffectiveToolRequest,
    ModelClient,
    ModelResponse,
    NativeToolCall,
    StructuredModelOutputError,
    ToolModelResponse,
)
from leave_information_bubble.runtime.errors import AgentError, ErrorCode

logger = logging.getLogger(__name__)


def _native_assistant_message(
    message: Any,
    calls: list[Any],
    *,
    preserve_reasoning: bool = False,
) -> dict[str, Any]:
    """Return the provider continuation message needed for the next call.

    DeepSeek thinking-mode tool calls require their opaque
    ``reasoning_content`` to be echoed in every later request.  Keep it only
    for those tool-call continuations; ordinary non-thinking messages retain
    the provider-neutral OpenAI-compatible shape.
    """
    output: dict[str, Any] = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
    }
    if calls:
        if preserve_reasoning:
            reasoning_content = getattr(message, "reasoning_content", None)
            if not isinstance(reasoning_content, str):
                raise ValueError(
                    "thinking-mode tool call must include reasoning_content"
                )
            output["reasoning_content"] = reasoning_content
        output["tool_calls"] = [
            {
                "id": str(getattr(call, "id", "")),
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": str(getattr(getattr(call, "function", None), "name", "")),
                    "arguments": getattr(getattr(call, "function", None), "arguments", ""),
                },
            }
            for call in calls
        ]
    return output


def _native_tool_call(call: Any) -> NativeToolCall:
    """Validate and normalize one provider-native function call."""
    identifier = getattr(call, "id", None)
    function = getattr(call, "function", None)
    name = getattr(function, "name", None)
    raw_arguments = getattr(function, "arguments", None)
    if (
        getattr(call, "type", None) != "function"
        or not isinstance(identifier, str)
        or not identifier
        or not isinstance(name, str)
        or not name
        or not isinstance(raw_arguments, str)
    ):
        raise ValueError("tool call must include id, function name, and JSON arguments")
    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be a JSON object")
    return NativeToolCall(id=identifier, name=name, arguments=arguments)


class DeepSeekClient(ModelClient):
    """ModelClient implementation for DeepSeek API (OpenAI-compatible protocol).

    Args:
        api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
        base_url: API base URL. Defaults to https://api.deepseek.com.
        model: Model identifier. Defaults to ``deepseek-v4-flash``.
        max_retries: SDK-level retries for transient HTTP errors.

    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        max_retries: int = 0,
        timeout_seconds: float = 90.0,
    ) -> None:
        import os

        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        resolved_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not resolved_key:
            raise AgentError(
                ErrorCode.INTERNAL_ERROR,
                "DeepSeek API key not set — set DEEPSEEK_API_KEY env var or pass api_key",
                recoverable=False,
            )
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._model = model
        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            max_retries=max_retries,
            timeout=timeout_seconds,
        )

    # ------------------------------------------------------------------
    # ModelClient interface
    # ------------------------------------------------------------------

    async def invoke(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Free-text completion. Returns normalized ModelResponse."""
        messages = self._build_messages(system, prompt)
        start = time.monotonic()

        if reasoning_effort is not None and reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort must be 'high', 'max', or None")
        request_options = dict(kwargs)
        if thinking is not None:
            raw_extra_body = request_options.pop("extra_body", None)
            extra_body = dict(raw_extra_body) if isinstance(raw_extra_body, dict) else {}
            extra_body["thinking"] = {"type": "enabled" if thinking else "disabled"}
            request_options["extra_body"] = extra_body
        if thinking:
            request_options["reasoning_effort"] = reasoning_effort or "high"
        else:
            # DeepSeek documents temperature as unsupported in thinking mode.
            request_options["temperature"] = temperature

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                **request_options,
            )
        except (OpenAITimeoutError, APIConnectionError) as e:
            raise AgentError(ErrorCode.MODEL_TRANSIENT, str(e)) from e
        except RateLimitError as e:
            raise AgentError(ErrorCode.MODEL_TRANSIENT, f"Rate limited: {e}") from e
        except AuthenticationError as e:
            raise AgentError(ErrorCode.INTERNAL_ERROR, f"Auth failed: {e}", recoverable=False) from e
        except APIError as e:
            raise AgentError(ErrorCode.INTERNAL_ERROR, f"API error: {e}", recoverable=False) from e

        latency = (time.monotonic() - start) * 1000
        choice = response.choices[0]
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        cached_tokens = self._usage_token(response.usage, "prompt_cache_hit_tokens")
        if not cached_tokens and response.usage:
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        uncached_tokens = self._usage_token(response.usage, "prompt_cache_miss_tokens")
        if not uncached_tokens:
            uncached_tokens = max(0, prompt_tokens - cached_tokens)

        return ModelResponse(
            content=choice.message.content or "",
            model=self._model,
            prompt_tokens=prompt_tokens,
            cached_input_tokens=cached_tokens,
            uncached_input_tokens=uncached_tokens,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=round(latency, 1),
            cost_usd=self._estimate_cost(
                uncached_tokens,
                cached_tokens,
                response.usage.completion_tokens if response.usage else 0,
            ),
        )

    async def invoke_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Request one JSON object and validate its required top-level fields.

        Retries belong to the Agent runtime because it owns the model-call
        budget. Keeping this gateway call single-shot prevents hidden billable
        requests and makes the execution trace truthful.

        Uses the provider's native JSON Output contract.  A malformed, fenced,
        or empty body fails closed; the budget-owning runtime may perform at
        most one separately recorded retry.
        """
        schema_prompt = (
            "Your task is below. Use the configured reasoning mode internally. "
            "Return only one valid JSON object conforming to this JSON schema:\n"
            f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "CRITICAL: Your response MUST contain a valid JSON object. "
            "Do not put chain-of-thought or prose before the object. A schema field "
            "named 'thinking' is a concise auditable rationale, not hidden reasoning.\n\n"
            f"Task context:\n{prompt}"
        )
        response = await self.invoke(
            schema_prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            response_format={"type": "json_object"},
            **kwargs,
        )

        content = response.content.strip()
        if not content:
            raise StructuredModelOutputError("model returned an empty JSON Output body", response)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            repaired = self._repair_structural_json(content)
            if repaired is None:
                raise StructuredModelOutputError(
                    f"model violated native JSON Output: {error}", response
                ) from error
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as repaired_error:
                raise StructuredModelOutputError(
                    f"model violated native JSON Output: {error}", response
                ) from repaired_error
            logger.warning(
                "Repaired mismatched JSON container delimiters from model=%s",
                response.model or getattr(self, "_model", "unknown"),
            )

        if not isinstance(parsed, dict):
            raise StructuredModelOutputError(
                f"model returned {type(parsed).__name__}, expected an object",
                response,
            )

        missing = self._missing_required_fields(parsed, schema)
        if missing:
            raise StructuredModelOutputError(
                f"model output is missing required fields: {missing}",
                response,
            )

        response.structured_output = parsed
        return response

    async def invoke_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        tool_choice: dict[str, Any] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> ToolModelResponse:
        """Invoke DeepSeek using its native OpenAI-compatible function-call protocol."""
        if reasoning_effort is not None and reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort must be 'high', 'max', or None")
        start = time.monotonic()
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
        }
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if thinking:
            # DeepSeek documents temperature as unsupported in thinking mode;
            # mirror invoke()'s handling and omit it rather than risk a 400.
            request["extra_body"] = {"thinking": {"type": "enabled"}}
            request["reasoning_effort"] = reasoning_effort or "high"
        else:
            request["temperature"] = temperature
            if thinking is not None:
                request["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            provider_response = await self._client.chat.completions.create(**request)
        except (OpenAITimeoutError, APIConnectionError) as error:
            raise AgentError(ErrorCode.MODEL_TRANSIENT, str(error)) from error
        except RateLimitError as error:
            raise AgentError(ErrorCode.MODEL_TRANSIENT, f"Rate limited: {error}") from error
        except AuthenticationError as error:
            raise AgentError(ErrorCode.INTERNAL_ERROR, f"Auth failed: {error}", recoverable=False) from error
        except APIError as error:
            if getattr(error, "status_code", None) == 503:
                # Provider is busy (ServiceUnavailableError/InternalServerError with
                # status 503) — transient, same class as 429, so let the graph retry.
                raise AgentError(ErrorCode.MODEL_TRANSIENT, f"API error: {error}") from error
            raise AgentError(ErrorCode.INTERNAL_ERROR, f"API error: {error}", recoverable=False) from error

        choices = list(getattr(provider_response, "choices", None) or [])
        provider_model = getattr(provider_response, "model", None) or None
        # F5: snapshot the ACTUAL request after every adapter default was
        # applied — the envelope must fingerprint what the provider received
        # (effective reasoning effort, temperature, thinking extra_body), not
        # the caller's pre-adapter options.
        # Task 3 (closeout): the snapshot's model is ALWAYS the alias actually
        # sent (the adapter never rewrites the model field). The provider's
        # echoed model is a RESPONSE fact and lives only on
        # ToolModelResponse.model — request-side records must not leak it.
        effective_request = EffectiveToolRequest(
            model=str(request["model"]),
            tool_schemas=tuple(tools),
            tool_choice=request.get("tool_choice"),
            response_format=request.get("response_format"),
            max_tokens=int(request["max_tokens"]),
            provider_options={
                key: request[key]
                for key in ("temperature", "extra_body", "reasoning_effort")
                if key in request
            },
        )
        if not choices:
            response = self._tool_response(
                content="",
                message={},
                usage=provider_response.usage,
                latency_ms=(time.monotonic() - start) * 1000,
                model=provider_model,
                effective_request=effective_request,
            )
            raise StructuredModelOutputError("model returned no choices", response)
        choice = choices[0]
        raw_message = choice.message
        raw_calls = list(getattr(raw_message, "tool_calls", None) or [])
        response = self._tool_response(
            content=getattr(raw_message, "content", None) or "",
            message={},
            usage=provider_response.usage,
            latency_ms=(time.monotonic() - start) * 1000,
            model=provider_model,
            effective_request=effective_request,
        )
        try:
            response.tool_calls = [_native_tool_call(call) for call in raw_calls]
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            # Keep the exact provider call on the billable error response.  A
            # runtime with a structured tool-error loop can then return bad
            # JSON/object arguments to the model without weakening the
            # provider-neutral adapter contract: this call still fails closed.
            identities_are_strict = all(
                getattr(call, "type", None) == "function"
                and isinstance(getattr(call, "id", None), str)
                and bool(getattr(call, "id", None))
                and isinstance(getattr(getattr(call, "function", None), "name", None), str)
                and bool(getattr(getattr(call, "function", None), "name", None))
                and isinstance(
                    getattr(getattr(call, "function", None), "arguments", None), str
                )
                for call in raw_calls
            )
            if identities_are_strict:
                with contextlib.suppress(ValueError):
                    response.message = _native_assistant_message(
                        raw_message,
                        raw_calls,
                        preserve_reasoning=bool(thinking and raw_calls),
                    )
            call_id = getattr(raw_calls[0], "id", "unknown") if raw_calls else "unknown"
            raise StructuredModelOutputError(
                f"invalid native tool arguments for {call_id}: {error}", response
            ) from error
        try:
            response.message = _native_assistant_message(
                raw_message,
                raw_calls,
                preserve_reasoning=bool(thinking and raw_calls),
            )
        except ValueError as error:
            raise StructuredModelOutputError(str(error), response) from error
        return response

    def estimate_tokens(self, text: str) -> int:
        """Rough token count for budget pre-checks.

        Uses 1.5 chars/token as a conservative approximation for mixed Chinese/JSON
        prompts.  The older 2 chars/token estimate under-reserved several real calls,
        even though the provider total still remained inside the scope ceiling.
        Production should use a proper tokenizer (tiktoken for OpenAI-compatible).
        Measured actuals run ~2.47 chars/token (docs/token-estimation-accounting.md
        §2); pre-call figures here are never authoritative — settle with the
        provider's usage object after the call.
        """
        return max(1, (len(text) * 2) // 3 + 1)

    def _tool_response(
        self,
        *,
        content: str,
        message: dict[str, Any],
        usage: object,
        latency_ms: float,
        model: str | None = None,
        effective_request: EffectiveToolRequest | None = None,
    ) -> ToolModelResponse:
        """Normalize provider usage while preserving native continuation data."""
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        cached_tokens = self._usage_token(usage, "prompt_cache_hit_tokens")
        if not cached_tokens:
            details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        uncached_tokens = self._usage_token(usage, "prompt_cache_miss_tokens") or max(
            0, prompt_tokens - cached_tokens
        )
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return ToolModelResponse(
            content=content,
            message=message,
            model=model or self._model,
            effective_request=effective_request,
            prompt_tokens=prompt_tokens,
            cached_input_tokens=cached_tokens,
            uncached_input_tokens=uncached_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 1),
            cost_usd=self._estimate_cost(uncached_tokens, cached_tokens, completion_tokens),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(system: str, prompt: str) -> list[ChatCompletionMessageParam]:
        """Build the messages list for the chat API."""
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _missing_required_fields(
        data: dict[str, Any],
        schema: dict[str, Any],
    ) -> list[str]:
        """Return missing top-level fields for a useful repair instruction."""
        required: list[str] = schema.get("required", [])
        return [field for field in required if field not in data]

    @staticmethod
    def _repair_structural_json(content: str) -> str | None:
        """Repair only unambiguous container-delimiter mistakes.

        Flash occasionally emits an object close where the surrounding array
        must close first, followed by a duplicate trailing close.  This pass
        never edits quoted content, keys, values, commas, or truncated JSON;
        it only inserts the closer implied by the open-container stack and
        removes unmatched trailing closers.  Semantic/schema validation still
        runs afterward.
        """
        output: list[str] = []
        stack: list[str] = []
        in_string = False
        escaped = False
        repairs = 0
        matching_open = {"}": "{", "]": "["}
        matching_close = {"{": "}", "[": "]"}
        for index, char in enumerate(content):
            if in_string:
                output.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                remainder = content[index + 1 :]
                if stack and not remainder.strip().strip("}]").strip():
                    # A quote cannot begin a JSON string when the physical
                    # tail contains only container closers. Flash sometimes
                    # emits exactly this orphan after a completed object.
                    repairs += 1
                    if repairs > 4:
                        return None
                    continue
                in_string = True
                output.append(char)
                continue
            if char in "{[":
                stack.append(char)
                output.append(char)
                continue
            if char not in "}]":
                output.append(char)
                continue
            expected = matching_open[char]
            while stack and stack[-1] != expected:
                repairs += 1
                if repairs > 4:
                    return None
                output.append(matching_close[stack.pop()])
            if stack and stack[-1] == expected:
                stack.pop()
                output.append(char)
                continue
            # Only discard a duplicate closer at the physical tail. Any
            # non-closer token afterward would make the repair ambiguous.
            remainder = content[index + 1 :]
            if remainder.strip().strip("}]").strip():
                return None
            repairs += 1
            if repairs > 4:
                return None
        if in_string or stack or repairs == 0:
            return None
        return "".join(output)

    def _estimate_cost(
        self,
        uncached_prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate USD cost using the configured V4 model's public rates.

        Rates follow DeepSeek's peak/off-peak pricing (effective 2026-08-17):
        this ledger-side estimate uses the OFF-PEAK tier (half of peak),
        because live batches are scheduled in off-peak windows by manifest.
        The LiveCostGuard safety reservation deliberately uses peak rates
        (an upper bound); settlement here uses the actual off-peak tier.
        FX assumption ~7.1 CNY/USD; rates are documented in
        docs/plans/2026-08-19-phase2-batch1-preparation.md §4.0.
        """
        if self._model == "deepseek-v4-flash":
            cache_rate, input_rate, output_rate = 0.007, 0.211, 0.634
        elif self._model == "deepseek-v4-pro":
            # v4-pro official post-2026-08-17 rates not yet verified; unchanged.
            cache_rate, input_rate, output_rate = 0.003625, 0.435, 0.87
        else:
            cache_rate, input_rate, output_rate = 0.07, 0.27, 1.10
        input_cost = (uncached_prompt_tokens / 1_000_000) * input_rate
        cache_cost = (cached_prompt_tokens / 1_000_000) * cache_rate
        output_cost = (completion_tokens / 1_000_000) * output_rate
        return round(input_cost + cache_cost + output_cost, 6)

    @staticmethod
    def _usage_token(usage: object, field: str) -> int:
        """Read provider-specific cache counters without coupling business code."""
        if usage is None:
            return 0
        value = getattr(usage, field, 0)
        if not value and hasattr(usage, "model_extra"):
            extra = getattr(usage, "model_extra", {}) or {}
            value = extra.get(field, 0)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
