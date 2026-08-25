# Split from tests/world_agent/test_graph.py (audit 2026-08-18, baseline e50bce4).
# G5b-2: the legacy submit_cognition / proposal / repair / digest fixtures were
# retired with the legacy runtime; only the shell-neutral model fakes remain.

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import (
    dataclass,
    field,
)
from typing import Any

from leave_information_bubble.gateway.client import (
    EffectiveToolRequest,
    NativeToolCall,
    ToolModelResponse,
)


@dataclass
class _ScriptedModel:
    """A deterministic native-tool model that records the real graph transcript."""

    responses: list[ToolModelResponse | Exception]
    request_model: str = "actual-model"
    attach_snapshot: bool = True
    requests: list[list[dict[str, Any]]] = field(default_factory=list)
    tool_names: list[list[str]] = field(default_factory=list)
    tool_schemas: list[list[dict[str, Any]]] = field(default_factory=list)
    request_options: list[dict[str, Any]] = field(default_factory=list)

    async def invoke_tools(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]], **kwargs: Any
    ) -> ToolModelResponse:
        self.requests.append(messages)
        self.tool_names.append([item["function"]["name"] for item in tools])
        self.tool_schemas.append(tools)
        self.request_options.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        # F5: mirror the DeepSeek adapter — attach the effective request
        # snapshot (after adapter defaults), so the graph envelopes audit the
        # provider-effective request rather than the caller's raw options.
        # Task 3: the snapshot's model is the alias ACTUALLY SENT (a
        # request-side fact), independent of the response's echoed model —
        # a fake response with a different echoed model proves the two
        # values never bleed into each other. ``attach_snapshot=False``
        # mirrors an older adapter (no snapshot), exercising the
        # caller_requested_fallback envelope path.
        if self.attach_snapshot:
            result.effective_request = EffectiveToolRequest(
                model=self.request_model,
                tool_schemas=tuple(tools),
                tool_choice=kwargs.get("tool_choice"),
                response_format=kwargs.get("response_format"),
                max_tokens=kwargs.get("max_tokens"),
                provider_options=_adapter_provider_options(kwargs),
            )
        return result


def _response(content: str, *calls: NativeToolCall) -> ToolModelResponse:
    """Build a realistic normalized provider response for the scripted model."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ]
    return ToolModelResponse(content=content, message=message, tool_calls=list(calls))


def _adapter_provider_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror DeepSeekClient.invoke_tools' adapter defaults from raw options."""
    thinking = options.get("thinking")
    provider_options: dict[str, Any] = {}
    if thinking:
        provider_options["extra_body"] = {"thinking": {"type": "enabled"}}
        provider_options["reasoning_effort"] = options.get("reasoning_effort") or "high"
    else:
        provider_options["temperature"] = options.get("temperature", 0.2)
        if thinking is not None:
            provider_options["extra_body"] = {"thinking": {"type": "disabled"}}
    return provider_options
