"""Model gateway — thin abstraction over LLM providers.

Agents never import provider-specific clients directly.
They call ModelClient.invoke() / ModelClient.invoke_structured().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from leave_information_bubble.runtime.errors import AgentError, ErrorCode


@dataclass
class ModelResponse:
    """Normalized response from any LLM provider."""

    content: str
    structured_output: dict[str, Any] | None = None
    model: str = ""
    prompt_tokens: int = 0
    cached_input_tokens: int = 0
    uncached_input_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class NativeToolCall:
    """One normalized provider-native function call."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class EffectiveToolRequest:
    """Canonical snapshot of the effective provider tool-call request (F5).

    The provider adapter builds this from the ACTUAL request dict after all
    adapter defaults were applied (thinking extra_body, temperature,
    reasoning_effort), so the envelope fingerprints what the provider
    received — not what the caller asked for. Messages are excluded because
    transcript byte identity is audited separately; the ordered tool schemas
    and every effective non-message option are included.

    ``model`` is the alias actually sent in the request (a request-side
    fact; the adapter never rewrites it). The provider-echoed model is a
    response-side fact carried by ``ToolModelResponse.model`` instead.
    """

    model: str
    tool_schemas: tuple[dict[str, Any], ...]
    tool_choice: dict[str, Any] | str | None
    response_format: dict[str, Any] | None
    max_tokens: int | None
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolModelResponse(ModelResponse):
    """Model response retaining the exact assistant message for tool continuation."""

    message: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[NativeToolCall] = field(default_factory=list)
    effective_request: EffectiveToolRequest | None = None


class StructuredModelOutputError(AgentError):
    """Invalid structured content that still carries billable provider usage."""

    def __init__(self, message: str, response: ModelResponse) -> None:
        super().__init__(ErrorCode.MODEL_OUTPUT_INVALID, message)
        self.response = response


class ModelClient(ABC):
    """Provider-agnostic LLM interface.

    Business code depends on this, never on a specific SDK response type.
    """

    @abstractmethod
    async def invoke(self, prompt: str, *, system: str = "", **kwargs: Any) -> ModelResponse:
        """Free-text completion."""
        ...

    @abstractmethod
    async def invoke_structured(
        self, prompt: str, *, schema: dict[str, Any], system: str = "", **kwargs: Any
    ) -> ModelResponse:
        """Return one structured response; the budget-owning caller controls retries."""
        ...

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Rough token count for budget pre-checks."""
        ...
