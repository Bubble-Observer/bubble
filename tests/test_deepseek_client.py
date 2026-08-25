"""Offline contract tests for the DeepSeek gateway."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import InternalServerError, RateLimitError

from leave_information_bubble.gateway.client import ModelResponse, StructuredModelOutputError
from leave_information_bubble.gateway.deepseek_client import DeepSeekClient
from leave_information_bubble.runtime.errors import AgentError, ErrorCode


def _client_with_content(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> tuple[DeepSeekClient, list[dict[str, Any]]]:
    client = object.__new__(DeepSeekClient)
    calls: list[dict[str, Any]] = []

    async def fake_invoke(prompt: str, **kwargs: Any) -> ModelResponse:
        calls.append({"prompt": prompt, **kwargs})
        return ModelResponse(content=content)

    monkeypatch.setattr(client, "invoke", fake_invoke)
    return client, calls


@pytest.mark.asyncio
async def test_structured_call_is_single_shot_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _client_with_content(monkeypatch, '{"title":"观察"}')
    response = await client.invoke_structured(
        "生成手记",
        schema={"type": "object", "required": ["title"]},
    )

    assert response.structured_output == {"title": "观察"}
    assert len(calls) == 1
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert '\n  "type"' not in calls[0]["prompt"]


@pytest.mark.asyncio
async def test_structured_call_does_not_guess_json_from_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _calls = _client_with_content(
        monkeypatch, '```json\n{"title":"观察"}\n```'
    )

    with pytest.raises(StructuredModelOutputError):
        await client.invoke_structured(
            "生成手记",
            schema={"type": "object", "required": ["title"]},
        )


@pytest.mark.asyncio
async def test_structured_call_repairs_only_container_delimiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _client_with_content(
        monkeypatch,
        '{"title":"观察","items":[{"id":"one"}}},',
    )

    with pytest.raises(StructuredModelOutputError):
        await client.invoke_structured(
            "生成手记",
            schema={"type": "object", "required": ["title", "items"]},
        )
    assert len(calls) == 1

    client, calls = _client_with_content(
        monkeypatch,
        '{"title":"观察","items":[{"id":"one"}"}]}',
    )
    response = await client.invoke_structured(
        "生成手记",
        schema={"type": "object", "required": ["title", "items"]},
    )
    assert response.structured_output == {
        "title": "观察",
        "items": [{"id": "one"}],
    }
    assert len(calls) == 1

    client, calls = _client_with_content(
        monkeypatch,
        '{"title":"观察","items":[{"id":"one"}}}}',
    )
    response = await client.invoke_structured(
        "生成手记",
        schema={"type": "object", "required": ["title", "items"]},
    )
    assert response.structured_output == {
        "title": "观察",
        "items": [{"id": "one"}],
    }
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_structured_call_does_not_complete_truncated_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _calls = _client_with_content(
        monkeypatch, '{"title":"观察","items":[{"id":"one"}'
    )

    with pytest.raises(StructuredModelOutputError):
        await client.invoke_structured(
            "生成手记",
            schema={"type": "object", "required": ["title", "items"]},
        )


@pytest.mark.asyncio
async def test_structured_call_reports_exact_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _client_with_content(monkeypatch, '{"title":"观察"}')

    with pytest.raises(StructuredModelOutputError) as error_info:
        await client.invoke_structured(
            "生成手记",
            schema={"type": "object", "required": ["title", "summary", "source_refs"]},
        )

    assert error_info.value.code is ErrorCode.MODEL_OUTPUT_INVALID
    assert "summary" in str(error_info.value)
    assert "source_refs" in str(error_info.value)
    assert error_info.value.response.content.startswith('{"title":')
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thinking", "expected_type", "has_temperature"),
    [(True, "enabled", False), (False, "disabled", True)],
)
async def test_v4_thinking_mode_is_explicit_at_provider_boundary(
    thinking: bool,
    expected_type: str,
    has_temperature: bool,
) -> None:
    captured: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
            usage=None,
        )

    client = object.__new__(DeepSeekClient)
    client._model = "deepseek-v4-flash"
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    await client.invoke(
        "test",
        thinking=thinking,
        reasoning_effort="high" if thinking else None,
        temperature=0.2,
    )

    assert captured[0]["extra_body"] == {
        "thinking": {"type": expected_type}
    }
    assert ("temperature" in captured[0]) is has_temperature
    if thinking:
        assert captured[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thinking", "expected_type", "has_temperature"),
    [(True, "enabled", False), (False, "disabled", True), (None, None, True)],
)
async def test_invoke_tools_thinking_mode_omits_temperature(
    thinking: bool | None,
    expected_type: str | None,
    has_temperature: bool,
) -> None:
    """Thinking-mode tool calls must not carry temperature (DeepSeek rejects it).

    invoke() already skips temperature under thinking; invoke_tools must mirror
    that so the exploration rounds that enable thinking stay provider-valid.
    """
    captured: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=None, tool_calls=[]))],
            usage=None,
        )

    client = object.__new__(DeepSeekClient)
    client._model = "deepseek-v4-flash"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    await client.invoke_tools(
        [{"role": "user", "content": "hi"}],
        tools=[],
        temperature=0.2,
        thinking=thinking,
    )

    if has_temperature:
        assert captured[0]["temperature"] == 0.2
    else:
        assert "temperature" not in captured[0]
    if expected_type is None:
        assert "extra_body" not in captured[0]
    else:
        assert captured[0]["extra_body"] == {"thinking": {"type": expected_type}}
    if thinking:
        # mirror invoke(): thinking requests default to "high" reasoning effort
        assert captured[0]["reasoning_effort"] == "high"
    else:
        assert "reasoning_effort" not in captured[0]


@pytest.mark.asyncio
async def test_invoke_tools_thinking_reasoning_effort_passthrough() -> None:
    """Explicit reasoning_effort reaches the provider; invalid values are rejected.

    invoke_tools mirrors invoke()'s contract: reasoning_effort is sent as a
    top-level request field only under thinking, and only "high"/"max" pass.
    """
    captured: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=None, tool_calls=[]))],
            usage=None,
        )

    client = object.__new__(DeepSeekClient)
    client._model = "deepseek-v4-flash"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    await client.invoke_tools(
        [{"role": "user", "content": "hi"}],
        tools=[],
        thinking=True,
        reasoning_effort="max",
    )
    assert captured[0]["reasoning_effort"] == "max"

    with pytest.raises(ValueError):
        await client.invoke_tools(
            [{"role": "user", "content": "hi"}],
            tools=[],
            thinking=True,
            reasoning_effort="low",
        )
    assert len(captured) == 1  # the invalid call never reached the provider


def test_chinese_token_estimate_is_conservative() -> None:
    client = object.__new__(DeepSeekClient)

    assert client.estimate_tokens("中文内容" * 10) == 27


def _client_raising(error: Exception) -> DeepSeekClient:
    """Build a DeepSeekClient whose provider call raises the given error."""
    async def create(**kwargs: Any) -> Any:
        raise error

    client = object.__new__(DeepSeekClient)
    client._model = "deepseek-v4-flash"
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client


def _provider_error(status_code: int, cls: Callable[..., Exception]) -> Exception:
    """Construct a realistic provider status error carrying an httpx response."""
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return cls("provider error", response=httpx.Response(status_code, request=request), body=None)


@pytest.mark.asyncio
async def test_invoke_tools_maps_provider_503_to_transient() -> None:
    error = _provider_error(503, InternalServerError)
    client = _client_raising(error)

    with pytest.raises(AgentError) as error_info:
        await client.invoke_tools([{"role": "user", "content": "hi"}], tools=[])

    assert error_info.value.code is ErrorCode.MODEL_TRANSIENT
    assert error_info.value.recoverable is True
    assert error_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_invoke_tools_maps_other_api_errors_to_internal_error() -> None:
    error = _provider_error(500, InternalServerError)
    client = _client_raising(error)

    with pytest.raises(AgentError) as error_info:
        await client.invoke_tools([{"role": "user", "content": "hi"}], tools=[])

    assert error_info.value.code is ErrorCode.INTERNAL_ERROR
    assert error_info.value.recoverable is False
    assert error_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_invoke_tools_maps_provider_429_to_transient() -> None:
    error = _provider_error(429, RateLimitError)
    client = _client_raising(error)

    with pytest.raises(AgentError) as error_info:
        await client.invoke_tools([{"role": "user", "content": "hi"}], tools=[])

    assert error_info.value.code is ErrorCode.MODEL_TRANSIENT
    assert error_info.value.recoverable is True


@pytest.mark.asyncio
async def test_invoke_tools_preserves_mid_stream_system_message() -> None:
    """A system message not in first position passes through to the provider.

    P1 risk artifact: DeepSeek's gateway must accept mid-conversation system
    messages. The OpenAI-compatible contract forwards the messages list
    verbatim, so neither the content nor the position of a mid-stream system
    message may be reordered, rewritten, or dropped by the client.
    """
    captured: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=[]),
                )
            ],
            usage=None,
        )

    client = object.__new__(DeepSeekClient)
    client._model = "deepseek-v4-flash"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    messages = [
        {"role": "user", "content": "first request"},
        {"role": "system", "content": "mid-stream instruction"},
        {"role": "user", "content": "follow-up"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    await client.invoke_tools(messages, tools=tools)

    forwarded = captured[0]["messages"]
    assert forwarded == messages  # verbatim forwarding, nothing reordered or dropped
    assert [message["role"] for message in forwarded] == ["user", "system", "user"]
    assert forwarded[1] == {"role": "system", "content": "mid-stream instruction"}
    assert captured[0]["tools"] == tools
