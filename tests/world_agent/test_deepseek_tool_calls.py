"""Contract tests for the DeepSeek provider-native tool-call path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from leave_information_bubble.gateway.client import StructuredModelOutputError
from leave_information_bubble.gateway.deepseek_client import DeepSeekClient
from leave_information_bubble.runtime.errors import ErrorCode
from leave_information_bubble.world import submit_cognition_schema


def _client(response: Any) -> tuple[DeepSeekClient, list[dict[str, Any]]]:
    """Build a DeepSeek client whose complete provider response is scripted."""
    requests: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return response

    client = object.__new__(DeepSeekClient)
    client._model = "deepseek-v4-flash"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, requests


@pytest.mark.asyncio
async def test_deepseek_native_tool_call_preserves_id_name_arguments_and_usage() -> None:
    """Catch native tool calls whose provider identity, payload, or usage is discarded."""
    provider_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="I will inspect the available memory.",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_memory_7",
                            type="function",
                            function=SimpleNamespace(name="memory_search", arguments='{"query":"opera"}'),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=25,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        ),
    )
    client, requests = _client(provider_response)

    result = await client.invoke_tools(
        [{"role": "user", "content": "Explore historical links."}],
        tools=[{"type": "function", "function": {"name": "memory_search", "parameters": {}}}],
    )

    assert result.tool_calls[0].id == "call_memory_7"
    assert result.tool_calls[0].name == "memory_search"
    assert result.tool_calls[0].arguments == {"query": "opera"}
    assert result.message == {
        "role": "assistant",
        "content": "I will inspect the available memory.",
        "tool_calls": [
            {
                "id": "call_memory_7",
                "type": "function",
                "function": {"name": "memory_search", "arguments": '{"query":"opera"}'},
            }
        ],
    }
    assert result.prompt_tokens == 120
    assert result.cached_input_tokens == 20
    assert result.uncached_input_tokens == 100
    assert result.completion_tokens == 25
    assert requests[0]["tools"][0]["function"]["name"] == "memory_search"
    assert requests[0]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_thinking_tool_call_preserves_opaque_reasoning_for_later_requests() -> None:
    """DeepSeek requires tool-call reasoning state to be echoed on later calls."""
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        reasoning_content="opaque-provider-continuation",
                        tool_calls=[
                            SimpleNamespace(
                                id="thinking-call",
                                type="function",
                                function=SimpleNamespace(
                                    name="memory_recent",
                                    arguments="{}",
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="done",
                        reasoning_content="not-needed-without-tools",
                        tool_calls=[],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=2),
        ),
    ]
    requests: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return responses.pop(0)

    client = object.__new__(DeepSeekClient)
    client._model = "deepseek-v4-flash"
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "inspect memory"}]

    first = await client.invoke_tools(messages, tools=[], thinking=True)
    assert first.message["reasoning_content"] == "opaque-provider-continuation"
    messages.extend(
        [
            first.message,
            {"role": "tool", "tool_call_id": "thinking-call", "content": "{}"},
        ]
    )
    second = await client.invoke_tools(messages, tools=[], thinking=True)

    assert requests[1]["messages"][1]["reasoning_content"] == (
        "opaque-provider-continuation"
    )
    assert "reasoning_content" not in second.message


@pytest.mark.asyncio
async def test_thinking_tool_call_without_reasoning_fails_before_continuation() -> None:
    """Do not accept a continuation that DeepSeek will reject on the next turn."""
    client, _requests = _client(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="missing-reasoning",
                                type="function",
                                function=SimpleNamespace(
                                    name="memory_recent",
                                    arguments="{}",
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )
    )

    with pytest.raises(StructuredModelOutputError, match="reasoning_content"):
        await client.invoke_tools([], tools=[], thinking=True)


@pytest.mark.asyncio
async def test_deepseek_terminal_tool_call_disables_thinking_and_names_the_tool() -> None:
    """Catch proposal repair relying on auto tool choice in V4 thinking mode.

    The fixture carries the real ``submit_cognition`` schema so the native
    path is proven against the exact provider-facing contract, not a stub.
    """
    schema = submit_cognition_schema()
    provider_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="terminal-proposal",
                            type="function",
                            function=SimpleNamespace(
                                name="submit_cognition",
                                arguments='{"schema_version":"1"}',
                            ),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4),
    )
    client, requests = _client(provider_response)
    choice = {"type": "function", "function": {"name": "submit_cognition"}}

    result = await client.invoke_tools(
        [{"role": "system", "content": "Submit the final cognition proposal."}],
        tools=[schema],
        tool_choice=choice,
        thinking=False,
    )

    assert result.tool_calls[0].name == "submit_cognition"
    assert requests[0]["tools"][0] == schema
    assert requests[0]["tool_choice"] == choice
    assert requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def _is_json_native(value: object) -> bool:
    """Return whether a value is made only of JSON-native types with string keys."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_json_native(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) for key in value) and all(
            _is_json_native(item) for item in value.values()
        )
    return False


def _assert_refs_resolve(node: object, definitions: dict[str, Any]) -> None:
    """Fail on any $ref whose target is absent from the schema's $defs."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            assert ref.startswith("#/$defs/"), ref
            assert ref.removeprefix("#/$defs/") in definitions, f"dangling $ref {ref}"
        for value in node.values():
            _assert_refs_resolve(value, definitions)
    elif isinstance(node, list):
        for value in node:
            _assert_refs_resolve(value, definitions)


def _assert_enum_members_are_strings(node: object) -> None:
    """Fail on non-string enum members some providers reject."""
    if isinstance(node, dict):
        enum = node.get("enum")
        if isinstance(enum, list):
            assert all(isinstance(member, str) for member in enum), "non-string enum member"
        for value in node.values():
            _assert_enum_members_are_strings(value)
    elif isinstance(node, list):
        for value in node:
            _assert_enum_members_are_strings(value)


def test_submit_cognition_schema_serializes_without_provider_unsupported_combinations() -> None:
    """The full submit_cognition tool schema is provider-serializable JSON.

    OpenAI-compatible/DeepSeek tool schemas pass through as opaque JSON, so
    the failure modes are non-JSON values, dangling $refs, and non-string
    enum members. Each is checked so the model-visible schema cannot drift
    into a shape no provider accepts.
    """
    schema = submit_cognition_schema()
    parameters = schema["function"]["parameters"]

    assert json.loads(json.dumps(schema, ensure_ascii=False)) == schema
    assert _is_json_native(schema)
    assert isinstance(parameters["$defs"], dict)
    _assert_refs_resolve(schema, parameters["$defs"])
    _assert_enum_members_are_strings(schema)


@pytest.mark.asyncio
async def test_deepseek_native_tool_call_rejects_invalid_argument_json() -> None:
    """Catch malformed provider arguments being silently converted to an empty object."""
    client, _requests = _client(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_bad_json",
                                type="function",
                                function=SimpleNamespace(name="memory_recent", arguments="{not-json"),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )
    )

    with pytest.raises(StructuredModelOutputError) as raised:
        await client.invoke_tools([], tools=[])

    assert raised.value.response.prompt_tokens == 10
    assert raised.value.response.message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bad_json",
                "type": "function",
                "function": {"name": "memory_recent", "arguments": "{not-json"},
            }
        ],
    }
    assert "call_bad_json" in str(raised.value)


@pytest.mark.asyncio
async def test_deepseek_does_not_preserve_malformed_args_when_raw_identity_is_invalid() -> None:
    """An invalid provider identity must not be string-coerced into a recoverable call."""
    client, _requests = _client(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id=7,
                                type="function",
                                function=SimpleNamespace(name="graph_patch", arguments="{not-json"),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )
    )

    with pytest.raises(StructuredModelOutputError) as raised:
        await client.invoke_tools([], tools=[])

    assert raised.value.response.message == {}


@pytest.mark.asyncio
async def test_deepseek_native_final_message_has_no_tool_calls() -> None:
    """Catch final assistant messages being treated as calls merely because tools were offered."""
    client, _requests = _client(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant", content="The evidence is inconclusive.", tool_calls=None
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
    )

    result = await client.invoke_tools([], tools=[])

    assert result.content == "The evidence is inconclusive."
    assert result.tool_calls == []
    assert result.message == {"role": "assistant", "content": "The evidence is inconclusive."}


@pytest.mark.asyncio
async def test_deepseek_native_tool_call_rejects_empty_choices_with_billable_usage() -> None:
    """Catch an empty successful provider response escaping as IndexError and losing usage."""
    client, _requests = _client(
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=200,
                completion_tokens=10,
                prompt_tokens_details=SimpleNamespace(cached_tokens=50),
            ),
        )
    )

    with pytest.raises(StructuredModelOutputError) as raised:
        await client.invoke_tools([], tools=[])

    response = raised.value.response
    assert raised.value.code is ErrorCode.MODEL_OUTPUT_INVALID
    assert response.model == "deepseek-v4-flash"
    assert response.prompt_tokens == 200
    assert response.cached_input_tokens == 50
    assert response.uncached_input_tokens == 150
    assert response.completion_tokens == 10
    # off-peak flash rates (2026-08-17): 150*0.211 + 50*0.007 + 10*0.634 over 1e6
    assert response.cost_usd == 0.000038
    assert response.latency_ms >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("identifier", "name", "call_type"),
    [
        (None, "memory_recent", "function"),
        ("", "memory_recent", "function"),
        (7, "memory_recent", "function"),
        ("call-name-none", None, "function"),
        ("call-name-empty", "", "function"),
        ("call-name-number", 7, "function"),
        ("call-type", "memory_recent", "other"),
    ],
)
async def test_deepseek_native_tool_call_rejects_invalid_call_identity(
    identifier: object, name: object, call_type: object
) -> None:
    """Catch null, blank, non-string, or non-function provider call identities being accepted."""
    client, _requests = _client(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id=identifier,
                                type=call_type,
                                function=SimpleNamespace(name=name, arguments="{}"),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=4),
        )
    )

    with pytest.raises(StructuredModelOutputError) as raised:
        await client.invoke_tools([], tools=[])

    assert raised.value.code is ErrorCode.MODEL_OUTPUT_INVALID
    assert raised.value.response.prompt_tokens == 30


@pytest.mark.asyncio
async def test_effective_request_snapshot_thinking_enabled_defaults() -> None:
    """The envelope must fingerprint the adapter-applied thinking defaults."""
    provider_response = SimpleNamespace(
        model="provider-resolved-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant", content=None, tool_calls=[]
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )
    client, requests = _client(provider_response)

    result = await client.invoke_tools(
        [{"role": "user", "content": "think"}],
        tools=[{"type": "function", "function": {"name": "memory_search", "parameters": {}}}],
        thinking=True,
        reasoning_effort="max",
    )

    assert result.model == "provider-resolved-model"
    assert result.effective_request is not None
    # Task 3: the request snapshot's model is the alias actually sent, never
    # the provider-echoed response model — the two values stay independent
    assert result.effective_request.model == "deepseek-v4-flash"
    assert result.effective_request.provider_options == {
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "max",
    }
    assert result.effective_request.tool_schemas[0]["function"]["name"] == "memory_search"
    assert result.effective_request.max_tokens == 4096
    assert requests[0]["reasoning_effort"] == "max"


@pytest.mark.asyncio
async def test_effective_request_snapshot_thinking_enabled_high_fallback() -> None:
    """reasoning_effort defaults to high in thinking mode, exactly as sent."""
    provider_response = SimpleNamespace(
        model="provider-resolved-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant", content=None, tool_calls=[]
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )
    client, requests = _client(provider_response)

    result = await client.invoke_tools(
        [{"role": "user", "content": "think"}],
        tools=[{"type": "function", "function": {"name": "memory_search", "parameters": {}}}],
        thinking=True,
    )

    assert result.effective_request is not None
    assert result.effective_request.provider_options == {
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "high",
    }
    assert requests[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_effective_request_snapshot_thinking_disabled_defaults() -> None:
    """Non-thinking requests snapshot temperature and the explicit disabled extra_body."""
    provider_response = SimpleNamespace(
        model="provider-resolved-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant", content=None, tool_calls=[]
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )
    client, requests = _client(provider_response)

    result = await client.invoke_tools(
        [{"role": "user", "content": "no-think"}],
        tools=[{"type": "function", "function": {"name": "memory_search", "parameters": {}}}],
        thinking=False,
        temperature=0.4,
        tool_choice={"type": "function", "function": {"name": "submit_cognition"}},
    )

    assert result.effective_request is not None
    assert result.effective_request.provider_options == {
        "extra_body": {"thinking": {"type": "disabled"}},
        "temperature": 0.4,
    }
    assert result.effective_request.tool_choice == {
        "type": "function",
        "function": {"name": "submit_cognition"},
    }
    assert requests[0]["temperature"] == 0.4
    assert requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_effective_request_snapshot_falls_back_to_configured_model() -> None:
    """A provider response without a model keeps the configured model."""
    provider_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=None, tool_calls=[])
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )
    client, _requests = _client(provider_response)

    result = await client.invoke_tools([], tools=[])

    assert result.model == "deepseek-v4-flash"
    assert result.effective_request is not None
    assert result.effective_request.model == "deepseek-v4-flash"
