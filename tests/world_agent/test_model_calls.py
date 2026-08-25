"""Per-call model invocation ledger: recorder rows and the graph write path."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from leave_information_bubble.gateway.client import (
    NativeToolCall,
    StructuredModelOutputError,
    ToolModelResponse,
)
from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world_agent.graph import (
    build_world_agent_graph,
    graph_shell_initial_state,
)
from leave_information_bubble.world_agent.model_calls import (
    ModelCallRecorder,
    ModelRequestEnvelope,
)
from tests.world_agent._graph_helpers import (
    _response as _snapshot_response,
)
from tests.world_agent._graph_helpers import (
    _ScriptedModel as _SnapshotScriptedModel,
)


@dataclass
class _ScriptedModel:
    """A deterministic native-tool model with configurable usage/cost fields."""

    responses: list[ToolModelResponse | Exception]
    cost_usd: float = 0.001
    latency_ms: float = 42.0

    async def invoke_tools(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]], **kwargs: Any
    ) -> ToolModelResponse:
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return ToolModelResponse(
            content=result.content,
            message=result.message,
            tool_calls=result.tool_calls,
            model="fake-model",
            prompt_tokens=10,
            cached_input_tokens=4,
            uncached_input_tokens=6,
            completion_tokens=7,
            cost_usd=self.cost_usd,
            latency_ms=self.latency_ms,
        )


def _response(content: str, *calls: NativeToolCall) -> ToolModelResponse:
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


def _rows(path: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return list(
            connection.execute(
                "SELECT thread_id, turn, purpose, model, prompt_tokens,"
                " cached_input_tokens, uncached_input_tokens, completion_tokens,"
                " cost_usd, latency_ms FROM model_calls ORDER BY id"
            )
        )
    finally:
        connection.close()


def _initial(
    store: WorldStore, content: str, thread_id: str, wake_id: str
) -> dict[str, Any]:
    """Build a checkpointed Graph Shell input carrying the protocol identity."""
    return graph_shell_initial_state(
        messages=[{"role": "user", "content": content}],
        store=store,
        thread_id=thread_id,
        wake_id=wake_id,
        domain_key="lol_cn",
        mode="broad",
        object_id=None,
    )


def _phase_rows(path: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return list(
            connection.execute(
                "SELECT wake_protocol, phase FROM model_calls ORDER BY id"
            )
        )
    finally:
        connection.close()


def test_recorder_writes_one_row_with_all_fields(tmp_path: Path) -> None:
    """append() persists every response field into one model_calls row."""
    recorder = ModelCallRecorder(tmp_path / "runtime.sqlite3")
    response = ToolModelResponse(
        content="ok",
        message={"role": "assistant", "content": "ok"},
        model="deepseek-chat",
        prompt_tokens=10,
        cached_input_tokens=4,
        uncached_input_tokens=6,
        completion_tokens=7,
        cost_usd=0.00123,
        latency_ms=55.5,
    )
    recorder.append(thread_id="round-1", turn=3, purpose="proposal", response=response)

    rows = _rows(tmp_path / "runtime.sqlite3")
    assert len(rows) == 1
    assert rows[0]["thread_id"] == "round-1"
    assert rows[0]["turn"] == 3
    assert rows[0]["purpose"] == "proposal"
    assert rows[0]["model"] == "deepseek-chat"
    assert rows[0]["prompt_tokens"] == 10
    assert rows[0]["cached_input_tokens"] == 4
    assert rows[0]["uncached_input_tokens"] == 6
    assert rows[0]["completion_tokens"] == 7
    assert rows[0]["cost_usd"] == 0.00123
    assert rows[0]["latency_ms"] == 55.5


def test_recorder_table_creation_is_idempotent(tmp_path: Path) -> None:
    """Two recorders over the same path never double-create the table."""
    path = tmp_path / "runtime.sqlite3"
    ModelCallRecorder(path)
    ModelCallRecorder(path)  # must not raise
    recorder = ModelCallRecorder(path)
    recorder.append(
        thread_id="round-1",
        turn=1,
        purpose="explore",
        response=ToolModelResponse(content="", message={"role": "assistant", "content": ""}),
    )
    assert len(_rows(path)) == 1


def test_recorder_migrates_pre_p5_table_for_digest_metadata(tmp_path: Path) -> None:
    """An existing runtime DB gains nullable digest columns in place."""
    path = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE model_calls ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,"
            "turn INTEGER NOT NULL, purpose TEXT NOT NULL,"
            "wake_protocol TEXT NOT NULL DEFAULT 'current',"
            "phase TEXT NOT NULL DEFAULT 'exploration', model TEXT,"
            "prompt_tokens INTEGER, cached_input_tokens INTEGER,"
            "uncached_input_tokens INTEGER, completion_tokens INTEGER,"
            "cost_usd REAL, latency_ms REAL)"
        )
        connection.commit()
    finally:
        connection.close()

    ModelCallRecorder(path)

    connection = sqlite3.connect(path)
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(model_calls)")}
    finally:
        connection.close()
    assert {
        "digest_target_ids_json",
        "digest_primary_observation_ids_json",
        "digest_contract_version",
    } <= columns


def test_recorder_migrates_digest_era_table_for_wake_id(tmp_path: Path) -> None:
    """A runtime DB created by this build before wake identity gains the
    nullable wake_id column additively, keeping old rows readable."""
    path = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE model_calls ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,"
            "turn INTEGER NOT NULL, purpose TEXT NOT NULL,"
            "wake_protocol TEXT NOT NULL DEFAULT 'current',"
            "phase TEXT NOT NULL DEFAULT 'exploration', model TEXT,"
            "request_model TEXT,"
            "prompt_tokens INTEGER, cached_input_tokens INTEGER,"
            "uncached_input_tokens INTEGER, completion_tokens INTEGER,"
            "cost_usd REAL, latency_ms REAL,"
            "digest_target_ids_json TEXT,"
            "digest_primary_observation_ids_json TEXT,"
            "digest_contract_version TEXT,"
            "request_fingerprint TEXT, request_source TEXT)"
        )
        connection.commit()
    finally:
        connection.close()

    ModelCallRecorder(path)

    connection = sqlite3.connect(path)
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(model_calls)")}
    finally:
        connection.close()
    assert "wake_id" in columns


def test_recorder_stamps_wake_id(tmp_path: Path) -> None:
    """append() persists the wake identity alongside the thread id."""
    path = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(path)
    recorder.append(
        thread_id="thread-x",
        wake_id="wake-a",
        turn=1,
        purpose="explore",
        response=ToolModelResponse(content="", message={"role": "assistant", "content": ""}),
    )

    connection = sqlite3.connect(path)
    try:
        row = connection.execute("SELECT thread_id, wake_id FROM model_calls").fetchone()
    finally:
        connection.close()
    assert tuple(row) == ("thread-x", "wake-a")


def test_recorder_records_protocol_and_phase(tmp_path: Path) -> None:
    """Runtime traces preserve the experiment boundary without changing purpose."""
    path = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(path)
    recorder.append(
        thread_id="round-1",
        turn=1,
        purpose="proposal",
        wake_protocol="separated",
        phase="consolidation",
        response=ToolModelResponse(content="", message={"role": "assistant", "content": ""}),
    )
    assert [tuple(row) for row in _phase_rows(path)] == [("separated", "consolidation")]


def test_recorder_keeps_material_targets_and_primary_observation_audit_ids(tmp_path: Path) -> None:
    """Digest ledger records opaque material targets plus durable primaries."""
    path = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(path)
    recorder.append(
        thread_id="round-1",
        turn=1,
        purpose="digest",
        digest_target_ids=["material-full", "material-full"],
        digest_primary_observation_ids=["obs-primary", "obs-primary"],
        response=ToolModelResponse(content="", message={"role": "assistant", "content": ""}),
    )

    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT digest_target_ids_json, digest_primary_observation_ids_json FROM model_calls"
        ).fetchone()
    finally:
        connection.close()
    assert json.loads(row[0]) == ["material-full"]
    assert json.loads(row[1]) == ["obs-primary"]


def test_recorder_validates_thread_and_turn(tmp_path: Path) -> None:
    """Empty thread ids and non-positive turns are rejected before any write."""
    recorder = ModelCallRecorder(tmp_path / "runtime.sqlite3")
    response = ToolModelResponse(content="", message={"role": "assistant", "content": ""})
    with pytest.raises(ValueError, match="thread_id"):
        recorder.append(thread_id="", turn=1, purpose="explore", response=response)
    with pytest.raises(ValueError, match="turn"):
        recorder.append(thread_id="round-1", turn=0, purpose="explore", response=response)
    assert _rows(tmp_path / "runtime.sqlite3") == []


async def test_graph_writes_one_row_per_successful_invocation(tmp_path: Path) -> None:
    """A two-turn Graph Shell run persists exactly two rows with turn and purpose."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        digested_ids=set(),
        thread_id="test-mc",
        wake_id="wake-mc",
    )
    model = _ScriptedModel(
        [
            _response(
                "Recall first.",
                NativeToolCall(id="call-1", name="memory_recent", arguments={}),
            ),
            _response("", NativeToolCall(id="finish", name="finalize_graph", arguments={})),
        ],
        cost_usd=0.0005,
        latency_ms=11.0,
    )
    runtime_db = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime_db)
    async with aiosqlite.connect(runtime_db, isolation_level=None) as connection:
        graph = build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            checkpointer=AsyncSqliteSaver(connection),
            model_calls=recorder,
            thread_id="test-mc",
            wake_id="wake-mc",
        )
        await graph.ainvoke(
            _initial(store, "Explore.", "test-mc", "wake-mc"),
            {"configurable": {"thread_id": "test-mc"}},
        )

    rows = _rows(runtime_db)
    assert [row["thread_id"] for row in rows] == ["test-mc", "test-mc"]
    assert [row["turn"] for row in rows] == [1, 2]
    assert [row["purpose"] for row in rows] == ["graph_shell", "graph_shell"]
    assert [row["cost_usd"] for row in rows] == [0.0005, 0.0005]
    assert [row["latency_ms"] for row in rows] == [11.0, 11.0]
    assert [row["cached_input_tokens"] for row in rows] == [4, 4]


async def test_graph_records_provider_effective_envelope_without_crash(
    tmp_path: Path,
) -> None:
    """F-1 live-path regression: snapshot-carrying responses record cleanly.

    The offline suite previously split the two envelope branches across
    files: graph-level recorder tests used a fake that drops
    ``effective_request`` (fallback branch), while snapshot-attaching fakes
    ran without a recorder (envelope construction skipped entirely). This
    test couples them exactly like the live DeepSeek path does — an
    ``EffectiveToolRequest`` snapshot with a recorder attached must append a
    row via the provider_effective branch without raising.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        digested_ids=set(),
        thread_id="test-mc-live",
        wake_id="wake-live",
    )
    model = _SnapshotScriptedModel(
        [
            _snapshot_response(
                "Recall first.",
                NativeToolCall(id="call-1", name="memory_recent", arguments={}),
            ),
            _snapshot_response(
                "", NativeToolCall(id="finish", name="finalize_graph", arguments={})
            ),
        ]
    )
    runtime_db = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime_db)
    async with aiosqlite.connect(runtime_db, isolation_level=None) as connection:
        graph = build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            checkpointer=AsyncSqliteSaver(connection),
            model_calls=recorder,
            thread_id="test-mc-live",
            wake_id="wake-live",
        )
        await graph.ainvoke(
            _initial(store, "Explore.", "test-mc-live", "wake-live"),
            {"configurable": {"thread_id": "test-mc-live"}},
        )

    connection = sqlite3.connect(runtime_db)
    try:
        rows = list(
            connection.execute(
                "SELECT request_source, request_model FROM model_calls ORDER BY id"
            ).fetchall()
        )
    finally:
        connection.close()
    assert [tuple(row) for row in rows] == [
        ("provider_effective", "actual-model"),
        ("provider_effective", "actual-model"),
    ]


async def test_graph_records_explicit_wake_id_into_model_calls(tmp_path: Path) -> None:
    """The graph stamps every recorded call with the effective wake identity."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        digested_ids=set(),
        thread_id="test-mc-wake",
        wake_id="wake-a",
    )
    model = _ScriptedModel(
        [
            _response(
                "Recall first.",
                NativeToolCall(id="call-1", name="memory_recent", arguments={}),
            ),
            _response("", NativeToolCall(id="finish", name="finalize_graph", arguments={})),
        ]
    )
    runtime_db = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime_db)
    async with aiosqlite.connect(runtime_db, isolation_level=None) as connection:
        graph = build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            checkpointer=AsyncSqliteSaver(connection),
            model_calls=recorder,
            thread_id="test-mc-wake",
            wake_id="wake-a",
        )
        await graph.ainvoke(
            _initial(store, "Explore.", "test-mc-wake", "wake-a"),
            {"configurable": {"thread_id": "test-mc-wake"}},
        )

    connection = sqlite3.connect(runtime_db)
    try:
        rows = list(
            connection.execute(
                "SELECT thread_id, wake_id FROM model_calls ORDER BY id"
            ).fetchall()
        )
    finally:
        connection.close()
    assert len(rows) == 2
    assert all(tuple(row) == ("test-mc-wake", "wake-a") for row in rows)


async def test_graph_records_recovered_output_error_exactly_once(tmp_path: Path) -> None:
    """A recoverable StructuredModelOutputError is recorded once, not retried.

    G5b-2: the legacy repair/retry loop is gone — an intact native call with
    broken JSON arguments is recovered in place, its response is recorded for
    that turn, and the run continues on the next model call.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        digested_ids=set(),
        thread_id="test-mc",
        wake_id="wake-mc",
    )
    malformed = ToolModelResponse(
        content="",
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "bad-json",
                    "type": "function",
                    "function": {"name": "graph_patch", "arguments": "{not-json"},
                }
            ],
        },
        tool_calls=[],
        prompt_tokens=11,
        completion_tokens=2,
        cost_usd=0.001,
    )
    model = _ScriptedModel(
        [
            StructuredModelOutputError(
                "invalid native tool arguments for bad-json", malformed
            ),
            _response(
                "retry with valid json",
                NativeToolCall(
                    id="fixed",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "op-1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Recovered concept",
                                    "kind": "concept",
                                    "provisional": True,
                                },
                            }
                        ]
                    },
                ),
            ),
            _response("finish", NativeToolCall(id="finish", name="finalize_graph", arguments={})),
        ]
    )
    runtime_db = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime_db)
    async with aiosqlite.connect(runtime_db, isolation_level=None) as connection:
        graph = build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            checkpointer=AsyncSqliteSaver(connection),
            model_calls=recorder,
            thread_id="test-mc",
            wake_id="wake-mc",
        )
        result = await graph.ainvoke(
            _initial(store, "Explore.", "test-mc", "wake-mc"),
            {"configurable": {"thread_id": "test-mc"}},
        )

    rows = _rows(runtime_db)
    assert len(rows) == 3  # recovered turn + repaired patch + finalize, no retry rows
    assert [row["turn"] for row in rows] == [1, 2, 3]
    assert [row["purpose"] for row in rows] == ["graph_shell", "graph_shell", "graph_shell"]
    assert result["terminal_status"] == "published"
    assert result["raw_exception"]["recoverable"] is True


async def test_graph_records_nothing_when_invocation_fails(tmp_path: Path) -> None:
    """An unrecoverable model failure leaves the ledger empty."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        digested_ids=set(),
        thread_id="test-mc",
        wake_id="wake-mc",
    )
    model = _ScriptedModel([RuntimeError("provider down")])
    runtime_db = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime_db)
    async with aiosqlite.connect(runtime_db, isolation_level=None) as connection:
        graph = build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            checkpointer=AsyncSqliteSaver(connection),
            model_calls=recorder,
            thread_id="test-mc",
            wake_id="wake-mc",
        )
        with pytest.raises(RuntimeError, match="provider down"):
            await graph.ainvoke(
                _initial(store, "Explore.", "test-mc", "wake-mc"),
                {"configurable": {"thread_id": "test-mc"}},
            )

    assert _rows(runtime_db) == []


def test_recorder_requires_thread_id_and_wake_id_at_graph_build(tmp_path: Path) -> None:
    """Missing Graph Shell identity is a build-time error, not silent loss."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(store=store, adapters={}, digested_ids=set())
    model = _ScriptedModel([])
    with pytest.raises(ValueError, match="thread_id is required"):
        build_world_agent_graph(model=model, tools=tools, store=store, wake_id="wake-mc")
    with pytest.raises(ValueError, match="wake_id is required"):
        build_world_agent_graph(model=model, tools=tools, store=store, thread_id="test-mc")
    with pytest.raises(ValueError, match="thread_id is required"):
        build_world_agent_graph(model=model, tools=tools, store=store)


def test_request_fingerprint_is_deterministic_and_tool_order_sensitive() -> None:
    """Fingerprints are byte-stable across re-creation and NEVER schema-sorted."""
    memory_schema = {
        "type": "function",
        "function": {
            "name": "memory_recent",
            "description": "检索最近的记忆，用于当前探索锚点",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    submit_schema = {
        "type": "function",
        "function": {
            "name": "submit_cognition",
            "description": "提交本轮认知增量",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    # Deliberately NOT alphabetical: submit_cognition precedes memory_recent,
    # so a schema-sorting implementation would hash them the other way round.
    envelope = ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=True,
        reasoning_effort="high",
        tool_schemas=[submit_schema, memory_schema],
        tool_choice=None,
        response_format=None,
        max_tokens=8192,
    )
    assert envelope.fingerprint() == envelope.fingerprint()
    assert envelope.fingerprint() == ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=True,
        reasoning_effort="high",
        tool_schemas=[submit_schema, memory_schema],
        tool_choice=None,
        response_format=None,
        max_tokens=8192,
    ).fingerprint()
    reordered = ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=True,
        reasoning_effort="high",
        tool_schemas=[memory_schema, submit_schema],
        tool_choice=None,
        response_format=None,
        max_tokens=8192,
    )
    assert reordered.fingerprint() != envelope.fingerprint()
    # The digest pins the exact compact UTF-8 serialization contract —
    # F5: the payload now includes source and provider_options.
    assert (
        envelope.fingerprint()
        == "449bfbe774089dce999c1326a443ff287789c81eb1cd04d5060615bbe18d5bfe"
    )


def test_full_fingerprint_changes_for_tool_choice_reasoning_or_max_tokens() -> None:
    """Every envelope field participates in the fingerprint, not only cache-relevant ones."""
    schema = {
        "type": "function",
        "function": {"name": "submit_cognition", "parameters": {"type": "object"}},
    }
    base = ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=False,
        reasoning_effort=None,
        tool_schemas=[schema],
        tool_choice=None,
        response_format=None,
        max_tokens=4096,
    )
    variants = [
        ModelRequestEnvelope(
            model="other-model",
            thinking=False,
            reasoning_effort=None,
            tool_schemas=[schema],
            tool_choice=None,
            response_format=None,
            max_tokens=4096,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=True,
            reasoning_effort=None,
            tool_schemas=[schema],
            tool_choice=None,
            response_format=None,
            max_tokens=4096,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=False,
            reasoning_effort="high",
            tool_schemas=[schema],
            tool_choice=None,
            response_format=None,
            max_tokens=4096,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=False,
            reasoning_effort=None,
            tool_schemas=[schema, schema],
            tool_choice=None,
            response_format=None,
            max_tokens=4096,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=False,
            reasoning_effort=None,
            tool_schemas=[schema],
            tool_choice={"type": "function", "function": {"name": "submit_cognition"}},
            response_format=None,
            max_tokens=4096,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=False,
            reasoning_effort=None,
            tool_schemas=[schema],
            tool_choice=None,
            response_format={"type": "json_object"},
            max_tokens=4096,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=False,
            reasoning_effort=None,
            tool_schemas=[schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
        ),
        # F5: the effective-request source and provider options participate
        # in the fingerprint too — a fallback envelope must never hash the
        # same as an adapter-verified one
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=False,
            reasoning_effort=None,
            tool_schemas=[schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
            source="caller_requested_fallback",
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=False,
            reasoning_effort=None,
            tool_schemas=[schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
            provider_options={"temperature": 0.4},
        ),
    ]
    assert len({variant.fingerprint() for variant in variants}) == len(variants)
    for variant in variants:
        assert variant.fingerprint() != base.fingerprint()


def test_stable_projection_ignores_only_declared_tool_choice_delta() -> None:
    """tool_choice is the only allowed phase delta; every other field projects."""
    memory_schema = {
        "type": "function",
        "function": {
            "name": "memory_recent",
            "description": "检索最近的记忆",
            "parameters": {"type": "object"},
        },
    }
    submit_schema = {
        "type": "function",
        "function": {
            "name": "submit_cognition",
            "description": "提交本轮认知增量",
            "parameters": {"type": "object"},
        },
    }
    base = ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=True,
        reasoning_effort="high",
        tool_schemas=[submit_schema, memory_schema],
        tool_choice=None,
        response_format=None,
        max_tokens=8192,
    )
    with_choice = ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=True,
        reasoning_effort="high",
        tool_schemas=[submit_schema, memory_schema],
        tool_choice={"type": "function", "function": {"name": "submit_cognition"}},
        response_format=None,
        max_tokens=8192,
    )
    assert base.stable_wake_projection() == with_choice.stable_wake_projection()
    projection = base.stable_wake_projection()
    assert set(projection) == {
        "model",
        "thinking",
        "reasoning_effort",
        "tool_schemas",
        "response_format",
        "max_tokens",
        "source",
        "provider_options",
    }
    assert projection["model"] == "deepseek-v4-flash"
    assert projection["thinking"] is True
    assert projection["reasoning_effort"] == "high"
    assert projection["response_format"] is None
    assert projection["max_tokens"] == 8192
    # tool schema order is preserved inside the projection, never sorted
    assert projection["tool_schemas"] == [submit_schema, memory_schema]
    for other in [
        ModelRequestEnvelope(
            model="other-model",
            thinking=True,
            reasoning_effort="high",
            tool_schemas=[submit_schema, memory_schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=False,
            reasoning_effort="high",
            tool_schemas=[submit_schema, memory_schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=True,
            reasoning_effort=None,
            tool_schemas=[submit_schema, memory_schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=True,
            reasoning_effort="high",
            tool_schemas=[memory_schema, submit_schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=True,
            reasoning_effort="high",
            tool_schemas=[submit_schema, memory_schema],
            tool_choice=None,
            response_format={"type": "json_object"},
            max_tokens=8192,
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=True,
            reasoning_effort="high",
            tool_schemas=[submit_schema, memory_schema],
            tool_choice=None,
            response_format=None,
            max_tokens=4096,
        ),
        # F5: effective-request source and provider options project too — a
        # fallback or a changed effective option must change the cacheable key
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=True,
            reasoning_effort="high",
            tool_schemas=[submit_schema, memory_schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
            source="caller_requested_fallback",
        ),
        ModelRequestEnvelope(
            model="deepseek-v4-flash",
            thinking=True,
            reasoning_effort="high",
            tool_schemas=[submit_schema, memory_schema],
            tool_choice=None,
            response_format=None,
            max_tokens=8192,
            provider_options={"extra_body": {"thinking": {"type": "enabled"}}},
        ),
    ]:
        assert other.stable_wake_projection() != projection


def test_model_call_recorder_adds_fingerprint_column_idempotently(tmp_path: Path) -> None:
    """A pre-fingerprint ledger gains the nullable column; appends store the digest."""
    path = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE model_calls ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,"
            "turn INTEGER NOT NULL, purpose TEXT NOT NULL,"
            "wake_protocol TEXT NOT NULL DEFAULT 'current',"
            "phase TEXT NOT NULL DEFAULT 'exploration', model TEXT,"
            "prompt_tokens INTEGER, cached_input_tokens INTEGER,"
            "uncached_input_tokens INTEGER, completion_tokens INTEGER,"
            "cost_usd REAL, latency_ms REAL)"
        )
        connection.commit()
    finally:
        connection.close()

    ModelCallRecorder(path)  # migrates in place
    ModelCallRecorder(path)  # idempotent

    envelope = ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=True,
        reasoning_effort="high",
        tool_schemas=[{"type": "function", "function": {"name": "memory_recent"}}],
        tool_choice=None,
        response_format=None,
        max_tokens=None,
    )
    recorder = ModelCallRecorder(path)
    recorder.append(
        thread_id="round-1",
        turn=1,
        purpose="explore",
        request_envelope=envelope,
        response=ToolModelResponse(content="", message={"role": "assistant", "content": ""}),
    )
    recorder.append(
        thread_id="round-1",
        turn=2,
        purpose="proposal",
        request_source="caller_requested_fallback",
        response=ToolModelResponse(content="", message={"role": "assistant", "content": ""}),
    )
    connection = sqlite3.connect(path)
    try:
        rows = list(
            connection.execute(
                "SELECT request_fingerprint, request_source FROM model_calls ORDER BY id"
            )
        )
    finally:
        connection.close()
    assert [row[0] for row in rows] == [envelope.fingerprint(), None]
    assert [row[1] for row in rows] == ["provider_effective", "caller_requested_fallback"]
