"""Per-call model invocation ledger in the runtime database.

The world-agent graph only carries ``total_cost_usd`` in its state; this
recorder persists one row per successful model invocation (thread, turn,
purpose, token split including cached/uncached input, cost, latency) into a
``model_calls`` table in the same SQLite file that holds the LangGraph
checkpoints. The verification-round report script aggregates these rows for
the per-round cost/cache breakdown; rounds recorded before this table existed
(b7-b10) fall back to the checkpoint ``total_cost_usd``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from leave_information_bubble.gateway.client import ToolModelResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    purpose TEXT NOT NULL,
    wake_protocol TEXT NOT NULL DEFAULT 'current',
    phase TEXT NOT NULL DEFAULT 'exploration',
    model TEXT,
    request_model TEXT,
    prompt_tokens INTEGER,
    cached_input_tokens INTEGER,
    uncached_input_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    latency_ms REAL,
    digest_target_ids_json TEXT,
    digest_primary_observation_ids_json TEXT,
    digest_contract_version TEXT,
    request_fingerprint TEXT,
    request_source TEXT
)
"""


@dataclass(frozen=True)
class ModelRequestEnvelope:
    """The complete provider request envelope one model call carried.

    The fingerprint identifies exactly which envelope produced a ledger
    row's cached/uncached token split. Tool schemas keep their passed-in
    order — order is part of the identity and they are never sorted.
    """

    model: str
    thinking: bool
    reasoning_effort: str | None
    tool_schemas: Sequence[Mapping[str, Any]]
    tool_choice: Mapping[str, Any] | str | None
    response_format: Mapping[str, Any] | None
    max_tokens: int | None
    source: str = "provider_effective"
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def _payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort,
            "tool_schemas": list(self.tool_schemas),
            "tool_choice": self.tool_choice,
            "response_format": self.response_format,
            "max_tokens": self.max_tokens,
            "source": self.source,
            "provider_options": dict(self.provider_options),
        }

    def fingerprint(self) -> str:
        """Return the compact UTF-8 SHA-256 digest covering every field."""
        encoded = json.dumps(
            self._payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def stable_wake_projection(self) -> Mapping[str, Any]:
        """Return every field except the one allowed phase delta, tool_choice."""
        payload = self._payload()
        payload.pop("tool_choice")
        return payload


class ModelCallRecorder:
    """Append one row per model invocation to the runtime database.

    The table is created idempotently at construction. Each append opens its
    own short-lived connection (the InquiryLeaseStore pattern) so writes never
    contend with the checkpoint saver's connection on the same file. Only
    successful invocations are recorded; failed calls never reach the recorder
    because the graph raises instead of returning a response.
    """

    def __init__(self, path: str | Path) -> None:
        """Build a recorder bound to one runtime database file.

        Args:
            path: The runtime SQLite database path (same file as the
                LangGraph checkpoints).

        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute(_SCHEMA)
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(model_calls)")
            }
            if "wake_id" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN wake_id TEXT"
                )
            if "wake_protocol" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN wake_protocol TEXT NOT NULL DEFAULT 'current'"
                )
            if "phase" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN phase TEXT NOT NULL DEFAULT 'exploration'"
                )
            if "digest_target_ids_json" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN digest_target_ids_json TEXT"
                )
            if "digest_contract_version" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN digest_contract_version TEXT"
                )
            if "digest_primary_observation_ids_json" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN digest_primary_observation_ids_json TEXT"
                )
            if "request_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN request_fingerprint TEXT"
                )
            if "request_source" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN request_source TEXT"
                )
            if "request_model" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN request_model TEXT"
                )
            connection.commit()
        finally:
            connection.close()

    def append(
        self,
        *,
        thread_id: str,
        wake_id: str | None = None,
        turn: int,
        purpose: str,
        wake_protocol: str = "current",
        phase: str = "exploration",
        digest_target_ids: list[str] | None = None,
        digest_primary_observation_ids: list[str] | None = None,
        digest_contract_version: str | None = None,
        request_envelope: ModelRequestEnvelope | None = None,
        request_source: str = "provider_effective",
        request_model: str | None = None,
        response: ToolModelResponse,
    ) -> int:
        """Persist one successful model invocation.

        Args:
            thread_id: The LangGraph thread the invocation belongs to.
            wake_id: The wake identity this invocation belongs to; when None
                the row records the thread id as a compat fallback for
                callers that predate wake identity.
            turn: The 1-based agent turn that issued the invocation.
            purpose: ``"explore"``, ``"proposal"``, or ``"digest"`` — the
                graph route the invocation served.
            wake_protocol: The explicit ``current`` or ``separated`` protocol.
            phase: The graph phase, such as ``exploration`` or
                ``consolidation``.
            digest_target_ids: Exact Digest material ids presented to an
                isolated forced call, or requested by the model in an ordinary
                call; None for non-digest calls.  A recorded target does not
                imply that the tool executed or the material was digested.
            digest_primary_observation_ids: Durable primary observation ids
                that resolve the recorded digest targets, retained for audit.
                An empty list can therefore identify an unresolved ordinary
                request without using an empty-string sentinel.
            digest_contract_version: Cache/prompt contract used by a digest
                call; None for non-digest calls.
            request_envelope: The full provider request envelope that produced
                this response; only its fingerprint is persisted.
            request_source: ``"provider_effective"`` when the envelope was
                snapshotted from the adapter's actual request, or
                ``"caller_requested_fallback"`` when the caller's pre-adapter
                options had to stand in (no effective snapshot available).
            request_model: The model alias actually sent in the request
                (request-side fact, e.g. the envelope's ``model``); the
                ``model`` column keeps the provider-echoed response model.
            response: The provider response carrying usage and cost fields.

        Returns:
            The inserted runtime-ledger row id.

        """
        if not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        if turn < 1:
            raise ValueError("turn must be positive")
        connection = self._connect()
        try:
            cursor = connection.execute(
                "INSERT INTO model_calls"
                " (thread_id, wake_id, turn, purpose, wake_protocol, phase, model, request_model,"
                "  prompt_tokens, cached_input_tokens, uncached_input_tokens,"
                "  completion_tokens, cost_usd, latency_ms, digest_target_ids_json,"
                " digest_primary_observation_ids_json, digest_contract_version,"
                " request_fingerprint, request_source)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    wake_id,
                    turn,
                    purpose,
                    wake_protocol,
                    phase,
                    response.model or "",
                    request_model,
                    response.prompt_tokens,
                    response.cached_input_tokens,
                    response.uncached_input_tokens,
                    response.completion_tokens,
                    response.cost_usd,
                    response.latency_ms,
                    (
                        json.dumps(list(dict.fromkeys(digest_target_ids)), separators=(",", ":"))
                        if digest_target_ids is not None
                        else None
                    ),
                    (
                        json.dumps(
                            list(dict.fromkeys(digest_primary_observation_ids)), separators=(",", ":")
                        )
                        if digest_primary_observation_ids is not None
                        else None
                    ),
                    digest_contract_version,
                    (
                        request_envelope.fingerprint()
                        if request_envelope is not None
                        else None
                    ),
                    request_source,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection
