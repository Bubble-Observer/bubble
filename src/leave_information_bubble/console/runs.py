"""In-process supervision primitives for local console runs.

The console intentionally owns only one active run.  The world/runtime SQLite
files are long-lived single-writer stores, so this small process-local guard is
preferable to a queue or a distributed scheduler for the personal console.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import re
import sqlite3
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

RunStatus = Literal["queued", "running", "succeeded", "failed"]
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Runner = Callable[["RunSpec"], Awaitable[Mapping[str, JsonValue] | None]]

_SENSITIVE_TEXT = re.compile(
    r"(?i)(api[_ -]?key|authorization|bearer|token|secret|password)\s*[:=]\s*[^\s,;]+"
)


class RunAlreadyActiveError(RuntimeError):
    """Raised when a second run is requested before the current one finishes."""


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Stable inputs for one console-launched world-agent wake."""

    thread_id: str
    runtime_db: Path
    config: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.thread_id.strip():
            raise ValueError("thread_id must be non-empty")


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    """Aggregated model-call ledger values safe to show in the console."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    latest_turn: int | None = None
    latest_phase: str | None = None
    latest_model: str | None = None
    latest_latency_ms: float | None = None
    latest_prompt_tokens: int | None = None
    latest_cached_input_tokens: int | None = None
    latest_completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ConsoleEvent:
    """A short, sanitized console activity event."""

    at: datetime
    run_id: str | None
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Console-visible lifecycle record for one run."""

    run_id: str
    thread_id: str
    config_snapshot: Mapping[str, JsonValue]
    runtime_db: Path
    status: RunStatus
    queued_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    result_summary: Mapping[str, JsonValue] | None = None


def read_runtime_metrics(runtime_db: str | Path, thread_id: str | None = None) -> RuntimeMetrics:
    """Read the optional ``model_calls`` ledger without changing its database.

    Missing databases and older runtime databases without the ledger are normal
    during first launch, and therefore return empty metrics rather than errors.
    """
    path = Path(runtime_db)
    if not path.is_file():
        return RuntimeMetrics()
    try:
        connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_calls'"
            ).fetchone()
            if table is None:
                return RuntimeMetrics()
            columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(model_calls)")}
            required = {"turn", "prompt_tokens", "completion_tokens", "cached_input_tokens", "cost_usd"}
            if not required.issubset(columns):
                return RuntimeMetrics()
            where, params = ("", ()) if thread_id is None else (" WHERE thread_id = ?", (thread_id,))
            totals = connection.execute(
                "SELECT COUNT(*) AS calls, "
                "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
                "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
                "COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens, "
                "COALESCE(SUM(cost_usd), 0) AS cost_usd "
                f"FROM model_calls{where}",
                params,
            ).fetchone()
            if totals is None or not int(totals["calls"]):
                return RuntimeMetrics()
            latest_columns = ["turn"] + [
                column
                for column in (
                    "phase",
                    "model",
                    "latency_ms",
                    "prompt_tokens",
                    "cached_input_tokens",
                    "completion_tokens",
                )
                if column in columns
            ]
            latest = connection.execute(
                f"SELECT {', '.join(latest_columns)} FROM model_calls{where} ORDER BY id DESC LIMIT 1",
                params,
            ).fetchone()
            return RuntimeMetrics(
                calls=int(totals["calls"]),
                prompt_tokens=int(totals["prompt_tokens"]),
                completion_tokens=int(totals["completion_tokens"]),
                cached_input_tokens=int(totals["cached_input_tokens"]),
                cost_usd=float(totals["cost_usd"]),
                latest_turn=int(latest["turn"]) if latest is not None else None,
                latest_phase=_optional_text(latest, "phase"),
                latest_model=_optional_text(latest, "model"),
                latest_latency_ms=_optional_float(latest, "latency_ms"),
                latest_prompt_tokens=_optional_int(latest, "prompt_tokens"),
                latest_cached_input_tokens=_optional_int(latest, "cached_input_tokens"),
                latest_completion_tokens=_optional_int(latest, "completion_tokens"),
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return RuntimeMetrics()


class RunManager:
    """Run one injected async runner at a time and retain bounded local history."""

    def __init__(self, runner: Runner, *, history_limit: int = 30, event_limit: int = 100) -> None:
        if history_limit < 1 or event_limit < 1:
            raise ValueError("history_limit and event_limit must be positive")
        self._runner = runner
        self._history: deque[RunRecord] = deque(maxlen=history_limit)
        self._events: deque[ConsoleEvent] = deque(maxlen=event_limit)
        self._active_run_id: str | None = None
        self._lock = asyncio.Lock()
        self._runner_accepts_publish = len(inspect.signature(runner).parameters) >= 2

    async def start(self, spec: RunSpec) -> RunRecord:
        """Queue a run and return its immutable initial record."""
        async with self._lock:
            if self._active_run_id is not None:
                raise RunAlreadyActiveError("A console run is already active.")
            now = _now()
            record = RunRecord(
                run_id=uuid.uuid4().hex,
                thread_id=spec.thread_id,
                config_snapshot=_safe_mapping(spec.config),
                runtime_db=Path(spec.runtime_db),
                status="queued",
                queued_at=now,
            )
            self._history.append(record)
            self._active_run_id = record.run_id
            self.publish("run.queued", f"Run {record.run_id[:8]} queued.", run_id=record.run_id)
            asyncio.create_task(self._execute(record.run_id, spec), name=f"console-run-{record.run_id}")
            return _copy_record(record)

    async def get(self, run_id: str) -> RunRecord | None:
        """Return one record, if still retained in local history."""
        async with self._lock:
            return _copy_record(record) if (record := self._find(run_id)) else None

    async def list_runs(self) -> list[RunRecord]:
        """Return newest-first bounded run history."""
        async with self._lock:
            return [_copy_record(record) for record in reversed(self._history)]

    async def metrics(self, run_id: str) -> RuntimeMetrics:
        """Read current ledger metrics for one retained run."""
        record = await self.get(run_id)
        if record is None:
            return RuntimeMetrics()
        return read_runtime_metrics(record.runtime_db, record.thread_id)

    async def restore(self, record: RunRecord) -> RunRecord:
        """Add a completed durable record reconstructed from local runtime state."""
        if record.status in {"queued", "running"}:
            raise ValueError("only terminal runs can be restored")
        async with self._lock:
            existing = self._find(record.run_id)
            if existing is not None:
                return _copy_record(existing)
            self._history.append(record)
            return _copy_record(record)

    def publish(self, kind: str, message: str, *, run_id: str | None = None) -> ConsoleEvent:
        """Append a bounded, credential-scrubbed UI event."""
        event = ConsoleEvent(
            at=_now(),
            run_id=run_id,
            kind=_clean_text(kind, limit=60),
            message=_clean_text(message, limit=400),
        )
        self._events.append(event)
        return event

    def list_events(self, *, run_id: str | None = None, limit: int = 50) -> list[ConsoleEvent]:
        """Return newest-first events from the ring buffer."""
        if limit < 1:
            return []
        events = reversed(self._events)
        if run_id is not None:
            return [event for event in events if event.run_id == run_id][:limit]
        return list(events)[:limit]

    async def _execute(self, run_id: str, spec: RunSpec) -> None:
        await self._replace(run_id, status="running", started_at=_now())
        self.publish("run.started", f"Run {run_id[:8]} started.", run_id=run_id)
        try:

            def publish(kind: str, message: str) -> ConsoleEvent:
                return self.publish(kind, message, run_id=run_id)

            result = (
                await self._runner(spec, publish)  # type: ignore[call-arg]
                if self._runner_accepts_publish
                else await self._runner(spec)
            )
        except Exception as exc:  # Runner boundaries must never expose exception contents to the UI.
            error = f"Runner failed ({type(exc).__name__})."
            await self._replace(run_id, status="failed", ended_at=_now(), error=error)
            self.publish("run.failed", error, run_id=run_id)
        else:
            await self._replace(
                run_id,
                status="succeeded",
                ended_at=_now(),
                result_summary=_safe_mapping(result or {}),
            )
            self.publish("run.succeeded", f"Run {run_id[:8]} completed.", run_id=run_id)
        finally:
            async with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

    async def _replace(
        self,
        run_id: str,
        *,
        status: RunStatus,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        error: str | None = None,
        result_summary: Mapping[str, JsonValue] | None = None,
    ) -> None:
        async with self._lock:
            for index, record in enumerate(self._history):
                if record.run_id == run_id:
                    self._history[index] = replace(
                        record,
                        status=status,
                        started_at=started_at if started_at is not None else record.started_at,
                        ended_at=ended_at if ended_at is not None else record.ended_at,
                        error=error,
                        result_summary=result_summary,
                    )
                    return

    def _find(self, run_id: str) -> RunRecord | None:
        return next((record for record in self._history if record.run_id == run_id), None)


def _now() -> datetime:
    return datetime.now(UTC)


def _optional_text(row: sqlite3.Row | None, key: str) -> str | None:
    if row is None or key not in row.keys() or row[key] is None:  # noqa: SIM118 - sqlite rows test values.
        return None
    return str(row[key]) or None


def _optional_float(row: sqlite3.Row | None, key: str) -> float | None:
    if row is None or key not in row.keys() or row[key] is None:  # noqa: SIM118 - sqlite rows test values.
        return None
    return float(row[key])


def _optional_int(row: sqlite3.Row | None, key: str) -> int | None:
    if row is None or key not in row.keys() or row[key] is None:  # noqa: SIM118 - sqlite rows test values.
        return None
    return int(row[key])


def _safe_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {str(key): _safe_value(item) for key, item in value.items()}


def _safe_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return _safe_mapping(value)
    if isinstance(value, list | tuple):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _clean_text(value, limit=2_000)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return _clean_text(str(value), limit=400)


def _copy_record(record: RunRecord) -> RunRecord:
    return replace(
        record,
        config_snapshot=copy.deepcopy(dict(record.config_snapshot)),
        result_summary=(copy.deepcopy(dict(record.result_summary)) if record.result_summary else None),
    )


def _clean_text(value: str, *, limit: int) -> str:
    cleaned = _SENSITIVE_TEXT.sub(r"\1=[redacted]", value).replace("\n", " ").strip()
    return cleaned[:limit]
