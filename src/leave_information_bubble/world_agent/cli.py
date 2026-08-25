"""Run the small world agent against live adapters or deterministic replay fixtures."""

# ruff: noqa: D103, T201

import argparse
import asyncio
import json
import math
import sqlite3
import sys
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from leave_information_bubble import channels, tools
from leave_information_bubble.config import Settings, get_settings
from leave_information_bubble.gateway.client import NativeToolCall, ToolModelResponse
from leave_information_bubble.gateway.deepseek_client import DeepSeekClient
from leave_information_bubble.runtime.inquiry_lease import InquiryLeaseStore
from leave_information_bubble.tools.transcription import FasterWhisperTranscriber
from leave_information_bubble.tools.web_search import PublicWebSearchTool
from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world.domain_config import DomainFocus, resolve_domain_focus
from leave_information_bubble.world.finalize import FinalizeReceipt, finalize_graph
from leave_information_bubble.world.writer_lease import (
    STAGED_TABLES,
    abandon_writer_lease,
    read_writer_lease,
    release_writer_lease,
    restore_writer_lease,
    wake_has_active_staging,
    writer_lease_action,
)
from leave_information_bubble.world_agent.graph import (
    DEFAULT_CONTEXT_HARD_CUT_TOKENS,
    DEFAULT_CONTEXT_WARNING_TOKENS,
    WakeReasoningProfile,
    build_world_agent_graph,
    graph_shell_initial_state,
)
from leave_information_bubble.world_agent.live_cost_guard import (
    FROZEN_G1_MODEL,
    LiveCostGuard,
    validate_fresh_audit_path,
)
from leave_information_bubble.world_agent.live_deadline import LiveDeadlineGuard
from leave_information_bubble.world_agent.model_calls import ModelCallRecorder
from leave_information_bubble.world_agent.prompt import render_wake_input
from leave_information_bubble.world_agent.run_report import build_run_report, render_run_report

_GRAPH_SHELL_WORLD_LOCKS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = WeakKeyDictionary()


@asynccontextmanager
async def _graph_shell_world_run_guard(
    world_store_identity: str, *, enabled: bool
) -> AsyncIterator[None]:
    """Serialize Graph Shell runs that target one world within an event loop."""
    if not enabled:
        yield
        return
    loop = asyncio.get_running_loop()
    locks = _GRAPH_SHELL_WORLD_LOCKS.setdefault(loop, {})
    lock = locks.setdefault(world_store_identity.casefold(), asyncio.Lock())
    async with lock:
        yield


class _ScriptedModel:
    def __init__(self, path: Path) -> None:
        self._turns = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(self._turns, list):
            raise ValueError("malformed scripted model fixture: expected a turn list")

    async def invoke_tools(self, _messages: list[dict[str, Any]], **_kwargs: object) -> ToolModelResponse:
        if not self._turns:
            raise RuntimeError("scripted model fixture exhausted before graph completion")
        try:
            turn = self._turns.pop(0)
            content, raw_calls = turn.get("content", ""), turn.get("tool_calls", [])
            cost_usd = turn.get("cost_usd", 0.0)
            if (
                not isinstance(content, str)
                or not isinstance(raw_calls, list)
                or not isinstance(cost_usd, (int, float))
            ):
                raise TypeError
            cost_usd = float(cost_usd)
            calls = [NativeToolCall(**item) for item in raw_calls]
            if any(
                not isinstance(call.id, str)
                or not isinstance(call.name, str)
                or not isinstance(call.arguments, dict)
                for call in calls
            ):
                raise TypeError
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("malformed scripted model fixture turn") from error
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if raw_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in calls
            ]
        return ToolModelResponse(
            content=message["content"],
            message=message,
            tool_calls=calls,
            cost_usd=cost_usd,
        )


class WorldWriteInProgressError(ValueError):
    """A different Graph Shell wake owns this world's single active writer.

    Raised at the composition root when a fresh wake or a foreign resume
    races the world-scoped writer lease, so a second process never enters
    the model or patches the shared world. The message carries the stable
    code "world_write_in_progress" for typed failure matching.
    """


_EPILOG = """\
required: --perspective/--mission, --thread-id, --world-db, --runtime-db
examples:
  fresh live wake (safety defaults, nothing else to learn):
    --perspective m --thread-id my-thread --world-db data/demo-world.sqlite3
    --runtime-db data/demo-runtime.sqlite3
  resumed wake after an interrupted one (deadline still applies):
    --resume --live-deadline-seconds 1800
  offline replay (no API key, deterministic):
    --replay-fixture fixture.json --scripted-model-fixture model.json
all live-safety flags are optional; default guardrails: 96 turns,
700k context warn / 800k hard cut, no cost cap (set --max-cost-usd to stop
exploring once cumulative spend reaches a USD amount)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=_EPILOG)
    perspective = parser.add_mutually_exclusive_group()
    perspective.add_argument(
        "--perspective",
        help=(
            "Optional attention guidance for this wake; effective within system, tool, "
            "evidence, and persistence boundaries."
        ),
    )
    perspective.add_argument(
        "--mission",
        help="Deprecated compatibility alias for --perspective.",
    )
    parser.add_argument(
        "--world-db",
        type=Path,
        required=True,
        help="Explicit durable world SQLite path; use an isolated copy for tests and experiments.",
    )
    parser.add_argument(
        "--runtime-db",
        type=Path,
        required=True,
        help="Explicit runtime/checkpoint SQLite path, different from --world-db.",
    )
    parser.add_argument("--thread-id", required=True)
    parser.add_argument(
        "--graph-shell",
        action="store_true",
        help=(
            "Deprecated no-op since G5b-2: the working-graph (Graph Shell) "
            "runtime is the default and only normal runtime. Accepted for old "
            "run snapshots and explicit callers; behavior is identical with "
            "and without it."
        ),
    )
    parser.add_argument(
        "--graph-shell-status",
        action="store_true",
        help=(
            "Read-only management entry: report the world's active Graph Shell "
            "writer lease (owner wake/thread/claimed_at, checkpoint existence, "
            "active staging, finalize receipt) and the deterministic recovery "
            "action. Runs no model and mutates nothing."
        ),
    )
    parser.add_argument(
        "--graph-shell-abandon",
        metavar="WAKE_ID",
        help=(
            "Explicit management entry: mark the named wake's active staged "
            "working graph abandoned and release the singleton writer lease in "
            "one world transaction, with idempotent runtime claim cleanup. "
            "Refuses an owner mismatch and runs no model; the wake id must be "
            "typed verbatim (see --graph-shell-status for the current owner)."
        ),
    )
    parser.add_argument(
        "--graph-shell-restore",
        metavar="WAKE_ID",
        help=(
            "Explicit management entry (reverse of --graph-shell-abandon): "
            "re-activate the named wake's abandoned staged working graph and "
            "re-acquire the singleton writer lease for the same wake/thread in "
            "one world transaction, so the wake can be resumed and published. "
            "Runs no model; refuses an owner mismatch, a published wake, a "
            "missing wake claim or checkpoint, and abandoned rows without a "
            "successful patch receipt (all fail closed)."
        ),
    )
    parser.add_argument(
        "--graph-shell-finalize-wake",
        metavar="WAKE_ID",
        help=(
            "Explicit deterministic management entry: publish the named wake's "
            "active staged working graph through the same finalize_graph the "
            "agent uses (I1 one-publish-per-wake, idempotent replay, "
            "wake_closed for items staged after publishing). Runs no model and "
            "needs no writer lease; fail-closed on a missing wake claim or "
            "empty active staging, and the receipt carries blocked / "
            "compile_failed / commit_rejected reasons for operator reading."
        ),
    )
    parser.add_argument(
        "--wake-id",
        help=(
            "Explicit wake identity. Fresh wakes mint a collision-resistant "
            "identity by default; on --resume the checkpoint's wake identity "
            "is restored and must match an explicitly passed value (fail closed)."
        ),
    )
    parser.add_argument("--mode", choices=("broad", "deep"), default="broad")
    parser.add_argument(
        "--wake-protocol",
        choices=("current", "separated"),
        default="current",
        help=(
            "Retired protocol selector (G5b-2): accepted as a no-op so old "
            "run snapshots replay; Graph Shell is the only wake protocol."
        ),
    )
    parser.add_argument(
        "--memory-navigation",
        choices=("legacy", "overview_v1"),
        default="overview_v1",
        help=(
            "Retired navigation selector (G5b-2): accepted as a no-op so old "
            "run snapshots replay; memory navigation is fixed."
        ),
    )
    parser.add_argument(
        "--digest-cache-reuse",
        action="store_true",
        help=(
            "Retired experiment switch (G5b-2): accepted as a no-op so old "
            "run snapshots replay."
        ),
    )
    parser.add_argument("--object-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-turns", type=int, default=96)
    parser.add_argument(
        "--context-warning-tokens",
        type=int,
        default=DEFAULT_CONTEXT_WARNING_TOKENS,
        help=(
            "Absolute pre-call context threshold that injects the wind-down "
            "notice once (default %(default)s; below the provider window "
            "so the run winds down before a bare provider overflow)."
        ),
    )
    parser.add_argument(
        "--context-hard-cut-tokens",
        type=int,
        default=DEFAULT_CONTEXT_HARD_CUT_TOKENS,
        help=(
            "Absolute pre-call context limit that hard-stops the wake as "
            "staged_unpublished (default %(default)s; must be >= "
            "--context-warning-tokens)."
        ),
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Stop exploring once cumulative model cost reaches this USD amount",
    )
    parser.add_argument("--adapters", default="bilibili,nga,hupu,public-web")
    parser.add_argument(
        "--adapter-defaults",
        type=json.loads,
        default=None,
        help=(
            "Per-adapter domain surface parameters as JSON, e.g. "
            '{"hupu": {"board": "gaming"}, "nga": {"fid": "-152678"}}. '
            "Adapters without a configured default degrade to typed "
            "no-default-board limitations instead of scanning the wrong board."
        ),
    )
    parser.add_argument(
        "--domain",
        type=resolve_domain_focus,
        default=resolve_domain_focus("lol_cn"),
        help="registered domain focus and subtitle-ledger key (currently: lol_cn)",
    )
    parser.add_argument("--replay-fixture", type=Path)
    parser.add_argument("--scripted-model-fixture", type=Path)
    parser.add_argument(
        "--live-hard-cap-usd",
        type=Decimal,
        help=(
            "G1-only pre-call projection cost cap for the frozen live model; must pair "
            "with --live-cost-audit-path; not available with --resume or --replay-fixture "
            "(use --max-cost-usd for cumulative spend); default disabled."
        ),
    )
    parser.add_argument(
        "--live-cost-audit-path",
        type=Path,
        help="Append-only JSONL audit path; required together with --live-hard-cap-usd.",
    )
    parser.add_argument(
        "--live-deadline-seconds",
        type=float,
        help=(
            "Opt-in wall-clock safety net for a live wake (fresh or --resume): blocks new "
            "model dispatch, lets an already-dispatched model call settle, blocks new "
            "acquisition dispatch, and lets an already-dispatched acquisition settle under "
            "its own timeout; independent of cost caps and unavailable with "
            "--replay-fixture; not a cognitive time limit, normal exit signal, or default."
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Enable provider reasoning mode (thinking) for this wake: the agent "
            "produces reasoning tokens before each response. Requires provider "
            "support; raises reasoning latency and per-wake cost."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("high", "max"),
        default=None,
        help=(
            "Optional reasoning strength for --thinking; the provider maps "
            "high/max onto its supported reasoning_effort levels. Ignored "
            "unless --thinking is passed."
        ),
    )
    args = parser.parse_args(argv)
    if args.mission is not None:
        warnings.warn(
            "--mission is deprecated; use --perspective instead",
            FutureWarning,
            stacklevel=2,
        )
    if args.graph_shell:
        warnings.warn(
            "--graph-shell is deprecated (G5b-2); the Graph Shell is the default "
            "and only normal runtime, and the flag is a no-op",
            FutureWarning,
            stacklevel=2,
        )
    passed_flags = set(argv) if argv is not None else set(sys.argv[1:])
    for retired, detail in (
        ("wake_protocol", "Graph Shell is the only wake protocol"),
        ("memory_navigation", "memory navigation is fixed"),
        ("digest_cache_reuse", "the digest cache switch is retired"),
    ):
        flag = f"--{retired.replace('_', '-')}"
        if any(item == flag or item.startswith(f"{flag}=") for item in passed_flags):
            warnings.warn(
                f"{flag} is retired (G5b-2) and ignored; {detail}",
                FutureWarning,
                stacklevel=2,
            )
    args.perspective = args.perspective if args.perspective is not None else args.mission
    graph_shell_restore_id = getattr(args, "graph_shell_restore", None)
    graph_shell_finalize_id = getattr(args, "graph_shell_finalize_wake", None)
    management_entry = bool(getattr(args, "graph_shell_status", False)) or bool(
        getattr(args, "graph_shell_abandon", None)
    ) or bool(graph_shell_restore_id) or bool(graph_shell_finalize_id)
    if bool(args.replay_fixture) != bool(args.scripted_model_fixture) and not management_entry:
        parser.error("replay and scripted-model fixtures must be supplied together")
    if management_entry and args.resume:
        parser.error(
            "--graph-shell-status / --graph-shell-abandon / --graph-shell-restore / "
            "--graph-shell-finalize-wake do not combine with --resume"
        )
    if (
        bool(getattr(args, "graph_shell_status", False))
        + bool(getattr(args, "graph_shell_abandon", None))
        + bool(graph_shell_restore_id)
        + bool(graph_shell_finalize_id)
    ) > 1:
        parser.error(
            "--graph-shell-status, --graph-shell-abandon, --graph-shell-restore and "
            "--graph-shell-finalize-wake are mutually exclusive"
        )
    if args.object_id and args.mode != "deep":
        parser.error("--object-id is valid only with --mode deep")
    if args.max_cost_usd is not None and args.max_cost_usd < 0:
        parser.error("--max-cost-usd must be non-negative")
    if args.context_hard_cut_tokens <= 0:
        parser.error("--context-hard-cut-tokens must be positive")
    if args.context_warning_tokens > args.context_hard_cut_tokens:
        parser.error("--context-warning-tokens must not exceed --context-hard-cut-tokens")
    if (args.live_hard_cap_usd is None) != (args.live_cost_audit_path is None):
        parser.error(
            "--live-hard-cap-usd and --live-cost-audit-path must be supplied together; "
            "to run without a pre-call projection cap, omit both"
        )
    if args.live_hard_cap_usd is not None:
        if not args.live_hard_cap_usd.is_finite() or args.live_hard_cap_usd <= 0:
            parser.error("--live-hard-cap-usd must be a finite value greater than zero")
        if args.replay_fixture:
            parser.error("--live-hard-cap-usd is available only for live runs, not replay")
        if args.resume:
            parser.error(
                "--live-hard-cap-usd is unavailable with --resume (it is a per-run "
                "projection guard, while a resumed wake keeps its checkpoint's cumulative "
                "spend); bound cumulative spend with --max-cost-usd instead"
            )
    if args.live_deadline_seconds is not None and (
        not math.isfinite(args.live_deadline_seconds)
        or args.live_deadline_seconds <= 0
        or args.replay_fixture
    ):
        parser.error(
            "--live-deadline-seconds must be a finite positive value and is available "
            "only for live runs (fresh or --resume); it is a pure wall-clock safety "
            "net and needs no --live-hard-cap-usd"
        )
    return args


def _replay(path: Path) -> dict[str, channels.ReplayChannelAdapter]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    batches = lambda model, items: {  # noqa: E731
        key: model.model_validate(value) for key, value in items.items()
    }
    adapter = channels.ReplayChannelAdapter(
        adapter_id=fixture["adapter_id"],
        adapter_version=fixture["adapter_version"],
        discoveries=batches(channels.DiscoveryBatch, fixture["discoveries"]),
        hydrations=batches(channels.ObservationBatch, fixture["hydrations"]),
    )
    return {adapter.adapter_id: adapter}


def _stderr_progress(event: dict[str, object]) -> None:
    """Emit one line-delimited operational event for humans and log collectors."""
    payload = {"type": "progress", "at_monotonic": round(time.monotonic(), 3), **event}
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), file=sys.stderr, flush=True)


def _live(
    names: str,
    settings: Settings,
    progress: Callable[[dict[str, object]], None] = _stderr_progress,
    *,
    adapter_defaults: Mapping[str, Mapping[str, object]] | None = None,
    domain_key: str | None = None,
) -> dict[str, Any]:
    transcriber = (
        FasterWhisperTranscriber(
            settings.asr_model,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
            local_files_only=settings.asr_local_files_only,
            language=settings.asr_language,
            hotwords=settings.asr_hotwords,
            lock_path=Path(settings.data_dir) / "runtime" / "asr-gpu.lock",
            lock_timeout_seconds=settings.asr_lock_timeout_seconds,
            timeout_seconds=settings.asr_timeout_seconds,
            max_audio_bytes=settings.asr_max_audio_bytes,
            progress=progress,
        )
        if settings.asr_enabled
        else None
    )
    if adapter_defaults is not None:
        unknown_adapters = set(adapter_defaults) - {"bilibili", "nga", "hupu", "public-web"}
        if unknown_adapters:
            raise ValueError(
                f"unknown adapter defaults: {', '.join(sorted(unknown_adapters))}"
            )

    def hupu_default_board() -> str | None:
        if adapter_defaults is not None and "hupu" in adapter_defaults:
            value = adapter_defaults["hupu"].get("board")
            return str(value) if value else None
        # bare CLI runs keep the built-in LOL board only for the LOL domain;
        # any other domain degrades to an explicit surface_key instead of
        # silently scanning the wrong board
        return "lol" if domain_key == "lol_cn" else None

    def nga_default_fid() -> str | None:
        if adapter_defaults is not None and "nga" in adapter_defaults:
            value = adapter_defaults["nga"].get("fid")
            return str(value) if value else None
        return "-152678" if domain_key == "lol_cn" else None

    available = {
        "bilibili": lambda: channels.BilibiliChannelAdapter(
            tools.BilibiliSearchTool(sessdata=settings.bilibili_sessdata),
            transcriber=transcriber,
            asr_max_duration_seconds=settings.asr_max_duration_seconds,
            asr_max_audio_bytes=settings.asr_max_audio_bytes,
            progress=progress,
        ),
        "nga": lambda: channels.NgaChannelAdapter(
            tools.NgaPublicTool(session_cookie=settings.nga_cookie),
            default_fid=nga_default_fid(),
        ),
        "hupu": lambda: channels.HupuChannelAdapter(
            tools.HupuPublicTool(),
            default_board=hupu_default_board(),
        ),
        "public-web": lambda: channels.PublicWebChannelAdapter(PublicWebSearchTool()),
    }
    requested = [name.strip() for name in names.split(",") if name.strip()]
    if unknown := set(requested) - available.keys():
        raise ValueError(f"unknown adapters: {', '.join(sorted(unknown))}")
    return {name: available[name]() for name in requested}


async def _checkpoint_terminal(
    runtime_path: Path, thread_id: str
) -> tuple[bool, str]:
    """Return whether *thread_id* has a checkpoint in *runtime_path*.

    The checkpoint's terminal status is returned alongside; this is the
    read-only probe used by the status entry.
    """
    if not runtime_path.exists():
        return False, ""
    connection = await aiosqlite.connect(runtime_path, isolation_level=None)
    try:
        saver = AsyncSqliteSaver(connection)
        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
    finally:
        await connection.close()
    values = (checkpoint.checkpoint or {}).get("channel_values") if checkpoint is not None else None
    if not values:
        return False, ""
    return True, str(dict(values).get("terminal_status") or "")


async def graph_shell_status(world_path: Path, runtime_path: Path) -> dict[str, Any]:
    """Read-only projection of the world's Graph Shell writer gate.

    Reports the singleton lease (owner wake/thread/claimed_at), the owner
    wake's checkpoint existence and terminal, active staging counts, the
    finalize-receipt existence, and one deterministic recovery action derived
    from that durable state. Never opens the world for writing and never
    mutates any database; a missing world/runtime reports a typed status.
    """
    if not world_path.exists():
        return {
            "status": "no_world",
            "world_path": str(world_path),
            "lease": None,
            "recovery_action": "world database does not exist; nothing to recover",
        }
    store = WorldStore(world_path, initialize_schema=False)
    with store.read_connection() as connection:
        try:
            lease = read_writer_lease(connection)
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error):
                # corruption stays loud — only the missing lease table (a
                # genuine pre-v17 world) is the typed legacy marker
                raise
            return {
                "status": "no_active_writer",
                "world_path": str(world_path),
                "schema": "legacy_schema_no_writer_lease",
                "lease": None,
                "recovery_action": (
                    "world predates the Graph Shell writer lease (no "
                    "graph_shell_writer_leases table); no active writer — a "
                    "fresh Graph Shell wake upgrades the schema under the "
                    "normal migration discipline"
                ),
            }
        if lease is None:
            abandoned_by_wake: dict[str, dict[str, int]] = {}
            for table in STAGED_TABLES:
                for row in connection.execute(
                    f"SELECT wake_id, COUNT(*) AS n FROM {table} "
                    "WHERE status = 'abandoned' GROUP BY wake_id"
                ):
                    wake = str(row["wake_id"])
                    abandoned_by_wake.setdefault(wake, {})[table] = int(row["n"])
            if abandoned_by_wake:
                # every ledger appears per wake, zeros included (same shape as
                # the abandon entry's abandoned_items)
                for wake in abandoned_by_wake:
                    for table in STAGED_TABLES:
                        abandoned_by_wake[wake].setdefault(table, 0)
                total = sum(sum(counts.values()) for counts in abandoned_by_wake.values())
                biggest = max(
                    abandoned_by_wake,
                    key=lambda wake: sum(abandoned_by_wake[wake].values()),
                )
                return {
                    "status": "no_active_writer",
                    "world_path": str(world_path),
                    "lease": None,
                    "abandoned_staging": abandoned_by_wake,
                    "recovery_action": (
                        f"{total} abandoned staged item(s) across "
                        f"{len(abandoned_by_wake)} wake(s) await restore (e.g. "
                        f"{biggest}: {abandoned_by_wake[biggest]}); run "
                        f"--graph-shell-restore {biggest} to re-activate the "
                        "working graph, then --resume --wake-id "
                        f"{biggest} to publish it"
                    ),
                }
            return {
                "status": "no_active_writer",
                "world_path": str(world_path),
                "lease": None,
                "recovery_action": "no active writer lease; no recovery needed",
            }
        owner = str(lease["owner_wake_id"])
        owner_thread = str(lease["owner_thread_id"])
        claim = connection.execute(
            "SELECT thread_id, runtime_store_identity "
            "FROM graph_shell_wake_claims WHERE wake_id = ?",
            (owner,),
        ).fetchone()
        receipt = connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (owner,)
        ).fetchone()
        active: dict[str, list[str]] = {}
        objects_without_aliases: list[str] = []
        for table in STAGED_TABLES:
            select = "staged_id, aliases_json" if table == "staged_objects" else "staged_id"
            rows = connection.execute(
                f"SELECT {select} FROM {table} "
                "WHERE wake_id = ? AND status = 'active' ORDER BY staged_id",
                (owner,),
            ).fetchall()
            active[table] = [str(row["staged_id"]) for row in rows]
            if table == "staged_objects":
                # identity observation (F-B follow-up): new objects with no
                # alias declaration can silently split a full/short name into
                # twin objects later; surface them so the operator sees the
                # declaration rate instead of discovering splits in recall.
                objects_without_aliases = [
                    str(row["staged_id"])
                    for row in rows
                    if not json.loads(row["aliases_json"] or "[]")
                ]
    active_total = sum(len(ids) for ids in active.values())
    checkpoint_exists, checkpoint_terminal = await _checkpoint_terminal(
        runtime_path, owner_thread
    )
    if receipt is not None and active_total:
        recovery_action = (
            f"resume fails closed (durable finalize receipt plus {active_total} "
            f"active staged item(s)); run --graph-shell-abandon {owner} to mark "
            "the staging abandoned and release the lease"
        )
    elif receipt is not None:
        recovery_action = (
            f"resume with --resume --wake-id {owner}; the durable receipt replays "
            "as already_published and releases the lease"
        )
    elif checkpoint_exists:
        recovery_action = (
            f"resume with --resume --wake-id {owner}; same-wake resume re-enters "
            "the lease and continues the wake"
        )
    else:
        recovery_action = (
            f"crash before first checkpoint; run --graph-shell-abandon {owner} to "
            "release the lease, then start a fresh wake"
        )
    return {
        "status": "lease_held",
        "world_path": str(world_path),
        "lease": {
            "owner_wake_id": owner,
            "owner_thread_id": owner_thread,
            "claimed_at": float(lease["claimed_at"]),
        },
        "owner_wake": {
            "wake_claim": (
                None
                if claim is None
                else {
                    "thread_id": str(claim["thread_id"]),
                    "runtime_store_identity": str(claim["runtime_store_identity"]),
                }
            ),
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_terminal_status": checkpoint_terminal,
            "active_staging_count": active_total,
            "active_staging": active,
            "active_objects_without_aliases": objects_without_aliases,
            "finalize_receipt": receipt is not None,
        },
        "recovery_action": recovery_action,
    }


async def graph_shell_abandon(
    world_path: Path, runtime_path: Path, wake_id: str
) -> dict[str, Any]:
    """Explicit abandon entry: release the world's writer gate for *wake_id*.

    The typed wake id is the operator's confirmation: the entry refuses
    unless it equals the lease owner verbatim (fail closed on mismatch), then
    marks the wake's active staging abandoned and releases the lease in one
    world transaction, and finally cleans the runtime wake-claim row
    idempotently. Runs no model. A world with no lease reports
    ``already_abandoned`` so repeated runs are harmless.
    """
    if not world_path.exists():
        return {
            "status": "no_world",
            "world_path": str(world_path),
            "wake_id": wake_id,
        }
    store = WorldStore(world_path)
    with store.read_connection() as connection:
        lease = read_writer_lease(connection)
    if lease is None:
        return {
            "status": "already_abandoned",
            "world_path": str(world_path),
            "wake_id": wake_id,
            "abandoned_items": 0,
            "lease_released": False,
        }
    owner = str(lease["owner_wake_id"])
    if owner != wake_id:
        raise ValueError(
            f"graph_shell_abandon_owner_mismatch: the world's writer lease is "
            f"held by {owner}, not {wake_id}; abandon requires the owner wake "
            "id typed verbatim (run --graph-shell-status to confirm the owner)"
        )
    abandoned, released = abandon_writer_lease(store, wake_id)
    claims_cleaned = False
    if runtime_path.exists():
        connection = sqlite3.connect(runtime_path)
        try:
            cursor = connection.execute(
                "DELETE FROM graph_shell_wake_claims WHERE wake_id = ?", (wake_id,)
            )
            connection.commit()
            claims_cleaned = cursor.rowcount > 0
        except sqlite3.OperationalError:
            # no claims table yet (a wake never reached the runtime): nothing
            # to clean; the world transaction above already released the gate
            pass
        finally:
            connection.close()
    return {
        "status": "abandoned",
        "world_path": str(world_path),
        "wake_id": wake_id,
        "abandoned_items": abandoned,
        "lease_released": released,
        "runtime_claims_cleaned": claims_cleaned,
    }


async def graph_shell_restore(
    world_path: Path, runtime_path: Path, wake_id: str
) -> dict[str, Any]:
    """Explicit restore entry: re-activate *wake_id*'s abandoned working graph.

    The reverse of ``--graph-shell-abandon``: re-activates the wake's
    abandoned staged rows (objects / assertions / inquiries) and re-acquires
    the singleton writer lease for the same wake/thread in one world
    transaction, so the documented resume workflow can continue the wake and
    publish the recovered graph. Runs no model.

    Fails closed on: a lease held by a different wake (owner mismatch), a
    wake that still holds the lease (its abandoned rows are mid-run drops, not
    a whole-wake abandon), a published wake (its abandoned rows are dropped
    items), missing world claim or runtime checkpoint (resume could not run),
    a runtime identity mismatch (resume would fail the same way), and
    abandoned rows without a successful patch receipt (untraceable working
    graph). A world with nothing abandoned reports ``nothing_to_restore``.
    """
    if not world_path.exists():
        return {
            "status": "no_world",
            "world_path": str(world_path),
            "wake_id": wake_id,
        }
    store = WorldStore(world_path)
    with store.read_connection() as connection:
        lease = read_writer_lease(connection)
        receipt = connection.execute(
            "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()
        claim = connection.execute(
            "SELECT thread_id, runtime_store_identity "
            "FROM graph_shell_wake_claims WHERE wake_id = ?",
            (wake_id,),
        ).fetchone()
        patch_receipt_rows = connection.execute(
            "SELECT op_id, result_json FROM staged_patch_receipts WHERE wake_id = ?",
            (wake_id,),
        ).fetchall()
        abandoned: dict[str, list[str]] = {}
        for table in STAGED_TABLES:
            rows = connection.execute(
                f"SELECT staged_id FROM {table} "
                "WHERE wake_id = ? AND status = 'abandoned' ORDER BY staged_id",
                (wake_id,),
            ).fetchall()
            abandoned[table] = [str(row["staged_id"]) for row in rows]
    abandoned_total = sum(len(ids) for ids in abandoned.values())
    if lease is not None:
        owner = str(lease["owner_wake_id"])
        if owner != wake_id:
            raise ValueError(
                f"graph_shell_restore_owner_mismatch: the world's writer lease is "
                f"held by {owner}, not {wake_id}; restore requires the owner wake "
                "id typed verbatim (run --graph-shell-status to confirm the owner)"
            )
        return {
            "status": "nothing_to_restore",
            "world_path": str(world_path),
            "wake_id": wake_id,
            "reason": "wake_still_active",
            "restored_items": 0,
        }
    if receipt is not None or abandoned_total == 0:
        return {
            "status": "nothing_to_restore",
            "world_path": str(world_path),
            "wake_id": wake_id,
            "reason": "already_published" if receipt is not None else "no_abandoned_staging",
            "restored_items": 0,
        }
    if claim is None:
        raise ValueError(
            f"graph_shell_restore_claim_missing: wake {wake_id} has abandoned "
            "staging but no world wake claim; restore fails closed"
        )
    if str(claim["runtime_store_identity"]) != str(runtime_path):
        raise ValueError(
            f"graph_shell_restore_runtime_mismatch: wake {wake_id} was claimed in "
            f"runtime {claim['runtime_store_identity']}, not {runtime_path}; restore "
            "fails closed (resume would fail the same way)"
        )
    checkpoint_exists, _ = await _checkpoint_terminal(
        runtime_path, str(claim["thread_id"])
    )
    if not checkpoint_exists:
        raise ValueError(
            f"graph_shell_restore_checkpoint_missing: wake {wake_id} has abandoned "
            "staging but no checkpoint for thread "
            f"{claim['thread_id']} in {runtime_path}; restore would wedge the "
            "world with active staging no resume could continue; restore fails closed"
        )
    # traceability: every abandoned row must be backed by a successful patch
    # receipt (mirror of the resume reconciliation, direction 2)
    successful_ledger_staged_ids: set[str] = set()
    for row in patch_receipt_rows:
        try:
            patch_result = json.loads(str(row["result_json"]))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"durable patch receipt {row['op_id']!r} for wake {wake_id} "
                "contains invalid JSON; restore fails closed"
            ) from error
        if not (isinstance(patch_result, dict) and patch_result.get("status") == "ok"):
            continue
        staged_id = str(patch_result.get("staged_id") or "")
        if staged_id:
            successful_ledger_staged_ids.add(staged_id)
    untraceable = sorted(
        {sid for ids in abandoned.values() for sid in ids} - successful_ledger_staged_ids
    )
    if untraceable:
        raise ValueError(
            f"abandoned staged item(s) {untraceable} for wake {wake_id} have no "
            "successful durable patch receipt in the patch ledger; restore fails "
            "closed — the working graph cannot be traced to an accepted patch"
        )
    restored, acquired = restore_writer_lease(
        store, wake_id, str(claim["thread_id"])
    )
    return {
        "status": "restored",
        "world_path": str(world_path),
        "wake_id": wake_id,
        "restored_items": restored,
        "lease_acquired": acquired,
        "recovery_action": (
            f"run --resume --wake-id {wake_id} (thread {claim['thread_id']}) "
            "to continue the wake and publish the restored working graph"
        ),
    }


async def graph_shell_finalize_wake(world_path: Path, wake_id: str) -> dict[str, Any]:
    """Deterministic management entry: publish *wake_id*'s active staged graph.

    The same finalize_graph the agent calls (I1 one-publish-per-wake, idempotent
    replay, wake_closed for items staged after publishing), so this entry cannot
    diverge from agent behavior and needs no writer lease and no model. Fail
    closed on a missing world, a missing wake claim, or empty active staging;
    the receipt itself carries blocked / compile_failed / commit_rejected
    reasons for operator reading.
    """
    if not world_path.exists():
        return {
            "status": "no_world",
            "world_path": str(world_path),
            "wake_id": wake_id,
        }
    store = WorldStore(world_path)
    with store.read_connection() as connection:
        claim = connection.execute(
            "SELECT thread_id FROM graph_shell_wake_claims WHERE wake_id = ?",
            (wake_id,),
        ).fetchone()
        receipt = connection.execute(
            "SELECT wake_id FROM finalize_receipts WHERE wake_id = ?",
            (wake_id,),
        ).fetchone()
        active_total = 0
        for table in STAGED_TABLES:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {table} "
                "WHERE wake_id = ? AND status = 'active'",
                (wake_id,),
            ).fetchone()
            active_total += int(row[0])
    if claim is None:
        return {
            "status": "wake_unknown",
            "world_path": str(world_path),
            "wake_id": wake_id,
        }
    if receipt is None and active_total == 0:
        return {
            "status": "nothing_to_finalize",
            "world_path": str(world_path),
            "wake_id": wake_id,
            "hint": (
                "run --graph-shell-status to confirm the wake's staging shape; "
                "--graph-shell-restore may apply if its rows were abandoned"
            ),
        }
    # a stored receipt replays as already_published; active rows staged after
    # publishing answer wake_closed (I1); otherwise finalize publishes
    result = finalize_graph(store, wake_id)
    return result.model_dump()


def _storage_fallback_result(error: sqlite3.Error) -> dict[str, Any]:
    """Last-resort storage boundary for the CLI (T1-2).

    A world/checkpoint fault that escaped the typed tool and graph handlers
    (typically ``database is locked``) must not crash the CLI in any phase —
    fresh run or resume. The wake keeps its lease and staged work, so resume
    is the retry.
    """
    return {
        "terminal_status": "staged_unpublished",
        "terminal_summary": (
            "World storage error stopped Graph Shell: "
            f"{type(error).__name__}: {error}; the wake retains its "
            "staged work — resume this wake to retry"
        ),
        "halted": True,
        "resume_allowed": True,
        "finalize_status": "",
    }


async def run(
    args: argparse.Namespace,
    *,
    progress: Callable[[dict[str, object]], None] = _stderr_progress,
    print_report: bool = True,
    operator_instructions: str = "",
) -> dict[str, Any]:
    wake_started_at = time.perf_counter()
    perspective = getattr(args, "perspective", None)
    if perspective is None:
        perspective = getattr(args, "mission", None)
    raw_focus = args.domain
    focus = raw_focus if isinstance(raw_focus, DomainFocus) else resolve_domain_focus(str(raw_focus))
    world_path, runtime_path = args.world_db.resolve(), args.runtime_db.resolve()
    if world_path == runtime_path:
        raise ValueError("world and runtime databases must be distinct")
    if getattr(args, "graph_shell_status", False):
        # management entry: read-only projection of the writer gate; runs no
        # model and never opens the world for writing (a missing world is
        # reported, not created)
        result = await graph_shell_status(world_path, runtime_path)
        if print_report:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    abandon_wake_id = getattr(args, "graph_shell_abandon", None)
    if abandon_wake_id:
        # management entry: explicit operator confirmation to release the
        # gate; runs no model
        result = await graph_shell_abandon(world_path, runtime_path, str(abandon_wake_id))
        if print_report:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    restore_wake_id = getattr(args, "graph_shell_restore", None)
    if restore_wake_id:
        # management entry: re-activate an abandoned working graph so the same
        # wake can be resumed and published; runs no model
        result = await graph_shell_restore(world_path, runtime_path, str(restore_wake_id))
        if print_report:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    finalize_wake_id = getattr(args, "graph_shell_finalize_wake", None)
    if finalize_wake_id:
        # management entry: deterministically publish the wake's active staged
        # working graph through finalize_graph; runs no model
        result = await graph_shell_finalize_wake(world_path, str(finalize_wake_id))
        if print_report:
            # FinalizeReceipt carries committed_at datetimes; serialize them as
            # ISO strings so the published report is JSON-round-trippable.
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return result
    if (args.live_hard_cap_usd is None) != (args.live_cost_audit_path is None):
        raise ValueError("live hard cap and audit path must be configured together")
    if args.live_hard_cap_usd is not None and (
        not args.live_hard_cap_usd.is_finite() or args.live_hard_cap_usd <= 0
    ):
        raise ValueError("live hard cap must be a finite value greater than zero")
    if args.live_hard_cap_usd is not None and args.replay_fixture:
        raise ValueError("live pre-call hard cap is available only for live runs, not replay")
    if args.live_hard_cap_usd is not None and args.resume:
        raise ValueError(
            "live pre-call hard cap is unavailable with --resume; "
            "bound cumulative spend with --max-cost-usd instead"
        )
    if args.live_deadline_seconds is not None and (
        not math.isfinite(args.live_deadline_seconds)
        or args.live_deadline_seconds <= 0
        or args.replay_fixture
    ):
        raise ValueError(
            "live safety deadline must be a finite positive value and is available "
            "only for live runs (fresh or --resume)"
        )
    settings: Settings | None = None
    if args.live_hard_cap_usd is not None:
        settings = get_settings()
        if settings.deepseek_model != FROZEN_G1_MODEL:
            raise ValueError(
                f"live pre-call hard cap requires configured model {FROZEN_G1_MODEL}, "
                f"got {settings.deepseek_model}"
            )
        validate_fresh_audit_path(args.live_cost_audit_path)
    store = WorldStore(world_path)
    wake_deadline_at: float | None = None
    if args.replay_fixture:
        adapters = _replay(args.replay_fixture)
        model = _ScriptedModel(args.scripted_model_fixture)
    else:
        settings = settings or get_settings()
        adapters = _live(
            args.adapters,
            settings,
            progress,
            adapter_defaults=args.adapter_defaults,
            domain_key=getattr(focus, "domain_key", None),
        )
        model = DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )
        if args.live_hard_cap_usd is not None:
            model = LiveCostGuard(
                model=model,
                model_name=settings.deepseek_model,
                hard_cap_usd=args.live_hard_cap_usd,
                audit_path=args.live_cost_audit_path,
            )
        if args.live_deadline_seconds is not None:
            model = LiveDeadlineGuard(model=model, deadline_seconds=args.live_deadline_seconds)
            wake_deadline_at = model.deadline_at
    # per-call cost/cache detail lands in the runtime DB (model_calls table);
    # rounds without a recorder fall back to checkpoint total_cost_usd only
    model_calls = ModelCallRecorder(runtime_path)
    config = {"configurable": {"thread_id": args.thread_id}}
    # G5b-2: the Graph Shell is the default and only normal runtime. Fresh
    # wakes mint a collision-resistant identity; resume restores it from the
    # latest checkpoint; explicit injection remains available for
    # deterministic tests.
    wake_id = args.wake_id or f"wake-{uuid.uuid4().hex}"
    checkpoint_values: dict[str, Any] = {}
    durable_resume_receipt: FinalizeReceipt | None = None
    recovery_summary = ""
    async with (
        _graph_shell_world_run_guard(str(world_path), enabled=True),
        aiosqlite.connect(runtime_path, isolation_level=None) as connection,
    ):
        if args.resume:
            checkpoint = await AsyncSqliteSaver(connection).aget_tuple(config)
            if checkpoint is None or not (checkpoint.checkpoint or {}).get("channel_values"):
                raise ValueError(f"no checkpoint found for thread: {args.thread_id}")
            checkpoint_values = dict((checkpoint.checkpoint or {}).get("channel_values") or {})
            checkpoint_wake_id = str(checkpoint_values.get("wake_id") or "")
            if not checkpoint_wake_id:
                raise ValueError(
                    "checkpoint predates wake identity; start a fresh wake (unset --resume) "
                    "or use a checkpoint created by this build"
                )
            if args.wake_id is not None and args.wake_id != checkpoint_wake_id:
                raise ValueError(
                    f"--wake-id {args.wake_id} does not match checkpoint wake identity "
                    f"{checkpoint_wake_id}"
                )
            wake_id = checkpoint_wake_id
            expected_protocol = {
                "thread_id": args.thread_id,
                "world_store_identity": str(world_path),
                "domain_key": focus.domain_key,
                "agent_mode": args.mode,
                "object_seed": args.object_id or "",
            }
            for field, requested in expected_protocol.items():
                if checkpoint_values.get(field) != requested:
                    raise ValueError(
                        f"checkpoint {field} mismatch: "
                        f"checkpoint={checkpoint_values.get(field)!r}, "
                        f"requested={requested!r}"
                    )
            with store.read_connection() as world_connection:
                staged_rows = sum(
                    int(
                        world_connection.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE wake_id = ?",
                            (wake_id,),
                        ).fetchone()[0]
                    )
                    for table in (
                        "staged_objects",
                        "staged_assertions",
                        "staged_inquiries",
                    )
                )
                staged_ids = {
                    str(row["staged_id"])
                    for table in (
                        "staged_objects",
                        "staged_assertions",
                        "staged_inquiries",
                    )
                    for row in world_connection.execute(
                        f"SELECT staged_id FROM {table} WHERE wake_id = ?",
                        (wake_id,),
                    ).fetchall()
                }
                active_staged_ids = {
                    str(row["staged_id"])
                    for table in (
                        "staged_objects",
                        "staged_assertions",
                        "staged_inquiries",
                    )
                    for row in world_connection.execute(
                        f"SELECT staged_id FROM {table} "
                        "WHERE wake_id = ? AND status = 'active'",
                        (wake_id,),
                    ).fetchall()
                }
                durable_receipt_row = world_connection.execute(
                    "SELECT receipt_json FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
                ).fetchone()
                world_claim_row = world_connection.execute(
                    "SELECT thread_id, runtime_store_identity "
                    "FROM graph_shell_wake_claims WHERE wake_id = ?",
                    (wake_id,),
                ).fetchone()
                active_staged_rows = sum(
                    int(
                        world_connection.execute(
                            f"SELECT COUNT(*) FROM {table} "
                            "WHERE wake_id = ? AND status = 'active'",
                            (wake_id,),
                        ).fetchone()[0]
                    )
                    for table in (
                        "staged_objects",
                        "staged_assertions",
                        "staged_inquiries",
                    )
                )
                active_items = {
                    label: [
                        str(row["staged_id"])
                        for row in world_connection.execute(
                            f"SELECT staged_id FROM {table} "
                            "WHERE wake_id = ? AND status = 'active' "
                            "ORDER BY staged_id LIMIT 10",
                            (wake_id,),
                        ).fetchall()
                    ]
                    for label, table in (
                        ("objects", "staged_objects"),
                        ("assertions", "staged_assertions"),
                        ("inquiries", "staged_inquiries"),
                    )
                }
                patch_ledger_count = int(
                    world_connection.execute(
                        "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?",
                        (wake_id,),
                    ).fetchone()[0]
                )
                patch_receipt_rows = world_connection.execute(
                    "SELECT op_id, result_json FROM staged_patch_receipts WHERE wake_id = ?",
                    (wake_id,),
                ).fetchall()
                foreign_active = sorted(
                    {
                        str(row["wake_id"])
                        for table in (
                            "staged_objects",
                            "staged_assertions",
                            "staged_inquiries",
                        )
                        for row in world_connection.execute(
                            f"SELECT DISTINCT wake_id FROM {table} "
                            "WHERE status = 'active' AND wake_id <> ?",
                            (wake_id,),
                        ).fetchall()
                    }
                )
            if world_claim_row is not None and (
                str(world_claim_row["thread_id"]) != args.thread_id
                or str(world_claim_row["runtime_store_identity"]) != str(runtime_path)
            ):
                raise ValueError(
                    f"world wake claim identity mismatch for {wake_id}: "
                    f"thread={world_claim_row['thread_id']!r}, "
                    f"runtime={world_claim_row['runtime_store_identity']!r}; "
                    "resume fails closed"
                )
            # F-02: bidirectional reconciliation between staging and the
            # patch ledger. Direction 1 (ledger -> staging) below flags
            # successful receipts whose staged item vanished; direction 2
            # (staging -> ledger) flags active staged items with no
            # successful receipt — an impossible state that must never
            # reach the model or finalize.
            successful_ledger_staged_ids: set[str] = set()
            missing_staged_receipts: list[str] = []
            for row in patch_receipt_rows:
                try:
                    patch_result = json.loads(str(row["result_json"]))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"durable patch receipt {row['op_id']!r} for wake {wake_id} "
                        "contains invalid JSON; resume fails closed"
                    ) from error
                if not (isinstance(patch_result, dict) and patch_result.get("status") == "ok"):
                    continue
                staged_id = str(patch_result.get("staged_id") or "")
                if not staged_id:
                    continue
                successful_ledger_staged_ids.add(staged_id)
                if staged_id not in staged_ids:
                    missing_staged_receipts.append(str(row["op_id"]))
            if missing_staged_receipts and durable_receipt_row is None:
                raise ValueError(
                    "successful durable patch receipt(s) "
                    f"{missing_staged_receipts} exist for wake {wake_id}, but their "
                    "durable staging is missing; resume fails closed"
                )
            untraceable_active = sorted(active_staged_ids - successful_ledger_staged_ids)
            if untraceable_active and durable_receipt_row is None:
                raise ValueError(
                    f"active staged item(s) {untraceable_active} for wake {wake_id} "
                    "have no successful durable patch receipt in the patch ledger; "
                    "resume fails closed — the working graph cannot be traced to an "
                    "accepted patch; restore the ledger from backup or abandon the "
                    "wake before retrying"
                )
            if int(checkpoint_values.get("patch_success_count", 0) or 0) > 0 and not (
                staged_rows or patch_ledger_count or durable_receipt_row is not None
            ):
                ledger_note = "" if patch_ledger_count else " no patch ledger exists, and"
                raise ValueError(
                    f"checkpoint records successful patch work for wake {wake_id}, "
                    f"but durable staging is missing,{ledger_note} no finalize receipt "
                    "exists; resume fails closed"
                )
            if foreign_active:
                raise WorldWriteInProgressError(
                    "world_write_in_progress: another Graph Shell wake owns active "
                    f"staging {foreign_active}; resolve that writer before resuming "
                    f"{wake_id}"
                )
            # F-05: same-wake resume is the idempotent re-entry into the
            # world's singleton writer lease. A foreign lease is busy; a
            # missing lease (pre-lease wake or an abandoned owner) is the
            # crash-recovery path that may re-acquire.
            if durable_receipt_row is None:
                with store.write_connection() as world_connection:
                    # v17 schema citizen; WorldStore migrated the world on
                    # open, so no lazy CREATE TABLE here.
                    lease_row = world_connection.execute(
                        "SELECT owner_wake_id, owner_thread_id "
                        "FROM graph_shell_writer_leases WHERE singleton_id = 1"
                    ).fetchone()
                    if (
                        lease_row is not None
                        and str(lease_row["owner_wake_id"]) != wake_id
                    ):
                        raise WorldWriteInProgressError(
                            "world_write_in_progress: the active Graph Shell writer "
                            f"for world {world_path} is wake "
                            f"{lease_row['owner_wake_id']} in thread "
                            f"{lease_row['owner_thread_id']}; resume {wake_id} fails "
                            "closed; run --graph-shell-status to inspect, then "
                            "--graph-shell-abandon "
                            f"{lease_row['owner_wake_id']} to release the lease"
                        )
                    if lease_row is None:
                        try:
                            world_connection.execute(
                                "INSERT INTO graph_shell_writer_leases "
                                "(singleton_id, owner_wake_id, owner_thread_id, "
                                "claimed_at) VALUES (1, ?, ?, ?)",
                                (wake_id, args.thread_id, time.time()),
                            )
                        except sqlite3.IntegrityError as error:
                            raise WorldWriteInProgressError(
                                "world_write_in_progress: the active Graph Shell "
                                f"writer for world {world_path} was claimed "
                                f"concurrently; resume {wake_id} fails closed"
                            ) from error
            checkpoint_terminal = str(checkpoint_values.get("terminal_status") or "")
            if (
                checkpoint_terminal in {"published", "already_published", "wake_closed"}
                and durable_receipt_row is None
            ):
                raise ValueError(
                    f"checkpoint claims terminal publication {checkpoint_terminal!r} "
                    f"for wake {wake_id}, but the durable finalize receipt is missing; "
                    "resume fails closed"
                )
            if durable_receipt_row is not None:
                if active_staged_rows:
                    raise ValueError(
                        f"wake {wake_id} has both a durable finalize receipt and "
                        f"{active_staged_rows} active staged item(s); resume fails closed"
                    )
                durable_resume_receipt = FinalizeReceipt.model_validate_json(
                    durable_receipt_row["receipt_json"]
                )
            else:
                recovery_summary = "Graph Shell durable recovery:\n" + json.dumps(
                    {
                        "wake_id": wake_id,
                        "active": active_items,
                        "active_total": active_staged_rows,
                        "patch_ledger_count": patch_ledger_count,
                        "patch_success_count": int(
                            checkpoint_values.get("patch_success_count", 0) or 0
                        ),
                        "patch_replay_count": int(
                            checkpoint_values.get("patch_replay_count", 0) or 0
                        ),
                        "finalize_receipt": False,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        else:
            with store.read_connection() as world_connection:
                active_wakes = sorted(
                    {
                        str(row["wake_id"])
                        for table in (
                            "staged_objects",
                            "staged_assertions",
                            "staged_inquiries",
                        )
                        for row in world_connection.execute(
                            f"SELECT DISTINCT wake_id FROM {table} WHERE status = 'active'"
                        ).fetchall()
                    }
                )
                closed_wake = world_connection.execute(
                    "SELECT 1 FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
                ).fetchone()
                staged_history = sum(
                    int(
                        world_connection.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE wake_id = ?",
                            (wake_id,),
                        ).fetchone()[0]
                    )
                    for table in (
                        "staged_objects",
                        "staged_assertions",
                        "staged_inquiries",
                    )
                )
                patch_history = int(
                    world_connection.execute(
                        "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?",
                        (wake_id,),
                    ).fetchone()[0]
                )
                world_claim = world_connection.execute(
                    "SELECT thread_id, runtime_store_identity "
                    "FROM graph_shell_wake_claims WHERE wake_id = ?",
                    (wake_id,),
                ).fetchone()
            if closed_wake is not None:
                raise ValueError(f"Graph Shell wake {wake_id} is already finalized")
            if staged_history or patch_history:
                raise ValueError(
                    f"Graph Shell wake {wake_id} already has durable Graph Shell history "
                    f"(staged_rows={staged_history}, patch_receipts={patch_history}); "
                    "fresh wake identities are single-use"
                )
            if active_wakes:
                raise WorldWriteInProgressError(
                    "world_write_in_progress: active Graph Shell staging already "
                    f"exists for wake(s) {active_wakes}; use --resume for the owning "
                    "thread/wake"
                )
            if world_claim is not None:
                raise ValueError(
                    f"Graph Shell wake {wake_id} is already claimed by thread "
                    f"{world_claim['thread_id']} in runtime "
                    f"{world_claim['runtime_store_identity']}; fresh wake identities "
                    "are single-use"
                )
            # F-05: the singleton writer lease is the atomic cross-process gate.
            # The SELECT checks above are early exits; this INSERT is the single
            # decision that makes this process the world's only active writer.
            try:
                with store.write_connection() as world_connection:
                    # The writer-lease table is a v17 world schema citizen
                    # (world/schema.py migration), never lazily created here;
                    # WorldStore already migrated the world on open.
                    world_connection.execute(
                        "INSERT INTO graph_shell_writer_leases "
                        "(singleton_id, owner_wake_id, owner_thread_id, claimed_at) "
                        "VALUES (1, ?, ?, ?)",
                        (wake_id, args.thread_id, time.time()),
                    )
            except sqlite3.IntegrityError as error:
                raise WorldWriteInProgressError(
                    "world_write_in_progress: another process owns the active Graph "
                    f"Shell writer lease for world {world_path}; fresh wakes are "
                    "refused; run --graph-shell-status to inspect, then "
                    "--graph-shell-abandon <owner wake id> to release the lease"
                ) from error
            try:
                try:
                    with store.write_connection() as world_connection:
                        world_connection.execute(
                            "INSERT INTO graph_shell_wake_claims "
                            "(wake_id, thread_id, runtime_store_identity, claimed_at) "
                            "VALUES (?, ?, ?, ?)",
                            (wake_id, args.thread_id, str(runtime_path), time.time()),
                        )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"Graph Shell wake {wake_id} was claimed concurrently in world "
                        f"{world_path}; fresh wake identities are single-use"
                    ) from error
                # F-08: fresh runtimes get this table eagerly from
                # initialize_blank_runtime (RUNTIME_TABLES contract); this
                # CREATE TABLE IF NOT EXISTS is the idempotent fallback for
                # runtimes created before the registry included the table.
                await connection.execute(
                    "CREATE TABLE IF NOT EXISTS graph_shell_wake_claims ("
                    "wake_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, "
                    "world_store_identity TEXT NOT NULL, claimed_at REAL NOT NULL)"
                )
                prior_claim = await (
                    await connection.execute(
                        "SELECT thread_id FROM graph_shell_wake_claims WHERE wake_id = ?",
                        (wake_id,),
                    )
                ).fetchone()
                prior_model_call = await (
                    await connection.execute(
                        "SELECT thread_id FROM model_calls WHERE wake_id = ? LIMIT 1",
                        (wake_id,),
                    )
                ).fetchone()
                if prior_claim is not None or prior_model_call is not None:
                    owner = str((prior_claim or prior_model_call)[0])
                    raise ValueError(
                        f"Graph Shell wake {wake_id} is already claimed by thread "
                        f"{owner}; fresh wake identities are single-use"
                    )
                try:
                    await connection.execute(
                        "INSERT INTO graph_shell_wake_claims "
                        "(wake_id, thread_id, world_store_identity, claimed_at) "
                        "VALUES (?, ?, ?, ?)",
                        (wake_id, args.thread_id, str(world_path), time.time()),
                    )
                except aiosqlite.IntegrityError as error:
                    raise ValueError(
                        f"Graph Shell wake {wake_id} was claimed concurrently; "
                        "fresh wake identities are single-use"
                    ) from error
            except Exception:
                # A fresh wake that cannot complete its claim must not wedge the
                # world: release the writer lease before propagating.
                with store.write_connection() as world_connection:
                    world_connection.execute(
                        "DELETE FROM graph_shell_writer_leases WHERE owner_wake_id = ?",
                        (wake_id,),
                    )
                raise
        # One shared set per run: digest execution mutates it by reference.
        # The separately resolved wake owns all working-graph operations.
        tools = WorldTools(
            store=store,
            adapters=adapters,
            leases=InquiryLeaseStore(runtime_path),
            digested_ids=set(),
            domain=focus.domain_key,
            thread_id=args.thread_id,
            wake_id=wake_id,
            mode=args.mode,
            discovery_timeout_seconds=(
                settings.tool_discovery_timeout_seconds if settings is not None else 45.0
            ),
            hydration_timeout_seconds=(
                settings.tool_hydration_timeout_seconds if settings is not None else 90.0
            ),
            full_hydration_timeout_seconds=(
                settings.tool_full_hydration_timeout_seconds if settings is not None else 900.0
            ),
            wake_deadline_at=wake_deadline_at,
            closed_wake_guard=True,
            progress=progress,
        )
        graph = build_world_agent_graph(
            model=model,
            tools=tools,
            store=store,
            checkpointer=AsyncSqliteSaver(connection),
            max_turns=args.max_turns,
            max_cost_usd=args.max_cost_usd,
            mode=args.mode,
            object_id=args.object_id,
            focus=focus,
            # Task 5.3: the composition root freezes ONE reasoning profile for
            # the whole wake, so the request envelope is one cacheable key
            reasoning_profile=WakeReasoningProfile(
                thinking=args.thinking,
                reasoning_effort=args.reasoning_effort,
            ),
            model_calls=model_calls,
            thread_id=args.thread_id,
            wake_id=wake_id,
            operator_instructions=operator_instructions,
            # C-1e: production runs re-validate the writer lease before every
            # world mutation and fail closed on writer_lease_lost
            enforce_writer_lease=True,
            context_warning_tokens=args.context_warning_tokens,
            context_hard_cut_tokens=args.context_hard_cut_tokens,
        )
        if args.resume and not (await graph.aget_state(config)).values:
            raise ValueError(f"no checkpoint found for thread: {args.thread_id}")
        initial_messages = [
            {
                "role": "user",
                "content": render_wake_input(perspective),
            }
        ]
        initial_state = None if args.resume else {"messages": initial_messages}
        if not args.resume:
            initial_state = graph_shell_initial_state(
                messages=initial_messages,
                store=store,
                thread_id=args.thread_id,
                wake_id=wake_id,
                domain_key=focus.domain_key,
                mode=args.mode,
                object_id=args.object_id,
            )
        result: dict[str, Any] | None = None
        if durable_resume_receipt is not None:
            replayed_receipt = durable_resume_receipt.model_copy(
                update={"status": "already_published", "replayed": True}
            )
            try:
                await graph.aupdate_state(
                    config,
                    {
                        "terminal_status": "already_published",
                        "terminal_summary": (
                            f"Durable finalize receipt already published wake {wake_id}; "
                            "no model or tool call resumed."
                        ),
                        "halted": True,
                        "resume_allowed": False,
                        "finalize_status": "already_published",
                        "finalize_receipt": replayed_receipt.model_dump(mode="json"),
                        "resume_count": int(checkpoint_values.get("resume_count", 0) or 0) + 1,
                    },
                    as_node="tools",
                )
            except sqlite3.Error as error:
                # A checkpoint write fault during the replay path is the same
                # typed storage boundary as a fresh run: nothing was consumed,
                # the lease is kept, resume is the retry.
                result = _storage_fallback_result(error)
            else:
                release_writer_lease(store, wake_id)
                result = dict((await graph.aget_state(config)).values)
        elif args.resume:
            try:
                snapshot = await graph.aget_state(config)
                resume_update: dict[str, Any] = {
                    "resume_count": int(snapshot.values.get("resume_count", 0) or 0) + 1,
                    "pending_recovery_summary": recovery_summary,
                }
                if snapshot.values.get("terminal_status") == "staged_unpublished":
                    # Resume must not rewrite wake-scoped control state: the
                    # checkpoint's completion_reminder_used (durable once-per-wake
                    # reminder contract) survives resume untouched.
                    await graph.aupdate_state(
                        config,
                        {
                            **resume_update,
                            "terminal_status": "",
                            "terminal_summary": "",
                            "halted": False,
                        },
                        as_node="completion_reminder",
                    )
                else:
                    await graph.aupdate_state(config, resume_update)
            except sqlite3.Error as error:
                # The resume checkpoint read/write itself hit storage: the
                # checkpoint is still the pre-resume state, so a later resume
                # retries cleanly (idempotent re-entry, F-05).
                result = _storage_fallback_result(error)
        if durable_resume_receipt is None and result is None:
            # ``result`` is already set when a resume-path storage fault
            # produced the fallback: the checkpoint write never happened, so
            # starting a fresh ainvoke would resume a wake whose checkpoint
            # was not updated and hide the storage fault behind new work.
            try:
                result = await graph.ainvoke(initial_state, config)
            except sqlite3.Error as error:
                # Last-resort storage boundary: a world/checkpoint fault that
                # escaped the typed tool and graph handlers (typically
                # ``database is locked``) must not crash the CLI. The wake
                # keeps its lease and staged work, so resume is the retry.
                result = _storage_fallback_result(error)
            # F-05 frozen lease lifecycle (world/writer_lease.py): published /
            # already_published release the gate; wake_closed releases it only
            # when nothing active remains staged for this wake — a
            # wake_closed wake with stranded active staging keeps the gate and
            # gets an explicit recovery action instead. blocked /
            # compile_failed / commit_rejected / staged_unpublished and
            # reconciliation / provider failures keep the gate.
            terminal_status = str(result.get("terminal_status") or "")
            action = writer_lease_action(terminal_status, wake_has_active_staging(store, wake_id))
            if action == "release":
                release_writer_lease(store, wake_id)
            elif action == "keep_recovery_action":
                result["terminal_summary"] = (
                    f"{result.get('terminal_summary', '')} "
                    "Wake holds stranded active staging that can never publish "
                    f"under {wake_id}; run --graph-shell-status then "
                    f"--graph-shell-abandon {wake_id} to mark it abandoned and "
                    "release the writer lease."
                )
    if isinstance(model, _ScriptedModel) and model._turns:
        raise RuntimeError(f"{len(model._turns)} unconsumed scripted model turn(s)")
    try:
        report = build_run_report(
            thread_id=args.thread_id,
            wake_id=wake_id,
            domain=focus.domain_key,
            mission=perspective or "",
            mode=args.mode,
            object_id=args.object_id,
            result=result,
            store=store,
            runtime_path=runtime_path,
            wall_ms=(time.perf_counter() - wake_started_at) * 1000,
        )
        # Keep the structured, evidence-backed report available to non-CLI
        # callers such as the local observatory. It is a read-only projection
        # of existing ledgers and never feeds back into graph state or routing.
        result["run_report"] = report
        if print_report:
            print(render_run_report(report))
    except Exception as error:  # noqa: BLE001 - reporting cannot undo a successful commit
        if print_report:
            print(f"run report unavailable: {type(error).__name__}: {error}", file=sys.stderr)
    if print_report:
        print(f"world_db={world_path}\nruntime_db={runtime_path}")
    return result


def main() -> None:
    """Run the world-agent command-line entrypoint."""
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
