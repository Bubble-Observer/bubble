"""Loopback-only personal Web console for the retained world agent."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sqlite3
import webbrowser
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiosqlite
import uvicorn
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from leave_information_bubble.config import Settings, get_settings
from leave_information_bubble.world_agent.cli import (
    graph_shell_abandon,
    graph_shell_finalize_wake,
)
from leave_information_bubble.world_agent.cli import (
    parse_args as parse_world_args,
)
from leave_information_bubble.world_agent.cli import (
    run as run_world_agent,
)

from .inspection import ReadOnlyInspection
from .local_settings import local_settings_status, update_local_settings
from .profiles import (
    AgentProfile,
    AgentProfileRegistry,
    build_prompt_preview,
    build_quick_profile,
    clone_profile_design,
)
from .runs import JsonValue, RunAlreadyActiveError, RunManager, RunRecord, RunSpec

_ROOT = Path.cwd().resolve()
_STATIC_DIR = Path(__file__).with_name("static")
_PROFILE_DIR = _ROOT / "data" / "run-configs" / "agents"
_NO_STORE_HEADERS = {"Cache-Control": "no-store, max-age=0"}


class _NoStoreStaticFiles(StaticFiles):
    """Serve local console assets without retaining cross-edit browser caches."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(_NO_STORE_HEADERS)
        return response


class PromptPreviewRequest(BaseModel):
    """Per-preview mode inputs; editable content remains in the profile."""

    model_config = ConfigDict(extra="forbid")

    mode: str | None = None
    wake_protocol: str | None = None
    object_id: str | None = Field(default=None, max_length=200)
    perspective: str | None = Field(default=None, max_length=20_000)

    @field_validator("perspective")
    @classmethod
    def trim_perspective(cls, value: str | None) -> str | None:
        """Preview the same normalized optional wake perspective as a live run."""
        if value is None:
            return None
        return value.strip() or None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str | None) -> str | None:
        """Accept only the two retained world-agent postures."""
        if value is not None and value not in {"broad", "deep"}:
            raise ValueError("mode must be broad or deep")
        return value

    @field_validator("wake_protocol")
    @classmethod
    def validate_protocol(cls, value: str | None) -> str | None:
        """Accept only a graph protocol implemented by the engine."""
        if value is not None and value not in {"current", "separated"}:
            raise ValueError("wake_protocol must be current or separated")
        return value


class CreateRunRequest(BaseModel):
    """Validated browser request for one immutable world-agent wake."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    perspective: str | None = Field(default=None, max_length=20_000)
    mission: str | None = Field(default=None, max_length=20_000)
    thread_id: str | None = Field(default=None, max_length=160)
    mode: str | None = None
    object_id: str | None = Field(default=None, max_length=200)
    max_turns: int | None = Field(default=None, ge=1, le=200)
    max_cost_usd: float | None = Field(default=None, gt=0, le=1000)
    adapters: str | list[str] | None = None
    thinking: bool | None = None
    reasoning_effort: str | None = None
    wake_protocol: str | None = None
    memory_navigation: str | None = None
    digest_cache_reuse: bool | None = None

    @field_validator("profile_id")
    @classmethod
    def trim_required(cls, value: str) -> str:
        """Reject a whitespace-only profile identifier."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @field_validator("perspective", "mission")
    @classmethod
    def trim_optional_perspective(cls, value: str | None) -> str | None:
        """Treat an empty observation perspective as an intentional neutral wake."""
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def perspective_alias_is_unambiguous(self) -> CreateRunRequest:
        """Accept legacy mission payloads without allowing two competing values."""
        if self.perspective is not None and self.mission is not None:
            raise ValueError("perspective and legacy mission cannot both be provided")
        return self

    @property
    def wake_perspective(self) -> str | None:
        """Return the canonical optional perspective from either API spelling."""
        return self.perspective if self.perspective is not None else self.mission

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str | None) -> str | None:
        """Accept only Broad or Deep mode overrides."""
        if value is not None and value not in {"broad", "deep"}:
            raise ValueError("mode must be broad or deep")
        return value

    @field_validator("wake_protocol")
    @classmethod
    def validate_protocol(cls, value: str | None) -> str | None:
        """Accept only graph protocols implemented by the engine."""
        if value is not None and value not in {"current", "separated"}:
            raise ValueError("wake_protocol must be current or separated")
        return value

    @field_validator("memory_navigation")
    @classmethod
    def validate_navigation(cls, value: str | None) -> str | None:
        """Accept only supported memory navigation modes."""
        if value is not None and value not in {"legacy", "overview_v1"}:
            raise ValueError("memory_navigation must be legacy or overview_v1")
        return value

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, value: str | None) -> str | None:
        """Accept only the reasoning strengths the CLI maps to the provider."""
        if value is not None and value not in {"high", "max"}:
            raise ValueError("reasoning_effort must be high or max")
        return value


class QuickProfileRequest(BaseModel):
    """Low-burden Agent creation input; shared cognition stays out of the profile."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=64)
    display_name: str = Field(min_length=1, max_length=120)
    observation_center: str = Field(min_length=1, max_length=2000)
    locale: str = Field(default="zh-CN", min_length=1, max_length=80)
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=80)
    adapters: tuple[str, ...] = ("bilibili", "public-web")


class LocalSettingsRequest(BaseModel):
    """Allowlisted machine-level connection values submitted by the local console."""

    model_config = ConfigDict(extra="forbid")

    deepseek_api_key: str | None = Field(default=None, max_length=20_000)
    deepseek_base_url: str | None = Field(default=None, max_length=2_000)
    deepseek_model: str | None = Field(default=None, max_length=500)
    bilibili_sessdata: str | None = Field(default=None, max_length=20_000)
    nga_cookie: str | None = Field(default=None, max_length=20_000)

    def env_updates(self) -> dict[str, str]:
        """Translate only explicitly supplied values to the supported `.env` spelling."""
        return {
            name.upper(): value
            for name, value in self.model_dump(exclude_unset=True, exclude_none=True).items()
        }


class CloneProfileRequest(BaseModel):
    """Copy only one Agent's design into a fresh, empty memory identity."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=64)
    display_name: str = Field(min_length=1, max_length=120)


def _generated_profile_id(display_name: str, requested: str | None, existing: set[str]) -> str:
    """Return an explicit safe id, or derive a collision-free local identifier."""
    if requested is not None and requested.strip():
        candidate = requested.strip()
        if candidate in existing:
            raise FileExistsError(f"profile already exists: {candidate}")
        return candidate
    base = re.sub(r"[^a-z0-9_-]+", "-", display_name.casefold()).strip("-_")
    base = (base or "agent")[:64]
    candidate = base
    index = 2
    while candidate in existing:
        suffix = f"-{index}"
        candidate = f"{base[: 64 - len(suffix)]}{suffix}"
        index += 1
    return candidate


def _workspace_path(relative: str) -> Path:
    path = (_ROOT / relative).resolve()
    if path != _ROOT and _ROOT not in path.parents:
        raise ValueError("configured database path escapes the workspace")
    return path


def _thread_id(profile_id: str) -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"console-{profile_id}-{stamp}-{secrets.token_hex(2)}"


def _adapter_text(value: str | list[str] | None, profile: AgentProfile) -> str:
    if value is None:
        return ",".join(profile.defaults.adapters)
    items = value.split(",") if isinstance(value, str) else value
    cleaned = [item.strip() for item in items if item.strip()]
    if not cleaned:
        raise ValueError("at least one adapter is required")
    return ",".join(cleaned)


def _world_namespace(profile: AgentProfile, request: CreateRunRequest) -> argparse.Namespace:
    mode = request.mode or profile.defaults.mode
    object_id = request.object_id.strip() if request.object_id else None
    if object_id and mode != "deep":
        raise ValueError("object_id is valid only with deep mode")
    # G5b-2: wake-protocol / memory-navigation / digest-cache-reuse are
    # retired and must not be emitted (the profile/request fields remain
    # accepted for schema compatibility but are no longer threaded to the CLI).
    arguments = [
        "--thread-id",
        request.thread_id or _thread_id(profile.id),
        "--world-db",
        str(_workspace_path(profile.world_db)),
        "--runtime-db",
        str(_workspace_path(profile.runtime_db)),
        "--mode",
        mode,
        "--max-turns",
        str(request.max_turns or profile.defaults.max_turns),
        "--adapters",
        _adapter_text(request.adapters, profile),
    ]
    if request.wake_perspective:
        arguments.extend(("--perspective", request.wake_perspective))
    cost = request.max_cost_usd if request.max_cost_usd is not None else profile.defaults.max_cost_usd
    if cost is not None:
        arguments.extend(("--max-cost-usd", str(cost)))
    if object_id:
        arguments.extend(("--object-id", object_id))
    thinking = request.thinking if request.thinking is not None else profile.defaults.thinking
    reasoning_effort = (
        request.reasoning_effort
        if request.reasoning_effort is not None
        else profile.defaults.reasoning_effort
    )
    if thinking:
        arguments.append("--thinking")
    if reasoning_effort:
        arguments.extend(("--reasoning-effort", reasoning_effort))
    if profile.adapter_defaults:
        arguments.extend(("--adapter-defaults", json.dumps(profile.adapter_defaults)))
    namespace = parse_world_args(arguments)
    namespace.domain = profile.focus
    return namespace


async def _default_runner(
    spec: RunSpec,
    publish: Callable[[str, str], object],
) -> Mapping[str, JsonValue]:
    config = dict(spec.config)
    arguments = config.pop("world_args")
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError("run snapshot has invalid world arguments")
    namespace = _parse_world_arguments(arguments)
    raw_focus = config.pop("domain_focus")
    if not isinstance(raw_focus, dict):
        raise ValueError("run snapshot has invalid domain focus")
    profile = AgentProfile.model_validate(config.pop("profile_snapshot"))
    if raw_focus != _serialize(profile.focus):
        raise ValueError("run snapshot domain focus does not match the profile snapshot")
    namespace.domain = profile.focus
    # the profile is the single source of truth for per-domain adapter
    # surfaces; snapshot world_args may predate adapter_defaults entirely
    namespace.adapter_defaults = profile.adapter_defaults

    def progress(event: dict[str, object]) -> None:
        kind = str(event.get("type") or event.get("stage") or "tool.progress")
        message = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        publish(kind, message)

    result = await run_world_agent(
        namespace,
        progress=progress,
        print_report=False,
        operator_instructions=profile.operator_instructions or "",
    )
    return _result_summary(result)


def _result_summary(result: Mapping[str, Any]) -> dict[str, JsonValue]:
    fields = (
        "terminal_summary",
        "terminal_status",
        "turn_count",
        "total_cost_usd",
        "halted",
        "phase",
        "transition_reason",
        "wake_id",
        "finalize_status",
        "finalize_receipt",
        "commit_receipt",
        "run_report",
    )
    summary: dict[str, JsonValue] = {}
    for field in fields:
        value = result.get(field)
        try:
            json.dumps(value)
        except TypeError:
            continue
        summary[field] = value
    return summary


def _serialize(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[union-attr]
    if hasattr(value, "__dataclass_fields__"):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(item) for item in value]
    return value


async def _payload(request: Request) -> dict[str, Any]:
    try:
        value = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("request body must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _error(detail: str, status: int) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def _query_int(
    request: Request,
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int = 100,
) -> int:
    raw = request.query_params.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _inquiry_subject_views(items: object) -> list[dict[str, object]]:
    if not isinstance(items, list):
        return []
    output: list[dict[str, object]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        subject_id = item.pop("subject_id", None)
        subject_name = item.pop("subject_name", None)
        item["subject"] = {"id": subject_id, "name": subject_name}
        output.append(item)
    return output


def _commit_summaries(items: object) -> list[dict[str, object]]:
    if not isinstance(items, list):
        return []
    output: list[dict[str, object]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        receipt = raw.get("receipt")
        receipt = receipt if isinstance(receipt, dict) else {}
        output.append(
            {
                "commit_id": raw.get("commit_id"),
                "committed_at": raw.get("committed_at"),
                "counts": {
                    "objects": len(receipt.get("object_ids", [])),
                    "assertions": len(receipt.get("assertion_ids", [])),
                    "inquiries": len(receipt.get("inquiry_ids", [])),
                    "resolved_inquiries": len(receipt.get("resolved_inquiry_ids", [])),
                },
            }
        )
    return output


def _run_inspection_view(
    raw: Mapping[str, object],
    report: object,
    result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    report = report if isinstance(report, dict) else {}
    result = result or {}
    proposal = raw.get("proposal_review")
    proposal = proposal if isinstance(proposal, dict) else {}
    attempts = proposal.get("attempts")
    attempts = attempts if isinstance(attempts, list) else []
    review = report.get("review")
    review = dict(review) if isinstance(review, dict) else {}
    review.setdefault("attempts", len(attempts))
    review.setdefault("omitted", proposal.get("omissions", {}))
    model = report.get("model")
    if not isinstance(model, dict):
        model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    if "successful_calls" not in model:
        model = {**model, "successful_calls": model.get("calls", 0)}
    execution = report.get("execution")
    if not isinstance(execution, dict) or not execution:
        execution = {
            "status": "incomplete" if result.get("halted") else "complete",
            "turns": int(result.get("turn_count", 0) or 0),
            "stop_reason": str(
                result.get("transition_reason")
                or result.get("terminal_summary")
                or "record restored without a persisted run report"
            ),
        }
    durable = report.get("durable_diff")
    if not isinstance(durable, dict):
        raw_diff = raw.get("durable_diff")
        raw_diff = raw_diff if isinstance(raw_diff, dict) else {}
        durable = {
            key: len(raw_diff.get(key, [])) if isinstance(raw_diff.get(key), list) else 0
            for key in ("objects", "assertions", "inquiries", "resolved_inquiries")
        }
    return {
        "entry": report.get("entry", {}),
        "execution": execution,
        "model": model,
        "tools": report.get("tools", {}),
        "review": review,
        "issues": report.get("issues", {"hard": [], "soft": []}),
        "durable": durable,
        "publication": report.get("publication", {}),
        "writes": raw.get("writes", {}),
        "limitations": raw.get("limitations", []),
    }


def create_app(
    *,
    registry: AgentProfileRegistry | None = None,
    manager: RunManager | None = None,
    settings_file: Path | None = None,
) -> Starlette:
    """Build the local console app with injectable offline dependencies."""
    profiles = registry or AgentProfileRegistry(_PROFILE_DIR)
    runs = manager or RunManager(_default_runner)
    workspace_root = Path.cwd().resolve() if registry is None else registry.directory.parent.parent.parent
    resolved_settings_file = (settings_file or workspace_root / ".env").resolve()
    profiles.bootstrap_lol_profile()

    def current_settings() -> Settings:
        return get_settings() if settings_file is None else Settings(_env_file=resolved_settings_file)

    async def restore_runtime_history() -> None:
        for profile in profiles.list():
            for record in await _runtime_run_records(profile, workspace_root):
                await runs.restore(record)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        await restore_runtime_history()
        yield

    async def index(_request: Request) -> Response:
        return FileResponse(_STATIC_DIR / "index.html", headers=_NO_STORE_HEADERS)

    async def system(_request: Request) -> Response:
        settings = current_settings()
        return JSONResponse(
            {
                "provider": {
                    "name": "DeepSeek",
                    "model": settings.deepseek_model,
                    "base_url": settings.deepseek_base_url,
                    "key_configured": bool(settings.deepseek_api_key),
                },
                "limits": {"single_writer": True, "loopback_only": True},
            }
        )

    async def local_settings(request: Request) -> Response:
        try:
            if request.method == "PUT":
                payload = LocalSettingsRequest.model_validate(await _payload(request))
                update_local_settings(resolved_settings_file, payload.env_updates())
                get_settings.cache_clear()
            return JSONResponse(local_settings_status(current_settings()))
        except ValidationError as error:
            fields = sorted(
                {str(issue["loc"][-1]) for issue in error.errors(include_input=False) if issue.get("loc")}
            )
            suffix = f": {', '.join(fields)}" if fields else ""
            return _error(f"invalid local settings{suffix}", 422)
        except ValueError as error:
            return _error(str(error), 422)

    async def profile_collection(request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse([_profile_view(profile, workspace_root) for profile in profiles.list()])
        try:
            profile = AgentProfile.model_validate(await _payload(request))
            return JSONResponse(_profile_view(profiles.create(profile), workspace_root), status_code=201)
        except FileExistsError as error:
            return _error(str(error), 409)
        except (ValidationError, ValueError) as error:
            return _error(str(error), 422)

    async def profile_quick_create(request: Request) -> Response:
        try:
            payload = QuickProfileRequest.model_validate(await _payload(request))
            agent_id = _generated_profile_id(
                payload.display_name,
                payload.id,
                {profile.id for profile in profiles.list()},
            )
            profile = build_quick_profile(
                agent_id=agent_id,
                display_name=payload.display_name.strip(),
                observation_center=payload.observation_center.strip(),
                locale=payload.locale.strip(),
                timezone=payload.timezone.strip(),
                adapters=payload.adapters,
            )
            return JSONResponse(
                _profile_view(profiles.create(profile), workspace_root),
                status_code=201,
            )
        except FileExistsError as error:
            return _error(str(error), 409)
        except (ValidationError, ValueError) as error:
            return _error(str(error), 422)

    async def profile_clone(request: Request) -> Response:
        try:
            source = profiles.get(request.path_params["profile_id"])
            payload = CloneProfileRequest.model_validate(await _payload(request))
            agent_id = _generated_profile_id(
                payload.display_name,
                payload.id,
                {profile.id for profile in profiles.list()},
            )
            profile = clone_profile_design(
                source,
                agent_id=agent_id,
                display_name=payload.display_name.strip(),
            )
            return JSONResponse(
                _profile_view(profiles.create(profile), workspace_root),
                status_code=201,
            )
        except KeyError as error:
            return _error(str(error), 404)
        except FileExistsError as error:
            return _error(str(error), 409)
        except (ValidationError, ValueError) as error:
            return _error(str(error), 422)

    async def profile_item(request: Request) -> Response:
        try:
            existing = profiles.get(request.path_params["profile_id"])
            if request.method == "GET":
                return JSONResponse(_profile_view(existing, workspace_root))
            payload = await _payload(request)
            if payload.get("id", existing.id) != existing.id:
                return _error("profile id cannot be changed", 409)
            # GET views include derived, read-only memory counts so the browser can
            # render them.  Accept that same view as an update without treating the
            # derived field as persisted profile configuration.
            payload.pop("memory", None)
            payload["id"] = existing.id
            return JSONResponse(
                _profile_view(profiles.update(AgentProfile.model_validate(payload)), workspace_root)
            )
        except KeyError as error:
            return _error(str(error), 404)
        except (ValidationError, ValueError) as error:
            return _error(str(error), 422)

    async def prompt_preview(request: Request) -> Response:
        try:
            profile = profiles.get(request.path_params["profile_id"])
            payload = PromptPreviewRequest.model_validate(await _payload(request))
            preview = build_prompt_preview(
                profile,
                mode=payload.mode,  # type: ignore[arg-type]
                wake_protocol=payload.wake_protocol,  # type: ignore[arg-type]
                object_id=payload.object_id,
                perspective=payload.perspective,
            )
            return JSONResponse(_serialize(preview))
        except KeyError as error:
            return _error(str(error), 404)
        except (ValidationError, ValueError) as error:
            return _error(str(error), 422)

    def inspection_for(profile: AgentProfile) -> ReadOnlyInspection:
        return ReadOnlyInspection(
            (workspace_root / profile.world_db).resolve(),
            (workspace_root / profile.runtime_db).resolve(),
        )

    async def memory_summary(request: Request) -> Response:
        try:
            reader = inspection_for(profiles.get(request.path_params["profile_id"]))
            headline = reader.summary()
            counts = headline["counts"]
            object_samples = reader.search_objects(limit=8)["items"]
            return JSONResponse(
                {
                    "initialized": headline["initialized"],
                    "summary": {
                        "objects": counts["objects"],
                        "current_assertions": counts["current_assertions"],
                        "open_inquiries": counts["open_inquiries"],
                        "observations": counts["observations"],
                        "cognition_commits": counts["cognition_commits"],
                    },
                    # This is a stable alphabetical sample, not a recency claim.
                    "object_samples": object_samples,
                    "recent_objects": object_samples,
                    "open_inquiries": _inquiry_subject_views(
                        reader.inquiries(statuses=("open",), limit=8)["items"]
                    ),
                    "recent_commits": _commit_summaries(reader.recent_commits(limit=8)["items"]),
                }
            )
        except KeyError as error:
            return _error(str(error), 404)
        except ValueError as error:
            return _error(str(error), 422)

    async def memory_graph(request: Request) -> Response:
        try:
            reader = inspection_for(profiles.get(request.path_params["profile_id"]))
            limit = _query_int(request, "limit", 24, maximum=32)
            window = _query_int(request, "window", 0, minimum=0, maximum=10_000)
            return JSONResponse(reader.graph_snapshot(limit=limit, window=window))
        except KeyError as error:
            return _error(str(error), 404)
        except ValueError as error:
            return _error(str(error), 422)

    async def memory_objects(request: Request) -> Response:
        try:
            reader = inspection_for(profiles.get(request.path_params["profile_id"]))
            limit = _query_int(request, "limit", 25)
            offset = _query_int(request, "offset", 0, minimum=0, maximum=10_000)
            result = reader.search_objects(
                request.query_params.get("query", ""),
                kind=request.query_params.get("kind") or None,
                limit=min(100, limit + offset),
            )
            items = result["items"][offset : offset + limit]
            return JSONResponse(
                {
                    "items": items,
                    "limit_applied": limit,
                    "offset": offset,
                    "has_more": len(result["items"]) > offset + limit,
                }
            )
        except KeyError as error:
            return _error(str(error), 404)
        except ValueError as error:
            return _error(str(error), 422)

    async def memory_object_detail(request: Request) -> Response:
        try:
            reader = inspection_for(profiles.get(request.path_params["profile_id"]))
            detail = reader.object_detail(request.path_params["object_id"])
            return JSONResponse(detail) if detail is not None else _error("unknown object", 404)
        except KeyError as error:
            return _error(str(error), 404)

    async def memory_inquiries(request: Request) -> Response:
        try:
            reader = inspection_for(profiles.get(request.path_params["profile_id"]))
            raw_statuses = request.query_params.getlist("status") or ["open", "dormant"]
            statuses = tuple(
                item.strip() for value in raw_statuses for item in value.split(",") if item.strip()
            )
            limit = _query_int(request, "limit", 25)
            offset = _query_int(request, "offset", 0, minimum=0, maximum=10_000)
            result = reader.inquiries(
                statuses=statuses,
                object_id=request.query_params.get("object_id") or None,
                limit=min(100, limit + offset),
            )
            all_items = _inquiry_subject_views(result["items"])
            return JSONResponse(
                {
                    "items": all_items[offset : offset + limit],
                    "limit_applied": limit,
                    "offset": offset,
                    "has_more": len(all_items) > offset + limit,
                }
            )
        except KeyError as error:
            return _error(str(error), 404)
        except ValueError as error:
            return _error(str(error), 422)

    async def run_collection(request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse([_run_view(record) for record in await runs.list_runs()])
        try:
            payload = CreateRunRequest.model_validate(await _payload(request))
            profile = profiles.get(payload.profile_id)
            namespace = _world_namespace(profile, payload)
            world_args = _namespace_arguments(namespace)
            snapshot: dict[str, JsonValue] = {
                "profile_id": profile.id,
                "mode": namespace.mode,
                "perspective": namespace.perspective,
                "object_id": namespace.object_id,
                "max_turns": namespace.max_turns,
                "max_cost_usd": namespace.max_cost_usd,
                "adapters": namespace.adapters,
                "thinking": namespace.thinking,
                "reasoning_effort": namespace.reasoning_effort,
                "wake_protocol": namespace.wake_protocol,
                "memory_navigation": namespace.memory_navigation,
                "digest_cache_reuse": namespace.digest_cache_reuse,
                "world_args": world_args,
                "domain_focus": _serialize(profile.focus),
                "profile_snapshot": profile.model_dump(mode="json"),
            }
            record = await runs.start(
                RunSpec(
                    thread_id=namespace.thread_id,
                    runtime_db=Path(namespace.runtime_db),
                    config=snapshot,
                )
            )
            return JSONResponse(_run_view(record), status_code=202)
        except KeyError as error:
            return _error(str(error), 404)
        except RunAlreadyActiveError as error:
            return _error(str(error), 409)
        except (ValidationError, ValueError, SystemExit) as error:
            return _error(str(error), 422)

    async def run_item(request: Request) -> Response:
        run_id = request.path_params["run_id"]
        record = await runs.get(run_id)
        if record is None:
            return _error(f"unknown run: {run_id}", 404)
        metrics = await runs.metrics(run_id)
        view = _run_view(record)
        inspection: dict[str, object] = {}
        config = view.get("config_snapshot")
        if isinstance(config, dict):
            profile_id = config.get("profile_id")
            if isinstance(profile_id, str):
                try:
                    reader = inspection_for(profiles.get(profile_id))
                    result = view.get("result_summary")
                    report = result.get("run_report") if isinstance(result, dict) else None
                    wake_id = _result_wake_id(result if isinstance(result, dict) else {})
                    raw = reader.run_inspection(record.thread_id, wake_id=wake_id)
                    inspection = _run_inspection_view(
                        raw,
                        report,
                        result if isinstance(result, dict) else None,
                    )
                    view["outcome"] = _outcome_view(
                        str(view.get("status", "")),
                        result if isinstance(result, dict) else {},
                        inspection,
                    )
                except KeyError:
                    inspection = {}
        return JSONResponse(
            {
                **view,
                "metrics": _serialize(metrics),
                "events": _serialize(runs.list_events(run_id=run_id)),
                "inspection": inspection,
            }
        )

    async def pending_wakes(request: Request) -> Response:
        """List wakes whose staging a finalize attempt can still publish."""
        try:
            reader = inspection_for(profiles.get(request.path_params["profile_id"]))
            return JSONResponse({"wakes": reader.pending_wakes()})
        except KeyError as error:
            return _error(str(error), 404)
        except ValueError as error:
            return _error(str(error), 422)

    async def finalize_pending_wake(request: Request) -> Response:
        """Deterministically publish one pending wake (idempotent, no model).

        Reuses the same deterministic finalize entry as the CLI management
        flag, so the console can never diverge from it: the wake's active
        staging is validated and published in one transaction, and a repeat
        call replays the stored receipt instead of writing again. Unknown
        wakes and missing worlds answer 404; every other outcome — including
        blocked, compile_failed or nothing_to_finalize — is the finalize
        machinery's authoritative report and answers 200.
        """
        try:
            profile = profiles.get(request.path_params["profile_id"])
        except KeyError as error:
            return _error(str(error), 404)
        wake_id = request.path_params["wake_id"]
        if not wake_id or len(wake_id) > 200:
            return _error("wake_id must be a non-empty string of at most 200 characters", 422)
        world_db = (workspace_root / profile.world_db).resolve()
        try:
            outcome = await graph_shell_finalize_wake(world_db, wake_id)
        except (OSError, sqlite3.Error) as error:
            return _error(f"finalize failed: {type(error).__name__}", 422)
        if outcome.get("status") in {"no_world", "wake_unknown"}:
            return JSONResponse(outcome, status_code=404)
        return JSONResponse(_serialize(outcome))

    async def abandon_pending_wake(request: Request) -> Response:
        """Discard one pending wake's staging and release its writer lease.

        Reuses the same deterministic abandon entry as the CLI management
        flag, so the console can never diverge from it: the wake's active
        staging is marked abandoned and the singleton writer lease released
        in one world transaction (fail closed on owner mismatch), with
        idempotent runtime claim cleanup. Unknown wakes and missing worlds
        answer 404; an owner mismatch answers 422; a repeat call reports
        ``already_abandoned`` instead of failing.
        """
        try:
            profile = profiles.get(request.path_params["profile_id"])
        except KeyError as error:
            return _error(str(error), 404)
        wake_id = request.path_params["wake_id"]
        if not wake_id or len(wake_id) > 200:
            return _error("wake_id must be a non-empty string of at most 200 characters", 422)
        world_db = (workspace_root / profile.world_db).resolve()
        runtime_db = (workspace_root / profile.runtime_db).resolve()
        try:
            outcome = await graph_shell_abandon(world_db, runtime_db, wake_id)
        except ValueError as error:
            return _error(str(error), 422)
        except (OSError, sqlite3.Error) as error:
            return _error(f"abandon failed: {type(error).__name__}", 422)
        if outcome.get("status") == "no_world":
            return JSONResponse(outcome, status_code=404)
        return JSONResponse(_serialize(outcome))

    routes = [
        Route("/", index),
        Route("/api/system", system),
        Route("/api/local-settings", local_settings, methods=["GET", "PUT"]),
        Route("/api/profiles/quick", profile_quick_create, methods=["POST"]),
        Route("/api/profiles", profile_collection, methods=["GET", "POST"]),
        Route("/api/profiles/{profile_id}/clone", profile_clone, methods=["POST"]),
        Route("/api/profiles/{profile_id}/memory", memory_summary, methods=["GET"]),
        Route("/api/profiles/{profile_id}/memory/graph", memory_graph, methods=["GET"]),
        Route("/api/profiles/{profile_id}/memory/objects", memory_objects, methods=["GET"]),
        Route(
            "/api/profiles/{profile_id}/memory/objects/{object_id}",
            memory_object_detail,
            methods=["GET"],
        ),
        Route("/api/profiles/{profile_id}/memory/inquiries", memory_inquiries, methods=["GET"]),
        Route(
            "/api/profiles/{profile_id}/pending-wakes",
            pending_wakes,
            methods=["GET"],
        ),
        Route(
            "/api/profiles/{profile_id}/pending-wakes/{wake_id}/finalize",
            finalize_pending_wake,
            methods=["POST"],
        ),
        Route(
            "/api/profiles/{profile_id}/pending-wakes/{wake_id}/abandon",
            abandon_pending_wake,
            methods=["POST"],
        ),
        Route("/api/profiles/{profile_id}", profile_item, methods=["GET", "PUT"]),
        Route("/api/profiles/{profile_id}/prompt-preview", prompt_preview, methods=["POST"]),
        Route("/api/runs", run_collection, methods=["GET", "POST"]),
        Route("/api/runs/{run_id}", run_item, methods=["GET"]),
        Mount("/static", app=_NoStoreStaticFiles(directory=_STATIC_DIR), name="static"),
    ]
    return Starlette(debug=False, routes=routes, lifespan=lifespan)


def _namespace_arguments(namespace: argparse.Namespace) -> list[str]:
    # G5b-2: the wake protocol / memory navigation / digest-cache flags are
    # retired; new snapshots must not persist them (old snapshots that carry
    # them still replay, because cli.parse_args accepts them as no-ops).
    arguments = [
        "--thread-id",
        namespace.thread_id,
        "--world-db",
        str(namespace.world_db),
        "--runtime-db",
        str(namespace.runtime_db),
        "--mode",
        namespace.mode,
        "--max-turns",
        str(namespace.max_turns),
        "--adapters",
        namespace.adapters,
    ]
    if getattr(namespace, "adapter_defaults", None):
        arguments.extend(("--adapter-defaults", json.dumps(namespace.adapter_defaults)))
    if namespace.perspective:
        arguments.extend(("--perspective", namespace.perspective))
    if namespace.object_id:
        arguments.extend(("--object-id", namespace.object_id))
    if namespace.max_cost_usd is not None:
        arguments.extend(("--max-cost-usd", str(namespace.max_cost_usd)))
    if namespace.thinking:
        arguments.append("--thinking")
    if namespace.reasoning_effort:
        arguments.extend(("--reasoning-effort", namespace.reasoning_effort))
    return arguments


def _parse_world_arguments(arguments: list[str]) -> argparse.Namespace:
    """Turn a validated snapshot into the CLI namespace without exiting the server."""
    try:
        return parse_world_args(arguments)
    except SystemExit as error:
        raise ValueError("saved run configuration is invalid") from error


def _run_view(record: object) -> dict[str, object]:
    view = _serialize(record)
    assert isinstance(view, dict)
    config = view.get("config_snapshot")
    if isinstance(config, dict):
        view["profile_id"] = config.get("profile_id")
    result = view.get("result_summary")
    view["outcome"] = _outcome_view(str(view.get("status", "")), result if isinstance(result, dict) else {})
    return view


def _result_wake_id(result: Mapping[str, object]) -> str | None:
    """Recover the Graph Shell wake identity from live or restored results."""
    direct = result.get("wake_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    finalize = result.get("finalize_receipt")
    if isinstance(finalize, dict):
        wake_id = finalize.get("wake_id")
        if isinstance(wake_id, str) and wake_id.strip():
            return wake_id.strip()
    report = result.get("run_report")
    entry = report.get("entry") if isinstance(report, dict) else None
    wake_id = entry.get("wake_id") if isinstance(entry, dict) else None
    return wake_id.strip() if isinstance(wake_id, str) and wake_id.strip() else None


def _profile_view(profile: AgentProfile, workspace_root: Path) -> dict[str, object]:
    """Add small read-only memory counts for the Agent lobby and header."""
    view = _serialize(profile)
    assert isinstance(view, dict)
    view["memory"] = _memory_counts((workspace_root / profile.world_db).resolve())
    return view


def _memory_counts(world_db: Path) -> dict[str, int]:
    """Read stable headline counts without creating or mutating a database."""
    counts = {"objects": 0, "assertions": 0, "inquiries": 0, "commits": 0}
    if not world_db.is_file():
        return counts
    try:
        connection = sqlite3.connect(f"file:{world_db.as_posix()}?mode=ro", uri=True)
        try:
            tables = {
                str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            for key, table in (
                ("objects", "objects"),
                ("assertions", "assertions"),
                ("inquiries", "inquiries"),
                ("commits", "world_audit"),
            ):
                if table in tables:
                    counts[key] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return {"objects": 0, "assertions": 0, "inquiries": 0, "commits": 0}
    return counts


def _outcome_view(
    status: str,
    result: Mapping[str, object],
    inspection: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Separate lifecycle completion from the durable publication authority."""
    receipt = result.get("commit_receipt")
    commit = receipt.get("commit", {}) if isinstance(receipt, dict) else {}
    review_outcome = receipt.get("review_outcome") if isinstance(receipt, dict) else None
    counts = {
        "objects": len(commit.get("object_ids", [])) if isinstance(commit, dict) else 0,
        "assertions": len(commit.get("assertion_ids", [])) if isinstance(commit, dict) else 0,
        "inquiries": len(commit.get("inquiry_ids", [])) if isinstance(commit, dict) else 0,
    }
    finalize = result.get("finalize_receipt")
    finalize = finalize if isinstance(finalize, dict) else {}
    finalize_status = str(finalize.get("status") or result.get("finalize_status") or "")
    finalize_stats = finalize.get("stats")
    finalize_stats = finalize_stats if isinstance(finalize_stats, dict) else {}
    if finalize_stats:
        counts = {
            "objects": int(finalize_stats.get("objects_created", 0) or 0)
            + int(finalize_stats.get("objects_updated", 0) or 0),
            "assertions": int(finalize_stats.get("assertions_created", 0) or 0),
            "inquiries": int(finalize_stats.get("inquiries_created", 0) or 0),
        }
    report = result.get("run_report")
    report = report if isinstance(report, dict) else {}
    durable_diff = report.get("durable_diff")
    durable_diff = durable_diff if isinstance(durable_diff, dict) else {}
    if durable_diff:
        counts = {
            key: int(durable_diff.get(key, counts[key]) or 0)
            for key in ("objects", "assertions", "inquiries")
        }
    if inspection:
        inspected = inspection.get("durable")
        inspected = inspected if isinstance(inspected, dict) else {}
        if inspected:
            counts = {
                key: int(inspected.get(key, counts[key]) or 0)
                for key in ("objects", "assertions", "inquiries")
            }
    publication = report.get("publication")
    publication = publication if isinstance(publication, dict) else {}
    terminal_summary = str(result.get("terminal_summary") or "")
    terminal_status = str(result.get("terminal_status") or "")
    durable = (
        bool(commit)
        or bool(receipt)
        or bool(finalize.get("commit_id"))
        or finalize_status in {"published", "already_published"}
        or terminal_status in {"published", "already_published"}
        or publication.get("published") is True
        or any(counts.values())
    )
    finalize_warnings = finalize.get("warnings")
    has_warnings = isinstance(finalize_warnings, list) and bool(finalize_warnings)
    amendment_failed = "amendment failed" in terminal_summary.casefold()
    if status == "failed":
        level, label = "error", "运行失败"
    elif status in {"queued", "running"}:
        level, label = "active", "正在运行"
    elif durable and (review_outcome == "COMMIT_WITH_WARNINGS" or amendment_failed or has_warnings):
        level, label = "warning", "已完成 · 已写入（有提示）"
    elif durable:
        level, label = "success", "已完成 · 已写入"
    else:
        level, label = "neutral", "已完成 · 无持久化变更"
    return {
        "level": level,
        "label": label,
        "durable": durable,
        "review_outcome": review_outcome,
        "last_phase": result.get("phase"),
        "amendment_failed": amendment_failed,
        "message": terminal_summary,
        "written": counts,
    }


async def _runtime_run_records(profile: AgentProfile, workspace_root: Path) -> list[RunRecord]:
    """Recover completed console runs from LangGraph checkpoints after a restart."""
    runtime_path = (workspace_root / profile.runtime_db).resolve()
    if not runtime_path.is_file():
        return []
    try:
        async with aiosqlite.connect(runtime_path, isolation_level=None) as connection:
            rows = await (
                await connection.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'console-%'"
                )
            ).fetchall()
            saver = AsyncSqliteSaver(connection)
            records: list[RunRecord] = []
            for (thread_id,) in rows:
                checkpoint = await saver.aget_tuple({"configurable": {"thread_id": str(thread_id)}})
                if checkpoint is None:
                    continue
                values = checkpoint.checkpoint.get("channel_values", {})
                if not isinstance(values, dict) or not str(values.get("terminal_summary") or "").strip():
                    continue
                result = _result_summary(values)
                commit_receipt = result.get("commit_receipt")
                commit = commit_receipt.get("commit", {}) if isinstance(commit_receipt, dict) else {}
                finalize_receipt = result.get("finalize_receipt")
                finalize_receipt = finalize_receipt if isinstance(finalize_receipt, dict) else {}
                committed_at = (
                    commit.get("committed_at") if isinstance(commit, dict) else None
                ) or finalize_receipt.get("committed_at")
                ended_at = _parse_datetime(committed_at) or datetime.now().astimezone()
                mode = _infer_thread_mode(runtime_path, str(thread_id))
                records.append(
                    RunRecord(
                        run_id=f"restored-{str(thread_id)}",
                        thread_id=str(thread_id),
                        config_snapshot={
                            "profile_id": profile.id,
                            "mode": mode,
                            "restored": True,
                        },
                        runtime_db=runtime_path,
                        status="succeeded",
                        queued_at=ended_at,
                        started_at=ended_at,
                        ended_at=ended_at,
                        result_summary=result,
                    )
                )
            return records
    except (OSError, sqlite3.Error):
        return []


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _infer_thread_mode(runtime_path: Path, thread_id: str) -> str:
    """Use the model ledger as a conservative hint; old rows lack explicit mode."""
    try:
        connection = sqlite3.connect(f"file:{runtime_path.as_posix()}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT purpose FROM model_calls WHERE thread_id = ? ORDER BY id LIMIT 1",
                (thread_id,),
            ).fetchone()
            return "broad" if row is not None else "unknown"
        finally:
            connection.close()
    except sqlite3.Error:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately loopback-only console options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open the system browser.")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("bubble-console is local-only; host must be loopback")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    return args


def main() -> None:
    """Run the personal console without enabling network-facing deployment."""
    args = parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_open:
        webbrowser.open(url)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
