"""Offline HTTP contracts for the personal Agent console."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from leave_information_bubble.console.app import (
    CreateRunRequest,
    _default_runner,
    _namespace_arguments,
    _outcome_view,
    _parse_world_arguments,
    _result_summary,
    _serialize,
    _world_namespace,
    create_app,
    parse_args,
)
from leave_information_bubble.console.profiles import AgentProfileRegistry
from leave_information_bubble.console.runs import RunManager, RunSpec


def test_graph_shell_summary_keeps_durable_receipt_for_restored_run_outcome() -> None:
    summary = _result_summary(
        {
            "terminal_summary": "published",
            "terminal_status": "published",
            "wake_id": "wake-one",
            "finalize_status": "published",
            "finalize_receipt": {
                "wake_id": "wake-one",
                "status": "published",
                "commit_id": "wake-one:finalize",
                "committed_at": "2026-08-24T11:27:15+00:00",
                "stats": {
                    "objects_created": 7,
                    "objects_updated": 1,
                    "assertions_created": 28,
                    "inquiries_created": 1,
                },
                "warnings": [],
            },
        }
    )

    assert summary["wake_id"] == "wake-one"
    assert summary["finalize_receipt"]["commit_id"] == "wake-one:finalize"
    assert _outcome_view("succeeded", summary)["written"] == {
        "objects": 8,
        "assertions": 28,
        "inquiries": 1,
    }
    assert _outcome_view("succeeded", summary)["label"] == "已完成 · 已写入"


def test_console_html_and_static_assets_disable_browser_caching(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")

    async def runner(_: RunSpec) -> None:
        return None

    with TestClient(create_app(registry=registry, manager=RunManager(runner))) as client:
        index = client.get("/")
        stylesheet = client.get("/static/app.css?v=20260824-5")

    assert index.headers["cache-control"] == "no-store, max-age=0"
    assert stylesheet.headers["cache-control"] == "no-store, max-age=0"
    assert "app.css?v=20260824-5" in index.text


def test_console_bootstraps_profile_and_previews_prompt(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")

    async def runner(_: RunSpec) -> None:
        return None

    with TestClient(create_app(registry=registry, manager=RunManager(runner))) as client:
        profiles = client.get("/api/profiles")
        assert profiles.status_code == 200
        assert profiles.json()[0]["id"] == "lol_cn"
        assert profiles.json()[0]["defaults"]["broad_perspective"] is None
        assert profiles.json()[0]["defaults"]["deep_perspective"] is None
        assert set(profiles.json()[0]["memory"]) == {"objects", "assertions", "inquiries", "commits"}

        preview = client.post(
            "/api/profiles/lol_cn/prompt-preview",
            json={
                "mode": "deep",
                "object_id": "seed-1",
                "wake_protocol": "separated",
                "perspective": "  follow an unexpected roster claim  ",
            },
        )
        assert preview.status_code == 200
        assert "operator-provided possible entry is seed-1" in preview.json()["posture"]
        # G5b-2: the retired wake_protocol field is accepted and neutralized —
        # the preview always compiles the Graph Shell prompt surface
        assert "Publication is always an explicit model decision" in preview.json()["mechanics"]
        assert "separate submit-only closing" not in preview.json()["mechanics"]
        assert "Familiar vantage points" in preview.json()["compiled"]
        assert "LPL 赛事与战队" in preview.json()["compiled"]
        assert preview.json()["wake_input"] == (
            "Wake perspective for this run:\n"
            "follow an unexpected roster claim\n\n"
            "Use this as the operator's intended attention for this wake within the configured "
            "observation center and all system, tool, evidence, and persistence boundaries. "
            "Let it guide what you prioritize, deprioritize, and treat as relevant; do not turn "
            "it into a fixed route, required conclusion, or coverage quota."
        )


def test_console_run_api_enforces_single_writer(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    release = asyncio.Event()

    async def runner(_: RunSpec) -> dict[str, str]:
        await release.wait()
        return {"terminal_summary": "done"}

    with TestClient(create_app(registry=registry, manager=RunManager(runner))) as client:
        request = {"profile_id": "lol_cn", "perspective": "offline angle", "max_turns": 2}
        first = client.post("/api/runs", json=request)
        assert first.status_code == 202
        second = client.post("/api/runs", json=request)
        assert second.status_code == 409
        detail = client.get(f"/api/runs/{first.json()['run_id']}")
        assert detail.status_code == 200
        assert detail.json()["profile_id"] == "lol_cn"
        assert detail.json()["config_snapshot"]["perspective"] == "offline angle"
        assert "mission" not in detail.json()["config_snapshot"]
        assert detail.json()["config_snapshot"]["world_args"][-2:] == [
            "--perspective",
            "offline angle",
        ]
        release.set()


def test_console_run_api_accepts_legacy_mission_and_neutral_empty_wake(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    release = asyncio.Event()

    async def runner(_: RunSpec) -> dict[str, str]:
        await release.wait()
        return {"terminal_summary": "done"}

    with TestClient(create_app(registry=registry, manager=RunManager(runner))) as client:
        legacy = client.post(
            "/api/runs",
            json={"profile_id": "lol_cn", "mission": "legacy angle", "max_turns": 2},
        )
        assert legacy.status_code == 202
        assert legacy.json()["config_snapshot"]["perspective"] == "legacy angle"
        assert "--mission" not in legacy.json()["config_snapshot"]["world_args"]
        release.set()

    neutral_registry = AgentProfileRegistry(tmp_path / "neutral-profiles")
    neutral_release = asyncio.Event()

    async def neutral_runner(_: RunSpec) -> dict[str, str]:
        await neutral_release.wait()
        return {"terminal_summary": "done"}

    with TestClient(create_app(registry=neutral_registry, manager=RunManager(neutral_runner))) as client:
        neutral = client.post(
            "/api/runs",
            json={"profile_id": "lol_cn", "perspective": "   ", "max_turns": 2},
        )
        assert neutral.status_code == 202
        snapshot = neutral.json()["config_snapshot"]
        assert snapshot["perspective"] is None
        assert "--perspective" not in snapshot["world_args"]
        assert "--mission" not in snapshot["world_args"]
        neutral_release.set()


def test_console_rejects_competing_perspective_and_legacy_mission() -> None:
    response = CreateRunRequest.model_validate
    try:
        response({"profile_id": "lol_cn", "perspective": "new", "mission": "old"})
    except ValueError as error:
        assert "cannot both be provided" in str(error)
    else:
        raise AssertionError("competing perspective fields were accepted")


def test_console_cli_snapshot_only_adds_non_empty_perspective(tmp_path: Path) -> None:
    profile = AgentProfileRegistry(tmp_path / "profiles").bootstrap_lol_profile()
    with_perspective = _world_namespace(
        profile,
        CreateRunRequest(profile_id=profile.id, perspective="current angle"),
    )
    without_perspective = _world_namespace(
        profile,
        CreateRunRequest(profile_id=profile.id),
    )

    assert _namespace_arguments(with_perspective)[-2:] == ["--perspective", "current angle"]
    neutral_arguments = _namespace_arguments(without_perspective)
    assert "--perspective" not in neutral_arguments
    assert "--mission" not in neutral_arguments


def test_system_api_never_returns_secret(tmp_path: Path) -> None:
    async def runner(_: RunSpec) -> None:
        return None

    app = create_app(
        registry=AgentProfileRegistry(tmp_path / "profiles"),
        manager=RunManager(runner),
    )
    with TestClient(app) as client:
        response = client.get("/api/system")
    assert response.status_code == 200
    text = response.text.casefold()
    assert "key_configured" in text
    assert "deepseek_api_key" not in text


def test_memory_graph_api_degrades_to_an_empty_read_only_snapshot(tmp_path: Path) -> None:
    async def runner(_: RunSpec) -> None:
        return None

    # A profile whose world database does not exist anywhere must degrade
    # to an empty read-only snapshot instead of failing (the builtin
    # lol_cn profile points at the real data/agents/lol_cn store, which is
    # present in this workspace).
    registry = AgentProfileRegistry(tmp_path / "profiles")
    base = registry.bootstrap_lol_profile()
    profile = base.model_copy(
        update={
            "id": "phantom",
            "display_name": "Phantom",
            "world_db": "data/agents/phantom/world.sqlite3",
            "runtime_db": "data/agents/phantom/runtime.sqlite3",
        }
    )
    registry.create(profile)
    with TestClient(
        create_app(
            registry=registry,
            manager=RunManager(runner),
        )
    ) as client:
        response = client.get("/api/profiles/phantom/memory/graph?limit=8")

    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "nodes": [],
        "edges": [],
        "truncated": False,
    }


def test_local_settings_api_persists_allowlisted_secrets_without_returning_them(
    tmp_path: Path,
) -> None:
    async def runner(_: RunSpec) -> None:
        return None

    env_file = tmp_path / ".env"
    secret = "sk-test-value-that-must-not-leak"
    with TestClient(
        create_app(
            registry=AgentProfileRegistry(tmp_path / "profiles"),
            manager=RunManager(runner),
            settings_file=env_file,
        )
    ) as client:
        saved = client.put(
            "/api/local-settings",
            json={
                "deepseek_api_key": secret,
                "deepseek_model": "deepseek-v4",
                "nga_cookie": "uid=one; token=two",
            },
        )
        rejected = client.put("/api/local-settings", json={"arbitrary_secret": "no"})

    assert saved.status_code == 200
    assert secret not in saved.text
    assert saved.json() == {
        "provider": {
            "name": "DeepSeek",
            "model": "deepseek-v4",
            "base_url": "https://api.deepseek.com",
        },
        "configured": {"deepseek": True, "bilibili": False, "nga": True},
    }
    persisted = env_file.read_text(encoding="utf-8")
    assert f'DEEPSEEK_API_KEY="{secret}"' in persisted
    assert 'NGA_COOKIE="uid=one; token=two"' in persisted
    assert rejected.status_code == 422


def test_local_settings_validation_never_echoes_rejected_secret(tmp_path: Path) -> None:
    async def runner(_: RunSpec) -> None:
        return None

    env_file = tmp_path / ".env"
    canary = "secret-canary-must-not-return"
    with TestClient(
        create_app(
            registry=AgentProfileRegistry(tmp_path / "profiles"),
            manager=RunManager(runner),
            settings_file=env_file,
        )
    ) as client:
        response = client.put(
            "/api/local-settings",
            json={"deepseek_api_key": canary + ("x" * 20_000)},
        )

    assert response.status_code == 422
    assert canary not in response.text
    assert response.json() == {"detail": "invalid local settings: deepseek_api_key"}
    assert not env_file.exists()


def test_profile_get_view_can_be_put_back_with_operator_update(tmp_path: Path) -> None:
    """Catch derived memory counts making the browser's save-operator PUT fail."""

    async def runner(_: RunSpec) -> None:
        return None

    with TestClient(
        create_app(
            registry=AgentProfileRegistry(tmp_path / "profiles"),
            manager=RunManager(runner),
        )
    ) as client:
        profile = client.get("/api/profiles/lol_cn").json()
        assert "memory" in profile
        profile["operator_instructions"] = "Prefer primary sources."

        updated = client.put("/api/profiles/lol_cn", json=profile)

    assert updated.status_code == 200
    assert updated.json()["operator_instructions"] == "Prefer primary sources."


def test_profile_api_round_trips_search_experience(tmp_path: Path) -> None:
    async def runner(_: RunSpec) -> None:
        return None

    with TestClient(
        create_app(
            registry=AgentProfileRegistry(tmp_path / "profiles"),
            manager=RunManager(runner),
        )
    ) as client:
        profile = client.get("/api/profiles/lol_cn").json()
        profile["search_experience"] = ["Use current tournament names and post-match language."]
        saved = client.put("/api/profiles/lol_cn", json=profile)

    assert saved.status_code == 200
    assert saved.json()["search_experience"] == profile["search_experience"]


def test_profile_editor_wires_search_experience_as_newline_delimited_guidance() -> None:
    static_dir = Path(__file__).parents[2] / "src" / "leave_information_bubble" / "console" / "static"
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    javascript = (static_dir / "app.js").read_text(encoding="utf-8")

    assert 'textarea name="search_experience"' in html
    assert "(profile.search_experience || []).join('\\n')" in javascript
    assert "search_experience: []" in javascript
    assert "search_experience: toLines(raw.search_experience)" in javascript


def test_console_rejects_non_loopback_host() -> None:
    try:
        parse_args(["--host", "0.0.0.0"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("non-loopback host was accepted")


def test_console_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort must be high or max"):
        CreateRunRequest.model_validate({"profile_id": "lol_cn", "reasoning_effort": "medium"})


def test_profile_defaults_and_request_overrides_feed_thinking_controls(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    base = registry.bootstrap_lol_profile()

    # Neutral profile: thinking off, no effort — no flags are emitted.
    neutral = _world_namespace(base, CreateRunRequest(profile_id=base.id))
    neutral_arguments = _namespace_arguments(neutral)
    assert "--thinking" not in neutral_arguments
    assert "--reasoning-effort" not in neutral_arguments

    # Profile defaults with thinking on + effort flow through by themselves.
    tuned = base.model_copy(
        update={"defaults": base.defaults.model_copy(update={"thinking": True, "reasoning_effort": "high"})}
    )
    defaulted = _world_namespace(tuned, CreateRunRequest(profile_id=tuned.id))
    defaulted_arguments = _namespace_arguments(defaulted)
    assert "--thinking" in defaulted_arguments
    assert defaulted_arguments[-2:] == ["--reasoning-effort", "high"]

    # An explicit request override wins over profile defaults.
    overridden = _world_namespace(tuned, CreateRunRequest(profile_id=tuned.id, thinking=False))
    overridden_arguments = _namespace_arguments(overridden)
    assert "--thinking" not in overridden_arguments
    assert "--reasoning-effort" in overridden_arguments


def test_world_args_round_trip_preserves_thinking_controls(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    profile = registry.bootstrap_lol_profile()
    namespace = _world_namespace(
        profile,
        CreateRunRequest(profile_id=profile.id, thinking=True, reasoning_effort="max"),
    )
    arguments = _namespace_arguments(namespace)
    assert "--thinking" in arguments
    assert arguments[-2:] == ["--reasoning-effort", "max"]

    rebuilt = _parse_world_arguments(arguments)
    assert rebuilt.thinking is True
    assert rebuilt.reasoning_effort == "max"


def test_default_runner_rejects_damaged_snapshot(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    profile = registry.bootstrap_lol_profile()
    namespace = _world_namespace(profile, CreateRunRequest(profile_id=profile.id))
    config: dict[str, object] = {
        "world_args": _namespace_arguments(namespace),
        "domain_focus": _serialize(profile.focus),
        "profile_snapshot": profile.model_dump(mode="json"),
    }
    spec = RunSpec(
        thread_id=namespace.thread_id,
        runtime_db=Path(namespace.runtime_db),
        config=config,
    )

    async def publish(_kind: str, _message: str) -> None:
        return None

    # A drifted domain focus must fail closed before any world work starts.
    drifted = replace(spec, config={**config, "domain_focus": {"domain_key": "somewhere-else"}})
    with pytest.raises(ValueError, match="domain focus does not match"):
        asyncio.run(_default_runner(drifted, publish))

    # Malformed world arguments must surface as a validation error, not exit.
    malformed = replace(spec, config={**config, "world_args": ["--not-a-real-flag"]})
    with pytest.raises(ValueError, match="saved run configuration is invalid"):
        asyncio.run(_default_runner(malformed, publish))
