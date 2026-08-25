"""Failing-first HTTP contracts for the personal Agent observatory.

These tests deliberately exercise only JSON APIs.  They keep the observatory
read-only, profile-scoped, and separate from the world-agent's cognitive
control plane.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from starlette.testclient import TestClient

from leave_information_bubble.console.app import create_app
from leave_information_bubble.console.profiles import (
    AgentProfile,
    AgentProfileRegistry,
    RunDefaults,
)
from leave_information_bubble.console.runs import RunManager, RunSpec
from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    InquiryInput,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)
from leave_information_bubble.world_agent.model_calls import ModelCallRecorder

NOW = datetime(2026, 8, 14, 9, 0, tzinfo=UTC)


def _workspace_relative(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def _profile(
    tmp_path: Path,
    profile_id: str,
    *,
    defaults: RunDefaults | None = None,
    operator_instructions: str | None = None,
) -> AgentProfile:
    directory = tmp_path / profile_id
    return AgentProfile(
        id=profile_id,
        display_name=f"{profile_id} Agent",
        domain_key=f"{profile_id}_domain",
        observation_center=f"{profile_id} community",
        relevance_rule=f"Use {profile_id} as an observation center, not an information boundary.",
        attention_examples=("recent changes",),
        source_preferences=("public sources",),
        branches=(),
        locale="zh-CN",
        timezone="Asia/Shanghai",
        defaults=defaults or RunDefaults(adapters=("public-web",)),
        world_db=_workspace_relative(directory / "world.sqlite3"),
        runtime_db=_workspace_relative(directory / "runtime.sqlite3"),
        operator_instructions=operator_instructions,
    )


def _seed_world(profile: AgentProfile, *, object_id: str, name: str, commit_id: str) -> None:
    store = WorldStore(Path(profile.world_db))
    observation_id = f"observation-{object_id}"
    assertion_id = f"assertion-{object_id}"
    inquiry_id = f"inquiry-{object_id}"
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id=object_id,
                    kind=ObjectKind.ENTITY,
                    canonical_name=name,
                    aliases=[f"{name}别名"],
                )
            ],
            observations=[
                ObservationInput(
                    id=observation_id,
                    source_uri=f"https://example.test/{object_id}",
                    source_kind="web",
                    title=f"{name}来源标题",
                    excerpt=f"{name}的有界摘要",
                    content_ref=f"body://{observation_id}",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                )
            ],
            assertions=[
                AssertionInput(
                    id=assertion_id,
                    subject_id=object_id,
                    predicate="has_current_state",
                    literal=f"{name}当前状态",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id=observation_id, role="supports")],
                )
            ],
            inquiries=[
                InquiryInput(
                    id=inquiry_id,
                    subject_id=object_id,
                    prompt=f"{name}接下来会怎样？",
                    rationale="仍有一个值得保留的开放边缘。",
                    created_at=NOW,
                )
            ],
        ),
        commit_id,
    )


async def _empty_runner(_: RunSpec) -> None:
    return None


def _client(registry: AgentProfileRegistry, manager: RunManager | None = None) -> TestClient:
    return TestClient(create_app(registry=registry, manager=manager or RunManager(_empty_runner)))


def test_memory_summary_exposes_only_readable_cognition_and_recent_commits(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    profile = _profile(tmp_path, "alpha")
    registry.create(profile)
    _seed_world(profile, object_id="object-alpha", name="阿尔法对象", commit_id="agent:wake-alpha")

    with _client(registry) as client:
        response = client.get("/api/profiles/alpha/memory")

    assert response.status_code == 200
    payload = response.json()
    assert payload["initialized"] is True
    assert payload["summary"] == {
        "objects": 1,
        "current_assertions": 1,
        "open_inquiries": 1,
        "observations": 1,
        "cognition_commits": 1,
    }
    assert payload["object_samples"][0]["id"] == "object-alpha"
    # Compatibility alias is deliberately not a recency guarantee: the sample is alphabetical.
    assert payload["recent_objects"] == payload["object_samples"]
    assert payload["open_inquiries"][0]["id"] == "inquiry-object-alpha"
    assert payload["recent_commits"][0]["commit_id"] == "agent:wake-alpha"
    assert payload["recent_commits"][0]["counts"] == {
        "objects": 1,
        "assertions": 1,
        "inquiries": 1,
        "resolved_inquiries": 0,
    }


def test_memory_object_search_and_detail_are_bounded_and_evidence_aware(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    profile = _profile(tmp_path, "alpha")
    registry.create(profile)
    _seed_world(profile, object_id="object-alpha", name="阿尔法对象", commit_id="agent:wake-alpha")

    with _client(registry) as client:
        search = client.get(
            "/api/profiles/alpha/memory/objects",
            params={"query": "阿尔法", "limit": 20, "offset": 0},
        )
        detail = client.get("/api/profiles/alpha/memory/objects/object-alpha")

    assert search.status_code == 200
    result = search.json()
    assert result["items"][0]["id"] == "object-alpha"
    assert result["limit_applied"] == 20
    assert result["offset"] == 0
    assert result["has_more"] is False

    assert detail.status_code == 200
    payload = detail.json()
    assert payload["object"]["canonical_name"] == "阿尔法对象"
    assert payload["aliases"] == ["阿尔法对象别名"]
    assert payload["assertions"][0]["epistemic_role"] == "fact"
    assert payload["assertions"][0]["evidence"][0]["observation"]["title"] == "阿尔法对象来源标题"
    assert payload["inquiries"][0]["status"] == "open"
    observation = payload["observations"][0]
    assert observation["excerpt"] == "阿尔法对象的有界摘要"
    assert "body" not in observation
    assert "content" not in observation


def test_memory_inquiries_are_filterable_but_remain_observation_not_a_task_queue(
    tmp_path: Path,
) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    profile = _profile(tmp_path, "alpha")
    registry.create(profile)
    _seed_world(profile, object_id="object-alpha", name="阿尔法对象", commit_id="agent:wake-alpha")

    with _client(registry) as client:
        response = client.get(
            "/api/profiles/alpha/memory/inquiries",
            params={"status": "open", "limit": 10, "offset": 0},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["id"] == "inquiry-object-alpha"
    assert payload["items"][0]["subject"]["id"] == "object-alpha"
    assert payload["items"][0]["status"] == "open"
    assert payload["limit_applied"] == 10
    assert "priority" not in payload["items"][0]
    assert "recommended_next" not in payload
    assert "completion_rate" not in payload


def test_missing_memory_database_returns_empty_without_creating_either_database(
    tmp_path: Path,
) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    profile = _profile(tmp_path, "empty")
    registry.create(profile)
    world_path = Path(profile.world_db)
    runtime_path = Path(profile.runtime_db)
    assert not world_path.exists()
    assert not runtime_path.exists()

    with _client(registry) as client:
        summary = client.get("/api/profiles/empty/memory")
        objects = client.get("/api/profiles/empty/memory/objects")
        inquiries = client.get("/api/profiles/empty/memory/inquiries")
        missing_detail = client.get("/api/profiles/empty/memory/objects/not-there")

    assert summary.status_code == 200
    assert summary.json()["initialized"] is False
    assert summary.json()["summary"] == {
        "objects": 0,
        "current_assertions": 0,
        "open_inquiries": 0,
        "observations": 0,
        "cognition_commits": 0,
    }
    assert objects.status_code == 200
    assert objects.json()["items"] == []
    assert inquiries.status_code == 200
    assert inquiries.json()["items"] == []
    assert missing_detail.status_code == 404
    assert not world_path.exists()
    assert not runtime_path.exists()


def test_memory_routes_are_profile_scoped_and_unknown_profiles_fail_closed(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    alpha = _profile(tmp_path, "alpha")
    beta = _profile(tmp_path, "beta")
    registry.create(alpha)
    registry.create(beta)
    _seed_world(alpha, object_id="object-alpha", name="阿尔法对象", commit_id="agent:wake-alpha")
    _seed_world(beta, object_id="object-beta", name="贝塔对象", commit_id="agent:wake-beta")

    with _client(registry) as client:
        alpha_items = client.get("/api/profiles/alpha/memory/objects").json()["items"]
        beta_items = client.get("/api/profiles/beta/memory/objects").json()["items"]
        cross_profile = client.get("/api/profiles/beta/memory/objects/object-alpha")
        unknown_profile = client.get("/api/profiles/not-registered/memory")

    assert {item["id"] for item in alpha_items} == {"object-alpha"}
    assert {item["id"] for item in beta_items} == {"object-beta"}
    assert cross_profile.status_code == 404
    assert unknown_profile.status_code == 404


def test_quick_create_builds_a_stable_profile_but_defers_database_initialization(
    tmp_path: Path,
) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    agent_id = f"quick_{tmp_path.name[-8:]}"

    with _client(registry) as client:
        response = client.post(
            "/api/profiles/quick",
            json={
                "id": agent_id,
                "display_name": "独立游戏观察员",
                "observation_center": "中文独立游戏社区",
                "locale": "zh-CN",
                "timezone": "Asia/Shanghai",
                "adapters": ["public-web", "bilibili"],
            },
        )

    assert response.status_code == 201
    profile = response.json()
    assert profile["domain_key"] == agent_id
    assert profile["branches"] == []
    assert profile["defaults"]["wake_protocol"] == "current"
    assert profile["defaults"]["memory_navigation"] == "overview_v1"
    assert profile["defaults"]["digest_cache_reuse"] is False
    assert profile["defaults"]["broad_perspective"] is None
    assert profile["defaults"]["deep_perspective"] is None
    assert profile["world_db"] == f"data/agents/{agent_id}/world.sqlite3"
    assert profile["runtime_db"] == f"data/agents/{agent_id}/runtime.sqlite3"
    assert not Path(profile["world_db"]).exists()
    assert not Path(profile["runtime_db"]).exists()


def test_clone_copies_design_but_not_memory_operator_or_experimental_switches(
    tmp_path: Path,
) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    source_defaults = RunDefaults(
        broad_perspective="old broad pull",
        deep_perspective="old deep pull",
        adapters=("public-web",),
        wake_protocol="separated",
        memory_navigation="overview_v1",
        digest_cache_reuse=True,
    )
    source = _profile(
        tmp_path,
        "source",
        defaults=source_defaults,
        operator_instructions="old private preference",
    )
    registry.create(source)
    _seed_world(source, object_id="object-source", name="源对象", commit_id="agent:wake-source")
    clone_id = f"clone_{tmp_path.name[-8:]}"

    with _client(registry) as client:
        response = client.post(
            "/api/profiles/source/clone",
            json={"id": clone_id, "display_name": "复制后的观察员"},
        )

    assert response.status_code == 201
    cloned = response.json()
    assert cloned["observation_center"] == source.observation_center
    assert cloned["locale"] == source.locale
    assert cloned["operator_instructions"] is None
    assert cloned["defaults"]["wake_protocol"] == "current"
    assert cloned["defaults"]["memory_navigation"] == "overview_v1"
    assert cloned["defaults"]["digest_cache_reuse"] is False
    assert cloned["defaults"]["broad_perspective"] is None
    assert cloned["defaults"]["deep_perspective"] is None
    assert cloned["world_db"] == f"data/agents/{clone_id}/world.sqlite3"
    assert cloned["runtime_db"] == f"data/agents/{clone_id}/runtime.sqlite3"
    assert not Path(cloned["world_db"]).exists()
    assert not Path(cloned["runtime_db"]).exists()


def test_run_inspection_joins_report_ledgers_and_exact_committed_rows(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "profiles")
    profile = _profile(tmp_path, "alpha")
    registry.create(profile)
    thread_id = "console-observatory-contract"
    commit_id = f"agent:{thread_id}"
    _seed_world(profile, object_id="object-alpha", name="阿尔法对象", commit_id=commit_id)

    ModelCallRecorder(Path(profile.runtime_db))
    with sqlite3.connect(profile.runtime_db) as connection:
        connection.execute(
            "INSERT INTO model_calls(thread_id, turn, purpose, wake_protocol, phase, model, "
            "prompt_tokens, cached_input_tokens, uncached_input_tokens, completion_tokens, "
            "cost_usd, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (thread_id, 1, "explore", "current", "exploration", "offline-model", 100, 25, 75, 20, 0.01, 50.0),
        )
    with sqlite3.connect(profile.world_db) as connection:
        connection.execute(
            "INSERT INTO proposal_attempts(commit_id, thread_id, attempted_at, outcome, new_objects, "
            "assertions, inquiries, omitted_assertions, omitted_inquiries, omitted_resolutions, "
            "resolved_inquiries, evidence_missing_assertions, error, delta_json, issues_json) "
            "VALUES (?, ?, ?, 'committed', 1, 1, 1, 0, 0, 0, 0, 0, NULL, '{}', '[]')",
            (commit_id, thread_id, NOW.isoformat()),
        )

    async def runner(_: RunSpec) -> dict[str, object]:
        return {
            "terminal_summary": "terminal commit completed",
            "turn_count": 1,
            "total_cost_usd": 0.01,
            "halted": False,
            "phase": "commit",
            "transition_reason": "terminal_commit",
            "run_report": {
                "execution": {"status": "complete", "turns": 1, "stop_reason": "terminal_commit"},
                "model": {"successful_calls": 1, "by_purpose": {"explore": 1}},
                "tools": {
                    "total": 1,
                    "failed": 0,
                    "limited": 0,
                    "by_name": {"discover_sources": 1},
                    "diagnostics": [
                        {
                            "call_id": "discover-1",
                            "name": "discover_sources",
                            "ok": True,
                            "limitations": [],
                            "outcome": "success",
                            "scope": {"window_applied": True, "precision": "exact"},
                            "completeness": {"returned": 5, "limit": 5, "truncated": False},
                        }
                    ],
                    "diagnostics_truncated": False,
                },
                "review": {"attempts": 1, "outcome": "committed", "omitted": {}},
                "durable_diff": {"objects": 1, "assertions": 1, "inquiries": 1},
            },
        }

    manager = RunManager(runner)
    with _client(registry, manager) as client:
        started = client.post(
            "/api/runs",
            json={"profile_id": "alpha", "thread_id": thread_id, "max_turns": 2},
        )
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        detail = None
        for _ in range(50):
            detail = client.get(f"/api/runs/{run_id}")
            if detail.json()["status"] == "succeeded":
                break
            asyncio.run(asyncio.sleep(0.01))

    assert detail is not None
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["outcome"] == {
        "level": "success",
        "label": "已完成 · 已写入",
        "durable": True,
        "review_outcome": None,
        "last_phase": "commit",
        "amendment_failed": False,
        "message": "terminal commit completed",
        "written": {"objects": 1, "assertions": 1, "inquiries": 1},
    }
    inspection = payload["inspection"]
    assert inspection["durable"] == {"objects": 1, "assertions": 1, "inquiries": 1}
    assert inspection["execution"]["stop_reason"] == "terminal_commit"
    assert inspection["model"]["successful_calls"] == 1
    assert inspection["model"]["by_purpose"] == {"explore": 1}
    assert inspection["tools"]["by_name"] == {"discover_sources": 1}
    assert inspection["tools"]["diagnostics"][0] == {
        "call_id": "discover-1",
        "name": "discover_sources",
        "ok": True,
        "limitations": [],
        "outcome": "success",
        "scope": {"window_applied": True, "precision": "exact"},
        "completeness": {"returned": 5, "limit": 5, "truncated": False},
    }
    assert inspection["review"]["attempts"] == 1
    assert inspection["writes"]["objects"][0]["id"] == "object-alpha"
    assert inspection["writes"]["assertions"][0]["id"] == "assertion-object-alpha"
    assert inspection["writes"]["assertions"][0]["literal"] == "阿尔法对象当前状态"
    assert inspection["writes"]["inquiries"][0]["id"] == "inquiry-object-alpha"
    assert inspection["writes"]["evidence_links"][0]["observation"]["id"] == ("observation-object-alpha")
    assert "hidden_reasoning" not in inspection
