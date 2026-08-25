"""Offline HTTP contracts for pending-wake listing and manual finalize.

A wake is "pending" exactly when the deterministic finalize entry would
still publish it: it has a world-scoped writer claim, at least one active
staged row, and no published receipt.  The list must never promise what the
finalize POST cannot deliver (claim-less rows belong to restore, published
wakes are excluded even with later staging), and the POST must be
idempotent: a repeat call replays the stored receipt.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from starlette.testclient import TestClient

from leave_information_bubble.console.app import create_app
from leave_information_bubble.console.profiles import (
    AgentProfile,
    AgentProfileRegistry,
    RunDefaults,
)
from leave_information_bubble.console.runs import RunManager, RunSpec
from leave_information_bubble.world import WorldStore, WorldTools


def _workspace_relative(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def _profile(tmp_path: Path, profile_id: str) -> AgentProfile:
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
        defaults=RunDefaults(adapters=("public-web",)),
        world_db=_workspace_relative(directory / "world.sqlite3"),
        runtime_db=_workspace_relative(directory / "runtime.sqlite3"),
    )


def _registry(tmp_path: Path) -> AgentProfileRegistry:
    return AgentProfileRegistry(tmp_path / "profiles")


def _stage_object(world_db: Path, wake_id: str, op_id: str, name: str) -> None:
    """Stage one provisional concept the same way a halted wake does."""
    store = WorldStore(world_db)
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="console-pending-thread",
        wake_id=wake_id,
    )
    result = asyncio.run(
        tools.execute(
            "graph_patch",
            {
                "items": [
                    {
                        "op_id": op_id,
                        "kind": "object",
                        "action": "create",
                        "payload": {
                            "canonical_name": name,
                            "kind": "concept",
                            "provisional": True,
                        },
                    }
                ]
            },
            f"call-{op_id}",
        )
    )
    assert result["ok"] is True


def _claim_wake(world_db: Path, wake_id: str, thread_id: str) -> None:
    with sqlite3.connect(world_db) as connection:
        connection.execute(
            "INSERT INTO graph_shell_wake_claims (wake_id, thread_id, "
            "runtime_store_identity, claimed_at) VALUES (?, ?, ?, ?)",
            (wake_id, thread_id, "console-runtime", 1.0),
        )


def _lease_wake(world_db: Path, wake_id: str, thread_id: str) -> None:
    """Give *wake_id* the world's singleton writer lease, like a live wake."""
    with sqlite3.connect(world_db) as connection:
        connection.execute(
            "INSERT INTO graph_shell_writer_leases VALUES (1, ?, ?, 123456.0)",
            (wake_id, thread_id),
        )


def _client(registry: AgentProfileRegistry) -> TestClient:
    async def runner(_: RunSpec) -> None:
        return None

    return TestClient(create_app(registry=registry, manager=RunManager(runner)))


def _setup_pending(tmp_path: Path, profile_id: str, wake_id: str) -> Path:
    registry = _registry(tmp_path)
    profile = _profile(tmp_path, profile_id)
    registry.create(profile)
    world_db = Path(profile.world_db).resolve()
    _stage_object(world_db, wake_id, "op-1", "Pending Probe")
    _claim_wake(world_db, wake_id, f"{profile_id}-thread")
    return world_db


def test_pending_wakes_lists_only_finalizeable_unpublished_wakes(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    profile = _profile(tmp_path, "alpha")
    registry.create(profile)
    world_db = Path(profile.world_db).resolve()
    _stage_object(world_db, "wake-a", "op-1", "First Probe")
    _claim_wake(world_db, "wake-a", "alpha-thread")
    # Abandoned staging without a claim is a restore case, never a finalize
    # candidate: the deterministic entry fails closed on wake_unknown.
    _stage_object(world_db, "wake-orphan", "op-2", "Orphaned Probe")

    with _client(registry) as client:
        response = client.get("/api/profiles/alpha/pending-wakes")

    assert response.status_code == 200
    assert response.json() == {
        "wakes": [
            {
                "wake_id": "wake-a",
                "staging": {
                    "staged_objects": 1,
                    "staged_assertions": 0,
                    "staged_inquiries": 0,
                },
                "staging_total": 1,
                "claimed_by": "alpha-thread",
            }
        ]
    }


def test_pending_wakes_degrades_to_empty_for_missing_store(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    profile = _profile(tmp_path, "phantom").model_copy(
        update={"id": "phantom", "display_name": "Phantom"}
    )
    registry.create(profile)

    with _client(registry) as client:
        response = client.get("/api/profiles/phantom/pending-wakes")

    assert response.status_code == 200
    assert response.json() == {"wakes": []}


def test_pending_wakes_excludes_wakes_after_publication(tmp_path: Path) -> None:
    world_db = _setup_pending(tmp_path, "alpha", "wake-a")
    registry = _registry(tmp_path)
    with _client(registry) as client:
        client.post("/api/profiles/alpha/pending-wakes/wake-a/finalize")
        response = client.get("/api/profiles/alpha/pending-wakes")

    assert response.status_code == 200
    assert response.json() == {"wakes": []}
    with sqlite3.connect(world_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 1


def test_finalize_pending_wake_publishes_and_replays_idempotently(tmp_path: Path) -> None:
    world_db = _setup_pending(tmp_path, "alpha", "wake-a")
    registry = _registry(tmp_path)
    with _client(registry) as client:
        published = client.post("/api/profiles/alpha/pending-wakes/wake-a/finalize")
        replayed = client.post("/api/profiles/alpha/pending-wakes/wake-a/finalize")

    assert published.status_code == 200
    body = published.json()
    assert body["status"] == "published"
    assert body["commit_id"] == "wake-a:finalize"
    assert body["stats"]["objects_created"] == 1
    assert body["stats"]["total_items"] == 1
    assert body["committed_at"].endswith("+00:00")
    assert replayed.status_code == 200
    assert replayed.json()["status"] == "already_published"
    assert replayed.json()["commit_id"] == "wake-a:finalize"
    with sqlite3.connect(world_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 1


def test_finalize_pending_wake_rejects_unknown_wake(tmp_path: Path) -> None:
    world_db = _setup_pending(tmp_path, "alpha", "wake-a")
    registry = _registry(tmp_path)
    with _client(registry) as client:
        response = client.post("/api/profiles/alpha/pending-wakes/wake-nope/finalize")

    assert response.status_code == 404
    assert response.json()["status"] == "wake_unknown"
    with sqlite3.connect(world_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 0


def test_finalize_pending_wake_rejects_missing_world(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    profile = _profile(tmp_path, "phantom").model_copy(
        update={"id": "phantom", "display_name": "Phantom"}
    )
    registry.create(profile)
    with _client(registry) as client:
        response = client.post("/api/profiles/phantom/pending-wakes/wake-a/finalize")

    assert response.status_code == 404
    assert response.json()["status"] == "no_world"


def test_finalize_pending_wake_rejects_oversized_wake_id(tmp_path: Path) -> None:
    _setup_pending(tmp_path, "alpha", "wake-a")
    registry = _registry(tmp_path)
    with _client(registry) as client:
        response = client.post(
            f"/api/profiles/alpha/pending-wakes/{'wake-' + 'x' * 200}/finalize"
        )

    assert response.status_code == 422
    assert "wake_id" in response.json()["detail"]


def test_finalize_pending_wake_unknown_profile_fails_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with _client(registry) as client:
        listed = client.get("/api/profiles/nobody/pending-wakes")
        finalized = client.post("/api/profiles/nobody/pending-wakes/wake-a/finalize")

    assert listed.status_code == 404
    assert finalized.status_code == 404


def test_abandon_pending_wake_marks_staging_abandoned_and_releases_lease(
    tmp_path: Path,
) -> None:
    world_db = _setup_pending(tmp_path, "alpha", "wake-a")
    _lease_wake(world_db, "wake-a", "alpha-thread")
    registry = _registry(tmp_path)

    with _client(registry) as client:
        response = client.post("/api/profiles/alpha/pending-wakes/wake-a/abandon")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "abandoned"
    assert body["abandoned_items"] == {
        "staged_objects": 1,
        "staged_assertions": 0,
        "staged_inquiries": 0,
    }
    assert body["lease_released"] is True
    assert body["runtime_claims_cleaned"] is False  # no runtime db exists
    with sqlite3.connect(world_db) as connection:
        status = connection.execute(
            "SELECT status FROM staged_objects WHERE wake_id = 'wake-a'"
        ).fetchone()[0]
        lease_count = connection.execute(
            "SELECT COUNT(*) FROM graph_shell_writer_leases WHERE singleton_id = 1"
        ).fetchone()[0]
    assert status == "abandoned"
    assert lease_count == 0


def test_abandon_pending_wake_replays_idempotently(tmp_path: Path) -> None:
    world_db = _setup_pending(tmp_path, "alpha", "wake-a")
    _lease_wake(world_db, "wake-a", "alpha-thread")
    registry = _registry(tmp_path)

    with _client(registry) as client:
        first = client.post("/api/profiles/alpha/pending-wakes/wake-a/abandon")
        replayed = client.post("/api/profiles/alpha/pending-wakes/wake-a/abandon")

    assert first.status_code == 200
    assert first.json()["status"] == "abandoned"
    assert replayed.status_code == 200
    assert replayed.json()["status"] == "already_abandoned"
    with sqlite3.connect(world_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = 'wake-a' "
            "AND status = 'active'"
        ).fetchone()[0] == 0


def test_abandon_pending_wake_owner_mismatch_fails_closed(tmp_path: Path) -> None:
    world_db = _setup_pending(tmp_path, "alpha", "wake-a")
    _lease_wake(world_db, "wake-other", "other-thread")
    registry = _registry(tmp_path)

    with _client(registry) as client:
        response = client.post("/api/profiles/alpha/pending-wakes/wake-a/abandon")

    assert response.status_code == 422
    assert "owner_mismatch" in response.json()["detail"]
    with sqlite3.connect(world_db) as connection:
        status = connection.execute(
            "SELECT status FROM staged_objects WHERE wake_id = 'wake-a'"
        ).fetchone()[0]
        owner = connection.execute(
            "SELECT owner_wake_id FROM graph_shell_writer_leases WHERE singleton_id = 1"
        ).fetchone()[0]
    assert status == "active"
    assert owner == "wake-other"


def test_abandon_pending_wake_without_lease_reports_already_abandoned(
    tmp_path: Path,
) -> None:
    world_db = _setup_pending(tmp_path, "alpha", "wake-a")
    registry = _registry(tmp_path)

    with _client(registry) as client:
        response = client.post("/api/profiles/alpha/pending-wakes/wake-a/abandon")

    assert response.status_code == 200
    assert response.json()["status"] == "already_abandoned"
    assert response.json()["abandoned_items"] == 0


def test_abandon_pending_wake_rejects_missing_world(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    profile = _profile(tmp_path, "phantom").model_copy(
        update={"id": "phantom", "display_name": "Phantom"}
    )
    registry.create(profile)
    with _client(registry) as client:
        response = client.post("/api/profiles/phantom/pending-wakes/wake-a/abandon")

    assert response.status_code == 404
    assert response.json()["status"] == "no_world"


def test_abandon_pending_wake_rejects_oversized_wake_id(tmp_path: Path) -> None:
    _setup_pending(tmp_path, "alpha", "wake-a")
    registry = _registry(tmp_path)
    with _client(registry) as client:
        response = client.post(
            f"/api/profiles/alpha/pending-wakes/{'wake-' + 'x' * 200}/abandon"
        )

    assert response.status_code == 422
    assert "wake_id" in response.json()["detail"]


def test_abandon_pending_wake_unknown_profile_fails_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with _client(registry) as client:
        response = client.post("/api/profiles/nobody/pending-wakes/wake-a/abandon")

    assert response.status_code == 404
