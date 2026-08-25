"""Four-wake offline demo: durable cognition with the Graph Shell (no network, no API key).

Runs the real composition root (``src/leave_information_bubble/world_agent/cli.py``)
against an isolated throwaway database pair with deterministic scripted-model
fixtures:

  wake 1  —  creates an event with participant anchors and a score assertion,
             reviews readiness, finalizes
  wake 2  —  reuses the wake-1 event id (referent reuse), supersedes the wake-1
             score assertion, finalizes again
  wake 3  —  stages a new player entity (Rookie) but stops without finalizing
             (a wake that ran out of budget/deadline ends exactly like this),
             so the run is ``staged_unpublished`` and the world keeps its work
  wake 4  —  ``--resume --wake-id demo-w3`` re-enters the same wake from its
             checkpoint and finalizes, publishing Rookie

Expected result: three ``published`` wakes (commits demo-w1/demo-w2/demo-w3),
one supersede chain ``demo-w1:a3 -> demo-w2:a1``, one interrupted wake whose
staging survives and is later published by resume.

This is scripted evaluation: it proves the protocol and the state machine, not
that a real LLM will take these actions. Real-wake evidence is reported
separately in ``docs/evaluation.md``.

Usage (from the repository root, after ``pip install -e .[dev]``):

    python examples/offline_demo.py
"""

# ruff: noqa: T201 — this is a standalone demo; prints are the CLI surface

from __future__ import annotations

import asyncio
import gc
import importlib
import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leave_information_bubble.world import (
    CognitiveDelta,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)

_WA_RUNNER = "leave_information_bubble.world_agent.cli"
_NOW = datetime(2026, 8, 22, tzinfo=UTC)


# ── deterministic fixtures ──────────────────────────────────────────────────

def _tool_turn(identifier: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "content": "",
        "tool_calls": [{"id": identifier, "name": name, "arguments": arguments}],
    }


def _text_turn(content: str) -> dict[str, object]:
    """Return a plain-text model response (no tool call)."""
    return {"content": content, "tool_calls": []}


def _write_model(path: Path, turns: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")


def _write_replay_fixture(path: Path) -> None:
    """Empty replay adapter: the scripted wakes never discover or hydrate."""
    path.write_text(
        json.dumps(
            {"adapter_id": "replay", "adapter_version": "1", "discoveries": {}, "hydrations": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _seed_observation(world_path: Path) -> None:
    """Seed one observation so the scripted assertions have evidence to cite."""
    WorldStore(world_path).memory_commit(
        CognitiveDelta(
            objects=[],
            observations=[
                ObservationInput(
                    id="demo:obs-1",
                    source_uri="https://fixture.test/demo",
                    source_kind="community_post",
                    title="demo observation",
                    excerpt="Replay record: IG defeated LNG 3-1 in the playoff series.",
                    content_ref="https://fixture.test/demo",
                    depth=ObservationDepth.CONTENT,
                    observed_at=_NOW,
                )
            ],
            assertions=[],
        ),
        "offline-demo-seed",
    )


# ── wake runner ─────────────────────────────────────────────────────────────

async def _run_wake(
    workdir: Path,
    thread_id: str,
    wake_id: str,
    turns: list[dict[str, object]],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one wake through the default composition root (no legacy flags).

    With ``resume=True`` the run re-enters the wake from its durable
    checkpoint (``--resume --wake-id`` must match the checkpoint identity).
    """
    runner = importlib.import_module(_WA_RUNNER)
    model_path = workdir / f"{thread_id}-{wake_id}-model.json"
    _write_model(model_path, turns)
    args = runner.parse_args(
        [
            "--perspective",
            "Offline demo wake.",
            "--world-db",
            str(workdir / "world.sqlite3"),
            "--runtime-db",
            str(workdir / "runtime.sqlite3"),
            "--thread-id",
            thread_id,
            "--replay-fixture",
            str(workdir / "replay.json"),
            "--scripted-model-fixture",
            str(model_path),
            "--wake-id",
            wake_id,
            *(["--resume"] if resume else []),
        ]
    )
    return await runner.run(args, print_report=False)


async def _run_status(workdir: Path, thread_id: str) -> dict[str, Any]:
    """Show the read-only projection of the world's writer gate.

    Runs no model and needs no fixtures — the same command a user would run
    to see why their wake did not publish and what to do about it.
    """
    runner = importlib.import_module(_WA_RUNNER)
    args = runner.parse_args(
        [
            "--perspective",
            "Offline demo status.",
            "--world-db",
            str(workdir / "world.sqlite3"),
            "--runtime-db",
            str(workdir / "runtime.sqlite3"),
            "--thread-id",
            thread_id,
            "--graph-shell-status",
        ]
    )
    return await runner.run(args, print_report=False)


# ── formal-graph probes ─────────────────────────────────────────────────────

def _connect(world_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(world_path)
    connection.row_factory = sqlite3.Row
    return connection


def _formal_count(world_path: Path, table: str) -> int:
    connection = _connect(world_path)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def _staged_count(world_path: Path, table: str) -> int:
    connection = _connect(world_path)
    try:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE status = 'active'"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def _commit_count(world_path: Path) -> int:
    return _formal_count(world_path, "finalize_receipts")


def _supersede_chain(world_path: Path) -> list[dict[str, Any]]:
    connection = _connect(world_path)
    try:
        rows = connection.execute(
            "SELECT id, literal_json, supersedes_id FROM assertions "
            "WHERE predicate = 'score' ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    return [{"id": str(row["id"]), "literal": json.loads(row["literal_json"]),
             "supersedes_id": str(row["supersedes_id"] or "")} for row in rows]


# ── main ────────────────────────────────────────────────────────────────────

async def _demo(keep_world: Path | None) -> int:
    failures: list[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="bubble-demo-"))
    world_path = workdir / "world.sqlite3"
    try:
        print(f"workdir (isolated, removed on exit): {workdir}")
        _seed_observation(world_path)
        _write_replay_fixture(workdir / "replay.json")

        wake1 = await _run_wake(
            workdir,
            "demo-thread",
            "demo-w1",
            [
                _tool_turn("demo-w1-search", "memory_search", {"query": "IG vs LNG", "limit": 5}),
                _tool_turn(
                    "demo-w1-patch",
                    "graph_patch",
                    {
                        "items": [
                            {
                                "op_id": "demo-w1-event",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "IG vs LNG playoff series",
                                    "kind": "event",
                                    "event_time_start": _NOW.isoformat(),
                                    "domain_hints": ["lol"],
                                },
                            },
                            {
                                "op_id": "demo-w1-ig",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "IG",
                                    "kind": "entity",
                                    "domain_hints": ["lol"],
                                },
                            },
                            {
                                "op_id": "demo-w1-lng",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "LNG",
                                    "kind": "entity",
                                    "domain_hints": ["lol"],
                                },
                            },
                            {
                                "op_id": "demo-w1-anchor-ig",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "demo-w1:s1",
                                    "predicate": "participant",
                                    "object_ref": "demo-w1:s2",
                                    "epistemic_role": "fact",
                                    "confidence": 0.9,
                                    "evidence": ["demo:obs-1"],
                                },
                            },
                            {
                                "op_id": "demo-w1-anchor-lng",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "demo-w1:s1",
                                    "predicate": "participant",
                                    "object_ref": "demo-w1:s3",
                                    "epistemic_role": "fact",
                                    "confidence": 0.9,
                                    "evidence": ["demo:obs-1"],
                                },
                            },
                            {
                                "op_id": "demo-w1-score",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "demo-w1:s1",
                                    "predicate": "score",
                                    "literal": "3-0",
                                    "epistemic_role": "fact",
                                    "confidence": 0.9,
                                    "evidence": ["demo:obs-1"],
                                },
                            },
                        ]
                    },
                ),
                _tool_turn("demo-w1-inspect", "graph_inspect", {}),
                _tool_turn("demo-w1-finalize", "finalize_graph", {}),
            ],
        )
        print(f"wake 1: {wake1['terminal_status']}  (commit {wake1['finalize_receipt']['commit_id']})")
        if wake1["terminal_status"] != "published":
            failures.append(f"wake 1 not published: {wake1['terminal_status']}")

        wake2 = await _run_wake(
            workdir,
            "demo-thread",
            "demo-w2",
            [
                _tool_turn("demo-w2-search", "memory_search", {"query": "IG vs LNG", "limit": 5}),
                _tool_turn(
                    "demo-w2-patch",
                    "graph_patch",
                    {
                        "items": [
                            {
                                "op_id": "demo-w2-score",
                                "kind": "assertion",
                                "action": "supersede",
                                "payload": {
                                    "subject_ref": "demo-w1:s1",
                                    "predicate": "score",
                                    "literal": "3-1",
                                    "epistemic_role": "fact",
                                    "confidence": 0.95,
                                    "evidence": ["demo:obs-1"],
                                    "supersedes_ref": "demo-w1:a3",
                                },
                            }
                        ]
                    },
                ),
                _tool_turn("demo-w2-finalize", "finalize_graph", {}),
            ],
        )
        print(f"wake 2: {wake2['terminal_status']}  (commit {wake2['finalize_receipt']['commit_id']})")
        if wake2["terminal_status"] != "published":
            failures.append(f"wake 2 not published: {wake2['terminal_status']}")

        wake3 = await _run_wake(
            workdir,
            "demo-thread",
            "demo-w3",
            [
                _tool_turn(
                    "demo-w3-patch",
                    "graph_patch",
                    {
                        "items": [
                            {
                                "op_id": "demo-w3-rookie",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "Rookie",
                                    "kind": "entity",
                                    "domain_hints": ["lol"],
                                },
                            },
                            {
                                "op_id": "demo-w3-anchor",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "demo-w1:s1",
                                    "predicate": "participant",
                                    "object_ref": "demo-w3:s1",
                                    "epistemic_role": "fact",
                                    "confidence": 0.9,
                                    "evidence": ["demo:obs-1"],
                                },
                            },
                        ]
                    },
                ),
                # no finalize_graph: the wake stops first (real wakes end like
                # this when they run out of budget/deadline). The Graph Shell
                # host never auto-publishes — staging stays durable.
                _text_turn("This wake is out of budget; stop without publishing."),
                _text_turn("Confirmed: stop. No further actions."),
            ],
        )
        print(
            f"wake 3: {wake3['terminal_status']}  "
            f"(staged 1 object, nothing published)"
        )
        if wake3["terminal_status"] != "staged_unpublished":
            failures.append(f"wake 3 not staged_unpublished: {wake3['terminal_status']}")
        staged = _staged_count(world_path, "staged_objects")
        if staged != 1:
            failures.append(f"expected 1 active staged object after wake 3, got {staged}")

        status = await _run_status(workdir, "demo-thread")
        print(
            f"status: {status.get('status')}  (owner {status['lease']['owner_wake_id']}, "
            f"checkpoint {status['owner_wake']['checkpoint_terminal_status']}, "
            f"{status['owner_wake']['active_staging_count']} active staged item)"
        )
        if status.get("status") != "lease_held":
            failures.append(f"expected lease_held status, got {status.get('status')}")
        if status["lease"]["owner_wake_id"] != "demo-w3":
            failures.append(f"status owner mismatch: {status['lease']['owner_wake_id']}")

        wake4 = await _run_wake(
            workdir,
            "demo-thread",
            "demo-w3",
            [_tool_turn("demo-w3-finalize", "finalize_graph", {})],
            resume=True,
        )
        print(
            f"wake 4: {wake4['terminal_status']}  "
            f"(commit {wake4['finalize_receipt']['commit_id']})"
        )
        if wake4["terminal_status"] != "published":
            failures.append(f"wake 4 (resume) not published: {wake4['terminal_status']}")
        if wake4["finalize_receipt"]["commit_id"] != "demo-w3:finalize":
            failures.append(
                f"wake 4 commit id wrong: {wake4['finalize_receipt']['commit_id']}"
            )
        staged = _staged_count(world_path, "staged_objects")
        if staged != 0:
            failures.append(f"staging must be consumed by finalize, still {staged} active")

        objects = _formal_count(world_path, "objects")
        assertions = _formal_count(world_path, "assertions")
        commits = _commit_count(world_path)
        print(f"formal graph: {objects} objects, {assertions} assertions, {commits} commits")
        if objects != 4:
            failures.append(
                f"expected 4 formal objects (wake 3's Rookie published by resume), got {objects}"
            )
        if assertions != 5:
            failures.append(f"expected 5 formal assertions, got {assertions}")
        if commits != 3:
            failures.append(f"expected 3 finalize receipts, got {commits}")
        chain = _supersede_chain(world_path)
        print(f"supersede chain: {chain}")
        if len(chain) != 2 or chain[1].get("supersedes_id") != "demo-w1:a3":
            failures.append(f"supersede chain wrong: {chain}")
    finally:
        if keep_world is not None:
            keep_world.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(world_path, keep_world)
            print(
                f"world kept at: {keep_world}  "
                f"(render it: python scripts/render_world_graph.py {keep_world})"
            )
        # Windows may hold a transient sqlite handle until GC; best-effort
        # cleanup of the throwaway scratch dir must not fail the demo.
        gc.collect()
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print("\nDEMO FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nDEMO PASSED: four wakes. wake 2 reused the wake-1 event id and "
          "superseded its score; wake 3 stopped without publishing (staging "
          "survived); wake 4 resumed wake 3 and published Rookie.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the four-wake demo; exit 0 on success, 1 on any failure.

    ``--keep-world PATH`` copies the demo's world database to PATH before the
    scratch dir is removed, so the result can be rendered with
    ``scripts/render_world_graph.py``.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the four-wake offline demo.")
    parser.add_argument(
        "--keep-world",
        type=Path,
        default=None,
        help="copy the demo world database to this path before cleanup",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_demo(args.keep_world))


if __name__ == "__main__":
    sys.exit(main())
