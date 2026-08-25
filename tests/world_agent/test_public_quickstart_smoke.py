"""Public quickstart smoke: the two-wake offline demo path from a fresh checkout.

Mirrors ``examples/offline_demo.py`` at the pytest level: seeds one observation,
runs two scripted wakes through the default composition root on an isolated
database pair, and asserts the formal-graph outcome (referent reuse +
supersede). Guards the quickstart story the public README advertises.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from leave_information_bubble.world import (
    CognitiveDelta,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)

_NOW = datetime(2026, 8, 22, tzinfo=UTC)
_WA_RUNNER = "leave_information_bubble.world_agent.cli"


def _tool_turn(identifier: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"content": "", "tool_calls": [{"id": identifier, "name": name, "arguments": arguments}]}


def _write_model(path: Path, turns: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")


def _seed_observation(world_path: Path) -> None:
    WorldStore(world_path).memory_commit(
        CognitiveDelta(
            objects=[],
            observations=[
                ObservationInput(
                    id="smoke:obs-1",
                    source_uri="https://fixture.test/smoke",
                    source_kind="community_post",
                    title="smoke observation",
                    excerpt="Replay record: IG defeated LNG 3-1 in the playoff series.",
                    content_ref="https://fixture.test/smoke",
                    depth=ObservationDepth.CONTENT,
                    observed_at=_NOW,
                )
            ],
            assertions=[],
        ),
        "public-quickstart-smoke-seed",
    )


async def _run_wake(
    workdir: Path,
    thread_id: str,
    wake_id: str,
    turns: list[dict[str, object]],
) -> dict[str, object]:
    runner = importlib.import_module(_WA_RUNNER)
    model_path = workdir / f"{thread_id}-{wake_id}-model.json"
    _write_model(model_path, turns)
    args = runner.parse_args(
        [
            "--perspective",
            "Public quickstart smoke wake.",
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
        ]
    )
    return await runner.run(args, print_report=False)


def _formal_count(world_path: Path, table: str) -> int:
    connection = sqlite3.connect(world_path)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


async def test_public_quickstart_two_wake_demo(tmp_path: Path) -> None:
    """The README quickstart: two wakes, one supersede chain, no new objects in wake 2."""
    (tmp_path / "replay.json").write_text(
        json.dumps(
            {"adapter_id": "replay", "adapter_version": "1", "discoveries": {}, "hydrations": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    world_path = tmp_path / "world.sqlite3"
    _seed_observation(world_path)

    wake1 = await _run_wake(
        tmp_path,
        "quickstart-thread",
        "qs-w1",
        [
            _tool_turn("qs-w1-search", "memory_search", {"query": "IG vs LNG", "limit": 5}),
            _tool_turn(
                "qs-w1-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "qs-w1-event",
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
                            "op_id": "qs-w1-ig",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "IG",
                                "kind": "entity",
                                "domain_hints": ["lol"],
                            },
                        },
                        {
                            "op_id": "qs-w1-lng",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "LNG",
                                "kind": "entity",
                                "domain_hints": ["lol"],
                            },
                        },
                        {
                            "op_id": "qs-w1-anchor-ig",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "qs-w1:s1",
                                "predicate": "participant",
                                "object_ref": "qs-w1:s2",
                                "epistemic_role": "fact",
                                "confidence": 0.9,
                                "evidence": ["smoke:obs-1"],
                            },
                        },
                        {
                            "op_id": "qs-w1-anchor-lng",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "qs-w1:s1",
                                "predicate": "participant",
                                "object_ref": "qs-w1:s3",
                                "epistemic_role": "fact",
                                "confidence": 0.9,
                                "evidence": ["smoke:obs-1"],
                            },
                        },
                        {
                            "op_id": "qs-w1-score",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "qs-w1:s1",
                                "predicate": "score",
                                "literal": "3-0",
                                "epistemic_role": "fact",
                                "confidence": 0.9,
                                "evidence": ["smoke:obs-1"],
                            },
                        },
                    ]
                },
            ),
            _tool_turn("qs-w1-inspect", "graph_inspect", {}),
            _tool_turn("qs-w1-finalize", "finalize_graph", {}),
        ],
    )
    assert wake1["terminal_status"] == "published"
    assert wake1["finalize_receipt"]["commit_id"] == "qs-w1:finalize"

    wake2 = await _run_wake(
        tmp_path,
        "quickstart-thread",
        "qs-w2",
        [
            _tool_turn("qs-w2-search", "memory_search", {"query": "IG vs LNG", "limit": 5}),
            _tool_turn(
                "qs-w2-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "qs-w2-score",
                            "kind": "assertion",
                            "action": "supersede",
                            "payload": {
                                "subject_ref": "qs-w1:s1",
                                "predicate": "score",
                                "literal": "3-1",
                                "epistemic_role": "fact",
                                "confidence": 0.95,
                                "evidence": ["smoke:obs-1"],
                                "supersedes_ref": "qs-w1:a3",
                            },
                        }
                    ]
                },
            ),
            _tool_turn("qs-w2-finalize", "finalize_graph", {}),
        ],
    )
    assert wake2["terminal_status"] == "published"
    assert wake2["finalize_receipt"]["commit_id"] == "qs-w2:finalize"

    # wake 2 reused every wake-1 object id: no new formal objects, one supersede.
    assert _formal_count(world_path, "objects") == 3
    assert _formal_count(world_path, "assertions") == 4
    connection = sqlite3.connect(world_path)
    try:
        rows = connection.execute(
            "SELECT id, literal_json, supersedes_id FROM assertions "
            "WHERE predicate = 'score' ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == 2
    assert rows[0][0] == "qs-w1:a3" and json.loads(rows[0][1]) == "3-0"
    assert rows[1][2] == "qs-w1:a3" and json.loads(rows[1][1]) == "3-1"
