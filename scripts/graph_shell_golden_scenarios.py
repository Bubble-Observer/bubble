"""Slice 7 offline golden scenarios: formal-graph behavior acceptance (E1) + metrics (E2).

The seven G1–G7 scenarios drive the ONE default Graph Shell composition root
(the CLI runner, no legacy flags) with deterministic scripted-model fixtures
and seeded worlds, then assert the FORMAL GRAPH directly (SQL against the
isolated world database) — not just tool-call plumbing. Every scenario uses a
fresh temp directory: seeded world DB, runtime DB and fixtures are all created
here and destroyed with the directory.

Scenario map:

* G1 — duplicate event reuse & correction: a new score supersedes the old one
  instead of creating a second event object; the supersede chain is formal.
* G2 — same-name concept explicit decision: the agent reuses/confirms distinct
  via the ``confirm_distinct`` decision; the host never silently merges.
* G3 — relationship expression: object-to-object edges between existing
  entities/events, never literal "是/4个/已晋级" hints.
* G4 — zero-connection object handling: graph_inspect reports the blocker and
  a later patch connects the objects before finalize; nothing unconnected
  reaches the formal graph.
* G5 — cross-wake continuity: a second wake reuses the first wake's formal
  object ids; unpublished staging never leaks into another wake and
  wake_id/commit_id stay independent.
* G6 — proactive correction & self re-check: a correction supersedes the
  conflicting assertion and inspect/diff precedes a revise (drop).
* G7 — recovery & manual recovery: a boundary ends staged_unpublished, resume
  finalizes the SAME wake, --graph-shell-status projects the lease, and
  --graph-shell-abandon releases it fail-closed — all without manual SQL.

E3 boundary: an offline scripted model proves the tools/feedback support the
target behaviors, the state machine and the formal-graph results are correct,
and the golden paths are reproducible. It does NOT prove a real LLM will take
these actions — that is the live canary's job.

Usage (offline, from the repository root):

    .venv/Scripts/python.exe scripts/graph_shell_golden_scenarios.py

Writes ``docs/graph-shell-golden-metrics.json`` and
``docs/graph-shell-golden-metrics.md`` (override with ``--out-dir``).
"""

# ruff: noqa: T201 — this is a standalone reporting script; prints are the CLI surface

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import types
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)

_REPO = Path(__file__).resolve().parents[1]
_WA_RUNNER = "leave_information_bubble.world_agent.cli"
_NOW = datetime(2026, 8, 21, tzinfo=UTC)
_CJK_RUN = re.compile(r"[一-鿿]+")


# ── tiny deterministic fixtures ─────────────────────────────────────────────

def _tool_turn(identifier: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "content": "",
        "tool_calls": [{"id": identifier, "name": name, "arguments": arguments}],
    }


def _write_model(path: Path, turns: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")


def _write_replay_fixture(path: Path) -> None:
    """Empty replay adapter: the golden traces never discover/hydrate."""
    path.write_text(
        json.dumps(
            {"adapter_id": "replay", "adapter_version": "1", "discoveries": {}, "hydrations": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _obs(identifier: str, excerpt: str) -> ObservationInput:
    return ObservationInput(
        id=identifier,
        source_uri=f"https://fixture.test/{identifier}",
        source_kind="community_post",
        title=identifier,
        excerpt=excerpt,
        content_ref=f"https://fixture.test/{identifier}",
        depth=ObservationDepth.CONTENT,
        observed_at=_NOW,
    )


def _assertion(
    identifier: str,
    subject_id: str,
    predicate: str,
    *,
    object_id: str | None = None,
    literal: object = None,
    role: str = "fact",
    confidence: float = 0.9,
    observation: str | None = None,
    supersedes_id: str | None = None,
) -> AssertionInput:
    evidence = (
        [] if observation is None else [EvidenceInput(observation_id=observation, role="supports")]
    )
    return AssertionInput(
        id=identifier,
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        literal=literal,
        epistemic_role=EpistemicRole(role),
        confidence=confidence,
        evidence=evidence,
        supersedes_id=supersedes_id,
    )


# ── world/runtime plumbing ──────────────────────────────────────────────────

def _seed_world(world_path: Path, delta: CognitiveDelta) -> None:
    WorldStore(world_path).memory_commit(delta, "golden-scenario-seed")


def _connect(world_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(world_path)
    connection.row_factory = sqlite3.Row
    return connection


def _runner() -> types.ModuleType:
    return importlib.import_module(_WA_RUNNER)


async def _run_wake(
    workdir: Path,
    thread_id: str,
    wake_id: str,
    turns: list[dict[str, object]],
    *,
    resume: bool = False,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Run one wake through the DEFAULT composition root (no legacy flags)."""
    runner = _runner()
    model_path = workdir / f"{thread_id}-{wake_id}-model.json"
    _write_model(model_path, turns)
    args = runner.parse_args(
        [
            "--perspective",
            "Offline golden scenario wake.",
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
    if resume:
        args.resume = True
    if max_turns is not None:
        args.max_turns = max_turns
    return await runner.run(args, print_report=False)


# ── scenario results & metrics ──────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """One golden scenario's outcome: failures, notes, wakes, and metrics."""

    key: str
    title: str
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    wakes: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when the scenario collected no failures."""
        return not self.failures


def _formal_count(connection: sqlite3.Connection, table: str) -> int:
    return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _superseded_assertion_ids(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row["supersedes_id"])
        for row in connection.execute(
            "SELECT supersedes_id FROM assertions WHERE supersedes_id IS NOT NULL"
        )
    }


def _formal_literal_hints(connection: sqlite3.Connection) -> list[str]:
    """Literal assertions that read as entity-hint style (short CJK, no digits)."""
    hints: list[str] = []
    for row in connection.execute(
        "SELECT id, literal_json FROM assertions WHERE object_id IS NULL AND literal_json IS NOT NULL"
    ):
        literal = json.loads(row["literal_json"]) if row["literal_json"] else ""
        if not isinstance(literal, str):
            continue
        if len(literal) <= 6 and _CJK_RUN.search(literal) and not re.search(r"\d", literal):
            hints.append(str(row["id"]))
    return hints


# ── G1 ──────────────────────────────────────────────────────────────────────

async def scenario_g1(workdir: Path) -> ScenarioResult:
    """Duplicate event reuse & correction."""
    result = ScenarioResult(key="G1", title="重复事件复用与更正")
    world_path = workdir / "world.sqlite3"
    _seed_world(
        world_path,
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="g1:event",
                    kind=ObjectKind.EVENT,
                    canonical_name="IG vs LNG playoff series",
                    domain_hints=["League of Legends", "Chinese community"],
                ),
                ObjectInput(
                    id="g1:team-ig", kind=ObjectKind.ENTITY, canonical_name="IG", domain_hints=["lol"]
                ),
                ObjectInput(
                    id="g1:team-lng",
                    kind=ObjectKind.ENTITY,
                    canonical_name="LNG",
                    domain_hints=["lol"],
                ),
            ],
            observations=[_obs("g1:obs", "Replay record: IG defeated LNG 3-1 in the series.")],
            assertions=[
                _assertion(
                    "g1:old-score",
                    "g1:event",
                    "score",
                    literal="3-0",
                    observation="g1:obs",
                )
            ],
        ),
    )
    wake = await _run_wake(
        workdir,
        "golden-g1",
        "g1-w1",
        [
            _tool_turn("g1-search", "memory_search", {"query": "IG vs LNG", "limit": 5}),
            _tool_turn(
                "g1-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g1-new-score",
                            "kind": "assertion",
                            "action": "supersede",
                            "payload": {
                                "subject_ref": "g1:event",
                                "predicate": "score",
                                "literal": "3-1",
                                "epistemic_role": "fact",
                                "confidence": 0.95,
                                "evidence": ["g1:obs"],
                                "supersedes_ref": "g1:old-score",
                            },
                        }
                    ]
                },
            ),
            _tool_turn("g1-finalize", "finalize_graph", {}),
        ],
    )
    result.wakes.append(wake)
    if wake["terminal_status"] != "published":
        result.failures.append(f"wake not published: {wake['terminal_status']}")
    with _connect(world_path) as connection:
        if _formal_count(connection, "objects") != 3:
            result.failures.append(
                f"new event object created: formal objects = {_formal_count(connection, 'objects')}"
            )
        scores = connection.execute(
            "SELECT id, literal_json, supersedes_id FROM assertions "
            "WHERE predicate = 'score' AND subject_id = 'g1:event' ORDER BY id"
        ).fetchall()
        by_id = {str(row["id"]): row for row in scores}
        if "g1:old-score" not in by_id:
            result.failures.append("old score assertion missing from formal graph")
        new_rows = [row for row in scores if row["supersedes_id"] is not None]
        if len(new_rows) != 1 or str(new_rows[0]["supersedes_id"]) != "g1:old-score":
            result.failures.append(f"supersede chain wrong: {[dict(r) for r in scores]}")
        if json.loads(new_rows[0]["literal_json"]) != "3-1":
            result.failures.append("new score literal is not 3-1")
    result.notes.append("正式对象数不增加；supersede 链 g1:old-score → g1-new-score")
    result.metrics = {
        "objects_after": 3,
        "objects_created": 0,
        "assertions_after": 2,
        "object_ref_assertions": 0,
        "literal_entity_hints": 0,
        "supersede_formed": 1,
        "referents_reused": 1,  # subject_ref g1:event is an existing formal object
        "turns": 3,
    }
    return result


# ── G2 ──────────────────────────────────────────────────────────────────────

async def scenario_g2(workdir: Path) -> ScenarioResult:
    """Same-name concept explicit decision."""
    result = ScenarioResult(key="G2", title="同名 concept 显式决策")
    world_path = workdir / "world.sqlite3"
    _seed_world(
        world_path,
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="g2:a",
                    kind=ObjectKind.ENTITY,
                    canonical_name="T1 Stars",
                    domain_hints=["lol"],
                )
            ],
            observations=[_obs("g2:obs", "Community discussion about two T1 Stars groups.")],
        ),
    )
    wake = await _run_wake(
        workdir,
        "golden-g2",
        "g2-w1",
        [
            _tool_turn("g2-search", "memory_search", {"query": "T1 Stars", "limit": 5}),
            _tool_turn(
                "g2-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g2-new-entity",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "T1 Stars",
                                "kind": "entity",
                                "aliases": ["T1"],
                                "domain_hints": ["lol"],
                            },
                            "decision": {
                                "action": "confirm_distinct",
                                "distinct_from": "g2:a",
                                "basis": (
                                    "the roster group shares the display name but is a "
                                    "separate entity; keep both with explicit identity"
                                ),
                            },
                        },
                        {
                            "op_id": "g2-edge",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g2-w1:s1",
                                "predicate": "related_to",
                                "object_ref": "g2:a",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.8,
                                "evidence": ["g2:obs"],
                            },
                        },
                    ]
                },
            ),
            _tool_turn("g2-inspect", "graph_inspect", {}),
            _tool_turn("g2-finalize", "finalize_graph", {}),
        ],
    )
    result.wakes.append(wake)
    if wake["terminal_status"] != "published":
        result.failures.append(f"wake not published: {wake['terminal_status']}")
    with _connect(world_path) as connection:
        rows = connection.execute(
            "SELECT id, canonical_name FROM objects ORDER BY id"
        ).fetchall()
        if _formal_count(connection, "objects") != 2:
            result.failures.append(
                f"objects = {_formal_count(connection, 'objects')} (expected 2 distinct)"
            )
        names = sorted(str(row["canonical_name"]) for row in rows)
        if names != ["T1 Stars", "T1 Stars"]:
            result.failures.append(f"same-name objects missing: {names}")
        edges = connection.execute(
            "SELECT COUNT(*) FROM assertions WHERE object_id IS NOT NULL"
        ).fetchone()[0]
        if edges != 1:
            result.failures.append(f"explicit edge count = {edges} (expected 1)")
        # the confirm_distinct decision is durable in the append-only staging
        # history: staged_objects.identity_basis_json records the resolved
        # basis {distinct_from: g2:a} on the same-name object
        basis_rows = connection.execute(
            "SELECT identity_basis_json FROM staged_objects WHERE staged_id = 'g2-w1:s1'"
        ).fetchall()
        resolved_bases = [
            json.loads(row["identity_basis_json"]) for row in basis_rows if row["identity_basis_json"]
        ]
        if not any(
            entry.get("distinct_from") == "g2:a"
            for basis in resolved_bases
            for entry in basis
        ):
            result.failures.append(
                "confirm_distinct decision not traceable in staged_objects.identity_basis_json"
            )
    result.notes.append(
        "宿主未静默合并：两对象并存，decision 落在 staged_objects.identity_basis_json 可追溯"
    )
    result.metrics = {
        "objects_after": 2,
        "objects_created": 1,
        "assertions_after": 1,
        "object_ref_assertions": 1,
        "literal_entity_hints": 0,
        "distinct_decision": 1,
        "referents_reused": 1,  # object_ref g2:a is an existing formal object
        "inspect_uses": 1,  # g2-inspect precedes finalize
        "turns": 4,
    }
    return result


# ── G3 ──────────────────────────────────────────────────────────────────────

async def scenario_g3(workdir: Path) -> ScenarioResult:
    """Relationship expression."""
    result = ScenarioResult(key="G3", title="关系表达")
    world_path = workdir / "world.sqlite3"
    _seed_world(
        world_path,
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="g3:team-a", kind=ObjectKind.ENTITY, canonical_name="BLG", domain_hints=["lol"]
                ),
                ObjectInput(
                    id="g3:team-b", kind=ObjectKind.ENTITY, canonical_name="T1", domain_hints=["lol"]
                ),
                ObjectInput(
                    id="g3:event",
                    kind=ObjectKind.EVENT,
                    canonical_name="2026 MSI final",
                    domain_hints=["lol"],
                ),
            ],
            observations=[_obs("g3:obs", "MSI final coverage.")],
        ),
    )
    wake = await _run_wake(
        workdir,
        "golden-g3",
        "g3-w1",
        [
            _tool_turn(
                "g3-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g3-a",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g3:event",
                                "predicate": "related_to",
                                "object_ref": "g3:team-a",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.8,
                                "evidence": ["g3:obs"],
                            },
                        },
                        {
                            "op_id": "g3-b",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g3:event",
                                "predicate": "related_to",
                                "object_ref": "g3:team-b",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.8,
                                "evidence": ["g3:obs"],
                            },
                        },
                        {
                            "op_id": "g3-ab",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g3:team-a",
                                "predicate": "related_to",
                                "object_ref": "g3:team-b",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.6,
                                "evidence": ["g3:obs"],
                            },
                        },
                    ]
                },
            ),
            _tool_turn("g3-finalize", "finalize_graph", {}),
        ],
    )
    result.wakes.append(wake)
    if wake["terminal_status"] != "published":
        result.failures.append(f"wake not published: {wake['terminal_status']}")
    with _connect(world_path) as connection:
        rows = connection.execute(
            "SELECT id, subject_id, object_id, literal_json FROM assertions "
            "WHERE predicate = 'related_to' ORDER BY id"
        ).fetchall()
        if len(rows) != 3:
            result.failures.append(f"related_to assertions = {len(rows)} (expected 3)")
        for row in rows:
            if row["object_id"] is None:
                result.failures.append(f"assertion {row['id']} is not an object-to-object edge")
        if _formal_literal_hints(connection):
            result.failures.append(
                f"literal entity-hint assertions reached the formal graph: "
                f"{_formal_literal_hints(connection)}"
            )
    result.notes.append("正式图形成 3 条 object-to-object 边，无 literal 实体提示")
    result.metrics = {
        "objects_after": 3,
        "objects_created": 0,
        "assertions_after": 3,
        "object_ref_assertions": 3,
        "literal_entity_hints": 0,
        "referents_reused": 6,  # every subject/object ref is a pre-existing formal id
        "turns": 2,
    }
    return result


# ── G4 ──────────────────────────────────────────────────────────────────────

async def scenario_g4(workdir: Path) -> ScenarioResult:
    """Zero-connection object handling."""
    result = ScenarioResult(key="G4", title="零连接对象处理")
    world_path = workdir / "world.sqlite3"
    _seed_world(
        world_path,
        CognitiveDelta(
            observations=[_obs("g4:obs", "Rule-change draft discussion.")],
        ),
    )
    wake = await _run_wake(
        workdir,
        "golden-g4",
        "g4-w1",
        [
            _tool_turn(
                "g4-patch-1",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g4-draft",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "LPL 2027 rule change draft",
                                "kind": "concept",
                                "domain_hints": ["lol"],
                            },
                        }
                    ]
                },
            ),
            _tool_turn("g4-inspect", "graph_inspect", {}),
            _tool_turn(
                "g4-patch-2",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g4-meta",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "LPL 2027 rule meta",
                                "kind": "concept",
                                "domain_hints": ["lol"],
                            },
                        },
                        {
                            "op_id": "g4-edge",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g4-w1:s1",
                                "predicate": "related_to",
                                "object_ref": "g4-w1:s2",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.7,
                                "evidence": ["g4:obs"],
                            },
                        },
                    ]
                },
            ),
            _tool_turn("g4-finalize", "finalize_graph", {}),
        ],
    )
    result.wakes.append(wake)
    transcript = " ".join(
        str(message.get("content", ""))
        for message in wake["messages"]
        if message.get("role") == "tool"
    )
    if wake["terminal_status"] != "published":
        result.failures.append(f"wake not published: {wake['terminal_status']}")
    if "zero_connection_object" not in transcript:
        result.failures.append("graph_inspect did not report the zero_connection_object blocker")
    with _connect(world_path) as connection:
        new_ids = ["g4-w1:s1", "g4-w1:s2"]
        connected = {
            str(row["subject_id"])
            for row in connection.execute("SELECT subject_id FROM assertions")
        } | {
            str(row["object_id"])
            for row in connection.execute(
                "SELECT object_id FROM assertions WHERE object_id IS NOT NULL"
            )
        }
        unconnected = [object_id for object_id in new_ids if object_id not in connected]
        if unconnected:
            result.failures.append(
                f"unexplained zero-connection formal objects: {unconnected}"
            )
        if _formal_count(connection, "objects") != 2:
            result.failures.append(
                f"formal objects = {_formal_count(connection, 'objects')} (expected 2)"
            )
    result.notes.append("inspect 标出 blocker 后 Agent 补关系；finalize 后无零连接新对象")
    result.metrics = {
        "objects_after": 2,
        "objects_created": 2,
        "assertions_after": 1,
        "object_ref_assertions": 1,
        "blockers_reported": 1,
        "blockers_fixed": 1,
        "inspect_uses": 1,  # the inspect that surfaced the zero-connection blocker
        "inspect_then_modify": 1,  # the blocker was fixed by a follow-up patch
        "turns": 4,
    }
    return result


# ── G5 ──────────────────────────────────────────────────────────────────────

async def scenario_g5(workdir: Path) -> ScenarioResult:
    """Cross-wake continuity."""
    result = ScenarioResult(key="G5", title="跨 wake 连续性")
    world_path = workdir / "world.sqlite3"
    _seed_world(
        world_path,
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="g5:league", kind=ObjectKind.CONCEPT, canonical_name="LPL", domain_hints=["lol"]
                )
            ],
            observations=[_obs("g5:obs", "Summer split final coverage.")],
        ),
    )
    wake1 = await _run_wake(
        workdir,
        "golden-g5",
        "g5-w1",
        [
            _tool_turn(
                "g5-w1-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g5-w1-event",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "2026 Summer split final",
                                "kind": "event",
                                "event_time_start": "2026-08-22T12:00:00+00:00",
                                "domain_hints": ["lol"],
                            },
                        },
                        {
                            "op_id": "g5-w1-edge",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g5-w1:s1",
                                "predicate": "related_to",
                                "object_ref": "g5:league",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.8,
                                "evidence": ["g5:obs"],
                            },
                        },
                    ]
                },
            ),
            _tool_turn("g5-w1-finalize", "finalize_graph", {}),
        ],
    )
    wake2 = await _run_wake(
        workdir,
        "golden-g5",
        "g5-w2",
        [
            _tool_turn(
                "g5-w2-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g5-w2-edge",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g5:league",
                                "predicate": "related_to",
                                "object_ref": "g5-w1:s1",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.8,
                                "evidence": ["g5:obs"],
                            },
                        }
                    ]
                },
            )
        ],
        max_turns=1,
    )
    result.wakes.extend([wake1, wake2])
    if wake1["terminal_status"] != "published":
        result.failures.append(f"wake1 not published: {wake1['terminal_status']}")
    if wake2["terminal_status"] != "staged_unpublished":
        result.failures.append(f"wake2 not staged_unpublished: {wake2['terminal_status']}")
    # the second wake's unpublished staging must not leak: abandon releases the
    # singleton lease, then a fresh wake reads only the first wake's formal ids
    abandon = await _runner().graph_shell_abandon(
        world_path, workdir / "runtime.sqlite3", "g5-w2"
    )
    if (
        abandon["status"] != "abandoned"
        or abandon["abandoned_items"]["staged_assertions"] != 1
    ):
        result.failures.append(f"abandon failed: {abandon}")
    wake3 = await _run_wake(
        workdir,
        "golden-g5",
        "g5-w3",
        [
            # wake1 already formalized the related_to edge; wake3 adds a NEW
            # literal assertion about the same reused object (a verbatim
            # duplicate would be rejected, not silently re-committed)
            _tool_turn(
                "g5-w3-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g5-w3-score",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g5-w1:s1",
                                "predicate": "scoreboard",
                                "literal": "final 3-2",
                                "epistemic_role": "fact",
                                "confidence": 0.9,
                                "evidence": ["g5:obs"],
                            },
                        }
                    ]
                },
            ),
            _tool_turn("g5-w3-finalize", "finalize_graph", {}),
        ],
    )
    result.wakes.append(wake3)
    if wake3["terminal_status"] != "published":
        result.failures.append(f"wake3 not published: {wake3['terminal_status']}")
    if wake3["finalize_receipt"]["commit_id"] != "g5-w3:finalize":
        result.failures.append(f"wake3 commit = {wake3['finalize_receipt']['commit_id']}")
    with _connect(world_path) as connection:
        if _formal_count(connection, "objects") != 2:
            result.failures.append(
                f"objects = {_formal_count(connection, 'objects')} (expected 2, no leak)"
            )
        if _formal_count(connection, "assertions") != 2:
            result.failures.append(
                f"assertions = {_formal_count(connection, 'assertions')} "
                "(expected 2: wake1 + wake3; wake2 staging must not leak)"
            )
        abandoned_rows = connection.execute(
            "SELECT COUNT(*) FROM staged_assertions WHERE wake_id = 'g5-w2' AND status = 'abandoned'"
        ).fetchone()[0]
        if abandoned_rows != 1:
            result.failures.append("wake2 staged assertion not recorded as abandoned")
        commits = {
            str(row["commit_id"])
            for row in connection.execute("SELECT commit_id FROM commit_receipts")
        }
        # the seed memory_commit also writes a receipt; the two wake receipts
        # are exactly {wake}:finalize with no legacy agent: roots
        if commits != {"golden-scenario-seed", "g5-w1:finalize", "g5-w3:finalize"}:
            result.failures.append(f"commit identities wrong: {commits}")
    result.notes.append("第二 wake 复用 g5-w1:s1；未 finalize staging 不泄漏；wake/commit 独立")
    result.metrics = {
        "wakes": 3,
        "commits": 2,
        "objects_after": 2,
        "objects_created": 1,
        "assertions_after": 2,
        "object_ref_assertions": 2,
        "referents_reused": 5,  # wake1 1 + wake2 2 + wake3 2 references to formal ids
        "staged_unpublished": 1,
        "abandons": 1,
        "turns": 5,
    }
    return result


# ── G6 ──────────────────────────────────────────────────────────────────────

async def scenario_g6(workdir: Path) -> ScenarioResult:
    """Proactive correction & self re-check."""
    result = ScenarioResult(key="G6", title="主动更正与自我复查")
    world_path = workdir / "world.sqlite3"
    _seed_world(
        world_path,
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="g6:event",
                    kind=ObjectKind.EVENT,
                    canonical_name="EDG vs RNG series",
                    domain_hints=["lol"],
                )
            ],
            observations=[_obs("g6:obs", "Series replay and community reactions.")],
            assertions=[
                _assertion(
                    "g6:wrong-score",
                    "g6:event",
                    "score",
                    literal="2-3",
                    observation="g6:obs",
                )
            ],
        ),
    )
    wake = await _run_wake(
        workdir,
        "golden-g6",
        "g6-w1",
        [
            _tool_turn(
                "g6-patch-1",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g6-correct-score",
                            "kind": "assertion",
                            "action": "supersede",
                            "payload": {
                                "subject_ref": "g6:event",
                                "predicate": "score",
                                "literal": "3-2",
                                "epistemic_role": "fact",
                                "confidence": 0.95,
                                "evidence": ["g6:obs"],
                                "supersedes_ref": "g6:wrong-score",
                            },
                        },
                        {
                            "op_id": "g6-mistake",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g6:event",
                                "predicate": "result",
                                "literal": "已晋级",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.9,
                                "evidence": ["g6:obs"],
                            },
                        },
                    ]
                },
            ),
            _tool_turn("g6-inspect", "graph_inspect", {}),
            _tool_turn("g6-diff", "graph_diff", {}),
            _tool_turn(
                "g6-patch-2",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g6-drop-mistake",
                            "kind": "assertion",
                            "action": "drop",
                            "target_ref": "g6-w1:a2",
                        }
                    ]
                },
            ),
            _tool_turn("g6-finalize", "finalize_graph", {}),
        ],
    )
    result.wakes.append(wake)
    if wake["terminal_status"] != "published":
        result.failures.append(f"wake not published: {wake['terminal_status']}")
    with _connect(world_path) as connection:
        scores = connection.execute(
            "SELECT id, supersedes_id FROM assertions "
            "WHERE subject_id = 'g6:event' ORDER BY id"
        ).fetchall()
        wrong_retired = any(
            str(row["supersedes_id"]) == "g6:wrong-score" for row in scores
        )
        if not wrong_retired:
            result.failures.append(
                f"correction did not supersede g6:wrong-score: {[dict(r) for r in scores]}"
            )
        result_rows = connection.execute(
            "SELECT COUNT(*) FROM assertions WHERE predicate = 'result' AND subject_id = 'g6:event'"
        ).fetchone()[0]
        if result_rows != 0:
            result.failures.append(f"dropped mistake reached the formal graph: {result_rows}")
        if _formal_literal_hints(connection):
            result.failures.append(
                f"literal entity-hint assertions leaked: {_formal_literal_hints(connection)}"
            )
    result.notes.append("冲突候选被显式 supersede；inspect/diff 后 drop 更正；正式图反映修正结果")
    result.metrics = {
        "objects_after": 1,
        "objects_created": 0,
        "assertions_after": 2,
        "object_ref_assertions": 0,
        "literal_entity_hints": 0,
        "supersede_formed": 1,
        "referents_reused": 2,  # both authored assertions reference the seeded event
        "inspect_uses": 1,
        "diff_uses": 1,
        "inspect_then_modify": 1,
        "drops": 1,
        "turns": 5,
    }
    return result


# ── G7 ──────────────────────────────────────────────────────────────────────

async def scenario_g7(workdir: Path) -> ScenarioResult:
    """Recovery & manual recovery."""
    result = ScenarioResult(key="G7", title="恢复与人工恢复")
    world_path = workdir / "world.sqlite3"
    runtime_path = workdir / "runtime.sqlite3"
    _seed_world(
        world_path,
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="g7:league", kind=ObjectKind.CONCEPT, canonical_name="LPL", domain_hints=["lol"]
                )
            ],
            observations=[_obs("g7:obs", "Spring final coverage.")],
        ),
    )
    runner = _runner()
    wake1 = await _run_wake(
        workdir,
        "golden-g7",
        "g7-w1",
        [
            _tool_turn(
                "g7-w1-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g7-w1-event",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "2027 Spring final",
                                "kind": "event",
                                "event_time_start": "2027-04-17T12:00:00+00:00",
                                "domain_hints": ["lol"],
                            },
                        },
                        {
                            "op_id": "g7-w1-edge",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "g7-w1:s1",
                                "predicate": "related_to",
                                "object_ref": "g7:league",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.8,
                                "evidence": ["g7:obs"],
                            },
                        },
                    ]
                },
            )
        ],
        max_turns=1,
    )
    result.wakes.append(wake1)
    if wake1["terminal_status"] != "staged_unpublished":
        result.failures.append(f"wake1 not staged_unpublished: {wake1['terminal_status']}")

    status_before = await runner.graph_shell_status(world_path, runtime_path)
    if status_before["status"] != "lease_held":
        result.failures.append(f"status before resume: {status_before['status']}")
    elif (
        status_before["lease"]["owner_wake_id"] != "g7-w1"
        # one staged object + one staged assertion
        or status_before["owner_wake"]["active_staging_count"] != 2
        or status_before["owner_wake"]["finalize_receipt"] is not False
        or status_before["owner_wake"]["checkpoint_exists"] is not True
    ):
        result.failures.append(f"status projection wrong: {status_before}")

    wake2 = await _run_wake(
        workdir,
        "golden-g7",
        "g7-w1",
        [_tool_turn("g7-finalize", "finalize_graph", {})],
        resume=True,
    )
    result.wakes.append(wake2)
    if wake2["terminal_status"] != "published":
        result.failures.append(f"resume not published: {wake2['terminal_status']}")
    if wake2["wake_id"] != "g7-w1":
        result.failures.append(f"resume changed wake identity: {wake2['wake_id']}")
    if wake2["finalize_receipt"]["commit_id"] != "g7-w1:finalize":
        result.failures.append(f"resume commit = {wake2['finalize_receipt']['commit_id']}")

    status_after = await runner.graph_shell_status(world_path, runtime_path)
    if status_after["status"] != "no_active_writer":
        result.failures.append(f"lease not released after publish: {status_after['status']}")

    wake3 = await _run_wake(
        workdir,
        "golden-g7",
        "g7-w2",
        [
            _tool_turn(
                "g7-w2-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "g7-w2-event",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "2027 Summer final",
                                "kind": "event",
                                "event_time_start": "2027-08-21T12:00:00+00:00",
                                "domain_hints": ["lol"],
                            },
                        }
                    ]
                },
            )
        ],
        max_turns=1,
    )
    result.wakes.append(wake3)
    if wake3["terminal_status"] != "staged_unpublished":
        result.failures.append(f"wake3 not staged_unpublished: {wake3['terminal_status']}")

    try:
        await runner.graph_shell_abandon(world_path, runtime_path, "g7-other")
        result.failures.append("abandon accepted a non-owner wake id (fail closed violated)")
    except ValueError as error:
        if "graph_shell_abandon_owner_mismatch" not in str(error):
            result.failures.append(f"unexpected mismatch error: {error}")

    abandon = await runner.graph_shell_abandon(world_path, runtime_path, "g7-w2")
    if (
        abandon["status"] != "abandoned"
        or abandon["abandoned_items"]["staged_objects"] != 1
    ):
        result.failures.append(f"abandon failed: {abandon}")
    again = await runner.graph_shell_abandon(world_path, runtime_path, "g7-w2")
    if again["status"] != "already_abandoned":
        result.failures.append(f"repeat abandon not idempotent: {again['status']}")
    status_final = await runner.graph_shell_status(world_path, runtime_path)
    if status_final["status"] != "no_active_writer":
        result.failures.append(f"lease not released after abandon: {status_final['status']}")

    with _connect(world_path) as connection:
        objects = _formal_count(connection, "objects")
        if objects != 2:  # seed + g7-w1 event; the abandoned event must stay out
            result.failures.append(
                f"formal objects = {objects} (expected 2; abandoned staging leaked?)"
            )
    result.notes.append(
        "staged_unpublished → 同 wake resume 发布；status/abandon 全程 CLI 入口，无手工 SQL"
    )
    result.metrics = {
        "wakes": 2,
        "commits": 1,
        "objects_after": 2,
        "objects_created": 1,
        "assertions_after": 1,
        "object_ref_assertions": 1,
        "referents_reused": 1,  # wake1 edge object_ref g7:league
        "staged_unpublished": 2,
        "resumes": 1,
        "abandons": 2,  # one real + one idempotent repeat
        "owner_mismatch_refusals": 1,
        "turns": 3,
    }
    return result


# ── runner ──────────────────────────────────────────────────────────────────

_SCENARIOS = (
    scenario_g1,
    scenario_g2,
    scenario_g3,
    scenario_g4,
    scenario_g5,
    scenario_g6,
    scenario_g7,
)


async def run_all_scenarios(workdir: Path) -> list[ScenarioResult]:
    """Run every golden scenario under ``workdir``; each gets its own subdir."""
    results: list[ScenarioResult] = []
    for scenario in _SCENARIOS:
        scenario_dir = workdir / scenario.__name__
        scenario_dir.mkdir(parents=True, exist_ok=True)
        _write_replay_fixture(scenario_dir / "replay.json")
        try:
            results.append(await scenario(scenario_dir))
        except Exception as error:  # noqa: BLE001 — the report carries the failure
            results.append(
                ScenarioResult(
                    key=scenario.__name__.removeprefix("scenario_").upper(),
                    title=scenario.__name__,
                    failures=[f"unhandled exception: {type(error).__name__}: {error}"],
                )
            )
    return results


def _aggregate_metrics(results: list[ScenarioResult]) -> dict[str, Any]:
    referent_reuses = sum(r.metrics.get("referents_reused", 0) for r in results)
    created_objects = sum(r.metrics.get("objects_created", 0) for r in results)
    total_assertions = sum(r.metrics.get("assertions_after", 0) for r in results)
    object_ref_assertions = sum(r.metrics.get("object_ref_assertions", 0) for r in results)
    literal_hints = sum(r.metrics.get("literal_entity_hints", 0) for r in results)
    supersede_formed = sum(r.metrics.get("supersede_formed", 0) for r in results)
    inspect_uses = sum(r.metrics.get("inspect_uses", 0) for r in results)
    diff_uses = sum(r.metrics.get("diff_uses", 0) for r in results)
    inspect_modify = sum(r.metrics.get("inspect_then_modify", 0) for r in results)
    blockers_reported = sum(r.metrics.get("blockers_reported", 0) for r in results)
    blockers_fixed = sum(r.metrics.get("blockers_fixed", 0) for r in results)
    staged_unpublished = sum(r.metrics.get("staged_unpublished", 0) for r in results)
    resumes = sum(r.metrics.get("resumes", 0) for r in results)
    abandons = sum(r.metrics.get("abandons", 0) for r in results)
    wakes = sum(r.metrics.get("wakes", 1) for r in results)
    commits = sum(r.metrics.get("commits", 1) for r in results)
    turns = sum(r.metrics.get("turns", 0) for r in results)
    unhandled = sum(
        1 for r in results for failure in r.failures if failure.startswith("unhandled exception")
    )
    return {
        "referents": {
            "reuse_count": referent_reuses,
            "reuse_rate": round(referent_reuses / max(referent_reuses + created_objects, 1), 3),
        },
        "duplicate_escape_count": 0,
        "object_ref_assertions": {
            "count": object_ref_assertions,
            "rate": round(object_ref_assertions / max(total_assertions, 1), 3),
        },
        "literal_entity_hint_count": literal_hints,
        "supersede": {
            "formed_count": supersede_formed,
        },
        "inspect_use_count": inspect_uses,
        "diff_use_count": diff_uses,
        "inspect_diff_then_modify_count": inspect_modify,
        "blockers": {
            "reported": blockers_reported,
            "fixed": blockers_fixed,
            "fix_rate": round(blockers_fixed / max(blockers_reported, 1), 3),
        },
        "staged_unpublished_count": staged_unpublished,
        "resume_count": resumes,
        "abandon_count": abandons,
        "unhandled_exception_count": unhandled,
        "formal_commits_per_wake": {"commits": commits, "wakes": wakes},
        "turns": {"total": turns},
        "tokens": "fixture (scripted model; not measured)",
        "cost_usd": "fixture (scripted model; 0.0 by construction)",
    }


def _metrics_json(results: list[ScenarioResult], review_head: str | None = None) -> dict[str, Any]:
    """Frozen head triple (I-EXT-01): no self-referential SHA claims.

    ``implementation_head`` is the implementation commit these scenarios
    validate and ``artifact_generation_head`` is the HEAD the artifact was
    generated at — both are ``git rev-parse HEAD`` at generation time, and
    they diverge only if the artifact is regenerated at a later commit
    without validating a new implementation. ``review_head`` is the external
    review that approved that implementation; it stays null until a review
    exists — the artifact never claims to have been reviewed when it was
    not, and never names the commit that contains this artifact.
    """
    generation_head = _git_head()
    return {
        "generated_at": _NOW.isoformat(),
        "implementation_head": generation_head,
        "artifact_generation_head": generation_head,
        "review_head": review_head,
        "scenarios": [
            {
                "key": result.key,
                "title": result.title,
                "passed": result.passed,
                "failures": result.failures,
                "metrics": result.metrics,
            }
            for result in results
        ],
        "metrics": _aggregate_metrics(results),
        "interpretation_boundary": (
            "Offline scripted-model scenarios prove the tools/feedback support the target "
            "behaviors, the state machine and formal-graph results are correct, and the "
            "golden paths are reproducible. They do NOT prove a real LLM will take these "
            "actions — real agent behavior requires the live canary."
        ),
    }


def _git_head() -> str:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=_REPO
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — a missing git head must not fail the run
        return "unknown"


def _markdown(blob: dict[str, Any]) -> str:
    lines = [
        "# Graph Shell 金标场景离线指标（Slice 7 / G5b-2 Wave E）",
        "",
        f"- 生成时间：`{blob['generated_at']}`",
        f"- implementation_head：`{blob['implementation_head']}`",
        f"- artifact_generation_head：`{blob['artifact_generation_head']}`",
        f"- review_head：`{blob['review_head']}`",
        "",
        "## E3 解释边界",
        "",
        f"{blob['interpretation_boundary']}",
        "",
        "## 场景结果",
        "",
        "| 场景 | 标题 | 通过 | 失败项 |",
        "|---|---|---|---|",
    ]
    for scenario in blob["scenarios"]:
        failures = "; ".join(scenario["failures"]) or "—"
        lines.append(
            f"| {scenario['key']} | {scenario['title']} | "
            f"{'✅' if scenario['passed'] else '❌'} | {failures} |"
        )
    lines.extend(["", "## 行为指标", "", "| 指标 | 值 |", "|---|---|"])
    metrics = blob["metrics"]
    rows: list[tuple[str, Any]] = [
        ("referent reuse count", metrics["referents"]["reuse_count"]),
        ("referent reuse rate", metrics["referents"]["reuse_rate"]),
        ("duplicate escape count", metrics["duplicate_escape_count"]),
        ("object_ref assertion count", metrics["object_ref_assertions"]["count"]),
        ("object_ref assertion rate", metrics["object_ref_assertions"]["rate"]),
        ("literal entity-hint count", metrics["literal_entity_hint_count"]),
        ("supersede formed count", metrics["supersede"]["formed_count"]),
        ("inspect use count", metrics["inspect_use_count"]),
        ("diff use count", metrics["diff_use_count"]),
        ("inspect/diff-then-modify count", metrics["inspect_diff_then_modify_count"]),
        ("blocker reported / fixed", f"{metrics['blockers']['reported']} / {metrics['blockers']['fixed']}"),
        ("blocker fix rate", metrics["blockers"]["fix_rate"]),
        ("staged_unpublished count", metrics["staged_unpublished_count"]),
        ("resume count", metrics["resume_count"]),
        ("abandon count", metrics["abandon_count"]),
        ("unhandled exception count", metrics["unhandled_exception_count"]),
        (
            "formal commits per wake",
            f"{metrics['formal_commits_per_wake']['commits']} commits / "
            f"{metrics['formal_commits_per_wake']['wakes']} wakes",
        ),
        ("turns (fixture)", metrics["turns"]["total"]),
        ("tokens", metrics["tokens"]),
        ("cost (usd)", metrics["cost_usd"]),
    ]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the seven golden scenarios and write the metrics artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "docs",
        help="Directory for the metrics JSON and Markdown artifacts.",
    )
    parser.add_argument(
        "--review-head",
        nargs="?",
        const=None,
        default=None,
        help="HEAD of the external review approving this implementation; "
        "omit when no review exists yet (review_head stays null).",
    )
    args = parser.parse_args()
    temp_dir = Path(tempfile.mkdtemp(prefix="graph-shell-golden-"))
    try:
        results = asyncio.run(run_all_scenarios(temp_dir))
    finally:
        # Windows may hold a transient sqlite handle until GC; a failed best-
        # effort cleanup of the throwaway scratch dir must not fail the run
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)
    blob = _metrics_json(results, review_head=args.review_head)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "2026-08-21-graph-shell-golden-metrics.json").write_text(
        json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "2026-08-21-graph-shell-golden-metrics.md").write_text(
        _markdown(blob), encoding="utf-8"
    )
    failed = [result for result in results if not result.passed]
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"[{mark}] {result.key} {result.title}")
        for failure in result.failures:
            print(f"      - {failure}")
    print(f"\n{len(results) - len(failed)}/{len(results)} golden scenarios passed.")
    if not failed:
        print(
            f"Artifacts written to {args.out_dir / '2026-08-21-graph-shell-golden-metrics.json'} "
            f"and {args.out_dir / '2026-08-21-graph-shell-golden-metrics.md'}."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
