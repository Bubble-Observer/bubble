"""Fresh-thread replay proof for Graph Shell world cognition (G5b-2).

The legacy proposal fixtures (submit_cognition turns, ``agent:`` commit
identities, the --digest-off experiment switch) were retired with the
proposal runtime; every wake here runs the one normal graph —
search/read, patch, inspect/diff, explicit finalize — and no legacy
commit or recovery path is reachable (D6).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from leave_information_bubble.channels import (
    DiscoveryBatch,
    ObservationBatch,
    SourceOccurrence,
)
from leave_information_bubble.gateway.client import NativeToolCall, ToolModelResponse
from leave_information_bubble.models.epistemics import (
    AccessDepth,
    ObservationModality,
    SourceObservation,
)
from leave_information_bubble.world import CognitiveDelta, ObjectInput, ObjectKind, WorldStore, WorldTools
from leave_information_bubble.world.finalize import finalize_graph
from leave_information_bubble.world.store import observation_id
from leave_information_bubble.world_agent.graph import WakeReasoningProfile

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)


def test_runner_entrypoint_resolves_the_src_package() -> None:
    """Catch the direct script command requiring an undeclared PYTHONPATH."""
    runner = Path(__file__).parents[2] / "scripts" / "run_world_agent.py"

    completed = subprocess.run([sys.executable, str(runner), "--help"], check=False)

    assert completed.returncode == 0


def test_runner_defaults_to_broad_and_rejects_object_id_outside_deep() -> None:
    """Catch ambiguous CLI mode selection or a center accepted by a Broad run."""
    runner = _load_runner()
    required = _required_runner_args("broad-default")

    assert runner.parse_args(required).mode == "broad"
    with pytest.raises(SystemExit):
        runner.parse_args([*required, "--object-id", "event-ig-lng"])


def test_runner_accepts_optional_perspective_and_deprecated_mission_alias() -> None:
    runner = _load_runner()
    base = [
        "--thread-id",
        "optional-perspective",
        "--world-db",
        "data/test/optional-world.sqlite3",
        "--runtime-db",
        "data/test/optional-runtime.sqlite3",
    ]

    assert runner.parse_args(base).perspective is None
    assert runner.parse_args([*base, "--perspective", "A temporary lens."]).perspective == (
        "A temporary lens."
    )
    with pytest.warns(FutureWarning, match="--mission is deprecated"):
        legacy = runner.parse_args([*base, "--mission", "Legacy lens."])
    assert legacy.perspective == "Legacy lens."
    with pytest.raises(SystemExit):
        runner.parse_args([*base, "--perspective", "new", "--mission", "old"])


def test_runner_accepts_an_optional_center_only_for_deep_mode() -> None:
    """Catch the Deep center selector missing from the public runner boundary."""
    runner = _load_runner()

    args = runner.parse_args(
        [
            *_required_runner_args("deep-center", mission="Explain one center."),
            "--mode",
            "deep",
            "--object-id",
            "event-ig-lng",
        ]
    )

    assert args.mode == "deep"
    assert args.object_id == "event-ig-lng"


def test_runner_exposes_explicit_checkpoint_resume() -> None:
    """Catch a failed live run lacking an explicit continuation boundary."""
    runner = _load_runner()

    args = runner.parse_args(
        [
            *_required_runner_args("resume-thread", mission="Continue the interrupted run."),
            "--resume",
        ]
    )

    assert args.resume is True


def test_runner_keeps_graph_shell_opt_in() -> None:
    """G5b-1 is selectable without changing the legacy production default."""
    runner = _load_runner()
    required = _required_runner_args("graph-shell-flag")

    assert runner.parse_args(required).graph_shell is False
    assert runner.parse_args([*required, "--graph-shell"]).graph_shell is True


def test_runner_keeps_digest_cache_reuse_opt_in() -> None:
    """P5 instrumentation is on, but reuse stays off before the live default decision."""
    runner = _load_runner()
    required = _required_runner_args("cache-default")

    assert runner.parse_args(required).digest_cache_reuse is False
    assert runner.parse_args([*required, "--digest-cache-reuse"]).digest_cache_reuse is True


def test_runner_exposes_thinking_profile_flags() -> None:
    """Live thinking mode is an explicit per-wake opt-in, off by default."""
    runner = _load_runner()
    required = _required_runner_args("thinking-default")

    defaults = runner.parse_args(required)
    assert defaults.thinking is False
    assert defaults.reasoning_effort is None

    on = runner.parse_args([*required, "--thinking"])
    assert on.thinking is True
    assert on.reasoning_effort is None

    maxed = runner.parse_args([*required, "--thinking", "--reasoning-effort", "max"])
    assert maxed.thinking is True
    assert maxed.reasoning_effort == "max"

    with pytest.raises(SystemExit):
        runner.parse_args([*required, "--reasoning-effort", "low"])


def test_runner_resolves_the_default_domain_focus_and_rejects_unknown() -> None:
    """The CLI and programmatic runner share the fail-closed focus resolver."""
    runner = _load_runner()
    required = _required_runner_args("domain-default")

    assert runner.parse_args(required).domain.domain_key == "lol_cn"
    assert runner.parse_args([*required, "--domain", "lol_cn"]).domain.domain_key == "lol_cn"
    with pytest.raises(SystemExit):
        runner.parse_args([*required, "--domain", "unknown"])


async def test_runner_forwards_domain_into_world_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch --domain dropping before the WorldTools constructor."""
    runner = _load_runner()
    received: list[str] = []

    class _RecordingWorldTools(WorldTools):
        def __init__(self, *, domain: str = "", **kwargs: object) -> None:
            received.append(domain)
            super().__init__(domain=domain, **kwargs)

    monkeypatch.setattr(runner, "WorldTools", _RecordingWorldTools)
    args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("domain-finish", "finalize_graph", {})],
        "domain-run",
    )
    args.domain = "lol_cn"

    result = await runner.run(args)

    assert received == ["lol_cn"]
    assert result["run_report"]["entry"]["domain"] == "lol_cn"


async def test_runner_report_failure_does_not_undo_completed_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A presentation failure cannot turn a committed cognition run into a failure."""
    runner = _load_runner()

    def fail_report(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("simulated report failure")

    monkeypatch.setattr(runner, "build_run_report", fail_report)
    result = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            [_tool_turn("report-finish", "finalize_graph", {})],
            "report-failure",
        )
    )

    captured = capsys.readouterr()
    assert result["finalize_receipt"]["commit_id"] == f"{result['wake_id']}:finalize"
    assert "run report unavailable: RuntimeError: simulated report failure" in captured.err


def _tool_turn(identifier: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "content": "",
        "tool_calls": [{"id": identifier, "name": name, "arguments": arguments}],
    }


def _model_response(content: str, call: NativeToolCall | None = None) -> ToolModelResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    calls = [] if call is None else [call]
    if call is not None:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
        ]
    return ToolModelResponse(content=content, message=message, tool_calls=calls)


class _AdaptiveRecallModel:
    """Choose the expansion root from the actual durable search result."""

    def __init__(self) -> None:
        self.turn = 0
        self.search_memory: dict[str, Any] = {}
        self.expansion_memory: dict[str, Any] = {}
        self.expanded_object_id = ""

    async def invoke_tools(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]], **kwargs: Any
    ) -> ToolModelResponse:
        del tools
        del kwargs
        self.turn += 1
        if self.turn == 1:
            return _model_response(
                "Search durable memory.",
                NativeToolCall(
                    id="recall-search",
                    name="memory_search",
                    arguments={"query": "IG", "limit": 5},
                ),
            )
        memory = json.loads(messages[-1]["content"])["memory"]
        if self.turn == 2:
            self.search_memory = memory
            center = next(
                item
                for item in memory["anchor_objects"]
                if item["canonical_name"] == "IG vs LNG playoff series"
            )
            self.expanded_object_id = center["id"]
            return _model_response(
                "Expand the located center.",
                NativeToolCall(
                    id="recall-expand",
                    name="memory_expand",
                    arguments={"object_ids": [center["id"]], "depth": 2, "limit": 15},
                ),
            )
        self.expansion_memory = memory
        return _model_response(
            "",
            NativeToolCall(
                id="recall-finish",
                name="finalize_graph",
                arguments={},
            ),
        )


_COMMUNITY_EXCERPT = (
    "In this bounded sample, commenters used Renchuanren to connect the IG "
    "win to the club's 2018 championship identity while disputing roster continuity."
)


def _obs_id(source: str) -> str:
    """Return the unified observation id the replay fixture derives for one source."""
    return observation_id("replay", f"https://fixture.test/{source}")


def _community_obs_id() -> str:
    """Return the comment sub-id the fixture's ig-lng-community hydrate writes.

    A sampled comment keeps its own content-derived sub-id under the source
    id (task T13 fix F2), so evidence and links cite that sub-id — the id
    the model actually sees on the discussion card.
    """
    digest = hashlib.sha256(_COMMUNITY_EXCERPT.encode("utf-8")).hexdigest()[:32]
    return f"{_obs_id('ig-lng-community')}-comment-{digest}"


def _load_runner() -> ModuleType:
    return importlib.import_module("leave_information_bubble.world_agent.cli")


def _write_replay_fixture(path: Path) -> None:
    occurrences = [
        SourceOccurrence(
            id=identifier,
            adapter_id="replay",
            adapter_version="1",
            source_ref=f"https://fixture.test/{identifier}",
            canonical_url=f"https://fixture.test/{identifier}",
            title=title,
            content_type="community_post",
            language="zh-Hans",
            captured_at=NOW,
        )
        for identifier, title in (
            ("ig-lng-series", "IG vs LNG playoff series replay record"),
            ("ig-lng-community", "Chinese community reactions to IG vs LNG"),
            ("blg-roster", "BLG roster announcement discussion"),
            ("fearless-draft", "LPL fearless-draft rule change discussion"),
        )
    ]
    discovery = DiscoveryBatch(
        request_id="tool:scan-centers",
        adapter_id="replay",
        adapter_version="1",
        occurrences=occurrences,
    )

    def hydration(source: str, observation: SourceObservation) -> ObservationBatch:
        return ObservationBatch(
            request_id=f"tool:hydrate:{_obs_id(source)}",
            adapter_id="replay",
            adapter_version="1",
            observations=[observation],
        )

    fixture = {
        "adapter_id": "replay",
        "adapter_version": "1",
        "discoveries": {"tool:scan-centers": discovery.model_dump(mode="json")},
        "hydrations": {
            "https://fixture.test/ig-lng-series": hydration(
                "ig-lng-series",
                SourceObservation(
                    id="ig-lng-score",
                    source_ref="https://fixture.test/ig-lng-series",
                    modality=ObservationModality.DOCUMENT_TEXT,
                    access_depth=AccessDepth.CONTENT_TEXT,
                    excerpt="Replay match record: IG defeated LNG 3-1 in the playoff series.",
                    acquisition_method="replay_fixture",
                    captured_at=NOW,
                ),
            ).model_dump(mode="json"),
            "https://fixture.test/ig-lng-community": hydration(
                "ig-lng-community",
                SourceObservation(
                    id="ig-lng-reactions",
                    source_ref="https://fixture.test/ig-lng-community",
                    modality=ObservationModality.COMMENT,
                    access_depth=AccessDepth.REACTIONS,
                    excerpt=_COMMUNITY_EXCERPT,
                    acquisition_method="replay_fixture",
                    captured_at=NOW,
                    sampling_scope="twelve replayed top-level comments",
                ),
            ).model_dump(mode="json"),
        },
    }
    path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")


def _write_model(path: Path, turns: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")


def _required_runner_args(thread_id: str, *, mission: str = "Observe LoL.") -> list[str]:
    """Return explicit non-running paths for parser-only CLI contract tests."""
    return [
        "--perspective",
        mission,
        "--world-db",
        f"data/test/{thread_id}-world.sqlite3",
        "--runtime-db",
        f"data/test/{thread_id}-runtime.sqlite3",
        "--thread-id",
        thread_id,
    ]


def _runner_args(runner: ModuleType, tmp_path: Path, turns: object, thread_id: str) -> object:
    replay_path = tmp_path / f"{thread_id}-replay.json"
    model_path = tmp_path / f"{thread_id}-model.json"
    _write_replay_fixture(replay_path)
    model_path.write_text(json.dumps(turns), encoding="utf-8")
    return runner.parse_args(
        [
            "--perspective",
            "Validate a replay fixture.",
            "--world-db",
            str(tmp_path / f"{thread_id}-world.sqlite3"),
            "--runtime-db",
            str(tmp_path / f"{thread_id}-runtime.sqlite3"),
            "--thread-id",
            thread_id,
            "--replay-fixture",
            str(replay_path),
            "--scripted-model-fixture",
            str(model_path),
        ]
    )


async def test_scripted_model_rejects_a_malformed_turn(tmp_path: Path) -> None:
    """Catch replay accepting a normalized assistant content value that is not text."""
    runner = _load_runner()

    with pytest.raises(ValueError, match="malformed scripted model fixture"):
        await runner.run(_runner_args(runner, tmp_path, [{"content": 7}], "malformed"))


async def test_scripted_model_rejects_invalid_native_call_field_types(
    tmp_path: Path,
) -> None:
    """Catch malformed native-call fields crossing the scripted-model boundary."""
    runner = _load_runner()
    turns = [{"content": "", "tool_calls": [{"id": 1, "name": 2, "arguments": []}]}]

    with pytest.raises(ValueError) as error_info:
        await runner.run(_runner_args(runner, tmp_path, turns, "bad-call-types"))
    assert error_info.value.args == ("malformed scripted model fixture turn",)


async def test_scripted_model_reports_exhaustion_before_graph_completion(
    tmp_path: Path,
) -> None:
    """Catch an empty fixture leaking an unhelpful list-index error."""
    runner = _load_runner()

    with pytest.raises(RuntimeError, match="exhausted before graph completion"):
        await runner.run(_runner_args(runner, tmp_path, [], "exhausted"))


async def test_runner_rejects_resume_without_a_checkpoint(tmp_path: Path) -> None:
    """Catch resume silently bootstrapping a new run when its checkpoint is missing."""
    runner = _load_runner()
    args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "missing-checkpoint",
    )
    args.resume = True

    with pytest.raises(ValueError, match="checkpoint"):
        await runner.run(args)


def test_runner_parses_explicit_wake_id() -> None:
    """The CLI exposes an explicit wake identity, defaulting to the thread id."""
    runner = _load_runner()
    base = _required_runner_args("wake-parser")

    assert runner.parse_args(base).wake_id is None
    assert runner.parse_args([*base, "--wake-id", "wake-explicit"]).wake_id == "wake-explicit"


async def test_runner_commits_under_an_explicit_wake_id(tmp_path: Path) -> None:
    """A fresh run with --wake-id finalizes under the wake-explicit:finalize identity."""
    runner = _load_runner()
    args = _runner_args(
        runner, tmp_path, [_tool_turn("finish", "finalize_graph", {})], "wake-commit"
    )
    args.wake_id = "wake-explicit"

    result = await runner.run(args)

    assert result["wake_id"] == "wake-explicit"
    assert result["finalize_receipt"]["commit_id"] == "wake-explicit:finalize"


async def test_graph_shell_fresh_wakes_on_one_thread_get_distinct_commit_identities(
    tmp_path: Path,
) -> None:
    """Fresh Graph Shell wakes mint wake IDs while the durable thread stays stable."""
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finish-1", "finalize_graph", {})],
        "shared-shell-thread",
    )
    first_args.graph_shell = True
    first = await runner.run(first_args, print_report=False)

    second_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finish-2", "finalize_graph", {})],
        "shared-shell-thread",
    )
    second_args.graph_shell = True
    second = await runner.run(second_args, print_report=False)

    assert first["thread_id"] == second["thread_id"] == "shared-shell-thread"
    assert first["wake_id"] != second["wake_id"]
    assert first["finalize_receipt"]["commit_id"] == f"{first['wake_id']}:finalize"
    assert second["finalize_receipt"]["commit_id"] == f"{second['wake_id']}:finalize"
    assert first["run_report"]["entry"]["execution_mode"] == "graph_shell"
    assert first["run_report"]["entry"]["thread_id"] == "shared-shell-thread"
    assert first["run_report"]["entry"]["wake_id"] == first["wake_id"]
    assert first["run_report"]["publication"]["published"] is True


async def test_graph_shell_fresh_explicit_wake_cannot_reuse_durable_receipt(tmp_path: Path) -> None:
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finish", "finalize_graph", {})],
        "closed-wake-owner",
    )
    first_args.graph_shell = True
    first_args.wake_id = "fixed-closed-wake"
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "published"

    second_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "closed-wake-reuser",
    )
    second_args.graph_shell = True
    second_args.wake_id = "fixed-closed-wake"
    second_args.world_db = first_args.world_db

    with pytest.raises(ValueError, match="already finalized"):
        await runner.run(second_args, print_report=False)


async def test_graph_shell_fresh_explicit_wake_cannot_reuse_abandoned_history(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "create then drop",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "abandoned-create",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Abandoned draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        },
                        {
                            "op_id": "abandoned-drop",
                            "kind": "object",
                            "action": "drop",
                            "target_ref": "reused-abandoned:s1",
                        },
                    ]
                },
            )
        ],
        "abandoned-owner",
    )
    first_args.graph_shell = True
    first_args.wake_id = "reused-abandoned"
    first_args.max_turns = 1

    first = await runner.run(first_args, print_report=False)

    assert first["terminal_status"] == "staged_unpublished"
    with sqlite3.connect(first_args.world_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = ? AND status = 'abandoned'",
            (first_args.wake_id,),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?",
            (first_args.wake_id,),
        ).fetchone()[0] == 2

    second_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "abandoned-reuser",
    )
    second_args.graph_shell = True
    second_args.wake_id = first_args.wake_id
    second_args.world_db = first_args.world_db

    with pytest.raises(ValueError, match="durable Graph Shell history"):
        await runner.run(second_args, print_report=False)


async def test_graph_shell_fresh_explicit_wake_cannot_reuse_empty_checkpoint_history(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [{"content": "pause without world mutation", "tool_calls": []}],
        "empty-wake-owner",
    )
    first_args.graph_shell = True
    first_args.wake_id = "reused-empty-wake"
    first_args.max_turns = 1

    first = await runner.run(first_args, print_report=False)

    assert first["terminal_status"] == "staged_unpublished"

    second_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "empty-wake-reuser",
    )
    second_args.graph_shell = True
    second_args.wake_id = first_args.wake_id
    second_args.world_db = first_args.world_db

    with pytest.raises(ValueError, match="already claimed"):
        await runner.run(second_args, print_report=False)


async def test_graph_shell_resume_treats_durable_receipt_as_terminal_authority(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "resume-authority-op",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Receipt authority draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            )
        ],
        "receipt-authority-thread",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"
    assert finalize_graph(runner.WorldStore(first_args.world_db), first["wake_id"]).status == "published"

    resume_args = _runner_args(runner, tmp_path, [], "receipt-authority-thread")
    resume_args.graph_shell = True
    resume_args.resume = True

    resumed = await runner.run(resume_args, print_report=False)

    assert resumed["terminal_status"] == "already_published"
    assert resumed["finalize_receipt"]["commit_id"] == f"{first['wake_id']}:finalize"
    assert resumed["resume_allowed"] is False
    assert resumed["run_report"]["publication"]["resume_count"] == 1
    assert resumed["run_report"]["publication"]["finalize_status_counts"] == {
        "published": 1,
        "already_published": 1,
    }
    with sqlite3.connect(first_args.world_db) as connection:
        assert sum(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE wake_id = ? AND status = 'active'",
                (first["wake_id"],),
            ).fetchone()[0]
            for table in ("staged_objects", "staged_assertions", "staged_inquiries")
        ) == 0


async def test_graph_shell_resume_rejects_a_different_world_database(tmp_path: Path) -> None:
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finish", "finalize_graph", {})],
        "wrong-world-thread",
    )
    first_args.graph_shell = True
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "published"

    resume_args = _runner_args(runner, tmp_path, [], "wrong-world-thread")
    resume_args.graph_shell = True
    resume_args.resume = True
    resume_args.runtime_db = first_args.runtime_db
    resume_args.world_db = tmp_path / "different-world.sqlite3"

    with pytest.raises(ValueError, match="world_store_identity mismatch"):
        await runner.run(resume_args, print_report=False)


async def test_graph_shell_resume_rejects_a_copied_runtime_database(tmp_path: Path) -> None:
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [{"content": "pause without world mutation", "tool_calls": []}],
        "runtime-claim-owner",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"

    copied_runtime = tmp_path / "copied-runtime.sqlite3"
    shutil.copy2(first_args.runtime_db, copied_runtime)
    resume_args = _runner_args(runner, tmp_path, [], "runtime-claim-owner")
    resume_args.graph_shell = True
    resume_args.resume = True
    resume_args.world_db = first_args.world_db
    resume_args.runtime_db = copied_runtime

    with pytest.raises(ValueError, match="world wake claim identity mismatch"):
        await runner.run(resume_args, print_report=False)


async def test_graph_shell_resume_rejects_terminal_checkpoint_without_durable_receipt(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finish", "finalize_graph", {})],
        "missing-receipt-thread",
    )
    first_args.graph_shell = True
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "published"
    with sqlite3.connect(first_args.world_db) as connection:
        connection.execute("DELETE FROM finalize_receipts WHERE wake_id = ?", (first["wake_id"],))
        connection.commit()

    resume_args = _runner_args(runner, tmp_path, [], "missing-receipt-thread")
    resume_args.graph_shell = True
    resume_args.resume = True

    with pytest.raises(ValueError, match="durable finalize receipt is missing"):
        await runner.run(resume_args, print_report=False)


async def test_runner_graph_shell_first_invoke_input_carries_full_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-bootstrap checkpoint can recover the wake without inferred identity."""
    runner = _load_runner()
    original = runner.build_world_agent_graph
    captured: list[dict[str, object]] = []

    def recording_builder(**kwargs: object) -> object:
        delegate = original(**kwargs)

        class RecordingGraph:
            async def ainvoke(self, initial_state: object, config: object) -> object:
                assert isinstance(initial_state, dict)
                captured.append(initial_state)
                return await delegate.ainvoke(initial_state, config)

        return RecordingGraph()

    monkeypatch.setattr(runner, "build_world_agent_graph", recording_builder)
    args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finish", "finalize_graph", {})],
        "identity-first-thread",
    )
    args.graph_shell = True

    result = await runner.run(args, print_report=False)

    assert len(captured) == 1
    assert captured[0]["wake_id"] == result["wake_id"]
    assert captured[0]["thread_id"] == "identity-first-thread"
    assert captured[0]["execution_mode"] == "graph_shell"
    assert captured[0]["world_store_identity"] == str(args.world_db.resolve())
    assert captured[0]["domain_key"] == "lol_cn"


async def test_graph_shell_staged_unpublished_resume_reuses_same_wake(tmp_path: Path) -> None:
    """A non-finalizing boundary keeps staging and CLI resume reopens that checkpoint."""
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Resumable Draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            )
        ],
        "resume-shell-thread",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    first = await runner.run(first_args, print_report=False)

    second_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finalize", "finalize_graph", {})],
        "resume-shell-thread",
    )
    second_args.graph_shell = True
    second_args.resume = True
    second = await runner.run(second_args, print_report=False)

    assert first["terminal_status"] == "staged_unpublished"
    assert second["terminal_status"] == "published"
    assert second["wake_id"] == first["wake_id"]
    assert second["finalize_receipt"]["commit_id"] == f"{first['wake_id']}:finalize"
    assert second["run_report"]["publication"]["resume_count"] == 1
    recovery_messages = [
        message["content"]
        for message in second["messages"]
        if message.get("role") == "user"
        and str(message.get("content", "")).startswith("Graph Shell durable recovery:\n")
    ]
    assert len(recovery_messages) == 1
    recovery = json.loads(recovery_messages[0].split("\n", 1)[1])
    assert recovery["wake_id"] == first["wake_id"]
    assert recovery["active"] == {
        "objects": [f"{first['wake_id']}:s1"],
        "assertions": [],
        "inquiries": [],
    }
    assert recovery["active_total"] == 1
    assert recovery["patch_ledger_count"] == 1
    assert recovery["patch_success_count"] == 1
    assert recovery["patch_replay_count"] == 0
    assert recovery["finalize_receipt"] is False
    # F-04 H-matrix max-turns case: exactly one formal commit, no re-patch.
    with sqlite3.connect(first_args.world_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?",
            (first["wake_id"],),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?",
            (first["wake_id"],),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = ?",
            (f"{first['wake_id']}:finalize",),
        ).fetchone()[0] == 1


async def test_graph_shell_resume_fails_closed_when_checkpoint_staging_vanished(
    tmp_path: Path,
) -> None:
    """Checkpoint counters cannot overrule missing durable staging/receipt authority."""
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Vanishing Draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            )
        ],
        "corrupt-shell-thread",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"
    with sqlite3.connect(first_args.world_db) as connection:
        connection.execute("DELETE FROM staged_objects WHERE wake_id = ?", (first["wake_id"],))
        connection.commit()

    resume_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "corrupt-shell-thread",
    )
    resume_args.graph_shell = True
    resume_args.resume = True

    with pytest.raises(ValueError, match="durable staging is missing"):
        await runner.run(resume_args, print_report=False)


async def test_graph_shell_resume_rejects_staging_without_patch_ledger(
    tmp_path: Path,
) -> None:
    """Active staged work without a successful durable patch ledger fails closed.

    F-02 direction 2: resume must reconcile staging -> ledger, not only
    ledger -> staging. A staged item with no successful receipt is an
    impossible state and must never reach the model or finalize.
    """
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Untraceable Draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            )
        ],
        "ledgerless-shell-thread",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"
    wake_id = first["wake_id"]
    with sqlite3.connect(first_args.world_db) as connection:
        connection.execute(
            "DELETE FROM staged_patch_receipts WHERE wake_id = ?", (wake_id,)
        )
        connection.commit()

    resume_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "ledgerless-shell-thread",
    )
    resume_args.graph_shell = True
    resume_args.resume = True

    with pytest.raises(ValueError, match="no successful durable patch receipt"):
        await runner.run(resume_args, print_report=False)

    # fail closed: zero model/patch/finalize calls, staging preserved, no
    # formal/receipt/commit writes
    with sqlite3.connect(first_args.world_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = ? AND status = 'active'",
            (wake_id,),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = ?",
            (f"{wake_id}:finalize",),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM world_audit WHERE commit_id = ?",
            (f"{wake_id}:finalize",),
        ).fetchone()[0] == 0


def _patch_turn(identifier: str, canonical_name: str) -> dict[str, object]:
    return _tool_turn(
        identifier,
        "graph_patch",
        {
            "items": [
                {
                    "op_id": f"op-{identifier}",
                    "kind": "object",
                    "action": "create",
                    "payload": {
                        "canonical_name": canonical_name,
                        "kind": "concept",
                        "provisional": True,
                    },
                }
            ]
        },
    )


def _fresh_reminder_count(result: object) -> int:
    """Count natural-pause reminders added after the durable recovery marker.

    Checkpointed history (including the first wake's single reminder) is
    replayed into resume results, so only messages after the last recovery
    summary can be freshly added by the resumed wake.
    """
    messages = result["messages"]
    recovery_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user" and str(message.get("content", "")).startswith(
            "Graph Shell durable recovery:\n"
        ):
            recovery_index = index
    return sum(
        1
        for message in messages[recovery_index + 1 :]
        if message.get("role") == "user"
        and str(message.get("content", "")).startswith("You made no tool call.")
    )


async def test_graph_shell_natural_pause_reminder_is_once_per_wake_across_resume(
    tmp_path: Path,
) -> None:
    """A natural pause reminds at most once per wake; resume never re-fires it.

    F-04: completion_reminder_used is durable wake-scoped control state and
    must survive resume. The first wake pauses (one reminder, staged
    unpublished); a resumed pause must stop silently instead of reminding
    again; a second resume still reaches finalize on the same wake.
    """
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _patch_turn("pause", "Paused Draft"),
            {"content": "Pause, no tool call."},
            {"content": "Pause again, no tool call."},
        ],
        "pause-shell-thread",
    )
    first_args.graph_shell = True
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"
    assert first["completion_reminder_used"] is True
    assert _fresh_reminder_count(first) == 1
    wake_id = first["wake_id"]

    second_args = _runner_args(
        runner,
        tmp_path,
        [{"content": "Pause on resume; no second reminder allowed."}],
        "pause-shell-thread",
    )
    second_args.graph_shell = True
    second_args.resume = True
    second = await runner.run(second_args, print_report=False)
    assert second["wake_id"] == wake_id
    assert second["terminal_status"] == "staged_unpublished"
    assert second["completion_reminder_used"] is True
    assert _fresh_reminder_count(second) == 0
    recovery_messages = [
        message["content"]
        for message in second["messages"]
        if message.get("role") == "user"
        and str(message.get("content", "")).startswith("Graph Shell durable recovery:\n")
    ]
    assert len(recovery_messages) == 1
    recovery = json.loads(recovery_messages[0].split("\n", 1)[1])
    assert recovery["wake_id"] == wake_id
    assert recovery["active"] == {
        "objects": [f"{wake_id}:s1"],
        "assertions": [],
        "inquiries": [],
    }
    assert recovery["active_total"] == 1
    assert recovery["patch_ledger_count"] == 1
    assert recovery["patch_success_count"] == 1

    third_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finalize", "finalize_graph", {})],
        "pause-shell-thread",
    )
    third_args.graph_shell = True
    third_args.resume = True
    third = await runner.run(third_args, print_report=False)
    assert third["wake_id"] == wake_id
    assert third["terminal_status"] == "published"
    assert third["completion_reminder_used"] is True
    assert _fresh_reminder_count(third) == 0
    assert third["finalize_receipt"]["commit_id"] == f"{wake_id}:finalize"
    with sqlite3.connect(first_args.world_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = ?",
            (f"{wake_id}:finalize",),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM world_audit WHERE commit_id = ?",
            (f"{wake_id}:finalize",),
        ).fetchone()[0] == 1


async def test_graph_shell_max_cost_boundary_resume_finalizes_same_wake(
    tmp_path: Path,
) -> None:
    """The cost boundary stops staged work and CLI resume finalizes the same wake.

    F-04 H-matrix: cumulative scripted-model cost crossing --max-cost-usd is a
    real composition-root stop with no auto-finalize; resume reopens the same
    wake, does not re-patch, and finalizes exactly once.
    """
    runner = _load_runner()
    patch_turn = _patch_turn("cost", "Costed Draft")
    patch_turn["cost_usd"] = 0.02
    first_args = _runner_args(runner, tmp_path, [patch_turn], "cost-shell-thread")
    first_args.graph_shell = True
    first_args.max_cost_usd = 0.01
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"
    assert first["total_cost_usd"] == 0.02
    assert first["finalize_receipt"] == {}
    wake_id = first["wake_id"]
    with sqlite3.connect(first_args.world_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = ? AND status = 'active'",
            (wake_id,),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = ?",
            (f"{wake_id}:finalize",),
        ).fetchone()[0] == 0

    second_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finalize", "finalize_graph", {})],
        "cost-shell-thread",
    )
    second_args.graph_shell = True
    second_args.resume = True
    second = await runner.run(second_args, print_report=False)
    assert second["wake_id"] == wake_id
    assert second["terminal_status"] == "published"
    assert second["finalize_receipt"]["commit_id"] == f"{wake_id}:finalize"
    with sqlite3.connect(first_args.world_db) as connection:
        # no duplicate patch on resume
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = ?",
            (f"{wake_id}:finalize",),
        ).fetchone()[0] == 1


async def test_graph_shell_deadline_boundary_resume_finalizes_same_wake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline stop keeps staged work and CLI resume finalizes the same wake.

    F-04 H-matrix: --live-deadline-seconds is available for fresh and --resume
    live runs (independent of cost caps) and cannot be exercised offline, so the
    deadline is injected at the model seam — the same invoke boundary
    LiveDeadlineGuard occupies in live runs. The deterministic subclass fires
    LiveDeadlineExceeded on the second dispatch so wall-clock timing cannot
    flake; the halt, staging preservation and resume path are the real graph
    code.
    """
    runner = _load_runner()
    from leave_information_bubble.world_agent.live_deadline import LiveDeadlineExceeded

    class DeadlineOnSecondDispatch(runner._ScriptedModel):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self._dispatches = 0

        async def invoke_tools(
            self, messages: object, **kwargs: object
        ) -> object:
            self._dispatches += 1
            if self._dispatches >= 2:
                raise LiveDeadlineExceeded()
            return await super().invoke_tools(messages, **kwargs)

    monkeypatch.setattr(runner, "_ScriptedModel", DeadlineOnSecondDispatch)
    # The second dispatch raises before consuming its fixture turn, so the
    # fixture holds only the one turn the wake actually executes.
    first_args = _runner_args(
        runner,
        tmp_path,
        [_patch_turn("deadline", "Deadlined Draft")],
        "deadline-shell-thread",
    )
    first_args.graph_shell = True
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"
    assert first["terminal_summary"].startswith("Live deadline reached before a model call")
    assert first["finalize_receipt"] == {}
    wake_id = first["wake_id"]
    with sqlite3.connect(first_args.world_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_objects WHERE wake_id = ? AND status = 'active'",
            (wake_id,),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = ?",
            (f"{wake_id}:finalize",),
        ).fetchone()[0] == 0

    second_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finalize", "finalize_graph", {})],
        "deadline-shell-thread",
    )
    second_args.graph_shell = True
    second_args.resume = True
    second = await runner.run(second_args, print_report=False)
    assert second["wake_id"] == wake_id
    assert second["terminal_status"] == "published"
    assert second["finalize_receipt"]["commit_id"] == f"{wake_id}:finalize"
    with sqlite3.connect(first_args.world_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM commit_receipts WHERE commit_id = ?",
            (f"{wake_id}:finalize",),
        ).fetchone()[0] == 1


async def test_graph_shell_resume_rejects_checkpoint_success_without_staging_or_ledger(
    tmp_path: Path,
) -> None:
    """Checkpoint patch-success counters cannot outlive both staging and ledger."""
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Ghost Success Draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            )
        ],
        "ghost-success-thread",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"
    wake_id = first["wake_id"]
    with sqlite3.connect(first_args.world_db) as connection:
        connection.execute("DELETE FROM staged_objects WHERE wake_id = ?", (wake_id,))
        connection.execute(
            "DELETE FROM staged_patch_receipts WHERE wake_id = ?", (wake_id,)
        )
        connection.commit()

    resume_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "ghost-success-thread",
    )
    resume_args.graph_shell = True
    resume_args.resume = True

    with pytest.raises(ValueError, match="no patch ledger exists"):
        await runner.run(resume_args, print_report=False)


async def test_graph_shell_cp1_resume_rejects_success_ledger_without_staged_row(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    world_path = tmp_path / "cp1-missing-world.sqlite3"
    runtime_path = tmp_path / "cp1-missing-runtime.sqlite3"
    store = runner.WorldStore(world_path)
    config = {"configurable": {"thread_id": "cp1-missing-thread"}}

    class FailAfterPatch(runner.WorldTools):
        async def execute(self, name: str, arguments: object, call_id: str) -> object:
            result = await super().execute(name, arguments, call_id)
            if name == "graph_patch":
                raise RuntimeError("CP1 after durable patch before tool checkpoint")
            return result

    model_path = tmp_path / "cp1-seed-model.json"
    _write_model(
        model_path,
        [
            _tool_turn(
                "cp1-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "cp1-durable-op",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Must not disappear",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            )
        ],
    )
    shell_tools = FailAfterPatch(
        store=store,
        adapters={},
        thread_id="cp1-missing-thread",
        wake_id="cp1-missing-wake",
        closed_wake_guard=True,
    )
    async with runner.aiosqlite.connect(runtime_path, isolation_level=None) as connection:
        graph = runner.build_world_agent_graph(
            model=runner._ScriptedModel(model_path),
            tools=shell_tools,
            store=store,
            checkpointer=runner.AsyncSqliteSaver(connection),
            thread_id="cp1-missing-thread",
            wake_id="cp1-missing-wake",
        )
        with pytest.raises(RuntimeError, match="CP1 after durable patch"):
            await graph.ainvoke(
                runner.graph_shell_initial_state(
                    messages=[{"role": "user", "content": "Patch once."}],
                    store=store,
                    thread_id="cp1-missing-thread",
                    wake_id="cp1-missing-wake",
                    domain_key="lol_cn",
                    mode="broad",
                    object_id=None,
                ),
                config,
            )

    with sqlite3.connect(world_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM staged_patch_receipts WHERE wake_id = 'cp1-missing-wake'"
        ).fetchone()[0] == 1
        connection.execute("DELETE FROM staged_objects WHERE wake_id = 'cp1-missing-wake'")
        connection.commit()

    resume_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "cp1-missing-thread",
    )
    resume_args.graph_shell = True
    resume_args.resume = True
    resume_args.world_db = world_path
    resume_args.runtime_db = runtime_path

    with pytest.raises(ValueError, match="successful durable patch receipt.*staging is missing"):
        await runner.run(resume_args, print_report=False)


async def test_graph_shell_fresh_start_rejects_another_active_wake(tmp_path: Path) -> None:
    """A fresh writer cannot silently abandon durable staging owned by another wake."""
    runner = _load_runner()
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "op-1",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Owned Draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            )
        ],
        "active-owner",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    first = await runner.run(first_args, print_report=False)
    assert first["terminal_status"] == "staged_unpublished"

    competing = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("must-not-run", "finalize_graph", {})],
        "competing-writer",
    )
    competing.graph_shell = True
    competing.world_db = first_args.world_db

    with pytest.raises(ValueError, match="active Graph Shell staging already exists"):
        await runner.run(competing, print_report=False)


async def test_graph_shell_serializes_different_wakes_before_active_staging_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent different wakes cannot both pass the world-writer startup check."""
    runner = _load_runner()
    original_scripted_model = runner._ScriptedModel
    both_invoked = asyncio.Event()
    invoked_count = 0

    class _CoordinatedScriptedModel(original_scripted_model):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self._owner = path.name
            self._invoke_count = 0

        async def invoke_tools(self, *args: Any, **kwargs: Any) -> ToolModelResponse:
            nonlocal invoked_count
            self._invoke_count += 1
            if self._invoke_count == 1:
                invoked_count += 1
                if invoked_count == 2:
                    both_invoked.set()
                with suppress(TimeoutError):
                    await asyncio.wait_for(both_invoked.wait(), timeout=0.5)
            return await super().invoke_tools(*args, **kwargs)

    monkeypatch.setattr(runner, "_ScriptedModel", _CoordinatedScriptedModel)
    first_args = _runner_args(
        runner,
        tmp_path,
        [
            _tool_turn(
                "first-patch",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "first-op",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "First Draft",
                                "kind": "concept",
                                "provisional": True,
                            },
                        }
                    ]
                },
            ),
        ],
        "first-writer",
    )
    first_args.graph_shell = True
    first_args.max_turns = 1
    second_args = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("second-finalize", "finalize_graph", {})],
        "second-writer",
    )
    second_args.graph_shell = True
    second_args.world_db = first_args.world_db
    second_args.runtime_db = first_args.runtime_db

    outcomes = await asyncio.gather(
        runner.run(first_args, print_report=False),
        runner.run(second_args, print_report=False),
        return_exceptions=True,
    )
    results = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    errors = [outcome for outcome in outcomes if isinstance(outcome, ValueError)]

    assert [result["terminal_status"] for result in results] == ["staged_unpublished"]
    assert len(errors) == 1
    assert "active Graph Shell staging already exists" in str(errors[0])


_DRIVER_SOURCE = """\
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import leave_information_bubble.world_agent.cli as cli

READY = os.environ.get("BARRIER_READY", "")
PROCEED = os.environ.get("BARRIER_PROCEED", "")


class _BarrierScriptedModel(cli._ScriptedModel):
    \"\"\"Replay model parking its FIRST dispatch on a cross-process barrier.\"\"\"

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._dispatched = False

    async def invoke_tools(self, messages: object, **kwargs: object) -> object:
        if READY and PROCEED and not self._dispatched:
            Path(READY).write_text("ready", encoding="utf-8")
            deadline = time.monotonic() + 120.0
            while not Path(PROCEED).exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("barrier wait timed out")
                await asyncio.sleep(0.05)
        self._dispatched = True
        return await super().invoke_tools(messages, **kwargs)


cli._ScriptedModel = _BarrierScriptedModel
args = cli.parse_args(sys.argv[1:])
result = asyncio.run(cli.run(args, print_report=False))
print(
    json.dumps(
        {
            "terminal_status": result.get("terminal_status"),
            "wake_id": result.get("wake_id"),
            "finalize_commit_id": (result.get("finalize_receipt") or {}).get("commit_id"),
        },
        ensure_ascii=False,
    )
)
"""


async def test_graph_shell_single_writer_lease_two_process_barrier(
    tmp_path: Path,
) -> None:
    """Only one process may become the world's active writer.

    F-05: two fresh CLI processes racing for the same world DB. The atomic
    singleton writer lease admits exactly one process into the model; the
    other fails closed with the typed world_write_in_progress error before
    any patch; the world holds at most one active wake and one owner. The
    loser stays rejected while the winner is staged; the winner resumes and
    publishes; only after publish may a new fresh wake take ownership.
    """
    repo_root = Path(__file__).resolve().parents[2]
    driver = tmp_path / "writer_driver.py"
    driver.write_text(_DRIVER_SOURCE, encoding="utf-8")
    world_db = tmp_path / "shared-world.sqlite3"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    def _argv(
        thread_id: str,
        turns: list[dict[str, object]],
        *,
        resume: bool = False,
        max_turns: int | None = None,
    ) -> list[str]:
        replay = tmp_path / f"{thread_id}-replay.json"
        model = tmp_path / f"{thread_id}-model.json"
        _write_replay_fixture(replay)
        _write_model(model, turns)
        argv = [
            "--perspective",
            "Validate a replay fixture.",
            "--world-db",
            str(world_db),
            "--runtime-db",
            str(tmp_path / f"{thread_id}-runtime.sqlite3"),
            "--thread-id",
            thread_id,
            "--replay-fixture",
            str(replay),
            "--scripted-model-fixture",
            str(model),
            "--graph-shell",
        ]
        if max_turns is not None:
            argv += ["--max-turns", str(max_turns)]
        if resume:
            argv.append("--resume")
        return argv

    def _spawn(thread_id: str, ready: Path) -> subprocess.Popen[bytes]:
        out = tmp_path / f"{thread_id}-out.txt"
        err = tmp_path / f"{thread_id}-err.txt"
        return subprocess.Popen(
            [
                sys.executable,
                str(driver),
                *_argv(
                    thread_id,
                    [_patch_turn(f"race-{thread_id}", "Racing Draft")],
                    max_turns=1,
                ),
            ],
            cwd=repo_root,
            env={**env, "BARRIER_READY": str(ready), "BARRIER_PROCEED": str(proceed)},
            stdout=out.open("wb"),
            stderr=err.open("wb"),
        )

    # Pre-migrate the shared world with one quiet published wake so the
    # racing processes contend on the writer lease, not on fresh-DB schema
    # migration (which fails closed untyped with "database is locked").
    setup_run = subprocess.run(
        [
            sys.executable,
            str(driver),
            *_argv(
                "setup-writer",
                [
                    _patch_turn("setup", "Setup Draft"),
                    _tool_turn("finalize", "finalize_graph", {}),
                ],
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert setup_run.returncode == 0, setup_run.stderr.decode("utf-8")
    setup_wake = json.loads(setup_run.stdout.decode("utf-8"))["wake_id"]

    proceed = tmp_path / "proceed.txt"
    racing = {"race-a": _spawn("race-a", tmp_path / "ready-race-a.txt"),
              "race-b": _spawn("race-b", tmp_path / "ready-race-b.txt")}
    try:
        deadline = time.monotonic() + 90.0
        loser_name: str | None = None
        while time.monotonic() < deadline:
            for name, process in racing.items():
                if process.poll() is not None:
                    loser_name = name
                    break
            if loser_name is not None:
                break
            await asyncio.sleep(0.05)
        assert loser_name is not None, "neither racing process exited; writer race hung"
        winner_name = "race-b" if loser_name == "race-a" else "race-a"
        loser, winner = racing[loser_name], racing[winner_name]
        assert loser.returncode != 0
        assert "world_write_in_progress" in (
            tmp_path / f"{loser_name}-err.txt"
        ).read_text(encoding="utf-8")
        # The winner is parked at its first model dispatch; release it.
        winner_ready = tmp_path / f"ready-{winner_name}.txt"
        ready_deadline = time.monotonic() + 30.0
        while not winner_ready.exists():
            assert time.monotonic() < ready_deadline, "winner never reached the barrier"
            await asyncio.sleep(0.05)
        proceed.write_text("go", encoding="utf-8")
        assert winner.wait(timeout=60) == 0
        winner_result = json.loads(
            (tmp_path / f"{winner_name}-out.txt").read_text(encoding="utf-8")
        )
        assert winner_result["terminal_status"] == "staged_unpublished"
        winner_wake = winner_result["wake_id"]
        with sqlite3.connect(world_db) as connection:
            active_wakes = {
                str(row[0])
                for table in ("staged_objects", "staged_assertions", "staged_inquiries")
                for row in connection.execute(
                    f"SELECT DISTINCT wake_id FROM {table} WHERE status = 'active'"
                ).fetchall()
            }
            assert active_wakes == {winner_wake}
            owners = [
                row[0]
                for row in connection.execute(
                    "SELECT owner_wake_id FROM graph_shell_writer_leases"
                ).fetchall()
            ]
            assert owners == [winner_wake]
            claim_wakes = {
                str(row[0])
                for row in connection.execute(
                    "SELECT wake_id FROM graph_shell_wake_claims"
                ).fetchall()
            }
            assert claim_wakes == {setup_wake, winner_wake}

        # While the winner still owns staging, a fresh retry stays rejected.
        loser_retry = subprocess.run(
            [
                sys.executable,
                str(driver),
                *_argv(
                    loser_name,
                    [_patch_turn(f"retry-{loser_name}", "Retry Draft")],
                    max_turns=1,
                ),
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert loser_retry.returncode != 0
        assert "world_write_in_progress" in loser_retry.stderr.decode("utf-8")

        # Same-wake resume continues and publishes, releasing the lease.
        resume_run = subprocess.run(
            [
                sys.executable,
                str(driver),
                *_argv(winner_name, [_tool_turn("finalize", "finalize_graph", {})], resume=True),
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert resume_run.returncode == 0, resume_run.stderr.decode("utf-8")
        resume_result = json.loads(resume_run.stdout.decode("utf-8"))
        assert resume_result["terminal_status"] == "published"
        assert resume_result["finalize_commit_id"] == f"{winner_wake}:finalize"
        with sqlite3.connect(world_db) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM graph_shell_writer_leases"
            ).fetchone()[0] == 0

        # After publish, a new fresh wake obtains ownership and publishes.
        fresh_run = subprocess.run(
            [
                sys.executable,
                str(driver),
                *_argv(
                    "post-publish",
                    [
                        _patch_turn("fresh", "Fresh Draft"),
                        _tool_turn("finalize", "finalize_graph", {}),
                    ],
                ),
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert fresh_run.returncode == 0, fresh_run.stderr.decode("utf-8")
        assert json.loads(fresh_run.stdout.decode("utf-8"))["terminal_status"] == "published"
        with sqlite3.connect(world_db) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM graph_shell_writer_leases"
            ).fetchone()[0] == 0
    finally:
        for process in racing.values():
            if process.poll() is None:
                process.kill()


async def test_runner_resume_restores_checkpoint_wake_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    """--resume restores the checkpoint's wake identity; an explicit --wake-id
    that differs fails closed; the resumed halted wake continues to finalize
    under the same identity (G5b-2: resume reopens the wake and re-dispatches
    the model; no legacy commit identity is ever minted)."""
    runner = _load_runner()
    halted_fixture = [
        _tool_turn("fail-1", "discover_sources", {"adapter": "nope", "query": "x"}),
        _tool_turn("fail-2", "discover_sources", {"adapter": "nope", "query": "x"}),
        _tool_turn("fail-3", "discover_sources", {"adapter": "nope", "query": "x"}),
    ]
    args = _runner_args(runner, tmp_path, halted_fixture, "resume-wake")
    args.wake_id = "wake-original"

    first = await runner.run(args)

    assert first["halted"] is True
    assert "Tool error loop" in first["terminal_summary"]
    assert first["terminal_status"] == "staged_unpublished"
    assert first["finalize_receipt"] == {}

    mismatch = _runner_args(runner, tmp_path, [], "resume-wake")
    mismatch.resume = True
    mismatch.wake_id = "wake-wrong"
    with pytest.raises(ValueError, match="does not match checkpoint wake identity"):
        await runner.run(mismatch)

    restored = _runner_args(
        runner,
        tmp_path,
        [_tool_turn("finish-resumed", "finalize_graph", {})],
        "resume-wake",
    )
    restored.resume = True
    second = await runner.run(restored)

    assert second["wake_id"] == "wake-original"
    assert second["terminal_status"] == "published"
    assert second["finalize_receipt"]["commit_id"] == "wake-original:finalize"


async def test_runner_rejects_unconsumed_scripted_turns(tmp_path: Path) -> None:
    """Catch replay declaring success while ignoring unexpected model turns."""
    runner = _load_runner()
    turns = [
        _tool_turn("finish", "finalize_graph", {}),
        _tool_turn("unexpected", "finalize_graph", {}),
    ]

    with pytest.raises(RuntimeError, match="1 unconsumed scripted model turn"):
        await runner.run(_runner_args(runner, tmp_path, turns, "extra"))


async def test_runner_reports_total_open_inquiries_beyond_summary_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catch the ten-row summary limit being reported as the total open count.

    G5b-2: the wake creates eleven inquiries in one graph_patch against the
    seeded formal object and finalizes; the report must count all eleven as
    created (staging stats are not capped) while the receipt shows the wake
    published exactly those eleven formal inquiries and no objects of its
    own (the seed commit belongs to a different commit id).
    """
    runner = _load_runner()
    WorldStore(tmp_path / "inquiry-count-world.sqlite3").memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="center", kind=ObjectKind.EVENT, canonical_name="Center")]
        ),
        "seed-inquiry-count",
    )
    # Prompts must stay below the calibrated inquiry-similarity threshold
    # (0.2) or commit-time dedup interception rejects them as duplicates.
    prompts = [
        "首局的禁用选择是否针对对方中单？",
        "第二局经济曲线为何出现断档？",
        "第三局的团战处理与龙魂节奏怎样？",
        "第四局地图资源被压制的原因是什么？",
        "第五局英雄池深度能否支撑翻盘？",
        "打野选手野区规划是否存在漏洞？",
        "下路对线期换血策略是否过于激进？",
        "中单选手对线压制的细节表现如何？",
        "上单选手单带时机的选择是否合理？",
        "辅助视野布控的覆盖率是否达标？",
        "教练组临场 BP 的应变能力怎样评价？",
    ]
    patch_items = [
        {
            "op_id": f"question-{index}",
            "kind": "inquiry",
            "action": "create",
            "payload": {
                "subject_ref": "center",
                "prompt": prompt,
                "rationale": "Keep this branch open.",
            },
        }
        for index, prompt in enumerate(prompts)
    ]

    await runner.run(
        _runner_args(
            runner,
            tmp_path,
            [
                _tool_turn("eleven-inquiries", "graph_patch", {"items": patch_items}),
                _tool_turn("finalize-inquiries", "finalize_graph", {}),
            ],
            "inquiry-count",
        )
    )

    output = capsys.readouterr().out
    # all eleven inquiry creations count, and the receipt attributes the
    # inquiries to this wake while the seeded object stays a separate commit
    assert "持久化：objects +0 / assertions +0 / inquiries +11" in output
    assert "质询动作：新建=11" in output
    assert "finalized=11" in output
    assert "下轮入口：" in output


def _run_args(
    runner: ModuleType,
    *,
    mission: str,
    world_db: Path,
    runtime_db: Path,
    thread_id: str,
    replay_path: Path,
    model_path: Path,
    mode: str = "broad",
    object_id: str | None = None,
) -> object:
    values = [
        "--perspective",
        mission,
        "--world-db",
        str(world_db),
        "--runtime-db",
        str(runtime_db),
        "--thread-id",
        thread_id,
        "--mode",
        mode,
        "--max-turns",
        "12",
        "--replay-fixture",
        str(replay_path),
        "--scripted-model-fixture",
        str(model_path),
    ]
    if object_id is not None:
        values.extend(["--object-id", object_id])
    return runner.parse_args(values)


async def test_fresh_broad_repeat_broad_deep_and_recall_share_only_world_memory(
    tmp_path: Path,
) -> None:
    """D6 vertical regression: one world graph across replaceable wakes.

    Fresh broad wake builds and publishes the map (search → discover →
    patch → finalize); a repeat wake adds no novelty and finalizes an empty
    working graph honestly; a deep wake expands, reads, patches, answers and
    resolves an inquiry; a direct recall graph re-reads the same formal ids.
    Every wake runs the single normal Graph Shell surface — no submit
    proposal, no legacy commit or recovery path is reachable (D2/D6).
    """
    runner = _load_runner()
    replay_path = tmp_path / "replay.json"
    world_db = tmp_path / "world.sqlite3"
    runtime_db = tmp_path / "runtime.sqlite3"
    broad_model = tmp_path / "broad.json"
    repeat_model = tmp_path / "repeat.json"
    deep_model = tmp_path / "deep.json"
    _write_replay_fixture(replay_path)
    _write_model(
        broad_model,
        [
            _tool_turn("broad-recall", "memory_recent", {}),
            _tool_turn(
                "scan-centers",
                "discover_sources",
                {"adapter": "replay", "query": "League of Legends Chinese community", "limit": 5},
            ),
            _tool_turn(
                "patch-centers",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "center-ig-lng",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "IG vs LNG playoff series",
                                "kind": "event",
                                "event_time_start": "2026-08-22T12:00:00+00:00",
                                "aliases": ["IG"],
                                "domain_hints": ["League of Legends", "Chinese community"],
                            },
                        },
                        {
                            "op_id": "center-blg",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "BLG roster announcement",
                                "kind": "event",
                                "event_time_start": "2026-08-23T12:00:00+00:00",
                                "domain_hints": ["League of Legends", "Chinese community"],
                            },
                        },
                        {
                            "op_id": "center-fearless",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "LPL fearless-draft rule change",
                                "kind": "event",
                                "event_time_start": "2026-08-23T12:00:00+00:00",
                                "domain_hints": ["League of Legends", "Chinese community"],
                            },
                        },
                        # the deterministic compile gate rejects isolated new
                        # objects and unanchored events: the three centers form
                        # a relation cycle so every one is connected and anchored
                        {
                            "op_id": "rel-ig-blg",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "ac-b1:s1",
                                "predicate": "related_to",
                                "object_ref": "ac-b1:s2",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.7,
                                "evidence": [_obs_id("ig-lng-series")],
                            },
                        },
                        {
                            "op_id": "rel-blg-fearless",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "ac-b1:s2",
                                "predicate": "related_to",
                                "object_ref": "ac-b1:s3",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.7,
                                "evidence": [_obs_id("ig-lng-series")],
                            },
                        },
                        {
                            "op_id": "rel-fearless-ig",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "ac-b1:s3",
                                "predicate": "related_to",
                                "object_ref": "ac-b1:s1",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.7,
                                "evidence": [_obs_id("ig-lng-series")],
                            },
                        },
                        {
                            "op_id": "inq-ig-lng",
                            "kind": "inquiry",
                            "action": "create",
                            "payload": {
                                "subject_ref": "ac-b1:s1",
                                "prompt": "How did the score and community language combine "
                                "around IG vs LNG?",
                                "rationale": "Resolving this would materially improve "
                                "the center explanation.",
                            },
                        },
                        {
                            "op_id": "inq-blg",
                            "kind": "inquiry",
                            "action": "create",
                            "payload": {
                                "subject_ref": "ac-b1:s2",
                                "prompt": "What changed in the BLG roster?",
                                "rationale": "Resolving this would materially improve "
                                "the center explanation.",
                            },
                        },
                        {
                            "op_id": "inq-fearless",
                            "kind": "inquiry",
                            "action": "create",
                            "payload": {
                                "subject_ref": "ac-b1:s3",
                                "prompt": "How will the new draft format shift champion priority?",
                                "rationale": "Resolving this would materially improve "
                                "the center explanation.",
                            },
                        },
                    ]
                },
            ),
            _tool_turn("finalize-broad", "finalize_graph", {}),
        ],
    )

    first_broad_args = _run_args(
        runner,
        mission="Observe broadly and build a reusable map.",
        world_db=world_db,
        runtime_db=runtime_db,
        thread_id="acceptance-broad-1",
        replay_path=replay_path,
        model_path=broad_model,
    )
    first_broad_args.wake_id = "ac-b1"
    first_broad = await runner.run(first_broad_args)

    assert first_broad["wake_id"] == "ac-b1"
    assert first_broad["terminal_status"] == "published"
    assert first_broad["finalize_receipt"]["commit_id"] == "ac-b1:finalize"
    with sqlite3.connect(world_db) as connection:
        connection.row_factory = sqlite3.Row
        centers = {
            row["canonical_name"]: row["id"]
            for row in connection.execute("SELECT id, canonical_name FROM objects")
        }
        center_id = centers["IG vs LNG playoff series"]
        inquiry_id = connection.execute(
            "SELECT id FROM inquiries WHERE subject_id = ?", (center_id,)
        ).fetchone()["id"]
        before_repeat = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("objects", "assertions", "observations", "inquiries")
        }
        # the staged ids become the formal ids at finalize — the recall wake
        # and the deep fixture reference them verbatim (never invented)
        assert center_id == "ac-b1:s1"
        assert inquiry_id == "ac-b1:i1"

    _write_model(
        repeat_model,
        [
            _tool_turn("repeat-recall", "memory_search", {"query": "LoL current centers"}),
            _tool_turn(
                "scan-centers",
                "discover_sources",
                {"adapter": "replay", "query": "League of Legends Chinese community", "limit": 5},
            ),
            # no material novelty: finalizing the empty working graph is an
            # honest explicit publication decision (mechanics), and the repeat
            # wake's commit receipt carries no object/assertion/inquiry ids
            _tool_turn("finalize-repeat", "finalize_graph", {}),
        ],
    )
    repeated_broad = await runner.run(
        _run_args(
            runner,
            mission="Observe the same window and add only material novelty.",
            world_db=world_db,
            runtime_db=runtime_db,
            thread_id="acceptance-broad-2",
            replay_path=replay_path,
            model_path=repeat_model,
        )
    )
    with sqlite3.connect(world_db) as connection:
        after_repeat = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("objects", "assertions", "observations", "inquiries")
        }

    _write_model(
        deep_model,
        [
            _tool_turn(
                "deep-expand",
                "memory_expand",
                {"object_ids": [center_id], "depth": 2, "limit": 15},
            ),
            _tool_turn(
                "open-ig-lng",
                "open_source",
                {"observation_id": _obs_id("ig-lng-series")},
            ),
            _tool_turn(
                "sample-ig-lng",
                "sample_discussion",
                {"observation_id": _obs_id("ig-lng-community")},
            ),
            _tool_turn(
                "patch-deep",
                "graph_patch",
                {
                    "items": [
                        {
                            "op_id": "obj-renchuanren",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Renchuanren trans-roster fan identity",
                                "kind": "concept",
                            },
                        },
                        {
                            "op_id": "obj-ig-2018",
                            "kind": "object",
                            "action": "create",
                            "payload": {
                                "canonical_name": "Invictus Gaming wins the 2018 World Championship",
                                "kind": "event",
                                "event_time_start": "2018-11-03T00:00:00Z",
                            },
                        },
                        {
                            "op_id": "assert-result",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": center_id,
                                "predicate": "has_result",
                                "literal": "The replay match record reports that IG defeated LNG 3-1.",
                                "epistemic_role": "fact",
                                "confidence": 0.95,
                                "answers_ref": inquiry_id,
                                "evidence": [_obs_id("ig-lng-series")],
                            },
                        },
                        {
                            "op_id": "assert-expresses",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": center_id,
                                "predicate": "expresses",
                                "literal": "Sampled commenters framed the win through the label Renchuanren.",
                                "epistemic_role": "community_view",
                                "confidence": 0.76,
                                "evidence": [_community_obs_id()],
                            },
                        },
                        {
                            "op_id": "assert-explains",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": center_id,
                                "predicate": "explains",
                                "object_ref": "ac-deep:s1",
                                "epistemic_role": "semantic_explanation",
                                "confidence": 0.78,
                                "evidence": [_community_obs_id()],
                            },
                        },
                        {
                            "op_id": "assert-related",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": center_id,
                                "predicate": "related_to",
                                "object_ref": "ac-deep:s2",
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.7,
                                "evidence": [_community_obs_id()],
                            },
                        },
                        # deterministic compile: the 2018 event is only ever an
                        # object_ref elsewhere, so it needs its own relation
                        # anchor to survive the event_identity_anchor gate
                        {
                            "op_id": "assert-anchor-2018",
                            "kind": "assertion",
                            "action": "create",
                            "payload": {
                                "subject_ref": "ac-deep:s2",
                                "predicate": "participant_in",
                                "object_ref": center_id,
                                "epistemic_role": "agent_synthesis",
                                "confidence": 0.7,
                                "evidence": [_community_obs_id()],
                            },
                        },
                        {
                            "op_id": "inq-framing",
                            "kind": "inquiry",
                            "action": "create",
                            "payload": {
                                "subject_ref": center_id,
                                "prompt": "How widely was Renchuanren used beyond the twelve-comment sample?",
                                "rationale": "A broader sample would bound how representative "
                                "the framing was.",
                            },
                        },
                        {
                            "op_id": "resolve-score",
                            "kind": "inquiry",
                            "action": "resolve",
                            "target_ref": inquiry_id,
                            "payload": {
                                "expected_version": 1,
                                "answers_ref": "ac-deep:a1",
                            },
                        },
                    ]
                },
            ),
            _tool_turn("finalize-deep", "finalize_graph", {}),
        ],
    )
    deep_args = _run_args(
        runner,
        mission="Build a coherent local explanation around IG vs LNG.",
        world_db=world_db,
        runtime_db=runtime_db,
        thread_id="acceptance-deep",
        replay_path=replay_path,
        model_path=deep_model,
        mode="deep",
        object_id=center_id,
    )
    deep_args.wake_id = "ac-deep"
    deep = await runner.run(deep_args)
    assert deep["wake_id"] == "ac-deep"
    assert deep["terminal_status"] == "published"
    assert deep["finalize_receipt"]["commit_id"] == "ac-deep:finalize"

    recall_model = _AdaptiveRecallModel()
    recall_store = runner.WorldStore(world_db)
    recall_tools = runner.WorldTools(
        store=recall_store,
        adapters=runner._replay(replay_path),
        thread_id="acceptance-recall",
        wake_id="ac-recall",
        closed_wake_guard=True,
    )
    async with runner.aiosqlite.connect(runtime_db, isolation_level=None) as connection:
        recalled = await runner.build_world_agent_graph(
            model=recall_model,
            tools=recall_tools,
            store=recall_store,
            checkpointer=runner.AsyncSqliteSaver(connection),
            thread_id="acceptance-recall",
            wake_id="ac-recall",
            max_turns=6,
        ).ainvoke(
            runner.graph_shell_initial_state(
                messages=[{"role": "user", "content": "Recall the IG vs LNG graph."}],
                store=recall_store,
                thread_id="acceptance-recall",
                wake_id="ac-recall",
                domain_key="lol_cn",
                mode="broad",
                object_id=None,
            ),
            {"configurable": {"thread_id": "acceptance-recall"}},
        )

    assert first_broad["turn_count"] == 4
    assert repeated_broad["turn_count"] == 3
    assert (
        before_repeat
        == after_repeat
        == {
            "objects": 3,
            "assertions": 3,
            "observations": 4,
            "inquiries": 3,
        }
    )
    assert deep["turn_count"] == 5
    assert recalled["turn_count"] == 3

    with sqlite3.connect(world_db) as connection:
        connection.row_factory = sqlite3.Row
        assertions = connection.execute(
            "SELECT * FROM assertions WHERE subject_id = ? ORDER BY epistemic_role", (center_id,)
        ).fetchall()
        inquiries = connection.execute("SELECT * FROM inquiries ORDER BY id").fetchall()
        evidence = connection.execute(
            "SELECT a.epistemic_role, ae.observation_id, ae.role "
            "FROM assertion_evidence ae JOIN assertions a ON a.id = ae.assertion_id "
            "WHERE a.subject_id = ? ORDER BY a.epistemic_role",
            (center_id,),
        ).fetchall()

    assert {row["epistemic_role"] for row in assertions} == {
        "fact",
        "community_view",
        "semantic_explanation",
        "agent_synthesis",
    }
    # the center's relation assertions reach both new deep-wake objects (the
    # broad wake's relation cycle adds its own center refs, so containment
    # is the truthful check here)
    assert {"ac-deep:s1", "ac-deep:s2"} <= {
        row["object_id"] for row in assertions if row["object_id"] is not None
    }
    assert any(row["observation_id"] == _obs_id("ig-lng-series") for row in evidence)
    assert any(row["observation_id"] == _community_obs_id() for row in evidence)
    assert next(row for row in inquiries if row["id"] == inquiry_id)["status"] == "resolved"
    open_prompts = {row["prompt"] for row in inquiries if row["status"] == "open"}
    assert "How widely was Renchuanren used beyond the twelve-comment sample?" in open_prompts

    search_center = next(
        item
        for item in recall_model.search_memory["anchor_objects"]
        if item["canonical_name"] == "IG vs LNG playoff series"
    )
    assert search_center["id"] == recall_model.expanded_object_id == center_id
    neighbor_names = {item["canonical_name"] for item in recall_model.expansion_memory["neighboring_objects"]}
    assert {
        "Renchuanren trans-roster fan identity",
        "Invictus Gaming wins the 2018 World Championship",
    } <= neighbor_names
    # G5b-2: the legacy observation-links tables are retired; durable
    # provenance rides the assertion evidence refs, which must cite the
    # exact observation ids the deep wake attached to its assertions
    assert recall_model.expansion_memory["evidence_refs"]

    with sqlite3.connect(runtime_db) as connection:
        thread_ids = {row[0] for row in connection.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    assert thread_ids == {
        "acceptance-broad-1",
        "acceptance-broad-2",
        "acceptance-deep",
        "acceptance-recall",
    }
    assert world_db.resolve() != runtime_db.resolve()


async def test_runner_forwards_reasoning_profile_into_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The composition root owns the ONE frozen reasoning profile for the wake.

    Task 5.3: cli.py threads the single WakeReasoningProfile instance into
    build_world_agent_graph, so the whole wake — exploration and the forced
    proposal call — is built from the same frozen profile and differs only in
    tool_choice.
    """
    runner = _load_runner()
    received: list[object] = []
    original = runner.build_world_agent_graph

    def recording(**kwargs: object) -> object:
        received.append(kwargs.get("reasoning_profile"))
        return original(**kwargs)

    monkeypatch.setattr(runner, "build_world_agent_graph", recording)
    result = await runner.run(
        _runner_args(
            runner,
            tmp_path,
            [_tool_turn("profile-finish", "finalize_graph", {})],
            "profile-run",
        )
    )

    assert received == [WakeReasoningProfile()]
    assert result["finalize_receipt"]["commit_id"] == f"{result['wake_id']}:finalize"


def test_wake_reasoning_profile_max_tokens_follows_thinking() -> None:
    """Thinking wakes need the provider reasoning budget, not the off-mode cap.

    Live canary w2 showed completion pinned at the 8192 ceiling with zero
    tool calls escaping; the chain consumed the whole output budget before a
    tool call was written. Off-mode keeps the original ceiling, and an
    explicit max_tokens always wins.
    """
    assert WakeReasoningProfile().max_tokens == 8192
    assert WakeReasoningProfile(thinking=True).max_tokens == 32768
    assert WakeReasoningProfile(thinking=True, max_tokens=16384).max_tokens == 16384
    assert (
        WakeReasoningProfile(thinking=True, reasoning_effort="high").max_tokens
        == 32768
    )
