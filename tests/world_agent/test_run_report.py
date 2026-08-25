"""Focused contracts for the compact end-of-wake cognition report (G5b-2).

The legacy proposal/amendment review pins (safe-subset completeness, review
issue actions, omitted-resolution reasons) were retired with the proposal
runtime; replacement coverage lives in the working-graph staging stats,
explicit finalize receipt, and wake-scoped model ledger asserted here.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from leave_information_bubble.gateway.client import ToolModelResponse
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
from leave_information_bubble.world_agent.model_calls import (
    ModelCallRecorder,
    ModelRequestEnvelope,
)
from leave_information_bubble.world_agent.run_report import build_run_report, render_run_report


def test_graph_shell_report_projects_identity_staging_and_publication(tmp_path: Path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO staged_objects (staged_id, wake_id, status, target_ref, kind,"
            " canonical_name, domain_hints_json, aliases_json, created_at, updated_at, version)"
            " VALUES ('wake-report:s1', 'wake-report', 'active', NULL, 'concept',"
            " 'Draft', '[]', '[]', '2026-08-20', '2026-08-20', 1)"
        )
    result = {
        "execution_mode": "graph_shell",
        "thread_id": "thread-report",
        "wake_id": "wake-report",
        "terminal_status": "staged_unpublished",
        "terminal_summary": "resume this wake",
        "resume_allowed": True,
        "patch_success_count": 1,
        "tool_events": [],
    }

    report = build_run_report(
        thread_id="thread-report",
        wake_id="wake-report",
        domain="lol_cn",
        mission="draft",
        mode="broad",
        object_id=None,
        result=result,
        store=store,
        runtime_path=tmp_path / "runtime.sqlite3",
        wall_ms=10,
    )

    assert report["entry"] | {} == {
        "thread_id": "thread-report",
        "wake_id": "wake-report",
        "execution_mode": "graph_shell",
        "domain": "lol_cn",
        "mission": "draft",
        "mode": "broad",
        "deep_seed": None,
    }
    assert report["execution"]["status"] == "staged_unpublished"
    assert report["working_graph"]["active_total"] == 1
    assert report["working_graph"]["by_status"] == {
        "active": 1,
        "abandoned": 0,
        "finalized": 0,
    }
    assert report["publication"] == {
        "published": False,
        "status": "",
        "commit_id": None,
        "receipt": {},
        "resume_allowed": True,
        "resume_count": 0,
        "patch_success_count": 1,
        "patch_replay_count": 0,
        "finalize_status_counts": {},
        "replayed": False,
        "recovery_action": (
            "working graph remains staged; resume this wake and call finalize_graph to publish"
        ),
    }
    assert report["graph_actions"] == {
        "graph_inspect": 0,
        "graph_diff": 0,
        "graph_patch": 0,
        "finalize_graph": 0,
    }


def test_report_uses_only_this_wakes_receipt_and_explicit_signals(tmp_path: Path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    now = datetime(2026, 8, 12, tzinfo=UTC)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="old", kind=ObjectKind.CONCEPT, canonical_name="旧对象")]),
        "seed-old",
    )
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="new-a", kind=ObjectKind.CONCEPT, canonical_name="新中心"),
                ObjectInput(id="new-b", kind=ObjectKind.CONCEPT, canonical_name="关联对象"),
            ],
            observations=[
                ObservationInput(
                    id="obs-1",
                    source_uri="https://example.test/1",
                    source_kind="web",
                    title="来源一",
                    depth=ObservationDepth.CONTENT,
                    observed_at=now,
                )
            ],
            assertions=[
                AssertionInput(
                    id="assert-1",
                    subject_id="new-a",
                    predicate="related_to",
                    object_id="new-b",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                )
            ],
            inquiries=[
                InquiryInput(
                    id="inq-1",
                    subject_id="new-a",
                    prompt="这条连接为何出现？",
                    rationale="仍需深挖",
                )
            ],
        ),
        "wake-1:finalize",
    )
    runtime = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime)
    recorder.append(
        thread_id="thread-1",
        wake_id="wake-1",
        turn=1,
        purpose="explore",
        response=ToolModelResponse(content="", model="test", cost_usd=0.001, latency_ms=25),
    )
    result = {
        "turn_count": 2,
        "terminal_summary": "Committed",
        "commit_receipt": {"commit": {}},
        "tool_events": [
            {
                "call_id": "scan-1",
                "turn": 1,
                "name": "search_sources",
                "arguments": {"adapter": "public-web", "query": "新中心"},
                "ok": True,
                "limitations": [],
                "elapsed_ms": 20.0,
                "card": {"id": "obs-1", "title": "来源一"},
                "unresolved": ["来源之间是否存在历史联系"],
            }
        ],
    }

    report = build_run_report(
        thread_id="thread-1",
        wake_id="wake-1",
        domain="lol_cn",
        mission="开放探索",
        mode="deep",
        object_id="new-a",
        result=result,
        store=store,
        runtime_path=runtime,
        wall_ms=100,
    )

    assert report["durable_diff"] == {
        "objects": 2,
        "assertions": 1,
        "inquiries": 1,
        "resolved": 0,
        "evidence_links": 1,
        "object_names": ["新中心", "关联对象"],
        "inquiry_prompts": ["这条连接为何出现？"],
    }
    assert "旧对象" not in render_run_report(report)
    assert report["connections"] == ["新中心 -related_to→ 关联对象"]
    assert report["curiosity"]["observed"] is True
    assert report["model"]["successful_calls"] == 1
    assert report["next_entries"][0].startswith("新中心：这条连接为何出现？")


def test_report_degrades_without_model_ledger_and_does_not_invent_curiosity(tmp_path: Path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    report = build_run_report(
        thread_id="deadline",
        wake_id="deadline",
        domain="lol_cn",
        mission="运行到边界",
        mode="broad",
        object_id=None,
        result={
            "turn_count": 1,
            "halted": True,
            "transition_reason": "live_deadline_pre_call",
            "terminal_summary": "deadline",
            "tool_events": [
                {
                    "call_id": "limited-1",
                    "name": "open_source",
                    "ok": False,
                    "limitations": ["tool_timeout"],
                    "elapsed_ms": 10,
                    "diagnostic": {
                        "outcome": "unavailable",
                        "error": {
                            "code": "tool_timeout",
                            "stage": "full_hydrate",
                            "attempts": 1,
                            "same_request_can_change": True,
                            "change_condition": "provider_response_before_configured_timeout",
                        },
                    },
                }
            ],
        },
        store=store,
        runtime_path=tmp_path / "missing.sqlite3",
        wall_ms=10,
    )

    assert report["execution"]["status"] == "incomplete"
    assert report["model"]["available"] is False
    assert report["durable_diff"]["objects"] == 0
    assert report["curiosity"] == {
        "observed": False,
        "topics": [],
        "note": "no explicit curiosity signal observed; no inference made",
    }
    assert report["issues"]["hard"] == ["deadline"]
    assert report["tools"]["failed"] == 1
    assert report["tools"]["limited"] == 1
    assert report["tools"]["diagnostics"] == [
        {
            "call_id": "limited-1",
            "name": "open_source",
            "ok": False,
            "limitations": ["tool_timeout"],
            "outcome": "unavailable",
            "error": {
                "code": "tool_timeout",
                "stage": "full_hydrate",
                "attempts": 1,
                "same_request_can_change": True,
                "change_condition": "provider_response_before_configured_timeout",
            },
        }
    ]
    assert report["tools"]["diagnostics_truncated"] is False
    assert report["issues"]["soft"] == []


def test_model_summary_filters_calls_by_wake_id(tmp_path: Path) -> None:
    """A runtime ledger holding several wakes under one thread reports only the
    current wake's calls when a wake identity is given (Slice 1: run-report
    model section follows the same wake, not the whole thread; D5: thread_id
    is never a stand-in for wake_id)."""
    runtime = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime)
    recorder.append(
        thread_id="shared-thread",
        wake_id="wake-other",
        turn=1,
        purpose="explore",
        response=ToolModelResponse(content="", model="test", cost_usd=0.01, latency_ms=30),
    )
    recorder.append(
        thread_id="shared-thread",
        wake_id="wake-current",
        turn=1,
        purpose="explore",
        response=ToolModelResponse(content="", model="test", cost_usd=0.001, latency_ms=25),
    )
    base_result = {"turn_count": 1, "terminal_summary": "", "tool_events": []}

    current = build_run_report(
        thread_id="shared-thread",
        wake_id="wake-current",
        domain="lol_cn",
        mission="开放探索",
        mode="broad",
        object_id=None,
        result=base_result,
        store=WorldStore(tmp_path / "world.sqlite3"),
        runtime_path=runtime,
        wall_ms=10,
    )
    assert current["model"]["successful_calls"] == 1
    assert current["model"]["cost_usd"] == 0.001


def test_recovery_without_a_durable_receipt_is_incomplete(tmp_path: Path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    report = build_run_report(
        thread_id="empty-recovery",
        wake_id="empty-recovery",
        domain="lol_cn",
        mission="recover",
        mode="deep",
        object_id=None,
        result={
            "in_recovery": True,
            "terminal_summary": "Recovery: no recoverable cognition",
            "tool_events": [],
        },
        store=store,
        runtime_path=tmp_path / "runtime.sqlite3",
        wall_ms=1,
    )

    assert report["execution"]["status"] == "incomplete"
    assert report["issues"]["hard"] == ["Recovery: no recoverable cognition"]


def test_run_report_displays_fingerprint_and_existing_cache_token_split(
    tmp_path: Path,
) -> None:
    """The model section shows per-call purpose/phase/fingerprint and the cache split."""
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime)
    envelope = ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=True,
        reasoning_effort="high",
        tool_schemas=[{"type": "function", "function": {"name": "memory_recent"}}],
        tool_choice=None,
        response_format=None,
        max_tokens=None,
    )
    recorder.append(
        thread_id="fp-wake",
        wake_id="fp-wake",
        turn=1,
        purpose="explore",
        phase="exploration",
        request_envelope=envelope,
        request_source="provider_effective",
        request_model=envelope.model,
        response=ToolModelResponse(
            content="",
            message={"role": "assistant", "content": ""},
            model="echoed-model",
            cached_input_tokens=4,
            uncached_input_tokens=6,
            completion_tokens=7,
            cost_usd=0.001,
            latency_ms=25,
        ),
    )
    report = build_run_report(
        thread_id="fp-wake",
        wake_id="fp-wake",
        domain="lol_cn",
        mission="指纹审计",
        mode="deep",
        object_id=None,
        result={"turn_count": 1, "tool_events": []},
        store=store,
        runtime_path=runtime,
        wall_ms=10,
    )

    calls = report["model"]["calls"]
    assert calls == [
        {
            "purpose": "explore",
            "phase": "exploration",
            "request_model": "deepseek-v4-flash",
            "response_model": "echoed-model",
            "request_source": "provider_effective",
            "request_fingerprint": envelope.fingerprint(),
            "cached_input_tokens": 4,
            "uncached_input_tokens": 6,
        }
    ]
    rendered = render_run_report(report)
    assert "explore/exploration" in rendered
    assert envelope.fingerprint()[:12] in rendered
    assert "cached=4" in rendered
    assert "uncached=6" in rendered
    # request alias and echoed response model render as separate values, and
    # the persisted request_source never degrades to "-"
    assert "req=deepseek-v4-flash resp=echoed-model" in rendered
    assert "src=provider_effective" in rendered
    assert "src=-" not in rendered


def test_run_report_renders_both_request_sources_without_degrading_to_dash(
    tmp_path: Path,
) -> None:
    """A report mixing provider_effective and caller_requested_fallback rows
    renders each source and its request alias; neither source degrades to "-"
    and the fallback's unknown alias stays unknown (empty, never the echo)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    recorder = ModelCallRecorder(runtime)
    envelope = ModelRequestEnvelope(
        model="deepseek-v4-flash",
        thinking=True,
        reasoning_effort="high",
        tool_schemas=[{"type": "function", "function": {"name": "memory_recent"}}],
        tool_choice=None,
        response_format=None,
        max_tokens=None,
    )
    recorder.append(
        thread_id="both-src",
        wake_id="both-src",
        turn=1,
        purpose="explore",
        phase="exploration",
        request_envelope=envelope,
        request_source="provider_effective",
        request_model=envelope.model,
        response=ToolModelResponse(content="", model="echoed-model", cost_usd=0.001, latency_ms=10),
    )
    recorder.append(
        thread_id="both-src",
        wake_id="both-src",
        turn=2,
        purpose="proposal",
        phase="proposal",
        request_envelope=None,
        request_source="caller_requested_fallback",
        request_model="",
        response=ToolModelResponse(content="", model="echoed-model", cost_usd=0.001, latency_ms=10),
    )
    report = build_run_report(
        thread_id="both-src",
        wake_id="both-src",
        domain="lol_cn",
        mission="双源渲染",
        mode="deep",
        object_id=None,
        result={"turn_count": 2, "tool_events": []},
        store=store,
        runtime_path=runtime,
        wall_ms=10,
    )
    rendered = render_run_report(report)
    assert "req=deepseek-v4-flash resp=echoed-model" in rendered
    assert "src=provider_effective" in rendered
    assert "req=- resp=echoed-model" in rendered
    assert "src=caller_requested_fallback" in rendered
    assert "src=-" not in rendered


def test_run_report_degrades_fingerprint_for_pre_fingerprint_ledger(tmp_path: Path) -> None:
    """Ledger snapshots recorded before the fingerprint column still render."""
    store = WorldStore(tmp_path / "world.sqlite3")
    runtime = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(runtime)
    try:
        connection.execute(
            "CREATE TABLE model_calls ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,"
            "turn INTEGER NOT NULL, purpose TEXT NOT NULL,"
            "wake_protocol TEXT NOT NULL DEFAULT 'current',"
            "phase TEXT NOT NULL DEFAULT 'exploration', model TEXT,"
            "prompt_tokens INTEGER, cached_input_tokens INTEGER,"
            "uncached_input_tokens INTEGER, completion_tokens INTEGER,"
            "cost_usd REAL, latency_ms REAL)"
        )
        connection.execute(
            "INSERT INTO model_calls (thread_id, turn, purpose, cached_input_tokens,"
            " uncached_input_tokens, cost_usd, latency_ms)"
            " VALUES ('legacy-fp', 1, 'explore', 4, 6, 0.001, 25)"
        )
        connection.commit()
    finally:
        connection.close()

    report = build_run_report(
        thread_id="legacy-fp",
        wake_id="legacy-fp",
        domain="lol_cn",
        mission="旧快照",
        mode="broad",
        object_id=None,
        result={"turn_count": 1, "tool_events": []},
        store=store,
        runtime_path=runtime,
        wall_ms=10,
    )

    assert report["model"]["available"] is True
    assert report["model"]["calls"] == [
        {
            "purpose": "explore",
            "phase": "exploration",
            "request_model": "",
            "response_model": "",
            "request_source": "",
            "request_fingerprint": "",
            "cached_input_tokens": 4,
            "uncached_input_tokens": 6,
        }
    ]
    rendered = render_run_report(report)
    assert "cached=4" in rendered
    assert "fp=-" in rendered
    # pre-fingerprint rows render dashes for the absent model pair and source
    assert "req=- resp=-" in rendered
    assert "src=-" in rendered
