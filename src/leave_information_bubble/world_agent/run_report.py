"""Small, factual end-of-wake report for the world-agent runner.

G5b-2: Graph Shell is the default and only normal runtime, so the report no
longer reads legacy proposal-attempt fields; ``review`` and its render line
are retired and not emitted.  ``wake_id`` is mandatory and never falls back
to ``thread_id``.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from leave_information_bubble.world import WorldRecall, WorldStore

_CAP = 5
_TOOL_DIAGNOSTIC_CAP = 12


def build_run_report(
    *,
    thread_id: str,
    domain: str,
    mission: str,
    mode: str,
    object_id: str | None,
    result: Mapping[str, Any],
    store: WorldStore,
    runtime_path: Path,
    wall_ms: float,
    wake_id: str,
) -> dict[str, Any]:
    """Join one final graph state with its existing runtime and world ledgers."""
    # G5b-2: Graph Shell is the only normal runtime; the protocol field is
    # still surfaced for API compatibility, never read as a selector.
    execution_mode = str(result.get("execution_mode") or "graph_shell")
    receipt = _durable_receipt(store, wake_id)
    object_names = _object_names(store, receipt["object_ids"])
    inquiry_rows = _inquiries(store, receipt["inquiry_ids"])
    tool_events = _unique_events(result.get("tool_events", []))
    model = _model_summary(runtime_path, thread_id, wake_id)
    status = _status(result)
    curiosity_topics = _curiosity_topics(tool_events, inquiry_rows)
    connections = _connections(store, receipt["assertion_ids"])
    next_entries = _next_entries(store, inquiry_rows)
    hard = []
    if status == "incomplete":
        hard.append(str(result.get("terminal_summary") or "wake ended without a durable terminal commit"))
    working_graph = _working_graph_summary(store, wake_id)
    finalize_receipt = (
        dict(result.get("finalize_receipt", {}))
        if isinstance(result.get("finalize_receipt"), Mapping)
        else {}
    )
    finalize_status_counts = Counter(
        str(event.get("diagnostic", {}).get("status"))
        for event in tool_events
        if event.get("name") == "finalize_graph"
        and isinstance(event.get("diagnostic"), Mapping)
        and event.get("diagnostic", {}).get("status")
    )
    if (
        execution_mode == "graph_shell"
        and finalize_receipt.get("replayed") is True
        and finalize_receipt.get("status") == "already_published"
    ):
        # CP6 may persist the durable receipt before the tool event reaches a
        # checkpoint. Reconciliation is then the only truthful evidence for
        # both the original publication and this already-published replay.
        finalize_status_counts["published"] = max(finalize_status_counts["published"], 1)
        finalize_status_counts["already_published"] = max(
            finalize_status_counts["already_published"], 1
        )
    return {
        "entry": {
            "thread_id": thread_id,
            "wake_id": wake_id,
            "execution_mode": execution_mode,
            "domain": domain,
            "mission": mission,
            "mode": mode,
            "deep_seed": (
                {"id": object_id, "name": _one_object_name(store, object_id)} if object_id else None
            ),
        },
        "execution": {
            "status": status,
            "turns": int(result.get("turn_count", 0) or 0),
            "wall_ms": max(0.0, round(float(wall_ms), 3)),
            "raw_exception": (
                dict(result.get("raw_exception", {}))
                if isinstance(result.get("raw_exception"), Mapping)
                else {}
            ),
            "stop_reason": str(
                result.get("transition_reason")
                or result.get("terminal_summary")
                or "terminal_commit"
            ),
        },
        "model": model,
        "tools": _tool_summary(tool_events),
        "path": _exploration_path(tool_events),
        "curiosity": {
            "observed": bool(curiosity_topics),
            "topics": curiosity_topics[:3],
            "note": (
                "explicit inquiry/unresolved signals observed"
                if curiosity_topics
                else "no explicit curiosity signal observed; no inference made"
            ),
        },
        "connections": connections,
        "durable_diff": {
            "objects": len(receipt["object_ids"]),
            "assertions": len(receipt["assertion_ids"]),
            "inquiries": len(receipt["inquiry_ids"]),
            "resolved": len(receipt["resolved_inquiry_ids"]),
            "evidence_links": _evidence_count(store, receipt["assertion_ids"]),
            "object_names": object_names[:_CAP],
            "inquiry_prompts": [row["prompt"] for row in inquiry_rows[:_CAP]],
        },
        "working_graph": working_graph,
        "publication": {
            "published": status in {"published", "already_published"},
            "status": str(result.get("finalize_status") or ""),
            "commit_id": finalize_receipt.get("commit_id"),
            "receipt": finalize_receipt,
            "resume_allowed": bool(result.get("resume_allowed", False)),
            "resume_count": int(result.get("resume_count", 0) or 0),
            "patch_success_count": int(result.get("patch_success_count", 0) or 0),
            "patch_replay_count": int(result.get("patch_replay_count", 0) or 0),
            "finalize_status_counts": dict(finalize_status_counts),
            "replayed": bool(finalize_receipt.get("replayed")),
            "recovery_action": _recovery_action(status, finalize_receipt),
        },
        "graph_actions": _graph_actions(tool_events),
        # G5b-2: legacy proposal-attempt review is retired; no soft issue
        # codes exist without it, so soft stays an empty, schema-stable list.
        "issues": {"hard": hard, "soft": []},
        "next_entries": next_entries,
    }


def render_run_report(report: Mapping[str, Any]) -> str:
    """Render a compact Chinese report; detailed values remain in the mapping."""
    entry = report["entry"]
    execution = report["execution"]
    model = report["model"]
    tools = report["tools"]
    durable = report["durable_diff"]
    curiosity = report["curiosity"]
    seed = entry.get("deep_seed")
    seed_text = f"；起点={seed['name']} ({seed['id']})" if seed else ""
    lines = [
        "== 本轮认知报告 ==",
        (
            f"入口：{entry['mode']} / {entry['domain']} / {entry['execution_mode']}；"
            f"thread={entry['thread_id']}；wake={entry['wake_id']}；"
            f"mission={_clip(entry['mission'], 160)}{seed_text}"
        ),
        (
            f"执行：{execution['status']}；turns={execution['turns']}；"
            f"总耗时={_seconds(execution['wall_ms'])}；停止原因={_clip(execution['stop_reason'], 120)}"
        ),
        (
            f"模型：成功调用={model['successful_calls']}；耗时={_seconds(model['latency_ms'])}；"
            f"成本=${model['cost_usd']:.6f}；用途={_pairs(model['by_purpose'])}"
            if model["available"]
            else "模型：成功调用明细不可用（运行继续有效）"
        ),
        (
            "调用：" + "；".join(_call_line(call) for call in model["calls"])
            if model["available"] and model["calls"]
            else "调用：无"
        ),
        (
            f"工具：{tools['total']} 次，失败={tools['failed']}，带限制={tools['limited']}，"
            f"耗时={_seconds(tools['elapsed_ms'])}；分布={_pairs(tools['by_name'])}"
        ),
        "探索路径：" + (" → ".join(report["path"]) if report["path"] else "未形成可复核路径"),
        "开放好奇：" + ("；".join(curiosity["topics"]) if curiosity["observed"] else curiosity["note"]),
        "跨对象连接："
        + ("；".join(report["connections"]) if report["connections"] else "未观察到已提交的结构连接"),
        (
            f"持久化：objects +{durable['objects']} / assertions +{durable['assertions']} / "
            f"inquiries +{durable['inquiries']} / resolved +{durable['resolved']} / "
            f"evidence links +{durable['evidence_links']}"
        ),
        (
            f"工作图：active={report['working_graph']['active_total']}；"
            f"状态={_pairs(report['working_graph']['by_status'])}；"
            f"图操作={_pairs(report['graph_actions'])}"
        ),
        (
            f"质询动作：新建={report['working_graph']['inquiry_actions']['created']}；"
            f"解决={report['working_graph']['inquiry_actions']['resolved']}"
        ),
        (
            f"发布：{report['publication']['status'] or '未调用'}；"
            f"commit={report['publication']['commit_id'] or '-'}；"
            f"resume={report['publication']['resume_count']}；"
            f"patch 重放={report['publication']['patch_replay_count']}"
            + (
                f"；恢复动作={report['publication']['recovery_action']}"
                if report["publication"]["recovery_action"]
                else ""
            )
        ),
        "新增中心：" + ("；".join(durable["object_names"]) if durable["object_names"] else "无"),
        "问题：硬="
        + ("；".join(report["issues"]["hard"]) or "无")
        + "；软="
        + ("；".join(report["issues"]["soft"]) or "无"),
        "下轮入口：" + ("；".join(report["next_entries"]) if report["next_entries"] else "暂无明确入口"),
    ]
    return "\n".join(lines)


def _durable_receipt(store: WorldStore, wake_id: str) -> dict[str, list[str]]:
    """One wake's formal finalize receipt — exactly one per wake (Core §6.2).

    G5b-2: the ``agent:{thread}:...`` legacy commit family is retired; the
    receipt root is always ``{wake_id}:finalize`` and is never LIKE-matched.
    """
    output = {key: [] for key in ("object_ids", "assertion_ids", "inquiry_ids", "resolved_inquiry_ids")}
    root = f"{wake_id}:finalize"
    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT receipt_json FROM commit_receipts WHERE commit_id = ? ORDER BY committed_at",
            (root,),
        ).fetchall()
    for row in rows:
        try:
            receipt = json.loads(str(row["receipt_json"]))
        except (TypeError, ValueError):
            continue
        if not isinstance(receipt, dict):
            continue
        for key in output:
            values = receipt.get(key, [])
            if isinstance(values, list):
                output[key].extend(str(value) for value in values if str(value))
    return {key: list(dict.fromkeys(values)) for key, values in output.items()}


def _empty_working_graph() -> dict[str, Any]:
    return {
        "active_total": 0,
        "by_status": {"active": 0, "abandoned": 0, "finalized": 0},
        "by_kind": {"objects": 0, "assertions": 0, "inquiries": 0},
        "inquiry_actions": {"created": 0, "resolved": 0},
        "active_ids": [],
    }


def _working_graph_summary(store: WorldStore, wake_id: str) -> dict[str, Any]:
    summary = _empty_working_graph()
    by_status: Counter[str] = Counter()
    by_kind: dict[str, int] = {}
    active_ids: list[str] = []
    with store.read_connection() as connection:
        for kind, table in (
            ("objects", "staged_objects"),
            ("assertions", "staged_assertions"),
            ("inquiries", "staged_inquiries"),
        ):
            rows = connection.execute(
                f"SELECT staged_id, status FROM {table} WHERE wake_id = ? ORDER BY staged_id",
                (wake_id,),
            ).fetchall()
            by_kind[kind] = len(rows)
            for row in rows:
                status = str(row["status"])
                by_status[status] += 1
                if status == "active" and len(active_ids) < _CAP:
                    active_ids.append(str(row["staged_id"]))
        # Wave B rows: creates carry a subject kind (factual/semantic/stateful),
        # resolve rows carry the row-class marker 'resolution' — so a per-wake
        # kind split is the truthful "inquiry actions" count (D5).
        inquiry_kinds = connection.execute(
            "SELECT kind FROM staged_inquiries WHERE wake_id = ?", (wake_id,)
        ).fetchall()
    summary.update(
        active_total=by_status["active"],
        by_status={
            "active": by_status["active"],
            "abandoned": by_status["abandoned"],
            "finalized": by_status["finalized"],
        },
        by_kind=by_kind,
        inquiry_actions={
            "created": sum(1 for row in inquiry_kinds if str(row["kind"]) != "resolution"),
            "resolved": sum(1 for row in inquiry_kinds if str(row["kind"]) == "resolution"),
        },
        active_ids=active_ids,
    )
    return summary


def _model_summary(path: Path, thread_id: str, wake_id: str | None = None) -> dict[str, Any]:
    empty = {
        "available": False,
        "successful_calls": 0,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
        "by_purpose": {},
        "calls": [],
    }
    rows = _model_ledger_rows(path, thread_id, wake_id)
    if rows is None:
        return empty
    purposes = Counter(str(row["purpose"] or "unknown") for row in rows)
    return {
        "available": True,
        "successful_calls": len(rows),
        "latency_ms": round(sum(float(row["latency_ms"] or 0) for row in rows), 3),
        "cost_usd": sum(float(row["cost_usd"] or 0) for row in rows),
        "by_purpose": dict(purposes),
        "calls": [
            {
                "purpose": str(row.get("purpose") or "unknown"),
                "phase": str(row.get("phase") or "exploration"),
                "request_model": str(row.get("request_model") or ""),
                "response_model": str(row.get("model") or ""),
                "request_source": str(row.get("request_source") or ""),
                "request_fingerprint": str(row.get("request_fingerprint") or ""),
                "cached_input_tokens": int(row.get("cached_input_tokens") or 0),
                "uncached_input_tokens": int(row.get("uncached_input_tokens") or 0),
            }
            for row in rows
        ],
    }


def _model_ledger_rows(
    path: Path, thread_id: str, wake_id: str | None = None
) -> list[dict[str, Any]] | None:
    """Read one wake's ledger rows, degrading missing columns to empty values.

    Snapshots recorded before a column existed must still render: the query
    falls back stepwise to older column sets and absent fields report as
    empty instead of failing the whole report.  Returns None when the ledger
    table itself is missing, so the caller reports the ledger unavailable.

    When wake_id is given and the ledger has a wake_id column, rows filter by
    wake identity so one thread's wakes never bleed into each other's model
    ledger (Core §6.2); without the column the thread filter remains as the
    compat fallback.
    """
    try:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error:
        return None
    try:
        has_wake_id = False
        with contextlib.suppress(sqlite3.OperationalError):
            # missing table surfaces through the column loop below
            has_wake_id = any(
                str(row["name"]) == "wake_id"
                for row in connection.execute("PRAGMA table_info(model_calls)")
            )
        if wake_id is not None and has_wake_id:
            filter_sql, filter_param = "wake_id = ?", (wake_id,)
        else:
            filter_sql, filter_param = "thread_id = ?", (thread_id,)
        for columns in (
            "purpose, phase, request_model, model, request_fingerprint, request_source,"
            " cached_input_tokens, uncached_input_tokens, cost_usd, latency_ms",
            "purpose, phase, request_fingerprint, request_source,"
            " cached_input_tokens, uncached_input_tokens, cost_usd, latency_ms",
            "purpose, phase, request_fingerprint, cached_input_tokens,"
            " uncached_input_tokens, cost_usd, latency_ms",
            "purpose, phase, cached_input_tokens, uncached_input_tokens,"
            " cost_usd, latency_ms",
            "purpose, cost_usd, latency_ms",
        ):
            try:
                rows = connection.execute(
                    f"SELECT {columns} FROM model_calls"
                    f" WHERE {filter_sql} ORDER BY id",
                    filter_param,
                ).fetchall()
            except sqlite3.OperationalError as error:
                if "no such table" in str(error):
                    return None
                continue
            return [dict(row) for row in rows]
        return []
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def _unique_events(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("call_id", ""))
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        output.append(item)
    return output


def _tool_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for event in events:
        diagnostic = event.get("diagnostic")
        diagnostic = dict(diagnostic) if isinstance(diagnostic, Mapping) else {}
        limitations = [str(item)[:160] for item in event.get("limitations", [])[:8] if str(item)]
        if not diagnostic and bool(event.get("ok")) and not limitations:
            continue
        diagnostics.append(
            {
                "call_id": str(event.get("call_id", ""))[:160],
                "name": str(event.get("name", ""))[:160],
                "ok": bool(event.get("ok")),
                "limitations": limitations,
                **diagnostic,
            }
        )
    return {
        "total": len(events),
        "failed": sum(not bool(event.get("ok")) for event in events),
        "limited": sum(bool(event.get("limitations")) for event in events),
        "elapsed_ms": round(sum(float(event.get("elapsed_ms", 0) or 0) for event in events), 3),
        "by_name": dict(Counter(str(event.get("name", "unknown")) for event in events)),
        "slowest": [
            {"name": str(item.get("name", "")), "elapsed_ms": float(item.get("elapsed_ms", 0) or 0)}
            for item in sorted(
                events,
                key=lambda event: float(event.get("elapsed_ms", 0) or 0),
                reverse=True,
            )[:3]
        ],
        "diagnostics": diagnostics[:_TOOL_DIAGNOSTIC_CAP],
        "diagnostics_truncated": len(diagnostics) > _TOOL_DIAGNOSTIC_CAP,
    }


def _exploration_path(events: Sequence[Mapping[str, Any]]) -> list[str]:
    path: list[str] = []
    for event in events:
        name = str(event.get("name", ""))
        args = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        if name.startswith("memory_"):
            anchor = args.get("query") or args.get("inquiry_id") or args.get("observation_id") or ""
            label = f"memory:{_clip(anchor, 40)}" if anchor else f"memory:{name.removeprefix('memory_')}"
        elif name in {"discover_sources", "search_sources"}:
            label = f"{args.get('adapter', '?')}:{_clip(args.get('query', 'surface'), 40)}"
        elif name in {"open_source", "sample_discussion", "inspect_media"}:
            card = event.get("card") if isinstance(event.get("card"), dict) else {}
            title = card.get("title") or args.get("observation_id") or "source"
            label = f"open[{args.get('depth', name)}]:{_clip(title, 48)}"
        elif name == "follow_related":
            label = "related:" + _clip(args.get("observation_id", "source"), 40)
        else:
            continue
        if not path or path[-1] != label:
            path.append(label)
    return path[:8]


def _curiosity_topics(
    events: Sequence[Mapping[str, Any]], inquiries: Sequence[Mapping[str, str]]
) -> list[str]:
    # G5b-2: log_inquiry_point / propose_inquiry are retired from the shell
    # tool surface; unresolved markers and formal inquiry rows are the only
    # curiosity signals left, and both are grounded in the durable store.
    topics: list[str] = []
    for event in events:
        values = event.get("unresolved", [])
        if isinstance(values, list):
            topics.extend(str(value) for value in values if str(value))
    topics.extend(row["prompt"] for row in inquiries)
    return list(dict.fromkeys(_clip(topic, 120) for topic in topics if topic.strip()))


def _connections(store: WorldStore, assertion_ids: Sequence[str]) -> list[str]:
    if not assertion_ids:
        return []
    marks = ",".join("?" for _ in assertion_ids)
    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT s.canonical_name AS subject, a.predicate, o.canonical_name AS object"
            " FROM assertions a JOIN objects s ON s.id = a.subject_id"
            " JOIN objects o ON o.id = a.object_id"
            f" WHERE a.id IN ({marks}) ORDER BY a.id",
            tuple(assertion_ids),
        ).fetchall()
    return [f"{row['subject']} -{row['predicate']}→ {row['object']}" for row in rows[:_CAP]]


def _object_names(store: WorldStore, identifiers: Sequence[str]) -> list[str]:
    if not identifiers:
        return []
    marks = ",".join("?" for _ in identifiers)
    with store.read_connection() as connection:
        rows = connection.execute(
            f"SELECT id, canonical_name FROM objects WHERE id IN ({marks})", tuple(identifiers)
        ).fetchall()
    names = {str(row["id"]): str(row["canonical_name"]) for row in rows}
    return [names[identifier] for identifier in identifiers if identifier in names]


def _one_object_name(store: WorldStore, identifier: str) -> str:
    names = _object_names(store, [identifier])
    return names[0] if names else identifier


def _inquiries(store: WorldStore, identifiers: Sequence[str]) -> list[dict[str, str]]:
    if not identifiers:
        return []
    marks = ",".join("?" for _ in identifiers)
    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT i.id, i.prompt, i.status, o.canonical_name AS subject"
            " FROM inquiries i JOIN objects o ON o.id = i.subject_id"
            f" WHERE i.id IN ({marks})",
            tuple(identifiers),
        ).fetchall()
    by_id = {str(row["id"]): dict(row) for row in rows}
    return [by_id[identifier] for identifier in identifiers if identifier in by_id]


def _evidence_count(store: WorldStore, assertion_ids: Sequence[str]) -> int:
    if not assertion_ids:
        return 0
    marks = ",".join("?" for _ in assertion_ids)
    with store.read_connection() as connection:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM assertion_evidence WHERE assertion_id IN ({marks})",
                tuple(assertion_ids),
            ).fetchone()[0]
        )


def _next_entries(store: WorldStore, new_inquiries: Sequence[Mapping[str, str]]) -> list[str]:
    entries = [
        f"{row['subject']}：{_clip(row['prompt'], 120)} ({row['id']})"
        for row in new_inquiries
        if row.get("status") == "open"
    ]
    if entries:
        return entries[:3]
    try:
        overview = WorldRecall(store).overview(limit=3)
    except (sqlite3.Error, ValueError):
        return []
    return [f"{item.name}：{_clip(item.prompt or '', 120)} ({item.id})" for item in overview.coverage_gaps]


def _status(result: Mapping[str, Any]) -> str:
    # G5b-2: the graph's finalize node owns the terminal status; there is no
    # legacy recovery branch anymore.
    terminal = str(result.get("terminal_status") or "")
    if terminal in {
        "published",
        "already_published",
        "wake_closed",
        "staged_unpublished",
    }:
        return terminal
    return "incomplete"


_GRAPH_ACTION_TOOLS = ("graph_inspect", "graph_diff", "graph_patch", "finalize_graph")


def _graph_actions(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count the working-graph tool calls this wake (inspect/diff/patch/finalize)."""
    return {
        name: sum(1 for event in events if event.get("name") == name)
        for name in _GRAPH_ACTION_TOOLS
    }


def _recovery_action(terminal_status: str, finalize_receipt: Mapping[str, Any]) -> str:
    """One actionable line for a non-published terminal state (D5)."""
    if terminal_status == "staged_unpublished":
        return "working graph remains staged; resume this wake and call finalize_graph to publish"
    if terminal_status == "already_published":
        return "receipt already durable; replay confirmed, nothing left to publish"
    if terminal_status == "wake_closed":
        return "wake closed without a new durable commit; inspect staging before resuming"
    if finalize_receipt.get("replayed") is True:
        return "finalize replayed from an existing receipt"
    return ""


def _seconds(milliseconds: object) -> str:
    return f"{float(milliseconds or 0) / 1000:.2f}s"


def _pairs(values: Mapping[str, Any]) -> str:
    return ",".join(f"{key}={value}" for key, value in values.items()) or "无"


def _call_line(call: Mapping[str, Any]) -> str:
    """One compact per-call envelope line: purpose/phase, model pair, cache split, source, fingerprint.

    Request-side (alias actually sent) and response-side (provider-echoed
    model) stay separate values; ``src`` carries the persisted envelope
    source and never renders as ``-`` for rows that recorded one.
    """
    fingerprint = str(call.get("request_fingerprint") or "")
    request_model = call.get("request_model") or "-"
    response_model = call.get("response_model") or "-"
    return (
        f"{call.get('purpose') or 'unknown'}/{call.get('phase') or 'exploration'} "
        f"req={request_model} resp={response_model} "
        f"cached={int(call.get('cached_input_tokens') or 0)} "
        f"uncached={int(call.get('uncached_input_tokens') or 0)} "
        f"src={call.get('request_source') or '-'} "
        f"fp={fingerprint[:12] or '-'}"
    )


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["build_run_report", "render_run_report"]
