"""Tests for the append-only curiosity-log runtime module."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from leave_information_bubble.runtime.curiosity_log import (
    aggregate_curiosity,
    append_curiosity,
    load_curiosity,
)


def test_append_writes_json_line(tmp_path: Path) -> None:
    log = tmp_path / "curiosity.jsonl"
    append_curiosity(log, "t-1", 1, "comment", "obs-1", "48梗是什么意思", "社区反复使用但我不懂")
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["thread_id"] == "t-1"
    assert row["round"] == 1
    assert row["topic"] == "48梗是什么意思"


def test_append_appends_not_overwrites(tmp_path: Path) -> None:
    log = tmp_path / "curiosity.jsonl"
    append_curiosity(log, "t-1", 1, "comment", "o1", "a", "r")
    append_curiosity(log, "t-1", 2, "comment", "o2", "b", "r")
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_append_missing_parent_creates_dir(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "curiosity.jsonl"
    append_curiosity(log, "t-1", 1, "comment", "o1", "a", "r")
    assert log.exists()


def test_append_swallows_write_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Write failures must not raise: exploration is never blocked by log problems."""
    bad_path = tmp_path / "blocked"
    bad_path.mkdir()
    with caplog.at_level(logging.WARNING, logger="leave_information_bubble.runtime.curiosity_log"):
        append_curiosity(bad_path, "t-1", 1, "comment", "o1", "a", "r")
    assert "curiosity-log append failed" in caplog.text


def test_load_filters_thread_and_skips_corrupt(tmp_path: Path) -> None:
    log = tmp_path / "curiosity.jsonl"
    row_t1 = '{"thread_id": "t-1", "round": 1, "topic": "a", "source_ref": "o1",'
    row_t1 += ' "reason": "r", "source_type": "c"}\n'
    row_t2 = '{"thread_id": "t-2", "round": 1, "topic": "b", "source_ref": "o2",'
    row_t2 += ' "reason": "r", "source_type": "c"}\n'
    log.write_text(row_t1 + "not-json\n" + row_t2, encoding="utf-8")
    rows = load_curiosity(log, "t-1")
    assert len(rows) == 1
    assert rows[0]["topic"] == "a"


def test_aggregate_counts_topic_and_round_filter(tmp_path: Path) -> None:
    rows = [
        {"thread_id": "t-1", "round": 1, "topic": "梗", "source_ref": "o1"},
        {"thread_id": "t-1", "round": 2, "topic": "梗", "source_ref": "o2"},
        {"thread_id": "t-1", "round": 1, "topic": "其他", "source_ref": "o3"},
    ]
    agg = aggregate_curiosity(rows, min_count=2, max_rounds_back=3, current_round=2)
    assert len(agg) == 1
    assert agg[0]["topic"] == "梗"
    assert agg[0]["count"] == 2
    # round 淘汰：current_round=5 时 round 2 的 last_round 过期
    agg2 = aggregate_curiosity(rows, min_count=2, max_rounds_back=3, current_round=5)
    assert agg2 == []
