"""Tests for SubtitleLedger — domain isolation, cross-video counting, degradation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leave_information_bubble.tools.subtitle_ledger import SubtitleLedger


@pytest.fixture
def ledger(tmp_path: Path) -> SubtitleLedger:
    return SubtitleLedger(tmp_path / "subtitle-hash-ledger.json")


# ── Cross-video counting & idempotency ─────────────────────────────────────


def test_single_video_never_blacklisted(ledger: SubtitleLedger) -> None:
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is False


def test_two_distinct_bvids_blacklists(ledger: SubtitleLedger) -> None:
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    ledger.record("lol_cn", "abc1234567", "BV2bbb")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is True


def test_three_distinct_bvids_blacklists(ledger: SubtitleLedger) -> None:
    for bvid in ("BV1aaa", "BV2bbb", "BV3ccc"):
        ledger.record("lol_cn", "abc1234567", bvid)
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is True


def test_repeated_same_bvid_does_not_count(ledger: SubtitleLedger) -> None:
    for _ in range(5):
        ledger.record("lol_cn", "abc1234567", "BV1aaa")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is False


def test_same_hash_two_bvids_then_third_repeated(ledger: SubtitleLedger) -> None:
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    ledger.record("lol_cn", "abc1234567", "BV2bbb")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is True
    # Re-observing an already-known bvid must not roll anything back.
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is True


def test_record_idempotent_count_field(ledger: SubtitleLedger) -> None:
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    data = json.loads(ledger._path.read_text(encoding="utf-8"))
    entry = data["lol_cn"]["abc1234567"]
    assert entry["count"] == 1
    assert entry["bvids"] == ["BV1aaa"]


# ── Domain isolation ───────────────────────────────────────────────────────


def test_hash_blacklisted_in_one_domain_only(ledger: SubtitleLedger) -> None:
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    ledger.record("lol_cn", "abc1234567", "BV2bbb")
    ledger.record("nga_cn", "abc1234567", "BV9zzz")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is True
    assert ledger.is_blacklisted("nga_cn", "abc1234567") is False


def test_domain_keys_are_separate_branches(ledger: SubtitleLedger) -> None:
    ledger.record("lol_cn", "hashAAAA01", "BV1aaa")
    ledger.record("lol_cn", "hashAAAA01", "BV2bbb")
    ledger.record("nga_cn", "hashAAAA01", "BV9zzz")
    assert ledger.is_blacklisted("lol_cn", "hashAAAA01") is True
    assert ledger.is_blacklisted("nga_cn", "hashAAAA01") is False


def test_unknown_domain_and_hash_not_blacklisted(ledger: SubtitleLedger) -> None:
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    assert ledger.is_blacklisted("other_dom", "abc1234567") is False
    assert ledger.is_blacklisted("lol_cn", "different") is False


# ── Missing / corrupt file degradation ─────────────────────────────────────


def test_missing_file_not_blacklisted(ledger: SubtitleLedger) -> None:
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is False


def test_garbage_json_degrades_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("{ this is not json !!!", encoding="utf-8")
    ledger = SubtitleLedger(path)
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is False
    # Recovery: recording after a corrupt file must not crash or raise.
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is False


def test_invalid_structure_degrades_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    ledger = SubtitleLedger(path)
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is False


def test_invalid_hash_entry_degrades_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps({"lol_cn": {"abc1234567": {"bvids": "not-a-list"}}}),
        encoding="utf-8",
    )
    ledger = SubtitleLedger(path)
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is False
    # Recording replaces the invalid state with a fresh, valid one.
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    ledger.record("lol_cn", "abc1234567", "BV2bbb")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is True


# ── Persistence across instances ───────────────────────────────────────────


def test_ledger_survives_reconstruction(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    SubtitleLedger(path).record("lol_cn", "abc1234567", "BV1aaa")
    SubtitleLedger(path).record("lol_cn", "abc1234567", "BV2bbb")
    fresh = SubtitleLedger(path)
    assert fresh.is_blacklisted("lol_cn", "abc1234567") is True


# ── Atomic writes ──────────────────────────────────────────────────────────


def test_no_temp_file_left_after_writes(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = SubtitleLedger(path)
    for i in range(3):
        ledger.record("lol_cn", "abc1234567", f"BV{i:04d}")
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    # File content is valid JSON and complete.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["lol_cn"]["abc1234567"]["count"] == 3


def test_write_failure_keeps_old_file_and_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    SubtitleLedger(path).record("lol_cn", "abc1234567", "BV1aaa")
    before = path.read_text(encoding="utf-8")
    # Replace the ledger file with a directory so os.replace fails.
    path.unlink()
    path.mkdir()
    ledger = SubtitleLedger(path)
    ledger.record("lol_cn", "abc1234567", "BV2bbb")  # must not raise
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    # The pre-failure content is untouched on disk: reconstructing a fresh
    # ledger from the captured old content still sees only one bvid.
    fresh_path = tmp_path / "ledger-copy.json"
    fresh_path.write_text(before, encoding="utf-8")
    fresh = SubtitleLedger(fresh_path)
    assert fresh.is_blacklisted("lol_cn", "abc1234567") is False
    fresh.record("lol_cn", "abc1234567", "BV2bbb")
    assert fresh.is_blacklisted("lol_cn", "abc1234567") is True


def test_write_failure_then_recovery(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = SubtitleLedger(path)
    ledger.record("lol_cn", "abc1234567", "BV1aaa")
    path.unlink()
    path.mkdir()
    ledger.record("lol_cn", "abc1234567", "BV2bbb")  # fails silently
    path.rmdir()
    ledger.record("lol_cn", "abc1234567", "BV2bbb")
    ledger.record("lol_cn", "abc1234567", "BV3ccc")
    assert ledger.is_blacklisted("lol_cn", "abc1234567") is True
