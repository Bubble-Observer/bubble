"""Deterministic tests for inquiry text similarity primitives."""

from __future__ import annotations

from leave_information_bubble.world.similarity import bigrams, jaccard, normalize


def test_normalize_strips_punctuation_and_casefolds():
    assert normalize("Bin 在 2026 LPL 第三赛段的实际状态!") == "bin在2026lpl第三赛段的实际状态"
    assert normalize("") == ""


def test_bigrams_chinese_and_ascii_mixed():
    chars = normalize("Bin状态")
    assert chars == "bin状态"
    assert bigrams("Bin状态") == {"bi", "in", "n状", "状态"}


def test_bigrams_short_text_uses_single_chars():
    assert bigrams("ab") == {"ab"}
    assert bigrams("a") == {"a"}
    assert bigrams("") == set()


def test_jaccard_basics():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"a", "c"}) == 1 / 3
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
