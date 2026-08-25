"""Contracts for the P4 domain-focus registry."""

from __future__ import annotations

import pytest

from leave_information_bubble.world.domain_config import DomainFocus, domain_focus_keys, resolve_domain_focus


def test_lol_cn_is_the_only_registered_focus() -> None:
    assert domain_focus_keys() == ("lol_cn",)
    focus = resolve_domain_focus("lol_cn")
    assert isinstance(focus, DomainFocus)
    assert focus.locale == "zh-CN"
    assert focus.timezone == "Asia/Shanghai"
    assert "Explore developments that concern League of Legends" in focus.relevance_rule
    assert "evidence of a direct change" in focus.relevance_rule
    assert any("patches, versions" in item for item in focus.attention_examples)
    assert any("memes, slang, nicknames, and competing uses" in item for item in focus.attention_examples)
    assert any("foreign-language" in item for item in focus.source_preferences)


@pytest.mark.parametrize("key", ("", "  ", "finance", "LOL_CN"))
def test_focus_resolution_fails_closed(key: str) -> None:
    with pytest.raises(ValueError):
        resolve_domain_focus(key)


def test_lol_cn_carries_branch_dimensions() -> None:
    focus = resolve_domain_focus("lol_cn")
    assert len(focus.branches) == 4
    names = [name for name, _ in focus.branches]
    assert names == ["LPL 赛事与战队", "其他赛区", "游戏本体", "社区文化"]
    assert all(description for _, description in focus.branches)
