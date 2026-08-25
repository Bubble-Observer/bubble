"""Small, explicit domain lenses for world-agent prompt injection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainFocus:
    """A configured observation lens, separate from the stable agent identity."""

    domain_key: str
    observation_center: str
    relevance_rule: str
    attention_examples: tuple[str, ...]
    source_preferences: tuple[str, ...]
    locale: str
    timezone: str
    branches: tuple[tuple[str, str], ...] = ()
    search_experience: tuple[str, ...] = ()


_LOL_CN = DomainFocus(
    domain_key="lol_cn",
    observation_center="Chinese League of Legends community understanding",
    relevance_rule=(
        "Explore developments that concern League of Legends or its communities, or that "
        "provide evidence of a direct change to something that does."
    ),
    attention_examples=(
        "Chinese community language, reactions, and local context",
        "global tournaments, players, teams, organizations, and competitive mechanics",
        "patches, versions, platform changes, business decisions, and historical links with material impact",
        "memes, slang, nicknames, and competing uses whose meaning matters in context",
    ),
    source_preferences=(
        "Chinese community discussion and primary materials are useful starting points",
        "Use international, foreign-language, or historical sources when they materially "
        "clarify the same world",
    ),
    locale="zh-CN",
    timezone="Asia/Shanghai",
    branches=(
        ("LPL 赛事与战队", "近期比赛、战队动态、选手状态与转会"),
        ("其他赛区", "LCK、欧美赛区的重要动态与跨赛区联系"),
        ("游戏本体", "版本更新、平衡改动、玩法变化"),
        ("社区文化", "梗、黑话、圈层语言与社区情绪"),
    ),
    search_experience=(
        "宽入口可把英雄联盟、LOL、LPL等领域称呼与今日、昨晚、刚刚、赛后、官宣、热议等及时性或社区表达自然组合；这些只是语言经验。",
        "当前材料出现赛事、战队、选手、版本变化、争议或新说法后，可据此形成更具体的查询；不要把固定branch当成搜索清单。",
        "视频社区常用赛后、复盘、爆料等表达；公开网页查询适合加入对象、事件、日期或结果词；论坛板块本身也能提供当前刺激。",
    ),
)

_FOCUSES: dict[str, DomainFocus] = {_LOL_CN.domain_key: _LOL_CN}


def domain_focus_keys() -> tuple[str, ...]:
    """Return registered domain keys in deterministic order."""
    return tuple(sorted(_FOCUSES))


def resolve_domain_focus(domain_key: str) -> DomainFocus:
    """Resolve one registered lens, failing closed for empty or unknown keys."""
    key = domain_key.strip()
    if not key:
        raise ValueError("domain key must be non-empty")
    try:
        return _FOCUSES[key]
    except KeyError as error:
        available = ", ".join(domain_focus_keys())
        raise ValueError(f"unknown domain key: {key}; available: {available}") from error
