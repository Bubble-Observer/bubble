"""Offline contracts for local-console Agent profiles and prompt previews."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leave_information_bubble.console.profiles import (
    AgentProfile,
    AgentProfileRegistry,
    RunDefaults,
    build_prompt_preview,
    build_quick_profile,
    clone_profile_design,
)
from leave_information_bubble.world.domain_config import resolve_domain_focus
from leave_information_bubble.world_agent.prompt import graph_shell_prompt


def _profile(**changes: object) -> AgentProfile:
    values = {
        "id": "test_agent",
        "display_name": "Test Agent",
        "domain_key": "test_domain",
        "observation_center": "A local test community",
        "relevance_rule": "Keep material relevant to the local test community.",
        "attention_examples": ("community language",),
        "source_preferences": ("primary sources",),
        "branches": (("current events", "recent changes and participants"),),
        "locale": "en-US",
        "timezone": "UTC",
        "defaults": RunDefaults(max_turns=12, max_cost_usd=2.5),
        "world_db": "data/agents/test_agent/world.sqlite3",
        "runtime_db": "data/agents/test_agent/runtime.sqlite3",
        "operator_instructions": "Prefer clearly dated primary sources.",
    }
    values.update(changes)
    return AgentProfile.model_validate(values)


def _tracked_lol_profile_json() -> dict[str, object]:
    """Return the bundled LOL profile shape exactly as tracked before
    search_experience was persisted — a fixture, not the live data file.

    The live profile (data/run-configs/agents/lol_cn.json) is user-editable
    data; tests must not couple to it (a legitimate edit, e.g. pointing the
    profile at a new long-term DB, must not change test outcomes). The key
    set must equal _TRACKED_LOL_PROFILE_KEYS_WITHOUT_SEARCH_EXPERIENCE and
    the values must satisfy the bootstrap identity check, including the
    original world/runtime DB paths.
    """
    focus = resolve_domain_focus("lol_cn")
    return {
        "id": "lol_cn",
        "display_name": "英雄联盟中文社区观察员",
        "domain_key": focus.domain_key,
        "observation_center": focus.observation_center,
        "relevance_rule": focus.relevance_rule,
        "attention_examples": list(focus.attention_examples),
        "source_preferences": list(focus.source_preferences),
        "branches": [list(branch) for branch in focus.branches],
        "locale": focus.locale,
        "timezone": focus.timezone,
        "defaults": {
            "mode": "broad",
            "broad_perspective": None,
            "deep_perspective": None,
            "adapters": ["bilibili", "nga", "hupu", "public-web"],
            "max_turns": 24,
            "max_cost_usd": None,
            "wake_protocol": "current",
            "memory_navigation": "legacy",
            "digest_cache_reuse": False,
        },
        "world_db": "data/agents/lol_cn/world.sqlite3",
        "runtime_db": "data/agents/lol_cn/runtime.sqlite3",
        "operator_instructions": None,
        "search_experience": list(focus.search_experience),
    }


def _write_profile(directory: Path, payload: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{payload['id']}.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def test_registry_round_trip_and_atomic_json(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "agents")
    profile = _profile()

    assert registry.create(profile) == profile
    assert registry.get(profile.id) == profile
    assert registry.list() == [profile]
    payload = json.loads((tmp_path / "agents" / "test_agent.json").read_text(encoding="utf-8"))
    assert payload["world_db"] == "data/agents/test_agent/world.sqlite3"
    assert not list((tmp_path / "agents").glob("*.tmp"))

    updated = profile.model_copy(update={"display_name": "Updated Agent"})
    assert registry.update(updated) == updated
    assert registry.get(profile.id).display_name == "Updated Agent"


def test_profile_search_experience_round_trips_through_json(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "agents")
    profile = _profile(search_experience=("使用领域词与及时性语言形成宽入口",))

    registry.create(profile)

    assert registry.get(profile.id).search_experience == profile.search_experience
    payload = json.loads((registry.directory / "test_agent.json").read_text(encoding="utf-8"))
    assert payload["search_experience"] == ["使用领域词与及时性语言形成宽入口"]


def test_profile_focus_preserves_search_experience() -> None:
    profile = _profile(search_experience=("使用领域词与及时性语言形成宽入口",))

    assert profile.focus.search_experience == profile.search_experience


def test_profile_search_experience_trims_entries_and_allows_explicit_empty_tuple() -> None:
    assert _profile(search_experience=("  timely language  ",)).search_experience == (
        "timely language",
    )
    assert _profile(search_experience=()).search_experience == ()


@pytest.mark.parametrize(
    "search_experience",
    (
        ("",),
        ("x" * 241,),
        ("same", " same "),
    ),
)
def test_profile_search_experience_rejects_invalid_entries(
    search_experience: tuple[str, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="search experience entries must (contain 1..240 characters|be unique)",
    ):
        _profile(search_experience=search_experience)


def test_profile_search_experience_rejects_more_than_24_entries() -> None:
    with pytest.raises(ValueError, match="at most 24 items"):
        _profile(search_experience=tuple(f"entry-{index}" for index in range(25)))


def test_custom_profile_without_search_experience_preserves_sentinel(tmp_path: Path) -> None:
    payload = _profile().model_dump(mode="json")
    payload["id"] = "custom"
    payload["world_db"] = "data/agents/custom/world.sqlite3"
    payload["runtime_db"] = "data/agents/custom/runtime.sqlite3"
    payload.pop("search_experience", None)
    _write_profile(tmp_path, payload)

    profile = AgentProfileRegistry(tmp_path).get("custom")

    assert profile.search_experience is None
    assert profile.focus.search_experience == ()


def test_exact_tracked_legacy_search_experience_is_only_injected_by_bootstrap(
    tmp_path: Path,
) -> None:
    payload = _tracked_lol_profile_json()
    payload.pop("search_experience", None)
    _write_profile(tmp_path, payload)
    registry = AgentProfileRegistry(tmp_path)

    loaded = registry.get("lol_cn")

    assert loaded.search_experience is None
    assert loaded.focus.search_experience == ()

    bootstrapped = registry.bootstrap_lol_profile()

    assert bootstrapped.search_experience == resolve_domain_focus("lol_cn").search_experience
    persisted = json.loads((tmp_path / "lol_cn.json").read_text(encoding="utf-8"))
    assert persisted["search_experience"] == list(bootstrapped.search_experience)


def test_matching_lol_id_and_domain_do_not_trigger_search_experience_defaults(
    tmp_path: Path,
) -> None:
    payload = _tracked_lol_profile_json()
    payload["display_name"] = "My custom LOL observer"
    payload.pop("search_experience", None)
    _write_profile(tmp_path, payload)

    profile = AgentProfileRegistry(tmp_path).bootstrap_lol_profile()

    assert profile.search_experience is None
    assert profile.focus.search_experience == ()


def test_lol_domain_search_experience_is_weak_editable_guidance() -> None:
    assert resolve_domain_focus("lol_cn").search_experience == (
        "宽入口可把英雄联盟、LOL、LPL等领域称呼与今日、昨晚、刚刚、赛后、官宣、热议等及时性或社区表达自然组合；这些只是语言经验。",
        "当前材料出现赛事、战队、选手、版本变化、争议或新说法后，可据此形成更具体的查询；不要把固定branch当成搜索清单。",
        "视频社区常用赛后、复盘、爆料等表达；公开网页查询适合加入对象、事件、日期或结果词；论坛板块本身也能提供当前刺激。",
    )


def test_tracked_lol_profile_persists_registered_search_experience() -> None:
    payload = _tracked_lol_profile_json()

    assert payload["search_experience"] == list(
        resolve_domain_focus("lol_cn").search_experience
    )


@pytest.mark.parametrize(
    ("profile_id", "database_path"),
    (
        ("../escape", "data/world.sqlite3"),
        ("UPPER", "data/world.sqlite3"),
        ("safe", "../world.sqlite3"),
        ("safe", "C:/world.sqlite3"),
    ),
)
def test_profile_identifiers_and_paths_fail_closed(profile_id: str, database_path: str) -> None:
    values = _profile().model_dump()
    values["id"] = profile_id
    values["world_db"] = database_path
    with pytest.raises(ValueError):
        AgentProfile.model_validate(values)


def test_profile_database_paths_are_distinct_case_insensitively() -> None:
    values = _profile().model_dump()
    values["runtime_db"] = values["world_db"].upper()
    with pytest.raises(ValueError, match="different paths"):
        AgentProfile.model_validate(values)


def test_registry_rejects_profile_filename_mismatch(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path)
    (tmp_path / "other.json").write_text(_profile().model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        registry.list()


def test_registry_rejects_database_paths_used_by_another_profile(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "agents")
    first = _profile()
    registry.create(first)

    second = first.model_copy(update={"id": "second_agent", "display_name": "Second Agent"})
    with pytest.raises(ValueError, match="already used"):
        registry.create(second)

    registry.create(
        second.model_copy(
            update={
                "world_db": "data/agents/second_agent/world.sqlite3",
                "runtime_db": "data/agents/second_agent/runtime.sqlite3",
            }
        )
    )
    with pytest.raises(ValueError, match="already used"):
        registry.update(
            second.model_copy(
                update={
                    "world_db": "data/agents/second_agent/world.sqlite3",
                    "runtime_db": first.runtime_db,
                }
            )
        )


def test_quick_profile_uses_neutral_design_and_defers_database_creation(tmp_path: Path) -> None:
    profile = build_quick_profile(
        agent_id="indie_games",
        display_name="Indie Games Observer",
        observation_center="Chinese independent-game communities and their changing interests.",
        locale="zh-CN",
        timezone="Asia/Shanghai",
        adapters=("public-web", "bilibili"),
    )

    assert profile.domain_key == "indie_games"
    assert profile.relevance_rule == (
        "Explore material that concerns the stated observation center, or that provides "
        "evidence of a direct change to something that does."
    )
    assert profile.attention_examples
    assert profile.source_preferences
    assert profile.branches == ()
    assert profile.search_experience == ()
    assert profile.defaults == RunDefaults(adapters=("public-web", "bilibili"))
    assert profile.world_db == "data/agents/indie_games/world.sqlite3"
    assert profile.runtime_db == "data/agents/indie_games/runtime.sqlite3"
    assert not (tmp_path / profile.world_db).exists()
    assert not (tmp_path / profile.runtime_db).exists()


def test_clone_profile_design_keeps_design_but_not_memory_or_operator(tmp_path: Path) -> None:
    source = _profile(search_experience=("copy this domain guidance",)).model_copy(
        update={
            "defaults": RunDefaults(
                mode="deep",
                broad_perspective="a saved broad angle",
                deep_perspective="a saved deep angle",
                adapters=("public-web",),
                max_turns=36,
                max_cost_usd=3.0,
                wake_protocol="separated",
                memory_navigation="overview_v1",
                digest_cache_reuse=True,
            )
        }
    )
    copied = clone_profile_design(source, agent_id="second_agent", display_name="Second Agent")

    assert copied.id == "second_agent"
    assert copied.display_name == "Second Agent"
    assert copied.focus == source.focus
    assert copied.search_experience == source.search_experience
    assert copied.defaults.mode == "deep"
    assert copied.defaults.adapters == ("public-web",)
    assert copied.defaults.max_turns == 36
    assert copied.defaults.max_cost_usd == 3.0
    assert copied.defaults.broad_perspective is None
    assert copied.defaults.deep_perspective is None
    assert copied.defaults.wake_protocol == "current"
    assert copied.defaults.memory_navigation == "overview_v1"
    assert copied.defaults.digest_cache_reuse is False
    assert copied.operator_instructions is None
    assert copied.world_db == "data/agents/second_agent/world.sqlite3"
    assert copied.runtime_db == "data/agents/second_agent/runtime.sqlite3"
    assert copied.world_db != source.world_db
    assert copied.runtime_db != source.runtime_db
    assert not (tmp_path / copied.world_db).exists()
    assert not (tmp_path / copied.runtime_db).exists()
    assert source.defaults.wake_protocol == "separated"
    assert source.operator_instructions == "Prefer clearly dated primary sources."


def test_prompt_preview_keeps_core_layers_read_only_and_adds_operator() -> None:
    # G5b-2: the protocol selector is retired — the preview always compiles
    # the one Graph Shell prompt surface (profiles.py ignores wake_protocol)
    preview = build_prompt_preview(_profile(), mode="deep", object_id="seed-7")

    assert "current segment of a continuing cognition" in preview.stable
    assert "A local test community" in preview.domain
    assert "operator-provided possible entry is seed-7" in preview.posture
    assert "Epistemic Discipline" in preview.epistemic
    assert "Titles, comments, automatic transcripts" in preview.epistemic
    assert "without a valid evidence link" in preview.epistemic
    assert "Publication is always an explicit model decision" in preview.mechanics
    assert "the formal graph is durable published memory" in preview.mechanics.casefold()
    assert "Before declaring an Object or alias" not in preview.mechanics
    assert preview.operator == "Prefer clearly dated primary sources."
    assert preview.compiled == graph_shell_prompt(
        "deep",
        "seed-7",
        _profile().focus,
        operator_instructions=_profile().operator_instructions,
    )
    assert "## Operator Guidance" in preview.compiled
    assert "should inform choices" in preview.compiled
    assert "system, tool, evidence, and persistence boundaries" in preview.compiled
    assert "may be ignored" not in preview.compiled
    assert preview.compiled.endswith("Prefer clearly dated primary sources.")


def test_prompt_preview_renders_search_experience_for_broad_only() -> None:
    profile = _profile(search_experience=("combine domain and recency language",))

    broad = build_prompt_preview(profile, mode="broad")
    deep = build_prompt_preview(profile, mode="deep")

    assert "Human search experience — optional and revisable" in broad.domain
    assert "combine domain and recency language" in broad.domain
    assert "combine domain and recency language" in broad.compiled
    assert "Human search experience — optional and revisable" not in deep.domain
    assert "combine domain and recency language" not in deep.domain
    assert "combine domain and recency language" not in deep.compiled


def test_lol_bootstrap_is_idempotent_and_uses_isolated_storage(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "agents")
    first = registry.bootstrap_lol_profile()
    second = registry.bootstrap_lol_profile()

    assert first == second
    assert first.domain_key == "lol_cn"
    assert first.display_name == "英雄联盟中文社区观察员"
    assert first.world_db != first.runtime_db
    assert first.world_db == "data/agents/lol_cn/world.sqlite3"
    assert first.branches == resolve_domain_focus("lol_cn").branches
    assert "LPL 赛事与战队" in build_prompt_preview(first).compiled
    assert first.defaults.broad_perspective is None
    assert first.defaults.deep_perspective is None


def test_lol_bootstrap_migrates_profile_without_branches(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "agents")
    profile = registry.bootstrap_lol_profile()
    legacy = profile.model_dump(mode="json")
    legacy.pop("branches")
    (tmp_path / "agents" / "lol_cn.json").write_text(
        json.dumps(legacy, ensure_ascii=False),
        encoding="utf-8",
    )

    migrated = registry.bootstrap_lol_profile()

    assert migrated.branches == resolve_domain_focus("lol_cn").branches
    assert "Familiar vantage points" in build_prompt_preview(migrated).compiled


def test_lol_bootstrap_neutralizes_exact_legacy_builtin_profile(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(tmp_path / "agents")
    focus = resolve_domain_focus("lol_cn")
    legacy = {
        "id": "lol_cn",
        "display_name": "英雄联盟中文社区观察员",
        "domain_key": focus.domain_key,
        "observation_center": focus.observation_center,
        "relevance_rule": focus.relevance_rule,
        "attention_examples": list(focus.attention_examples),
        "source_preferences": list(focus.source_preferences),
        "locale": focus.locale,
        "timezone": focus.timezone,
        "defaults": {
            "mode": "broad",
            "broad_mission": (
                "观察最近 24 小时中文英雄联盟社区正在发生的变化。优先识别值得持续"
                "跟踪的赛事、战队、选手、版本与社区语义变化，比较不同来源和观点，"
                "并把有证据支持的新对象、关系和信息缺口沉淀到世界记忆。"
            ),
            "deep_mission": (
                "从选定的英雄联盟世界对象出发，沿已有关系、社区讨论、历史背景和"
                "未解决问题继续深入。区分事实、社区观点、综合判断与不确定性，"
                "只保留本轮新增或发生变化的长期认知。"
            ),
            "adapters": ["bilibili", "nga", "hupu", "public-web"],
            "max_turns": 24,
            "max_cost_usd": None,
            "wake_protocol": "current",
            "memory_navigation": "legacy",
            "digest_cache_reuse": False,
        },
        "world_db": "data/agents/lol_cn/world.sqlite3",
        "runtime_db": "data/agents/lol_cn/runtime.sqlite3",
        "operator_instructions": None,
    }
    registry.directory.mkdir(parents=True)
    (registry.directory / "lol_cn.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )

    migrated = registry.bootstrap_lol_profile()

    assert migrated.defaults.broad_perspective is None
    assert migrated.defaults.deep_perspective is None
    assert migrated.branches == focus.branches


def test_lol_bootstrap_preserves_user_perspectives_equal_to_old_default_text(
    tmp_path: Path,
) -> None:
    registry = AgentProfileRegistry(tmp_path / "agents")
    profile = registry.bootstrap_lol_profile()
    broad = (
        "观察最近 24 小时中文英雄联盟社区正在发生的变化。优先识别值得持续"
        "跟踪的赛事、战队、选手、版本与社区语义变化，比较不同来源和观点，"
        "并把有证据支持的新对象、关系和信息缺口沉淀到世界记忆。"
    )
    deep = (
        "从选定的英雄联盟世界对象出发，沿已有关系、社区讨论、历史背景和"
        "未解决问题继续深入。区分事实、社区观点、综合判断与不确定性，"
        "只保留本轮新增或发生变化的长期认知。"
    )
    registry.update(
        profile.model_copy(
            update={"defaults": profile.defaults.model_copy(
                update={"broad_perspective": broad, "deep_perspective": deep}
            )}
        )
    )

    preserved = registry.bootstrap_lol_profile()

    assert preserved.defaults.broad_perspective == broad
    assert preserved.defaults.deep_perspective == deep


def test_run_defaults_accept_legacy_missions_and_emit_canonical_perspectives() -> None:
    defaults = RunDefaults.model_validate(
        {"broad_mission": "  broad angle  ", "deep_mission": "  deep angle  "}
    )

    assert defaults.broad_perspective == "broad angle"
    assert defaults.deep_perspective == "deep angle"
    assert defaults.model_dump() == {
        "mode": "broad",
        "broad_perspective": "broad angle",
        "deep_perspective": "deep angle",
        "adapters": ("bilibili", "nga", "hupu", "public-web"),
        "max_turns": 96,
        "max_cost_usd": None,
        "thinking": False,
        "reasoning_effort": None,
        "wake_protocol": "current",
        "memory_navigation": "overview_v1",
        "digest_cache_reuse": False,
    }


def test_run_defaults_allow_empty_optional_perspectives_and_reject_conflicts() -> None:
    defaults = RunDefaults(broad_perspective="   ", deep_perspective=None)
    assert defaults.broad_perspective is None
    assert defaults.deep_perspective is None

    with pytest.raises(ValueError, match="cannot disagree"):
        RunDefaults.model_validate(
            {"broad_perspective": "new", "broad_mission": "legacy"}
        )


def test_profile_adapter_defaults_round_trips_through_json(tmp_path: Path) -> None:
    """Per-domain adapter surface parameters survive registry storage."""
    registry = AgentProfileRegistry(tmp_path / "profiles")
    payload = {
        "id": "indie_games",
        "display_name": "Indie Games Observer",
        "domain_key": "indie_games",
        "observation_center": "Independent game community understanding",
        "relevance_rule": "Keep material relevant to independent games.",
        "attention_examples": ("recent developments",),
        "source_preferences": ("configured sources",),
        "locale": "zh-CN",
        "timezone": "Asia/Shanghai",
        "defaults": {"mode": "broad", "adapters": ["public-web", "bilibili", "hupu", "nga"]},
        "adapter_defaults": {"hupu": {"board": "gaming"}},
        "world_db": "data/agents/indie_games/world.sqlite3",
        "runtime_db": "data/agents/indie_games/runtime.sqlite3",
    }
    registry.create(AgentProfile.model_validate(payload))
    loaded = registry.get("indie_games")

    assert loaded.adapter_defaults == {"hupu": {"board": "gaming"}}


@pytest.mark.parametrize(
    "adapter_defaults",
    (
        {"typo": {"board": "gaming"}},
        {"hupu": {"board": "gaming"}, "bilibili": {}},
    ),
)
def test_profile_adapter_defaults_reject_unknown_adapters(
    adapter_defaults: dict[str, dict[str, str]],
) -> None:
    with pytest.raises(ValueError, match="unknown adapter defaults"):
        _profile(adapter_defaults=adapter_defaults)


def test_profile_adapter_defaults_optional() -> None:
    profile = _profile()
    assert profile.adapter_defaults is None
