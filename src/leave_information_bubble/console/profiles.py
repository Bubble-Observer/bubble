"""Persisted Agent profiles and safe, read-only core Prompt previews."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from leave_information_bubble.world.domain_config import DomainFocus, resolve_domain_focus
from leave_information_bubble.world_agent.prompt import (
    AgentMode,
    WakeProtocol,
    graph_shell_prompt,
    prompt_layers,
    render_wake_input,
)

_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
# Adapters that accept per-domain surface parameters (board/fid); a profile
# entry for any other adapter id is a typo and rejected by validation.
_SURFACE_PARAMETER_ADAPTERS = frozenset({"hupu", "nga"})
_SECRET_MARKERS = ("api_key", "authorization:", "bearer ", "sk-")
_QUICK_RELEVANCE_RULE = (
    "Explore material that concerns the stated observation center, or that provides evidence "
    "of a direct change to something that does."
)
_QUICK_ATTENTION_EXAMPLES = (
    "recent developments, community language, and unresolved context",
)
_QUICK_SOURCE_PREFERENCES = (
    "Use configured sources as entry points; assess relevance and evidence independently.",
)
_LEGACY_DEFAULT_BROAD_MISSION = (
    "观察目标领域最近 24 小时正在发生的变化，识别值得持续跟踪的新事件、"
    "重要对象、观点分歧和信息缺口，并把有证据支持的认知沉淀到世界记忆。"
)
_LEGACY_DEFAULT_DEEP_MISSION = (
    "从选定的世界对象出发，沿已有关系、证据缺口和未解决问题继续深入，"
    "补充值得长期保留的事实、解释、关联与不确定性。"
)
_LEGACY_LOL_BROAD_MISSION = (
    "观察最近 24 小时中文英雄联盟社区正在发生的变化。优先识别值得持续"
    "跟踪的赛事、战队、选手、版本与社区语义变化，比较不同来源和观点，"
    "并把有证据支持的新对象、关系和信息缺口沉淀到世界记忆。"
)
_LEGACY_LOL_DEEP_MISSION = (
    "从选定的英雄联盟世界对象出发，沿已有关系、社区讨论、历史背景和"
    "未解决问题继续深入。区分事实、社区观点、综合判断与不确定性，"
    "只保留本轮新增或发生变化的长期认知。"
)

_LEGACY_LOL_PROFILE_KEYS = {
    "id",
    "display_name",
    "domain_key",
    "observation_center",
    "relevance_rule",
    "attention_examples",
    "source_preferences",
    "locale",
    "timezone",
    "defaults",
    "world_db",
    "runtime_db",
    "operator_instructions",
}
_LEGACY_RUN_DEFAULT_KEYS = {
    "mode",
    "broad_mission",
    "deep_mission",
    "adapters",
    "max_turns",
    "max_cost_usd",
    "wake_protocol",
    "memory_navigation",
    "digest_cache_reuse",
}
_TRACKED_LOL_PROFILE_KEYS_WITHOUT_SEARCH_EXPERIENCE = {
    "id",
    "display_name",
    "domain_key",
    "observation_center",
    "relevance_rule",
    "attention_examples",
    "source_preferences",
    "branches",
    "locale",
    "timezone",
    "defaults",
    "world_db",
    "runtime_db",
    "operator_instructions",
}
_TRACKED_RUN_DEFAULT_KEYS = {
    "mode",
    "broad_perspective",
    "deep_perspective",
    "adapters",
    "max_turns",
    "max_cost_usd",
    "wake_protocol",
    "memory_navigation",
    "digest_cache_reuse",
}


def _safe_profile_id(value: str) -> str:
    """Validate a profile id which is also used as a registry filename."""
    candidate = value.strip()
    if not _PROFILE_ID.fullmatch(candidate):
        raise ValueError("profile id must be 1-64 lowercase letters, numbers, '_' or '-'")
    return candidate


def _safe_relative_path(value: str) -> str:
    """Accept only relative, normalized database paths."""
    candidate = value.strip().replace("\\", "/")
    path = Path(candidate)
    if not candidate or path.is_absolute() or ":" in candidate or ".." in path.parts:
        raise ValueError("database path must be a non-empty relative path without traversal")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError("database path must be normalized")
    return path.as_posix()


def _is_legacy_builtin_lol_profile(value: object) -> bool:
    """Identify the exact pre-perspective bundled profile, not matching user text alone."""
    if not isinstance(value, Mapping) or set(value) != _LEGACY_LOL_PROFILE_KEYS:
        return False
    defaults = value.get("defaults")
    if not isinstance(defaults, Mapping) or set(defaults) != _LEGACY_RUN_DEFAULT_KEYS:
        return False
    focus = resolve_domain_focus("lol_cn")
    missions = (defaults.get("broad_mission"), defaults.get("deep_mission"))
    return (
        value.get("id") == "lol_cn"
        and value.get("display_name") in {"英雄联盟中文社区观察员", "LOL CN Observer"}
        and value.get("domain_key") == focus.domain_key
        and value.get("observation_center") == focus.observation_center
        and value.get("relevance_rule") == focus.relevance_rule
        and value.get("attention_examples") == list(focus.attention_examples)
        and value.get("source_preferences") == list(focus.source_preferences)
        and value.get("locale") == focus.locale
        and value.get("timezone") == focus.timezone
        and value.get("world_db") == "data/agents/lol_cn/world.sqlite3"
        and value.get("runtime_db") == "data/agents/lol_cn/runtime.sqlite3"
        and value.get("operator_instructions") is None
        and defaults.get("mode") == "broad"
        and defaults.get("adapters") == ["bilibili", "nga", "hupu", "public-web"]
        and defaults.get("max_turns") == 24
        and defaults.get("max_cost_usd") is None
        and defaults.get("wake_protocol") == "current"
        and defaults.get("memory_navigation") == "legacy"
        and defaults.get("digest_cache_reuse") is False
        and missions
        in {
            (_LEGACY_DEFAULT_BROAD_MISSION, _LEGACY_DEFAULT_DEEP_MISSION),
            (_LEGACY_LOL_BROAD_MISSION, _LEGACY_LOL_DEEP_MISSION),
        }
    )


def _is_tracked_lol_profile_without_search_experience(value: object) -> bool:
    """Match the bundled profile immediately before search experience was persisted."""
    if (
        not isinstance(value, Mapping)
        or set(value) != _TRACKED_LOL_PROFILE_KEYS_WITHOUT_SEARCH_EXPERIENCE
    ):
        return False
    defaults = value.get("defaults")
    if not isinstance(defaults, Mapping) or set(defaults) != _TRACKED_RUN_DEFAULT_KEYS:
        return False
    focus = resolve_domain_focus("lol_cn")
    return (
        value.get("id") == "lol_cn"
        and value.get("display_name") == "英雄联盟中文社区观察员"
        and value.get("domain_key") == focus.domain_key
        and value.get("observation_center") == focus.observation_center
        and value.get("relevance_rule") == focus.relevance_rule
        and value.get("attention_examples") == list(focus.attention_examples)
        and value.get("source_preferences") == list(focus.source_preferences)
        and value.get("branches") == [list(branch) for branch in focus.branches]
        and value.get("locale") == focus.locale
        and value.get("timezone") == focus.timezone
        and value.get("world_db") == "data/agents/lol_cn/world.sqlite3"
        and value.get("runtime_db") == "data/agents/lol_cn/runtime.sqlite3"
        and value.get("operator_instructions") is None
        and defaults.get("mode") == "broad"
        and defaults.get("broad_perspective") is None
        and defaults.get("deep_perspective") is None
        and defaults.get("adapters") == ["bilibili", "nga", "hupu", "public-web"]
        and defaults.get("max_turns") == 24
        and defaults.get("max_cost_usd") is None
        and defaults.get("wake_protocol") == "current"
        and defaults.get("memory_navigation") == "legacy"
        and defaults.get("digest_cache_reuse") is False
    )


class RunDefaults(BaseModel):
    """Safe everyday run controls saved with one Agent profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: AgentMode = "broad"
    broad_perspective: str | None = Field(default=None, max_length=20_000)
    deep_perspective: str | None = Field(default=None, max_length=20_000)
    adapters: tuple[str, ...] = ("bilibili", "nga", "hupu", "public-web")
    max_turns: int = Field(default=96, ge=1, le=200)
    max_cost_usd: float | None = Field(default=None, gt=0, le=1000)
    thinking: bool = False
    reasoning_effort: Literal["high", "max"] | None = Field(default=None)
    wake_protocol: WakeProtocol = "current"
    memory_navigation: Literal["legacy", "overview_v1"] = "overview_v1"
    digest_cache_reuse: bool = False

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_missions(cls, value: object) -> object:
        """Accept legacy profile JSON while serializing only perspective fields."""
        if not isinstance(value, Mapping):
            return value
        migrated = dict(value)
        for legacy, current in (
            ("broad_mission", "broad_perspective"),
            ("deep_mission", "deep_perspective"),
        ):
            if legacy not in migrated:
                continue
            legacy_value = migrated.pop(legacy)
            if current in migrated and migrated[current] != legacy_value:
                raise ValueError(f"{legacy} and {current} cannot disagree")
            migrated.setdefault(current, legacy_value)
        return migrated

    @field_validator("broad_perspective", "deep_perspective")
    @classmethod
    def trim_perspective(cls, value: str | None) -> str | None:
        """Normalize optional saved perspectives without manufacturing a task."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("adapters")
    @classmethod
    def validate_adapters(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Keep adapter names useful as command/configuration values."""
        cleaned = tuple(item.strip() for item in value if item.strip())
        if not cleaned:
            raise ValueError("at least one adapter is required")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("adapter names must be unique")
        if any("/" in item or "\\" in item for item in cleaned):
            raise ValueError("adapter names cannot contain path separators")
        return cleaned


class AgentProfile(BaseModel):
    """One domain-specific world-agent configuration with isolated storage paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str = Field(min_length=1, max_length=120)
    domain_key: str = Field(min_length=1, max_length=80)
    observation_center: str = Field(min_length=1, max_length=2000)
    relevance_rule: str = Field(min_length=1, max_length=4000)
    attention_examples: tuple[str, ...] = Field(min_length=1, max_length=24)
    source_preferences: tuple[str, ...] = Field(min_length=1, max_length=24)
    branches: tuple[tuple[str, str], ...] | None = Field(default=None, max_length=24)
    search_experience: tuple[str, ...] | None = Field(default=None, max_length=24)
    locale: str = Field(min_length=1, max_length=80)
    timezone: str = Field(min_length=1, max_length=80)
    defaults: RunDefaults = Field(default_factory=RunDefaults)
    adapter_defaults: dict[str, dict[str, str]] | None = Field(default=None)
    world_db: str
    runtime_db: str
    operator_instructions: str | None = Field(default=None, max_length=6000)

    @field_validator("adapter_defaults")
    @classmethod
    def validate_adapter_defaults(
        cls, value: dict[str, dict[str, str]] | None
    ) -> dict[str, dict[str, str]] | None:
        """Reject unknown adapter ids so a typo can never silently degrade."""
        if value is None:
            return None
        unknown = set(value) - _SURFACE_PARAMETER_ADAPTERS
        if unknown:
            raise ValueError(
                f"unknown adapter defaults: {', '.join(sorted(unknown))}; "
                f"supported: {', '.join(sorted(_SURFACE_PARAMETER_ADAPTERS))}"
            )
        return value

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Keep profile identifiers safe for registry filenames."""
        return _safe_profile_id(value)

    @field_validator(
        "domain_key", "display_name", "observation_center", "relevance_rule", "locale", "timezone"
    )
    @classmethod
    def trim_required_text(cls, value: str) -> str:
        """Reject whitespace-only profile text."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @field_validator("attention_examples", "source_preferences")
    @classmethod
    def trim_text_list(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank entries in domain-lens lists."""
        cleaned = tuple(item.strip() for item in value if item.strip())
        if not cleaned:
            raise ValueError("list must include at least one non-empty item")
        return cleaned

    @field_validator("branches")
    @classmethod
    def trim_branches(
        cls, value: tuple[tuple[str, str], ...] | None
    ) -> tuple[tuple[str, str], ...] | None:
        """Keep optional branch names and descriptions useful for Prompt rendering."""
        if value is None:
            return None
        cleaned = tuple((name.strip(), description.strip()) for name, description in value)
        if any(not name or not description for name, description in cleaned):
            raise ValueError("branch names and descriptions must not be blank")
        if len({name for name, _ in cleaned}) != len(cleaned):
            raise ValueError("branch names must be unique")
        return cleaned

    @field_validator("search_experience")
    @classmethod
    def validate_search_experience(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """Normalize portable search guidance while preserving the legacy sentinel."""
        if value is None:
            return None
        cleaned = tuple(item.strip() for item in value)
        if any(not item or len(item) > 240 for item in cleaned):
            raise ValueError("search experience entries must contain 1..240 characters")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("search experience entries must be unique")
        return cleaned

    @field_validator("world_db", "runtime_db")
    @classmethod
    def validate_database_path(cls, value: str) -> str:
        """Keep database locations relative to the chosen workspace root."""
        return _safe_relative_path(value)

    @field_validator("operator_instructions")
    @classmethod
    def validate_operator_instructions(cls, value: str | None) -> str | None:
        """Keep optional instructions separate from credentials and blank text."""
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if any(marker in cleaned.casefold() for marker in _SECRET_MARKERS):
            raise ValueError("operator instructions must not contain credentials")
        return cleaned

    @model_validator(mode="after")
    def separate_database_paths(self) -> AgentProfile:
        """Do not let runtime checkpoints share the durable world database."""
        if self.world_db.casefold() == self.runtime_db.casefold():
            raise ValueError("world_db and runtime_db must be different paths")
        return self

    @property
    def focus(self) -> DomainFocus:
        """Build the engine's immutable domain lens from editable profile fields."""
        return DomainFocus(
            domain_key=self.domain_key,
            observation_center=self.observation_center,
            relevance_rule=self.relevance_rule,
            attention_examples=self.attention_examples,
            source_preferences=self.source_preferences,
            locale=self.locale,
            timezone=self.timezone,
            branches=self.branches or (),
            search_experience=self.search_experience or (),
        )


def build_quick_profile(
    *,
    agent_id: str,
    display_name: str,
    observation_center: str,
    locale: str,
    timezone: str,
    adapters: tuple[str, ...],
) -> AgentProfile:
    """Build a neutral, isolated profile without opening either database.

    The short creation flow asks only for a durable observation identity and
    available adapters.  The common cognition mechanics remain shared code,
    so this function deliberately emits stable, non-experimental run defaults.
    """
    safe_id = _safe_profile_id(agent_id)
    defaults = RunDefaults(adapters=adapters)
    database_directory = f"data/agents/{safe_id}"
    return AgentProfile(
        id=safe_id,
        display_name=display_name,
        domain_key=safe_id,
        observation_center=observation_center,
        relevance_rule=_QUICK_RELEVANCE_RULE,
        attention_examples=_QUICK_ATTENTION_EXAMPLES,
        source_preferences=_QUICK_SOURCE_PREFERENCES,
        branches=(),
        search_experience=(),
        locale=locale,
        timezone=timezone,
        defaults=defaults,
        world_db=f"{database_directory}/world.sqlite3",
        runtime_db=f"{database_directory}/runtime.sqlite3",
    )


def clone_profile_design(
    source: AgentProfile,
    *,
    agent_id: str,
    display_name: str,
) -> AgentProfile:
    """Copy an Agent's domain design into a fresh, memory-free identity.

    Persistent cognition, operator additions, saved wake perspectives, and
    experimental run switches are intentionally not copied.  The returned
    profile has new database paths but does not create files itself.
    """
    safe_id = _safe_profile_id(agent_id)
    database_directory = f"data/agents/{safe_id}"
    stable_defaults = source.defaults.model_copy(
        update={
            "broad_perspective": None,
            "deep_perspective": None,
            "wake_protocol": "current",
            "memory_navigation": "overview_v1",
            "digest_cache_reuse": False,
        }
    )
    return AgentProfile(
        id=safe_id,
        display_name=display_name,
        domain_key=source.domain_key,
        observation_center=source.observation_center,
        relevance_rule=source.relevance_rule,
        attention_examples=source.attention_examples,
        source_preferences=source.source_preferences,
        branches=source.branches,
        search_experience=source.search_experience,
        locale=source.locale,
        timezone=source.timezone,
        defaults=stable_defaults,
        world_db=f"{database_directory}/world.sqlite3",
        runtime_db=f"{database_directory}/runtime.sqlite3",
        operator_instructions=None,
    )


class PromptPreview(BaseModel):
    """UI-facing, named Prompt layers; core layers remain derived and read-only."""

    model_config = ConfigDict(frozen=True)

    stable: str
    domain: str
    posture: str
    epistemic: str
    mechanics: str
    operator: str | None
    wake_input: str
    compiled: str


def build_prompt_preview(
    profile: AgentProfile,
    *,
    mode: AgentMode | None = None,
    wake_protocol: WakeProtocol | None = None,
    object_id: str | None = None,
    perspective: str | None = None,
) -> PromptPreview:
    """Compile a preview without changing the engine's default prompt behavior."""
    # G5b-2: the wake protocol selector is retired; Graph Shell is the only
    # normal runtime, so the preview always compiles graph_shell_prompt (the
    # parameter stays accepted for API compatibility but is ignored).
    del wake_protocol
    selected_mode = mode or profile.defaults.mode
    layers = prompt_layers(
        selected_mode,
        object_id,
        profile.focus,
        include_search_experience=(selected_mode == "broad"),
    )
    compiled = graph_shell_prompt(
        selected_mode,
        object_id,
        profile.focus,
        operator_instructions=profile.operator_instructions,
        include_search_experience=(selected_mode == "broad"),
    )
    return PromptPreview(
        stable=layers.stable,
        domain=layers.domain,
        posture=layers.posture,
        epistemic=layers.epistemic,
        mechanics=layers.mechanics,
        operator=profile.operator_instructions,
        wake_input=render_wake_input(perspective),
        compiled=compiled,
    )


class AgentProfileRegistry:
    """One-JSON-file-per-profile registry with filename and traversal safeguards."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory.resolve()

    @property
    def directory(self) -> Path:
        """Expose the resolved registry root for local companion data lookups."""
        return self._directory

    def list(self) -> list[AgentProfile]:
        """Return registered profiles in deterministic display order."""
        if not self._directory.exists():
            return []
        profiles = [self._read(path) for path in self._directory.glob("*.json")]
        return sorted(profiles, key=lambda item: (item.display_name.casefold(), item.id))

    def get(self, profile_id: str) -> AgentProfile:
        """Return one profile or fail clearly when its file does not exist."""
        path = self._profile_path(profile_id)
        if not path.is_file():
            raise KeyError(f"unknown profile: {profile_id}")
        return self._read(path)

    def create(self, profile: AgentProfile) -> AgentProfile:
        """Atomically create a new profile without replacing an existing one."""
        path = self._profile_path(profile.id)
        if path.exists():
            raise FileExistsError(f"profile already exists: {profile.id}")
        self._ensure_unique_database_paths(profile)
        self._write(path, profile)
        return profile

    def update(self, profile: AgentProfile) -> AgentProfile:
        """Atomically replace an existing profile with the same safe identifier."""
        path = self._profile_path(profile.id)
        if not path.is_file():
            raise KeyError(f"unknown profile: {profile.id}")
        self._ensure_unique_database_paths(profile, exclude_id=profile.id)
        self._write(path, profile)
        return profile

    def _ensure_unique_database_paths(
        self, profile: AgentProfile, *, exclude_id: str | None = None
    ) -> None:
        """Keep every registered Agent's world and runtime stores isolated."""
        requested = {profile.world_db.casefold(), profile.runtime_db.casefold()}
        for existing in self.list():
            if existing.id == exclude_id:
                continue
            collision = requested.intersection(
                {existing.world_db.casefold(), existing.runtime_db.casefold()}
            )
            if collision:
                paths = ", ".join(sorted(collision))
                raise ValueError(f"database path already used by profile {existing.id}: {paths}")

    def bootstrap_lol_profile(self) -> AgentProfile:
        """Create the default LOL profile once, retaining the registered engine lens."""
        try:
            path = self._profile_path("lol_cn")
            if not path.is_file():
                raise KeyError("unknown profile: lol_cn")
            raw_profile = json.loads(path.read_text(encoding="utf-8"))
            migrate_builtin_defaults = _is_legacy_builtin_lol_profile(raw_profile)
            migrate_search_experience = (
                migrate_builtin_defaults
                or _is_tracked_lol_profile_without_search_experience(raw_profile)
            )
            profile = self.get("lol_cn")
            updates: dict[str, object] = {}
            if profile.display_name == "LOL CN Observer":
                updates["display_name"] = "英雄联盟中文社区观察员"
            if migrate_builtin_defaults:
                updates["defaults"] = profile.defaults.model_copy(
                    update={"broad_perspective": None, "deep_perspective": None}
                )
            if profile.branches is None:
                updates["branches"] = resolve_domain_focus("lol_cn").branches
            if migrate_search_experience:
                updates["search_experience"] = resolve_domain_focus("lol_cn").search_experience
            if updates:
                profile = profile.model_copy(update=updates)
                return self.update(profile)
            return profile
        except KeyError:
            focus = resolve_domain_focus("lol_cn")
            return self.create(
                AgentProfile(
                    id="lol_cn",
                    display_name="英雄联盟中文社区观察员",
                    domain_key=focus.domain_key,
                    observation_center=focus.observation_center,
                    relevance_rule=focus.relevance_rule,
                    attention_examples=focus.attention_examples,
                    source_preferences=focus.source_preferences,
                    branches=focus.branches,
                    search_experience=focus.search_experience,
                    locale=focus.locale,
                    timezone=focus.timezone,
                    defaults=_lol_run_defaults(),
                    adapter_defaults={
                        "hupu": {"board": "lol"},
                        "nga": {"fid": "-152678"},
                    },
                    world_db="data/agents/lol_cn/world.sqlite3",
                    runtime_db="data/agents/lol_cn/runtime.sqlite3",
                )
            )

    def _profile_path(self, profile_id: str) -> Path:
        safe_id = _safe_profile_id(profile_id)
        path = (self._directory / f"{safe_id}.json").resolve()
        if path.parent != self._directory:
            raise ValueError("profile path escapes registry directory")
        return path

    def _read(self, path: Path) -> AgentProfile:
        if path.resolve().parent != self._directory:
            raise ValueError("profile path escapes registry directory")
        profile = AgentProfile.model_validate_json(path.read_text(encoding="utf-8"))
        if path.stem != profile.id:
            raise ValueError("profile filename must match its id")
        return profile

    def _write(self, path: Path, profile: AgentProfile) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._directory,
            prefix=f".{profile.id}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(profile.model_dump_json(indent=2) + "\n")
        try:
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _lol_run_defaults() -> RunDefaults:
    """Return neutral defaults; a wake perspective is always optional."""
    return RunDefaults()
