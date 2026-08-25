"""Immutable public contracts for durable world cognition."""

# ruff: noqa: D101

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeAliasType

from .graph_contract import (
    AliasAction,
    normalize_identity_alias,
    normalize_qualifiers,
    normalize_type_key,
)

JsonValue = TypeAliasType(
    "JsonValue", str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class ObjectKind(StrEnum):
    """Kinds of universal world objects.

    The six durable kinds are ``entity`` (fallback), ``person``,
    ``organization``, ``place``, ``event`` and ``concept``; they match the
    graph_patch ``OBJECT_KINDS`` enum, so staged kinds publish verbatim.
    ``SOURCE``, ``METHOD`` and ``RULE`` are legacy values kept for old
    commits, replay and console parsing; the historical terminal
    ``submit_cognition_schema`` narrows proposals to ``entity``, ``event``,
    ``concept`` and the committer routes legacy kinds into core kinds with a
    default ``type_key`` at that proposal boundary.
    """

    ENTITY = "entity"
    PERSON = "person"
    EVENT = "event"
    CONCEPT = "concept"
    SOURCE = "source"
    PLACE = "place"
    ORGANIZATION = "organization"
    METHOD = "method"
    RULE = "rule"


class EpistemicRole(StrEnum):
    """Epistemic roles available to assertions."""

    FACT = "fact"
    COMMUNITY_VIEW = "community_view"
    SEMANTIC_EXPLANATION = "semantic_explanation"
    AGENT_SYNTHESIS = "agent_synthesis"
    UNCERTAINTY = "uncertainty"
    META_KNOWLEDGE = "meta_knowledge"


class ObservationDepth(StrEnum):
    """How deeply an observation has been inspected."""

    SEEN = "seen"
    CONTENT = "content"
    DISCUSSION = "discussion"
    MEDIA = "media"


EventTimePrecision = Literal["exact", "interval", "period", "unknown"]
"""How precisely a stored event-time window is known (spec §7).

``exact`` — both bounds are a specific moment; ``interval`` — a closed
start/end range; ``period`` — a coarser recurring or ambiguous window;
``unknown`` — no temporal information (the v9 backfill default). Value
validation is the pydantic layer's job; the column itself is plain TEXT.
The field deliberately does NOT participate in the assertion signature:
precision is interpretation metadata, not an identity input, and adding it
would break replay of every stored commit.
"""


class WorldModel(BaseModel):
    """Base class that freezes public world contract attributes."""

    model_config = ConfigDict(frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def _normalize_datetimes(cls, value: object) -> object:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("datetime values must be timezone-aware UTC datetimes")
            return value.astimezone(UTC)
        return value


class ObjectInput(WorldModel):
    """An object proposed for durable world memory."""

    id: str
    kind: ObjectKind
    type_key: str | None = None
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    domain_hints: list[str] = Field(default_factory=list)
    provisional: bool = False
    event_time_start: datetime | None = None
    event_time_end: datetime | None = None
    event_time_precision: EventTimePrecision = "unknown"
    expected_version: int | None = None

    @field_validator("type_key", mode="before")
    @classmethod
    def _normalize_type_key(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return normalize_type_key(value)

    @model_validator(mode="after")
    def _validate_object(self) -> ObjectInput:
        _require_text(self.id, "object id")
        _require_text(self.canonical_name, "canonical name")
        _validate_interval(self.event_time_start, self.event_time_end)
        return self


class ObservationInput(WorldModel):
    """A source observation that may evidence an assertion."""

    id: str
    source_uri: str
    source_kind: str
    title: str = ""
    excerpt: str = ""
    content_ref: str = ""
    depth: ObservationDepth
    source_published_at: datetime | None = None
    observed_at: datetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_observation(self) -> ObservationInput:
        _require_text(self.id, "observation id")
        _require_text(self.source_uri, "source uri")
        _require_text(self.source_kind, "source kind")
        return self


class EvidenceInput(WorldModel):
    """A typed link from an assertion to an observation."""

    observation_id: str
    role: Literal["supports", "contradicts", "context"]

    @model_validator(mode="after")
    def _validate_evidence(self) -> EvidenceInput:
        _require_text(self.observation_id, "evidence observation id")
        return self


class AssertionInput(WorldModel):
    """A predicate assertion, optionally backed by observation evidence."""

    id: str
    subject_id: str
    predicate: str
    object_id: str | None = None
    literal: JsonValue | None = None
    epistemic_role: EpistemicRole
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceInput] = Field(default_factory=list)
    qualifiers: dict[str, str] = Field(default_factory=dict)
    event_time_start: datetime | None = None
    event_time_end: datetime | None = None
    event_time_precision: EventTimePrecision = "unknown"
    supersedes_id: str | None = None
    superseded_at: datetime | None = None
    supersede_reason: str | None = None
    answers_inquiry_id: str | None = None

    @field_validator("qualifiers", mode="before")
    @classmethod
    def _normalize_qualifiers(cls, value: object) -> object:
        return normalize_qualifiers(value)

    @model_validator(mode="after")
    def _validate_assertion(self) -> AssertionInput:
        _require_text(self.id, "assertion id")
        _require_text(self.subject_id, "assertion subject id")
        _require_text(self.predicate, "assertion predicate")
        if (self.object_id is None) == (self.literal is None):
            raise ValueError("assertion requires exactly one of object_id or literal")
        if self.object_id is not None:
            _require_text(self.object_id, "assertion object id")
        if self.answers_inquiry_id is not None:
            _require_text(self.answers_inquiry_id, "answers inquiry id")
        if self.supersede_reason is not None:
            _require_text(self.supersede_reason, "supersede reason")
        _validate_interval(self.event_time_start, self.event_time_end)
        return self


class InquiryInput(WorldModel):
    """A durable unresolved question about a world object."""

    id: str
    subject_id: str
    prompt: str
    rationale: str
    deepens_id: str | None = None
    kind: Literal["factual", "semantic", "stateful"] = "factual"
    created_at: datetime | None = None
    last_attempted_at: datetime | None = None
    attempt_count: int = 0
    resolved_at: datetime | None = None
    expected_version: int | None = None

    @model_validator(mode="after")
    def _validate_inquiry(self) -> InquiryInput:
        _require_text(self.id, "inquiry id")
        _require_text(self.subject_id, "inquiry subject id")
        _require_text(self.prompt, "inquiry prompt")
        _require_text(self.rationale, "inquiry rationale")
        return self


class InquiryResolution(WorldModel):
    """A version-checked request to resolve an inquiry."""

    id: str
    expected_version: int

    @model_validator(mode="after")
    def _validate_resolution(self) -> InquiryResolution:
        _require_text(self.id, "inquiry resolution id")
        return self


class ObservationLinkInput(WorldModel):
    """A candidate or context link from cognition to acquired material."""

    target_kind: Literal["object", "inquiry"]
    target_id: str
    observation_id: str
    role: Literal["candidate", "context"]

    @model_validator(mode="after")
    def _validate_link(self) -> ObservationLinkInput:
        _require_text(self.target_id, "observation link target id")
        _require_text(self.observation_id, "observation link observation id")
        return self


class AliasOperation(WorldModel):
    """One identity-alias mutation recorded in a durable cognitive delta."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_id: str
    raw_alias: str
    normalized_alias: str
    action: AliasAction

    @model_validator(mode="after")
    def _validate_alias_operation(self) -> AliasOperation:
        _require_text(self.object_id, "object id")
        if self.normalized_alias != normalize_identity_alias(self.raw_alias):
            raise ValueError("normalized_alias must equal the shared alias normalization")
        return self


class CognitiveDelta(WorldModel):
    """All durable cognition to be committed atomically."""

    observations: list[ObservationInput] = Field(default_factory=list)
    objects: list[ObjectInput] = Field(default_factory=list)
    assertions: list[AssertionInput] = Field(default_factory=list)
    inquiries: list[InquiryInput] = Field(default_factory=list)
    resolve_inquiries: list[InquiryResolution] = Field(default_factory=list)
    observation_links: list[ObservationLinkInput] = Field(default_factory=list)
    alias_operations: list[AliasOperation] = Field(default_factory=list)


class CommitReceipt(WorldModel):
    """The durable outcome of one world commit."""

    commit_id: str
    committed_at: datetime
    object_ids: list[str]
    observation_ids: list[str]
    assertion_ids: list[str]
    inquiry_ids: list[str]
    resolved_inquiry_ids: list[str]
    replayed: bool = False


# P3-A's overview is deliberately a small, read-only navigation contract.  It
# records durable evidence for every surfaced reason instead of exposing a
# rank/score that an agent cannot audit.
MemoryOverviewReason = Literal[
    "recent_observation",
    "recent_cognition_change",
    "related_active_neighbor",
    "repeated_inquiry_point",
    "underexplored",
    "dormant_reactivated",
]
MemoryOverviewDepth = Literal["seen", "content", "discussion", "media"]
MemoryOverviewCoverage = Literal["none", "attempted", "seen", "content", "discussion", "media", "answered"]


class InquiryCoverage(WorldModel):
    """Inquiry-local evidence summary; subject context is explicitly secondary."""

    direct_observation_count: int = Field(ge=0)
    direct_max_depth: MemoryOverviewDepth | None = None
    answering_assertion_count: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    last_attempted_at: datetime | None = None
    coverage: MemoryOverviewCoverage
    subject_context_depth: MemoryOverviewDepth | None = None


class MemoryOverviewFact(WorldModel):
    """One durable fact used to explain why a navigation candidate surfaced."""

    id: str
    kind: Literal["object", "observation", "assertion", "inquiry"]
    at: datetime | None = None
    related_id: str | None = None


class MemoryOverviewCandidate(WorldModel):
    """A candidate memory, not a recommendation or a task instruction."""

    id: str
    kind: Literal["object", "inquiry"]
    name: str
    prompt: str | None = None
    surfaced_because: list[MemoryOverviewReason]
    facts: list[MemoryOverviewFact]
    inquiry_coverage: InquiryCoverage | None = None


class MemoryOverview(WorldModel):
    """Bounded map of durable memory frontiers at a fixed UTC instant."""

    as_of: datetime
    counts: MemoryOverviewCounts
    active_fronts: list[MemoryOverviewCandidate] = Field(default_factory=list)
    reactivated_fronts: list[MemoryOverviewCandidate] = Field(default_factory=list)
    cold_bridges: list[MemoryOverviewCandidate] = Field(default_factory=list)
    coverage_gaps: list[MemoryOverviewCandidate] = Field(default_factory=list)
    truncated: bool = Field(default=False, exclude=True)


class MemoryOverviewCounts(WorldModel):
    """Exact library-presence counters, deliberately not an open-ended map."""

    objects: int = Field(ge=0)
    assertions: int = Field(ge=0)
    open_inquiries: int = Field(ge=0)
    dormant_inquiries: int = Field(ge=0)


def _require_text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _validate_interval(start: datetime | None, end: datetime | None) -> None:
    if start is not None and end is not None and start > end:
        raise ValueError("time interval start must be before or equal to end")


class RecallClock(StrEnum):
    WORLD = "world"
    KNOWLEDGE = "knowledge"


class MemoryBundle(BaseModel):
    anchor_objects: list[dict[str, Any]] = Field(default_factory=list)
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    neighboring_objects: list[dict[str, Any]] = Field(default_factory=list)
    inquiries: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    candidate_observation_refs: list[dict[str, Any]] = Field(default_factory=list)
    paths: list[list[str]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    ego_neighbors: list[dict[str, Any]] = Field(default_factory=list)
    status_assertions: list[dict[str, Any]] = Field(default_factory=list)
    event_timeline: list[dict[str, Any]] = Field(default_factory=list)
    event_edges: list[dict[str, Any]] = Field(default_factory=list)
    participated_events: list[dict[str, Any]] = Field(default_factory=list)
    omitted_counts: dict[str, int] = Field(default_factory=dict)
    sort_basis: str = ""
    event_next_cursor: str | None = None
    # memory_read additive fields (design §4.3): the identity portrait, the
    # expanded three-direction views with their cut flags, and the evidence
    # view's supersede chain.
    identity: dict[str, Any] | None = None
    views: dict[str, Any] = Field(default_factory=dict)
    view_truncated: dict[str, bool] = Field(default_factory=dict)
    assertion_chain: dict[str, Any] = Field(default_factory=dict)
    # memory_compare additive field (design §4.4): the side-by-side field
    # rows with shared/left_only/right_only splits. States facts and
    # differences only — never a merge verdict.
    compare: dict[str, Any] | None = None
    truncated: bool = Field(default=False, exclude=True)
