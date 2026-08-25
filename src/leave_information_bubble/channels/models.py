"""Contracts exchanged between visibility scheduling and channel adapters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from leave_information_bubble.models.epistemics import SourceObservation


class ChannelHealthStatus(StrEnum):
    """Operational state used by the scheduler without platform-specific logic."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    COOLDOWN = "cooldown"
    AUTHENTICATION_REQUIRED = "authentication_required"
    BLOCKED = "blocked"
    SCHEMA_CHANGED = "schema_changed"
    UNAVAILABLE = "unavailable"


class AcquisitionOutcome(StrEnum):
    """Operational result of one bounded adapter request."""

    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"


class HydrationDepth(StrEnum):
    """Bounded acquisition depth for one already discovered source."""

    METADATA = "metadata"
    TEXT = "text"
    REACTIONS = "reactions"
    MEDIA_TEXT = "media_text"
    STRUCTURED = "structured"


class ChannelCapabilityRole(StrEnum):
    """Platform-neutral retrieval role exposed by a channel adapter."""

    SEARCH_INDEX = "search_index"
    NEWS_INDEX = "news_index"
    OFFICIAL_FEED = "official_feed"
    PLATFORM_DISCOVERY = "platform_discovery"
    COMMUNITY_STREAM = "community_stream"
    STRUCTURED_CHANGE = "structured_change"
    PUBLIC_METRIC = "public_metric"
    WEB_BROWSER = "web_browser"
    MEDIA_TEXT = "media_text"
    ATTENTION_RANKING = "attention_ranking"
    RECENT_STREAM = "recent_stream"
    CURRENT_INDEX = "current_index"


class AcquisitionEntryKind(StrEnum):
    """The real acquisition surface an adapter enters."""

    PLATFORM_SEARCH = "platform_search"
    RANKED_SURFACE = "ranked_surface"
    BOUNDED_BOARD = "bounded_board"
    SOURCE_HYDRATION = "source_hydration"


class QuerySemantics(StrEnum):
    """How a query affects acquisition on a declared surface."""

    PLATFORM_SEARCH = "platform_search"
    BOUNDED_RERANK_HINT = "bounded_rerank_hint"
    UNSUPPORTED = "unsupported"


class TimeFilterPrecision(StrEnum):
    """Truthful precision of an adapter's provider-side time filtering."""

    UNSUPPORTED = "unsupported"
    COARSE = "coarse"
    EXACT = "exact"


class IndependenceStatus(StrEnum):
    """Conservative independence state for a possible content origin."""

    UNKNOWN = "unknown"
    SHARED_ORIGIN = "shared_origin"
    INDEPENDENT = "independent"


class CapabilityDescriptor(BaseModel):
    """Versioned declaration of one acquisition capability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="1.0", min_length=1, max_length=20)
    adapter_id: str = Field(min_length=1, max_length=200)
    adapter_version: str = Field(min_length=1, max_length=100)
    role: ChannelCapabilityRole
    entry_kind: AcquisitionEntryKind
    query_semantics: QuerySemantics
    supports_queryless: bool
    languages: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    supports_cursor: bool = False
    supports_metrics: bool = False
    supports_authentication: bool = False
    requires_authentication: bool = False
    time_filter_precision: TimeFilterPrecision = TimeFilterPrecision.UNSUPPORTED
    bounded_limit: int = Field(default=20, ge=1, le=1000)
    relative_cost: float = Field(default=1.0, ge=0.0)
    limitations: list[str] = Field(default_factory=list)


class ChannelHealth(BaseModel):
    """Current adapter health with an explicit retry horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    adapter_id: str = Field(min_length=1, max_length=200)
    adapter_version: str = Field(min_length=1, max_length=100)
    status: ChannelHealthStatus
    checked_at: datetime
    retry_after: datetime | None = None
    failure_code: str = ""
    detail: str = Field(default="", max_length=1000)
    data_freshness_seconds: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Seconds between checked_at and most recent source content timestamp, "
            "or None if never discovered"
        ),
    )

    @model_validator(mode="after")
    def validate_retry_horizon(self) -> ChannelHealth:
        """Require retry_after for time-bounded rate limits and cooldowns."""
        needs_retry = self.status in {
            ChannelHealthStatus.RATE_LIMITED,
            ChannelHealthStatus.COOLDOWN,
        }
        if needs_retry and self.retry_after is None:
            raise ValueError(f"{self.status.value} health requires retry_after")
        if self.retry_after is not None and self.retry_after < self.checked_at:
            raise ValueError("retry_after cannot precede checked_at")
        return self


class ScanRequest(BaseModel):
    """One bounded discovery request independent of platform APIs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=300)
    lane: str = Field(min_length=1, max_length=100)
    query: str = Field(default="", max_length=1000)
    domain_ids: list[str] = Field(default_factory=list)
    window_start: datetime | None = None
    window_end: datetime | None = None
    limit: int = Field(default=20, ge=1, le=500)
    cursor: str = Field(default="", max_length=2000)
    language: str = Field(default="", max_length=50)
    region: str = Field(default="", max_length=50)
    timezone: str = Field(default="", max_length=100)
    capability_roles: list[ChannelCapabilityRole] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_window(self) -> ScanRequest:
        """Accept an unfiltered scan or one complete, ordered interval."""
        if (self.window_start is None) != (self.window_end is None):
            raise ValueError("window_start and window_end must be supplied together")
        if (
            self.window_start is not None
            and self.window_end is not None
            and self.window_end < self.window_start
        ):
            raise ValueError("window_end cannot precede window_start")
        return self


class HydrationRequest(BaseModel):
    """One bounded request to deepen an already discovered source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=300)
    source_ref: str = Field(min_length=1, max_length=2000)
    depth: HydrationDepth
    max_bytes: int = Field(default=10_000_000, ge=1)
    max_duration_seconds: int = Field(default=1200, ge=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class SourceOccurrence(BaseModel):
    """One platform occurrence found before semantic interpretation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=300)
    adapter_id: str = Field(min_length=1, max_length=200)
    adapter_version: str = Field(min_length=1, max_length=100)
    source_ref: str = Field(min_length=1, max_length=2000)
    canonical_url: str = Field(default="", max_length=4000)
    title: str = Field(default="", max_length=2000)
    author: str = Field(default="", max_length=500)
    content_type: str = Field(default="", max_length=100)
    language: str = Field(default="", max_length=50)
    source_published_at: datetime | None = None
    captured_at: datetime
    content_hash: str = Field(default="", max_length=128)
    content_hash_verified: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_verified_hash(self) -> SourceOccurrence:
        """Accept a dedup hash only when an adapter attests a SHA-256 identity."""
        if not self.content_hash_verified:
            return self
        if len(self.content_hash) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.content_hash
        ):
            raise ValueError("verified content_hash must be a 64-character hexadecimal SHA-256")
        return self


class RetrievalHit(BaseModel):
    """One inexpensive index hit before document hydration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=300)
    request_id: str = Field(min_length=1, max_length=300)
    provider: str = Field(min_length=1, max_length=100)
    capability_role: ChannelCapabilityRole
    query: str = Field(min_length=1, max_length=1000)
    rank: int = Field(ge=1)
    url: str = Field(min_length=1, max_length=4000)
    canonical_url: str = Field(min_length=1, max_length=4000)
    title: str = Field(default="", max_length=2000)
    snippet: str = Field(default="", max_length=8000)
    detected_language: str = Field(default="", max_length=50)
    published_at: datetime | None = None
    captured_at: datetime
    content_hash: str = Field(default="", max_length=128)
    content_hash_verified: bool = False
    origin_cluster_id: str = Field(default="", max_length=300)
    independence_status: IndependenceStatus = IndependenceStatus.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_verified_hash(self) -> RetrievalHit:
        """Accept a verified identity only for a SHA-256-shaped digest."""
        if not self.content_hash_verified:
            return self
        if len(self.content_hash) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.content_hash
        ):
            raise ValueError("verified content_hash must be a 64-character hexadecimal SHA-256")
        return self


class RetrievalContractReport(BaseModel):
    """Observed search quality and provider-contract compliance for one batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_language: str = Field(default="zh-Hans", max_length=50)
    target_region: str = Field(default="CN", max_length=50)
    timezone: str = Field(default="Asia/Shanghai", max_length=100)
    result_count: int = Field(default=0, ge=0)
    target_language_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_anchor_hit_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    dated_result_count: int = Field(default=0, ge=0)
    in_window_result_count: int = Field(default=0, ge=0)
    time_filter_requested: bool = False
    time_filter_applied: bool = False
    time_filter_precision: TimeFilterPrecision = TimeFilterPrecision.UNSUPPORTED
    exact_duplicate_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    near_duplicate_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    dominant_host_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    empty_or_challenge_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    cursor_supported: bool = False
    cursor_trustworthy: bool = False
    count_trustworthy: bool = True
    provider_attempts: list[str] = Field(default_factory=list)
    degraded_reasons: list[str] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """Return whether an operational quality threshold was crossed."""
        return bool(self.degraded_reasons)


class RetrievalBatch(BaseModel):
    """Rich inexpensive retrieval result retained beside legacy occurrences."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=300)
    adapter_id: str = Field(min_length=1, max_length=200)
    adapter_version: str = Field(min_length=1, max_length=100)
    hits: list[RetrievalHit] = Field(default_factory=list)
    contract_report: RetrievalContractReport
    next_cursor: str = Field(default="", max_length=2000)
    partial: bool = False
    limitations: list[str] = Field(default_factory=list)
    physical_call_count: int = Field(default=0, ge=0)


class DiscoveryBatch(BaseModel):
    """Normalized, possibly partial result from one discovery request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=300)
    adapter_id: str = Field(min_length=1, max_length=200)
    adapter_version: str = Field(min_length=1, max_length=100)
    outcome: AcquisitionOutcome = AcquisitionOutcome.SUCCESS
    occurrences: list[SourceOccurrence] = Field(default_factory=list)
    next_cursor: str = Field(default="", max_length=2000)
    partial: bool = False
    limitations: list[str] = Field(default_factory=list)
    physical_call_count: int = Field(default=0, ge=0)
    retrieval: RetrievalBatch | None = None


class ObservationBatch(BaseModel):
    """Normalized result from one source hydration request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=300)
    adapter_id: str = Field(min_length=1, max_length=200)
    adapter_version: str = Field(min_length=1, max_length=100)
    outcome: AcquisitionOutcome = AcquisitionOutcome.SUCCESS
    observations: list[SourceObservation] = Field(default_factory=list)
    discovered_occurrences: list[SourceOccurrence] = Field(default_factory=list, max_length=8)
    partial: bool = False
    limitations: list[str] = Field(default_factory=list)
    physical_call_count: int = Field(default=0, ge=0)


__all__ = [
    "AcquisitionEntryKind",
    "AcquisitionOutcome",
    "CapabilityDescriptor",
    "ChannelCapabilityRole",
    "ChannelHealth",
    "ChannelHealthStatus",
    "DiscoveryBatch",
    "HydrationDepth",
    "HydrationRequest",
    "IndependenceStatus",
    "ObservationBatch",
    "QuerySemantics",
    "RetrievalBatch",
    "RetrievalContractReport",
    "RetrievalHit",
    "ScanRequest",
    "SourceOccurrence",
    "TimeFilterPrecision",
]
