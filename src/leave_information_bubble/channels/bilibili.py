"""Bilibili vNext channel adapter with bounded, typed hydration."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from leave_information_bubble.models.epistemics import (
    AccessDepth,
    ObservationModality,
    SourceObservation,
)
from leave_information_bubble.tools.bilibili_search import (
    SUBTITLE_ABSENT_LIMITATION,
    BilibiliAudiencePrecision,
    BilibiliAudienceValue,
    BilibiliCommentPage,
    BilibiliConcurrentAudienceResult,
    BilibiliSearchResult,
)
from leave_information_bubble.tools.transcription import (
    ASR_AUDIO_EMPTY_LIMITATION,
    ASR_DURATION_EXCEEDED_LIMITATION,
    AudioTranscriber,
)

from .models import (
    AcquisitionEntryKind,
    AcquisitionOutcome,
    CapabilityDescriptor,
    ChannelCapabilityRole,
    ChannelHealth,
    ChannelHealthStatus,
    DiscoveryBatch,
    HydrationDepth,
    HydrationRequest,
    ObservationBatch,
    QuerySemantics,
    ScanRequest,
    SourceOccurrence,
    TimeFilterPrecision,
)

_BVID_PATTERN = re.compile(r"(BV[0-9A-Za-z]+)", re.IGNORECASE)
_FULL_BODY_CAP = 32_000
logger = logging.getLogger(__name__)


class BilibiliCommentSort(StrEnum):
    """Public comment orderings supported by Bilibili."""

    HOT = "hot"
    NEWEST = "newest"


class BilibiliCommentSampling(BaseModel):
    """Bounded sampling parameters accepted through hydration arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sorts: list[BilibiliCommentSort] = Field(
        default_factory=lambda: [
            BilibiliCommentSort.HOT,
            BilibiliCommentSort.NEWEST,
        ],
        min_length=1,
        max_length=2,
    )
    per_sort_limit: int = Field(default=20, ge=1, le=50)
    sort_limits: dict[BilibiliCommentSort, int] = Field(default_factory=dict)
    max_total: int = Field(default=40, ge=0, le=200)
    reply_limit: int = Field(default=5, ge=0, le=10)
    page: int = Field(default=1, ge=1, le=100)
    include_danmaku: bool = True
    danmaku_limit: int = Field(default=300, ge=1, le=2000)


class _BilibiliTool(Protocol):
    async def search(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 20,
        *,
        order: str = "totalrank",
    ) -> BilibiliSearchResult: ...

    async def get_video_info(self, bvid: str) -> dict[str, Any]: ...

    async def get_concurrent_audience(
        self,
        bvid: str,
        cid: int,
    ) -> BilibiliConcurrentAudienceResult: ...

    async def get_video_tags(self, bvid: str) -> list[dict[str, Any]]: ...

    async def get_subtitles(
        self,
        bvid: str,
        *,
        video_info: dict[str, Any] | None = None,
        domain: str = "",
    ) -> list[dict[str, Any]]: ...

    #: Reason the most recent get_subtitles call could not produce trustworthy
    #: segments (``None`` on success); consumed by the channel adapter to emit
    #: a truthful limitation instead of a fixed "absent" text.
    last_subtitle_limitation: str | None

    async def get_audio_bytes(
        self,
        bvid: str,
        max_bytes: int = 50_000_000,
        *,
        video_info: dict[str, Any] | None = None,
    ) -> bytes: ...

    async def get_comment_page(
        self,
        oid: int | str,
        *,
        limit: int = 20,
        mode: int = 3,
        page: int = 1,
        reply_limit: int = 10,
    ) -> BilibiliCommentPage: ...

    async def get_danmaku(
        self,
        bvid: str,
        limit: int = 500,
        *,
        cid: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def get_related_videos(
        self,
        bvid: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]: ...


class BilibiliChannelAdapter:
    """Expose public Bilibili discovery and bounded source hydration."""

    adapter_id = "bilibili"
    adapter_version = "2.1.0"

    def __init__(
        self,
        tool: _BilibiliTool,
        *,
        transcriber: AudioTranscriber | None = None,
        asr_max_duration_seconds: int = 3600,
        asr_max_audio_bytes: int = 50_000_000,
        now_factory: Callable[[], datetime] | None = None,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._tool = tool
        self._transcriber = transcriber
        self._asr_max_duration_seconds = max(60, asr_max_duration_seconds)
        self._asr_max_audio_bytes = max(1_000_000, asr_max_audio_bytes)
        self._now_factory = now_factory or (lambda: datetime.now(UTC))
        self._progress = progress
        self._latest_content_at: datetime | None = None

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        """Declare platform discovery, community, metric, media-text, and surface roles."""
        common: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "languages": ["zh-Hans"],
            "regions": ["CN"],
            "supports_authentication": False,
            "requires_authentication": False,
        }
        return (
            CapabilityDescriptor(
                **common,
                role=ChannelCapabilityRole.PLATFORM_DISCOVERY,
                entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
                query_semantics=QuerySemantics.PLATFORM_SEARCH,
                supports_queryless=False,
                supports_cursor=False,
                supports_metrics=True,
                time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
                bounded_limit=50,
                relative_cost=1.0,
                limitations=[
                    "platform_search_order_is_not_a_cross-source_heat_ranking",
                    "provider_side_time_filter_unavailable",
                ],
            ),
            CapabilityDescriptor(
                **common,
                role=ChannelCapabilityRole.ATTENTION_RANKING,
                entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
                query_semantics=QuerySemantics.PLATFORM_SEARCH,
                supports_queryless=False,
                supports_cursor=False,
                supports_metrics=True,
                time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
                bounded_limit=50,
                relative_cost=1.0,
                limitations=[
                    "ranking_reflects_platform_sort_order_not_global_absolutes",
                    "provider_ranking_algorithm_opaque_and_may_change_without_notice",
                ],
            ),
            CapabilityDescriptor(
                **common,
                role=ChannelCapabilityRole.RECENT_STREAM,
                entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
                query_semantics=QuerySemantics.PLATFORM_SEARCH,
                supports_queryless=False,
                supports_cursor=False,
                supports_metrics=True,
                time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
                bounded_limit=50,
                relative_cost=1.0,
                limitations=[
                    "recent_stream_dependent_on_platform_pubdate_index_quality",
                    "provider_side_time_filter_unavailable",
                    "provider_time_sort_may_exclude_non_video_content",
                ],
            ),
            CapabilityDescriptor(
                **common,
                role=ChannelCapabilityRole.COMMUNITY_STREAM,
                entry_kind=AcquisitionEntryKind.SOURCE_HYDRATION,
                query_semantics=QuerySemantics.UNSUPPORTED,
                supports_queryless=False,
                supports_cursor=True,
                supports_metrics=True,
                bounded_limit=200,
                relative_cost=2.0,
                limitations=["comments_are_platform_ordered_bounded_samples"],
            ),
            CapabilityDescriptor(
                **common,
                role=ChannelCapabilityRole.PUBLIC_METRIC,
                entry_kind=AcquisitionEntryKind.SOURCE_HYDRATION,
                query_semantics=QuerySemantics.UNSUPPORTED,
                supports_queryless=False,
                supports_cursor=False,
                supports_metrics=True,
                bounded_limit=10,
                relative_cost=1.5,
                limitations=["concurrent_audience_precision_may_be_rounded_or_hidden"],
            ),
            CapabilityDescriptor(
                **common,
                role=ChannelCapabilityRole.MEDIA_TEXT,
                entry_kind=AcquisitionEntryKind.SOURCE_HYDRATION,
                query_semantics=QuerySemantics.UNSUPPORTED,
                supports_queryless=False,
                supports_cursor=False,
                supports_metrics=False,
                bounded_limit=5,
                relative_cost=4.0,
                limitations=["subtitle_or_asr_availability_is_content_dependent"],
            ),
        )

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Discover mixed Bilibili occurrences without assuming they are videos."""
        query = request.query.strip()
        if not query:
            return self._discovery_failure(request.id, "bilibili_query_required")
        page = self._bounded_int(request.arguments.get("page"), default=1, low=1, high=100)
        surface_role, order = self._discovery_surface(request)
        result = await self._tool.search(
            query,
            page=page,
            page_size=request.limit,
            order=order,
        )
        captured_at = self._utc_now()
        all_items = [item for item in result.items if str(item.get("id", "")).strip()]
        selected = all_items[: request.limit]
        limitations = [result.error] if result.error else []
        limitations.append("bilibili_provider_time_filter_unsupported")
        limitations.append(f"bilibili_surface_order:{order}")
        if len(all_items) > request.limit:
            limitations.append("platform_returned_more_than_requested_limit")
        occurrences = [
            self._occurrence(
                item,
                captured_at,
                search_rank=index + 1,
                capability_role=surface_role,
                provider_order=order,
            )
            for index, item in enumerate(selected)
        ]
        outside_window_count = (
            sum(
                occurrence.source_published_at is not None
                and not (
                    request.window_start
                    <= occurrence.source_published_at
                    <= request.window_end
                )
                for occurrence in occurrences
            )
            if request.window_start is not None and request.window_end is not None
            else 0
        )
        if outside_window_count:
            limitations.append(
                f"published_outside_requested_window_count:{outside_window_count}"
            )
        return DiscoveryBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=(
                AcquisitionOutcome.PARTIAL
                if result.error and occurrences
                else AcquisitionOutcome.UNAVAILABLE
                if result.error
                else AcquisitionOutcome.SUCCESS
                if occurrences
                else AcquisitionOutcome.EMPTY
            ),
            occurrences=occurrences,
            partial=bool(result.error),
            limitations=limitations,
            physical_call_count=1,
        )

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Hydrate a BV video to the requested bounded evidence depth."""
        bvid = self._bvid(request.source_ref)
        if not bvid:
            return self._hydration_failure(
                request.id,
                "bilibili_content_type_not_video_hydratable",
            )
        if request.depth is HydrationDepth.REACTIONS:
            return await self._hydrate_reactions(request, bvid)
        return await self._hydrate_content(request, bvid)

    async def changes_since(self, request: ScanRequest) -> DiscoveryBatch:
        """Use bounded rediscovery because public search has no stable cursor."""
        batch = await self.discover(request)
        return batch.model_copy(
            update={
                "limitations": list(
                    dict.fromkeys([*batch.limitations, "cursor_not_supported"])
                )
            }
        )

    async def related(
        self,
        *,
        request_id: str,
        source_ref: str,
        limit: int = 10,
    ) -> DiscoveryBatch:
        """Discover the platform's actual related-video recommendations."""
        bvid = self._bvid(source_ref)
        if not bvid:
            return self._discovery_failure(
                request_id,
                "bilibili_related_requires_video",
            )
        safe_limit = min(max(limit, 1), 20)
        rows = await self._tool.get_related_videos(bvid, limit=safe_limit)
        captured_at = self._utc_now()
        occurrences: list[SourceOccurrence] = []
        for index, row in enumerate(rows[:safe_limit], start=1):
            related_bvid = str(row.get("bvid", "")).strip()
            if not related_bvid:
                continue
            occurrences.append(
                self._occurrence(
                    {
                        "id": f"bilibili-{related_bvid}",
                        "canonical_url": (
                            f"https://www.bilibili.com/video/{related_bvid}"
                        ),
                        "content_type": "bilibili_video",
                        "title": str(row.get("title", "")),
                        "engagement": {
                            "play": row.get("play"),
                            "danmaku": row.get("danmaku"),
                            "reply": row.get("reply"),
                        },
                    },
                    captured_at,
                    search_rank=index,
                    capability_role=ChannelCapabilityRole.PLATFORM_DISCOVERY,
                    provider_order="platform_related",
                )
            )
        return DiscoveryBatch(
            request_id=request_id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            occurrences=occurrences,
            partial=not bool(occurrences),
            limitations=[
                "platform_related_recommendations_are_personalization_opaque"
            ],
            physical_call_count=1,
        )

    async def health(self) -> ChannelHealth:
        """Report configured public-read availability with freshness tracking."""
        checked_at = self._utc_now()
        freshness: int | None = None
        status = ChannelHealthStatus.HEALTHY
        detail = "configured; public endpoints are checked per operation"
        if self._latest_content_at is not None:
            freshness = int((checked_at - self._latest_content_at).total_seconds())
            if freshness > 7 * 86400:  # 7 days
                status = ChannelHealthStatus.DEGRADED
                detail = (
                    f"most recent source content is {freshness // 86400}d old; "
                    "platform may return predominantly stale results"
                )
        return ChannelHealth(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            status=status,
            checked_at=checked_at,
            detail=detail,
            data_freshness_seconds=freshness,
        )

    async def _hydrate_content(
        self,
        request: HydrationRequest,
        bvid: str,
    ) -> ObservationBatch:
        self._emit_progress("fetching_video_metadata", bvid=bvid)
        detail = await self._tool.get_video_info(bvid)
        if not detail:
            return self._hydration_failure(
                request.id,
                "bilibili_video_detail_unavailable",
                physical_calls=1,
            )
        tags = await self._tool.get_video_tags(bvid)
        metadata = {**detail, "tags": tags, "tags_hydrated": bool(tags)}
        full = bool(request.arguments.get("full"))
        description = str(detail.get("description", "") or "").strip()
        metadata_body = (
            description
            if full
            and request.depth not in {HydrationDepth.TEXT, HydrationDepth.MEDIA_TEXT}
            and description
            else None
        )
        observations = [
            self._metadata_observation(request.source_ref, metadata, body=metadata_body)
        ]
        limitations: list[str] = []
        if not tags:
            limitations.append("video_tags_empty_or_public_endpoint_unavailable")
        physical_calls = 2
        if request.arguments.get("include_concurrent_audience") is True:
            audience_observations, audience_limitations, audience_calls = (
                await self._observe_concurrent_audience(
                    source_ref=request.source_ref,
                    bvid=bvid,
                    detail=detail,
                    arguments=request.arguments,
                )
            )
            observations.extend(audience_observations)
            limitations.extend(audience_limitations)
            physical_calls += audience_calls
        if request.depth in {HydrationDepth.TEXT, HydrationDepth.MEDIA_TEXT}:
            if description:
                description_body: str | None = None
                description_limitations: list[str] | None = None
                if full:
                    description_body, description_truncated = self._capped_body(
                        description
                    )
                    if description_truncated:
                        description_limitations = ["body_truncated"]
                observations.append(
                    self._observation(
                        source_ref=request.source_ref,
                        modality=ObservationModality.DOCUMENT_TEXT,
                        depth=AccessDepth.CONTENT_TEXT,
                        excerpt=description,
                        location="video_description",
                        method="bilibili_video_description",
                        metadata={"bvid": bvid},
                        limitations=description_limitations,
                        body=description_body,
                    )
                )
            max_segments = self._bounded_int(
                request.arguments.get("max_subtitle_segments"),
                default=500,
                low=1,
                high=500,
            )
            domain = str(request.arguments.get("domain") or "")
            self._emit_progress("checking_subtitles", bvid=bvid)
            segments = await self._tool.get_subtitles(
                bvid, video_info=detail, domain=domain
            )
            physical_calls += 2 if segments else 1
            if not segments:
                platform_limitation = (
                    self._tool.last_subtitle_limitation or SUBTITLE_ABSENT_LIMITATION
                )
                limitations.append(platform_limitation)
                if full and self._transcriber is not None:
                    duration = self._selected_video_duration_seconds(detail)
                    if duration > self._asr_max_duration_seconds:
                        limitations.append(ASR_DURATION_EXCEEDED_LIMITATION)
                    else:
                        try:
                            self._emit_progress("downloading_audio", bvid=bvid)
                            audio = await self._tool.get_audio_bytes(
                                bvid,
                                max_bytes=self._asr_max_audio_bytes,
                                video_info=detail,
                            )
                            physical_calls += 2
                        except Exception:
                            logger.exception("Failed to acquire bounded audio for %s", bvid)
                            audio = b""
                        if not audio:
                            limitations.append(ASR_AUDIO_EMPTY_LIMITATION)
                        else:
                            self._emit_progress("starting_asr", bvid=bvid)
                            result = await self._transcriber.transcribe(audio)
                            limitations.extend(result.warnings)
                            if result.segments:
                                cid = int(detail.get("cid", 0) or 0)
                                segments = [
                                    {**segment.as_dict(), "cid": cid}
                                    for segment in result.segments
                                ]
                            elif result.limitation:
                                limitations.append(result.limitation)
            for segment in segments[:max_segments]:
                text = str(segment.get("content", "") or "").strip()
                if not text:
                    continue
                is_asr = str(segment.get("acquisition_method", "")).startswith(
                    "faster_whisper:"
                )
                observations.append(
                    self._observation(
                        source_ref=request.source_ref,
                        modality=ObservationModality.TRANSCRIPT,
                        depth=AccessDepth.CONTENT_TEXT,
                        excerpt=text,
                        location=(
                            f"{float(segment.get('start', 0.0)):.2f}-"
                            f"{float(segment.get('end', 0.0)):.2f}"
                        ),
                        method=str(
                            segment.get("acquisition_method", "platform_subtitle")
                        ),
                        confidence=float(segment.get("confidence", 1.0)),
                        metadata={
                            "language": segment.get("language", ""),
                            "cid": segment.get("cid"),
                            "subtitle_reliability": segment.get("reliability"),
                        },
                        limitations=(
                            ["automatic_transcript_may_misrecognize_names_or_overlapping_speech"]
                            if is_asr
                            else None
                        ),
                    )
                )
            if full and segments:
                transcript_text, transcript_count = self._aggregate_transcript(
                    segments
                )
                transcript_body, transcript_truncated = self._capped_body(
                    transcript_text
                )
                observations.append(
                    self._observation(
                        source_ref=request.source_ref,
                        modality=ObservationModality.TRANSCRIPT,
                        depth=AccessDepth.CONTENT_TEXT,
                        excerpt=transcript_body,
                        location="video_transcript:full",
                        method=(
                            "bilibili_asr_aggregate"
                            if str(segments[0].get("acquisition_method", "")).startswith(
                                "faster_whisper:"
                            )
                            else "bilibili_subtitle_aggregate"
                        ),
                        sampling_scope=(
                            "full transcript aggregated across subtitle segments;"
                            f"segments={transcript_count}"
                        ),
                        metadata={
                            "bvid": bvid,
                            "segment_count": transcript_count,
                            "body_truncated": transcript_truncated,
                            "subtitle_reliability": segments[0].get(
                                "reliability"
                            ),
                            "transcript_acquisition_method": segments[0].get(
                                "acquisition_method"
                            ),
                        },
                        limitations=(
                            list(
                                dict.fromkeys(
                                    [
                                        *(["body_truncated"] if transcript_truncated else []),
                                        *(
                                            [
                                                "automatic_transcript_may_misrecognize_names_or_overlapping_speech"
                                            ]
                                            if str(
                                                segments[0].get("acquisition_method", "")
                                            ).startswith("faster_whisper:")
                                            else []
                                        ),
                                    ]
                                )
                            )
                            or None
                        ),
                        body=transcript_body,
                    )
                )
            if len(segments) > max_segments:
                limitations.append("subtitle_segments_truncated")
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            observations=observations,
            partial=bool(limitations),
            limitations=limitations,
            physical_call_count=physical_calls,
        )

    async def _observe_concurrent_audience(
        self,
        *,
        source_ref: str,
        bvid: str,
        detail: dict[str, Any],
        arguments: dict[str, Any],
    ) -> tuple[list[SourceObservation], list[str], int]:
        page, selection_error = self._select_video_page(detail, arguments)
        if selection_error:
            unavailable = BilibiliAudienceValue(
                precision=BilibiliAudiencePrecision.UNAVAILABLE
            )
            observations = self._concurrent_audience_observations(
                source_ref=source_ref,
                bvid=bvid,
                page={},
                page_count=self._page_count(detail),
                result=BilibiliConcurrentAudienceResult(
                    bvid=bvid,
                    cid=0,
                    total=unavailable,
                    count=unavailable,
                    limitations=[selection_error],
                    error_kind="invalid_part_selection",
                ),
            )
            return observations, [selection_error], 0
        cid = int(page.get("cid", 0) or 0)
        try:
            result = await self._tool.get_concurrent_audience(bvid, cid)
        except Exception:
            logger.exception(
                "Unexpected Bilibili concurrent-audience adapter failure for %s/%s",
                bvid,
                cid,
            )
            unavailable = BilibiliAudienceValue(
                precision=BilibiliAudiencePrecision.UNAVAILABLE
            )
            result = BilibiliConcurrentAudienceResult(
                bvid=bvid,
                cid=cid,
                total=unavailable,
                count=unavailable,
                limitations=["concurrent_audience_unexpected_failure"],
                error_kind="unexpected_failure",
            )
        observations = self._concurrent_audience_observations(
            source_ref=source_ref,
            bvid=bvid,
            page=page,
            page_count=self._page_count(detail),
            result=result,
        )
        limitations = list(result.limitations)
        if result.error_kind:
            limitations.append(
                f"concurrent_audience_{result.error_kind}"
            )
        return observations, list(dict.fromkeys(limitations)), 1

    def _concurrent_audience_observations(
        self,
        *,
        source_ref: str,
        bvid: str,
        page: dict[str, Any],
        page_count: int,
        result: BilibiliConcurrentAudienceResult,
    ) -> list[SourceObservation]:
        page_number = self._positive_int(page.get("page"))
        cid = self._positive_int(page.get("cid"))
        common_metadata: dict[str, Any] = {
            "temporal_signal_kind": "concurrent_audience",
            "bvid": bvid,
            "selected_cid": cid,
            "selected_page": page_number,
            "page_count": page_count,
            "part": str(page.get("part", "") or ""),
            "part_duration_seconds": self._positive_int(page.get("duration")),
            "visibility_scope": "public_endpoint",
            "error_kind": result.error_kind,
            "error_code": result.error_code,
        }
        scope = (
            "single public concurrent-audience snapshot;"
            f"cid={cid};page={page_number};page_count={page_count}"
        )
        total = self._audience_observation(
            source_ref=source_ref,
            location="concurrent_audience:archive_total",
            value=result.total,
            metadata={
                **common_metadata,
                "metric_name": "concurrent_audience_archive_total",
                "scope": "video_archive_all_parts",
            },
            sampling_scope=scope,
            limitations=[
                "single_snapshot_does_not_establish_growth_or_heat",
                "archive_total_spans_all_video_parts",
                *result.limitations,
            ],
        )
        count = self._audience_observation(
            source_ref=source_ref,
            location="concurrent_audience:selected_part",
            value=result.count,
            metadata={
                **common_metadata,
                "metric_name": "concurrent_audience_selected_part",
                "scope": "selected_cid_only",
            },
            sampling_scope=scope,
            limitations=[
                "single_snapshot_does_not_establish_growth_or_heat",
                "part_count_is_for_selected_cid",
                *result.limitations,
            ],
        )
        return [total, count]

    def _audience_observation(
        self,
        *,
        source_ref: str,
        location: str,
        value: BilibiliAudienceValue,
        metadata: dict[str, Any],
        sampling_scope: str,
        limitations: list[str],
    ) -> SourceObservation:
        precision = value.precision.value
        local_limitations = list(limitations)
        if value.precision is BilibiliAudiencePrecision.ROUNDED:
            local_limitations.append("rounded_value_is_a_lower_bound")
        elif value.precision is BilibiliAudiencePrecision.HIDDEN:
            local_limitations.append("metric_hidden_by_platform_show_switch")
        elif value.precision is BilibiliAudiencePrecision.UNAVAILABLE:
            local_limitations.append("metric_value_unavailable")
        return self._observation(
            source_ref=source_ref,
            modality=ObservationModality.STRUCTURED_DATA,
            depth=AccessDepth.METADATA,
            excerpt=value.display,
            location=location,
            method="bilibili_player_online_total",
            sampling_scope=sampling_scope,
            metadata={
                **metadata,
                "value": value.value,
                "lower_bound": value.lower_bound,
                "display": value.display,
                "precision": precision,
                "show_switch": value.show_switch,
            },
            limitations=list(dict.fromkeys(local_limitations)),
        )

    @classmethod
    def _select_video_page(
        cls,
        detail: dict[str, Any],
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        raw_pages = detail.get("pages", [])
        pages = [item for item in raw_pages if isinstance(item, dict)]
        if not pages:
            cid = cls._positive_int(detail.get("cid"))
            if cid is not None:
                pages = [{"cid": cid, "page": 1, "part": "", "duration": None}]
        if not pages:
            return {}, "concurrent_audience_video_cid_unavailable"

        cid_requested = "cid" in arguments
        requested_cid = cls._positive_int(arguments.get("cid"))
        if cid_requested and requested_cid is None:
            return {}, "concurrent_audience_invalid_cid"
        page_key = "page_number" if "page_number" in arguments else "page"
        page_requested = page_key in arguments
        requested_page = cls._positive_int(arguments.get(page_key))
        if page_requested and requested_page is None:
            return {}, "concurrent_audience_invalid_page_number"

        by_cid = (
            next(
                (
                    item
                    for item in pages
                    if cls._positive_int(item.get("cid")) == requested_cid
                ),
                None,
            )
            if requested_cid is not None
            else None
        )
        if requested_cid is not None and by_cid is None:
            return {}, "concurrent_audience_cid_not_in_video_pages"
        by_page = (
            next(
                (
                    item
                    for item in pages
                    if cls._positive_int(item.get("page")) == requested_page
                ),
                None,
            )
            if requested_page is not None
            else None
        )
        if requested_page is not None and by_page is None:
            return {}, "concurrent_audience_page_not_in_video"
        if by_cid is not None and by_page is not None and by_cid is not by_page:
            return {}, "concurrent_audience_cid_page_mismatch"
        if by_cid is not None:
            return by_cid, ""
        if by_page is not None:
            return by_page, ""
        default_cid = cls._positive_int(detail.get("cid"))
        default_page = next(
            (
                item
                for item in pages
                if cls._positive_int(item.get("cid")) == default_cid
            ),
            pages[0],
        )
        return default_page, ""

    @staticmethod
    def _page_count(detail: dict[str, Any]) -> int:
        pages = detail.get("pages", [])
        return len(pages) if isinstance(pages, list) and pages else 1

    async def _hydrate_reactions(
        self,
        request: HydrationRequest,
        bvid: str,
    ) -> ObservationBatch:
        try:
            sampling = BilibiliCommentSampling.model_validate(
                self._sampling_arguments(request.arguments)
            )
        except ValidationError:
            return self._hydration_failure(
                request.id,
                "invalid_bilibili_comment_sampling",
            )
        detail = await self._tool.get_video_info(bvid)
        if not detail:
            return self._hydration_failure(
                request.id,
                "bilibili_video_detail_unavailable",
                physical_calls=1,
            )
        tags = await self._tool.get_video_tags(bvid)
        hydrated_detail = {**detail, "tags": tags, "tags_hydrated": bool(tags)}
        full = bool(request.arguments.get("full"))
        description = str(detail.get("description", "") or "").strip()
        observations = [
            self._metadata_observation(
                request.source_ref,
                hydrated_detail,
                body=description if full and description else None,
            )
        ]
        limitations = [
            "comments_are_ordered_bounded_samples_not_overall_audience_opinion"
        ]
        if not tags:
            limitations.append("video_tags_empty_or_public_endpoint_unavailable")
        physical_calls = 2
        if request.arguments.get("include_concurrent_audience") is True:
            audience_observations, audience_limitations, audience_calls = (
                await self._observe_concurrent_audience(
                    source_ref=request.source_ref,
                    bvid=bvid,
                    detail=detail,
                    arguments=request.arguments,
                )
            )
            observations.extend(audience_observations)
            limitations.extend(audience_limitations)
            physical_calls += audience_calls
        aid = int(detail.get("aid", 0) or 0)
        remaining = sampling.max_total
        seen: set[str] = set()
        sampled_comment_count = 0
        sampled_body_lines: list[str] = []
        for sort in sampling.sorts:
            if remaining <= 0:
                break
            requested = min(
                sampling.sort_limits.get(sort, sampling.per_sort_limit),
                remaining,
            )
            if requested <= 0:
                continue
            page = await self._tool.get_comment_page(
                aid or bvid,
                limit=requested,
                mode=3 if sort is BilibiliCommentSort.HOT else 2,
                page=sampling.page,
                reply_limit=sampling.reply_limit,
            )
            physical_calls += 1
            limitations.extend(page.limitations)
            if page.error:
                limitations.append(f"{sort.value}_comments_unavailable:{page.error}")
            for comment in page.comments:
                content = str(comment.get("content", "") or "").strip()
                comment_ref = self._stable_id(
                    "bilibili-comment",
                    str(comment.get("user", "")),
                    str(comment.get("ctime", "")),
                    content,
                )
                if content and comment_ref not in seen and remaining > 0:
                    seen.add(comment_ref)
                    remaining -= 1
                    sampled_comment_count += 1
                    sampled_body_lines.append(
                        self._comment_body_line(sort, comment)
                    )
                    observations.append(
                        self._comment_observation(
                            source_ref=request.source_ref,
                            content=content,
                            location=f"comment:{sort.value}:page:{sampling.page}:root",
                            sampling_scope=self._comment_sampling_scope(page),
                            metadata={
                                "comment_ref": comment_ref,
                                "comment_kind": "thread_root",
                                "thread_root_ref": comment_ref,
                                "sort": page.sort,
                                "page": page.page,
                                "total_comments": page.total_comments,
                                "sample_returned": page.returned_count,
                                "sample_limit": page.requested_limit,
                                "has_more": page.has_more,
                                "like_count": comment.get("like_count"),
                                "reply_count": comment.get("reply_count"),
                                "ctime": comment.get("ctime"),
                                "author_ref": self._author_ref(comment.get("user")),
                            },
                        )
                    )
                for child in comment.get("replies", []) or []:
                    if remaining <= 0:
                        break
                    if not isinstance(child, dict):
                        continue
                    child_content = str(child.get("content", "") or "").strip()
                    child_ref = self._stable_id(
                        "bilibili-comment",
                        comment_ref,
                        str(child.get("user", "")),
                        str(child.get("ctime", "")),
                        child_content,
                    )
                    if not child_content or child_ref in seen:
                        continue
                    seen.add(child_ref)
                    remaining -= 1
                    sampled_comment_count += 1
                    sampled_body_lines.append(
                        self._comment_body_line(sort, child)
                    )
                    observations.append(
                        self._comment_observation(
                            source_ref=request.source_ref,
                            content=child_content,
                            location=f"comment:{sort.value}:page:{sampling.page}:reply",
                            sampling_scope=self._comment_sampling_scope(page),
                            metadata={
                                "comment_ref": child_ref,
                                "comment_kind": "thread_reply",
                                "parent_comment_ref": comment_ref,
                                "thread_root_ref": comment_ref,
                                "sort": page.sort,
                                "page": page.page,
                                "total_comments": page.total_comments,
                                "like_count": child.get("like_count"),
                                "ctime": child.get("ctime"),
                                "author_ref": self._author_ref(child.get("user")),
                            },
                        )
                    )
        if full and sampled_body_lines:
            joined_body, joined_truncated = self._capped_body(
                "\n".join(sampled_body_lines)
            )
            observations.append(
                self._observation(
                    source_ref=request.source_ref,
                    modality=ObservationModality.COMMENT,
                    depth=AccessDepth.REACTIONS,
                    excerpt=joined_body,
                    location="comment:full",
                    method="bilibili_comment_aggregate",
                    sampling_scope=(
                        "sampled comment texts joined in sampling order;"
                        f"comments={len(sampled_body_lines)}"
                    ),
                    metadata={
                        "bvid": bvid,
                        "comment_count": len(sampled_body_lines),
                        "body_truncated": joined_truncated,
                    },
                    limitations=["body_truncated"] if joined_truncated else None,
                    body=joined_body,
                )
            )
        if sampled_comment_count < sampling.max_total:
            limitations.append(
                "comment_sample_below_requested_budget:"
                f"{sampled_comment_count}/{sampling.max_total}"
            )
        if sampling.include_danmaku:
            cid = int(detail.get("cid", 0) or 0)
            danmaku = await self._tool.get_danmaku(
                bvid,
                limit=sampling.danmaku_limit,
                cid=cid or None,
            )
            physical_calls += 1 if cid else 2
            if not danmaku:
                limitations.append("danmaku_empty_or_unavailable")
            for item in danmaku:
                content = str(item.get("content", "") or "").strip()
                if not content:
                    continue
                observations.append(
                    self._observation(
                        source_ref=request.source_ref,
                        modality=ObservationModality.DANMAKU,
                        depth=AccessDepth.REACTIONS,
                        excerpt=content,
                        location=(
                            f"video_time:{float(item.get('video_time', 0.0)):.2f}"
                        ),
                        method="bilibili_danmaku_sample",
                        sampling_scope=(
                            "bounded first-page danmaku in platform document order;"
                            f"limit={sampling.danmaku_limit}"
                        ),
                        metadata={
                            "video_time": item.get("video_time"),
                            "mode": item.get("mode"),
                            "cid": item.get("cid"),
                        },
                    )
                )
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            observations=observations,
            partial=bool(limitations),
            limitations=list(dict.fromkeys(limitations)),
            physical_call_count=physical_calls,
        )

    @staticmethod
    def _sampling_arguments(arguments: dict[str, Any]) -> object:
        explicit = arguments.get("comment_sampling")
        if explicit is not None:
            if not isinstance(explicit, dict):
                return {}
            merged = dict(explicit)
        else:
            raw_plan = arguments.get("comment_sampling_plan")
            if not isinstance(raw_plan, dict):
                return {}
            merged = BilibiliChannelAdapter._sampling_from_plan(raw_plan)
        # The request-level page is the pagination cursor the tools layer
        # forwards for page>1 calls (audit A3-1). Without forwarding it,
        # BilibiliCommentSampling.page keeps its default of 1 and every
        # hydrate silently fetches the first comment page while the caller
        # asked for a later one. get_comment_page treats page as a 1-based
        # page number (pn), so the raw value passes straight through,
        # bounded to the model's declared 1..100 range.
        page = arguments.get("page")
        if page is not None and not isinstance(page, bool):
            merged["page"] = BilibiliChannelAdapter._bounded_int(
                page, default=1, low=1, high=100
            )
        return merged

    @staticmethod
    def _sampling_from_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
        """Translate a generic comment sampling plan into bilibili sampling arguments."""
        allocations = raw_plan.get("allocations")
        by_stratum: dict[str, int] = {}
        if isinstance(allocations, list):
            for item in allocations:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("stratum", ""))
                requested = item.get("requested")
                if isinstance(requested, int) and not isinstance(requested, bool):
                    by_stratum[name] = max(0, requested)
        popular = by_stratum.get("popular", 0)
        recent = by_stratum.get("recent", 0)
        roots = by_stratum.get("thread_root", 0)
        replies = by_stratum.get("thread_reply", 0)
        target_raw = raw_plan.get("target_sample_count", 0)
        target = (
            max(0, min(200, target_raw))
            if isinstance(target_raw, int) and not isinstance(target_raw, bool)
            else 0
        )
        hot_limit = min(50, popular + roots)
        recent_limit = min(50, recent)
        root_budget = max(1, hot_limit + recent_limit)
        reply_limit = min(10, math.ceil(replies / root_budget))
        sorts = []
        if hot_limit:
            sorts.append(BilibiliCommentSort.HOT)
        if recent_limit:
            sorts.append(BilibiliCommentSort.NEWEST)
        if not sorts:
            sorts = [BilibiliCommentSort.HOT]
        return {
            "sorts": sorts,
            "per_sort_limit": max(1, hot_limit, recent_limit),
            "sort_limits": {
                BilibiliCommentSort.HOT: hot_limit,
                BilibiliCommentSort.NEWEST: recent_limit,
            },
            "max_total": target,
            "reply_limit": reply_limit,
        }

    def _comment_observation(
        self,
        *,
        source_ref: str,
        content: str,
        location: str,
        sampling_scope: str,
        metadata: dict[str, Any],
    ) -> SourceObservation:
        return self._observation(
            source_ref=source_ref,
            modality=ObservationModality.COMMENT,
            depth=AccessDepth.REACTIONS,
            excerpt=content,
            location=location,
            method="bilibili_comment_sample",
            sampling_scope=sampling_scope,
            metadata=metadata,
        )

    @staticmethod
    def _comment_body_line(
        sort: BilibiliCommentSort,
        comment: dict[str, Any],
    ) -> str:
        """Render one sampled comment as a readable aggregated-body line."""
        content = str(comment.get("content", "") or "").strip()
        likes = comment.get("like_count")
        like_display = likes if isinstance(likes, int) else 0
        return f"[{sort.value}] likes={like_display} {content}"

    @staticmethod
    def _aggregate_transcript(
        segments: list[dict[str, Any]],
    ) -> tuple[str, int]:
        """Join all subtitle segments into timestamped lines in time order."""
        ordered = sorted(segments, key=BilibiliChannelAdapter._segment_start)
        lines: list[str] = []
        for segment in ordered:
            text = str(segment.get("content", "") or "").strip()
            if not text:
                continue
            lines.append(
                f"{BilibiliChannelAdapter._subtitle_timestamp(segment)} {text}"
            )
        return "\n".join(lines), len(lines)

    @staticmethod
    def _selected_video_duration_seconds(detail: dict[str, Any]) -> int:
        """Return the duration of the same video page used for subtitle/audio fetches."""
        selected_cid = int(detail.get("cid", 0) or 0)
        pages = detail.get("pages", [])
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                if selected_cid and int(page.get("cid", 0) or 0) != selected_cid:
                    continue
                return max(0, int(page.get("duration", 0) or 0))
        return max(0, int(detail.get("duration", 0) or 0))

    def _emit_progress(self, stage: str, **details: object) -> None:
        """Publish one best-effort Bilibili hydration stage."""
        if self._progress is None:
            return
        try:
            self._progress({"component": "bilibili", "stage": stage, **details})
        except Exception:  # noqa: BLE001 - progress must never break hydration
            logger.debug("Bilibili progress sink failed", exc_info=True)

    @staticmethod
    def _segment_start(segment: dict[str, Any]) -> float:
        """Return the segment start offset in seconds, defaulting to zero."""
        try:
            return float(segment.get("start", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _subtitle_timestamp(segment: dict[str, Any]) -> str:
        """Format a segment start offset as a zero-padded mm:ss label."""
        total = max(0, int(BilibiliChannelAdapter._segment_start(segment)))
        return f"{total // 60:02d}:{total % 60:02d}"

    @staticmethod
    def _capped_body(text: str) -> tuple[str, bool]:
        """Cap a full-depth body at the shared limit, reporting truncation."""
        if len(text) <= _FULL_BODY_CAP:
            return text, False
        return text[:_FULL_BODY_CAP], True

    @staticmethod
    def _comment_sampling_scope(page: BilibiliCommentPage) -> str:
        return (
            f"sort={page.sort};page={page.page};"
            f"requested={page.requested_limit};"
            f"returned={page.returned_count};"
            f"total={page.total_comments};"
            f"reply_limit={page.reply_limit}"
        )

    @staticmethod
    def _author_ref(value: object) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            return ""
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
        return f"bilibili-author-{digest}"

    def _occurrence(
        self,
        item: dict[str, Any],
        captured_at: datetime,
        *,
        search_rank: int,
        capability_role: ChannelCapabilityRole = ChannelCapabilityRole.PLATFORM_DISCOVERY,
        provider_order: str = "totalrank",
    ) -> SourceOccurrence:
        source_ref = str(item.get("id", ""))
        published_at = self._parse_datetime(item.get("published_at"))
        if published_at is not None and (
            self._latest_content_at is None or published_at > self._latest_content_at
        ):
            self._latest_content_at = published_at
        content_type = str(item.get("content_type", "bilibili_unknown"))[:100]
        field_limitations = [
            "platform_search_result_is_discovery_only",
            "provider_side_time_filter_unavailable",
        ]
        if content_type != "bilibili_video":
            # Collections (media_ft), live rooms, and articles carry no BV id:
            # no depth can hydrate them, so say so at discovery time instead of
            # letting the model discover a card it can never open.
            field_limitations.append("bilibili_content_type_not_video_hydratable")
        metadata = {
            "author_mid": item.get("author_mid"),
            "engagement": self._normalized_engagement(item.get("engagement")),
            "tags": item.get("tags", []),
            "tags_hydrated": bool(item.get("tags_hydrated", False)),
            "text_snippet": item.get("text_snippet", ""),
            "search_rank": search_rank,
            "provider": "bilibili_public_search",
            "provider_order": provider_order,
            "capability_role": capability_role.value,
            "independence_status": "unknown",
            "field_limitations": field_limitations,
        }
        return SourceOccurrence(
            id=self._stable_id("occurrence", source_ref),
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            source_ref=source_ref,
            canonical_url=str(item.get("canonical_url", "")),
            title=str(item.get("title", ""))[:2000],
            author=str(item.get("author", ""))[:500],
            content_type=content_type,
            language="zh",
            source_published_at=published_at,
            captured_at=captured_at,
            metadata=metadata,
        )

    @staticmethod
    def _discovery_surface(
        request: ScanRequest,
    ) -> tuple[ChannelCapabilityRole, str]:
        """Translate platform-neutral surface roles to truthful provider ordering."""
        requested_mode = str(request.arguments.get("discovery_mode", "")).strip()
        roles = set(request.capability_roles)
        if (
            requested_mode == "ranking"
            or ChannelCapabilityRole.ATTENTION_RANKING in roles
        ):
            return ChannelCapabilityRole.ATTENTION_RANKING, "click"
        if requested_mode == "recent" or ChannelCapabilityRole.RECENT_STREAM in roles:
            return ChannelCapabilityRole.RECENT_STREAM, "pubdate"
        return ChannelCapabilityRole.PLATFORM_DISCOVERY, "totalrank"

    @staticmethod
    def _normalized_engagement(value: object) -> dict[str, object]:
        """Map platform counters to the channel-wide engagement vocabulary."""
        raw = value if isinstance(value, dict) else {}
        return {
            "views": raw.get("play"),
            "likes": raw.get("like"),
            "comments": raw.get("reply"),
            "realtime_reactions": raw.get("danmaku"),
            "saves": raw.get("favorite"),
            "supports": raw.get("coin"),
            "shares": raw.get("share"),
        }

    def _metadata_observation(
        self,
        source_ref: str,
        metadata: dict[str, Any],
        *,
        body: str | None = None,
    ) -> SourceObservation:
        local_limitations = [
            "platform metadata does not establish claims made inside the video"
        ]
        if body is not None:
            body, body_truncated = self._capped_body(body)
            if body_truncated:
                local_limitations.append("body_truncated")
        return self._observation(
            source_ref=source_ref,
            modality=ObservationModality.METADATA,
            depth=AccessDepth.METADATA,
            excerpt=str(metadata.get("title", "")),
            location="video_metadata",
            method="bilibili_video_info",
            metadata=metadata,
            limitations=local_limitations,
            body=body,
        )

    def _observation(
        self,
        *,
        source_ref: str,
        modality: ObservationModality,
        depth: AccessDepth,
        excerpt: str,
        location: str,
        method: str,
        confidence: float = 1.0,
        sampling_scope: str = "",
        metadata: dict[str, Any] | None = None,
        limitations: list[str] | None = None,
        body: str | None = None,
    ) -> SourceObservation:
        # The observation id is a pure function of the source reference so a
        # discovery card and a hydrated row collide on one durable id (task
        # T13): modality, location, and excerpt describe a read, not what was
        # read, and travel in the observation metadata instead.
        observation_id = self._stable_id("observation", source_ref)
        return SourceObservation(
            id=observation_id,
            source_ref=source_ref,
            modality=modality,
            access_depth=depth,
            excerpt=excerpt[:4000],
            location=location,
            acquisition_method=method,
            captured_at=self._utc_now(),
            confidence=confidence,
            sampling_scope=sampling_scope,
            metadata=metadata or {},
            limitations=limitations or [],
            body=body,
        )

    def _discovery_failure(self, request_id: str, limitation: str) -> DiscoveryBatch:
        return DiscoveryBatch(
            request_id=request_id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=AcquisitionOutcome.UNSUPPORTED,
            partial=True,
            limitations=[limitation],
        )

    def _hydration_failure(
        self,
        request_id: str,
        limitation: str,
        *,
        physical_calls: int = 0,
    ) -> ObservationBatch:
        return ObservationBatch(
            request_id=request_id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=(
                AcquisitionOutcome.UNSUPPORTED
                if physical_calls == 0
                else AcquisitionOutcome.UNAVAILABLE
            ),
            partial=True,
            limitations=[limitation],
            physical_call_count=physical_calls,
        )

    @staticmethod
    def _bvid(source_ref: str) -> str:
        match = _BVID_PATTERN.search(source_ref)
        return match.group(1) if match else ""

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    @staticmethod
    def _bounded_int(value: object, *, default: int, low: int, high: int) -> int:
        if isinstance(value, bool):
            return default
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            return default
        return max(low, min(parsed, high))

    @staticmethod
    def _positive_int(value: object) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _stable_id(kind: str, *parts: str) -> str:
        payload = "\0".join(parts).encode("utf-8")
        return f"{kind}-{hashlib.sha256(payload).hexdigest()[:32]}"

    def _utc_now(self) -> datetime:
        value = self._now_factory()
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "BilibiliChannelAdapter",
    "BilibiliCommentSampling",
    "BilibiliCommentSort",
]
