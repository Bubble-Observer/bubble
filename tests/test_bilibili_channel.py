"""Tests for the typed Bilibili vNext Channel Adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from leave_information_bubble.channels import (
    AcquisitionOutcome,
    BilibiliChannelAdapter,
    ChannelCapabilityRole,
    HydrationDepth,
    HydrationRequest,
    ScanRequest,
)
from leave_information_bubble.models.epistemics import ObservationModality
from leave_information_bubble.tools.bilibili_search import (
    SUBTITLE_ABSENT_LIMITATION,
    SUBTITLE_UNRELIABLE_LIMITATION,
    BilibiliAudiencePrecision,
    BilibiliAudienceValue,
    BilibiliCommentPage,
    BilibiliConcurrentAudienceResult,
    BilibiliSearchResult,
)
from leave_information_bubble.tools.transcription import (
    ASR_DURATION_EXCEEDED_LIMITATION,
    ASR_GPU_FALLBACK_LIMITATION,
    TranscriptionResult,
    TranscriptSegment,
)
from leave_information_bubble.world import (
    CognitiveDelta,
    ObservationDepth,
    ObservationInput,
    WorldStore,
    WorldTools,
)
from leave_information_bubble.world.store import observation_id

NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)


class FakeBilibiliTool:
    """Deterministic public-endpoint substitute."""

    def __init__(self) -> None:
        self.detail_calls = 0
        self.comment_calls: list[tuple[int | str, int, int, int]] = []
        self.comment_pages: list[int] = []
        self.audience_calls: list[tuple[str, int]] = []
        self.audience_result: BilibiliConcurrentAudienceResult | None = None
        self.raise_audience = False
        self.search_orders: list[str] = []
        self.subtitle_segments: list[dict[str, Any]] | None = None
        self.subtitle_domains: list[str] = []
        self.last_subtitle_limitation: str | None = None
        self.audio_bytes = b"encoded-audio"
        self.audio_calls: list[tuple[str, int, dict[str, Any] | None]] = []

    async def search(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 20,
        *,
        order: str = "totalrank",
    ) -> BilibiliSearchResult:
        del page, page_size
        self.search_orders.append(order)
        return BilibiliSearchResult(
            query=keyword,
            items=[
                {
                    "id": "bilibili-BV001",
                    "canonical_url": "https://www.bilibili.com/video/BV001",
                    "content_type": "bilibili_video",
                    "title": "current match review",
                    "author": "analyst",
                    "author_mid": 42,
                    "published_at": NOW.isoformat(),
                    "engagement": {
                        "play": 1200,
                        "like": 70,
                        "reply": 33,
                        "danmaku": 80,
                        "favorite": None,
                        "coin": None,
                    },
                    "tags": ["game"],
                    "tags_hydrated": False,
                    "text_snippet": "review",
                },
                {
                    "id": "bilibili-live_room-88",
                    "canonical_url": "https://live.bilibili.com/88",
                    "content_type": "bilibili_live_room",
                    "title": "live event",
                    "author": "caster",
                    "author_mid": None,
                    "published_at": None,
                    "engagement": {"play": 8000},
                    "tags": [],
                    "tags_hydrated": False,
                },
                {
                    "id": "bilibili-article-cv1",
                    "canonical_url": "https://www.bilibili.com/read/cv1",
                    "content_type": "bilibili_article",
                    "title": "article",
                    "author": "writer",
                    "published_at": None,
                    "engagement": {},
                    "tags": [],
                    "tags_hydrated": False,
                },
            ],
        )

    async def get_video_info(self, bvid: str) -> dict[str, Any]:
        assert bvid.upper() == "BV001"
        self.detail_calls += 1
        return {
            "aid": 101,
            "bvid": "BV001",
            "cid": 202,
            "title": "current match review",
            "author": "analyst",
            "owner_mid": 42,
            "description": "a bounded description",
            "pages": [
                {"cid": 202, "page": 1, "part": "first", "duration": 100},
                {"cid": 303, "page": 2, "part": "second", "duration": 120},
            ],
            "published_at": NOW.isoformat(),
            "stats": {
                "play": 1200,
                "like": 70,
                "reply": 33,
                "danmaku": 80,
                "favorite": 12,
                "coin": 9,
            },
        }

    async def get_related_videos(
        self, bvid: str, *, limit: int = 20,
    ) -> list[dict[str, Any]]:
        assert bvid.upper() == "BV001"
        return [
            {
                "bvid": "BV002",
                "title": "platform related result",
                "play": 900,
                "danmaku": 12,
                "reply": 8,
            }
        ][:limit]

    async def get_concurrent_audience(
        self,
        bvid: str,
        cid: int,
    ) -> BilibiliConcurrentAudienceResult:
        self.audience_calls.append((bvid, cid))
        if self.raise_audience:
            raise RuntimeError("simulated endpoint drift")
        if self.audience_result is not None:
            return self.audience_result
        return BilibiliConcurrentAudienceResult(
            bvid=bvid,
            cid=cid,
            total=BilibiliAudienceValue(
                precision=BilibiliAudiencePrecision.ROUNDED,
                lower_bound=6000,
                display="6000+",
                show_switch=True,
            ),
            count=BilibiliAudienceValue(
                precision=BilibiliAudiencePrecision.EXACT,
                value=3,
                display="3",
                show_switch=True,
            ),
            limitations=["total_rounded_lower_bound"],
        )

    async def get_video_tags(self, bvid: str) -> list[dict[str, Any]]:
        assert bvid.upper() == "BV001"
        return [{"tag_id": 1, "name": "game", "type": "topic", "music_id": ""}]

    async def get_subtitles(
        self,
        bvid: str,
        *,
        video_info: dict[str, Any] | None = None,
        domain: str = "",
    ) -> list[dict[str, Any]]:
        assert bvid.upper() == "BV001"
        assert video_info is not None
        self.subtitle_domains.append(domain)
        if self.subtitle_segments is not None:
            # Explicitly configured results (including []) are returned
            # verbatim; tests set ``last_subtitle_limitation`` themselves to
            # drive the adapter's limitation consumption.
            return self.subtitle_segments
        self.last_subtitle_limitation = None
        return [
            {
                "start": 1.0,
                "end": 2.0,
                "content": "spoken content",
                "language": "zh-CN",
                "acquisition_method": "platform_subtitle",
                "confidence": 1.0,
                "cid": 202,
                "reliability": "confirmed",
                "attempts": 1,
            }
        ]

    async def get_audio_bytes(
        self,
        bvid: str,
        max_bytes: int = 50_000_000,
        *,
        video_info: dict[str, Any] | None = None,
    ) -> bytes:
        self.audio_calls.append((bvid, max_bytes, video_info))
        return self.audio_bytes

    async def get_comment_page(
        self,
        oid: int | str,
        *,
        limit: int = 20,
        mode: int = 3,
        page: int = 1,
        reply_limit: int = 10,
    ) -> BilibiliCommentPage:
        self.comment_calls.append((oid, limit, mode, reply_limit))
        self.comment_pages.append(page)
        sort = "hot" if mode == 3 else "newest"
        comments = [
            {
                "user": sort,
                "content": f"{sort} comment",
                "like_count": 3,
                "reply_count": 2,
                "ctime": mode,
                "replies": [
                    {
                        "user": f"{sort}-child",
                        "content": f"{sort} child",
                        "like_count": 1,
                        "ctime": mode + 10,
                    }
                ][:reply_limit],
            }
        ]
        return BilibiliCommentPage(
            comments=comments,
            total_comments=330,
            sort=sort,
            requested_limit=limit,
            returned_count=len(comments),
            reply_limit=reply_limit,
            page=page,
            has_more=True,
            limitations=["platform_ordered_first_level_comment_sample"],
        )

    async def get_danmaku(
        self,
        bvid: str,
        limit: int = 500,
        *,
        cid: int | None = None,
    ) -> list[dict[str, Any]]:
        assert bvid.upper() == "BV001"
        assert cid == 202
        assert limit <= 2000
        return [{"content": "timed reaction", "video_time": 3.5, "mode": 1, "cid": cid}]


class FakeTranscriber:
    """Deterministic ASR substitute that never loads a model."""

    def __init__(self, result: TranscriptionResult | None = None) -> None:
        self.calls: list[bytes] = []
        self.result = result or TranscriptionResult(
            segments=(
                TranscriptSegment(
                    start=3.0,
                    end=5.0,
                    content="locally transcribed speech",
                    language="zh",
                    confidence=0.8,
                    acquisition_method="faster_whisper:small:cuda",
                ),
            )
        )

    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        self.calls.append(audio)
        return self.result


def test_bilibili_declares_platform_neutral_capabilities() -> None:
    adapter = BilibiliChannelAdapter(FakeBilibiliTool(), now_factory=lambda: NOW)

    descriptors = adapter.capability_descriptors

    assert {item.role for item in descriptors} == {
        ChannelCapabilityRole.PLATFORM_DISCOVERY,
        ChannelCapabilityRole.ATTENTION_RANKING,
        ChannelCapabilityRole.RECENT_STREAM,
        ChannelCapabilityRole.COMMUNITY_STREAM,
        ChannelCapabilityRole.PUBLIC_METRIC,
        ChannelCapabilityRole.MEDIA_TEXT,
    }
    metric = next(
        item for item in descriptors if item.role is ChannelCapabilityRole.PUBLIC_METRIC
    )
    assert metric.supports_metrics is True
    assert metric.languages == ["zh-Hans"]


async def test_related_uses_platform_recommendation_endpoint() -> None:
    adapter = BilibiliChannelAdapter(FakeBilibiliTool(), now_factory=lambda: NOW)

    batch = await adapter.related(
        request_id="related-1",
        source_ref="BV001",
        limit=5,
    )

    assert batch.physical_call_count == 1
    assert len(batch.occurrences) == 1
    assert batch.occurrences[0].source_ref == "bilibili-BV002"
    assert batch.occurrences[0].title == "platform related result"
    assert batch.occurrences[0].metadata["provider_order"] == "platform_related"


async def test_queryless_bilibili_discovery_reports_unsupported() -> None:
    """A query-required surface is not an ordinary empty success."""
    adapter = BilibiliChannelAdapter(FakeBilibiliTool(), now_factory=lambda: NOW)

    batch = await adapter.discover(
        ScanRequest(id="queryless", lane="ambient", query="")
    )

    assert batch.outcome is AcquisitionOutcome.UNSUPPORTED
    assert batch.occurrences == []
    assert batch.partial is True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("", AcquisitionOutcome.EMPTY),
        ("bilibili_search_unavailable", AcquisitionOutcome.UNAVAILABLE),
    ],
)
async def test_bilibili_discovery_distinguishes_empty_from_unavailable(
    error: str,
    expected: AcquisitionOutcome,
) -> None:
    """A completed empty query and a provider failure are different facts."""

    class EmptyTool(FakeBilibiliTool):
        async def search(
            self,
            keyword: str,
            page: int = 1,
            page_size: int = 20,
            *,
            order: str = "totalrank",
        ) -> BilibiliSearchResult:
            del page, page_size, order
            return BilibiliSearchResult(query=keyword, items=[], error=error)

    adapter = BilibiliChannelAdapter(EmptyTool(), now_factory=lambda: NOW)

    batch = await adapter.discover(_scan())

    assert batch.outcome is expected


def _scan(*, limit: int = 20) -> ScanRequest:
    return ScanRequest(
        id="scan-1",
        lane="emerging",
        query="open domain seed",
        window_start=NOW - timedelta(days=1),
        window_end=NOW,
        limit=limit,
    )


async def test_discovery_keeps_mixed_types_metrics_and_missing_values() -> None:
    adapter = BilibiliChannelAdapter(FakeBilibiliTool(), now_factory=lambda: NOW)

    batch = await adapter.discover(_scan())

    assert [item.content_type for item in batch.occurrences] == [
        "bilibili_video",
        "bilibili_live_room",
        "bilibili_article",
    ]
    video = batch.occurrences[0]
    assert video.source_published_at == NOW
    assert video.metadata["engagement"]["comments"] == 33
    assert video.metadata["engagement"]["saves"] is None
    assert video.metadata["engagement"]["views"] == 1200
    assert video.metadata["author_mid"] == 42
    assert video.metadata["tags"] == ["game"]
    assert batch.physical_call_count == 1


async def test_discovery_marks_non_video_cards_as_not_hydratable() -> None:
    """media_ft/live/article cards carry the not-hydratable marker (audit fix 3)."""
    tool = FakeBilibiliTool()

    class MarkedTool(FakeBilibiliTool):
        async def search(
            self,
            keyword: str,
            page: int = 1,
            page_size: int = 20,
            *,
            order: str = "totalrank",
        ) -> BilibiliSearchResult:
            del page, page_size, order
            return BilibiliSearchResult(
                query=keyword,
                items=[
                    {
                        "id": "bilibili-media_ft-42097",
                        "canonical_url": "https://www.bilibili.com/medialist/42097",
                        "content_type": "bilibili_media_ft",
                        "title": "英雄联盟手游赛事2021年度纪录片",
                        "author": "up",
                        "author_mid": None,
                        "published_at": None,
                        "engagement": {},
                        "tags": [],
                        "tags_hydrated": False,
                    },
                    {
                        "id": "bilibili-BV001",
                        "canonical_url": "https://www.bilibili.com/video/BV001",
                        "content_type": "bilibili_video",
                        "title": "current match review",
                        "author": "analyst",
                        "author_mid": 42,
                        "published_at": NOW.isoformat(),
                        "engagement": {},
                        "tags": ["game"],
                        "tags_hydrated": False,
                    },
                ],
            )

    adapter = BilibiliChannelAdapter(MarkedTool(), now_factory=lambda: NOW)
    batch = await adapter.discover(_scan())

    collection, video = batch.occurrences
    assert "bilibili_content_type_not_video_hydratable" in (
        collection.metadata["field_limitations"]
    )
    assert "bilibili_content_type_not_video_hydratable" not in (
        video.metadata["field_limitations"]
    )
    # the marker must actually reject hydrate for the collection card
    refused = await adapter.hydrate(HydrationRequest(
        id="hydrate-collection",
        source_ref=collection.source_ref,
        depth=HydrationDepth.METADATA,
    ))
    assert refused.limitations == ["bilibili_content_type_not_video_hydratable"]
    assert tool.detail_calls == 0


async def test_discovery_enforces_requested_limit_even_if_platform_ignores_it() -> None:
    adapter = BilibiliChannelAdapter(FakeBilibiliTool(), now_factory=lambda: NOW)

    batch = await adapter.discover(_scan(limit=1))

    assert len(batch.occurrences) == 1
    assert "platform_returned_more_than_requested_limit" in batch.limitations


async def test_discovery_maps_surface_roles_to_provider_native_orders() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    ranking = _scan().model_copy(
        update={"capability_roles": [ChannelCapabilityRole.ATTENTION_RANKING]}
    )
    recent = _scan().model_copy(
        update={"capability_roles": [ChannelCapabilityRole.RECENT_STREAM]}
    )
    ranking_batch = await adapter.discover(ranking)
    recent_batch = await adapter.discover(recent)

    assert tool.search_orders[-2:] == ["click", "pubdate"]
    assert ranking_batch.occurrences[0].metadata["capability_role"] == "attention_ranking"
    assert recent_batch.occurrences[0].metadata["capability_role"] == "recent_stream"


async def test_facade_recent_surface_uses_bilibili_pubdate_order(tmp_path: Path) -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)
    world_tools = WorldTools(
        store=WorldStore(tmp_path / "world.sqlite3"),
        adapters={"bilibili": adapter},
    )

    result = await world_tools.execute(
        "discover_sources",
        {"adapter": "bilibili", "query": "open domain seed", "surface_role": "recent_stream"},
        "facade-recent",
    )

    assert result["ok"] is True
    assert tool.search_orders == ["pubdate"]


async def test_non_video_occurrence_is_not_misrouted_to_video_detail() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-live",
            source_ref="bilibili-live_room-88",
            depth=HydrationDepth.METADATA,
        )
    )

    assert batch.partial is True
    assert batch.observations == []
    assert batch.limitations == ["bilibili_content_type_not_video_hydratable"]
    assert tool.detail_calls == 0


async def test_metadata_and_text_hydration_include_tags_metrics_and_subtitle() -> None:
    adapter = BilibiliChannelAdapter(FakeBilibiliTool(), now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-text",
            source_ref="https://www.bilibili.com/video/BV001",
            depth=HydrationDepth.TEXT,
        )
    )

    modalities = [item.modality for item in batch.observations]
    assert modalities == [
        ObservationModality.METADATA,
        ObservationModality.DOCUMENT_TEXT,
        ObservationModality.TRANSCRIPT,
    ]
    metadata = batch.observations[0].metadata
    assert metadata["stats"]["coin"] == 9
    assert metadata["tags"][0]["name"] == "game"
    assert metadata["tags_hydrated"] is True
    assert batch.physical_call_count == 4


async def test_non_full_subtitle_failure_never_triggers_local_asr() -> None:
    tool = FakeBilibiliTool()
    tool.subtitle_segments = []
    tool.last_subtitle_limitation = SUBTITLE_ABSENT_LIMITATION
    transcriber = FakeTranscriber()
    adapter = BilibiliChannelAdapter(
        tool,
        transcriber=transcriber,
        now_factory=lambda: NOW,
    )

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-preview-no-asr",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
        )
    )

    assert transcriber.calls == []
    assert tool.audio_calls == []
    assert SUBTITLE_ABSENT_LIMITATION in batch.limitations


async def test_full_falls_back_to_local_asr_and_builds_transcript_body() -> None:
    tool = FakeBilibiliTool()
    tool.subtitle_segments = []
    tool.last_subtitle_limitation = SUBTITLE_UNRELIABLE_LIMITATION
    transcriber = FakeTranscriber()
    adapter = BilibiliChannelAdapter(
        tool,
        transcriber=transcriber,
        now_factory=lambda: NOW,
    )

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-full-asr",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    assert transcriber.calls == [b"encoded-audio"]
    assert len(tool.audio_calls) == 1
    assert tool.audio_calls[0][2] is not None
    transcript = next(
        item
        for item in batch.observations
        if item.location == "video_transcript:full"
    )
    assert transcript.acquisition_method == "bilibili_asr_aggregate"
    assert "locally transcribed speech" in str(transcript.body)
    assert transcript.metadata["transcript_acquisition_method"] == (
        "faster_whisper:small:cuda"
    )
    assert SUBTITLE_UNRELIABLE_LIMITATION in batch.limitations


async def test_full_surfaces_successful_asr_cpu_fallback() -> None:
    tool = FakeBilibiliTool()
    tool.subtitle_segments = []
    transcriber = FakeTranscriber(
        TranscriptionResult(
            segments=(
                TranscriptSegment(
                    start=0.0,
                    end=1.0,
                    content="CPU transcript",
                    language="zh",
                    confidence=0.8,
                    acquisition_method="faster_whisper:small:cpu",
                ),
            ),
            warnings=(ASR_GPU_FALLBACK_LIMITATION,),
        )
    )
    adapter = BilibiliChannelAdapter(
        tool,
        transcriber=transcriber,
        now_factory=lambda: NOW,
    )

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-full-asr-cpu-fallback",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    assert ASR_GPU_FALLBACK_LIMITATION in batch.limitations
    transcript = next(
        item for item in batch.observations if item.location == "video_transcript:full"
    )
    assert transcript.metadata["transcript_acquisition_method"] == (
        "faster_whisper:small:cpu"
    )


async def test_full_uses_platform_subtitle_without_loading_asr() -> None:
    tool = FakeBilibiliTool()
    transcriber = FakeTranscriber()
    adapter = BilibiliChannelAdapter(
        tool,
        transcriber=transcriber,
        now_factory=lambda: NOW,
    )

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-full-platform",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    assert transcriber.calls == []
    assert tool.audio_calls == []
    transcript = next(
        item
        for item in batch.observations
        if item.location == "video_transcript:full"
    )
    assert transcript.acquisition_method == "bilibili_subtitle_aggregate"


async def test_full_rejects_asr_before_audio_download_when_duration_is_too_long() -> None:
    tool = FakeBilibiliTool()
    tool.subtitle_segments = []
    tool.last_subtitle_limitation = SUBTITLE_ABSENT_LIMITATION
    transcriber = FakeTranscriber()
    adapter = BilibiliChannelAdapter(
        tool,
        transcriber=transcriber,
        asr_max_duration_seconds=60,
        now_factory=lambda: NOW,
    )

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-full-too-long",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    assert tool.audio_calls == []
    assert transcriber.calls == []
    assert ASR_DURATION_EXCEEDED_LIMITATION in batch.limitations


async def test_optional_concurrent_audience_preserves_part_scope_and_precision() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-audience",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.STRUCTURED,
            arguments={
                "include_concurrent_audience": True,
                "page_number": 2,
            },
        )
    )

    audience = [
        item
        for item in batch.observations
        if item.metadata.get("temporal_signal_kind") == "concurrent_audience"
    ]
    assert len(audience) == 2
    assert tool.audience_calls == [("BV001", 303)]
    archive = next(
        item
        for item in audience
        if item.metadata["scope"] == "video_archive_all_parts"
    )
    selected = next(
        item for item in audience if item.metadata["scope"] == "selected_cid_only"
    )
    assert archive.metadata["precision"] == "rounded"
    assert archive.metadata["value"] is None
    assert archive.metadata["lower_bound"] == 6000
    assert archive.metadata["selected_page"] == 2
    assert archive.metadata["page_count"] == 2
    assert archive.metadata["part"] == "second"
    assert selected.metadata["precision"] == "exact"
    assert selected.metadata["value"] == 3
    assert "single_snapshot_does_not_establish_growth_or_heat" in selected.limitations
    assert batch.physical_call_count == 3


async def test_invalid_concurrent_part_is_explicit_without_endpoint_call() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-invalid-part",
            source_ref="BV001",
            depth=HydrationDepth.METADATA,
            arguments={"include_concurrent_audience": True, "cid": 999},
        )
    )

    audience = [
        item
        for item in batch.observations
        if item.metadata.get("temporal_signal_kind") == "concurrent_audience"
    ]
    assert len(audience) == 2
    assert all(item.metadata["precision"] == "unavailable" for item in audience)
    assert all(item.metadata["value"] is None for item in audience)
    assert tool.audience_calls == []
    assert "concurrent_audience_cid_not_in_video_pages" in batch.limitations
    assert batch.physical_call_count == 2


async def test_concurrent_endpoint_failure_does_not_discard_video_metadata() -> None:
    tool = FakeBilibiliTool()
    tool.raise_audience = True
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-audience-failure",
            source_ref="BV001",
            depth=HydrationDepth.METADATA,
            arguments={"include_concurrent_audience": True},
        )
    )

    assert batch.observations[0].modality is ObservationModality.METADATA
    audience = [
        item
        for item in batch.observations
        if item.metadata.get("temporal_signal_kind") == "concurrent_audience"
    ]
    assert len(audience) == 2
    assert all(item.metadata["precision"] == "unavailable" for item in audience)
    assert "concurrent_audience_unexpected_failure" in batch.limitations
    assert batch.physical_call_count == 3


async def test_reaction_hydration_can_also_observe_concurrent_audience() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-reactions-audience",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={
                "include_concurrent_audience": True,
                "comment_sampling": {
                    "sorts": ["hot"],
                    "per_sort_limit": 1,
                    "max_total": 0,
                    "reply_limit": 0,
                    "include_danmaku": False,
                },
            },
        )
    )

    audience = [
        item
        for item in batch.observations
        if item.metadata.get("temporal_signal_kind") == "concurrent_audience"
    ]
    assert len(audience) == 2
    assert tool.audience_calls == [("BV001", 202)]
    assert batch.physical_call_count == 3


async def test_reaction_hydration_discloses_total_sort_and_bounds() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-reactions",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={
                "comment_sampling": {
                    "sorts": ["hot", "newest"],
                    "per_sort_limit": 7,
                    "max_total": 9,
                    "reply_limit": 2,
                    "include_danmaku": True,
                    "danmaku_limit": 40,
                }
            },
        )
    )

    comments = [
        item
        for item in batch.observations
        if item.modality is ObservationModality.COMMENT
    ]
    assert len(comments) == 4
    assert {item.metadata["sort"] for item in comments} == {"hot", "newest"}
    assert all(item.metadata["total_comments"] == 330 for item in comments)
    assert all("total=330" in item.sampling_scope for item in comments)
    roots = [item for item in comments if item.metadata["comment_kind"] == "thread_root"]
    replies = [item for item in comments if item.metadata["comment_kind"] == "thread_reply"]
    assert len(roots) == 2
    assert len(replies) == 2
    assert all("user" not in item.metadata for item in comments)
    assert all("replies" not in item.metadata for item in comments)
    assert all(
        str(item.metadata["author_ref"]).startswith("bilibili-author-")
        for item in comments
    )
    assert replies[0].metadata["parent_comment_ref"] in {
        item.metadata["comment_ref"] for item in roots
    }
    assert tool.comment_calls == [(101, 7, 3, 2), (101, 7, 2, 2)]
    assert "comment_sample_below_requested_budget:4/9" in batch.limitations
    assert any(
        item.modality is ObservationModality.DANMAKU
        for item in batch.observations
    )
    assert batch.physical_call_count == 5


async def test_reaction_hydration_forwards_requested_page_to_comment_fetch() -> None:
    """Catch comment sampling silently dropping the request page (audit A3-1)."""
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-page2",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={
                "page": 2,
                "comment_sampling": {
                    "sorts": ["hot"],
                    "per_sort_limit": 1,
                    "max_total": 1,
                    "reply_limit": 0,
                    "include_danmaku": False,
                },
            },
        )
    )

    assert tool.comment_pages == [2]
    comments = [
        item
        for item in batch.observations
        if item.modality is ObservationModality.COMMENT
    ]
    assert len(comments) == 1
    assert comments[0].location == "comment:hot:page:2:root"
    assert comments[0].metadata["page"] == 2
    assert comments[0].sampling_scope == (
        "sort=hot;page=2;requested=1;returned=1;total=330;reply_limit=0"
    )


async def test_sampling_plan_also_forwards_requested_page() -> None:
    """Catch the plan-derived sampling path dropping the request page too."""
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-plan-page",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={
                "page": 2,
                "comment_sampling_plan": {
                    "target_sample_count": 3,
                    "allocations": [
                        {"stratum": "popular", "requested": 2},
                        {"stratum": "recent", "requested": 1},
                    ],
                },
            },
        )
    )

    assert tool.comment_pages == [2, 2]
    comments = [
        item
        for item in batch.observations
        if item.modality is ObservationModality.COMMENT
    ]
    roots = [item for item in comments if item.metadata["comment_kind"] == "thread_root"]
    assert {item.location for item in roots} == {
        "comment:hot:page:2:root",
        "comment:newest:page:2:root",
    }


async def test_default_sampling_keeps_first_page() -> None:
    """Regression: hydrates without a page argument still fetch page one."""
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-page1",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={
                "comment_sampling": {
                    "sorts": ["hot"],
                    "per_sort_limit": 1,
                    "max_total": 1,
                    "reply_limit": 0,
                    "include_danmaku": False,
                }
            },
        )
    )

    assert tool.comment_pages == [1]
    comments = [
        item
        for item in batch.observations
        if item.modality is ObservationModality.COMMENT
    ]
    assert comments[0].location == "comment:hot:page:1:root"


def test_sampling_arguments_rejects_non_dict_comment_sampling() -> None:
    """Non-dict comment_sampling falls back to empty sampling args (Minor-1).

    Before the isinstance guard, ``dict(explicit)`` raised TypeError outside
    the ValidationError envelope, escaping the adapter's structured failure
    path. Empty args model-validate to the default sampling profile, matching
    the non-dict comment_sampling_plan path below.
    """
    assert BilibiliChannelAdapter._sampling_arguments(
        {"comment_sampling": ["hot", "newest"]}
    ) == {}
    assert BilibiliChannelAdapter._sampling_arguments(
        {"comment_sampling": "hot"}
    ) == {}
    # None falls through to the plan path, which also yields empty args here
    assert BilibiliChannelAdapter._sampling_arguments(
        {"comment_sampling": None}
    ) == {}


def test_sampling_arguments_keeps_valid_comment_sampling() -> None:
    """A valid comment_sampling dict passes through unchanged (plus page)."""
    assert BilibiliChannelAdapter._sampling_arguments(
        {"comment_sampling": {"sorts": ["hot"], "per_sort_limit": 7}}
    ) == {"sorts": ["hot"], "per_sort_limit": 7}
    assert BilibiliChannelAdapter._sampling_arguments(
        {"comment_sampling": {"sorts": ["hot"]}, "page": 2}
    ) == {"sorts": ["hot"], "page": 2}


async def test_hydrate_survives_non_dict_comment_sampling() -> None:
    """Regression (Minor-1): a list comment_sampling must not crash hydrate.

    Before the fix the TypeError escaped hydrate entirely; now the adapter
    falls back to default sampling and proceeds to network access.
    """
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-minor1",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={"comment_sampling": ["hot"]},
        )
    )

    assert "invalid_bilibili_comment_sampling" not in batch.limitations
    assert tool.detail_calls == 1
    assert any(
        item.modality is ObservationModality.COMMENT
        for item in batch.observations
    )


async def test_invalid_sampling_parameters_fail_before_network_access() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-invalid",
            source_ref="BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={"comment_sampling": {"per_sort_limit": 5000}},
        )
    )

    assert batch.partial is True
    assert batch.limitations == ["invalid_bilibili_comment_sampling"]
    assert tool.detail_calls == 0


async def test_generic_comment_plan_controls_platform_fetch_and_reply_depth() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-generic-plan",
            source_ref="BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={
                "comment_sampling_plan": {
                    "target_sample_count": 11,
                    "allocations": [
                        {"stratum": "popular", "requested": 4},
                        {"stratum": "recent", "requested": 3},
                        {"stratum": "thread_root", "requested": 2},
                        {"stratum": "thread_reply", "requested": 2},
                    ],
                }
            },
        )
    )

    assert tool.comment_calls == [(101, 6, 3, 1), (101, 3, 2, 1)]
    comments = [
        item
        for item in batch.observations
        if item.modality is ObservationModality.COMMENT
    ]
    assert {item.metadata["comment_kind"] for item in comments} == {
        "thread_root",
        "thread_reply",
    }
    assert "comment_sample_below_requested_budget:4/11" in batch.limitations


async def test_comment_roots_and_replies_share_one_global_sample_cap() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-one-comment",
            source_ref="BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={
                "comment_sampling": {
                    "sorts": ["hot", "newest"],
                    "per_sort_limit": 5,
                    "max_total": 1,
                    "reply_limit": 5,
                    "include_danmaku": False,
                }
            },
        )
    )

    comments = [
        item
        for item in batch.observations
        if item.modality is ObservationModality.COMMENT
    ]
    assert len(comments) == 1
    assert comments[0].metadata["comment_kind"] == "thread_root"
    assert tool.comment_calls == [(101, 1, 3, 5)]
    assert batch.physical_call_count == 3


def _ten_segment_subtitles() -> list[dict[str, Any]]:
    """Return ten ordered subtitle segments matching the fake video's cid."""
    return [
        {
            "start": float(index),
            "end": float(index + 1),
            "content": f"segment {index} text",
            "language": "zh-CN",
            "acquisition_method": "platform_subtitle",
            "confidence": 1.0,
            "cid": 202,
            "reliability": "confirmed",
            "attempts": 1,
        }
        for index in range(1, 11)
    ]


@pytest.mark.parametrize("reliability", ["confirmed", "best_effort"])
async def test_hydrate_passthroughs_subtitle_reliability(reliability: str) -> None:
    """Per-segment reliability travels in observation metadata (contract 1.1)."""
    tool = FakeBilibiliTool()
    tool.subtitle_segments = [
        {
            "start": 1.0,
            "end": 2.0,
            "content": "spoken content",
            "language": "zh-CN",
            "acquisition_method": "platform_subtitle",
            "confidence": 1.0,
            "cid": 202,
            "reliability": reliability,
            "attempts": 3,
        }
    ]
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-reliability",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
        )
    )

    transcript = next(
        item
        for item in batch.observations
        if item.modality is ObservationModality.TRANSCRIPT
    )
    assert transcript.metadata["subtitle_reliability"] == reliability
    # Non-empty segments mean no subtitle limitation is attached.
    assert SUBTITLE_ABSENT_LIMITATION not in batch.limitations


async def test_hydrate_empty_subtitles_uses_tool_unreliable_limitation() -> None:
    """Empty segments surface the tool's attempts-exhausted reason verbatim."""
    tool = FakeBilibiliTool()
    tool.subtitle_segments = []
    tool.last_subtitle_limitation = SUBTITLE_UNRELIABLE_LIMITATION
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-empty-unreliable",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
        )
    )

    assert SUBTITLE_UNRELIABLE_LIMITATION in batch.limitations
    assert batch.partial is True


async def test_hydrate_empty_subtitles_without_reason_falls_back_to_absent_text() -> None:
    """A tool that records no reason still yields the legacy absent limitation."""
    tool = FakeBilibiliTool()
    tool.subtitle_segments = []
    tool.last_subtitle_limitation = None
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-empty-fallback",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
        )
    )

    assert SUBTITLE_ABSENT_LIMITATION in batch.limitations


async def test_hydrate_forwards_domain_to_subtitle_fetch() -> None:
    """The domain argument reaches the ledger-scoped subtitle fetch."""
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    await adapter.hydrate(
        HydrationRequest(
            id="hydrate-domain",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
            arguments={"domain": "lol_cn"},
        )
    )

    assert tool.subtitle_domains == ["lol_cn"]


async def test_hydrate_without_domain_forwards_empty_default() -> None:
    """Missing domain means an empty string, which skips the ledger (T2)."""
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    await adapter.hydrate(
        HydrationRequest(
            id="hydrate-no-domain",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
        )
    )

    assert tool.subtitle_domains == [""]


async def test_hydrate_full_returns_aggregated_subtitle_body() -> None:
    tool = FakeBilibiliTool()
    tool.subtitle_segments = _ten_segment_subtitles()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-full",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    transcripts = [
        item
        for item in batch.observations
        if item.modality is ObservationModality.TRANSCRIPT
    ]
    aggregate = next(
        item for item in transcripts if item.location == "video_transcript:full"
    )
    expected = "\n".join(
        f"00:{index:02d} segment {index} text" for index in range(1, 11)
    )
    assert aggregate.body == expected
    assert "body_truncated" not in aggregate.limitations
    assert aggregate.metadata["segment_count"] == 10
    assert aggregate.metadata["body_truncated"] is False
    assert aggregate.metadata["subtitle_reliability"] == "confirmed"
    per_segment = [
        item for item in transcripts if item.location != "video_transcript:full"
    ]
    assert len(per_segment) == 10
    assert all(item.body is None for item in per_segment)
    description = next(
        item
        for item in batch.observations
        if item.modality is ObservationModality.DOCUMENT_TEXT
    )
    assert description.body == "a bounded description"
    metadata = next(
        item
        for item in batch.observations
        if item.modality is ObservationModality.METADATA
    )
    assert metadata.body is None
    assert batch.physical_call_count == 4


async def test_hydrate_non_full_has_no_body_field() -> None:
    tool = FakeBilibiliTool()
    tool.subtitle_segments = _ten_segment_subtitles()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-not-full",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
        )
    )

    assert all(item.body is None for item in batch.observations)
    assert all(item.location != "video_transcript:full" for item in batch.observations)
    assert all(item.location != "comment:full" for item in batch.observations)


async def test_hydrate_full_body_is_capped_with_truncation_limitation() -> None:
    tool = FakeBilibiliTool()
    tool.subtitle_segments = [
        {
            "start": float(index),
            "end": float(index + 1),
            "content": "x" * 4000,
            "language": "zh-CN",
            "acquisition_method": "platform_subtitle",
            "confidence": 1.0,
            "cid": 202,
        }
        for index in range(1, 11)
    ]
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-full-capped",
            source_ref="BV001",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    aggregate = next(
        item
        for item in batch.observations
        if item.location == "video_transcript:full"
    )
    assert aggregate.body is not None
    assert len(aggregate.body) == 32_000
    assert "body_truncated" in aggregate.limitations
    assert aggregate.metadata["body_truncated"] is True


async def test_hydrate_full_reactions_joins_sampled_comments() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-full-reactions",
            source_ref="BV001",
            depth=HydrationDepth.REACTIONS,
            arguments={
                "full": True,
                "comment_sampling": {
                    "sorts": ["hot", "newest"],
                    "per_sort_limit": 7,
                    "max_total": 9,
                    "reply_limit": 2,
                    "include_danmaku": True,
                    "danmaku_limit": 40,
                },
            },
        )
    )

    aggregate = next(
        item for item in batch.observations if item.location == "comment:full"
    )
    assert aggregate.modality is ObservationModality.COMMENT
    assert aggregate.body == (
        "[hot] likes=3 hot comment\n"
        "[hot] likes=1 hot child\n"
        "[newest] likes=3 newest comment\n"
        "[newest] likes=1 newest child"
    )
    assert aggregate.metadata["comment_count"] == 4
    assert aggregate.metadata["body_truncated"] is False
    assert "body_truncated" not in aggregate.limitations
    danmaku = [
        item
        for item in batch.observations
        if item.modality is ObservationModality.DANMAKU
    ]
    assert len(danmaku) == 1
    assert danmaku[0].body is None


async def test_hydrate_full_metadata_depth_carries_description_body() -> None:
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-full-metadata",
            source_ref="BV001",
            depth=HydrationDepth.METADATA,
            arguments={"full": True},
        )
    )

    metadata = batch.observations[0]
    assert metadata.modality is ObservationModality.METADATA
    assert metadata.body == "a bounded description"
    assert batch.physical_call_count == 2


async def test_hydrate_observations_share_one_source_derived_id() -> None:
    """Catch observation ids drifting off the unified source-derived rule (T13).

    One TEXT hydration returns several observations (video metadata,
    description, transcript segment) with different modalities, locations,
    and excerpts, all for the same source. Under the unified id rule none of
    those fields participate in the id, so every observation carries the
    same ``observation-<hash(source_ref)>`` suffix the world store derives
    (observation_id) — discovery cards and hydrated rows collide on one row.
    """
    tool = FakeBilibiliTool()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="hydrate-id-unity",
            source_ref="bilibili-BV001",
            depth=HydrationDepth.TEXT,
            arguments={},
        )
    )

    assert len(batch.observations) >= 3  # metadata + description + transcript
    assert len({item.modality for item in batch.observations}) > 1
    assert len({item.location for item in batch.observations}) > 1
    expected = observation_id("bilibili", "bilibili-BV001").split(":", 1)[1]
    assert {item.id for item in batch.observations} == {expected}


async def test_open_source_full_assembles_bodies_across_observations(tmp_path: Path) -> None:
    """Catch the tools layer picking the first observation's (empty) body card.

    The real adapter's TEXT-depth full hydrate places the metadata observation
    first with body None, the description body on the next index, and the
    aggregated transcript on the last observation. Index-0 selection therefore
    produced an empty body card, no observation_bodies write, and a dead
    re-read path; the assembled body must carry the description plus the
    transcript and persist for a zero-network store serve.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="bilibili:seen:bilibili-BV001",
                    source_uri="bilibili-BV001",
                    source_kind="bilibili",
                    title="current match review",
                    depth=ObservationDepth.SEEN,
                    observed_at=NOW,
                    metadata={"adapter_version": "2.1.0"},
                )
            ]
        ),
        "seed-full",
    )
    tool = FakeBilibiliTool()
    tool.subtitle_segments = _ten_segment_subtitles()
    adapter = BilibiliChannelAdapter(tool, now_factory=lambda: NOW)
    tools = WorldTools(store=store, adapters={"bilibili": adapter})

    r1 = await tools.execute(
        "open_source", {"observation_id": "bilibili:seen:bilibili-BV001", "depth": "full"}, "c1"
    )

    assert r1["ok"] is True
    body = r1["cards"][0]["body"]
    assert "a bounded description" in body
    assert "00:01 segment 1 text" in body
    assert "00:10 segment 10 text" in body
    assert tool.detail_calls == 1
    # the assembled body was written to the store envelope
    row = store.read_observation_body("bilibili:seen:bilibili-BV001")
    assert row is not None
    assert json.loads(row["body_json"])["text"] == body
    # a second full call serves from the store with zero hydration
    r2 = await tools.execute(
        "open_source", {"observation_id": "bilibili:seen:bilibili-BV001", "depth": "full"}, "c2"
    )
    assert "served_from_store" in r2["limitations"]
    assert tool.detail_calls == 1
    assert r2["cards"][0]["body"] == body
