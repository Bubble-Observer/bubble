from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from leave_information_bubble.channels import (
    AcquisitionEntryKind,
    AcquisitionOutcome,
    BilibiliChannelAdapter,
    ChannelCapabilityRole,
    ChannelHealth,
    ChannelHealthStatus,
    DiscoveryBatch,
    HupuChannelAdapter,
    HydrationDepth,
    HydrationRequest,
    NgaChannelAdapter,
    ObservationBatch,
    PublicWebChannelAdapter,
    QuerySemantics,
    ScanRequest,
    SourceOccurrence,
    TimeFilterPrecision,
)
from leave_information_bubble.tools.hupu import HupuBoardPage
from leave_information_bubble.tools.nga import NgaBoardPage

NOW = datetime(2026, 7, 20, tzinfo=UTC)


class _FakeHupuTool:
    """Descriptor inspection does not invoke the board tool."""


class _FakeNgaTool:
    """Descriptor inspection does not invoke the board tool."""


class _FakeBilibiliTool:
    """Descriptor inspection does not invoke the platform tool."""


class _FakePublicWebTool:
    """Descriptor inspection does not invoke the search tool."""


def test_hupu_declares_board_scan_and_bounded_query_hint() -> None:
    """Catch a board scan being represented as an unrestricted search."""
    descriptors = HupuChannelAdapter(_FakeHupuTool()).capability_descriptors  # type: ignore[arg-type]
    community = next(
        item
        for item in descriptors
        if item.role is ChannelCapabilityRole.COMMUNITY_STREAM
    )

    assert community.entry_kind is AcquisitionEntryKind.BOUNDED_BOARD
    assert community.query_semantics is QuerySemantics.BOUNDED_RERANK_HINT
    assert community.supports_queryless is True


@pytest.mark.parametrize(
    "adapter",
    [
        HupuChannelAdapter(_FakeHupuTool()),  # type: ignore[arg-type]
        NgaChannelAdapter(_FakeNgaTool()),  # type: ignore[arg-type]
    ],
)
def test_community_board_adapters_declare_only_the_real_discovery_surface(
    adapter: HupuChannelAdapter | NgaChannelAdapter,
) -> None:
    """Catch one board behavior being advertised as a second attention surface."""
    assert [
        (descriptor.role, descriptor.entry_kind, descriptor.query_semantics)
        for descriptor in adapter.capability_descriptors
    ] == [
        (
            ChannelCapabilityRole.COMMUNITY_STREAM,
            AcquisitionEntryKind.BOUNDED_BOARD,
            QuerySemantics.BOUNDED_RERANK_HINT,
        )
    ]


def test_bilibili_and_public_web_declare_true_query_search() -> None:
    """Catch query-required search surfaces claiming queryless support."""
    bilibili = next(
        item
        for item in BilibiliChannelAdapter(_FakeBilibiliTool()).capability_descriptors  # type: ignore[arg-type]
        if item.role is ChannelCapabilityRole.RECENT_STREAM
    )
    public = next(
        item
        for item in PublicWebChannelAdapter(_FakePublicWebTool()).capability_descriptors  # type: ignore[arg-type]
        if item.role is ChannelCapabilityRole.NEWS_INDEX
    )

    assert bilibili.query_semantics is QuerySemantics.PLATFORM_SEARCH
    assert public.query_semantics is QuerySemantics.PLATFORM_SEARCH
    assert not bilibili.supports_queryless
    assert not public.supports_queryless


def test_bilibili_recent_stream_declares_time_filter_unsupported() -> None:
    """Catch pubdate ordering being mistaken for provider time filtering."""
    recent = next(
        item
        for item in BilibiliChannelAdapter(_FakeBilibiliTool()).capability_descriptors  # type: ignore[arg-type]
        if item.role is ChannelCapabilityRole.RECENT_STREAM
    )

    assert recent.time_filter_precision is TimeFilterPrecision.UNSUPPORTED


def test_scan_request_rejects_inverted_window() -> None:
    with pytest.raises(ValidationError, match="window_end"):
        ScanRequest(
            id="scan-1",
            lane="global",
            window_start=NOW,
            window_end=NOW - timedelta(minutes=1),
        )


def test_rate_limited_health_requires_retry_horizon() -> None:
    with pytest.raises(ValidationError, match="requires retry_after"):
        ChannelHealth(
            adapter_id="public-feed",
            adapter_version="1.0.0",
            status=ChannelHealthStatus.RATE_LIMITED,
            checked_at=NOW,
        )


def test_channel_contracts_are_frozen_and_forbid_unknown_fields() -> None:
    request = HydrationRequest(
        id="hydrate-1",
        source_ref="source:1",
        depth=HydrationDepth.TEXT,
    )

    with pytest.raises(ValidationError):
        request.depth = HydrationDepth.STRUCTURED
    with pytest.raises(ValidationError, match="extra"):
        SourceOccurrence(
            id="occurrence-1",
            adapter_id="feed",
            adapter_version="1.0.0",
            source_ref="item:1",
            captured_at=NOW,
            unsupported=True,
        )


def test_verified_occurrence_hash_requires_sha256_shape() -> None:
    with pytest.raises(ValidationError, match="64-character"):
        SourceOccurrence(
            id="occurrence-1",
            adapter_id="feed",
            adapter_version="1.0.0",
            source_ref="item:1",
            captured_at=NOW,
            content_hash="source-controlled-value",
            content_hash_verified=True,
        )


def test_discovery_batch_exposes_a_stable_operational_outcome() -> None:
    """Catch partial, empty, unsupported, and unavailable scans collapsing into limitations."""
    properties = DiscoveryBatch.model_json_schema()["properties"]

    assert "outcome" in properties


def test_observation_batch_exposes_the_same_operational_outcome_vocabulary() -> None:
    """Open and scan feedback share one stable outcome vocabulary."""
    properties = ObservationBatch.model_json_schema()["properties"]

    assert "outcome" in properties


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_kind", "error"),
    [("hupu", "hupu_board_unavailable"), ("nga", "nga_board_unavailable")],
)
async def test_community_board_failure_reports_unavailable(
    adapter_kind: str,
    error: str,
) -> None:
    """A failed board request must not look like an empty board page."""

    class FailingBoardTool:
        async def scan_board(
            self, board: str, *, page: int, limit: int
        ) -> HupuBoardPage | NgaBoardPage:
            del limit
            if adapter_kind == "hupu":
                return HupuBoardPage(
                    board=board, page=page, url="https://bbs.hupu.com", error=error
                )
            return NgaBoardPage(
                fid=board, page=page, url="https://bbs.nga.cn", error=error
            )

    if adapter_kind == "hupu":
        adapter = HupuChannelAdapter(FailingBoardTool())  # type: ignore[arg-type]
    else:
        adapter = NgaChannelAdapter(FailingBoardTool())  # type: ignore[arg-type]

    batch = await adapter.discover(
        ScanRequest(id=f"{adapter_kind}-failure", lane="ambient")
    )

    assert batch.outcome is AcquisitionOutcome.UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_kind", ["hupu", "nga"])
async def test_invalid_community_source_reports_unsupported(adapter_kind: str) -> None:
    """An invalid source identity is a capability mismatch, not empty content."""

    class InvalidThreadTool:
        @staticmethod
        def thread_id(source_ref: str) -> str:
            del source_ref
            return ""

    if adapter_kind == "hupu":
        adapter = HupuChannelAdapter(InvalidThreadTool())  # type: ignore[arg-type]
    else:
        adapter = NgaChannelAdapter(InvalidThreadTool())  # type: ignore[arg-type]

    batch = await adapter.hydrate(
        HydrationRequest(
            id=f"{adapter_kind}-invalid",
            source_ref="not-a-thread",
            depth=HydrationDepth.TEXT,
        )
    )

    assert batch.outcome is AcquisitionOutcome.UNSUPPORTED


# --- domain-scoped default surfaces (task: adapter domain parameters) -------


@pytest.mark.parametrize(
    ("adapter", "limitation_fragment"),
    [
        (HupuChannelAdapter(_FakeHupuTool(), default_board=None),  # type: ignore[arg-type]
         "no_default_board_for_domain"),
        (NgaChannelAdapter(_FakeNgaTool(), default_fid=None),  # type: ignore[arg-type]
         "no_default_fid_for_domain"),
    ],
)
def test_community_board_adapter_without_domain_default_degrades(
    adapter: HupuChannelAdapter | NgaChannelAdapter,
    limitation_fragment: str,
) -> None:
    """Without a configured board/fid the adapter must not silently scan the
    wrong platform section: it declares queryless=no and a typed limitation."""
    community = next(
        item
        for item in adapter.capability_descriptors
        if item.role is ChannelCapabilityRole.COMMUNITY_STREAM
    )
    assert community.supports_queryless is False
    assert any(limitation_fragment in limitation for limitation in community.limitations)


@pytest.mark.parametrize("adapter_kind", ["hupu", "nga"])
async def test_degraded_board_adapter_returns_unsupported_without_surface_key(
    adapter_kind: str,
) -> None:
    """A discover without surface_key on a domainless adapter must return a
    typed UNSUPPORTED instead of hitting the platform's default (LOL) board."""

    class ExplodingBoardTool:
        async def scan_board(self, board: str, *, page: int, limit: int) -> object:
            raise AssertionError(f"scan_board must not run without a surface key (board={board!r})")

    if adapter_kind == "hupu":
        adapter = HupuChannelAdapter(ExplodingBoardTool(), default_board=None)  # type: ignore[arg-type]
    else:
        adapter = NgaChannelAdapter(ExplodingBoardTool(), default_fid=None)  # type: ignore[arg-type]

    batch = await adapter.discover(
        ScanRequest(id=f"{adapter_kind}-degraded", lane="ambient")
    )

    assert batch.outcome is AcquisitionOutcome.UNSUPPORTED
    assert batch.partial is True
    assert any(
        "surface_key" in limitation for limitation in batch.limitations
    ), batch.limitations


@pytest.mark.parametrize(
    ("adapter_kind", "surface_key", "expected"),
    [
        ("hupu", "csgo", "csgo"),
        ("nga", "335", "335"),
    ],
)
async def test_degraded_board_adapter_honors_surface_key(
    adapter_kind: str,
    surface_key: str,
    expected: str,
) -> None:
    """surface_key reaches the platform scan as the explicit board/fid."""

    class RecordingBoardTool:
        def __init__(self) -> None:
            self.boards: list[str] = []

        async def scan_board(self, board: str, *, page: int, limit: int) -> object:
            self.boards.append(board)
            if adapter_kind == "hupu":
                return HupuBoardPage(board=board, page=page, url="https://bbs.hupu.com")
            return NgaBoardPage(fid=board, page=page, url="https://bbs.nga.cn")

    tool = RecordingBoardTool()
    if adapter_kind == "hupu":
        adapter = HupuChannelAdapter(tool, default_board=None)  # type: ignore[arg-type]
    else:
        adapter = NgaChannelAdapter(tool, default_fid=None)  # type: ignore[arg-type]

    batch = await adapter.discover(
        ScanRequest(
            id=f"{adapter_kind}-keyed",
            lane="ambient",
            arguments={"surface_key": surface_key},
        )
    )

    # an empty board page is EMPTY content, not an unsupported request —
    # the scan ran against the surface_key board/fid
    assert batch.outcome is AcquisitionOutcome.EMPTY
    assert batch.physical_call_count == 1
    assert tool.boards == [expected]


async def test_configured_board_adapter_keeps_queryless_scan() -> None:
    """A configured domain adapter (lol_cn bare CLI) scans its board by default."""
    tool = _FakeHupuTool()

    class ConfiguredTool:
        async def scan_board(self, board: str, *, page: int, limit: int) -> HupuBoardPage:
            return HupuBoardPage(board=board, page=page, url="https://bbs.hupu.com")

    adapter = HupuChannelAdapter(ConfiguredTool())  # type: ignore[arg-type]
    batch = await adapter.discover(ScanRequest(id="hupu-lol", lane="ambient"))
    assert batch.outcome is AcquisitionOutcome.EMPTY
    assert batch.physical_call_count == 1
