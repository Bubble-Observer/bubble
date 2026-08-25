# Split from tests/world/test_tools.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

import pytest

from leave_information_bubble.channels import (
    AcquisitionEntryKind,
    AcquisitionOutcome,
    CapabilityDescriptor,
    ChannelCapabilityRole,
    DiscoveryBatch,
    HupuChannelAdapter,
    ObservationBatch,
    QuerySemantics,
    ReplayChannelAdapter,
    ScanRequest,
    SourceOccurrence,
    TimeFilterPrecision,
)
from leave_information_bubble.runtime.errors import AgentError, ErrorCode
from leave_information_bubble.tools.hupu import HupuBoardPage, HupuThreadCard
from leave_information_bubble.world import (
    AssertionProposal,
    CognitionDeltaProposal,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    GraphRef,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    ProposalCommitter,
    WorldStore,
)
from leave_information_bubble.world.store import observation_id
from leave_information_bubble.world.tools import WorldTools
from tests.world._tools_helpers import (
    CAP,
    NOW,
    PUBLISHED,
    SOURCE,
    _adapter,
    _anchor,
    _observation,
    _occurrence,
    _tools,
)
from tests.world.test_tools_open_depth import (
    _anchor_observation,
    _EmptyRelatedReplayAdapter,
    _ForgedRelatedReplayAdapter,
    _FullBodyAdapter,
    _related_adapter,
)


async def test_discover_sources_persists_seen_observations_and_returns_bounded_cards(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch discovery cards that expose adapter-local rather than durable evidence identities."""
    tools, store = _tools(tmp_path)

    result = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "discover-call"
    )

    assert result["cards"] == [
        {
            "id": observation_id("replay", SOURCE),
            "title": "A discovered source",
            "source_uri": SOURCE,
            "source_kind": "replay",
            "depth": "seen",
            "published_at": PUBLISHED.isoformat(),
            "observed_at": NOW.isoformat(),
            "excerpt": "discovery excerpt",
            "material_reliability": "unknown",
            "limitations": [],
            "persisted": True,
            "adapter": {"id": "replay", "version": "1"},
        }
    ]
    with store.read_connection() as connection:
        row = connection.execute("SELECT id, depth, excerpt, metadata_json FROM observations").fetchone()
    assert row["id"] == result["cards"][0]["id"]
    assert row["depth"] == "seen"
    assert row["excerpt"] == "discovery excerpt"
    assert "unsafe_body" not in row["metadata_json"]


async def test_discover_success_bounds_adapter_limitations(tmp_path: pytest.TempPathFactory) -> None:
    """A successful discovery cannot return an unbounded adapter limitation payload."""
    raw_limitations = [f"{index}-" + "x" * 200 for index in range(10)]
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:bounded-discover": DiscoveryBatch(
                request_id="tool:bounded-discover",
                adapter_id="replay",
                adapter_version="1",
                occurrences=[_occurrence()],
                limitations=raw_limitations,
            )
        },
        hydrations={},
    )
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={"replay": adapter})

    result = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "bounded-discover"
    )

    assert result["ok"] is True
    assert result["limitations"][-1] == "limitations_truncated"
    assert len(result["limitations"]) <= 8
    assert max(map(len, result["limitations"])) <= 160
    assert sum(map(len, result["limitations"])) <= 640


async def test_discover_sources_runs_without_query(tmp_path: pytest.TempPathFactory) -> None:
    """Catch the scan schema making the targetless breadth survey impossible.

    discover_sources claims a survey without a precise target, so omitting
    the query must be legal and must not crash: forum-style adapters scan
    their board order, and query-driven adapters answer with a typed
    limitation (e.g. bilibili_query_required) instead of an exception.
    """
    tools, _ = _tools(tmp_path)

    result = await tools.execute("discover_sources", {"adapter": "replay"}, "schema-discover_sources")

    assert result["ok"] is True
    assert result["cards"] and result["cards"][0]["id"] == observation_id("replay", SOURCE)


class _ScanRecordingAdapter(ReplayChannelAdapter):
    """Record facade-normalized scans without making an external call."""

    def __init__(self) -> None:
        super().__init__(
            adapter_id="bilibili",
            adapter_version="1",
            discoveries={},
            hydrations={},
        )
        self.requests: list[ScanRequest] = []

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        return (
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.RECENT_STREAM,
                entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
                query_semantics=QuerySemantics.PLATFORM_SEARCH,
                supports_queryless=False,
            ),
        )

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        self.requests.append(request)
        return DiscoveryBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            occurrences=[],
        )


class _ConfiguredScanRecordingAdapter(_ScanRecordingAdapter):
    """Expose caller-supplied capability declarations to the facade."""

    def __init__(self, descriptors: tuple[CapabilityDescriptor, ...]) -> None:
        super().__init__()
        self._descriptors = descriptors

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        return self._descriptors


def _scan_descriptor(
    role: ChannelCapabilityRole,
    *,
    entry_kind: AcquisitionEntryKind = AcquisitionEntryKind.PLATFORM_SEARCH,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        adapter_id="bilibili",
        adapter_version="1",
        role=role,
        entry_kind=entry_kind,
        query_semantics=QuerySemantics.PLATFORM_SEARCH,
        supports_queryless=False,
    )


class _StaticHupuBoardTool:
    async def scan_board(self, board: str, *, page: int, limit: int) -> HupuBoardPage:
        del limit
        return HupuBoardPage(
            board=board,
            page=page,
            url="https://bbs.hupu.com/lol",
            threads=(
                HupuThreadCard(
                    thread_id="1",
                    title="BLG Bin discussion",
                    url="https://bbs.hupu.com/1.html",
                ),
            ),
        )


class _CountingHupuAdapter(HupuChannelAdapter):
    def __init__(self) -> None:
        super().__init__(_StaticHupuBoardTool(), now_factory=lambda: NOW)  # type: ignore[arg-type]
        self.calls = 0

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        self.calls += 1
        return await super().discover(request)


class _StableHupuBoardTool:
    async def scan_board(self, board: str, *, page: int, limit: int) -> HupuBoardPage:
        del limit
        return HupuBoardPage(
            board=board,
            page=page,
            url="https://bbs.hupu.com/lol",
            threads=tuple(
                HupuThreadCard(
                    thread_id=str(index),
                    title=f"BLG stable discussion {index}",
                    url=f"https://bbs.hupu.com/{index}.html",
                )
                for index in range(1, 6)
            ),
        )


class _StablePageAdapter(HupuChannelAdapter):
    def __init__(self) -> None:
        super().__init__(_StableHupuBoardTool(), now_factory=lambda: NOW)  # type: ignore[arg-type]


async def test_second_scan_reports_repeated_source_ids(tmp_path: Path) -> None:
    """A repeated page reports factual source overlap within the current wake."""
    tools = WorldTools(
        store=WorldStore(tmp_path / "world.sqlite3"),
        adapters={"hupu": _StablePageAdapter()},
    )

    first = await tools.execute("discover_sources", {"adapter": "hupu"}, "first")
    second = await tools.execute(
        "discover_sources", {"adapter": "hupu", "query": "BLG"}, "second"
    )

    expected_ids = [
        observation_id("hupu", f"hupu-thread-{index}")
        for index in range(1, 6)
    ]
    assert [card["id"] for card in first["cards"]] == expected_ids
    assert [card["id"] for card in second["cards"]] == expected_ids
    assert first["novelty"] == {
        "new_in_wake": 5,
        "repeated_in_wake": 0,
        "retained_history": 5,
        "history_truncated": False,
    }
    assert second["novelty"]["new_in_wake"] == 0
    assert second["novelty"]["repeated_in_wake"] == 5


async def test_scan_surface_role_reaches_adapter(tmp_path: Path) -> None:
    adapter = _ScanRecordingAdapter()
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={"bilibili": adapter})

    await tools.execute(
        "discover_sources",
        {"adapter": "bilibili", "query": "LPL", "surface_role": "recent_stream"},
        "recent",
    )

    assert adapter.requests[-1].capability_roles == [ChannelCapabilityRole.RECENT_STREAM]


async def test_scan_rejects_requested_surface_not_declared_by_adapter(tmp_path: Path) -> None:
    adapter = _ScanRecordingAdapter()
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={"bilibili": adapter})

    result = await tools.execute(
        "discover_sources",
        {"adapter": "bilibili", "query": "LPL", "surface_role": "current_index"},
        "unsupported-surface",
    )

    assert result == {
        "ok": False,
        "outcome": "unsupported",
        "error": {
            "code": "adapter_surface_role_unsupported",
            "stage": "discover",
            "attempts": 0,
            "same_request_can_change": False,
            "change_condition": "request_arguments_or_adapter_capability_changes",
            "declared_surface_roles": ["recent_stream"],
        },
        "limitations": ["adapter_surface_role_unsupported"],
    }
    assert adapter.requests == []


async def test_hupu_preflight_rejects_attention_but_allows_community(
    tmp_path: Path,
) -> None:
    """Catch a duplicate Hupu surface reaching the single real board call."""
    adapter = _CountingHupuAdapter()
    tools = WorldTools(
        store=WorldStore(tmp_path / "world.sqlite3"),
        adapters={"hupu": adapter},
    )

    rejected = await tools.execute(
        "discover_sources",
        {
            "adapter": "hupu",
            "query": "BLG Bin",
            "surface_role": "attention_ranking",
        },
        "hupu-attention",
    )

    assert rejected["outcome"] == "unsupported"
    assert rejected["error"] == {
        "code": "adapter_surface_role_unsupported",
        "stage": "discover",
        "attempts": 0,
        "same_request_can_change": False,
        "change_condition": "request_arguments_or_adapter_capability_changes",
        "declared_surface_roles": ["community_stream"],
    }
    assert rejected["limitations"] == ["adapter_surface_role_unsupported"]
    assert adapter.calls == 0

    allowed = await tools.execute(
        "discover_sources",
        {
            "adapter": "hupu",
            "query": "BLG Bin",
            "surface_role": "community_stream",
        },
        "hupu-community",
    )

    assert allowed["ok"] is True
    assert allowed["scope"]["query_semantics"] == "bounded_rerank_hint"
    assert adapter.calls == 1


async def test_scan_rejects_hydration_only_surface_without_adapter_call(tmp_path: Path) -> None:
    adapter = _ConfiguredScanRecordingAdapter(
        (
            _scan_descriptor(ChannelCapabilityRole.PLATFORM_DISCOVERY),
            _scan_descriptor(
                ChannelCapabilityRole.COMMUNITY_STREAM,
                entry_kind=AcquisitionEntryKind.SOURCE_HYDRATION,
            ),
        )
    )
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={"bilibili": adapter})

    result = await tools.execute(
        "discover_sources",
        {"adapter": "bilibili", "query": "LPL", "surface_role": "community_stream"},
        "hydration-only-surface",
    )

    assert result["outcome"] == "unsupported"
    assert result["error"] == {
        "code": "adapter_surface_role_unsupported",
        "stage": "discover",
        "attempts": 0,
        "same_request_can_change": False,
        "change_condition": "request_arguments_or_adapter_capability_changes",
        "declared_surface_roles": ["platform_discovery"],
    }
    assert adapter.requests == []


async def test_search_rejects_bounded_board_hint_without_dispatch(tmp_path: Path) -> None:
    adapter = _CountingHupuAdapter()
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={"hupu": adapter})

    result = await tools.execute(
        "search_sources", {"adapter": "hupu", "query": "BLG Bin"}, "search"
    )

    assert result["outcome"] == "unsupported"
    assert result["error"]["code"] == "adapter_targeted_search_unsupported"
    assert result["completeness"]["physical_calls"] == 0
    assert adapter.calls == 0


async def test_discover_allows_bounded_board_hint_and_exposes_selection(tmp_path: Path) -> None:
    adapter = _CountingHupuAdapter()
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={"hupu": adapter})

    result = await tools.execute(
        "discover_sources", {"adapter": "hupu", "query": "BLG Bin"}, "discover"
    )

    assert result["ok"] is True
    assert result["scope"]["query_semantics"] == "bounded_rerank_hint"
    assert result["cards"][0]["selection"]["match_method"] == "bounded_client_side"
    assert adapter.calls == 1


async def test_search_keeps_descriptor_free_replay_adapter_usable(tmp_path: Path) -> None:
    tools, _ = _tools(tmp_path)

    result = await tools.execute(
        "search_sources", {"adapter": "replay", "query": "source"}, "schema-search_sources"
    )

    assert result["ok"] is True
    assert result["cards"]


async def test_scan_unsupported_surface_reports_bounded_deduplicated_discovery_roles(
    tmp_path: Path,
) -> None:
    declared = (
        ChannelCapabilityRole.PLATFORM_DISCOVERY,
        ChannelCapabilityRole.ATTENTION_RANKING,
        ChannelCapabilityRole.SEARCH_INDEX,
        ChannelCapabilityRole.NEWS_INDEX,
        ChannelCapabilityRole.CURRENT_INDEX,
        ChannelCapabilityRole.COMMUNITY_STREAM,
    )
    adapter = _ConfiguredScanRecordingAdapter(
        tuple(_scan_descriptor(role) for role in declared for _ in range(3))
        + tuple(
            _scan_descriptor(
                ChannelCapabilityRole.COMMUNITY_STREAM,
                entry_kind=AcquisitionEntryKind.SOURCE_HYDRATION,
            )
            for _ in range(3)
        )
        + tuple(_scan_descriptor(ChannelCapabilityRole.WEB_BROWSER) for _ in range(3))
    )
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={"bilibili": adapter})

    result = await tools.execute(
        "discover_sources",
        {"adapter": "bilibili", "query": "LPL", "surface_role": "recent_stream"},
        "bounded-surface-feedback",
    )

    assert result["error"]["declared_surface_roles"] == [role.value for role in declared]
    assert len(result["error"]["declared_surface_roles"]) <= 7
    assert adapter.requests == []


class _CursorRecordingAdapter(ReplayChannelAdapter):
    """Return deterministic pages and retain each normalized scan request."""

    def __init__(self) -> None:
        super().__init__(
            adapter_id="cursor",
            adapter_version="1",
            discoveries={},
            hydrations={},
        )
        self.requests: list[ScanRequest] = []
        self.outcome = AcquisitionOutcome.SUCCESS
        self.partial = False
        self.limitations: list[str] = []
        self.empty = False

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        return (
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.COMMUNITY_STREAM,
                entry_kind=AcquisitionEntryKind.BOUNDED_BOARD,
                query_semantics=QuerySemantics.BOUNDED_RERANK_HINT,
                supports_queryless=True,
                time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
            ),
        )

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Select page two only when the model-visible cursor reached the adapter."""
        self.requests.append(request)
        page = 2 if request.cursor == "page-2" else 1
        return DiscoveryBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            occurrences=[]
            if self.empty
            else [
                SourceOccurrence(
                    id=f"page-{page}",
                    adapter_id=self.adapter_id,
                    adapter_version=self.adapter_version,
                    source_ref=f"cursor-page-{page}",
                    title=f"page-{page}",
                    captured_at=NOW,
                )
            ],
            next_cursor="" if self.empty else ("page-2" if page == 1 else ""),
            outcome=self.outcome,
            partial=self.partial,
            limitations=self.limitations,
        )


class _TargetedCursorRecordingAdapter(_CursorRecordingAdapter):
    """Cursor fixture whose declaration permits provider-side targeted search."""

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        return (
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.SEARCH_INDEX,
                entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
                query_semantics=QuerySemantics.PLATFORM_SEARCH,
                supports_queryless=False,
                time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
            ),
        )


async def test_discover_forwards_returned_cursor_and_echoes_effective_scope(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a next_cursor that the model can see but cannot send back."""
    adapter = _CursorRecordingAdapter()
    tools = WorldTools(
        store=WorldStore(tmp_path / "cursor.sqlite3"),
        adapters={"cursor": adapter},
    )

    first = await tools.execute("discover_sources", {"adapter": "cursor"}, "page-one")
    second = await tools.execute(
        "discover_sources",
        {"adapter": "cursor", "cursor": first["next_cursor"]},
        "page-two",
    )

    assert second["cards"][0]["title"] == "page-2"
    assert second["scope"]["cursor"] == "page-2"
    schema = {item["function"]["name"]: item["function"]["parameters"] for item in tools.schemas()}
    assert "cursor" in schema["discover_sources"]["properties"]


async def test_broad_discover_defaults_to_last_24_hours_and_echoes_scope(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a missing window silently becoming a zero-width instant."""
    adapter = _CursorRecordingAdapter()
    tools = WorldTools(
        store=WorldStore(tmp_path / "broad-window.sqlite3"),
        adapters={"cursor": adapter},
        mode="broad",
    )

    result = await tools.execute("discover_sources", {"adapter": "cursor"}, "broad-window")

    request = adapter.requests[-1]
    assert request.window_start is not None
    assert request.window_end is not None
    assert request.window_end - request.window_start == timedelta(hours=24)
    assert result["scope"]["time_window"] == {
        "mode": "broad_default_24h",
        "start": request.window_start.isoformat(),
        "end": request.window_end.isoformat(),
        "applied": False,
        "precision": "unsupported",
    }


async def test_deep_discover_and_every_search_leave_omitted_window_unfiltered(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a hidden recent-window policy leaking into Deep or targeted search."""
    adapter = _TargetedCursorRecordingAdapter()
    tools = WorldTools(
        store=WorldStore(tmp_path / "unfiltered-window.sqlite3"),
        adapters={"cursor": adapter},
        mode="deep",
    )

    deep = await tools.execute("discover_sources", {"adapter": "cursor"}, "deep-window")
    targeted = await tools.execute(
        "search_sources",
        {"adapter": "cursor", "query": "TES"},
        "search-window",
    )

    assert adapter.requests[0].window_start is None
    assert adapter.requests[0].window_end is None
    assert adapter.requests[1].window_start is None
    assert adapter.requests[1].window_end is None
    assert deep["scope"]["time_window"] == {
        "mode": "unspecified",
        "start": None,
        "end": None,
        "applied": False,
        "precision": "not_requested",
    }
    assert targeted["scope"]["time_window"]["mode"] == "unspecified"


async def test_scan_rejects_a_half_specified_explicit_window(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch an explicit start being paired with a hidden synthetic end."""
    adapter = _TargetedCursorRecordingAdapter()
    tools = WorldTools(
        store=WorldStore(tmp_path / "half-window.sqlite3"),
        adapters={"cursor": adapter},
        mode="deep",
    )

    result = await tools.execute(
        "search_sources",
        {
            "adapter": "cursor",
            "query": "TES",
            "window_start": "2026-08-14T00:00:00+00:00",
        },
        "half-window",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["attempts"] == 0
    assert adapter.requests == []


async def test_discovery_unsupported_is_not_reported_as_successful_empty_cards(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch query-required and unavailable adapters masquerading as ok empty scans."""
    adapter = _CursorRecordingAdapter()
    adapter.outcome = AcquisitionOutcome.UNSUPPORTED
    adapter.partial = True
    adapter.limitations = ["query_required"]
    adapter.empty = True
    tools = WorldTools(
        store=WorldStore(tmp_path / "unsupported.sqlite3"),
        adapters={"cursor": adapter},
    )

    result = await tools.execute("discover_sources", {"adapter": "cursor"}, "unsupported")

    assert result["ok"] is False
    assert result["outcome"] == "unsupported"
    assert result["error"] == {
        "code": "query_required",
        "stage": "discover",
        "attempts": 0,
        "same_request_can_change": False,
        "change_condition": "request_arguments_or_adapter_capability_changes",
    }
    assert result["completeness"] == {
        "returned": 0,
        "limit": 5,
        "partial": True,
        "truncated": False,
        "next_cursor_usable": False,
        "physical_calls": 0,
    }


async def test_partial_discovery_keeps_cards_and_exposes_completeness(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a usable partial result being flattened into ordinary success."""
    adapter = _CursorRecordingAdapter()
    adapter.outcome = AcquisitionOutcome.PARTIAL
    adapter.partial = True
    adapter.limitations = ["provider_returned_partial_page"]
    tools = WorldTools(
        store=WorldStore(tmp_path / "partial.sqlite3"),
        adapters={"cursor": adapter},
    )

    result = await tools.execute("discover_sources", {"adapter": "cursor"}, "partial")

    assert result["ok"] is True
    assert result["outcome"] == "partial"
    assert result["cards"][0]["title"] == "page-1"
    assert result["completeness"]["partial"] is True
    assert result["completeness"]["next_cursor_usable"] is True


async def test_acquisition_timeout_returns_factual_feedback_without_next_step_instruction(
    tmp_path: pytest.TempPathFactory,
) -> None:
    cancelled = asyncio.Event()
    events: list[dict[str, object]] = []

    class _SlowAdapter:
        adapter_id = "slow"
        adapter_version = "1"

        async def discover(self, request: ScanRequest) -> DiscoveryBatch:
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()
            return DiscoveryBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
            )

    store = WorldStore(tmp_path / "timeout.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={"slow": _SlowAdapter()},
        discovery_timeout_seconds=0.02,
        progress=events.append,
    )

    result = await tools.execute("discover_sources", {"adapter": "slow"}, "slow-call")

    assert cancelled.is_set()
    assert result["error"] == {
        "code": "tool_timeout",
        "stage": "discover",
        "attempts": 1,
        "same_request_can_change": True,
        "change_condition": "provider_response_before_configured_timeout",
    }
    assert result["outcome"] == "unavailable"
    assert result["limitations"] == ["tool_timeout"]
    assert [event["event"] for event in events] == ["tool_started", "tool_timed_out"]


async def test_expired_wake_deadline_prevents_adapter_dispatch(
    tmp_path: pytest.TempPathFactory,
) -> None:
    calls = 0

    class _CountingAdapter:
        adapter_id = "counting"
        adapter_version = "1"

        async def discover(self, request: ScanRequest) -> DiscoveryBatch:
            nonlocal calls
            calls += 1
            return DiscoveryBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
            )

    tools = WorldTools(
        store=WorldStore(tmp_path / "deadline.sqlite3"),
        adapters={"counting": _CountingAdapter()},
        wake_deadline_at=time.monotonic() - 1,
    )

    result = await tools.execute("discover_sources", {"adapter": "counting"}, "late-call")

    assert calls == 0
    assert result["error"] == {
        "code": "wake_deadline_exceeded",
        "stage": "before_dispatch",
        "attempts": 0,
        "same_request_can_change": False,
        "change_condition": "new_wake_dispatch_cutoff",
    }
    assert result["limitations"] == ["wake_deadline_exceeded"]


async def test_wake_dispatch_cutoff_does_not_cancel_an_in_flight_tool(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Once dispatched, the tool's own timeout governs settlement."""
    release = asyncio.Event()
    cancelled = False

    class _SettlingAdapter:
        adapter_id = "settling"
        adapter_version = "1"

        async def discover(self, request: ScanRequest) -> DiscoveryBatch:
            nonlocal cancelled
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled = True
                raise
            return DiscoveryBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
            )

    async def settle_after_cutoff() -> None:
        await asyncio.sleep(0.04)
        release.set()

    tools = WorldTools(
        store=WorldStore(tmp_path / "settling.sqlite3"),
        adapters={"settling": _SettlingAdapter()},
        wake_deadline_at=time.monotonic() + 0.02,
        discovery_timeout_seconds=0.2,
    )
    settler = asyncio.create_task(settle_after_cutoff())

    result = await tools.execute(
        "discover_sources", {"adapter": "settling"}, "settling-call"
    )
    await settler

    assert result["ok"] is True
    assert cancelled is False


async def test_agent_tool_facade_rejects_semantic_world_writes(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a research Agent retaining direct authority over the world graph."""
    tools, store = _tools(tmp_path)

    result = await tools.execute(
        "memory_commit",
        {"delta": {"objects": [{"id": "subject", "kind": "entity", "canonical_name": "Subject"}]}},
        "semantic-call",
    )

    assert "memory_commit" not in {item["function"]["name"] for item in tools.schemas()}
    assert result == {"ok": False, "limitations": ["unknown_tool:memory_commit"]}
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM world_audit").fetchone()[0] == 0


class _FailingReplayAdapter(ReplayChannelAdapter):
    def __init__(self, failures: list[AgentError]) -> None:
        super().__init__(
            adapter_id="replay",
            adapter_version="1",
            discoveries={
                "tool:transient-discover": DiscoveryBatch(
                    request_id="tool:transient-discover",
                    adapter_id="replay",
                    adapter_version="1",
                    occurrences=[_occurrence()],
                ),
                "tool:permanent-discover": DiscoveryBatch(
                    request_id="tool:permanent-discover",
                    adapter_id="replay",
                    adapter_version="1",
                    occurrences=[_occurrence()],
                ),
            },
            hydrations={},
        )
        self.failures = failures
        self.attempts = 0
        self.requests = []

    async def discover(self, request):
        self.attempts += 1
        self.requests.append(request)
        if self.failures:
            raise self.failures.pop(0)
        return await super().discover(request)


async def test_transient_source_failure_retries_once_then_returns_cards(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch one network interruption prematurely hiding an otherwise available source."""
    adapter = _FailingReplayAdapter([AgentError(ErrorCode.TOOL_TRANSIENT, "temporary")])
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(store=store, adapters={"replay": adapter})

    result = await tools.execute(
        "discover_sources",
        {"adapter": "replay", "query": "source"},
        "transient-discover",
    )

    assert result["ok"] is True
    assert [item["id"] for item in result["cards"]] == [observation_id("replay", SOURCE)]
    assert adapter.attempts == 2
    assert adapter.requests[0] == adapter.requests[1]


async def test_repeated_transient_source_failure_stops_after_two_attempts(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch source retries hammering a platform after their single retry is spent."""
    adapter = _FailingReplayAdapter(
        [
            AgentError(ErrorCode.TOOL_TRANSIENT, "temporary one"),
            AgentError(ErrorCode.TOOL_TRANSIENT, "temporary two"),
        ]
    )
    tools = WorldTools(
        store=WorldStore(tmp_path / "world.sqlite3"),
        adapters={"replay": adapter},
    )

    result = await tools.execute(
        "discover_sources",
        {"adapter": "replay", "query": "source"},
        "transient-discover",
    )

    assert result == {
        "ok": False,
        "outcome": "unavailable",
        "error": {
            "code": "TOOL_TRANSIENT",
            "stage": "discover",
            "attempts": 2,
            "same_request_can_change": True,
            "change_condition": "provider_or_network_conditions_change",
        },
        "limitations": ["adapter_failure:TOOL_TRANSIENT"],
    }
    assert adapter.attempts == 2


async def test_nontransient_source_failure_is_not_retried(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch authentication or platform limits being retried as network interruptions."""
    adapter = _FailingReplayAdapter([AgentError(ErrorCode.SOURCE_UNAVAILABLE, "authentication required")])
    tools = WorldTools(
        store=WorldStore(tmp_path / "world.sqlite3"),
        adapters={"replay": adapter},
    )

    result = await tools.execute(
        "discover_sources",
        {"adapter": "replay", "query": "source"},
        "permanent-discover",
    )

    assert result == {
        "ok": False,
        "outcome": "unavailable",
        "error": {
            "code": "SOURCE_UNAVAILABLE",
            "stage": "discover",
            "attempts": 1,
            "same_request_can_change": True,
            "change_condition": "source_access_or_provider_availability_changes",
        },
        "limitations": ["adapter_failure:SOURCE_UNAVAILABLE"],
    }
    assert adapter.attempts == 1


async def test_unknown_adapter_and_invalid_depth_return_typed_limitations(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch unsupported acquisition requests that escape the facade as exceptions."""
    tools, _ = _tools(tmp_path)
    anchor_id = await _anchor(tools, "open-seed")

    unknown = await tools.execute(
        "discover_sources", {"adapter": "missing", "query": "source"}, "missing-adapter"
    )
    invalid = await tools.execute(
        "open_source", {"observation_id": anchor_id, "depth": "unsupported"}, "bad-depth"
    )

    assert unknown["error"] == {
        "code": "unknown_adapter",
        "stage": "discover",
        "attempts": 0,
        "same_request_can_change": True,
        "change_condition": "registered_adapter_set_changes",
    }
    assert invalid["error"] == {
        "code": "invalid_arguments",
        "stage": "hydrate",
        "attempts": 0,
        "same_request_can_change": False,
        "change_condition": "request_arguments_change",
    }


class _ForgedReplayAdapter(ReplayChannelAdapter):
    async def discover(self, request):
        batch = await super().discover(request)
        return batch.model_copy(update={"adapter_id": "forged"})

    async def hydrate(self, request):
        batch = await super().hydrate(request)
        return batch.model_copy(update={"adapter_version": "forged"})


async def test_forged_batch_provenance_returns_limitations_without_persisting(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch accepted batches whose provenance differs from their registered adapter."""
    adapter = _ForgedReplayAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:forged-discover": DiscoveryBatch(
                request_id="tool:forged-discover", adapter_id="replay", adapter_version="1", occurrences=[]
            )
        },
        hydrations={
            SOURCE: ObservationBatch(
                request_id="tool:hydrate:replay:seen:anchor", adapter_id="replay", adapter_version="1"
            )
        },
    )
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(observations=[_anchor_observation("replay:seen:anchor")]), "anchor")
    tools = WorldTools(store=store, adapters={"replay": adapter})

    discovery = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "forged-discover"
    )
    hydration = await tools.execute("open_source", {"observation_id": "replay:seen:anchor"}, "forged-open")

    assert discovery["ok"] is False
    assert discovery["error"]["code"] == "adapter_provenance_mismatch"
    assert discovery["error"]["stage"] == "discover"
    assert hydration["ok"] is False
    assert hydration["error"]["code"] == "adapter_provenance_mismatch"
    assert hydration["error"]["stage"] == "hydrate"


async def test_facade_caps_replay_batches_before_cards_or_world_writes(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch an adapter batch that bypasses facade card caps before return or persistence."""
    # each fixture item needs its own source ref: the unified id merges
    # same-source observations, so a shared ref would collapse the batch
    # below the cap the test is guarding
    occurrences = [
        _occurrence(f"source-{index}").model_copy(
            update={
                "adapter_id": "public-web",
                "source_ref": f"{SOURCE}/occurrence-{index}",
            }
        )
        for index in range(CAP + 5)
    ]
    observations = [
        _observation(f"content-{index}").model_copy(update={"source_ref": f"{SOURCE}/content-{index}"})
        for index in range(CAP + 5)
    ]
    # public-web-like adapter: no native related(), but its hydrate fills
    # discovered_occurrences, so follow_related still exercises the cap path
    adapter = ReplayChannelAdapter(
        adapter_id="public-web",
        adapter_version="1",
        discoveries={
            "tool:cap-discover": DiscoveryBatch(
                request_id="tool:cap-discover",
                adapter_id="public-web",
                adapter_version="1",
                occurrences=occurrences,
            )
        },
        hydrations={
            SOURCE: ObservationBatch(
                request_id="tool:hydrate:replay:seen:anchor",
                adapter_id="public-web",
                adapter_version="1",
                observations=observations,
                discovered_occurrences=occurrences[:8],
            )
        },
    )
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(observations=[_anchor_observation("replay:seen:anchor", source_kind="public-web")]),
        "anchor",
    )
    tools = WorldTools(store=store, adapters={"public-web": adapter})

    discovery = await tools.execute(
        "discover_sources", {"adapter": "public-web", "query": "source", "limit": 500}, "cap-discover"
    )
    opened = await tools.execute("open_source", {"observation_id": "replay:seen:anchor"}, "cap-open")
    related = await tools.execute("follow_related", {"observation_id": "replay:seen:anchor"}, "cap-related")

    assert len(discovery["cards"]) == len(opened["cards"]) == len(related["cards"]) == CAP
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] <= 1 + 3 * CAP


class _ForgedOccurrenceReplayAdapter(ReplayChannelAdapter):
    async def discover(self, request):
        batch = await super().discover(request)
        forged = batch.occurrences[0].model_copy(update={"adapter_id": "forged"})
        return batch.model_copy(update={"occurrences": [forged]})

    async def hydrate(self, request):
        batch = await super().hydrate(request)
        forged = batch.discovered_occurrences[0].model_copy(update={"adapter_version": "forged"})
        return batch.model_copy(update={"discovered_occurrences": [forged]})


async def test_forged_discovery_occurrence_provenance_is_rejected_before_persistence(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a discovery batch whose individual occurrence impersonates another adapter."""
    adapter = _ForgedOccurrenceReplayAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:forged-occurrence": DiscoveryBatch(
                request_id="tool:forged-occurrence",
                adapter_id="replay",
                adapter_version="1",
                occurrences=[_occurrence()],
            )
        },
        hydrations={},
    )
    store = WorldStore(tmp_path / "world.sqlite3")
    result = await WorldTools(store=store, adapters={"replay": adapter}).execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "forged-occurrence"
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "adapter_provenance_mismatch"
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


async def test_forged_related_occurrence_provenance_is_rejected_before_persistence(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a hydration batch whose discovered relation impersonates another adapter version."""
    adapter = _ForgedOccurrenceReplayAdapter(
        adapter_id="public-web",
        adapter_version="1",
        discoveries={},
        hydrations={
            SOURCE: ObservationBatch(
                request_id="tool:hydrate:replay:seen:anchor",
                adapter_id="public-web",
                adapter_version="1",
                discovered_occurrences=[_occurrence().model_copy(update={"adapter_id": "public-web"})],
            )
        },
    )
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(observations=[_anchor_observation("replay:seen:anchor", source_kind="public-web")]),
        "anchor",
    )
    result = await WorldTools(store=store, adapters={"public-web": adapter}).execute(
        "follow_related", {"observation_id": "replay:seen:anchor"}, "forged-related"
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "adapter_provenance_mismatch"
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1


async def test_follow_related_answers_from_adapter_related_cards(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch follow_related routing through the adapter's real related() instead of an empty hydrate field."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(observations=[_anchor_observation("replay:seen:anchor")]),
        "anchor",
    )
    tools = WorldTools(store=store, adapters={"replay": _related_adapter()})

    result = await tools.execute("follow_related", {"observation_id": "replay:seen:anchor"}, "related-cards")

    assert result["ok"] is True
    assert len(result["cards"]) == CAP
    assert result["cards"][0]["title"] == "related related-0"
    assert result["cards"][0]["source_uri"] == "https://example.test/related/0"
    assert result["cards"][0]["adapter"] == {"id": "replay", "version": "1"}
    assert result["cards"][0]["engagement"] == {"views": 100}
    assert result["limitations"] == ["platform_related_recommendations_are_personalization_opaque"]
    with store.read_connection() as connection:
        related_rows = connection.execute(
            "SELECT id FROM observations WHERE source_uri LIKE 'https://example.test/related/%'"
        ).fetchall()
    assert len(related_rows) == CAP


async def test_follow_related_without_related_data_fails_explicit(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch follow_related masking an empty recommendation set with an ok card."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(observations=[_anchor_observation("replay:seen:anchor")]),
        "anchor",
    )
    tools = WorldTools(
        store=store,
        adapters={
            "replay": _EmptyRelatedReplayAdapter(
                adapter_id="replay",
                adapter_version="1",
                discoveries={},
                hydrations={},
            )
        },
    )

    result = await tools.execute("follow_related", {"observation_id": "replay:seen:anchor"}, "related-empty")

    assert result["ok"] is False
    assert result["outcome"] == "empty"
    assert result["error"]["code"] == "related_occurrences_unavailable"
    assert result["error"]["attempts"] == 1
    assert result["limitations"] == [
        "platform_related_recommendations_are_personalization_opaque",
        "related_occurrences_unavailable",
    ]


async def test_follow_related_fallback_never_masks_empty_occurrences(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch follow_related on an adapter without related content returning an empty ok card.

    Adapters whose hydrate never fills discovered_occurrences must fail with
    the typed limitation immediately — the old path burned a full TEXT
    hydrate whose discovered_occurrences stayed empty (audit item 2).
    """
    anchor_id = observation_id("replay", SOURCE)
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(observations=[_anchor_observation(anchor_id)]),
        "anchor",
    )
    adapter = _adapter()
    tools = WorldTools(store=store, adapters={"replay": adapter})

    result = await tools.execute("follow_related", {"observation_id": anchor_id}, "related-fallback")

    assert result["ok"] is False
    assert result["error"]["code"] == "related_occurrences_unavailable"
    assert result["error"]["attempts"] == 0
    # the short-circuit: no TEXT hydrate was burned for a guaranteed failure
    assert adapter.hydration_calls == []


async def test_follow_related_hydrate_fallback_serves_link_occurrences(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the short-circuit killing adapters whose hydrate fills discovered_occurrences.

    public-web has no native related() but its hydrate emits the page's
    outbound links, so follow_related must keep serving those cards from the
    hydrate instead of failing fast (audit item 2 keeps public-web's links).
    """
    anchor_id = "public-web:seen:anchor"
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id=anchor_id,
                    source_uri=SOURCE,
                    source_kind="public-web",
                    depth=ObservationDepth.SEEN,
                    observed_at=NOW,
                    metadata={"adapter_version": "1"},
                )
            ]
        ),
        "anchor",
    )
    link = SourceOccurrence(
        id="link-1",
        adapter_id="public-web",
        adapter_version="1",
        source_ref="https://example.test/link/1",
        canonical_url="https://example.test/link/1",
        title="https://example.test/link/1",
        captured_at=NOW,
    )
    adapter = ReplayChannelAdapter(
        adapter_id="public-web",
        adapter_version="1",
        discoveries={},
        hydrations={
            SOURCE: ObservationBatch(
                request_id=f"tool:hydrate:{anchor_id}",
                adapter_id="public-web",
                adapter_version="1",
                discovered_occurrences=[link],
            )
        },
    )
    tools = WorldTools(store=store, adapters={"public-web": adapter})

    result = await tools.execute("follow_related", {"observation_id": anchor_id}, "link-fallback")

    assert result["ok"] is True
    assert result["cards"][0]["source_uri"] == "https://example.test/link/1"
    assert adapter.hydration_calls == [SOURCE]


async def test_follow_related_rejects_forged_related_provenance(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a related() batch whose occurrence impersonates another adapter."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(observations=[_anchor_observation("replay:seen:anchor")]),
        "anchor",
    )
    tools = WorldTools(
        store=store,
        adapters={
            "replay": _ForgedRelatedReplayAdapter(
                adapter_id="replay",
                adapter_version="1",
                discoveries={},
                hydrations={},
            )
        },
    )

    result = await tools.execute("follow_related", {"observation_id": "replay:seen:anchor"}, "related-forged")

    assert result["ok"] is False
    assert result["error"]["code"] == "adapter_provenance_mismatch"
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1


class _StaleRequestReplayAdapter(ReplayChannelAdapter):
    async def discover(self, request):
        return (await super().discover(request)).model_copy(update={"request_id": "stale"})

    async def hydrate(self, request):
        return (await super().hydrate(request)).model_copy(update={"request_id": "stale"})


async def test_stale_batch_request_ids_are_rejected_before_persistence(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a valid but stale replay batch returned for a different internal request id."""
    adapter = _StaleRequestReplayAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:stale-discover": DiscoveryBatch(
                request_id="tool:stale-discover",
                adapter_id="replay",
                adapter_version="1",
                occurrences=[_occurrence()],
            )
        },
        hydrations={
            SOURCE: ObservationBatch(
                request_id="tool:hydrate:replay:seen:anchor", adapter_id="replay", adapter_version="1"
            )
        },
    )
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(observations=[_anchor_observation("replay:seen:anchor")]), "anchor")
    tools = WorldTools(store=store, adapters={"replay": adapter})

    discovery = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "stale-discover"
    )
    hydration = await tools.execute("open_source", {"observation_id": "replay:seen:anchor"}, "stale-open")

    assert discovery["error"]["code"] == hydration["error"]["code"] == "adapter_provenance_mismatch"
    assert discovery["error"]["stage"] == "discover"
    assert hydration["error"]["stage"] == "hydrate"


class _RotatingReplayAdapter(ReplayChannelAdapter):
    """Replay adapter that serves one configured occurrence per discover call."""

    def __init__(
        self,
        occurrences: list[SourceOccurrence],
        request_ids: list[str],
    ) -> None:
        super().__init__(
            adapter_id="replay",
            adapter_version="1",
            discoveries={
                f"tool:{request_id}": DiscoveryBatch(
                    request_id=f"tool:{request_id}",
                    adapter_id="replay",
                    adapter_version="1",
                    occurrences=[occurrences[0]],
                )
                for request_id in request_ids
            },
            hydrations={},
        )
        self._occurrences = list(occurrences)

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Replay the fixture then rotate in the next configured occurrence."""
        batch = await super().discover(request)
        if not self._occurrences:
            return batch
        occurrence = self._occurrences.pop(0)
        return batch.model_copy(update={"occurrences": [occurrence]})


async def test_discover_and_hydrate_share_one_source_derived_id(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the dual-track id spaces leaving two rows per source (T13)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:id-unity-discover": DiscoveryBatch(
                request_id="tool:id-unity-discover",
                adapter_id="replay",
                adapter_version="1",
                occurrences=[_occurrence()],
            )
        },
        hydrations={
            SOURCE: ObservationBatch(
                request_id=f"tool:hydrate:{observation_id('replay', SOURCE)}",
                adapter_id="replay",
                adapter_version="1",
                observations=[_observation()],
            )
        },
    )
    tools = WorldTools(store=store, adapters={"replay": adapter})

    discovered = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "id-unity-discover"
    )
    card_id = discovered["cards"][0]["id"]

    assert card_id == observation_id("replay", SOURCE)
    assert "seen" not in card_id
    assert "occurrence-" not in card_id

    opened = await tools.execute("open_source", {"observation_id": card_id}, "id-unity-open")

    assert opened["ok"] is True
    assert opened["cards"][0]["id"] == card_id
    with store.read_connection() as connection:
        rows = connection.execute("SELECT id, depth FROM observations").fetchall()
    assert [row["id"] for row in rows] == [card_id]
    assert rows[0]["depth"] == "content"


async def test_second_discover_same_source_upgrades_content_not_rows(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch re-discoveries of one source piling up dead SEEN rows (T13)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    second = _occurrence().model_copy(
        update={
            "title": "A discovered source v2",
            "metadata": {"snippet": "fresh discovery excerpt"},
        }
    )
    adapter = _RotatingReplayAdapter([_occurrence(), second], ["upgrade-1", "upgrade-2"])
    tools = WorldTools(store=store, adapters={"replay": adapter})

    r1 = await tools.execute("discover_sources", {"adapter": "replay", "query": "source"}, "upgrade-1")
    r2 = await tools.execute("discover_sources", {"adapter": "replay", "query": "source"}, "upgrade-2")

    obs_id = observation_id("replay", SOURCE)
    assert r1["cards"][0]["id"] == r2["cards"][0]["id"] == obs_id
    with store.read_connection() as connection:
        rows = connection.execute("SELECT id, title, excerpt FROM observations").fetchall()
    assert len(rows) == 1
    assert rows[0]["title"] == "A discovered source v2"
    assert rows[0]["excerpt"] == "fresh discovery excerpt"


async def test_already_opened_hits_the_hydrated_discovery_card(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch discovery cards re-hydrating a source that was already opened (T13)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    adapter = _adapter()
    tools = WorldTools(store=store, adapters={"replay": adapter})

    discovered = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "discover-call"
    )
    card_id = discovered["cards"][0]["id"]
    first_open = await tools.execute("open_source", {"observation_id": card_id}, "already-open-1")
    assert first_open["ok"] is True

    second_open = await tools.execute("open_source", {"observation_id": card_id}, "already-open-2")

    assert second_open["already_opened"] is True
    assert second_open["cards"][0]["id"] == card_id
    assert len(adapter.hydration_calls) == 1  # the second open served from store


async def test_unified_id_supports_assertion_through_committer(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the p2d hole: supports assertions citing the discovery id being dropped."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="BLG")],
        ),
        "seed-object",
    )
    adapter = _FullBodyAdapter()  # full reads write a body and upgrade depth
    tools = WorldTools(store=store, adapters={"counting": adapter})

    discovered = await tools.execute(
        "discover_sources", {"adapter": "counting", "query": "source"}, "evidence-discover"
    )
    obs_id = discovered["cards"][0]["id"]
    opened = await tools.execute("open_source", {"observation_id": obs_id, "depth": "full"}, "evidence-open")
    assert opened["ok"] is True

    proposal = CognitionDeltaProposal(
        assertions=[
            AssertionProposal(
                subject=GraphRef(memory_id="obj-1"),
                predicate="related_to",
                literal="confirms the full text",
                epistemic_role=EpistemicRole.FACT,
                confidence=0.8,
                evidence=[EvidenceInput(observation_id=obs_id, role="supports")],
            )
        ]
    )
    receipt = ProposalCommitter(store).commit(proposal, "proposal-unified-id")

    assert receipt.omitted_assertion_indexes == []
    assert receipt.evidence_missing_assertion_indexes == []
    assert len(receipt.commit.assertion_ids) == 1
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT role FROM assertion_evidence WHERE observation_id = ?", (obs_id,)
        ).fetchone()
    assert row["role"] == "supports"  # not downgraded to context or omitted
