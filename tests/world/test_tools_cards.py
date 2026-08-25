# Split from tests/world/test_tools.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from leave_information_bubble.channels import (
    DiscoveryBatch,
    ObservationBatch,
    ReplayChannelAdapter,
    SourceOccurrence,
)
from leave_information_bubble.models.epistemics import (
    AccessDepth,
    ObservationModality,
    SourceObservation,
)
from leave_information_bubble.world import (
    CognitiveDelta,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)
from leave_information_bubble.world.store import observation_id
from leave_information_bubble.world.tools import WorldTools
from tests.world._tools_helpers import (
    NOW,
    PUBLISHED,
    SOURCE,
    _occurrence,
    _tools,
)
from tests.world.test_tools_discovery import (
    _CursorRecordingAdapter,
    _ScanRecordingAdapter,
)
from tests.world.test_tools_open_depth import (
    _CountingHydrateAdapter,
    _related_adapter,
)


async def test_discover_cards_surface_engagement_counters(tmp_path: pytest.TempPathFactory) -> None:
    """Catch discovery cards dropping adapter-collected engagement counters (B2-3)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    occurrence = SourceOccurrence(
        id="source-1",
        adapter_id="replay",
        adapter_version="1",
        source_ref=SOURCE,
        canonical_url=SOURCE,
        title="A hot source",
        source_published_at=PUBLISHED,
        captured_at=NOW,
        metadata={
            "snippet": "discovery excerpt",
            "engagement": {
                "views": 123_456,
                "comments": 89,
                "realtime_reactions": 12,
                "likes": None,
            },
        },
    )
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:discover-call": DiscoveryBatch(
                request_id="tool:discover-call",
                adapter_id="replay",
                adapter_version="1",
                occurrences=[occurrence],
            )
        },
        hydrations={},
    )
    tools = WorldTools(store=store, adapters={"replay": adapter})

    result = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "discover-call"
    )

    assert result["cards"][0]["engagement"] == {
        "views": 123_456,
        "comments": 89,
        "realtime_reactions": 12,
    }
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT metadata_json FROM observations WHERE id = ?",
            (observation_id("replay", SOURCE),),
        ).fetchone()
    assert row is not None
    assert json.loads(row["metadata_json"])["engagement"] == {
        "views": 123_456,
        "comments": 89,
        "realtime_reactions": 12,
    }


async def test_discover_cards_preserve_bounded_selection_and_field_limitations(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The Agent can see why a bounded community listing returned this card."""
    occurrence = SourceOccurrence(
        id="selection-source",
        adapter_id="hupu",
        adapter_version="1",
        source_ref="https://bbs.hupu.com/selection",
        canonical_url="https://bbs.hupu.com/selection",
        title="JDG vs AL discussion",
        captured_at=NOW,
        metadata={
            "provider": "hupu_public_board",
            "capability_role": "community_stream",
            "board": "lol",
            "board_page": 2,
            "board_rank": 4,
            "query_relevance_rank": 1,
            "query_matched_terms": ["jdg", "al"],
            "query_match_is_client_side": True,
            "field_limitations": [
                "thread_listing_is_discovery_only",
                "community_thread_is_not_event_fact_authority",
            ],
        },
    )
    adapter = ReplayChannelAdapter(
        adapter_id="hupu",
        adapter_version="1",
        discoveries={
            "tool:selection": DiscoveryBatch(
                request_id="tool:selection",
                adapter_id="hupu",
                adapter_version="1",
                occurrences=[occurrence],
            )
        },
        hydrations={},
    )
    store = WorldStore(tmp_path / "selection.sqlite3")

    result = await WorldTools(store=store, adapters={"hupu": adapter}).execute(
        "search_sources", {"adapter": "hupu", "query": "JDG AL"}, "selection"
    )

    card = result["cards"][0]
    assert card["selection"] == {
        "provider": "hupu_public_board",
        "capability_role": "community_stream",
        "surface": "lol",
        "page": 2,
        "original_rank": 4,
        "relevance_rank": 1,
        "matched_terms": ["jdg", "al"],
        "match_method": "bounded_client_side",
    }
    assert card["limitations"] == [
        "thread_listing_is_discovery_only",
        "community_thread_is_not_event_fact_authority",
    ]
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT metadata_json FROM observations WHERE id = ?", (card["id"],)
        ).fetchone()
    assert row is not None
    persisted = json.loads(row["metadata_json"])
    assert persisted["selection"] == card["selection"]
    assert persisted["limitations"] == card["limitations"]


async def test_discover_cards_omit_engagement_when_adapter_has_none(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a fabricated zero-valued engagement key when the adapter reports none (B2-3)."""
    tools, _store = _tools(tmp_path)

    result = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "discover-call"
    )

    assert "engagement" not in result["cards"][0]


async def test_discover_preserves_verified_content_revision_for_refresh(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Only an adapter-verified fingerprint is retained for later comparison."""
    occurrence = _occurrence().model_copy(update={"content_hash": "a" * 64, "content_hash_verified": True})
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:revision": DiscoveryBatch(
                request_id="tool:revision", adapter_id="replay", adapter_version="1", occurrences=[occurrence]
            )
        },
        hydrations={},
    )
    store = WorldStore(tmp_path / "world.sqlite3")
    await WorldTools(store=store, adapters={"replay": adapter}).execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "revision"
    )
    with store.read_connection() as connection:
        row = connection.execute("SELECT metadata_json FROM observations").fetchone()
    metadata = json.loads(row["metadata_json"])
    assert metadata["content_hash_verified"] is True
    assert metadata["source_revision"] == "a" * 64
    assert metadata["source_revision_kind"] == "verified_content_hash"


async def test_hydrated_cards_surface_engagement_from_stats_and_comment_count(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the hydrate path dropping raw stat counters from agent-visible cards (B2-3)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id=observation_id("replay", SOURCE),
                    source_uri=SOURCE,
                    source_kind="replay",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=NOW,
                    metadata={"adapter_version": "1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={},
        hydrations={
            SOURCE: ObservationBatch(
                request_id=f"tool:hydrate:{observation_id('replay', SOURCE)}",
                adapter_id="replay",
                adapter_version="1",
                observations=[
                    SourceObservation(
                        id="content-1",
                        source_ref=SOURCE,
                        modality=ObservationModality.DOCUMENT_TEXT,
                        access_depth=AccessDepth.CONTENT_TEXT,
                        excerpt="bounded content excerpt",
                        location=SOURCE,
                        acquisition_method="replay_open",
                        captured_at=NOW,
                        metadata={
                            "title": "Opened source",
                            "published_at": PUBLISHED.isoformat(),
                            "stats": {
                                "play": 9_999,
                                "danmaku": 7,
                                "reply": 42,
                                "like": 100,
                                "coin": None,
                                "favorite": 5,
                                "share": 3,
                            },
                        },
                    ),
                    SourceObservation(
                        id="content-2",
                        source_ref=SOURCE,
                        modality=ObservationModality.COMMENT,
                        access_depth=AccessDepth.REACTIONS,
                        excerpt="comment sample",
                        location=SOURCE,
                        acquisition_method="replay_comments",
                        captured_at=NOW,
                        metadata={"comment_count": 25},
                    ),
                ],
            )
        },
    )
    tools = WorldTools(store=store, adapters={"replay": adapter})

    result = await tools.execute(
        "open_source", {"observation_id": observation_id("replay", SOURCE)}, "open-call"
    )

    assert result["outcome"] == "success"
    assert result["completeness"] == {
        "returned": 2,
        "limit": 5,
        "partial": False,
        "truncated": False,
        "next_cursor_usable": False,
        "physical_calls": 0,
    }
    # content-1 (document text) carries the source's main id; content-2 (a
    # sampled comment) keeps its own content-derived sub-id (fix F2)
    comment_id = (
        f"{observation_id('replay', SOURCE)}-comment-{hashlib.sha256(b'comment sample').hexdigest()[:32]}"
    )
    assert [item["id"] for item in result["cards"]] == [
        observation_id("replay", SOURCE),
        comment_id,
    ]
    assert result["cards"][0]["engagement"] == {
        "views": 9_999,
        "likes": 100,
        "comments": 42,
        "realtime_reactions": 7,
        "saves": 5,
        "shares": 3,
    }
    assert result["cards"][1]["engagement"] == {"comment_count": 25}
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT id, depth, metadata_json FROM observations WHERE id = ?",
            (observation_id("replay", SOURCE),),
        ).fetchone()
        comment_row = connection.execute(
            "SELECT id, depth FROM observations WHERE id = ?", (comment_id,)
        ).fetchone()
    assert row is not None
    assert row["depth"] == "content"
    assert json.loads(row["metadata_json"])["engagement"] == {
        "views": 9_999,
        "likes": 100,
        "comments": 42,
        "realtime_reactions": 7,
        "saves": 5,
        "shares": 3,
    }
    assert json.loads(row["metadata_json"])["observed"] == ["content", "comments"]
    assert comment_row is not None
    assert comment_row["depth"] == "discussion"


async def test_already_opened_card_serves_stored_engagement(tmp_path: pytest.TempPathFactory) -> None:
    """Catch the store-served card dropping previously committed engagement counters (B2-3)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/123",
                    source_kind="hupu",
                    title="IG news",
                    excerpt="excerpt",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime(2026, 8, 4, tzinfo=UTC),
                    metadata={
                        "adapter_version": "v1",
                        "engagement": {"views": 5_000, "replies": 30},
                    },
                )
            ]
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={"hupu": _CountingHydrateAdapter()})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1"}, "c1")

    assert result["already_opened"] is True
    assert result["outcome"] == "success"
    assert result["completeness"]["physical_calls"] == 0
    assert result["cards"][0]["engagement"] == {"views": 5_000, "replies": 30}


async def test_already_opened_card_omits_engagement_when_none_stored(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a fabricated engagement key on store-served cards with no signal data (B2-3)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/123",
                    source_kind="hupu",
                    title="IG news",
                    excerpt="excerpt",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime(2026, 8, 4, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={"hupu": _CountingHydrateAdapter()})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1"}, "c1")

    assert result["already_opened"] is True
    assert "engagement" not in result["cards"][0]


@pytest.mark.parametrize("metadata_json", ["not-json", "[]"])
async def test_already_opened_treats_malformed_metadata_as_empty(
    tmp_path: pytest.TempPathFactory, metadata_json: str
) -> None:
    """Malformed stored metadata cannot crash an already-opened material card."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/123",
                    source_kind="hupu",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime(2026, 8, 4, tzinfo=UTC),
                    metadata={"adapter_version": "v1", "observed": ["content"]},
                )
            ]
        ),
        "seed-malformed-metadata",
    )
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE observations SET metadata_json = ? WHERE id = ?", (metadata_json, "hupu:obs-1")
        )

    result = await WorldTools(store=store, adapters={}).execute(
        "open_source", {"observation_id": "hupu:obs-1"}, "malformed-metadata"
    )

    assert result["ok"] is True
    assert result["already_opened"] is True
    assert result["cards"][0]["limitations"] == []


async def test_already_opened_card_carries_stored_age_signal(tmp_path: pytest.TempPathFactory) -> None:
    """Catch store-served cards hiding how old an observation is (B2-8, audit A6).

    The already_opened short-circuit is a zero-network snapshot, so the model
    must see the storage age ("S16 标题" case: Bilibili retitled the video and
    the store kept the old title). A card observed 3 days ago reports
    ``stored_days_ago == 3`` and a limitation naming the age and the re-scan
    path (discover_sources/search_sources re-fetch and refresh the stored row).
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-stale",
                    source_uri="https://bbs.hupu.com/789",
                    source_kind="hupu",
                    title="IG news",
                    excerpt="excerpt",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime.now(UTC) - timedelta(days=3),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={"hupu": _CountingHydrateAdapter()})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-stale"}, "c1")

    assert result["already_opened"] is True
    assert result["cards"][0]["stored_days_ago"] == 3
    assert result["limitations"][0] == "observation already opened at depth=content"
    assert "stored 3 days ago" in result["limitations"][1]
    assert result["limitations"][1] == "stored 3 days ago; source content may have changed"


async def test_already_opened_fresh_observation_reports_zero_age_without_hint(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the freshness signal disappearing or a stale hint firing on fresh reads (B2-8).

    The structured ``stored_days_ago`` field stays on every store-served card
    (0 for a same-day observation), but the prose staleness hint only fires
    when the observation is at least a day old — a fresh card is not noise.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-fresh",
                    source_uri="https://bbs.hupu.com/999",
                    source_kind="hupu",
                    title="IG news",
                    excerpt="excerpt",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime.now(UTC) - timedelta(hours=2),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={"hupu": _CountingHydrateAdapter()})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-fresh"}, "c2")

    assert result["already_opened"] is True
    assert result["cards"][0]["stored_days_ago"] == 0
    assert result["limitations"] == ["observation already opened at depth=content"]


def test_memory_search_description_qualifies_empty_results_and_candidates() -> None:
    """The search description carries the empty-result caveat and the graded layers."""
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}
    search = descriptions["memory_search"]

    assert "An empty result does not establish that no other memory exists." in search
    assert "candidates" in search
    assert "identity_alias_exact" in search
    assert "canonical_exact" in search
    assert "name_usage" in search
    assert "legacy_name" in search
    assert "identity_authority" in search
    assert "possible_match" in search
    assert "text_match" in search
    assert "domain hints" in search


def test_all_tool_descriptions_are_differentiated_not_template() -> None:
    """Catch C4 regressions: no tool description may fall back to the old template."""
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}
    expected = {
        "memory_recent",
        "memory_search",
        "memory_read",
        "memory_compare",
        "memory_expand",
        "memory_evidence",
        "memory_inquiries",
        "memory_changes",
        "memory_overview",
        "claim_inquiry",
        "release_inquiry",
        "discover_sources",
        "search_sources",
        "open_source",
        "sample_discussion",
        "inspect_media",
        "follow_related",
        "digest_observation",
        "propose_inquiry",
        "log_inquiry_point",
        "graph_patch",
        "graph_inspect",
        "graph_diff",
    }
    assert set(descriptions) == expected
    template = "with bounded cards, provenance, and explicit limitations."
    for name, description in descriptions.items():
        assert template not in description
        assert description != f"{name.replace('_', ' ')} {template}"
        assert len(description) >= 120  # differentiated capability boundary, not a one-liner


def test_external_acquisition_descriptions_do_not_impose_memory_first_order() -> None:
    """Acquisition may precede recall unless durable identity work requires it."""
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}
    external = {
        "discover_sources",
        "search_sources",
        "open_source",
        "sample_discussion",
        "inspect_media",
        "follow_related",
    }
    for name in external:
        lowered = descriptions[name].casefold()
        assert "check your memory first" not in lowered, name
        assert "before you search externally" not in lowered, name
        assert "before you scan external" not in lowered, name

    memory_search = descriptions["memory_search"].casefold()
    assert "reusable durable identity" in memory_search
    assert "duplicate" in memory_search
    assert "update" in memory_search


def test_memory_descriptions_expose_bounded_recent_and_optional_associative_search() -> None:
    """Recall descriptions prevent both recent-slice amnesia and memory-first workflows."""
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}

    recent = descriptions["memory_recent"].casefold()
    assert "bounded" in recent
    assert "not all memory" in recent
    assert "absent item may still exist" in recent
    assert "not a required wake-opening step or an agenda" in recent
    assert "confirmed" not in recent

    inquiries = descriptions["memory_inquiries"].casefold()
    assert "current stimulus" in inquiries
    assert "continuing deep pull" in inquiries
    assert "not a backlog or default startup agenda" in inquiries
    assert "remembered judgments" in inquiries

    search = descriptions["memory_search"].casefold()
    for purpose in ("update", "duplicate", "conflict", "reusable durable identity"):
        assert purpose in search
    assert "optional" in search
    assert "not a required step before every external search" in search
    assert "confirmed" not in search

    expanded = descriptions["memory_expand"].casefold()
    assert "bounded local slice" in expanded
    assert "not a complete object history" in expanded
    assert "ego" in expanded
    assert "direct neighbors" in expanded
    assert "edge predicates" in expanded
    assert "current status" in expanded
    assert "recent events" in expanded
    assert "map" in expanded
    assert "zoom" in expanded

    evidence = descriptions["memory_evidence"].casefold()
    for boundary in ("bounded", "evidence roles", "depth", "source kind", "reliability", "limitation"):
        assert boundary in evidence
    assert "does not return source bodies" in evidence
    assert "already-surfaced remembered judgment" in evidence
    assert "current interpretation or correction" in evidence
    assert "not a default prerequisite" in evidence
    assert "need not be called for every judgment" in evidence

    changes = descriptions["memory_changes"].casefold()
    assert "since is omitted" in changes
    assert "latest proposal attempt" in changes
    assert "if no proposal attempt exists" in changes
    assert "all world audit rows" in changes
    assert "each returned list remains capped" in changes


def test_description_spot_checks_purpose_timing_and_neighbor_distinctions() -> None:
    """Catch C4: spot-checked tools name their purpose, timing, and adjacent tools."""
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}
    # open_source vs sample_discussion vs inspect_media: depth semantics
    open_source = descriptions["open_source"]
    assert "preview (metadata and excerpt)" in open_source
    assert "full (full-text body)" in open_source
    assert "depth" in open_source
    assert "Unlike sample_discussion and inspect_media" in open_source
    # memory_search vs memory_recent vs memory_changes: read semantics
    memory_search = descriptions["memory_search"]
    assert "full-text query" in memory_search
    assert "Unlike memory_recent" in memory_search
    assert "unlike memory_expand" in memory_search  # mid-sentence continuation
    assert "before searching external sources" not in memory_search
    # discover_sources vs search_sources: both scan the same surfaces with the
    # adapter's default ordering; discover is the query-optional breadth survey
    discover = descriptions["discover_sources"]
    assert "breadth survey" in discover
    assert "Unlike search_sources" in discover
    assert "default" in discover and "relevance" in discover
    search = descriptions["search_sources"]
    assert "query" in search and "Unlike discover_sources" in search
    assert "every adapter supports targeted retrieval" not in search
    # the old false claim is gone: no adapter surfaces freshness/attention
    # ordering (bilibili is always relevance-ranked search)
    assert "freshness" not in discover and "freshness" not in search


def test_acquisition_and_digest_descriptions_bound_search_depth_and_retries() -> None:
    """Descriptions expose real acquisition limits without creating a coverage workflow."""
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}

    discover = descriptions["discover_sources"].casefold()
    assert "query anchors" in discover
    assert "whole current surface" in discover
    assert "candidate" in discover and "not an assertion" in discover
    assert "access or content limitations" in discover

    search = descriptions["search_sources"].casefold()
    assert "does not by itself provide a broad view" in search
    assert "material, not assertions" in search
    assert "access or content limitations" in search

    opened = descriptions["open_source"].casefold()
    assert "need not be opened beyond" in opened
    assert "do not imply" in opened
    assert "supports a judgment" in opened

    digest = descriptions["digest_observation"].casefold()
    assert "does not require every candidate or point to become durable cognition" in digest
    assert "never to request more source content" in digest
    assert "honestly attributed, confidence-calibrated revisable cognition" in digest
    assert "must stay unresolved, not become an inferred fact" not in digest

    for name in ("discover_sources", "search_sources", "open_source"):
        lowered = descriptions[name].casefold()
        for scheduler in ("retrying blindly", "pivot or leave", "must open", "open every"):
            assert scheduler not in lowered


def test_scan_schemas_expose_adapter_enum_with_registered_platforms() -> None:
    """Schema-only callers retain the documented fallback adapter enum."""
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in tools.schemas()}
    expected = {"type": "string", "enum": ["bilibili", "nga", "hupu", "public-web"]}
    for name in ("discover_sources", "search_sources"):
        assert schemas[name]["properties"]["adapter"] == expected, name
        assert "adapter" in schemas[name]["required"], name
    # discover_sources is the targetless breadth survey: query optional; only
    # search_sources (targeted retrieval) keeps its query required
    assert "query" not in schemas["discover_sources"]["required"]
    assert "query" in schemas["search_sources"]["required"]


def test_scan_descriptions_list_available_adapters_without_content_labels() -> None:
    """Schema-only callers see fallback adapters as unclassified, never guessed."""
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}
    for name in ("discover_sources", "search_sources"):
        description = descriptions[name]
        for adapter_id in ("bilibili", "nga", "hupu", "public-web"):
            assert adapter_id in description
        assert "Available adapters:" not in description
        assert "check your memory first" not in description.casefold(), name
        assert "Unlike" in description, name
        # user ruling: no content-label claims attached to any platform, so
        # only the bare lowercase ids appear (no brand-name prose)
        assert "NGA" not in description and "Hupu" not in description, name
    assert "capability_unclassified" in descriptions["discover_sources"]
    assert "Targeted-search adapters allowed by this schema" in descriptions["search_sources"]


def test_scan_schemas_and_descriptions_expose_only_registered_adapters() -> None:
    """Configured tool facades must not advertise adapters they cannot dispatch."""
    tools = WorldTools(
        store=WorldStore(":memory:"),
        adapters={"public-web": _related_adapter(), "hupu": _related_adapter()},
    )
    functions = {item["function"]["name"]: item["function"] for item in tools.schemas()}

    for name in ("discover_sources", "search_sources"):
        assert functions[name]["parameters"]["properties"]["adapter"] == {
            "type": "string",
            "enum": ["public-web", "hupu"],
        }
        description = functions[name]["description"]
        assert "public-web" in description
        assert "hupu" in description
        assert "bilibili" not in description and "nga" not in description
    assert "capability_unclassified" in functions["discover_sources"]["description"]


def test_scan_descriptions_render_registered_adapter_capabilities() -> None:
    tools = WorldTools(
        store=WorldStore(":memory:"),
        adapters={"hupu": _CursorRecordingAdapter(), "bilibili": _ScanRecordingAdapter()},
    )
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}

    discover = descriptions["discover_sources"]
    assert "bilibili" in discover and "platform_search" in discover
    assert "query required" in discover
    assert "hupu" in discover and "bounded_board" in discover
    assert "bounded_rerank_hint" in discover
    assert "targeted search" not in discover.split("hupu", 1)[1]
    search = descriptions["search_sources"]
    assert "bilibili" in search
    assert "hupu" not in search


def _comment_observation(content: str) -> SourceObservation:
    return SourceObservation(
        id=f"comment-{hashlib.sha256(content.encode()).hexdigest()[:8]}",
        source_ref=SOURCE,
        modality=ObservationModality.COMMENT,
        access_depth=AccessDepth.REACTIONS,
        excerpt=content,
        location=SOURCE,
        acquisition_method="replay_comments",
        captured_at=NOW,
        metadata={},
    )


async def test_comments_keep_per_item_sub_rows_idempotent(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch comment granularity collapsing into the source row (fix F2)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    comments = [_comment_observation(f"sampled comment number {index}") for index in range(40)]
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:comments-discover": DiscoveryBatch(
                request_id="tool:comments-discover",
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
                observations=comments,
            )
        },
    )
    tools = WorldTools(store=store, adapters={"replay": adapter})
    discovered = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "comments-discover"
    )
    card_id = discovered["cards"][0]["id"]

    first = await tools.execute("sample_discussion", {"observation_id": card_id}, "comments-1")
    second = await tools.execute("sample_discussion", {"observation_id": card_id}, "comments-2")

    assert first["ok"] is True
    assert len(first["cards"]) == 5  # the card cap bounds one call, not the granularity
    ids = [item["id"] for item in first["cards"]]
    assert len(set(ids)) == 5  # distinct comment sub-ids, not one merged row
    assert all(identifier.startswith(f"{observation_id('replay', SOURCE)}-comment-") for identifier in ids)
    expected_first = (
        f"{observation_id('replay', SOURCE)}-comment-"
        f"{hashlib.sha256(b'sampled comment number 0').hexdigest()[:32]}"
    )
    assert ids[0] == expected_first
    with store.read_connection() as connection:
        rows = connection.execute(
            "SELECT id FROM observations WHERE id LIKE ?", (f"{observation_id('replay', SOURCE)}-comment-%",)
        ).fetchall()
    assert len(rows) == 5
    # re-fetching the same comments hits the same sub-rows: idempotent upsert
    assert second["ok"] is True
    assert [item["id"] for item in second["cards"]] == ids
    with store.read_connection() as connection:
        after = connection.execute(
            "SELECT id FROM observations WHERE id LIKE ?", (f"{observation_id('replay', SOURCE)}-comment-%",)
        ).fetchall()
    assert len(after) == 5


async def test_transcript_only_source_does_not_short_circuit_discussion(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch cross-dimension short-circuit: subtitles read must not block comments (fix F1)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    transcript = SourceObservation(
        id="segment-1",
        source_ref=SOURCE,
        modality=ObservationModality.TRANSCRIPT,
        access_depth=AccessDepth.CONTENT_TEXT,
        excerpt="00:01 segment 1 text",
        location="1.00-2.00",
        acquisition_method="replay_subtitles",
        captured_at=NOW,
        metadata={},
    )
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:f1-discover": DiscoveryBatch(
                request_id="tool:f1-discover",
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
                observations=[transcript],
            )
        },
    )
    tools = WorldTools(store=store, adapters={"replay": adapter})
    discovered = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "f1-discover"
    )
    card_id = discovered["cards"][0]["id"]

    preview = await tools.execute("open_source", {"observation_id": card_id}, "f1-preview")
    discussion = await tools.execute("sample_discussion", {"observation_id": card_id}, "f1-discussion")

    assert preview["ok"] is True
    assert "-transcript-" in preview["cards"][0]["id"]
    with store.read_connection() as connection:
        row = connection.execute("SELECT depth FROM observations WHERE id = ?", (card_id,)).fetchone()
    assert row["depth"] == "seen"  # subtitles never merged into the main row
    assert discussion["ok"] is True  # comments were never read: hydrate, not short-circuit
    assert "already_opened" not in discussion


async def test_comments_read_source_short_circuits_discussion(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch already_opened missing a source whose comments were already read (fix F1)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    metadata_observation = SourceObservation(
        id="meta-1",
        source_ref=SOURCE,
        modality=ObservationModality.METADATA,
        access_depth=AccessDepth.METADATA,
        excerpt="A discovered source",
        location=SOURCE,
        acquisition_method="replay_video_info",
        captured_at=NOW,
        metadata={"title": "A discovered source"},
    )
    comment = _comment_observation("sampled comment number 0")
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:f1b-discover": DiscoveryBatch(
                request_id="tool:f1b-discover",
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
                observations=[metadata_observation, comment],
            )
        },
    )
    tools = WorldTools(store=store, adapters={"replay": adapter})
    discovered = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "f1b-discover"
    )
    card_id = discovered["cards"][0]["id"]

    first = await tools.execute("sample_discussion", {"observation_id": card_id}, "f1b-first")
    second = await tools.execute("sample_discussion", {"observation_id": card_id}, "f1b-second")

    assert first["ok"] is True
    assert second["already_opened"] is True
    assert len(adapter.hydration_calls) == 1  # the re-open served from store
