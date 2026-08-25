# Split from tests/world/test_tools.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from leave_information_bubble.channels import (
    AcquisitionOutcome,
    DiscoveryBatch,
    HydrationDepth,
    HydrationRequest,
    ObservationBatch,
    ReplayChannelAdapter,
    ScanRequest,
    SourceOccurrence,
)
from leave_information_bubble.models.epistemics import (
    AccessDepth,
    ObservationModality,
    SourceObservation,
)
from leave_information_bubble.runtime.errors import AgentError, ErrorCode
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
from leave_information_bubble.world.materials import (
    BodyEnvelope,
    MaterialPartInput,
    build_body_envelope,
    parse_stored_body,
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


async def test_open_source_uses_durable_observation_id_and_persists_content_provenance(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch hydration that trusts caller source data or returns a non-citable content identity."""
    tools, store = _tools(tmp_path)
    anchor_id = await _anchor(tools, "open-seed")

    result = await tools.execute("open_source", {"observation_id": anchor_id}, "open-call")

    # hydrate writes the SAME durable id the discovery card carried (T13)
    assert result["cards"][0]["id"] == anchor_id
    assert result["cards"][0]["excerpt"] == "bounded content excerpt"
    assert result["cards"][0]["published_at"] == PUBLISHED.isoformat()
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT id, depth, source_published_at, metadata_json FROM observations WHERE id = ?",
            (result["cards"][0]["id"],),
        ).fetchone()
    assert row["depth"] == "content"
    assert row["source_published_at"] == PUBLISHED.isoformat()
    assert "raw" not in row["metadata_json"]


def test_open_source_schema_requires_full_depth_when_refresh_is_true(tmp_path) -> None:
    """The provider schema advertises refresh only under the full-depth contract."""
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={})
    schema = next(
        item["function"]["parameters"]
        for item in tools.schemas()
        if item["function"]["name"] == "open_source"
    )

    assert schema["properties"]["refresh"] == {"type": "boolean", "default": False}
    assert schema["allOf"] == [
        {
            "if": {
                "properties": {"refresh": {"const": True}},
                "required": ["refresh"],
            },
            "then": {
                "properties": {"depth": {"const": "full"}},
                "required": ["depth"],
            },
        }
    ]


def _anchor_observation(identifier: str, *, source_kind: str = "replay") -> ObservationInput:
    return ObservationInput(
        id=identifier,
        source_uri=SOURCE,
        source_kind=source_kind,
        depth=ObservationDepth.SEEN,
        observed_at=NOW,
        metadata={"adapter_version": "1"},
    )


def _related_occurrence(identifier: str, ref: str) -> SourceOccurrence:
    return SourceOccurrence(
        id=identifier,
        adapter_id="replay",
        adapter_version="1",
        source_ref=ref,
        canonical_url=ref,
        title=f"related {identifier}",
        captured_at=NOW,
        metadata={"engagement": {"views": 100}},
    )


class _RelatedReplayAdapter(ReplayChannelAdapter):
    """Replay adapter whose related() answers platform-style related batches."""

    async def related(
        self,
        *,
        request_id: str,
        source_ref: str,
        limit: int = 10,
    ) -> DiscoveryBatch:
        """Replay one bounded related batch with matching provenance."""
        del source_ref
        return DiscoveryBatch(
            request_id=request_id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            occurrences=[
                _related_occurrence(
                    f"related-{index}",
                    f"https://example.test/related/{index}",
                )
                for index in range(min(limit, CAP))
            ],
            limitations=["platform_related_recommendations_are_personalization_opaque"],
        )


class _EmptyRelatedReplayAdapter(_RelatedReplayAdapter):
    """Replay adapter whose related() returns no occurrences at all."""

    async def related(
        self,
        *,
        request_id: str,
        source_ref: str,
        limit: int = 10,
    ) -> DiscoveryBatch:
        """Replay an empty related batch with a platform disclosure."""
        del source_ref, limit
        return DiscoveryBatch(
            request_id=request_id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            occurrences=[],
            partial=True,
            limitations=["platform_related_recommendations_are_personalization_opaque"],
        )


class _ForgedRelatedReplayAdapter(_RelatedReplayAdapter):
    """Replay adapter whose related() occurrence impersonates another adapter."""

    async def related(
        self,
        *,
        request_id: str,
        source_ref: str,
        limit: int = 10,
    ) -> DiscoveryBatch:
        """Replay a related batch whose first occurrence carries forged provenance."""
        batch = await super().related(
            request_id=request_id,
            source_ref=source_ref,
            limit=limit,
        )
        forged = batch.occurrences[0].model_copy(update={"adapter_id": "forged"})
        return batch.model_copy(update={"occurrences": [forged]})


def _related_adapter() -> ReplayChannelAdapter:
    """The shared replay adapter extended with a native related() answer."""
    base = _adapter()
    return _RelatedReplayAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries=base._discoveries,
        hydrations=base._hydrations,
    )


class _CountingHydrateAdapter(ReplayChannelAdapter):
    """Replay adapter that counts hydrate calls, records their arguments, and replays long excerpts."""

    def __init__(self, *, adapter_id: str = "counting") -> None:
        super().__init__(
            adapter_id=adapter_id,
            adapter_version="1",
            discoveries={},
            hydrations={},
        )
        self.hydrate_calls = 0
        self.last_arguments: dict | None = None

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Replay one long-snippet occurrence so nav cards exceed the display cap."""
        return DiscoveryBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            occurrences=[
                SourceOccurrence(
                    id="source-1",
                    adapter_id=self.adapter_id,
                    adapter_version=self.adapter_version,
                    source_ref="https://bbs.hupu.com/1",
                    canonical_url="https://bbs.hupu.com/1",
                    title="A discovered source",
                    captured_at=NOW,
                    metadata={"snippet": "discovery excerpt " + "x" * 300},
                )
            ],
        )

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Count one hydration, record its arguments, and replay a fixed DISCUSSION-depth observation."""
        self.hydrate_calls += 1
        self.last_arguments = dict(request.arguments)
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            observations=[
                SourceObservation(
                    id="sample",
                    source_ref=request.source_ref,
                    modality=ObservationModality.COMMENT,
                    access_depth=AccessDepth.REACTIONS,
                    excerpt="bounded discussion excerpt " + "x" * 300,
                    location=request.source_ref,
                    acquisition_method="replay_discussion",
                    captured_at=NOW,
                    metadata={"title": "Discussion sample", "published_at": PUBLISHED.isoformat()},
                )
            ],
        )


class _PageAwareAdapter(_CountingHydrateAdapter):
    """Replay adapter that records the full HydrationRequest passed to hydrate."""

    def __init__(self) -> None:
        super().__init__()
        self.last_request: HydrationRequest | None = None

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Record the full request (including its depth) and replay a fixed observation."""
        self.last_request = request
        return await super().hydrate(request)


async def test_open_source_accepts_depth_param(tmp_path: pytest.TempPathFactory) -> None:
    """Catch open_source ignoring its depth argument and always hydrating at TEXT depth."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = _PageAwareAdapter()  # records HydrationRequest (extend with depth capture)
    tools = WorldTools(store=store, adapters={"hupu": adapter})
    # previously the "depth" in arguments guard returned _limited("unsupported_depth")
    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "media"}, "c1")
    assert result["ok"] is True
    assert adapter.last_request is not None
    assert adapter.last_request.depth == HydrationDepth.MEDIA_TEXT


async def test_open_source_rejects_unknown_depth(tmp_path: pytest.TempPathFactory) -> None:
    """Catch an unknown depth value escaping the facade as a typed limitation."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={"hupu": _PageAwareAdapter()})
    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "bogus"}, "c2")
    assert result["ok"] is False
    assert result["limitations"] == ["invalid_arguments"]


async def test_open_source_already_opened_skips_hydration(tmp_path: pytest.TempPathFactory) -> None:
    """Catch re-hydration of an observation already opened at the requested depth."""
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
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1"}, "c1")

    assert result["ok"] is True
    assert result["already_opened"] is True
    assert adapter.hydrate_calls == 0
    # the fixture's observed_at (2026-08-04) is now always at least a day old,
    # so the freshness signal fires: base marker first, staleness hint second
    assert result["limitations"][0] == "observation already opened at depth=content"
    assert len(result["limitations"]) == 2
    assert result["limitations"][1].startswith("stored ")
    assert result["limitations"][1].endswith("source content may have changed")
    assert len(result["cards"]) == 1
    assert result["cards"][0]["id"] == "hupu:obs-1"


async def test_open_source_shallow_still_hydrates(tmp_path: pytest.TempPathFactory) -> None:
    """Catch already-opened short-circuiting before a SEEN observation reaches CONTENT."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-2",
                    source_uri="https://bbs.hupu.com/456",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 4, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-2",
    )
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-2"}, "c2")

    assert "already_opened" not in result
    assert adapter.hydrate_calls == 1


async def test_sample_discussion_requires_discussion_depth(tmp_path: pytest.TempPathFactory) -> None:
    """Catch discussion sampling that skips hydration before DISCUSSION depth is stored."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="bilibili:v1",
                    source_uri="https://bilibili.com/video/BV1",
                    source_kind="bilibili",
                    title="t",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime(2026, 8, 4, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-3",
    )
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"bilibili": adapter})

    # CONTENT < DISCUSSION -> must hydrate
    result = await tools.execute("sample_discussion", {"observation_id": "bilibili:v1"}, "c3")
    assert adapter.hydrate_calls == 1
    # the fake returns a DISCUSSION-depth observation stored under a durable id
    hydrated_id = result["cards"][0]["id"]
    # re-opening the now-DISCUSSION observation skips hydration
    result2 = await tools.execute("sample_discussion", {"observation_id": hydrated_id}, "c4")
    assert result2["already_opened"] is True
    assert adapter.hydrate_calls == 1
    assert result2["cards"][0]["id"] == hydrated_id


async def test_search_cards_truncate_excerpt_display_only(tmp_path: pytest.TempPathFactory) -> None:
    """Catch nav cards leaking the full stored excerpt into scan-path responses."""
    store = WorldStore(tmp_path / "world.sqlite3")
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("search_sources", {"adapter": "hupu", "query": "ig"}, "c1")

    assert result["ok"] is True
    for card in result["cards"]:
        assert len(card["excerpt"]) <= 150
    # storage unchanged: the committed observation rows keep full 800-char excerpts
    with store.read_connection() as connection:
        row = connection.execute("SELECT excerpt FROM observations ORDER BY rowid LIMIT 1").fetchone()
    assert len(row["excerpt"]) > 150  # display truncation must not touch storage


async def test_sample_discussion_forwards_page_argument(tmp_path: pytest.TempPathFactory) -> None:
    """Catch comment sampling dropping the requested page before the adapter."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("sample_discussion", {"observation_id": "hupu:obs-1", "page": 3}, "c2")

    assert result["ok"] is True
    # the domain key rides every hydrate (empty default keeps the bilibili or-'' fallback)
    assert adapter.last_arguments == {"page": 3, "domain": ""}
    # preview semantics: hydrate cards keep the full stored excerpt, not the 150-char nav cap
    assert len(result["cards"][0]["excerpt"]) > 150


async def test_sample_discussion_forwards_domain_alongside_page(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the world path dropping the domain key before channel hydration."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter}, domain="lol_cn")

    result = await tools.execute("sample_discussion", {"observation_id": "hupu:obs-1", "page": 2}, "c2")

    assert result["ok"] is True
    # the merge must keep the page argument while adding the domain key
    assert adapter.last_arguments == {"page": 2, "domain": "lol_cn"}


async def test_open_source_full_forwards_domain_into_hydrate(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch full opens dropping the domain key from the HydrationRequest."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = _FullBodyAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter}, domain="lol_cn")

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is True
    assert adapter.last_arguments == {"full": True, "domain": "lol_cn"}


async def test_hydrate_domain_defaults_to_empty_string(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch an absent domain still crossing as "" (bilibili or-'' fallback keeps behavior)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("sample_discussion", {"observation_id": "hupu:obs-1"}, "c1")

    assert result["ok"] is True
    assert adapter.last_arguments == {"domain": ""}


class _FullBodyAdapter(_PageAwareAdapter):
    """Replay adapter that attaches a full body when hydrate is asked for it."""

    BODY = "FULL TEXT " + "x" * 100  # padded above the 100-char stripped-text write gate

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Attach a full body to the replayed observation for full-depth requests."""
        batch = await super().hydrate(request)
        if request.arguments.get("full"):
            observation = batch.observations[0].model_copy(update={"body": self.BODY})
            return batch.model_copy(update={"observations": [observation]})
        return batch


class _TwoLargeBodiesAdapter(_CountingHydrateAdapter):
    """Replay adapter whose full hydrate returns two large observations."""

    def __init__(self) -> None:
        super().__init__(adapter_id="hupu")

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Count one hydration and replay two observations whose bodies exceed the tools cap."""
        self.hydrate_calls += 1
        self.last_arguments = dict(request.arguments)
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            observations=[
                _observation().model_copy(update={"body": "a" * 20_000, "source_ref": request.source_ref}),
                _observation().model_copy(update={"body": "b" * 20_000, "source_ref": request.source_ref}),
            ],
            discovered_occurrences=[],
        )


class _BoundedBodyAdapter(_CountingHydrateAdapter):
    """Replay adapter whose full hydrate returns one observation with a fixed body."""

    def __init__(self, body: str) -> None:
        super().__init__(adapter_id="hupu")
        self._body = body

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Count one hydration and replay one observation carrying the configured body."""
        self.hydrate_calls += 1
        self.last_arguments = dict(request.arguments)
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            observations=[
                _observation().model_copy(update={"body": self._body, "source_ref": request.source_ref})
            ],
            discovered_occurrences=[],
        )


class _LimitedBodyAdapter(_CountingHydrateAdapter):
    """Replay adapter whose full hydrate returns a fixed body plus batch limitations."""

    def __init__(self, body: str | None = None, limitations: list[str] | None = None) -> None:
        super().__init__(adapter_id="hupu")
        self._body = body
        self._limitations = list(limitations or [])

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Count one hydration and replay the configured body and batch limitations."""
        self.hydrate_calls += 1
        self.last_arguments = dict(request.arguments)
        observations = []
        if self._body is not None:
            observations = [
                _observation().model_copy(update={"body": self._body, "source_ref": request.source_ref})
            ]
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            observations=observations,
            discovered_occurrences=[],
            limitations=list(self._limitations),
        )


class _ProvenanceBodiesAdapter(ReplayChannelAdapter):
    """Replay one direct description plus one subtitle or ASR aggregate."""

    def __init__(
        self,
        *,
        transcript_reliability: str,
        transcript_method: str,
        transcript_limitations: list[str] | None = None,
    ) -> None:
        super().__init__(
            adapter_id="hupu",
            adapter_version="material-v1",
            discoveries={},
            hydrations={},
        )
        self.transcript_reliability = transcript_reliability
        self.transcript_method = transcript_method
        self.transcript_limitations = list(transcript_limitations or [])
        self.hydrate_calls = 0

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Return two exact body parts with distinct provenance."""
        self.hydrate_calls += 1
        description = "官方短公告：赛程更新"
        transcript = "选手语音逐字稿"
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            observations=[
                SourceObservation(
                    id="adapter-description",
                    source_ref=request.source_ref,
                    modality=ObservationModality.DOCUMENT_TEXT,
                    access_depth=AccessDepth.CONTENT_TEXT,
                    excerpt=description,
                    body=description,
                    location="video_description",
                    acquisition_method="bilibili_video_description",
                    captured_at=NOW,
                    confidence=1.0,
                ),
                SourceObservation(
                    id="adapter-transcript",
                    source_ref=request.source_ref,
                    modality=ObservationModality.TRANSCRIPT,
                    access_depth=AccessDepth.CONTENT_TEXT,
                    excerpt=transcript,
                    body=transcript,
                    location="video_transcript:full",
                    acquisition_method=self.transcript_method,
                    captured_at=NOW + timedelta(seconds=1),
                    confidence=0.73,
                    sampling_scope="full transcript;segments=3",
                    limitations=self.transcript_limitations,
                    metadata={
                        "subtitle_reliability": self.transcript_reliability,
                        "transcript_acquisition_method": self.transcript_method,
                    },
                ),
            ],
        )


class _ForgedFullBodyAdapter(_BoundedBodyAdapter):
    """Replay a body inside a batch that forges the adapter version."""

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Return the normal body with an invalid batch provenance envelope."""
        batch = await super().hydrate(request)
        return batch.model_copy(update={"adapter_version": "forged"})


class _ForeignSourceBodyAdapter(_BoundedBodyAdapter):
    """Replay a valid batch envelope whose body belongs to another source."""

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Return a source-mismatched observation under an otherwise valid batch."""
        batch = await super().hydrate(request)
        foreign = batch.observations[0].model_copy(update={"source_ref": "https://foreign.example/material"})
        return batch.model_copy(update={"observations": [foreign]})


class _RaisingFullBodyAdapter(_BoundedBodyAdapter):
    """Raise one scripted adapter or provider failure during full hydration."""

    def __init__(self, failure: Exception) -> None:
        super().__init__("unused")
        self.failure = failure

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Raise the configured deterministic failure without external I/O."""
        self.hydrate_calls += 1
        self.last_arguments = dict(request.arguments)
        raise self.failure


class _SequenceBodyAdapter(_BoundedBodyAdapter):
    """Return a deterministic sequence of bodies across refresh attempts."""

    def __init__(self, bodies: list[str]) -> None:
        super().__init__(bodies[0])
        self.bodies = list(bodies)

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Return the next scripted body while retaining normal provenance."""
        index = min(self.hydrate_calls, len(self.bodies) - 1)
        self._body = self.bodies[index]
        return await super().hydrate(request)


async def test_open_source_full_writes_and_serves_body(tmp_path: pytest.TempPathFactory) -> None:
    """Catch full-depth opens that skip the body write path or re-hydrate a stored body."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = _FullBodyAdapter()  # hydrate returns one observation with a body above the write gate
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    r1 = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert r1["ok"] is True
    assert r1["cards"][0]["body"] == _FullBodyAdapter.BODY
    assert len(r1["cards"]) == 1  # full depth returns at most 1 card
    assert adapter.hydrate_calls == 1
    # re-read serves from store (no second hydrate)
    r2 = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c2")
    assert "served_from_store" in r2["limitations"]
    assert adapter.hydrate_calls == 1


@pytest.mark.parametrize(
    ("reliability", "method", "limitations", "expected_reliability"),
    [
        ("confirmed", "platform_subtitle", [], "confirmed"),
        (
            "automatic",
            "bilibili_asr_aggregate",
            ["automatic_transcript_may_misrecognize_names_or_overlapping_speech"],
            "automatic",
        ),
    ],
)
async def test_open_source_full_persists_v2_parts_and_exact_provenance(
    tmp_path: pytest.TempPathFactory,
    reliability: str,
    method: str,
    limitations: list[str],
    expected_reliability: str,
) -> None:
    """Body cards expose durable, reversible direct-text and transcript parts."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _ProvenanceBodiesAdapter(
        transcript_reliability=reliability,
        transcript_method=method,
        transcript_limitations=limitations,
    )
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full"},
        "material-provenance",
    )

    assert result["ok"] is True
    card = result["cards"][0]
    assert card["primary_observation_id"] == "hupu:obs-1"
    assert card["source_kind"] == "hupu"
    assert card["depth"] == "content"
    assert "published_at" in card and "observed_at" in card
    assert card["material_reliability"] == "mixed"
    assert card["limitations"] == limitations
    assert card["limitations_truncated"] is False
    assert card["persisted"] is True
    assert card["body"] == "官方短公告：赛程更新\n\n选手语音逐字稿"
    assert card["refresh_available"] is True
    assert card["quality_flags"] == ["short_text", "mixed_reliability"]
    assert [part["reliability"] for part in card["parts"]] == [
        "source_direct",
        expected_reliability,
    ]
    assert [card["body"][part["start_char"] : part["end_char"]] for part in card["parts"]] == [
        "官方短公告：赛程更新",
        "选手语音逐字稿",
    ]
    assert card["parts"][1]["confidence"] == 0.73
    assert card["parts"][1]["sampling_scope"] == "full transcript;segments=3"
    assert card["parts"][1]["acquisition_method"] == method
    assert card["parts"][1]["limitations"] == limitations
    assert card["parts"][1]["limitations_truncated"] is False

    stored = store.read_observation_body("hupu:obs-1")
    assert stored is not None
    envelope = json.loads(stored["body_json"])
    assert envelope["schema_version"] == 2
    assert envelope["material_hash"] == hashlib.sha256(card["body"].encode("utf-8")).hexdigest()
    assert stored["size_bytes"] == len(stored["body_json"].encode("utf-8"))
    part_ids = [part["observation_id"] for part in card["parts"]]
    with store.read_connection() as connection:
        rows = connection.execute(
            f"SELECT id, metadata_json FROM observations WHERE id IN ({','.join('?' for _ in part_ids)})",
            part_ids,
        ).fetchall()
    assert {str(row["id"]) for row in rows} == set(part_ids)
    transcript_metadata = next(
        json.loads(row["metadata_json"]) for row in rows if "body-transcript" in str(row["id"])
    )
    assert transcript_metadata["confidence"] == 0.73
    assert transcript_metadata["sampling_scope"] == "full transcript;segments=3"
    assert transcript_metadata["subtitle_reliability"] == reliability
    assert transcript_metadata["transcript_acquisition_method"] == method

    with store.write_connection() as connection:
        connection.execute("UPDATE observations SET depth = 'seen' WHERE id = 'hupu:obs-1'")
    reread = await tools.execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full"},
        "material-provenance-reread",
    )
    assert adapter.hydrate_calls == 1
    assert reread["limitations"] == ["served_from_store"]
    for field in (
        "body",
        "captured_at",
        "material_hash",
        "quality_flags",
        "parts",
        "refresh_available",
    ):
        assert reread["cards"][0][field] == card[field]
    with store.read_connection() as connection:
        primary = connection.execute("SELECT depth FROM observations WHERE id = 'hupu:obs-1'").fetchone()
    assert primary["depth"] == "content"


async def test_body_card_bounds_part_and_aggregate_limitations_without_rewriting_material(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Body-card limits protect response size while the durable envelope remains exact."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    raw_limitations = [f"{index}-" + "x" * 200 for index in range(10)]
    adapter = _ProvenanceBodiesAdapter(
        transcript_reliability="automatic",
        transcript_method="bilibili_asr_aggregate",
        transcript_limitations=raw_limitations,
    )
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    fresh = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "limits")
    reread = await tools.execute(
        "open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "limits-reread"
    )

    for card in (fresh["cards"][0], reread["cards"][0]):
        assert card["material_reliability"] == "mixed"
        assert card["limitations"][-1] == "limitations_truncated"
        assert card["limitations_truncated"] is True
        assert len(card["limitations"]) <= 8
        assert sum(map(len, card["limitations"])) <= 640
        transcript = card["parts"][1]
        assert transcript["limitations"][-1] == "limitations_truncated"
        assert transcript["limitations_truncated"] is True
        assert len(transcript["limitations"]) <= 8
        assert max(map(len, transcript["limitations"])) <= 160
        assert sum(map(len, transcript["limitations"])) <= 640


async def test_open_source_full_provenance_mismatch_writes_no_parts_or_body(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A forged hydrate batch fails closed before any material write."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _ForgedFullBodyAdapter("identity-correct text")
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full"},
        "forged-full",
    )

    assert result["ok"] is False
    assert result["error"] == {
        "code": "adapter_provenance_mismatch",
        "stage": "full_hydrate",
        "attempts": 1,
        "same_request_can_change": True,
        "change_condition": "adapter_response_identity_changes",
    }
    assert store.read_observation_body("hupu:obs-1") is None
    with store.read_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1


async def test_open_source_full_foreign_body_observation_writes_nothing(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A source-mismatched body fails before part, envelope, or depth writes."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _ForeignSourceBodyAdapter("foreign text")
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full"},
        "foreign-body",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "adapter_provenance_mismatch"
    assert result["error"]["stage"] == "full_hydrate"
    assert store.read_observation_body("hupu:obs-1") is None
    with store.read_connection() as connection:
        rows = connection.execute("SELECT id, depth FROM observations").fetchall()
    assert [(row["id"], row["depth"]) for row in rows] == [("hupu:obs-1", "seen")]


async def test_open_source_full_recaps_assembled_body_and_persists_truncation(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch cross-observation assembly exceeding the tools cap dropping the signal."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = _TwoLargeBodiesAdapter()  # 20K + 20K bodies exceed the 32K tools cap
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    r1 = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert r1["ok"] is True
    assert len(r1["cards"][0]["body"]) == 32_000
    assert "body_truncated" in r1["limitations"]
    # the envelope persists the flag so re-reads can re-emit it
    row = store.read_observation_body("hupu:obs-1")
    assert row is not None
    envelope = json.loads(row["body_json"])
    assert envelope["text"] == r1["cards"][0]["body"]
    assert envelope["truncated"] is True
    assert envelope["schema_version"] == 2
    assert envelope["quality_flags"] == ["body_truncated"]
    assert envelope["parts"][-1]["end_char"] == 32_000
    assert "body_truncated" in envelope["parts"][-1]["limitations"]
    assert [envelope["text"][part["start_char"] : part["end_char"]] for part in envelope["parts"]] == [
        "a" * 20_000,
        "b" * 11_998,
    ]
    # the store-served re-read re-emits the limitation without another hydrate
    r2 = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c2")
    assert "served_from_store" in r2["limitations"]
    assert "body_truncated" in r2["limitations"]
    assert adapter.hydrate_calls == 1


async def test_open_source_full_no_body_fails_typed(tmp_path: pytest.TempPathFactory) -> None:
    """Catch adapters ignoring full registering a phantom empty body card."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    adapter = _CountingHydrateAdapter()  # replays observations without any body
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result == {
        "ok": False,
        "outcome": "empty",
        "error": {
            "code": "no_full_content",
            "stage": "full_hydrate",
            "attempts": 1,
            "same_request_can_change": True,
            "change_condition": "source_content_or_public_access_changes",
        },
        "limitations": ["no_full_content", "full_content_unavailable"],
    }
    # nothing phantom written, so the graph's uncompacted accounting stays empty
    assert store.read_observation_body("hupu:obs-1") is None


class _EmptyHydrateAdapter(_CountingHydrateAdapter):
    """Replay adapter whose hydrate yields no observations plus adapter reasons."""

    def __init__(self, limitations: list[str] | None = None) -> None:
        super().__init__()
        self._limitations = list(limitations or ["hupu_thread_no_public_text"])
        self.outcome = AcquisitionOutcome.SUCCESS

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Count one hydration and replay an empty batch with the configured limitations."""
        self.hydrate_calls += 1
        self.last_arguments = dict(request.arguments)
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=self.outcome,
            observations=[],
            discovered_occurrences=[],
            limitations=list(self._limitations),
        )


def _seen_seed(store: WorldStore) -> None:
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )


async def test_open_source_empty_hydrate_fails_typed_with_no_content_marker(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch preview opens masking an empty platform answer with an ok card.

    A hydrate batch with neither observations nor discovered occurrences is a
    platform answer (e.g. a hupu thread whose main post has no public text),
    not a successful acquisition: returning ok:true cards:[] would teach the
    model "this platform returns nothing" (platform-visibility review). The
    response must fail with the adapter's own reasons first plus the
    no_content_observed marker.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    _seen_seed(store)
    adapter = _EmptyHydrateAdapter(limitations=["hupu_thread_no_public_text"])
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1"}, "c1")

    assert result == {
        "ok": False,
        "outcome": "empty",
        "error": {
            "code": "no_content_observed",
            "stage": "hydrate",
            "attempts": 1,
            "same_request_can_change": True,
            "change_condition": "source_content_or_public_access_changes",
        },
        "limitations": ["hupu_thread_no_public_text", "no_content_observed"],
    }
    assert adapter.hydrate_calls == 1


async def test_open_source_preserves_unsupported_material_outcome(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A capability boundary is not flattened into an ordinary empty source."""
    store = WorldStore(tmp_path / "unsupported-open.sqlite3")
    _seen_seed(store)
    adapter = _EmptyHydrateAdapter(limitations=["public_web_rendering_required"])
    adapter.outcome = AcquisitionOutcome.UNSUPPORTED

    result = await WorldTools(store=store, adapters={"hupu": adapter}).execute(
        "open_source", {"observation_id": "hupu:obs-1"}, "unsupported-open"
    )

    assert result["outcome"] == "unsupported"
    assert result["error"] == {
        "code": "public_web_rendering_required",
        "stage": "hydrate",
        "attempts": 1,
        "same_request_can_change": False,
        "change_condition": "request_or_adapter_capability_changes",
    }


async def test_open_source_empty_hydrate_bounds_adapter_limitations(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Empty hydrates preserve their typed marker without returning unbounded adapter text."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seen_seed(store)
    adapter = _EmptyHydrateAdapter(limitations=[f"{index}-" + "x" * 200 for index in range(10)])

    result = await WorldTools(store=store, adapters={"hupu": adapter}).execute(
        "open_source", {"observation_id": "hupu:obs-1"}, "bounded-empty"
    )

    assert result["ok"] is False
    assert "no_content_observed" in result["limitations"]
    assert result["limitations"][-1] == "limitations_truncated"
    assert len(result["limitations"]) <= 8
    assert max(map(len, result["limitations"])) <= 160
    assert sum(map(len, result["limitations"])) <= 640


async def test_sample_discussion_empty_hydrate_fails_typed_with_no_content_marker(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch sample_discussion masking an empty comment set with an ok card."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seen_seed(store)
    adapter = _EmptyHydrateAdapter(limitations=["reply_sample_is_not_whole_community_opinion"])
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("sample_discussion", {"observation_id": "hupu:obs-1"}, "c1")

    assert result["ok"] is False
    assert result["limitations"] == [
        "reply_sample_is_not_whole_community_opinion",
        "no_content_observed",
    ]
    assert adapter.hydrate_calls == 1


def _body_gate_seed(store: WorldStore) -> None:
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )


async def test_open_source_full_short_body_is_durable_advisory_material(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Short non-blank material is durable and carries an advisory flag."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _BoundedBodyAdapter("相关游戏：英雄联盟")  # 9 chars, below the stripped-text floor
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is True
    assert result["cards"][0]["body"] == "相关游戏：英雄联盟"
    assert result["cards"][0]["quality_flags"] == ["short_text"]
    assert store.read_observation_body("hupu:obs-1") is not None
    assert adapter.hydrate_calls == 1


@pytest.mark.parametrize(
    ("body",),
    [
        ("x" * 30,),
        ("x" * 80,),
        ("x" * 99,),
        ("x" * 100,),
        ("x" * 101,),
    ],
)
async def test_open_source_full_body_length_threshold_boundaries(
    tmp_path: pytest.TempPathFactory, body: str
) -> None:
    """Length never changes non-blank full-body success semantics."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _BoundedBodyAdapter(body)
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is True
    assert result["cards"][0]["body"] == body
    row = store.read_observation_body("hupu:obs-1")
    assert row is not None
    assert json.loads(row["body_json"])["text"] == body
    assert adapter.hydrate_calls == 1


async def test_open_source_full_whitespace_only_body_stays_no_full_content(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch stripped-to-empty bodies regressing off the no_full_content path."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _BoundedBodyAdapter("   ")  # whitespace strips to an empty body
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is False
    assert result["error"]["code"] == "no_full_content"
    assert result["limitations"] == ["no_full_content", "full_content_unavailable"]
    assert store.read_observation_body("hupu:obs-1") is None


async def test_open_source_full_no_body_passes_through_adapter_limitations(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the no-body failure dropping the adapter's specific reason (b7: generic feedback)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _LimitedBodyAdapter(
        limitations=["platform_subtitle_absent_or_unavailable_without_authentication"]
    )
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is False
    assert result["error"]["code"] == "no_full_content"
    assert result["limitations"] == [
        "platform_subtitle_absent_or_unavailable_without_authentication",
        "no_full_content",
        "full_content_unavailable",
    ]
    assert store.read_observation_body("hupu:obs-1") is None


async def test_open_source_full_unavailable_bounds_adapter_limitations(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Typed no-full-content responses cap oversized adapter-origin limitation strings."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _LimitedBodyAdapter(limitations=[f"{index}-" + "x" * 200 for index in range(10)])
    result = await WorldTools(store=store, adapters={"hupu": adapter}).execute(
        "open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "bounded-unavailable"
    )

    assert result["ok"] is False
    assert "no_full_content" in result["limitations"]
    assert "full_content_unavailable" in result["limitations"]
    assert result["limitations"][-1] == "limitations_truncated"
    assert len(result["limitations"]) <= 8
    assert max(map(len, result["limitations"])) <= 160
    assert sum(map(len, result["limitations"])) <= 640


async def test_open_source_full_short_body_keeps_adapter_limitations(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A short accepted body retains adapter limitations unchanged."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _LimitedBodyAdapter(
        body="相关游戏：英雄联盟",  # 9 chars, below the stripped-text floor
        limitations=["video_tags_empty_or_public_endpoint_unavailable"],
    )
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is True
    assert result["limitations"] == ["video_tags_empty_or_public_endpoint_unavailable"]
    assert result["cards"][0]["quality_flags"] == ["short_text"]
    assert store.read_observation_body("hupu:obs-1") is not None


async def test_open_source_full_failure_dedupes_adapter_reported_marker_and_hint(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the failure envelope duplicating reasons the adapter already reported."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _LimitedBodyAdapter(limitations=["no_full_content", "full_content_unavailable"])
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    # adapter order preserved, marker and hint deduped: nothing repeated
    assert result["ok"] is False
    assert result["error"]["code"] == "no_full_content"
    assert result["limitations"] == ["no_full_content", "full_content_unavailable"]
    assert store.read_observation_body("hupu:obs-1") is None


def _seed_stored_body(store: WorldStore, body: str) -> None:
    _body_gate_seed(store)
    store.write_observation_body(
        "hupu:obs-1",
        "mixed",
        json.dumps({"text": body, "truncated": False}, ensure_ascii=False),
        len(body),
    )


def _write_v2_body_for_part(store: WorldStore, part_id: str) -> None:
    envelope = build_body_envelope(
        [
            MaterialPartInput(
                observation_id=part_id,
                text="exact stored material",
                kind="description",
                location="video_description",
                acquisition_method="bilibili_video_description",
                confidence=1.0,
                reliability="source_direct",
                sampling_scope="",
                limitations=(),
                captured_at=NOW,
            )
        ],
        max_chars=32_000,
    )
    assert envelope is not None
    store.write_observation_body("hupu:obs-1", "mixed", envelope.to_json())


@pytest.mark.parametrize(
    ("part_source_uri", "part_depth"),
    [
        ("https://other.example/material", ObservationDepth.CONTENT),
        ("https://bbs.hupu.com/1", ObservationDepth.SEEN),
    ],
)
async def test_store_served_v2_unverifiable_part_never_upgrades_primary(
    tmp_path: pytest.TempPathFactory,
    part_source_uri: str,
    part_depth: ObservationDepth,
) -> None:
    """Foreign or shallow part rows cannot lend CONTENT authority to a primary."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    part_id = "hupu:body-part"
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id=part_id,
                    source_uri=part_source_uri,
                    source_kind="hupu",
                    title="part",
                    excerpt="exact stored material",
                    depth=part_depth,
                    observed_at=NOW,
                    metadata={
                        "adapter_version": "v1",
                        "body_part": True,
                        "material_kind": "description",
                        "material_reliability": "source_direct",
                        "location": "video_description",
                        "acquisition_method": "bilibili_video_description",
                    },
                )
            ]
        ),
        "seed-unverifiable-part",
    )
    _write_v2_body_for_part(store, part_id)
    adapter = _CountingHydrateAdapter()

    result = await WorldTools(store=store, adapters={"hupu": adapter}).execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full"},
        "unverifiable-v2",
    )

    assert result["ok"] is True
    assert result["limitations"] == ["served_from_store", "body_provenance_unverified"]
    assert adapter.hydrate_calls == 0
    with store.read_connection() as connection:
        primary = connection.execute("SELECT depth FROM observations WHERE id = 'hupu:obs-1'").fetchone()
    assert primary["depth"] == "seen"


async def test_store_served_v2_duplicate_part_id_validates_every_provenance_claim(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A repeated durable ID cannot hide one mismatched envelope part."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    part_id = "hupu:body-part"
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id=part_id,
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="part",
                    excerpt="second",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                    metadata={
                        "adapter_version": "v1",
                        "body_part": True,
                        "material_kind": "description",
                        "material_reliability": "source_direct",
                        "location": "video_description",
                        "acquisition_method": "bilibili_video_description",
                    },
                )
            ]
        ),
        "seed-duplicate-part",
    )
    envelope = build_body_envelope(
        [
            MaterialPartInput(
                observation_id=part_id,
                text="first",
                kind="transcript",
                location="video_transcript:full",
                acquisition_method="bilibili_asr_aggregate",
                confidence=0.7,
                reliability="automatic",
                sampling_scope="full transcript",
                limitations=("automatic_transcript",),
                captured_at=NOW,
            ),
            MaterialPartInput(
                observation_id=part_id,
                text="second",
                kind="description",
                location="video_description",
                acquisition_method="bilibili_video_description",
                confidence=1.0,
                reliability="source_direct",
                sampling_scope="",
                limitations=(),
                captured_at=NOW,
            ),
        ],
        max_chars=32_000,
    )
    assert envelope is not None
    store.write_observation_body("hupu:obs-1", "mixed", envelope.to_json())

    result = await WorldTools(store=store, adapters={"hupu": _CountingHydrateAdapter()}).execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full"},
        "duplicate-part",
    )

    assert result["limitations"] == ["served_from_store", "body_provenance_unverified"]
    with store.read_connection() as connection:
        primary = connection.execute("SELECT depth FROM observations WHERE id = 'hupu:obs-1'").fetchone()
    assert primary["depth"] == "seen"


async def test_store_served_invalid_v2_is_typed_and_keeps_seen_depth(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Malformed stored material fails closed without a hydrate or authority change."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    store.write_observation_body("hupu:obs-1", "mixed", "{not-json")
    adapter = _CountingHydrateAdapter()

    result = await WorldTools(store=store, adapters={"hupu": adapter}).execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full"},
        "invalid-stored-v2",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "stored_body_invalid"
    assert result["error"]["attempts"] == 0
    assert result["limitations"] == ["stored_body_invalid", "full_content_unavailable"]
    assert adapter.hydrate_calls == 0
    with store.read_connection() as connection:
        primary = connection.execute("SELECT depth FROM observations WHERE id = 'hupu:obs-1'").fetchone()
    assert primary["depth"] == "seen"


async def test_open_source_full_store_served_legacy_short_body_is_readable(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Legacy bodies remain readable but never claim verified provenance."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed_stored_body(store, "-")  # a stored garbage body, never hydration-written
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is True
    assert result["cards"][0]["body"] == "-"
    assert result["cards"][0]["provenance"] == "unknown"
    assert result["limitations"] == ["served_from_store", "body_provenance_unknown"]
    assert adapter.hydrate_calls == 0


@pytest.mark.parametrize(
    ("body",),
    [
        ("x" * 99,),
        ("x" * 100,),
        ("x" * 500,),
    ],
)
async def test_open_source_full_store_served_body_gate_boundaries(
    tmp_path: pytest.TempPathFactory, body: str
) -> None:
    """Legacy body length cannot become a semantic validity gate."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _seed_stored_body(store, body)
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is True
    assert "served_from_store" in result["limitations"]
    assert result["cards"][0]["body"] == body
    assert adapter.hydrate_calls == 0


async def test_open_source_full_upgrades_observation_depth_to_content(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch full reads that leave the observation at SEEN depth (spec D1 evidence-depth unity)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _FullBodyAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is True
    with store.read_connection() as connection:
        row = connection.execute("SELECT depth FROM observations WHERE id = 'hupu:obs-1'").fetchone()
    assert row["depth"] == "content"


async def test_full_read_then_preview_short_circuits_without_hydration(
    tmp_path: pytest.TempPathFactory,
) -> None:
    "Catch re-hydrating an upgraded observation when preview is requested (request depth <= stored depth)."
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _FullBodyAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    full = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")
    preview = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "preview"}, "c2")

    assert full["ok"] is True
    assert preview["ok"] is True
    assert preview["already_opened"] is True
    assert preview["cards"][0]["id"] == "hupu:obs-1"
    assert adapter.hydrate_calls == 1  # only the full hydrate; the preview served from store


async def test_repeated_full_read_is_idempotent_without_rehydrate(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch full re-reads re-hydrating, re-writing the body, or re-upgrading depth."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _FullBodyAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    r1 = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")
    r2 = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c2")

    assert r1["ok"] is True
    assert r2["ok"] is True
    assert r1["outcome"] == "success"
    assert r1["completeness"]["returned"] == 1
    assert r2["outcome"] == "success"
    assert r2["completeness"] == {
        "returned": 1,
        "limit": 1,
        "partial": False,
        "truncated": False,
        "next_cursor_usable": False,
        "physical_calls": 0,
    }
    assert "served_from_store" in r2["limitations"]
    assert adapter.hydrate_calls == 1
    with store.read_connection() as connection:
        row = connection.execute("SELECT depth FROM observations WHERE id = 'hupu:obs-1'").fetchone()
    assert row["depth"] == "content"


async def test_full_read_observation_supports_committer_assertion(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the p2d hole: a full-read observation still dropped by the committer's seen_support rule."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="BLG")],
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ],
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={"hupu": _FullBodyAdapter()})
    opened = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")
    assert opened["ok"] is True

    proposal = CognitionDeltaProposal(
        assertions=[
            AssertionProposal(
                subject=GraphRef(memory_id="obj-1"),
                predicate="related_to",
                literal="confirms the full text",
                epistemic_role=EpistemicRole.FACT,
                confidence=0.8,
                evidence=[EvidenceInput(observation_id="hupu:obs-1", role="supports")],
            )
        ]
    )
    receipt = ProposalCommitter(store).commit(proposal, "proposal-1")

    assert receipt.omitted_assertion_indexes == []
    assert receipt.evidence_missing_assertion_indexes == []
    assert len(receipt.commit.assertion_ids) == 1
    assertion_id = receipt.commit.assertion_ids[0]
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT role FROM assertion_evidence WHERE assertion_id = ? AND observation_id = 'hupu:obs-1'",
            (assertion_id,),
        ).fetchone()
    assert row["role"] == "supports"  # not silently downgraded to context


async def test_store_served_legacy_body_does_not_upgrade_seen_depth(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Legacy body presence alone cannot turn SEEN evidence into CONTENT."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    # legacy row: body already written but depth still SEEN (written before
    # the upgrade feature existed, or a partial write)
    store.write_observation_body(
        "hupu:obs-1", "article", json.dumps({"text": "legacy body " + "x" * 100}), 100
    )
    adapter = _FullBodyAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "c1")

    assert result["ok"] is True
    assert "served_from_store" in result["limitations"]
    assert "body_provenance_unknown" in result["limitations"]
    assert adapter.hydrate_calls == 0
    with store.read_connection() as connection:
        row = connection.execute("SELECT depth FROM observations WHERE id = 'hupu:obs-1'").fetchone()
    assert row["depth"] == "seen"


async def test_open_source_full_refresh_replaces_body_and_preserves_old_on_failure(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Explicit refresh is atomic: success replaces, failure serves exact old bytes."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _BoundedBodyAdapter("old official text")
    tools = WorldTools(store=store, adapters={"hupu": adapter})
    first = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "r1")
    assert first["ok"] is True
    before = store.read_observation_body("hupu:obs-1")
    assert before is not None
    adapter._body = "new official text"  # scripted fixture, never a live adapter
    refreshed = await tools.execute(
        "open_source", {"observation_id": "hupu:obs-1", "depth": "full", "refresh": True}, "r2"
    )
    assert refreshed["cards"][0]["body"] == "new official text"
    assert adapter.hydrate_calls == 2
    replacement = store.read_observation_body("hupu:obs-1")
    assert replacement is not None and replacement["body_json"] != before["body_json"]
    before_material = parse_stored_body(str(before["body_json"]))
    replacement_material = parse_stored_body(str(replacement["body_json"]))
    assert isinstance(before_material, BodyEnvelope)
    assert isinstance(replacement_material, BodyEnvelope)
    assert replacement_material.material_hash != before_material.material_hash
    assert [part.observation_id for part in replacement_material.parts] != [
        part.observation_id for part in before_material.parts
    ]
    assert store.body_parts_are_verifiable("hupu:obs-1", list(replacement_material.parts))
    failed = await WorldTools(
        store=store, adapters={"hupu": _LimitedBodyAdapter(limitations=["subtitle_unavailable"])}
    ).execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full", "refresh": True}, "r3")
    assert failed["cards"][0]["body"] == "new official text"
    assert "served_stale_after_refresh_failure" in failed["limitations"]
    assert "no_full_content" in failed["limitations"]
    assert store.read_observation_body("hupu:obs-1") == replacement


@pytest.mark.parametrize(
    ("failure_kind", "expected_reason"),
    [
        ("agent_error", "adapter_failure:SOURCE_UNAVAILABLE"),
        ("exception", "adapter_failure:RuntimeError"),
        ("batch_provenance", "adapter_provenance_mismatch"),
        ("source_identity", "adapter_provenance_mismatch"),
    ],
)
async def test_refresh_failures_preserve_exact_stored_material(
    tmp_path: pytest.TempPathFactory,
    failure_kind: str,
    expected_reason: str,
) -> None:
    """Every adapter/provenance refresh failure serves the exact old envelope."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    initial = _BoundedBodyAdapter("stable old material")
    await WorldTools(store=store, adapters={"hupu": initial}).execute(
        "open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "failure-seed"
    )
    before = store.read_observation_body("hupu:obs-1")
    assert before is not None
    if failure_kind == "agent_error":
        adapter = _RaisingFullBodyAdapter(
            AgentError(ErrorCode.SOURCE_UNAVAILABLE, "scripted unavailable")
        )
    elif failure_kind == "exception":
        adapter = _RaisingFullBodyAdapter(RuntimeError("scripted provider failure"))
    elif failure_kind == "batch_provenance":
        adapter = _ForgedFullBodyAdapter("forged replacement")
    else:
        adapter = _ForeignSourceBodyAdapter("foreign replacement")

    result = await WorldTools(store=store, adapters={"hupu": adapter}).execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full", "refresh": True},
        f"failure-{failure_kind}",
    )

    assert result["cards"][0]["body"] == "stable old material"
    assert "served_stale_after_refresh_failure" in result["limitations"]
    assert expected_reason in result["limitations"]
    assert store.read_observation_body("hupu:obs-1") == before


async def test_refresh_store_error_preserves_exact_stored_material(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transactional SQLite failure never exposes a partial replacement."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    await WorldTools(store=store, adapters={"hupu": _BoundedBodyAdapter("stable old material")}).execute(
        "open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "store-failure-seed"
    )
    before = store.read_observation_body("hupu:obs-1")
    assert before is not None

    def fail_write(*args, **kwargs):
        raise sqlite3.OperationalError("scripted write failure")

    monkeypatch.setattr(store, "replace_observation_body_with_parts", fail_write)
    result = await WorldTools(
        store=store, adapters={"hupu": _BoundedBodyAdapter("uncommitted replacement")}
    ).execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full", "refresh": True},
        "store-failure",
    )

    assert result["cards"][0]["body"] == "stable old material"
    assert "served_stale_after_refresh_failure" in result["limitations"]
    assert "refresh_material_write_failed" in result["limitations"]
    assert store.read_observation_body("hupu:obs-1") == before


async def test_first_full_read_store_failure_reports_persist_stage(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetched body is not reported as material when its atomic write fails."""
    store = WorldStore(tmp_path / "first-store-failure.sqlite3")
    _body_gate_seed(store)

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise sqlite3.OperationalError("scripted write failure")

    monkeypatch.setattr(store, "replace_observation_body_with_parts", fail_write)

    result = await WorldTools(
        store=store, adapters={"hupu": _FullBodyAdapter()}
    ).execute(
        "open_source",
        {"observation_id": "hupu:obs-1", "depth": "full"},
        "first-store-failure",
    )

    assert result["error"] == {
        "code": "material_write_failed",
        "stage": "persist",
        "attempts": 1,
        "same_request_can_change": True,
        "change_condition": "world_store_write_succeeds",
    }
    assert store.read_observation_body("hupu:obs-1") is None


async def test_same_call_id_refresh_race_replays_complete_material_transaction(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Concurrent same-id attempts cannot pair one envelope with another attempt's parts."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _SequenceBodyAdapter(["first candidate", "second candidate"])
    tools = WorldTools(store=store, adapters={"hupu": adapter})
    arguments = {"observation_id": "hupu:obs-1", "depth": "full", "refresh": True}

    results = await asyncio.gather(
        tools.execute("open_source", arguments, "same-call"),
        tools.execute("open_source", arguments, "same-call"),
    )

    stored = store.read_observation_body("hupu:obs-1")
    assert stored is not None
    material = parse_stored_body(str(stored["body_json"]))
    assert isinstance(material, BodyEnvelope)
    assert {result["cards"][0]["body"] for result in results} == {material.text}
    assert material.text in {"first candidate", "second candidate"}
    assert store.body_parts_are_verifiable("hupu:obs-1", list(material.parts))


async def test_open_source_full_default_store_serve_is_zero_call_and_refresh_requires_full(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Default full reads stay offline; refresh is invalid outside full depth."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    initial = WorldTools(store=store, adapters={"hupu": _BoundedBodyAdapter("stored text")})
    await initial.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "z1")
    counter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": counter})
    served = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "z2")
    assert served["ok"] is True and counter.hydrate_calls == 0
    invalid = await tools.execute(
        "open_source", {"observation_id": "hupu:obs-1", "depth": "preview", "refresh": True}, "z3"
    )
    assert invalid["ok"] is False
    assert invalid["error"]["code"] == "invalid_arguments"


async def test_verified_revision_mismatch_refreshes_but_unknown_revision_does_not(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Only a comparable verified fingerprint may trigger an implicit refresh."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _body_gate_seed(store)
    adapter = _BoundedBodyAdapter("replacement text")
    tools = WorldTools(store=store, adapters={"hupu": adapter})
    await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "m1")
    stored = store.read_observation_body("hupu:obs-1")
    assert stored is not None
    envelope = json.loads(stored["body_json"])
    envelope.update({"source_revision": "old", "source_revision_kind": "verified_content_hash"})
    store.write_observation_body("hupu:obs-1", "mixed", json.dumps(envelope, ensure_ascii=False))
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE observations SET metadata_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "content_hash_verified": True,
                        "source_revision": "new",
                        "source_revision_kind": "verified_content_hash",
                    }
                ),
                "hupu:obs-1",
            ),
        )
    refreshed = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "m2")
    assert refreshed["cards"][0]["body"] == "replacement text"
    assert adapter.hydrate_calls == 2
    with store.write_connection() as connection:
        connection.execute("UPDATE observations SET metadata_json = '{}' WHERE id = ?", ("hupu:obs-1",))
    served = await tools.execute("open_source", {"observation_id": "hupu:obs-1", "depth": "full"}, "m3")
    assert served["ok"] is True and adapter.hydrate_calls == 2


async def test_rescan_after_full_read_does_not_reject(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a re-discovered observation (SEEN re-commit) rejecting the whole scan batch after upgrade."""
    store = WorldStore(tmp_path / "world.sqlite3")
    adapter = _FullBodyAdapter()
    tools = WorldTools(store=store, adapters={"counting": adapter})

    first = await tools.execute(
        "discover_sources", {"adapter": "counting", "query": "source"}, "discover-call"
    )
    obs_id = first["cards"][0]["id"]
    full = await tools.execute("open_source", {"observation_id": obs_id, "depth": "full"}, "full-call")
    assert full["ok"] is True

    rescan = await tools.execute(
        "discover_sources", {"adapter": "counting", "query": "source"}, "discover-call-2"
    )

    assert rescan["ok"] is True
    assert rescan["cards"][0]["id"] == obs_id
    with store.read_connection() as connection:
        row = connection.execute("SELECT depth FROM observations WHERE id = ?", (obs_id,)).fetchone()
    assert row["depth"] == "content"


async def test_sample_discussion_rejects_non_integer_page(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a hostile page argument escaping the facade as a typed limitation."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-2",
                    source_uri="https://bbs.hupu.com/2",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-2",
    )
    adapter = _CountingHydrateAdapter()
    tools = WorldTools(store=store, adapters={"hupu": adapter})

    result = await tools.execute("sample_discussion", {"observation_id": "hupu:obs-2", "page": "abc"}, "c3")

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"
    assert adapter.hydrate_calls == 0


async def test_preview_hydrate_with_body_reaches_content_depth(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch B2-13 end-to-end: metadata(SEEN, long title) + body(CONTENT) merge into CONTENT.

    Reproduces the b8 verification failure through the real tool path: the
    preview hydrate returns both a metadata row whose title excerpt
    outlengthens the body row, and a DOCUMENT_TEXT body row; the merged row
    must carry CONTENT depth so the observed marker (which already claims
    content) and the depth agree, letting the committer's seen_support rule
    keep the body-backed assertion instead of omitting it.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    metadata_observation = SourceObservation(
        id="meta-1",
        source_ref=SOURCE,
        modality=ObservationModality.METADATA,
        access_depth=AccessDepth.METADATA,
        excerpt="T" * 74,  # bilibili titles routinely outlengthen the description
        location=SOURCE,
        acquisition_method="replay_video_info",
        captured_at=NOW,
        metadata={"title": "T" * 74},
    )
    description = SourceObservation(
        id="body-1",
        source_ref=SOURCE,
        modality=ObservationModality.DOCUMENT_TEXT,
        access_depth=AccessDepth.CONTENT_TEXT,
        excerpt="A much shorter description than the title.",
        location=SOURCE,
        acquisition_method="replay_open",
        captured_at=NOW,
        metadata={"published_at": PUBLISHED.isoformat()},
    )
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={
            "tool:b13-discover": DiscoveryBatch(
                request_id="tool:b13-discover",
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
                observations=[metadata_observation, description],
            )
        },
    )
    tools = WorldTools(store=store, adapters={"replay": adapter})
    discovered = await tools.execute(
        "discover_sources", {"adapter": "replay", "query": "source"}, "b13-discover"
    )
    card_id = discovered["cards"][0]["id"]

    opened = await tools.execute("open_source", {"observation_id": card_id}, "b13-preview")

    assert opened["ok"] is True
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT depth, metadata_json FROM observations WHERE id = ?", (card_id,)
        ).fetchone()
    assert row["depth"] == "content"  # was "seen" pre-fix when the title outlengthened the body
    observed = json.loads(row["metadata_json"])["observed"]
    assert observed == ["metadata", "content"]  # marker and depth agree, no divergence
