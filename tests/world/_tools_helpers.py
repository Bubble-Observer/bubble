# Split from tests/world/test_tools.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

from datetime import UTC, datetime

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
    WorldStore,
)
from leave_information_bubble.world.store import observation_id
from leave_information_bubble.world.tools import WorldTools

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)
PUBLISHED = datetime(2026, 8, 1, 9, tzinfo=UTC)
SOURCE = "https://example.test/source"
CAP = 5

def _occurrence(identifier: str = "source-1") -> SourceOccurrence:
    return SourceOccurrence(
        id=identifier,
        adapter_id="replay",
        adapter_version="1",
        source_ref=SOURCE,
        canonical_url=SOURCE,
        title="A discovered source",
        source_published_at=PUBLISHED,
        captured_at=NOW,
        metadata={"snippet": "discovery excerpt", "unsafe_body": "x" * 10_000},
    )


def _observation(identifier: str = "content-1") -> SourceObservation:
    return SourceObservation(
        id=identifier,
        source_ref=SOURCE,
        modality=ObservationModality.DOCUMENT_TEXT,
        access_depth=AccessDepth.CONTENT_TEXT,
        excerpt="bounded content excerpt",
        location=SOURCE,
        acquisition_method="replay_open",
        captured_at=NOW,
        metadata={"title": "Opened source", "published_at": PUBLISHED.isoformat(), "raw": "x" * 10_000},
    )


def _adapter() -> ReplayChannelAdapter:
    occurrence = _occurrence()
    requests = [
        "discover-call",
        "open-seed",
        "schema-discover_sources",
        "schema-search_sources",
        "schema-seed",
    ]
    discoveries = {
        f"tool:{request_id}": DiscoveryBatch(
            request_id=f"tool:{request_id}",
            adapter_id="replay",
            adapter_version="1",
            occurrences=[occurrence],
        )
        for request_id in requests
    }
    return ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries=discoveries,
        hydrations={
            SOURCE: ObservationBatch(
                request_id=f"tool:hydrate:{observation_id('replay', SOURCE)}",
                adapter_id="replay",
                adapter_version="1",
                observations=[_observation()],
            )
        },
    )


def _tools(tmp_path: pytest.TempPathFactory, *, thread_id: str = "") -> tuple[WorldTools, WorldStore]:
    store = WorldStore(tmp_path / "world.sqlite3")
    return (
        WorldTools(store=store, adapters={"replay": _adapter()}, thread_id=thread_id),
        store,
    )


async def _anchor(tools: WorldTools, call_id: str) -> str:
    result = await tools.execute("discover_sources", {"adapter": "replay", "query": "source"}, call_id)
    return str(result["cards"][0]["id"])
