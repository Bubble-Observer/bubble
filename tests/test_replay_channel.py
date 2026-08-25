from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leave_information_bubble.channels import (
    DiscoveryBatch,
    HydrationDepth,
    HydrationRequest,
    ObservationBatch,
    ReplayChannelAdapter,
    ScanRequest,
    SourceOccurrence,
)

NOW = datetime(2026, 7, 20, tzinfo=UTC)


def _request(identifier: str) -> ScanRequest:
    return ScanRequest(
        id=identifier,
        lane="global",
        window_start=NOW,
        window_end=NOW,
    )


async def test_replay_adapter_replays_discover_changes_hydrate_and_health() -> None:
    discovery = DiscoveryBatch(
        request_id="scan",
        adapter_id="fixture",
        adapter_version="1",
    )
    observation = ObservationBatch(
        request_id="hydrate",
        adapter_id="fixture",
        adapter_version="1",
    )
    adapter = ReplayChannelAdapter(
        adapter_id="fixture",
        adapter_version="1",
        discoveries={"scan": discovery},
        hydrations={"source": observation},
    )

    assert await adapter.discover(_request("scan")) is discovery
    assert await adapter.changes_since(_request("scan")) is discovery
    assert (
        await adapter.hydrate(
            HydrationRequest(
                id="hydrate",
                source_ref="source",
                depth=HydrationDepth.TEXT,
            )
        )
        is observation
    )
    assert (await adapter.health()).adapter_id == "fixture"
    assert adapter.discovery_calls == ["scan"]
    assert adapter.change_calls == ["scan"]
    assert adapter.hydration_calls == ["source"]


async def test_replay_adapter_exposes_missing_fixture_and_planned_failure() -> None:
    adapter = ReplayChannelAdapter(
        adapter_id="fixture",
        adapter_version="1",
        discoveries={},
        hydrations={},
        failures={"discover:broken": TimeoutError("planned")},
    )

    with pytest.raises(TimeoutError, match="planned"):
        await adapter.discover(_request("broken"))
    with pytest.raises(LookupError, match="no replay discovery"):
        await adapter.discover(_request("missing"))
    with pytest.raises(LookupError, match="no replay hydration"):
        await adapter.hydrate(
            HydrationRequest(
                id="hydrate",
                source_ref="missing",
                depth=HydrationDepth.TEXT,
            )
        )


def test_replay_adapter_requires_identity() -> None:
    with pytest.raises(ValueError, match="adapter_id"):
        ReplayChannelAdapter(
            adapter_id="",
            adapter_version="1",
            discoveries={},
            hydrations={},
        )
    with pytest.raises(ValueError, match="adapter_version"):
        ReplayChannelAdapter(
            adapter_id="fixture",
            adapter_version="",
            discoveries={},
            hydrations={},
        )


def test_replay_adapter_rejects_forged_occurrence_provenance() -> None:
    forged = SourceOccurrence(
        id="occ",
        adapter_id="other",
        adapter_version="1",
        source_ref="item",
        captured_at=NOW,
    )
    with pytest.raises(ValueError, match="provenance"):
        ReplayChannelAdapter(
            adapter_id="fixture",
            adapter_version="1",
            discoveries={
                "scan": DiscoveryBatch(
                    request_id="scan",
                    adapter_id="fixture",
                    adapter_version="1",
                    occurrences=[forged],
                )
            },
            hydrations={},
        )
