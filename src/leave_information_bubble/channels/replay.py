"""Deterministic in-memory channel adapter for replay and contract evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from .models import (
    ChannelHealth,
    ChannelHealthStatus,
    DiscoveryBatch,
    HydrationRequest,
    ObservationBatch,
    ScanRequest,
)


class ReplayChannelAdapter:
    """Replay immutable discovery and hydration fixtures without network access."""

    def __init__(
        self,
        *,
        adapter_id: str,
        adapter_version: str,
        discoveries: Mapping[str, DiscoveryBatch],
        hydrations: Mapping[str, ObservationBatch],
        failures: Mapping[str, Exception] | None = None,
        health: ChannelHealth | None = None,
    ) -> None:
        if not adapter_id.strip():
            raise ValueError("adapter_id is required")
        if not adapter_version.strip():
            raise ValueError("adapter_version is required")
        self._adapter_id = adapter_id
        self._adapter_version = adapter_version
        self._discoveries = dict(discoveries)
        self._hydrations = dict(hydrations)
        self._failures = dict(failures or {})
        self._health = health or ChannelHealth(
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            status=ChannelHealthStatus.HEALTHY,
            checked_at=datetime(1970, 1, 1, tzinfo=UTC),
        )
        self._validate_fixtures()
        self.discovery_calls: list[str] = []
        self.hydration_calls: list[str] = []
        self.hydration_requests: list[HydrationRequest] = []
        self.change_calls: list[str] = []

    @property
    def adapter_id(self) -> str:
        """Return the stable fixture adapter identifier."""
        return self._adapter_id

    @property
    def adapter_version(self) -> str:
        """Return the fixture schema and implementation version."""
        return self._adapter_version

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Replay the discovery batch registered for the request identifier."""
        self.discovery_calls.append(request.id)
        self._raise_planned(f"discover:{request.id}")
        try:
            return self._discoveries[request.id]
        except KeyError as exc:
            raise LookupError(f"no replay discovery for request {request.id}") from exc

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Replay observations registered for the exact source reference."""
        self.hydration_calls.append(request.source_ref)
        self.hydration_requests.append(request)
        self._raise_planned(f"hydrate:{request.source_ref}")
        try:
            return self._hydrations[request.source_ref]
        except KeyError as exc:
            raise LookupError(f"no replay hydration for source {request.source_ref}") from exc

    async def changes_since(self, request: ScanRequest) -> DiscoveryBatch:
        """Replay a cursor-based change batch using its request identifier."""
        self.change_calls.append(request.id)
        self._raise_planned(f"changes:{request.id}")
        try:
            return self._discoveries[request.id]
        except KeyError as exc:
            raise LookupError(f"no replay changes for request {request.id}") from exc

    async def health(self) -> ChannelHealth:
        """Return the fixed fixture health state."""
        self._raise_planned("health")
        return self._health

    def _raise_planned(self, operation: str) -> None:
        failure = self._failures.get(operation)
        if failure is not None:
            raise failure

    def _validate_fixtures(self) -> None:
        if (
            self._health.adapter_id != self.adapter_id
            or self._health.adapter_version != self.adapter_version
        ):
            raise ValueError("replay health provenance does not match adapter")
        for request_id, batch in self._discoveries.items():
            if request_id != batch.request_id:
                raise ValueError("replay discovery key does not match request_id")
            if (
                batch.adapter_id != self.adapter_id
                or batch.adapter_version != self.adapter_version
                or any(
                    occurrence.adapter_id != self.adapter_id
                    or occurrence.adapter_version != self.adapter_version
                    for occurrence in batch.occurrences
                )
            ):
                raise ValueError("replay discovery provenance does not match adapter")
        if any(
            batch.adapter_id != self.adapter_id
            or batch.adapter_version != self.adapter_version
            for batch in self._hydrations.values()
        ):
            raise ValueError("replay hydration provenance does not match adapter")


__all__ = ["ReplayChannelAdapter"]
