"""Minimal protocol implemented by every information channel adapter."""

from __future__ import annotations

from typing import Protocol

from .models import (
    ChannelHealth,
    DiscoveryBatch,
    HydrationRequest,
    ObservationBatch,
    ScanRequest,
)


class ChannelAdapter(Protocol):
    """Platform-neutral asynchronous discovery and hydration boundary."""

    @property
    def adapter_id(self) -> str:
        """Return a stable adapter identifier."""
        ...

    @property
    def adapter_version(self) -> str:
        """Return the implementation version used for provenance and replay."""
        ...

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Discover bounded source occurrences."""
        ...

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Read one source to the requested bounded depth."""
        ...

    async def changes_since(self, request: ScanRequest) -> DiscoveryBatch:
        """Return changes after a previously issued adapter cursor."""
        ...

    async def health(self) -> ChannelHealth:
        """Return current adapter availability without raising."""
        ...


__all__ = ["ChannelAdapter"]
