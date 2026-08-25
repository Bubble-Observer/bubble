"""Task 6: per-domain adapter surface parameters and no-domain fallback.

A bare LOL-domain run keeps the built-in board/fid; any other domain
without an explicit profile configuration must degrade to a typed
no-default limitation instead of silently scanning the LOL board.
surface_key threads through discover_sources/search_sources into the
adapter's ScanRequest.arguments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from leave_information_bubble.channels import (
    AcquisitionOutcome,
    ChannelCapabilityRole,
    DiscoveryBatch,
    ScanRequest,
)
from leave_information_bubble.config import Settings
from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world_agent import cli as runner

_OFFLINE_SETTINGS = Settings(asr_enabled=False)


def _hupu_adapter(adapters: dict[str, object]) -> object:
    return adapters["hupu"]


def _nga_adapter(adapters: dict[str, object]) -> object:
    return adapters["nga"]


def test_live_lol_domain_keeps_builtin_board_defaults() -> None:
    """A bare LOL-domain CLI run keeps hupu 'lol' and nga '-152678'."""
    adapters = runner._live(
        "hupu,nga", _OFFLINE_SETTINGS, domain_key="lol_cn"
    )
    assert _hupu_adapter(adapters)._default_board == "lol"
    assert _nga_adapter(adapters)._default_fid == "-152678"


def test_live_other_domain_without_defaults_degrades() -> None:
    """A bare CS-domain CLI run must NOT scan the LOL board: both adapters
    degrade to no-default (this was the LOL-leak root cause)."""
    adapters = runner._live(
        "hupu,nga", _OFFLINE_SETTINGS, domain_key="cs"
    )
    assert _hupu_adapter(adapters)._default_board is None
    assert _nga_adapter(adapters)._default_fid is None
    community = next(
        item
        for item in _hupu_adapter(adapters).capability_descriptors
        if item.role is ChannelCapabilityRole.COMMUNITY_STREAM
    )
    assert community.supports_queryless is False


def test_live_profile_defaults_override_builtin() -> None:
    """Profile adapter_defaults take precedence over the built-in LOL board."""
    adapters = runner._live(
        "hupu,nga",
        _OFFLINE_SETTINGS,
        adapter_defaults={"hupu": {"board": "csgo"}, "nga": {"fid": "335"}},
        domain_key="lol_cn",
    )
    assert _hupu_adapter(adapters)._default_board == "csgo"
    assert _nga_adapter(adapters)._default_fid == "335"


def test_live_partial_defaults_leave_others_degraded() -> None:
    """An entry for one adapter must not resurrect the LOL default for
    another adapter (cs profile configures hupu only)."""
    adapters = runner._live(
        "hupu,nga",
        _OFFLINE_SETTINGS,
        adapter_defaults={"hupu": {"board": "csgo"}},
        domain_key="cs",
    )
    assert _hupu_adapter(adapters)._default_board == "csgo"
    assert _nga_adapter(adapters)._default_fid is None


def test_live_rejects_unknown_adapter_defaults() -> None:
    with pytest.raises(ValueError, match="unknown adapter defaults"):
        runner._live(
            "hupu",
            _OFFLINE_SETTINGS,
            adapter_defaults={"hupu": {"board": "csgo"}, "typo": {}},
        )


def test_live_empty_configured_board_degrades() -> None:
    """An explicitly empty board value means no default, not the LOL board."""
    adapters = runner._live(
        "hupu",
        _OFFLINE_SETTINGS,
        adapter_defaults={"hupu": {"board": ""}},
        domain_key="lol_cn",
    )
    assert _hupu_adapter(adapters)._default_board is None


def test_cli_parser_accepts_adapter_defaults_json() -> None:
    namespace = runner.parse_args(
        [
            "--thread-id",
            "t",
            "--world-db",
            "data/w.sqlite3",
            "--runtime-db",
            "data/r.sqlite3",
            "--adapter-defaults",
            '{"hupu": {"board": "csgo"}}',
        ]
    )
    assert namespace.adapter_defaults == {"hupu": {"board": "csgo"}}


# --- surface_key passthrough through the WorldTools scan path ---------------


class _RecordingHupuAdapter:
    """Records the ScanRequest it receives and answers a bounded batch."""

    adapter_id = "hupu"
    adapter_version = "1.0.0"

    def __init__(self) -> None:
        self.requests: list[ScanRequest] = []
        self.surface_key: str | None = None

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        self.requests.append(request)
        self.surface_key = request.arguments.get("surface_key")
        return DiscoveryBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=AcquisitionOutcome.SUCCESS,
        )


async def test_surface_key_threads_through_scan_arguments(tmp_path: Path) -> None:
    """discover_sources surface_key reaches the adapter ScanRequest.arguments."""
    adapter = _RecordingHupuAdapter()
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={"hupu": adapter},  # type: ignore[arg-type]
        thread_id="surface-thread",
        wake_id="surface-wake",
    )
    result = await tools.execute(
        "discover_sources",
        {"adapter": "hupu", "surface_key": "csgo"},
        "call-1",
    )
    assert result.get("ok") is True
    assert adapter.surface_key == "csgo"
    assert adapter.requests[0].arguments == {"surface_key": "csgo"}


async def test_scan_without_surface_key_leaves_arguments_empty(tmp_path: Path) -> None:
    adapter = _RecordingHupuAdapter()
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={"hupu": adapter},  # type: ignore[arg-type]
        thread_id="surface-thread",
        wake_id="surface-wake",
    )
    result = await tools.execute("discover_sources", {"adapter": "hupu"}, "call-2")
    assert result.get("ok") is True
    assert adapter.requests[0].arguments == {}
