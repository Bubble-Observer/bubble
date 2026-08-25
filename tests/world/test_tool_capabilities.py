from __future__ import annotations

import jsonschema

from leave_information_bubble.channels import (
    AcquisitionEntryKind,
    BilibiliChannelAdapter,
    CapabilityDescriptor,
    ChannelCapabilityRole,
    HupuChannelAdapter,
    NgaChannelAdapter,
    PublicWebChannelAdapter,
    QuerySemantics,
    ReplayChannelAdapter,
    TimeFilterPrecision,
)
from leave_information_bubble.world import WorldStore, WorldTools
from leave_information_bubble.world.tool_capabilities import (
    render_scan_capabilities,
    scan_schema_contracts,
    supports_targeted_search,
)


def _bilibili_adapter() -> BilibiliChannelAdapter:
    return BilibiliChannelAdapter(object())  # type: ignore[arg-type]


def _hupu_adapter() -> HupuChannelAdapter:
    return HupuChannelAdapter(object())  # type: ignore[arg-type]


def _bounded_adapter() -> object:
    descriptors = tuple(
        CapabilityDescriptor(
            adapter_id="bounded",
            adapter_version="1",
            role=role,
            entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
            query_semantics=QuerySemantics.PLATFORM_SEARCH,
            supports_queryless=False,
            supports_cursor=True,
            time_filter_precision=TimeFilterPrecision.EXACT,
            limitations=[f"limitation-{index}-" + "x" * 200 for index in range(8)],
        )
        for role in ChannelCapabilityRole
    )

    class _BoundedAdapter:
        capability_descriptors = descriptors

    return _BoundedAdapter()


def test_renderer_distinguishes_true_search_from_board_hint() -> None:
    rendered = render_scan_capabilities(
        {
            "bilibili": _bilibili_adapter(),
            "hupu": _hupu_adapter(),
        }
    )

    assert "bilibili" in rendered and "platform_search" in rendered
    assert "query required" in rendered
    assert "hupu" in rendered and "bounded_board" in rendered
    assert "bounded_rerank_hint" in rendered
    assert "targeted search" not in rendered.split("hupu", 1)[1]


def test_renderer_marks_descriptor_free_adapter_unclassified() -> None:
    adapter = ReplayChannelAdapter(
        adapter_id="replay",
        adapter_version="1",
        discoveries={},
        hydrations={},
    )

    assert render_scan_capabilities({"replay": adapter}) == "replay: capability_unclassified"
    assert supports_targeted_search(adapter) is None


def test_renderer_keeps_real_enabled_adapter_lines_complete() -> None:
    rendered = render_scan_capabilities(
        {
            "bilibili": _bilibili_adapter(),
            "nga": NgaChannelAdapter(object()),  # type: ignore[arg-type]
            "hupu": _hupu_adapter(),
            "public-web": PublicWebChannelAdapter(object()),  # type: ignore[arg-type]
        }
    )
    lines = rendered.splitlines()

    assert [line.split(":", 1)[0] for line in lines] == ["bilibili", "hupu", "nga", "public-web"]
    assert all(len(line) <= 500 for line in lines)
    assert all("roles=" in line and "time=" in line and "cursor=" in line for line in lines)
    assert len(rendered) <= 2400


def test_renderer_bounds_lines_block_roles_and_limitations() -> None:
    rendered = render_scan_capabilities({"bounded": _bounded_adapter()})  # type: ignore[dict-item]

    assert len(rendered) <= 2400
    assert all(len(line) <= 500 for line in rendered.splitlines())
    assert all("roles=" in line and "time=" in line and "cursor=" in line for line in rendered.splitlines())
    assert all(line.count("limitation-") <= 4 for line in rendered.splitlines())
    assert all(line.count("_stream") + line.count("_index") <= 7 for line in rendered.splitlines())


def test_renderer_omits_only_whole_lines_when_block_is_full() -> None:
    adapters = {f"adapter-{index:02d}": _bounded_adapter() for index in range(12)}

    rendered = render_scan_capabilities(adapters)  # type: ignore[arg-type]
    lines = rendered.splitlines()

    assert lines[-1] == f"omitted_adapters={len(adapters) - len(lines) + 1}"
    assert len(lines) - 1 < len(adapters)
    assert len(rendered) <= 2400
    for line in lines[:-1]:
        assert len(line) <= 500
        assert "roles=" in line and "time=" in line and "cursor=" in line


def test_targeted_search_support_is_true_only_for_declared_platform_search() -> None:
    assert supports_targeted_search(_bilibili_adapter()) is True
    assert supports_targeted_search(_hupu_adapter()) is False


def test_machine_scan_contracts_match_real_adapter_declarations() -> None:
    contracts = scan_schema_contracts(
        {
            "bilibili": _bilibili_adapter(),
            "hupu": _hupu_adapter(),
            "nga": NgaChannelAdapter(object()),  # type: ignore[arg-type]
            "public-web": PublicWebChannelAdapter(object()),  # type: ignore[arg-type]
        }
    )

    assert contracts["bilibili"] == {
        "classified": True,
        "roles": ["platform_discovery", "attention_ranking", "recent_stream"],
        "query_required": True,
        "supports_cursor": False,
        "supports_surface_key": False,
        "targeted_search": True,
    }
    assert contracts["hupu"]["roles"] == ["community_stream"]
    assert contracts["hupu"]["supports_cursor"] is True
    assert contracts["hupu"]["supports_surface_key"] is True
    assert contracts["hupu"]["targeted_search"] is False


def test_scan_schemas_enforce_adapter_specific_external_capabilities() -> None:
    adapters = {
        "bilibili": _bilibili_adapter(),
        "public-web": PublicWebChannelAdapter(object()),  # type: ignore[arg-type]
        "hupu": _hupu_adapter(),
        "nga": NgaChannelAdapter(object()),  # type: ignore[arg-type]
    }
    tools = WorldTools(store=WorldStore(":memory:"), adapters=adapters)
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in tools.schemas()}
    discover = jsonschema.Draft202012Validator(
        schemas["discover_sources"], format_checker=jsonschema.FormatChecker()
    )

    assert discover.is_valid(
        {
            "adapter": "bilibili",
            "query": "LPL",
            "surface_role": "recent_stream",
        }
    )
    assert not discover.is_valid({"adapter": "bilibili"})
    assert not discover.is_valid({"adapter": "bilibili", "query": "LPL", "cursor": "page-2"})
    assert not discover.is_valid({"adapter": "bilibili", "query": "LPL", "surface_key": "lol"})
    assert not discover.is_valid(
        {
            "adapter": "bilibili",
            "query": "LPL",
            "surface_role": "community_stream",
        }
    )
    assert discover.is_valid(
        {
            "adapter": "hupu",
            "surface_role": "community_stream",
            "surface_key": "lol",
            "cursor": "page-2",
        }
    )
    assert not discover.is_valid({"adapter": "hupu", "surface_role": "attention_ranking"})
    assert not discover.is_valid({"adapter": "public-web", "query": "LPL", "cursor": "page-2"})
    assert not discover.is_valid({"adapter": "public-web", "query": "LPL", "domain_ids": ["lol"]})
    assert not discover.is_valid(
        {
            "adapter": "public-web",
            "query": "LPL",
            "window_start": "2026-08-24T00:00:00Z",
        }
    )

    search = schemas["search_sources"]
    assert search["properties"]["adapter"]["enum"] == [
        "bilibili",
        "public-web",
    ]
    assert len(
        next(
            item["function"]["description"]
            for item in tools.schemas()
            if item["function"]["name"] == "search_sources"
        )
    ) < len(
        next(
            item["function"]["description"]
            for item in tools.schemas()
            if item["function"]["name"] == "discover_sources"
        )
    )
