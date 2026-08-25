"""Bounded, declaration-only summaries of adapter scan capabilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from leave_information_bubble.channels import (
    AcquisitionEntryKind,
    CapabilityDescriptor,
    ChannelAdapter,
    QuerySemantics,
)

_DISCOVERY_ENTRY_KINDS = frozenset(
    {
        AcquisitionEntryKind.PLATFORM_SEARCH,
        AcquisitionEntryKind.RANKED_SURFACE,
        AcquisitionEntryKind.BOUNDED_BOARD,
    }
)
_LINE_CAP = 500
_BLOCK_CAP = 2400
_ROLE_CAP = 7
_LIMITATION_CAP = 4
_FIELD_TEXT_CAP = 80


def supports_targeted_search(adapter: ChannelAdapter) -> bool | None:
    """Return declared targeted-search support without guessing legacy adapters."""
    descriptors = _descriptors(adapter)
    if not descriptors:
        return None
    return any(
        descriptor.entry_kind in _DISCOVERY_ENTRY_KINDS
        and descriptor.query_semantics is QuerySemantics.PLATFORM_SEARCH
        for descriptor in descriptors
    )


def render_scan_capabilities(adapters: Mapping[str, ChannelAdapter]) -> str:
    """Render sorted complete lines, with a factual marker when the block fills."""
    items = sorted(adapters.items(), key=lambda item: str(item[0]))
    rendered = [_render_adapter(adapter_id, adapter) for adapter_id, adapter in items]
    lines: list[str] = []
    used = 0
    for index, line in enumerate(rendered):
        addition = len(line) + (1 if lines else 0)
        if used + addition <= _BLOCK_CAP:
            lines.append(line)
            used += addition
            continue
        omitted = len(rendered) - index
        marker = f"omitted_adapters={omitted}"
        while lines and used + 1 + len(marker) > _BLOCK_CAP:
            removed = lines.pop()
            used -= len(removed) + (1 if lines else 0)
            omitted += 1
            marker = f"omitted_adapters={omitted}"
        lines.append(marker)
        break
    return "\n".join(lines)


def scan_schema_contracts(
    adapters: Mapping[str, ChannelAdapter],
) -> dict[str, dict[str, object]]:
    """Project adapter declarations into machine-enforceable scan facts.

    Unknown/legacy adapters remain explicitly unclassified; callers can keep
    a compatibility schema for them instead of inventing capabilities.
    """
    contracts: dict[str, dict[str, object]] = {}
    for adapter_id, adapter in adapters.items():
        discovery = tuple(
            descriptor
            for descriptor in _descriptors(adapter)
            if descriptor.entry_kind in _DISCOVERY_ENTRY_KINDS
        )
        if not discovery:
            contracts[str(adapter_id)] = {"classified": False}
            continue
        contracts[str(adapter_id)] = {
            "classified": True,
            "roles": _unique(descriptor.role.value for descriptor in discovery),
            "query_required": all(not descriptor.supports_queryless for descriptor in discovery),
            "supports_cursor": any(descriptor.supports_cursor for descriptor in discovery),
            "supports_surface_key": any(
                descriptor.entry_kind is AcquisitionEntryKind.BOUNDED_BOARD for descriptor in discovery
            ),
            "targeted_search": supports_targeted_search(adapter) is True,
        }
    return contracts


def _render_adapter(adapter_id: str, adapter: ChannelAdapter) -> str:
    label = _bounded_text(adapter_id, 120)
    descriptors = _descriptors(adapter)
    if not descriptors:
        return f"{label}: capability_unclassified"
    discovery = tuple(
        descriptor for descriptor in descriptors if descriptor.entry_kind in _DISCOVERY_ENTRY_KINDS
    )
    if not discovery:
        return f"{label}: discovery_capability_unsupported; targeted_search=no"

    entry_kinds = _unique(descriptor.entry_kind.value for descriptor in discovery)
    query_semantics = _unique(descriptor.query_semantics.value for descriptor in discovery)
    all_roles = _unique(descriptor.role.value for descriptor in discovery)
    roles = all_roles[:_ROLE_CAP]
    time_precision = _unique(descriptor.time_filter_precision.value for descriptor in discovery)
    limitations = _unique(
        _bounded_text(limitation, _FIELD_TEXT_CAP)
        for descriptor in discovery
        for limitation in descriptor.limitations
        if _bounded_text(limitation, _FIELD_TEXT_CAP)
    )[:_LIMITATION_CAP]
    queryless = {descriptor.supports_queryless for descriptor in discovery}
    if queryless == {False}:
        query_support = "no (query required)"
    elif queryless == {True}:
        query_support = "yes"
    else:
        query_support = "mixed"
    targeted = supports_targeted_search(adapter)
    cursor = any(item.supports_cursor for item in discovery)
    fields = [
        f"entry={','.join(entry_kinds)}",
        f"query={','.join(query_semantics)}",
        f"queryless={query_support}",
        f"targeted_search={'yes' if targeted else 'no'}",
        f"roles={','.join(roles)}",
        f"time={','.join(time_precision)}",
        f"cursor={'yes' if cursor else 'no'}",
    ]
    line = f"{label}: " + "; ".join(fields)
    for limitation in limitations:
        candidate = (
            f"{line}; limitations={limitation}" if "limitations=" not in line else f"{line},{limitation}"
        )
        if len(candidate) > _LINE_CAP:
            break
        line = candidate
    return line


def _descriptors(adapter: ChannelAdapter) -> tuple[CapabilityDescriptor, ...]:
    return tuple(
        descriptor
        for descriptor in getattr(adapter, "capability_descriptors", ())
        if isinstance(descriptor, CapabilityDescriptor)
    )


def _unique(values: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _bounded_text(value: object, limit: int) -> str:
    return _normalized_text(value)[:limit]


def _normalized_text(value: object) -> str:
    return " ".join(str(value).split())


__all__ = ["render_scan_capabilities", "scan_schema_contracts", "supports_targeted_search"]
