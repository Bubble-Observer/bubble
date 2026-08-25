"""Bounded public graph vocabulary with no world-module dependencies.

Every layer (proposal, store, durable delta) imports normalization and
identity terms from here so that one shared rule set defines alias folding,
type keys, and qualifiers.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from enum import StrEnum


class AliasAction(StrEnum):
    """Durable identity-alias operation recorded in a graph delta."""

    ADD = "add"
    REMOVE = "remove"
    DEMOTE = "demote"


# Public bounds shared by the normalizers and the provider-facing schema, so
# the model-visible schema cannot drift from what the compiler validates.
TYPE_KEY_PATTERN = r"[a-z][a-z0-9._-]{0,63}"
QUALIFIER_KEYS = frozenset({"role", "language", "community", "scope", "granularity"})

_TYPE_KEY = re.compile(TYPE_KEY_PATTERN + r"\Z")


def normalize_identity_alias(raw: str) -> str:
    """Fold an identity alias to the canonical bounded form."""
    if not isinstance(raw, str):
        raise ValueError("identity alias must be a string")
    normalized = unicodedata.normalize("NFC", raw).casefold()
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("identity alias must be non-empty")
    return normalized


def normalize_type_key(raw: str | None) -> str | None:
    """Strip and casefold a bounded type key, or reject one outside the pattern."""
    if raw is None:
        return None
    normalized = raw.strip().casefold()
    if not normalized:
        return None
    if _TYPE_KEY.fullmatch(normalized) is None:
        raise ValueError("type_key must match [a-z][a-z0-9._-]{0,63}")
    return normalized


def normalize_qualifiers(raw: Mapping[str, object] | None) -> dict[str, str]:
    """Bound qualifier keys and values, returning the canonical trimmed form."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("qualifiers must be an object")
    if not raw:
        return {}
    if len(raw) > 5 or not set(raw).issubset(QUALIFIER_KEYS):
        raise ValueError("qualifiers contain unknown or excessive keys")
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 120:
            raise ValueError(f"invalid qualifier value for {key}")
        normalized[key] = value.strip()
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) > 768:
        raise ValueError("qualifiers exceed serialized limit")
    return normalized
