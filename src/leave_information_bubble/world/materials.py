"""Versioned, provenance-preserving stored body materials.

The world schema already has ``observation_bodies.body_json``.  This module
keeps its v2 payload deliberately small and self-validating so body material
can evolve without another world-schema migration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

MATERIAL_SCHEMA_VERSION = 2
MATERIAL_PART_SEPARATOR = "\n\n"
MATERIAL_RELIABILITIES = frozenset(
    {"source_direct", "confirmed", "best_effort", "automatic", "unknown"}
)


@dataclass(frozen=True)
class MaterialPart:
    """One exact, durable source contribution to a stored body."""

    observation_id: str
    start_char: int
    end_char: int
    kind: str
    location: str
    acquisition_method: str
    confidence: float
    reliability: str
    sampling_scope: str
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation of this part."""
        value = asdict(self)
        value["limitations"] = list(self.limitations)
        return value


@dataclass(frozen=True)
class MaterialPartInput:
    """The raw text and provenance used to build one material part."""

    observation_id: str
    text: str
    kind: str
    location: str
    acquisition_method: str
    confidence: float
    reliability: str
    sampling_scope: str
    limitations: tuple[str, ...]
    captured_at: datetime


@dataclass(frozen=True)
class BodyEnvelope:
    """The v2 stored-body envelope with exact text and part offsets."""

    text: str
    truncated: bool
    captured_at: datetime
    material_hash: str
    source_revision: str | None
    source_revision_kind: str | None
    quality_flags: tuple[str, ...]
    parts: tuple[MaterialPart, ...]
    schema_version: int = MATERIAL_SCHEMA_VERSION

    def to_json(self) -> str:
        """Serialize the envelope without changing model-visible text."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "text": self.text,
                "truncated": self.truncated,
                "captured_at": self.captured_at.isoformat(),
                "material_hash": self.material_hash,
                "source_revision": self.source_revision,
                "source_revision_kind": self.source_revision_kind,
                "quality_flags": list(self.quality_flags),
                "parts": [part.as_dict() for part in self.parts],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class LegacyBody:
    """A readable pre-v2 body whose provenance cannot be verified."""

    text: str
    truncated: bool


StoredBody = BodyEnvelope | LegacyBody


def build_body_envelope(
    inputs: Iterable[MaterialPartInput],
    *,
    max_chars: int,
    source_revision: str | None = None,
    source_revision_kind: str | None = None,
) -> BodyEnvelope | None:
    """Build a deterministic v2 envelope from non-blank body contributions.

    Whitespace-only contributions are absent material.  Non-blank text is
    retained byte-for-byte; no stripping or normalization is performed.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if (source_revision is None) != (source_revision_kind is None):
        raise ValueError("source revision and kind must be present together")
    candidates = [item for item in inputs if item.text.strip()]
    if not candidates:
        return None
    for item in candidates:
        if item.reliability not in MATERIAL_RELIABILITIES:
            raise ValueError("material part reliability is unsupported")
        if not 0.0 <= item.confidence <= 1.0:
            raise ValueError("material part confidence must be between zero and one")

    text = ""
    parts: list[MaterialPart] = []
    retained_inputs: list[MaterialPartInput] = []
    truncated = False
    for item in candidates:
        prefix = MATERIAL_PART_SEPARATOR if text else ""
        available = max_chars - len(text)
        if available <= 0:
            truncated = True
            break
        if prefix and available <= len(prefix):
            truncated = True
            break
        if prefix:
            text += prefix
        available = max_chars - len(text)
        part_text = item.text[:available]
        part_limitations = item.limitations
        if len(part_text) < len(item.text):
            truncated = True
            part_limitations = tuple(dict.fromkeys([*part_limitations, "body_truncated"]))
        start = len(text)
        text += part_text
        retained_inputs.append(item)
        parts.append(
            MaterialPart(
                observation_id=item.observation_id,
                start_char=start,
                end_char=len(text),
                kind=item.kind,
                location=item.location,
                acquisition_method=item.acquisition_method,
                confidence=item.confidence,
                reliability=item.reliability,
                sampling_scope=item.sampling_scope,
                limitations=part_limitations,
            )
        )
        if truncated:
            break

    flags: list[str] = []
    if len(text) < 100:
        flags.append("short_text")
    if truncated:
        flags.append("body_truncated")
    reliabilities = {part.reliability for part in parts}
    if len(reliabilities) > 1:
        flags.append("mixed_reliability")
    captured_at = max(item.captured_at for item in retained_inputs)
    return BodyEnvelope(
        text=text,
        truncated=truncated,
        captured_at=_utc(captured_at),
        material_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_revision=source_revision,
        source_revision_kind=source_revision_kind,
        quality_flags=tuple(flags),
        parts=tuple(parts),
    )


def parse_stored_body(body_json: str) -> StoredBody:
    """Parse a v2 envelope or return a deliberately marked legacy payload."""
    try:
        raw = json.loads(body_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("stored body is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("stored body must be an object")
    schema_version = raw.get("schema_version")
    if schema_version not in {None, 1, MATERIAL_SCHEMA_VERSION}:
        raise ValueError("stored body schema_version is unsupported")
    if schema_version != MATERIAL_SCHEMA_VERSION:
        return LegacyBody(text=str(raw.get("text", "")), truncated=bool(raw.get("truncated", False)))
    return _parse_v2(raw)


def is_verifiable_content_body(value: StoredBody) -> bool:
    """Return whether a body may justify a CONTENT depth upgrade."""
    return isinstance(value, BodyEnvelope) and bool(value.parts)


def _parse_v2(raw: dict[str, Any]) -> BodyEnvelope:
    text = raw.get("text")
    if not isinstance(text, str):
        raise ValueError("v2 body text must be a string")
    material_hash = raw.get("material_hash")
    expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if material_hash != expected_hash:
        raise ValueError("v2 body material_hash does not match exact text")
    truncated = raw.get("truncated")
    if not isinstance(truncated, bool):
        raise ValueError("v2 body truncated must be a boolean")
    captured_at = raw.get("captured_at")
    if not isinstance(captured_at, str):
        raise ValueError("v2 body captured_at must be an ISO timestamp")
    try:
        parsed_at = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("v2 body captured_at is invalid") from exc
    if parsed_at.tzinfo is None:
        raise ValueError("v2 body captured_at must be timezone aware")
    raw_parts = raw.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError("v2 body must contain parts")
    parts = tuple(_parse_part(part, text) for part in raw_parts)
    cursor = 0
    for index, part in enumerate(parts):
        if index == 0 and part.start_char != 0:
            raise ValueError("v2 body first part must start at zero")
        if part.start_char < cursor or part.end_char < part.start_char:
            raise ValueError("v2 body part offsets overlap or reverse")
        if index and text[cursor:part.start_char] != MATERIAL_PART_SEPARATOR:
            raise ValueError("v2 body part separator is not deterministic")
        cursor = part.end_char
    if cursor != len(text):
        raise ValueError("v2 body parts must cover the exact text")
    flags = raw.get("quality_flags", [])
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise ValueError("v2 body quality_flags must be strings")
    revision = raw.get("source_revision")
    revision_kind = raw.get("source_revision_kind")
    if revision is not None and not isinstance(revision, str):
        raise ValueError("v2 body source_revision must be a string or null")
    if revision_kind is not None and not isinstance(revision_kind, str):
        raise ValueError("v2 body source_revision_kind must be a string or null")
    if (revision is None) != (revision_kind is None):
        raise ValueError("v2 body source revision and kind must be present together")
    return BodyEnvelope(
        text=text,
        truncated=truncated,
        captured_at=_utc(parsed_at),
        material_hash=material_hash,
        source_revision=revision,
        source_revision_kind=revision_kind,
        quality_flags=tuple(flags),
        parts=parts,
    )


def _parse_part(raw: object, text: str) -> MaterialPart:
    if not isinstance(raw, dict):
        raise ValueError("v2 body part must be an object")
    required_strings = (
        "observation_id",
        "kind",
        "location",
        "acquisition_method",
        "reliability",
        "sampling_scope",
    )
    values: dict[str, str] = {}
    for field in required_strings:
        value = raw.get(field)
        if not isinstance(value, str):
            raise ValueError(f"v2 body part {field} must be a string")
        values[field] = value
    start = raw.get("start_char")
    end = raw.get("end_char")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or end > len(text)
    ):
        raise ValueError("v2 body part offsets are outside text")
    confidence = raw.get("confidence")
    if not isinstance(confidence, (float, int)) or isinstance(confidence, bool):
        raise ValueError("v2 body part confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("v2 body part confidence must be between zero and one")
    if not values["observation_id"]:
        raise ValueError("v2 body part observation_id must be non-empty")
    if values["reliability"] not in MATERIAL_RELIABILITIES:
        raise ValueError("v2 body part reliability is unsupported")
    limitations = raw.get("limitations", [])
    if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
        raise ValueError("v2 body part limitations must be strings")
    return MaterialPart(
        observation_id=values["observation_id"],
        start_char=start,
        end_char=end,
        kind=values["kind"],
        location=values["location"],
        acquisition_method=values["acquisition_method"],
        confidence=float(confidence),
        reliability=values["reliability"],
        sampling_scope=values["sampling_scope"],
        limitations=tuple(limitations),
    )


def _utc(value: datetime) -> datetime:
    """Normalize one material timestamp to UTC without changing its instant."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
