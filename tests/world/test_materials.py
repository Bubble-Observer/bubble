"""Stored body v2 envelope identity, offsets, and legacy compatibility."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from leave_information_bubble.world.materials import (
    MATERIAL_PART_SEPARATOR,
    BodyEnvelope,
    LegacyBody,
    MaterialPartInput,
    build_body_envelope,
    parse_stored_body,
)

CAPTURED = datetime(2026, 8, 11, 1, 2, 3, tzinfo=UTC)


def _part(
    observation_id: str,
    text: str,
    *,
    reliability: str = "source_direct",
    method: str = "bilibili_video_description",
) -> MaterialPartInput:
    return MaterialPartInput(
        observation_id=observation_id,
        text=text,
        kind="description" if reliability == "source_direct" else "transcript",
        location="video_description" if reliability == "source_direct" else "video_transcript:full",
        acquisition_method=method,
        confidence=0.91,
        reliability=reliability,
        sampling_scope="full available text",
        limitations=("bounded source material",),
        captured_at=CAPTURED,
    )


def test_v2_envelope_round_trips_exact_text_offsets_and_hash() -> None:
    """Each part offset cuts back to the exact adapter-provided text."""
    description = "短公告原文"
    transcript = "平台字幕逐字稿"

    envelope = build_body_envelope(
        [
            _part("obs-description", description),
            _part("obs-subtitle", transcript, reliability="confirmed", method="platform_subtitle"),
        ],
        max_chars=32_000,
        source_revision="a" * 64,
        source_revision_kind="verified_content_hash",
    )

    assert envelope is not None
    assert envelope.text == description + MATERIAL_PART_SEPARATOR + transcript
    assert envelope.material_hash == hashlib.sha256(envelope.text.encode("utf-8")).hexdigest()
    assert [envelope.text[item.start_char : item.end_char] for item in envelope.parts] == [
        description,
        transcript,
    ]
    parsed = parse_stored_body(envelope.to_json())
    assert parsed == envelope


def test_v2_envelope_truncation_preserves_last_part_boundary_and_limitation() -> None:
    """A tools cap may cut content, but the retained offsets stay reversible."""
    envelope = build_body_envelope(
        [_part("obs-a", "a" * 6), _part("obs-b", "b" * 8, reliability="automatic")],
        max_chars=12,
    )

    assert envelope is not None
    assert envelope.text == "a" * 6 + MATERIAL_PART_SEPARATOR + "b" * 4
    assert envelope.truncated is True
    assert "body_truncated" in envelope.quality_flags
    assert "mixed_reliability" in envelope.quality_flags
    assert envelope.text[envelope.parts[-1].start_char : envelope.parts[-1].end_char] == "b" * 4
    assert "body_truncated" in envelope.parts[-1].limitations


def test_parser_rejects_hash_or_offset_tampering() -> None:
    """A v2 body cannot silently relabel different text or uncovered offsets."""
    envelope = build_body_envelope([_part("obs-a", "exact")], max_chars=100)
    assert envelope is not None
    payload = json.loads(envelope.to_json())
    payload["material_hash"] = "0" * 64
    with pytest.raises(ValueError, match="material_hash"):
        parse_stored_body(json.dumps(payload))

    payload = json.loads(envelope.to_json())
    payload["parts"][0]["start_char"] = 1
    with pytest.raises(ValueError, match="start at zero"):
        parse_stored_body(json.dumps(payload))


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("separator", "separator"),
        ("overlap", "overlap"),
        ("tail", "cover the exact text"),
        ("reliability", "reliability"),
        ("confidence", "confidence"),
        ("revision_pair", "revision and kind"),
        ("truncated_type", "truncated must be a boolean"),
        ("schema", "schema_version is unsupported"),
    ],
)
def test_parser_rejects_non_exact_or_untyped_v2_payloads(tamper: str, message: str) -> None:
    """Stored v2 parsing fails closed on gaps, bad types, and unknown contracts."""
    envelope = build_body_envelope(
        [_part("obs-a", "left"), _part("obs-b", "right", reliability="confirmed")],
        max_chars=100,
    )
    assert envelope is not None
    payload = json.loads(envelope.to_json())
    if tamper == "separator":
        payload["text"] = "left--right"
        payload["material_hash"] = hashlib.sha256(payload["text"].encode("utf-8")).hexdigest()
    elif tamper == "overlap":
        payload["parts"][1]["start_char"] = payload["parts"][0]["end_char"] - 1
    elif tamper == "tail":
        payload["parts"][1]["end_char"] -= 1
    elif tamper == "reliability":
        payload["parts"][1]["reliability"] = "trusted"
    elif tamper == "confidence":
        payload["parts"][1]["confidence"] = 1.5
    elif tamper == "revision_pair":
        payload["source_revision"] = "revision-without-kind"
    elif tamper == "truncated_type":
        payload["truncated"] = "false"
    else:
        payload["schema_version"] = 3

    with pytest.raises(ValueError, match=message):
        parse_stored_body(json.dumps(payload))


def test_legacy_body_is_readable_without_claiming_provenance() -> None:
    """Pre-v2 payloads remain readable and are explicitly typed as legacy."""
    parsed = parse_stored_body('{"text":"legacy short body","truncated":false}')

    assert parsed == LegacyBody(text="legacy short body", truncated=False)
    assert not isinstance(parsed, BodyEnvelope)
