"""Domain-neutral models for question, provenance, and evidence depth."""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AccessDepth(IntEnum):
    """Maximum conclusion strength supported by acquired source material."""

    METADATA = 0
    CONTENT_TEXT = 1
    VISUAL_CONTENT = 2
    REACTIONS = 3
    AUTHORITATIVE = 4
    CORROBORATED = 5


class ObservationModality(StrEnum):
    """How an observation was perceived."""

    METADATA = "metadata"
    TRANSCRIPT = "transcript"
    DOCUMENT_TEXT = "document_text"
    IMAGE = "image"
    OCR = "ocr"
    COMMENT = "comment"
    DANMAKU = "danmaku"
    STRUCTURED_DATA = "structured_data"
    AUDIO_FINGERPRINT = "audio_fingerprint"


class SourceObservation(BaseModel):
    """A bounded observation tied to one source location and acquisition method."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    modality: ObservationModality
    access_depth: AccessDepth
    excerpt: str = Field(default="", max_length=4000)
    body: str | None = Field(
        default=None,
        max_length=32_000,
        description="Full-depth content aggregated on explicit request, beyond the bounded excerpt",
    )
    location: str = Field(default="", max_length=300)
    acquisition_method: str = Field(min_length=1, max_length=100)
    captured_at: datetime
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    authority_scopes: list[str] = Field(default_factory=list)
    sampling_scope: str = Field(default="", max_length=500)
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["AccessDepth", "ObservationModality", "SourceObservation"]
