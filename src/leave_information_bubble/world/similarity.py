"""Text-similarity primitives for inquiry deduplication at commit time."""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"\W+", re.UNICODE)


def normalize(text: str) -> str:
    """Casefold and strip punctuation/whitespace for stable comparison."""
    return _NON_WORD.sub("", text.casefold())


def bigrams(text: str) -> set[str]:
    """Return the set of consecutive two-character windows of normalized text.

    Texts shorter than two characters fall back to their single characters so
    short Chinese queries still compare meaningfully.
    """
    normalized = normalize(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Return the Jaccard index; both-empty sets compare as identical."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)
