"""Provider-neutral normalization, quality reporting, and card selection."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import (
    IndependenceStatus,
    RetrievalContractReport,
    RetrievalHit,
    TimeFilterPrecision,
)

_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "spm",
    "from",
    "source",
}
_GENERIC_QUERY_TERMS = {
    "latest",
    "news",
    "official",
    "current",
    "最近",
    "近期",
    "最新",
    "新闻",
    "消息",
    "官方",
    "发生",
    "什么",
}
_KANA_PATTERN = re.compile(r"[\u3040-\u30ff]")
_HAN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{1,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
_SPACE_PATTERN = re.compile(r"\s+")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)


def canonicalize_public_url(url: str) -> str:
    """Return a conservative canonical URL without tracking or fragments."""
    try:
        parsed = urlsplit(url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        return ""
    scheme = parsed.scheme.lower()
    hostname = hostname.lower().rstrip(".")
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    authority = hostname if port is None or default_port else f"{hostname}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_KEYS and not key.lower().startswith("utm_")
    ]
    return urlunsplit((scheme, authority, path, urlencode(sorted(query_items)), ""))


def detect_public_text_language(text: str) -> str:
    """Classify common public-index text without a paid language dependency."""
    if _KANA_PATTERN.search(text):
        return "ja"
    han_count = len(_HAN_PATTERN.findall(text))
    latin_count = len(_LATIN_PATTERN.findall(text))
    if han_count >= 2 and han_count >= latin_count // 2:
        return "zh-Hans"
    if latin_count:
        return "en"
    return "und"


def cluster_origin_candidates(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    """Cluster only exact or strongly matching origin candidates.

    Unique, unverified pages remain ``unknown`` and therefore cannot satisfy a
    strict independent-source requirement.
    """
    canonical_counts = Counter(hit.canonical_url for hit in hits)
    verified_counts = Counter(
        hit.content_hash.lower() for hit in hits if hit.content_hash_verified
    )
    fingerprint_counts = Counter(_text_fingerprint(hit) for hit in hits if _text_fingerprint(hit))
    clustered: list[RetrievalHit] = []
    for hit in hits:
        basis = "unverified_occurrence"
        value = hit.canonical_url
        shared = canonical_counts[hit.canonical_url] > 1
        if hit.content_hash_verified and verified_counts[hit.content_hash.lower()] > 1:
            basis = "verified_content_hash"
            value = hit.content_hash.lower()
            shared = True
        else:
            fingerprint = _text_fingerprint(hit)
            if fingerprint and fingerprint_counts[fingerprint] > 1:
                basis = "exact_title_snippet_fingerprint"
                value = fingerprint
                shared = True
            elif shared:
                basis = "canonical_url"
        cluster_id = f"origin-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"
        metadata = {
            **hit.metadata,
            "origin_cluster_basis": basis,
        }
        clustered.append(
            hit.model_copy(
                update={
                    "origin_cluster_id": cluster_id,
                    "independence_status": (
                        IndependenceStatus.SHARED_ORIGIN
                        if shared
                        else IndependenceStatus.UNKNOWN
                    ),
                    "metadata": metadata,
                }
            )
        )
    return clustered


def build_retrieval_contract_report(
    hits: Sequence[RetrievalHit],
    *,
    query: str,
    target_language: str,
    target_region: str,
    timezone: str,
    window_start: datetime | None,
    window_end: datetime | None,
    time_filter_requested: bool,
    time_filter_applied: bool,
    time_filter_precision: TimeFilterPrecision,
    provider_attempts: Sequence[str] = (),
    empty_or_challenge_attempts: int = 0,
    cursor_supported: bool = False,
    cursor_trustworthy: bool = False,
    count_trustworthy: bool = True,
    blind_spots: Sequence[str] = (),
) -> RetrievalContractReport:
    """Measure a retrieval batch against explicit language and coverage rules."""
    count = len(hits)
    anchors = _query_anchors(query)
    language_matches = sum(
        _language_matches(hit.detected_language, target_language) for hit in hits
    )
    anchor_matches = sum(
        not anchors
        or any(
            anchor in f"{hit.title} {hit.snippet} {hit.url}".casefold()
            for anchor in anchors
        )
        for hit in hits
    )
    dated = [hit for hit in hits if hit.published_at is not None]
    in_window = (
        dated
        if window_start is None or window_end is None
        else [
            hit
            for hit in dated
            if hit.published_at is not None and window_start <= hit.published_at <= window_end
        ]
    )
    canonical_counts = Counter(hit.canonical_url for hit in hits)
    exact_duplicates = sum(value - 1 for value in canonical_counts.values() if value > 1)
    fingerprint_counts = Counter(_text_fingerprint(hit) for hit in hits if _text_fingerprint(hit))
    near_duplicates = sum(value - 1 for value in fingerprint_counts.values() if value > 1)
    hosts = Counter(urlsplit(hit.canonical_url).hostname or "" for hit in hits)
    dominant_host = max(hosts.values(), default=0)
    attempt_count = max(1, len(provider_attempts))
    language_ratio = language_matches / count if count else 0.0
    anchor_ratio = anchor_matches / count if count else 0.0
    duplicate_ratio = exact_duplicates / count if count else 0.0
    near_duplicate_ratio = near_duplicates / count if count else 0.0
    host_ratio = dominant_host / count if count else 0.0
    empty_ratio = min(1.0, empty_or_challenge_attempts / attempt_count)
    degraded: list[str] = []
    if count and language_ratio < 0.8:
        degraded.append("target_language_below_80_percent")
    if count and anchors and anchor_ratio < 0.6:
        degraded.append("entity_anchor_hit_below_60_percent")
    if empty_ratio > 0.5:
        degraded.append("empty_or_challenge_above_50_percent")
    if duplicate_ratio > 0.7:
        degraded.append("exact_duplicates_above_70_percent")
    if host_ratio > 0.8:
        degraded.append("single_host_above_80_percent")
    if time_filter_requested and not time_filter_applied:
        degraded.append("provider_time_filter_not_applied")
    return RetrievalContractReport(
        target_language=target_language,
        target_region=target_region,
        timezone=timezone,
        result_count=count,
        target_language_ratio=language_ratio,
        entity_anchor_hit_ratio=anchor_ratio,
        dated_result_count=len(dated),
        in_window_result_count=len(in_window),
        time_filter_requested=time_filter_requested,
        time_filter_applied=time_filter_applied,
        time_filter_precision=time_filter_precision,
        exact_duplicate_ratio=duplicate_ratio,
        near_duplicate_ratio=near_duplicate_ratio,
        dominant_host_ratio=host_ratio,
        empty_or_challenge_ratio=empty_ratio,
        cursor_supported=cursor_supported,
        cursor_trustworthy=cursor_trustworthy,
        count_trustworthy=count_trustworthy,
        provider_attempts=list(provider_attempts),
        degraded_reasons=degraded,
        blind_spots=list(dict.fromkeys(blind_spots)),
    )


class StratifiedCardReservoir:
    """Select a bounded, source-diverse card slice without a heat score."""

    def select(
        self,
        hits: Sequence[RetrievalHit],
        *,
        limit: int,
        target_language: str = "zh-Hans",
        target_language_ratio: float = 0.85,
    ) -> list[RetrievalHit]:
        """Deduplicate and round-robin by language, capability, and publisher."""
        if limit <= 0:
            return []
        if not 0.0 <= target_language_ratio <= 1.0:
            raise ValueError("target_language_ratio must be between zero and one")
        merged = self._merge_appearances(hits)
        target = [
            hit
            for hit in merged
            if _language_matches(hit.detected_language, target_language)
        ]
        international = [hit for hit in merged if hit not in target]
        target_limit = min(len(target), round(limit * target_language_ratio))
        international_limit = min(len(international), limit - target_limit)
        if target_limit + international_limit < min(limit, len(merged)):
            target_limit = min(
                len(target),
                target_limit + min(limit, len(merged)) - target_limit - international_limit,
            )
        selected = [
            *self._round_robin(target, target_limit),
            *self._round_robin(international, international_limit),
        ]
        if len(selected) < min(limit, len(merged)):
            selected_ids = {hit.id for hit in selected}
            remainder = [hit for hit in merged if hit.id not in selected_ids]
            selected.extend(self._round_robin(remainder, limit - len(selected)))
        return selected[:limit]

    @staticmethod
    def _merge_appearances(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
        order: list[str] = []
        for hit in hits:
            key = hit.origin_cluster_id or hit.canonical_url
            if key not in grouped:
                order.append(key)
            grouped[key].append(hit)
        merged: list[RetrievalHit] = []
        for key in order:
            appearances = grouped[key]
            first = min(appearances, key=lambda item: item.rank)
            queries = list(dict.fromkeys(item.query for item in appearances))
            providers = list(dict.fromkeys(item.provider for item in appearances))
            merged.append(
                first.model_copy(
                    update={
                        "metadata": {
                            **first.metadata,
                            "appearance_count": len(appearances),
                            "cross_query_count": len(queries),
                            "queries": queries,
                            "providers": providers,
                        }
                    }
                )
            )
        return merged

    @staticmethod
    def _round_robin(hits: Iterable[RetrievalHit], limit: int) -> list[RetrievalHit]:
        buckets: dict[tuple[str, str], deque[RetrievalHit]] = defaultdict(deque)
        for hit in hits:
            host = (urlsplit(hit.canonical_url).hostname or "").lower()
            buckets[(hit.capability_role.value, host)].append(hit)
        selected: list[RetrievalHit] = []
        keys = deque(buckets)
        while keys and len(selected) < limit:
            key = keys.popleft()
            bucket = buckets[key]
            selected.append(bucket.popleft())
            if bucket:
                keys.append(key)
        return selected


def _language_matches(detected: str, target: str) -> bool:
    detected_prefix = detected.casefold().split("-", 1)[0]
    target_prefix = target.casefold().split("-", 1)[0]
    return bool(detected_prefix and detected_prefix == target_prefix)


def _query_anchors(query: str) -> list[str]:
    normalized = re.sub(r"\b(?:site|filetype|inurl|intitle):\S+", " ", query, flags=re.I)
    return list(
        dict.fromkeys(
            token.casefold()
            for token in _TOKEN_PATTERN.findall(normalized)
            if token.casefold() not in _GENERIC_QUERY_TERMS
        )
    )


def _text_fingerprint(hit: RetrievalHit) -> str:
    text = f"{hit.title}\n{hit.snippet}".casefold()
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    normalized = _SPACE_PATTERN.sub(" ", text).strip()
    if len(normalized) < 20:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = [
    "StratifiedCardReservoir",
    "build_retrieval_contract_report",
    "canonicalize_public_url",
    "cluster_origin_candidates",
    "detect_public_text_language",
]
