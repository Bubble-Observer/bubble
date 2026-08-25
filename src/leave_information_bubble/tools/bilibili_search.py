"""Bilibili platform search tool.

Wraps the public B站 search API (no cookie required for basic search — confirmed
by spikes/bilibili-no-cookie-test). Returns normalized content items suitable
for candidate signal formation.

API endpoints used:
- Search: https://api.bilibili.com/x/web-interface/wbi/search/all/v2
- Video info: https://api.bilibili.com/x/web-interface/view
- Concurrent audience: https://api.bilibili.com/x/player/online/total

Error mapping:
- HTTP 412 / 403 → SOURCE_UNAVAILABLE (blocked or geo-restricted)
- Network error / timeout → TOOL_TRANSIENT (retryable)
- API returns code != 0 → SOURCE_UNAVAILABLE (gentle degradation)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import httpx

from leave_information_bubble.runtime.errors import AgentError, ErrorCode
from leave_information_bubble.tools.subtitle_ledger import DEFAULT_PATH, SubtitleLedger

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.bilibili.com/x/web-interface/wbi/search/all/v2"
VIDEO_INFO_URL = "https://api.bilibili.com/x/web-interface/view"
VIDEO_TAGS_URL = "https://api.bilibili.com/x/tag/archive/tags"
ONLINE_TOTAL_URL = "https://api.bilibili.com/x/player/online/total"
PLAYER_INFO_URL = "https://api.bilibili.com/x/player/wbi/v2"
PLAY_URL = "https://api.bilibili.com/x/player/wbi/playurl"
DANMAKU_URL_TEMPLATE = "https://comment.bilibili.com/{cid}.xml"
RELATED_URL = "https://api.bilibili.com/x/web-interface/archive/related"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}

# ---------------------------------------------------------------------------
# Subtitle reliability parameters (plan binding constraint: thresholds must
# be named constants, no magic numbers).
# ---------------------------------------------------------------------------

#: Maximum subtitle fetch attempts per ``get_subtitles`` call.
MAX_SUBTITLE_ATTEMPTS = 5

#: Seconds between subtitle fetch attempts within one ``get_subtitles`` call.
SUBTITLE_RETRY_INTERVAL_S = 2.5

#: Minimum title Han-bigram hit rate for a subtitle to count as related to
#: the video title (calibrated on 30 manually labeled samples, strict
#: standard: precision 1.0 / recall 0.80 at 0.20 vs 0.67 at 0.15).
TITLE_BIGRAM_THRESHOLD = 0.20

#: Minimum number of valid Han bigrams a title must contribute for the
#: relevance signal to be usable. Below this the content is treated as not
#: related (letter-only titles like "BLG vs TES" must never flag content).
MIN_TITLE_HAN_BIGRAMS = 3

#: Times the same subtitle hash must recur within one ``get_subtitles`` call
#: to count as "same-video repetition".
SUBTITLE_REPEAT_THRESHOLD = 2

#: Length of the subtitle content-hash prefix (whitespace-stripped md5 hex).
SUBTITLE_HASH_PREFIX_LENGTH = 10

#: Limitation code when no attempt yielded a trustworthy subtitle. Derived
#: from ``MAX_SUBTITLE_ATTEMPTS`` so the code never drifts from the attempts.
SUBTITLE_UNRELIABLE_LIMITATION = f"platform_subtitle_unreliable_after_{MAX_SUBTITLE_ATTEMPTS}_attempts"

#: Limitation code when the video exposes no subtitle track at all (the
#: channel adapter emits the same code today for an empty subtitle result).
SUBTITLE_ABSENT_LIMITATION = "platform_subtitle_absent_or_unavailable_without_authentication"

#: Limitation code when the player endpoint offers a track that does not
#: belong to the canonical subtitle list returned for this video.
SUBTITLE_IDENTITY_MISMATCH_LIMITATION = "platform_subtitle_track_identity_mismatch"


def _segments_text(segments: list[dict[str, Any]]) -> str:
    """Concatenate subtitle segment contents into one text string.

    Used for content hashing and title-bigram relatedness checks.
    """
    return "".join(str(segment.get("content", "") or "") for segment in segments)


def _subtitle_content_hash(segments: list[dict[str, Any]]) -> str:
    """Return the whitespace-stripped md5 hex prefix of subtitle content.

    The hash identifies a subtitle track's content: the fallback pool serves
    the same content (hence the same hash) to many unrelated videos. Empty
    content yields "" — callers treat it as unhashable and skip recording.
    """
    text = re.sub(r"\s+", "", _segments_text(segments))
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:SUBTITLE_HASH_PREFIX_LENGTH]


def _is_han_bigram(bigram: str) -> bool:
    """Return True when both characters of ``bigram`` are Han (U+4E00–U+9FFF).

    Bigrams containing ASCII letters, digits, or punctuation are excluded
    from the title-relevance count, so letter-heavy titles (e.g. team names
    like BLG/TES/LPL) cannot flag arbitrary Latin content as related.
    """
    return all("一" <= char <= "鿿" for char in bigram)


def _title_bigram_hit_rate(title: str, content: str) -> float:
    """Return the fraction of the title's Han bigrams found in ``content``.

    Only bigrams whose two characters are both Han (U+4E00–U+9FFF) count on
    either side; whitespace is stripped first. When the title yields fewer
    than ``MIN_TITLE_HAN_BIGRAMS`` valid Han bigrams the relevance signal is
    unusable and the rate is 0.0 — the content is treated as not related.
    A rate of at least ``TITLE_BIGRAM_THRESHOLD`` marks the subtitle as
    plausibly related to the video title.
    """
    title_clean = re.sub(r"\s+", "", title)
    content_clean = re.sub(r"\s+", "", content)
    title_bigrams = {
        title_clean[i : i + 2]
        for i in range(len(title_clean) - 1)
        if _is_han_bigram(title_clean[i : i + 2])
    }
    if len(title_bigrams) < MIN_TITLE_HAN_BIGRAMS:
        return 0.0
    content_bigrams = {
        content_clean[i : i + 2]
        for i in range(len(content_clean) - 1)
        if _is_han_bigram(content_clean[i : i + 2])
    }
    return len(title_bigrams & content_bigrams) / len(title_bigrams)


def _annotate_segments(
    segments: list[dict[str, Any]],
    *,
    reliability: Literal["confirmed", "best_effort"],
    attempts: int,
) -> list[dict[str, Any]]:
    """Attach reliability metadata (``reliability`` / ``attempts``) to segments.

    ``attempts`` is the number of fetch attempts made when the decision was
    taken, so callers can weigh how hard the platform made us try.
    """
    return [
        {**segment, "reliability": reliability, "attempts": attempts}
        for segment in segments
    ]


@dataclass(frozen=True)
class BilibiliSearchResult:
    """Normalized result from a B站 search query."""

    items: list[dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    error: str = ""
    query: str = ""


@dataclass(frozen=True)
class BilibiliCommentPage:
    """One bounded comment page with explicit sampling provenance."""

    comments: list[dict[str, Any]] = field(default_factory=list)
    total_comments: int | None = None
    sort: str = "hot"
    requested_limit: int = 20
    returned_count: int = 0
    reply_limit: int = 10
    page: int = 1
    has_more: bool | None = None
    limitations: list[str] = field(default_factory=list)
    error: str = ""


class BilibiliAudiencePrecision(StrEnum):
    """Precision state of one platform concurrent-audience display."""

    EXACT = "exact"
    ROUNDED = "rounded"
    HIDDEN = "hidden"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BilibiliAudienceValue:
    """One exact, lower-bounded, hidden, or unavailable audience value."""

    precision: BilibiliAudiencePrecision
    value: int | None = None
    lower_bound: int | None = None
    display: str = ""
    show_switch: bool | None = None


@dataclass(frozen=True)
class BilibiliConcurrentAudienceResult:
    """Current archive-wide and selected-part audience from a public endpoint."""

    bvid: str
    cid: int
    total: BilibiliAudienceValue
    count: BilibiliAudienceValue
    limitations: list[str] = field(default_factory=list)
    error_kind: str = ""
    error_code: int | None = None
    error: str = ""


class BilibiliSearchTool:
    """Search Bilibili for content matching a keyword.

    Uses read-only platform endpoints for discovery, metadata, subtitles,
    comments and bounded danmaku samples. Missing capabilities degrade to
    explicit empty results; callers must not treat metadata as content.

    Args:
        timeout: HTTP request timeout in seconds.
        min_interval: Minimum seconds between requests (rate limiting).
        sessdata: Optional cookie for authenticated endpoints (not used yet).
        subtitle_retry_interval: Seconds between subtitle fetch attempts
            inside one ``get_subtitles`` call (defaults to
            ``SUBTITLE_RETRY_INTERVAL_S``; tests pass 0).
        subtitle_ledger_path: Ledger file for subtitle-hash blacklist tracking
            (defaults to ``DEFAULT_PATH``); its parent directory is created on
            construction so recording never silently fails.

    """

    def __init__(
        self,
        timeout: float = 10.0,
        min_interval: float = 1.0,
        sessdata: str = "",
        *,
        subtitle_retry_interval: float = SUBTITLE_RETRY_INTERVAL_S,
        subtitle_ledger_path: Path = DEFAULT_PATH,
    ) -> None:
        self._timeout = timeout
        self._min_interval = min_interval
        self._sessdata = sessdata
        self._subtitle_retry_interval = subtitle_retry_interval
        self._subtitle_ledger_path = Path(subtitle_ledger_path)
        # Task 1 review M-2: the ledger silently drops records when its parent
        # directory is missing — guarantee it here, at construction time.
        self._subtitle_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._subtitle_ledger = SubtitleLedger(self._subtitle_ledger_path)
        #: Reason the last ``get_subtitles`` call returned no trustworthy
        #: content (``None`` after a successful call; limitation codes
        #: otherwise). Task 3's channel adapter surfaces this as an
        #: ObservationBatch limitation.
        self.last_subtitle_limitation: str | None = None
        self._call_count: int = 0
        self._call_history: list[dict[str, Any]] = []
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def _respect_rate_limit(self) -> None:
        """Serialize this tool instance's requests and honor ``min_interval``."""
        async with self._rate_lock:
            remaining = self._min_interval - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_request_at = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 20,
        *,
        order: str = "totalrank",
    ) -> BilibiliSearchResult:
        """Search Bilibili by keyword.

        Args:
            keyword: Search query (Chinese recommended for B站).
            page: Page number (1-based).
            page_size: Results per page (max 50).
            order: Provider-native ordering: relevance, view attention, or publication time.

        Returns:
            BilibiliSearchResult with normalized items array.

        Raises:
            AgentError(TOOL_TRANSIENT): On network/timeout errors.
            AgentError(SOURCE_UNAVAILABLE): On API errors or blocks.

        """
        self._call_count += 1
        supported_orders = {"totalrank", "click", "pubdate"}
        if order not in supported_orders:
            raise ValueError(f"unsupported Bilibili search order: {order}")
        self._call_history.append(
            {
                "keyword": keyword,
                "page": page,
                "order": order,
                "call_index": self._call_count,
            }
        )

        params: dict[str, Any] = {
            "keyword": keyword,
            "page": page,
            "page_size": max(1, min(page_size, 50)),
            "order": order,
        }

        cookies = {}
        if self._sessdata:
            cookies["SESSDATA"] = self._sessdata

        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(
                    SEARCH_URL,
                    params=params,
                    headers=HEADERS,
                    cookies=cookies,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            raise AgentError(ErrorCode.TOOL_TRANSIENT, f"Bilibili search timeout: {e}") from e
        except httpx.NetworkError as e:
            raise AgentError(ErrorCode.TOOL_TRANSIENT, f"Bilibili network error: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 412):
                raise AgentError(
                    ErrorCode.SOURCE_UNAVAILABLE,
                    f"Bilibili blocked request (HTTP {e.response.status_code})",
                ) from e
            raise AgentError(
                ErrorCode.TOOL_TRANSIENT,
                f"Bilibili HTTP {e.response.status_code}",
            ) from e

        code = data.get("code", -1)
        if code != 0:
            return BilibiliSearchResult(
                error=data.get("message", f"API error code {code}"),
                query=keyword,
            )

        items = self._normalize_results(data)
        return BilibiliSearchResult(
            items=items,
            total_results=len(items),
            query=keyword,
        )

    async def get_video_info(self, bvid: str) -> dict[str, Any]:
        """Fetch metadata for a specific BV号.

        Args:
            bvid: Bilibili video ID (e.g., "BV1xx411c7mD").

        Returns:
            Dict with title, author, stats, pubdate, etc.
            Empty dict if the video is unavailable.

        """
        normalized = bvid.removeprefix("bilibili-")
        cookies = {"SESSDATA": self._sessdata} if self._sessdata else {}
        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    VIDEO_INFO_URL,
                    params={"bvid": normalized},
                    headers=HEADERS,
                    cookies=cookies,
                )
                response.raise_for_status()
                data = response.json()

            if data.get("code") != 0:
                logger.warning("Video info API error for %s: %s", normalized, data.get("message"))
                return {}

            video_data = data.get("data", {})
            return {
                "aid": int(video_data.get("aid", 0) or 0),
                "bvid": normalized,
                "cid": int(video_data.get("cid", 0) or 0),
                "title": video_data.get("title", ""),
                "author": video_data.get("owner", {}).get("name", ""),
                "owner_mid": int(video_data.get("owner", {}).get("mid", 0) or 0),
                "description": video_data.get("desc", ""),
                "subtitle_tracks": self._normalize_subtitle_tracks(
                    video_data.get("subtitle", {}).get("list", [])
                    if isinstance(video_data.get("subtitle"), dict)
                    else []
                ),
                "pages": [
                    {
                        "cid": int(page.get("cid", 0) or 0),
                        "page": int(page.get("page", 0) or 0),
                        "part": str(page.get("part", "") or ""),
                        "duration": int(page.get("duration", 0) or 0),
                    }
                    for page in video_data.get("pages", [])
                    if isinstance(page, dict)
                ],
                "published_at": self._timestamp_iso(video_data.get("pubdate")),
                "stats": {
                    "play": self._optional_int(video_data.get("stat", {}).get("view")),
                    "danmaku": self._optional_int(
                        video_data.get("stat", {}).get("danmaku")
                    ),
                    "like": self._optional_int(video_data.get("stat", {}).get("like")),
                    "coin": self._optional_int(video_data.get("stat", {}).get("coin")),
                    "favorite": self._optional_int(
                        video_data.get("stat", {}).get("favorite")
                    ),
                    "reply": self._optional_int(video_data.get("stat", {}).get("reply")),
                },
            }
        except Exception as e:
            logger.warning("Failed to fetch video info for %s: %s", normalized, e)
            return {}

    async def get_concurrent_audience(
        self,
        bvid: str,
        cid: int,
    ) -> BilibiliConcurrentAudienceResult:
        """Read one public concurrent-audience snapshot for a selected video part.

        Bilibili currently exposes ``total`` across the whole archive and
        ``count`` for the selected ``cid``. Values are presentation strings:
        an integer is exact, while forms such as ``6000+`` are lower bounds.
        This method preserves visibility and precision without interpreting
        the snapshot as heat or growth.
        """
        normalized = bvid.removeprefix("bilibili-")
        if not normalized or cid <= 0:
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind="invalid_input",
                error="bvid and positive cid are required",
            )
        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    ONLINE_TOTAL_URL,
                    params={"bvid": normalized, "cid": cid},
                    headers=HEADERS,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status == 429:
                kind = "rate_limited"
            elif status in {403, 412}:
                kind = "blocked"
            else:
                kind = "unavailable"
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind=kind,
                error_code=status,
                error=f"HTTP {status}",
            )
        except httpx.TimeoutException:
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind="timeout",
                error="request timed out",
            )
        except httpx.HTTPError:
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind="network",
                error="network request failed",
            )
        except (TypeError, ValueError):
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind="schema_changed",
                error="response was not valid JSON",
            )

        if not isinstance(payload, dict):
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind="schema_changed",
                error="response root is not an object",
            )
        code = self._optional_int(payload.get("code"))
        if code != 0:
            if code == -509:
                kind = "rate_limited"
            elif code in {-412, -403}:
                kind = "blocked"
            else:
                kind = "unavailable"
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind=kind,
                error_code=code,
                error=str(payload.get("message", "platform API error")),
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind="schema_changed",
                error="response data is not an object",
            )
        raw_switch = data.get("show_switch")
        limitations: list[str] = []
        if raw_switch is None:
            show_switch: dict[str, Any] = {}
            limitations.append("show_switch_missing")
        elif isinstance(raw_switch, dict):
            show_switch = raw_switch
        else:
            return self._concurrent_audience_failure(
                normalized,
                cid,
                kind="schema_changed",
                error="show_switch is not an object",
            )
        total_switch = self._optional_bool(show_switch.get("total"))
        count_switch = self._optional_bool(show_switch.get("count"))
        total, total_limitations = self._parse_audience_value(
            data.get("total"),
            total_switch,
        )
        count, count_limitations = self._parse_audience_value(
            data.get("count"),
            count_switch,
        )
        limitations.extend(f"total_{item}" for item in total_limitations)
        limitations.extend(f"count_{item}" for item in count_limitations)
        return BilibiliConcurrentAudienceResult(
            bvid=normalized,
            cid=cid,
            total=total,
            count=count,
            limitations=limitations,
        )

    async def get_video_tags(self, bvid: str) -> list[dict[str, Any]]:
        """Fetch public archive tags for a BV video.

        An empty result is ambiguous: the video can have no public tags, or
        the public endpoint can be unavailable. Callers must disclose that
        ambiguity instead of treating it as a verified empty tag set.
        """
        normalized = bvid.removeprefix("bilibili-")
        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    VIDEO_TAGS_URL,
                    params={"bvid": normalized},
                    headers=HEADERS,
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as error:
            logger.warning("Failed to fetch tags for %s: %s", normalized, error)
            return []
        if payload.get("code") != 0:
            logger.warning("Tag API error for %s: %s", normalized, payload.get("message"))
            return []
        tags: list[dict[str, Any]] = []
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("tag_name", "") or "").strip()
            if not name:
                continue
            tags.append(
                {
                    "tag_id": self._optional_int(item.get("tag_id")),
                    "name": name,
                    "type": str(item.get("tag_type", "") or ""),
                    "music_id": str(item.get("music_id", "") or ""),
                }
            )
        return tags

    async def get_related_videos(
        self, bvid: str, *, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return Bilibili's related-video recommendations.

        Args:
            bvid: 视频 BV 号。
            limit: 最大返回数量。

        Returns:
            相关视频列表。每个包含 ``bvid``, ``title``, ``play``,
            ``danmaku``, ``reply``。API 不可用时返回空列表。

        """
        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    RELATED_URL,
                    params={"bvid": bvid},
                    headers=HEADERS,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.warning(
                "Related videos API unavailable for %s", bvid, exc_info=True
            )
            return []

        if not isinstance(data, dict):
            return []

        items = data.get("data", [])
        if not isinstance(items, list):
            return []

        results: list[dict[str, Any]] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            stat = item.get("stat", {}) if isinstance(item.get("stat"), dict) else {}
            results.append({
                "bvid": str(item.get("bvid", "")),
                "title": str(item.get("title", "")),
                "play": stat.get("view", 0) or 0,
                "danmaku": stat.get("danmaku", 0) or 0,
                "reply": stat.get("reply", 0) or 0,
            })

        return results

    async def get_subtitles(
        self,
        bvid: str,
        *,
        video_info: dict[str, Any] | None = None,
        domain: str = "",
    ) -> list[dict[str, Any]]:
        """Return reliable platform subtitle segments for the first video page.

        The platform serves videos without subtitles of their own from a
        shared fallback pool: the same AI-subtitle content (same hash) recurs
        across many unrelated videos, so a single fetch is not trustworthy.
        This method fetches up to ``MAX_SUBTITLE_ATTEMPTS`` times (spaced by
        ``SUBTITLE_RETRY_INTERVAL_S``) and judges each result:

        * **confirmed** — the same content hash recurred at least
          ``SUBTITLE_REPEAT_THRESHOLD`` times within this call *and* the
          title Han-bigram hit rate is at least ``TITLE_BIGRAM_THRESHOLD``;
        * **best_effort** — the content is title-related (Han-bigram hit rate
          at least ``TITLE_BIGRAM_THRESHOLD``) but has not yet recurred
          within this call; the candidate with the highest hit rate is
          returned (ties broken by track rank: manual zh-CN first, then other
          manual, then AI);
        * **discarded** — the hash is blacklisted in ``domain`` (observed for
          at least ``BLACKLIST_THRESHOLD`` distinct bvids), or the content
          recurs without title relevance: the platform can serve *stable*
          fallback-pool content (the same unrelated hash on every attempt),
          so repetition alone is no evidence of the video's own subtitles.
          Discarded content is never returned, not even as best_effort.

        Every successful fetch records ``(domain, hash, bvid)`` in the
        subtitle ledger; with ``domain == ""`` the ledger is skipped entirely
        (no recording, no blacklist check). Each returned segment carries
        ``reliability`` (``"confirmed"`` or ``"best_effort"``) and
        ``attempts`` (fetch attempts made at decision time). When no attempt
        yields usable content the method returns ``[]`` and records the reason
        in ``last_subtitle_limitation`` (``SUBTITLE_UNRELIABLE_LIMITATION``
        after exhausting attempts, ``SUBTITLE_ABSENT_LIMITATION`` when the
        video has no subtitle track at all). Fallback-pool content is never
        returned.

        Args:
            bvid: Bilibili video ID (with or without the "bilibili-" prefix).
            video_info: Optional cached video metadata (cid, title); fetched
                when omitted.
            domain: Domain name for blacklist scoping (e.g. "lol_cn"); empty
                disables ledger recording and blacklist checks.

        Returns:
            Reliable subtitle segments (each with ``reliability`` and
            ``attempts`` keys) or ``[]`` when no trustworthy content could be
            obtained.

        """
        # A fresh call must not observe a limitation left by an earlier
        # call on this instance (shared-instance reuse / concurrent reads).
        self.last_subtitle_limitation = None
        normalized = bvid.removeprefix("bilibili-")
        info = video_info if video_info is not None else await self.get_video_info(normalized)
        cid = int(info.get("cid", 0) or 0)
        if not cid:
            pages = info.get("pages", [])
            if isinstance(pages, list) and pages and isinstance(pages[0], dict):
                cid = int(pages[0].get("cid", 0) or 0)
        if not cid:
            self.last_subtitle_limitation = SUBTITLE_ABSENT_LIMITATION
            return []

        title = str(info.get("title", "") or "")
        cookies = {"SESSDATA": self._sessdata} if self._sessdata else {}
        canonical_identities: set[tuple[str, str]] | None = None
        if "subtitle_tracks" in info:
            raw_canonical_tracks = info.get("subtitle_tracks", [])
            if not isinstance(raw_canonical_tracks, list):
                raw_canonical_tracks = []
            canonical_identities = {
                identity
                for track in raw_canonical_tracks
                if isinstance(track, dict)
                and (identity := self._subtitle_track_identity(track)) is not None
            }
            if not canonical_identities:
                # Authenticated video detail is the ownership source of truth.
                # When it explicitly has no tracks, avoid five player retries
                # and let the full-only ASR fallback start immediately.
                self.last_subtitle_limitation = SUBTITLE_ABSENT_LIMITATION
                return []
        seen_hashes: dict[str, int] = {}
        # best_effort candidate: (segments, track rank, title-bigram hit rate).
        # Only title-related, not-yet-repeated content qualifies; the higher
        # hit rate wins, ties go to the better-ranked track, then to the
        # first-seen candidate.
        best_effort: tuple[list[dict[str, Any]], int, float] | None = None
        attempts_made = 0
        identity_mismatch_seen = False
        for attempt in range(MAX_SUBTITLE_ATTEMPTS):
            if attempt > 0:
                await asyncio.sleep(self._subtitle_retry_interval)
            segments, rank, fetch_limitation = await self._fetch_subtitle_segments(
                normalized,
                cid,
                cookies,
                canonical_identities=canonical_identities,
            )
            attempts_made += 1
            identity_mismatch_seen = identity_mismatch_seen or (
                fetch_limitation == SUBTITLE_IDENTITY_MISMATCH_LIMITATION
            )
            if not segments:
                continue
            content_hash = _subtitle_content_hash(segments)
            if domain and content_hash:
                self._subtitle_ledger.record(domain, content_hash, normalized)
                if self._subtitle_ledger.is_blacklisted(domain, content_hash):
                    logger.debug(
                        "Discarding blacklisted subtitle hash %s for %s in domain %s",
                        content_hash,
                        normalized,
                        domain,
                    )
                    continue
            seen_hashes[content_hash] = seen_hashes.get(content_hash, 0) + 1
            repeated = seen_hashes[content_hash] >= SUBTITLE_REPEAT_THRESHOLD
            hit_rate = _title_bigram_hit_rate(title, _segments_text(segments))
            relevant = hit_rate >= TITLE_BIGRAM_THRESHOLD
            if repeated and relevant:
                self.last_subtitle_limitation = None
                return _annotate_segments(
                    segments, reliability="confirmed", attempts=attempts_made
                )
            if repeated or not relevant:
                # A repeated-but-unrelated track is *stable* fallback-pool
                # content: the platform can serve the same unrelated hash on
                # every attempt, so repetition alone is no evidence of the
                # video's own subtitles. Discard it like a blacklisted hash
                # and keep retrying; a single unrelated occurrence is equally
                # useless as a best_effort candidate.
                continue
            if (
                best_effort is None
                or hit_rate > best_effort[2]
                or (hit_rate == best_effort[2] and rank < best_effort[1])
            ):
                best_effort = (segments, rank, hit_rate)
        if best_effort is not None:
            self.last_subtitle_limitation = None
            return _annotate_segments(
                best_effort[0], reliability="best_effort", attempts=attempts_made
            )
        self.last_subtitle_limitation = (
            SUBTITLE_IDENTITY_MISMATCH_LIMITATION
            if identity_mismatch_seen
            else SUBTITLE_UNRELIABLE_LIMITATION
        )
        return []

    async def _fetch_subtitle_segments(
        self,
        bvid: str,
        cid: int,
        cookies: dict[str, str],
        *,
        canonical_identities: set[tuple[str, str]] | None,
    ) -> tuple[list[dict[str, Any]], int, str | None]:
        """Fetch and normalize one subtitle track.

        Returns the normalized segments plus the selected track's rank
        (``_subtitle_track_rank``) and an optional typed limitation. An empty
        list with a nominal rank of 2 means no trustworthy track was fetched.
        """
        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(
                    PLAYER_INFO_URL,
                    params={"bvid": bvid, "cid": cid},
                    headers=HEADERS,
                    cookies=cookies,
                )
                response.raise_for_status()
                payload = response.json()
                tracks = payload.get("data", {}).get("subtitle", {}).get("subtitles") or []
                if not tracks:
                    return [], 2, None

                if canonical_identities is not None:
                    matched_tracks = [
                        track
                        for track in tracks
                        if isinstance(track, dict)
                        and self._subtitle_track_identity(track) in canonical_identities
                    ]
                    if not matched_tracks:
                        logger.warning(
                            "Rejected player subtitle tracks with foreign identities for %s",
                            bvid,
                        )
                        return [], 2, SUBTITLE_IDENTITY_MISMATCH_LIMITATION
                    tracks = matched_tracks

                track = min(
                    (item for item in tracks if isinstance(item, dict)),
                    key=BilibiliSearchTool._subtitle_track_rank,
                    default=None,
                )
                if track is None:
                    return [], 2, None
                rank = BilibiliSearchTool._subtitle_track_rank(track)
                subtitle_url = str(track.get("subtitle_url", "") or "")
                if subtitle_url.startswith("//"):
                    subtitle_url = f"https:{subtitle_url}"
                if not subtitle_url:
                    return [], 2, None
                subtitle_response = await client.get(subtitle_url, headers=HEADERS, cookies=cookies)
                subtitle_response.raise_for_status()
                subtitle_payload = subtitle_response.json()
        except (httpx.HTTPError, ValueError, TypeError) as error:
            logger.warning("Failed to fetch subtitles for %s: %s", bvid, error)
            return [], 2, None

        language = str(track.get("lan", "") or track.get("lan_doc", "") or "unknown")
        segments: list[dict[str, Any]] = []
        for item in subtitle_payload.get("body", []):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            segments.append(
                {
                    "start": float(item.get("from", 0.0) or 0.0),
                    "end": float(item.get("to", 0.0) or 0.0),
                    "content": content,
                    "language": language,
                    "acquisition_method": "platform_subtitle",
                    "confidence": 1.0,
                    "cid": cid,
                }
            )
        return segments, rank, None

    @staticmethod
    def _normalize_subtitle_tracks(raw_tracks: object) -> list[dict[str, str]]:
        """Preserve canonical track identity fields from the video detail API."""
        if not isinstance(raw_tracks, list):
            return []
        result: list[dict[str, str]] = []
        for track in raw_tracks:
            if not isinstance(track, dict):
                continue
            track_id = str(track.get("id", "") or track.get("id_str", "") or "").strip()
            language = str(track.get("lan", "") or "").strip()
            if not track_id or not language:
                continue
            result.append(
                {
                    "id": track_id,
                    "lan": language,
                    "lan_doc": str(track.get("lan_doc", "") or ""),
                }
            )
        return result

    @staticmethod
    def _subtitle_track_identity(track: dict[str, Any]) -> tuple[str, str] | None:
        """Return the stable ID/language identity shared by detail and player APIs."""
        track_id = str(track.get("id", "") or track.get("id_str", "") or "").strip()
        language = str(track.get("lan", "") or "").strip().casefold()
        if not track_id or not language:
            return None
        return track_id, language

    @staticmethod
    def _subtitle_track_rank(track: dict[str, Any]) -> int:
        """Rank one subtitle track: manual zh-CN first, AI tracks last.

        The platform mixes manual tracks (``lan`` like ``zh-CN``, ``lan_doc``
        like ``中文（简体）``) with AI-generated ones (``ai-zh`` /
        ``中文（自动生成的）``) in a varying order. Deterministic selection
        keeps repeated fetches on the same track.
        """
        lan = str(track.get("lan", "") or "").strip().casefold()
        lan_doc = str(track.get("lan_doc", "") or "").strip()
        is_ai = lan.startswith("ai") or "自动生成" in lan_doc or "ai" in lan_doc.casefold()
        is_zh_cn = lan in {"zh-cn", "zh"} or "中文（简体）" in lan_doc or "简体" in lan_doc
        if is_zh_cn and not is_ai:
            return 0
        if not is_ai:
            return 1
        return 2

    async def get_danmaku(
        self,
        bvid: str,
        limit: int = 500,
        *,
        cid: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a bounded, timestamped danmaku sample from the first page."""
        normalized = bvid.removeprefix("bilibili-")
        resolved_cid = int(cid or 0)
        if not resolved_cid:
            info = await self.get_video_info(normalized)
            resolved_cid = int(info.get("cid", 0) or 0)
            if not resolved_cid:
                pages = info.get("pages", [])
                if isinstance(pages, list) and pages and isinstance(pages[0], dict):
                    resolved_cid = int(pages[0].get("cid", 0) or 0)
        if not resolved_cid:
            return []
        safe_limit = max(1, min(limit, 5000))
        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    DANMAKU_URL_TEMPLATE.format(cid=resolved_cid),
                    headers=HEADERS,
                )
                response.raise_for_status()
                root = ET.fromstring(response.content)
        except (httpx.HTTPError, ET.ParseError) as error:
            logger.warning("Failed to fetch danmaku for %s: %s", normalized, error)
            return []

        result: list[dict[str, Any]] = []
        for element in root.findall("d"):
            text = (element.text or "").strip()
            parts = str(element.attrib.get("p", "")).split(",")
            if not text or not parts:
                continue
            try:
                video_time = float(parts[0])
            except ValueError:
                continue
            result.append(
                {
                    "video_time": video_time,
                    "content": text,
                    "mode": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                    "sent_at": int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0,
                    "sampling": "first_page_bounded",
                    "cid": resolved_cid,
                }
            )
            if len(result) >= safe_limit:
                break
        return result

    async def get_audio_bytes(
        self,
        bvid: str,
        max_bytes: int = 50_000_000,
        *,
        video_info: dict[str, Any] | None = None,
    ) -> bytes:
        """Fetch a bounded audio stream for transient ASR processing.

        The caller must keep the payload temporary and delete derived media
        after transcription. Empty bytes indicate unavailable or oversized
        audio.
        """
        normalized = bvid.removeprefix("bilibili-")
        safe_max_bytes = max(1_000_000, min(max_bytes, 200_000_000))
        info = video_info if video_info is not None else await self.get_video_info(normalized)
        cid = int(info.get("cid", 0) or 0)
        if not cid:
            return b""
        cookies = {"SESSDATA": self._sessdata} if self._sessdata else {}
        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(
                    PLAY_URL,
                    params={
                        "bvid": normalized,
                        "cid": cid,
                        "fnval": 16,
                        "fourk": 0,
                    },
                    headers=HEADERS,
                    cookies=cookies,
                )
                response.raise_for_status()
                payload = response.json()
                audio_streams = payload.get("data", {}).get("dash", {}).get("audio") or []
                if not audio_streams:
                    return b""
                usable_streams = [
                    item
                    for item in audio_streams
                    if isinstance(item, dict)
                    and str(item.get("baseUrl", "") or item.get("base_url", "") or "")
                ]
                stream = min(
                    usable_streams,
                    key=lambda item: int(item.get("bandwidth", 0) or 0) or 2**63,
                    default=None,
                )
                if not stream:
                    return b""
                audio_url = str(stream.get("baseUrl", "") or stream.get("base_url", "") or "")
                content = bytearray()
                async with client.stream(
                    "GET",
                    audio_url,
                    headers=HEADERS,
                    cookies=cookies,
                ) as audio_response:
                    audio_response.raise_for_status()
                    async for chunk in audio_response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > safe_max_bytes:
                            logger.warning(
                                "Audio for %s exceeded bounded ASR payload (%d > %d)",
                                normalized,
                                len(content),
                                safe_max_bytes,
                            )
                            return b""
        except (httpx.HTTPError, ValueError, TypeError) as error:
            logger.warning("Failed to fetch audio for %s: %s", normalized, error)
            return b""
        return bytes(content)

    # ------------------------------------------------------------------
    # Comment API
    # ------------------------------------------------------------------

    COMMENT_URL = "https://api.bilibili.com/x/v2/reply/main"

    async def get_comments(self, oid: int | str, limit: int = 20, mode: int = 3) -> list[dict[str, Any]]:
        """Fetch hot comments for a video by its aid (av号) or bvid.

        B站 comment API is public — no cookie required for basic access.
        Mode 3 = hot comments (热度排序). Mode 2 = newest.

        Args:
            oid: Video aid (numeric) or bvid (string).
            limit: Max comments to return.
            mode: 2=newest, 3=hot (default).

        Returns:
            List of comments, each with: content, like_count, user, ctime.

        """
        resolved_oid: int | str = oid
        if isinstance(oid, str) and oid.upper().startswith("BV"):
            info = await self.get_video_info(oid)
            resolved_oid = int(info.get("aid", 0) or 0)
            if not resolved_oid:
                logger.warning("Unable to resolve bvid %s to aid for comments", oid)
                return []

        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    self.COMMENT_URL,
                    params={
                        "oid": resolved_oid,
                        "type": 1,
                        "mode": mode,
                        "ps": min(limit, 50),
                    },
                    headers=HEADERS,
                )
                response.raise_for_status()
                data = response.json()

            if data.get("code") != 0:
                logger.warning("Comment API error: %s", data.get("message"))
                return []

            replies = data.get("data", {}).get("replies") or []
            comments: list[dict[str, Any]] = []
            for r in replies:
                child_replies = [
                    {
                        "user": child.get("member", {}).get("uname", ""),
                        "content": child.get("content", {}).get("message", ""),
                        "like_count": child.get("like", 0),
                        "ctime": child.get("ctime", 0),
                    }
                    for child in (r.get("replies") or [])[:10]
                    if isinstance(child, dict)
                ]
                comments.append(
                    {
                        "user": r.get("member", {}).get("uname", ""),
                        "content": r.get("content", {}).get("message", ""),
                        "like_count": r.get("like", 0),
                        "reply_count": r.get("rcount", 0),
                        "ctime": r.get("ctime", 0),
                        "floor": r.get("floor", 0),
                        "sampling": "hot" if mode == 3 else "newest",
                        "replies": child_replies,
                    }
                )

            return comments[:limit]
        except Exception as e:
            logger.warning("Failed to fetch comments: %s", e)
            return []

    async def get_comment_page(
        self,
        oid: int | str,
        *,
        limit: int = 20,
        mode: int = 3,
        page: int = 1,
        reply_limit: int = 10,
    ) -> BilibiliCommentPage:
        """Fetch a bounded comment page and preserve sampling metadata.

        ``mode=3`` requests the platform hot ordering and ``mode=2`` newest.
        The public endpoint does not expose unbiased random sampling, so the
        ordering and first-page boundary are always disclosed.
        """
        safe_limit = max(1, min(limit, 50))
        safe_page = max(1, page)
        safe_reply_limit = max(0, min(reply_limit, 10))
        sort = "hot" if mode == 3 else "newest" if mode == 2 else f"mode_{mode}"
        resolved_oid: int | str = oid
        if isinstance(oid, str) and oid.upper().startswith("BV"):
            info = await self.get_video_info(oid)
            resolved_oid = int(info.get("aid", 0) or 0)
            if not resolved_oid:
                return BilibiliCommentPage(
                    sort=sort,
                    requested_limit=safe_limit,
                    reply_limit=safe_reply_limit,
                    page=safe_page,
                    limitations=["bvid_to_aid_resolution_failed"],
                    error="video aid unavailable",
                )
        try:
            await self._respect_rate_limit()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    self.COMMENT_URL,
                    params={
                        "oid": resolved_oid,
                        "type": 1,
                        "mode": mode,
                        "ps": safe_limit,
                        "pn": safe_page,
                    },
                    headers=HEADERS,
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as error:
            logger.warning("Failed to fetch comment page: %s", error)
            return BilibiliCommentPage(
                sort=sort,
                requested_limit=safe_limit,
                reply_limit=safe_reply_limit,
                page=safe_page,
                limitations=["comment_request_failed"],
                error=str(error),
            )
        if payload.get("code") != 0:
            message = str(payload.get("message", "comment API unavailable"))
            return BilibiliCommentPage(
                sort=sort,
                requested_limit=safe_limit,
                reply_limit=safe_reply_limit,
                page=safe_page,
                limitations=["comment_api_error"],
                error=message,
            )
        raw_response_data = payload.get("data", {})
        response_data = raw_response_data if isinstance(raw_response_data, dict) else {}
        comments: list[dict[str, Any]] = []
        for reply in response_data.get("replies") or []:
            if not isinstance(reply, dict):
                continue
            child_replies = [
                {
                    "user": child.get("member", {}).get("uname", ""),
                    "content": child.get("content", {}).get("message", ""),
                    "like_count": child.get("like", 0),
                    "ctime": child.get("ctime", 0),
                }
                for child in (reply.get("replies") or [])[:safe_reply_limit]
                if isinstance(child, dict)
            ]
            comments.append(
                {
                    "user": reply.get("member", {}).get("uname", ""),
                    "content": reply.get("content", {}).get("message", ""),
                    "like_count": reply.get("like", 0),
                    "reply_count": reply.get("rcount", 0),
                    "ctime": reply.get("ctime", 0),
                    "floor": reply.get("floor", 0),
                    "sampling": sort,
                    "replies": child_replies,
                }
            )
        cursor = response_data.get("cursor") or {}
        page_info = response_data.get("page") or {}
        total_comments = self._optional_int(cursor.get("all_count", page_info.get("count")))
        raw_is_end = cursor.get("is_end")
        has_more = None if raw_is_end is None else not bool(raw_is_end)
        limitations = ["platform_ordered_first_level_comment_sample"]
        if total_comments is None:
            limitations.append("total_comment_count_unavailable")
        bounded_comments = comments[:safe_limit]
        return BilibiliCommentPage(
            comments=bounded_comments,
            total_comments=total_comments,
            sort=sort,
            requested_limit=safe_limit,
            returned_count=len(bounded_comments),
            reply_limit=safe_reply_limit,
            page=safe_page,
            has_more=has_more,
            limitations=limitations,
        )

    async def get_video_aid(self, bvid: str) -> int:
        """Resolve bvid → aid (numeric ID needed for comment API)."""
        info = await self.get_video_info(bvid)
        aid_raw = info.get("aid", 0) or info.get("id", 0)
        return int(aid_raw) if aid_raw else 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of search calls made."""
        return self._call_count

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """Record of all search calls (for audit trail)."""
        return list(self._call_history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_results(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert B站 search API response to uniform content items.

        The B站 search API returns results grouped by type (video, article, etc.).
        We extract video results and normalize them into a flat list.
        """
        normalized: list[dict[str, Any]] = []
        results = data.get("data", {}).get("result", [])

        for section in results:
            if not isinstance(section, dict):
                continue
            result_type = str(section.get("result_type", "") or "unknown")
            items = section.get("data", [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "") or "").strip()
                effective_type = item_type if item_type else result_type
                bvid = str(item.get("bvid", "") or "")
                platform_id = BilibiliSearchTool._platform_id(
                    effective_type,
                    item,
                    bvid,
                )
                if not platform_id:
                    continue
                title_clean = BilibiliSearchTool._clean_title(str(item.get("title", "") or ""))
                author = str(
                    item.get("author", "")
                    or item.get("uname", "")
                    or item.get("name", "")
                    or ""
                )
                occurrence_id = (
                    f"bilibili-{bvid}"
                    if effective_type == "video" and bvid
                    else f"bilibili-{effective_type}-{platform_id}"
                )
                normalized.append(
                    {
                        "id": occurrence_id,
                        "source_id": "bilibili",
                        "canonical_url": BilibiliSearchTool._canonical_url(
                            effective_type, item, bvid
                        ),
                        "content_type": f"bilibili_{effective_type}",
                        "title": title_clean,
                        "author": author,
                        "author_mid": BilibiliSearchTool._optional_int(item.get("mid")),
                        "published_at": BilibiliSearchTool._publication_iso(item),
                        "engagement": {
                            "play": BilibiliSearchTool._optional_int(item.get("play")),
                            "danmaku": BilibiliSearchTool._optional_int(
                                item.get("danmaku", item.get("video_review"))
                            ),
                            "like": BilibiliSearchTool._optional_int(item.get("like")),
                            "reply": BilibiliSearchTool._optional_int(
                                item.get("reply", item.get("review"))
                            ),
                            "favorite": BilibiliSearchTool._optional_int(item.get("favorite")),
                            "coin": BilibiliSearchTool._optional_int(item.get("coin")),
                        },
                        "tags": BilibiliSearchTool._search_tags(item.get("tag")),
                        "tags_hydrated": False,
                        "text_snippet": str(item.get("description", "") or "")[:500],
                        "collected_at": datetime.now(UTC).isoformat(),
                        "content_hash": hashlib.sha256(
                            f"{effective_type}:{platform_id}:{title_clean}".encode()
                        ).hexdigest()[:16],
                    }
                )

        return normalized

    @staticmethod
    def _platform_id(result_type: str, item: dict[str, Any], bvid: str) -> str:
        if bvid:
            return bvid
        for key in (
            "roomid",
            "room_id",
            "id",
            "aid",
            "season_id",
            "media_id",
            "topic_id",
        ):
            value = str(item.get(key, "") or "").strip()
            if value:
                return value
        url = str(item.get("url", "") or item.get("goto_url", "") or "").strip()
        return hashlib.sha256(f"{result_type}:{url}".encode()).hexdigest()[:20] if url else ""

    @staticmethod
    def _canonical_url(result_type: str, item: dict[str, Any], bvid: str) -> str:
        if result_type == "video" and bvid:
            return f"https://www.bilibili.com/video/{bvid}"
        if result_type == "live_room":
            room_id = item.get("roomid", item.get("room_id", item.get("id", "")))
            return f"https://live.bilibili.com/{room_id}" if room_id else ""
        if result_type == "article":
            article_id = str(item.get("id", "") or "")
            return f"https://www.bilibili.com/read/{article_id}" if article_id else ""
        raw_url = str(item.get("url", "") or item.get("goto_url", "") or "")
        if raw_url.startswith("//"):
            return f"https:{raw_url}"
        if raw_url.startswith("/"):
            return f"https://www.bilibili.com{raw_url}"
        return raw_url

    @staticmethod
    def _search_tags(raw: object) -> list[str]:
        if isinstance(raw, list):
            return [str(value).strip() for value in raw if str(value).strip()]
        if not isinstance(raw, str):
            return []
        return [
            value.strip()
            for value in re.split(r"[,，;；\t]", BilibiliSearchTool._clean_title(raw))
            if value.strip()
        ]

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_bool(value: object) -> bool | None:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _parse_audience_value(
        raw: object,
        show_switch: bool | None,
    ) -> tuple[BilibiliAudienceValue, list[str]]:
        if show_switch is False:
            return (
                BilibiliAudienceValue(
                    precision=BilibiliAudiencePrecision.HIDDEN,
                    show_switch=False,
                ),
                ["hidden_by_show_switch"],
            )
        display = str(raw).strip() if raw is not None else ""
        if re.fullmatch(r"\d+", display):
            value = int(display)
            return (
                BilibiliAudienceValue(
                    precision=BilibiliAudiencePrecision.EXACT,
                    value=value,
                    display=display,
                    show_switch=show_switch,
                ),
                [],
            )
        rounded = re.fullmatch(r"(\d+)\+", display)
        if rounded:
            return (
                BilibiliAudienceValue(
                    precision=BilibiliAudiencePrecision.ROUNDED,
                    lower_bound=int(rounded.group(1)),
                    display=display,
                    show_switch=show_switch,
                ),
                ["rounded_lower_bound"],
            )
        if not display:
            return (
                BilibiliAudienceValue(
                    precision=BilibiliAudiencePrecision.UNAVAILABLE,
                    show_switch=show_switch,
                ),
                ["value_missing"],
            )
        return (
            BilibiliAudienceValue(
                precision=BilibiliAudiencePrecision.UNAVAILABLE,
                display=display,
                show_switch=show_switch,
            ),
            ["unrecognized_display_format"],
        )

    @staticmethod
    def _concurrent_audience_failure(
        bvid: str,
        cid: int,
        *,
        kind: str,
        error: str,
        error_code: int | None = None,
    ) -> BilibiliConcurrentAudienceResult:
        unavailable = BilibiliAudienceValue(
            precision=BilibiliAudiencePrecision.UNAVAILABLE
        )
        return BilibiliConcurrentAudienceResult(
            bvid=bvid,
            cid=cid,
            total=unavailable,
            count=unavailable,
            limitations=[f"concurrent_audience_{kind}"],
            error_kind=kind,
            error_code=error_code,
            error=error,
        )

    @staticmethod
    def _timestamp_iso(value: object) -> str | None:
        timestamp = BilibiliSearchTool._optional_int(value)
        if timestamp is None or timestamp <= 0:
            return None
        try:
            return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    @staticmethod
    def _publication_iso(item: dict[str, Any]) -> str | None:
        """Normalize common type-specific publication fields without title rules."""
        for key in (
            "pubdate",
            "pubtime",
            "pub_time",
            "publish_time",
            "ctime",
            "start_time",
        ):
            raw = item.get(key)
            timestamp = BilibiliSearchTool._timestamp_iso(raw)
            if timestamp:
                return timestamp
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
                except ValueError:
                    continue
                normalized = (
                    parsed.replace(tzinfo=UTC)
                    if parsed.tzinfo is None
                    else parsed.astimezone(UTC)
                )
                return normalized.isoformat()
        return None

    @staticmethod
    def _clean_title(raw: str) -> str:
        """Remove HTML tags from B站 titles.

        B站 wraps search hit terms in <em class="keyword"> tags.
        """
        return re.sub(r"<[^>]+>", "", raw)
