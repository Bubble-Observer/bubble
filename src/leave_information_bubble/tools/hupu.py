"""Bounded, read-only access to public Hupu forum pages.

The tool deliberately uses only anonymous public HTML.  It does not log in,
solve challenges, or call private application endpoints.  Parsers extract a
small semantic contract and fail closed when Hupu returns a challenge or an
unknown page shape.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

_BASE_URL = "https://bbs.hupu.com/"
_THREAD_RE = re.compile(r"/(\d+)(?:-\d+)?\.html(?:[?#].*)?$")
_BOARD_RE = re.compile(r"^[a-zA-Z0-9_-]{1,40}$")
_NUMBER_RE = re.compile(r"\d[\d,]*")
_CHALLENGE_MARKERS = ("安全验证", "访问验证", "captcha", "请完成验证")
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


@dataclass(frozen=True, slots=True)
class HupuThreadCard:
    """One thread listed on a public Hupu board page."""

    thread_id: str
    title: str
    url: str
    author: str = ""
    published_at: datetime | None = None
    reply_count: int | None = None
    view_count: int | None = None


@dataclass(frozen=True, slots=True)
class HupuReply:
    """One visible top-level reply from a bounded public thread page."""

    reply_ref: str
    content: str
    author: str = ""
    published_at: datetime | None = None
    highlighted: bool = False


@dataclass(frozen=True, slots=True)
class HupuThreadPage:
    """Parsed public thread content and a bounded visible reply sample."""

    thread_id: str
    url: str
    title: str = ""
    content: str = ""
    reply_count: int | None = None
    light_count: int | None = None
    view_count: int | None = None
    replies: tuple[HupuReply, ...] = ()
    error: str = ""
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HupuBoardPage:
    """Parsed board result with explicit operational status."""

    board: str
    page: int
    url: str
    threads: tuple[HupuThreadCard, ...] = ()
    error: str = ""
    limitations: tuple[str, ...] = ()


class _BoardParser(HTMLParser):
    def __init__(self, *, now: datetime) -> None:
        super().__init__(convert_charrefs=True)
        self.now = now
        self.depth = 0
        self.item_depth = -1
        self.current: dict[str, str] | None = None
        self.capture: tuple[str, int] | None = None
        self.buffer: list[str] = []
        self.items: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            return
        self.depth += 1
        values = dict(attrs)
        classes = values.get("class", "") or ""
        if tag == "li" and "bbs-sl-web-post-body" in classes:
            self.current = {}
            self.item_depth = self.depth
        if self.current is None:
            return
        field = ""
        if tag == "a" and "p-title" in classes:
            field = "title"
            self.current["href"] = values.get("href", "") or ""
        elif "post-datum" in classes:
            field = "datum"
        elif "post-auth" in classes:
            field = "author"
        elif "post-time" in classes:
            field = "time"
        if field:
            self.capture = (field, self.depth)
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.capture is not None:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if self.capture is not None and self.capture[1] == self.depth:
            if self.current is not None:
                self.current[self.capture[0]] = _clean_text(" ".join(self.buffer))
            self.capture = None
            self.buffer = []
        if self.current is not None and self.item_depth == self.depth and tag == "li":
            self.items.append(self.current)
            self.current = None
            self.item_depth = -1
        self.depth = max(0, self.depth - 1)


class _ThreadParser(HTMLParser):
    def __init__(self, *, thread_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.thread_id = thread_id
        self.depth = 0
        self.reply_depth = -1
        self.current_reply: dict[str, str] | None = None
        self.capture: tuple[str, int] | None = None
        self.buffer: list[str] = []
        self.title = ""
        self.content = ""
        self.metrics: dict[str, str] = {}
        self.replies: list[dict[str, str]] = []
        self.in_highlighted_section = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            return
        self.depth += 1
        values = dict(attrs)
        classes = values.get("class", "") or ""
        if "post-reply-list-container" in classes and self.current_reply is None:
            self.current_reply = {
                "highlighted": "1" if self.in_highlighted_section else "0"
            }
            self.reply_depth = self.depth
        field = ""
        if tag == "h1" and "index_name" in classes:
            field = "title"
        elif tag == "span" and "index_reply" in classes:
            field = "reply_count"
        elif tag == "span" and "index_light" in classes:
            field = "light_count"
        elif tag == "span" and "index_read" in classes:
            field = "view_count"
        elif self.current_reply is not None and "post-reply-list-user-info-top-name" in classes:
            field = "reply_author"
        elif self.current_reply is not None and "post-reply-list-user-info-top-time" in classes:
            field = "reply_time"
        elif "thread-content-detail" in classes:
            field = "reply_content" if self.current_reply is not None else "content"
        if field:
            self.capture = (field, self.depth)
            self.buffer = []

    def handle_data(self, data: str) -> None:
        normalized = _clean_text(data)
        if normalized == "这些回帖亮了":
            self.in_highlighted_section = True
        elif normalized in {"全部回复", "最新回复"}:
            self.in_highlighted_section = False
        if self.capture is not None:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if self.capture is not None and self.capture[1] == self.depth:
            field, _ = self.capture
            value = _clean_text(" ".join(self.buffer))
            if field == "title":
                self.title = value
            elif field == "content" and value and not self.content:
                self.content = value
            elif field.startswith("reply_") and self.current_reply is not None:
                self.current_reply[field.removeprefix("reply_")] = value
            else:
                self.metrics[field] = value
            self.capture = None
            self.buffer = []
        if (
            self.current_reply is not None
            and self.reply_depth == self.depth
            and tag == "div"
        ):
            if self.current_reply.get("content", "").strip():
                self.replies.append(self.current_reply)
            self.current_reply = None
            self.reply_depth = -1
        self.depth = max(0, self.depth - 1)


class HupuPublicTool:
    """Fetch and parse anonymous Hupu board/thread HTML with bounded caching."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 20.0,
        max_bytes: int = 4_000_000,
        cache_ttl_seconds: float = 120.0,
        min_interval_seconds: float = 0.35,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._transport = transport
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._cache_ttl = cache_ttl_seconds
        self._min_interval = min_interval_seconds
        self._now_factory = now_factory or (lambda: datetime.now(UTC))
        self._cache: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def scan_board(
        self,
        board: str = "lol",
        *,
        page: int = 1,
        limit: int = 20,
    ) -> HupuBoardPage:
        """Fetch and parse one bounded anonymous board page."""
        board = board.strip().casefold()
        if not _BOARD_RE.fullmatch(board):
            return HupuBoardPage(board=board, page=page, url="", error="invalid_hupu_board")
        page = min(max(int(page), 1), 100)
        limit = min(max(int(limit), 1), 100)
        url = urljoin(_BASE_URL, board if page == 1 else f"{board}-{page}")
        text, error = await self._fetch(url)
        if error:
            return HupuBoardPage(board=board, page=page, url=url, error=error)
        parser = _BoardParser(now=self._utc_now())
        parser.feed(text)
        cards: list[HupuThreadCard] = []
        for item in parser.items[:limit]:
            href = item.get("href", "")
            match = _THREAD_RE.search(href)
            title = item.get("title", "").strip()
            if match is None or not title:
                continue
            replies, views = _parse_pair(item.get("datum", ""))
            cards.append(HupuThreadCard(
                thread_id=match.group(1),
                title=title,
                url=urljoin(_BASE_URL, href),
                author=item.get("author", ""),
                published_at=_parse_board_time(item.get("time", ""), self._utc_now()),
                reply_count=replies,
                view_count=views,
            ))
        limitations: list[str] = []
        if not cards:
            marker = _challenge_marker(text)
            return HupuBoardPage(
                board=board,
                page=page,
                url=url,
                error=marker or "hupu_board_schema_changed",
            )
        limitations.append("hupu_board_order_is_platform_controlled")
        return HupuBoardPage(
            board=board,
            page=page,
            url=url,
            threads=tuple(cards),
            limitations=tuple(limitations),
        )

    async def get_thread(
        self,
        identifier: str,
        *,
        page: int = 1,
        reply_limit: int = 40,
    ) -> HupuThreadPage:
        """Fetch one public thread page and a bounded visible reply sample."""
        thread_id = self.thread_id(identifier)
        if not thread_id:
            return HupuThreadPage(thread_id="", url="", error="invalid_hupu_thread")
        page = min(max(int(page), 1), 100)
        reply_limit = min(max(int(reply_limit), 0), 100)
        suffix = f"{thread_id}.html" if page == 1 else f"{thread_id}-{page}.html"
        url = urljoin(_BASE_URL, suffix)
        text, error = await self._fetch(url)
        if error:
            return HupuThreadPage(thread_id=thread_id, url=url, error=error)
        parser = _ThreadParser(thread_id=thread_id)
        parser.feed(text)
        if not parser.title or not parser.content:
            marker = _challenge_marker(text)
            if marker:
                return HupuThreadPage(
                    thread_id=thread_id, url=url, error=marker
                )
            # a parsed title with no main-post text is a platform answer
            # (image-only / deleted-body / joke posts), not a layout drift:
            # label it honestly so callers do not read a broken parser into it
            return HupuThreadPage(
                thread_id=thread_id,
                url=url,
                error=(
                    "hupu_thread_no_public_text"
                    if parser.title and not parser.content
                    else "hupu_thread_schema_changed"
                ),
            )
        replies: list[HupuReply] = []
        seen: set[str] = set()
        for item in parser.replies:
            content = item.get("content", "").strip()
            author = item.get("author", "").strip()
            published_at = _parse_exact_time(item.get("time", ""))
            identity = f"{thread_id}\0{author}\0{item.get('time', '')}\0{content}"
            reply_ref = _sha_ref("hupu-reply", identity)
            if reply_ref in seen or not content:
                continue
            seen.add(reply_ref)
            replies.append(HupuReply(
                reply_ref=reply_ref,
                content=content,
                author=author,
                published_at=published_at,
                highlighted=item.get("highlighted") == "1",
            ))
            if len(replies) >= reply_limit:
                break
        return HupuThreadPage(
            thread_id=thread_id,
            url=url,
            title=parser.title,
            content=parser.content,
            reply_count=_parse_number(parser.metrics.get("reply_count", "")),
            light_count=_parse_number(parser.metrics.get("light_count", "")),
            view_count=_parse_number(parser.metrics.get("view_count", "")),
            replies=tuple(replies),
            limitations=(
                "visible_replies_are_platform_ordered_bounded_samples",
                "nested_reply_dialogs_are_not_expanded",
            ),
        )

    @staticmethod
    def thread_id(identifier: str) -> str:
        """Normalize a public URL, source reference, or numeric thread id."""
        value = identifier.strip()
        if value.startswith("hupu-thread-"):
            value = value.removeprefix("hupu-thread-")
        if value.isdigit():
            return value
        match = _THREAD_RE.search(urlparse(value).path)
        return match.group(1) if match else ""

    async def _fetch(self, url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "bbs.hupu.com":
            return "", "hupu_url_not_allowed"
        cached = self._cache.get(url)
        now_mono = time.monotonic()
        if cached and now_mono - cached[0] <= self._cache_ttl:
            return cached[1], ""
        async with self._lock:
            delay = self._min_interval - (time.monotonic() - self._last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                async with httpx.AsyncClient(
                    transport=self._transport,
                    timeout=self._timeout,
                    follow_redirects=True,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 Chrome/124 Safari/537.36"
                        ),
                        "Accept-Language": "zh-CN,zh;q=0.9",
                    },
                ) as client:
                    response = await client.get(url)
                self._last_request_at = time.monotonic()
            except httpx.TimeoutException:
                return "", "hupu_timeout"
            except httpx.HTTPError:
                return "", "hupu_http_error"
            if response.status_code in {401, 403}:
                return "", "hupu_access_blocked"
            if response.status_code == 429:
                return "", "hupu_rate_limited"
            if response.status_code >= 500:
                return "", f"hupu_upstream_{response.status_code}"
            if response.status_code != 200:
                return "", f"hupu_http_{response.status_code}"
            final_url = urlparse(str(response.url))
            if final_url.scheme != "https" or final_url.hostname != "bbs.hupu.com":
                return "", "hupu_redirect_not_allowed"
            payload = response.content[: self._max_bytes + 1]
            if len(payload) > self._max_bytes:
                return "", "hupu_document_too_large"
            text = payload.decode(response.encoding or "utf-8", errors="replace")
            marker = _challenge_marker(text)
            if marker:
                return "", marker
            self._cache[url] = (time.monotonic(), text)
            return text, ""

    def _utc_now(self) -> datetime:
        value = self._now_factory()
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _parse_number(value: str) -> int | None:
    match = _NUMBER_RE.search(value)
    return int(match.group(0).replace(",", "")) if match else None


def _parse_pair(value: str) -> tuple[int | None, int | None]:
    numbers = [int(item.replace(",", "")) for item in _NUMBER_RE.findall(value)]
    return (
        numbers[0] if numbers else None,
        numbers[1] if len(numbers) > 1 else None,
    )


def _parse_board_time(value: str, now: datetime) -> datetime | None:
    match = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", value)
    if match is None:
        return _parse_exact_time(value)
    cn = timezone(timedelta(hours=8))
    local_now = now.astimezone(cn)
    candidate = datetime(
        local_now.year,
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        tzinfo=cn,
    )
    if candidate > local_now + timedelta(days=1):
        candidate = candidate.replace(year=candidate.year - 1)
    return candidate.astimezone(UTC)


def _parse_exact_time(value: str) -> datetime | None:
    match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?",
        value,
    )
    if match is None:
        return None
    cn = timezone(timedelta(hours=8))
    return datetime(
        int(match.group(1)), int(match.group(2)), int(match.group(3)),
        int(match.group(4)), int(match.group(5)), int(match.group(6) or 0),
        tzinfo=cn,
    ).astimezone(UTC)


def _challenge_marker(text: str) -> str:
    lowered = text.casefold()
    return "hupu_challenge_page" if any(item.casefold() in lowered for item in _CHALLENGE_MARKERS) else ""


def _sha_ref(prefix: str, value: str) -> str:
    import hashlib

    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:32]}"


__all__ = [
    "HupuBoardPage",
    "HupuPublicTool",
    "HupuReply",
    "HupuThreadCard",
    "HupuThreadPage",
]
