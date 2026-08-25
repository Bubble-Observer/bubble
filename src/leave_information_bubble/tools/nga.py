"""Bounded, read-only access to public NGA forum pages.

The tool uses only public board and thread HTML.  It accepts an optional
caller-provided session cookie because NGA can require a browser-established
session even for otherwise public pages.  It never logs in, solves challenges,
or calls private application endpoints.  Parsers fail closed on access gates
and unfamiliar page shapes.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

_BASE_URL = "https://bbs.nga.cn/"
_FID_RE = re.compile(r"^-?\d{1,12}$")
_TID_RE = re.compile(r"^\d{1,20}$")
_FLOOR_ROW_RE = re.compile(r"^post1strow(\d+)$")
_POST_ANCHOR_RE = re.compile(r"^pid(\d+)Anchor$")
_DATE_RE = re.compile(
    r"(\d{2,4})-(\d{2})-(\d{2})(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?"
)
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}
_CHALLENGE_MARKERS = (
    "刷新过快",
    "请等候数秒再行访问",
    "访问频率过快",
    "captcha",
    "verify you are human",
)
_AUTH_MARKERS = (
    "你可能需要登录后访问",
    "你可能需要 登录 后访问",
    "请登录后访问",
)


@dataclass(frozen=True, slots=True)
class NgaThreadCard:
    """One thread listed on a public NGA board page."""

    thread_id: str
    title: str
    url: str
    author: str = ""
    published_at: datetime | None = None
    last_reply_at: datetime | None = None
    reply_count: int | None = None


@dataclass(frozen=True, slots=True)
class NgaReply:
    """One visible NGA floor from a bounded thread page."""

    reply_ref: str
    floor: int
    content: str
    author: str = ""
    author_ref: str = ""
    published_at: datetime | None = None
    support_count: int | None = None
    quoted_reply_ref: str = ""


@dataclass(frozen=True, slots=True)
class NgaThreadPage:
    """Parsed public thread content and visible floors from one page."""

    thread_id: str
    page: int
    url: str
    title: str = ""
    content: str = ""
    author: str = ""
    author_ref: str = ""
    published_at: datetime | None = None
    replies: tuple[NgaReply, ...] = ()
    error: str = ""
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NgaBoardPage:
    """Parsed board result with explicit operational status."""

    fid: str
    page: int
    url: str
    threads: tuple[NgaThreadCard, ...] = ()
    error: str = ""
    limitations: tuple[str, ...] = ()


class _BoardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.row_depth = -1
        self.current: dict[str, str] | None = None
        self.capture: tuple[str, int] | None = None
        self.buffer: list[str] = []
        self.items: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class", "") or "").split())
        if tag not in _VOID_TAGS:
            self.depth += 1
        if tag == "tr" and "topicrow" in classes:
            self.current = {}
            self.row_depth = self.depth
        if self.current is None:
            return
        field = ""
        if tag == "a" and "topic" in classes:
            field = "title"
            self.current["href"] = values.get("href", "") or ""
        elif tag == "a" and "author" in classes:
            field = "author"
        elif tag == "a" and "replies" in classes:
            field = "replies"
            self.current.setdefault("href", values.get("href", "") or "")
        elif tag == "span" and "postdate" in classes:
            field = "published"
            self.current["published_title"] = values.get("title", "") or ""
        elif tag == "a" and "replydate" in classes:
            field = "last_reply"
            self.current["last_reply_title"] = values.get("title", "") or ""
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
                self.current[self.capture[0]] = _clean_text("".join(self.buffer))
            self.capture = None
            self.buffer = []
        if self.current is not None and self.row_depth == self.depth and tag == "tr":
            self.items.append(self.current)
            self.current = None
            self.row_depth = -1
        self.depth = max(0, self.depth - 1)


class _ThreadParser(HTMLParser):
    def __init__(self, *, thread_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.thread_id = thread_id
        self.depth = 0
        self.row_depth = -1
        self.current: dict[str, str] | None = None
        self.capture: tuple[str, int] | None = None
        self.buffer: list[str] = []
        self.items: list[dict[str, str]] = []
        self.page_title = ""
        self._page_title_depth = -1
        self._page_title_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class", "") or "").split())
        if tag not in _VOID_TAGS:
            self.depth += 1
        if tag == "title" and self.current is None:
            self._page_title_depth = self.depth
            self._page_title_buffer = []
        row_match = _FLOOR_ROW_RE.fullmatch(values.get("id", "") or "")
        if tag == "tr" and "postrow" in classes and row_match is not None:
            self.current = {"floor": row_match.group(1)}
            self.row_depth = self.depth
        if self.current is None:
            return
        floor = self.current["floor"]
        identifier = values.get("id", "") or ""
        anchor_match = _POST_ANCHOR_RE.fullmatch(identifier)
        if anchor_match is not None and anchor_match.group(1) != "0":
            self.current["pid"] = anchor_match.group(1)
        if tag == "a" and identifier == f"postauthor{floor}":
            self.capture = ("author", self.depth)
            self.buffer = []
            self.current["author_href"] = values.get("href", "") or ""
            return
        if tag == "span" and identifier == f"postdate{floor}":
            self.capture = ("published", self.depth)
            self.buffer = []
            return
        if tag == "h3" and identifier == f"postsubject{floor}":
            self.capture = ("subject", self.depth)
            self.buffer = []
            return
        if tag in {"span", "p"} and identifier == f"postcontent{floor}":
            # live NGA wraps the main post in <p id='postcontent0'> while
            # reply floors keep <span id='postcontentN'>; accepting both
            # keeps the main-post body from being silently dropped (a TEXT
            # hydrate returned metadata only, losing the document)
            self.capture = ("content", self.depth)
            self.buffer = []
            return
        if tag == "span" and "recommendvalue" in classes:
            self.capture = ("support", self.depth)
            self.buffer = []
            return
        if self.capture is not None and self.capture[0] == "content":
            if tag in {"br", "p", "div", "blockquote", "h4"}:
                self.buffer.append("\n")
            if tag == "a":
                topid = parse_qs(urlparse(values.get("href", "") or "").query).get(
                    "topid", [""]
                )[0]
                if topid.isdigit():
                    self.current.setdefault("quoted_pid", topid)

    def handle_data(self, data: str) -> None:
        if self._page_title_depth >= 0:
            self._page_title_buffer.append(data)
        if self.capture is not None:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if self._page_title_depth == self.depth and tag == "title":
            self.page_title = _clean_text("".join(self._page_title_buffer))
            self._page_title_depth = -1
            self._page_title_buffer = []
        if self.capture is not None and self.capture[1] == self.depth:
            if self.current is not None:
                self.current[self.capture[0]] = _clean_text("".join(self.buffer))
            self.capture = None
            self.buffer = []
        if self.current is not None and self.row_depth == self.depth and tag == "tr":
            self.items.append(self.current)
            self.current = None
            self.row_depth = -1
        self.depth = max(0, self.depth - 1)


class NgaPublicTool:
    """Fetch and parse bounded public NGA board/thread HTML."""

    def __init__(
        self,
        *,
        session_cookie: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 25.0,
        max_bytes: int = 6_000_000,
        cache_ttl_seconds: float = 120.0,
        min_interval_seconds: float = 1.0,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_cookie = session_cookie.strip()
        self._transport = transport
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._cache_ttl = cache_ttl_seconds
        self._min_interval = min_interval_seconds
        self._now_factory = now_factory or (lambda: datetime.now(UTC))
        self._cache: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()
        self._last_request_at = 0.0

    @property
    def session_configured(self) -> bool:
        """Return whether the caller supplied an explicit NGA session cookie."""
        return bool(self._session_cookie)

    async def scan_board(
        self,
        fid: str = "-152678",
        *,
        page: int = 1,
        limit: int = 30,
    ) -> NgaBoardPage:
        """Fetch one public board page in platform order."""
        fid = str(fid).strip()
        if not _FID_RE.fullmatch(fid):
            return NgaBoardPage(fid=fid, page=page, url="", error="invalid_nga_fid")
        page = min(max(int(page), 1), 100)
        limit = min(max(int(limit), 1), 100)
        query = urlencode({"fid": fid, "page": page, "order_by": "postdatedesc"})
        url = f"{_BASE_URL}thread.php?{query}"
        text, error = await self._fetch(url)
        if error:
            return NgaBoardPage(fid=fid, page=page, url=url, error=error)
        parser = _BoardParser()
        parser.feed(text)
        cards: list[NgaThreadCard] = []
        seen: set[str] = set()
        for item in parser.items:
            thread_id = self.thread_id(item.get("href", ""))
            title = item.get("title", "").strip()
            if not thread_id or not title or thread_id in seen:
                continue
            seen.add(thread_id)
            cards.append(NgaThreadCard(
                thread_id=thread_id,
                title=title,
                url=f"{_BASE_URL}read.php?tid={thread_id}",
                author=item.get("author", "").strip(),
                published_at=_parse_nga_time(
                    item.get("published_title") or item.get("published", ""),
                    self._utc_now(),
                ),
                last_reply_at=_parse_nga_time(
                    item.get("last_reply_title") or item.get("last_reply", ""),
                    self._utc_now(),
                ),
                reply_count=_parse_number(item.get("replies", "")),
            ))
            if len(cards) >= limit:
                break
        if not cards:
            return NgaBoardPage(
                fid=fid,
                page=page,
                url=url,
                error=_page_error(text) or "nga_board_schema_changed",
            )
        return NgaBoardPage(
            fid=fid,
            page=page,
            url=url,
            threads=tuple(cards),
            limitations=(
                "nga_board_order_is_platform_controlled",
                "nga_board_contains_pinned_and_mirrored_topics",
            ),
        )

    async def get_thread(
        self,
        identifier: str,
        *,
        page: int = 1,
        reply_limit: int = 40,
    ) -> NgaThreadPage:
        """Fetch one thread page and parse its visible floor structure."""
        thread_id = self.thread_id(identifier)
        if not thread_id:
            return NgaThreadPage(
                thread_id="", page=page, url="", error="invalid_nga_thread"
            )
        page = min(max(int(page), 1), 100)
        reply_limit = min(max(int(reply_limit), 0), 100)
        url = f"{_BASE_URL}read.php?{urlencode({'tid': thread_id, 'page': page})}"
        text, error = await self._fetch(url)
        if error:
            return NgaThreadPage(
                thread_id=thread_id, page=page, url=url, error=error
            )
        parser = _ThreadParser(thread_id=thread_id)
        parser.feed(text)
        if not parser.items:
            return NgaThreadPage(
                thread_id=thread_id,
                page=page,
                url=url,
                error=_page_error(text) or "nga_thread_schema_changed",
            )
        main = next(
            (item for item in parser.items if item.get("floor") == "0"),
            None,
        )
        replies: list[NgaReply] = []
        for item in parser.items:
            floor = int(item.get("floor", "0") or 0)
            content = item.get("content", "").strip()
            if floor == 0 or not content:
                continue
            pid = item.get("pid", "").strip()
            reply_ref = f"nga-post-{pid}" if pid else f"nga-floor-{thread_id}-{floor}"
            quoted_pid = item.get("quoted_pid", "").strip()
            replies.append(NgaReply(
                reply_ref=reply_ref,
                floor=floor,
                content=content,
                author=item.get("author", "").strip(),
                author_ref=_author_ref(item.get("author_href", "")),
                published_at=_parse_nga_time(
                    item.get("published", ""), self._utc_now()
                ),
                support_count=_parse_number(item.get("support", "")),
                quoted_reply_ref=f"nga-post-{quoted_pid}" if quoted_pid else "",
            ))
            if len(replies) >= reply_limit:
                break
        title = (main or {}).get("subject", "").strip()
        if not title:
            title = _clean_page_title(parser.page_title)
        return NgaThreadPage(
            thread_id=thread_id,
            page=page,
            url=url,
            title=title,
            content=(main or {}).get("content", "").strip(),
            author=(main or {}).get("author", "").strip(),
            author_ref=_author_ref((main or {}).get("author_href", "")),
            published_at=_parse_nga_time(
                (main or {}).get("published", ""), self._utc_now()
            ),
            replies=tuple(replies),
            limitations=(
                "visible_floors_are_platform_ordered_bounded_samples",
                "collapsed_or_deleted_content_may_be_absent",
                "quoted_text_is_retained_with_quote_relation_when_available",
            ),
        )

    @staticmethod
    def thread_id(identifier: str) -> str:
        """Normalize an NGA URL, source reference, or numeric thread id."""
        value = str(identifier).strip()
        if value.startswith("nga-thread-"):
            value = value.removeprefix("nga-thread-")
        if _TID_RE.fullmatch(value):
            return value
        parsed = urlparse(value)
        if parsed.hostname not in {None, "bbs.nga.cn"}:
            return ""
        candidate = parse_qs(parsed.query).get("tid", [""])[0]
        return candidate if _TID_RE.fullmatch(candidate) else ""

    async def _fetch(self, url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "bbs.nga.cn":
            return "", "nga_url_not_allowed"
        cached = self._cache.get(url)
        now_mono = time.monotonic()
        if cached and now_mono - cached[0] <= self._cache_ttl:
            return cached[1], ""
        async with self._lock:
            delay = self._min_interval - (time.monotonic() - self._last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh-Hans;q=0.9",
                "Referer": _BASE_URL,
            }
            if self._session_cookie:
                headers["Cookie"] = self._session_cookie
            try:
                async with httpx.AsyncClient(
                    transport=self._transport,
                    timeout=self._timeout,
                    follow_redirects=True,
                    headers=headers,
                ) as client:
                    response = await client.get(url)
                self._last_request_at = time.monotonic()
            except httpx.TimeoutException:
                return "", "nga_timeout"
            except httpx.HTTPError:
                return "", "nga_http_error"
            if response.status_code in {401, 403}:
                return "", "nga_browser_session_required"
            if response.status_code == 429:
                return "", "nga_rate_limited"
            if response.status_code >= 500:
                return "", f"nga_upstream_{response.status_code}"
            if response.status_code != 200:
                return "", f"nga_http_{response.status_code}"
            final = urlparse(str(response.url))
            if final.scheme != "https" or final.hostname != "bbs.nga.cn":
                return "", "nga_redirect_not_allowed"
            payload = response.content[: self._max_bytes + 1]
            if len(payload) > self._max_bytes:
                return "", "nga_document_too_large"
            try:
                text = _decode_response(response, payload)
            except UnicodeError:
                return "", "nga_encoding_error"
            page_error = _page_error(text)
            if page_error:
                return "", page_error
            self._cache[url] = (time.monotonic(), text)
            return text, ""

    def _utc_now(self) -> datetime:
        value = self._now_factory()
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _clean_text(value: str) -> str:
    lines = [" ".join(line.replace("\xa0", " ").split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _clean_page_title(value: str) -> str:
    return re.sub(r"\s+NGA玩家社区.*$", "", value).strip()


def _parse_number(value: str) -> int | None:
    match = re.search(r"-?\d[\d,]*", value)
    return int(match.group(0).replace(",", "")) if match else None


def _parse_nga_time(value: str, now: datetime) -> datetime | None:
    match = _DATE_RE.search(value.strip())
    if match is None:
        # Live NGA board pages render post times as Unix epoch seconds in the
        # postdate span text (e.g. 1785291684) instead of a formatted date;
        # thread pages keep the formatted form. Accept both so discovery and
        # thread hydration agree on publish times.
        return _parse_epoch_time(value)
    cn = timezone(timedelta(hours=8))
    local_now = now.astimezone(cn)
    raw_year = int(match.group(1))
    year = raw_year + 2000 if raw_year < 100 else raw_year
    hour = int(match.group(4) or 0)
    minute = int(match.group(5) or 0)
    second = int(match.group(6) or 0)
    try:
        parsed = datetime(
            year,
            int(match.group(2)),
            int(match.group(3)),
            hour,
            minute,
            second,
            tzinfo=cn,
        )
    except ValueError:
        return None
    if raw_year < 100 and parsed > local_now + timedelta(days=1):
        parsed = parsed.replace(year=parsed.year - 100)
    return parsed.astimezone(UTC)


def _parse_epoch_time(value: str) -> datetime | None:
    """Parse a Unix epoch seconds (or milliseconds) literal, if plausible."""
    text = value.strip()
    if not re.fullmatch(r"\d{10,13}", text):
        return None
    try:
        parsed = datetime.fromtimestamp(int(text[:10]), tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    if parsed.year < 2000 or parsed.year > 2100:
        return None
    return parsed


def _author_ref(href: str) -> str:
    uid = parse_qs(urlparse(href).query).get("uid", [""])[0]
    return f"nga-user-{uid}" if uid.isdigit() else ""


def _page_error(text: str) -> str:
    normalized = "".join(text.split()).casefold()
    if any("".join(marker.split()).casefold() in normalized for marker in _CHALLENGE_MARKERS):
        return "nga_rate_limited"
    if any("".join(marker.split()).casefold() in normalized for marker in _AUTH_MARKERS):
        return "nga_browser_session_required"
    if "版面关闭" in normalized:
        return "nga_board_closed"
    return ""


def _decode_response(response: httpx.Response, payload: bytes) -> str:
    prefix = payload[:4096].decode("ascii", errors="ignore")
    candidates: list[str] = []
    header_match = re.search(
        r"charset\s*=\s*([a-zA-Z0-9._-]+)",
        response.headers.get("content-type", ""),
        re.I,
    )
    meta_match = re.search(
        r"charset\s*=\s*[\"']?([a-zA-Z0-9._-]+)", prefix, re.I
    )
    if header_match:
        candidates.append(header_match.group(1))
    if meta_match:
        candidates.append(meta_match.group(1))
    candidates.extend(("utf-8", "gb18030"))
    for encoding in dict.fromkeys(candidates):
        try:
            text = payload.decode(encoding, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
        if "\ufffd" not in text:
            return text
    raise UnicodeError("NGA response cannot be decoded without replacement")


__all__ = [
    "NgaBoardPage",
    "NgaPublicTool",
    "NgaReply",
    "NgaThreadCard",
    "NgaThreadPage",
]
