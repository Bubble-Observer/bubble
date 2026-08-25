"""Read-only web discovery and bounded document extraction."""

from __future__ import annotations

import asyncio
import html
import io
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
import trafilatura

from leave_information_bubble.runtime.errors import AgentError, ErrorCode
from leave_information_bubble.security import (
    UrlPolicyViolation,
    UrlSafetyPolicy,
)

DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
BING_RSS_URL = "https://www.bing.com/search"
BING_NEWS_RSS_URL = "https://www.bing.com/news/search"
WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh-Hans;q=0.9,en;q=0.5",
}


class WebSearchIndex(StrEnum):
    """Public index surface selected independently from a concrete provider."""

    GENERAL = "general"
    NEWS = "news"


class WebDocumentOutcome(StrEnum):
    """Qualification result for a fetched public document's static material."""

    QUALIFIED_FULL = "qualified_full"
    PARTIAL_CONTENT = "partial_content"
    PAGE_SHELL = "page_shell"
    LOGIN_OR_PAYWALL = "login_or_paywall"
    CHALLENGE_OR_BLOCKED = "challenge_or_blocked"
    RENDERING_REQUIRED = "rendering_required"
    EXTRACTION_FAILED = "extraction_failed"


@dataclass(frozen=True)
class WebSearchResult:
    """Normalized public web search result."""

    items: list[dict[str, str]] = field(default_factory=list)
    query: str = ""
    error: str = ""
    physical_call_count: int = 1
    provider: str = ""
    index: str = WebSearchIndex.GENERAL.value
    language: str = "zh-Hans"
    region: str = "CN"
    timezone: str = "Asia/Shanghai"
    time_filter_requested: bool = False
    time_filter_applied: bool = False
    time_filter_precision: str = "unsupported"
    provider_attempts: tuple[str, ...] = ()
    empty_or_challenge_attempts: int = 0
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class WebDocument:
    """Bounded text extracted from one public web document."""

    url: str
    title: str = ""
    text: str = ""
    content_type: str = ""
    links: tuple[str, ...] = ()
    error: str = ""
    outcome: WebDocumentOutcome = WebDocumentOutcome.QUALIFIED_FULL
    limitations: tuple[str, ...] = ()


class _TextExtractor(HTMLParser):
    """Collect links and visible-text diagnostics without claiming article extraction."""

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._chrome_depth = 0
        self.parts: list[str] = []
        self.content_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        if tag in {"title", "nav", "header", "footer", "aside"}:
            self._chrome_depth += 1
        if tag == "a" and not self._ignored_depth:
            href = next((value for key, value in attrs if key == "href"), None)
            if href and href.strip():
                self.links.append(href.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag in {"title", "nav", "header", "footer", "aside"} and self._chrome_depth:
            self._chrome_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            value = data.strip()
            self.parts.append(value)
            if not self._chrome_depth:
                self.content_parts.append(value)


class PublicWebSearchTool:
    """Search the public web and extract bounded HTML or PDF text."""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_document_bytes: int = 10_000_000,
        max_document_chars: int = 80_000,
        max_redirects: int = 5,
        url_policy: UrlSafetyPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
        self._timeout = timeout
        self._max_document_bytes = max_document_bytes
        self._max_document_chars = max_document_chars
        self._max_redirects = max_redirects
        self._url_policy = url_policy or UrlSafetyPolicy()
        self._transport = transport

    async def search(
        self,
        query: str,
        limit: int = 10,
        *,
        language: str = "zh-Hans",
        region: str = "CN",
        timezone: str = "Asia/Shanghai",
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        index: WebSearchIndex | str = WebSearchIndex.GENERAL,
    ) -> WebSearchResult:
        """Search a Chinese-localized general or news index with one fallback."""
        safe_limit = max(1, min(limit, 50))
        selected_index = WebSearchIndex(index)
        providers = self._provider_plan(
            selected_index,
            query=query,
            language=language,
            region=region,
            window_start=window_start,
            window_end=window_end,
        )
        attempts: list[str] = []
        limitations: list[str] = []
        empty_or_challenge_attempts = 0
        items: list[dict[str, str]] = []
        selected_provider = ""
        applied = False
        precision = "unsupported"
        best_candidate: (
            tuple[
                float,
                list[dict[str, str]],
                str,
                str,
            ]
            | None
        ) = None
        last_http_error: httpx.HTTPError | None = None
        extractor: _TextExtractor | None = None
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                for provider, url, params, parser, provider_precision in providers:
                    attempts.append(provider)
                    try:
                        response = await self._safe_get(client, url, params=params)
                        response.raise_for_status()
                    except UrlPolicyViolation as error:
                        # One public index can resolve through a blocked/private edge in
                        # the current network environment. Treat that as a provider-level
                        # coverage hole and try the one bounded alternative; never weaken
                        # SSRF validation or abort the entire logical search prematurely.
                        limitations.append(f"{provider}:url_policy:{error.code}")
                        empty_or_challenge_attempts += 1
                        continue
                    except httpx.HTTPError as error:
                        last_http_error = error
                        limitations.append(f"{provider}:http_error")
                        empty_or_challenge_attempts += 1
                        continue
                    try:
                        response_text = self._response_text(response)
                    except UnicodeError:
                        limitations.append(f"{provider}:encoding_error")
                        empty_or_challenge_attempts += 1
                        continue
                    if self._looks_like_search_challenge(response_text):
                        limitations.append(f"{provider}:challenge")
                        empty_or_challenge_attempts += 1
                        continue
                    parsed_items = parser(response_text, safe_limit)
                    if not parsed_items:
                        limitations.append(f"{provider}:empty")
                        empty_or_challenge_attempts += 1
                        continue
                    anchored_items = self._anchored_items(parsed_items, query=query)
                    dropped_count = len(parsed_items) - len(anchored_items)
                    if dropped_count:
                        limitations.append(f"{provider}:dropped_unanchored_results:{dropped_count}")
                    if not anchored_items:
                        limitations.append(f"{provider}:no_query_anchored_results")
                        _, language_ratio, anchor_ratio = self._candidate_quality(
                            parsed_items,
                            query=query,
                            language=language,
                        )
                        limitations.append(
                            f"{provider}:candidate_contract_below_threshold:"
                            f"language={language_ratio:.2f}:anchor={anchor_ratio:.2f}"
                        )
                        empty_or_challenge_attempts += 1
                        continue
                    quality, language_ratio, anchor_ratio = self._candidate_quality(
                        anchored_items,
                        query=query,
                        language=language,
                    )
                    candidate = (
                        quality,
                        anchored_items,
                        provider,
                        provider_precision,
                    )
                    if best_candidate is None or quality > best_candidate[0]:
                        best_candidate = candidate
                    if language_ratio >= 0.8 and anchor_ratio >= 0.6:
                        break
                    limitations.append(
                        f"{provider}:candidate_contract_below_threshold:"
                        f"language={language_ratio:.2f}:anchor={anchor_ratio:.2f}"
                    )
                if best_candidate is not None:
                    _, items, selected_provider, precision = best_candidate
                    applied = window_start is not None and window_end is not None
                    if applied:
                        limitations.append("provider_time_filter_forwarded_but_effect_not_verified")
        except UrlPolicyViolation as error:
            raise AgentError(
                ErrorCode.SOURCE_UNAVAILABLE,
                f"web search target rejected: {error.code}",
            ) from error
        except httpx.TimeoutException as error:
            raise AgentError(ErrorCode.TOOL_TRANSIENT, f"web search timeout: {error}") from error
        if not items and last_http_error is not None and len(attempts) == len(providers):
            limitations.append(f"all_providers_failed:{type(last_http_error).__name__}")
        return WebSearchResult(
            items=items,
            query=query,
            error="" if items else "public_web_no_results",
            physical_call_count=len(attempts),
            provider=selected_provider,
            index=selected_index.value,
            language=language,
            region=region,
            timezone=timezone,
            time_filter_requested=window_start is not None and window_end is not None,
            time_filter_applied=applied,
            time_filter_precision=precision,
            provider_attempts=tuple(attempts),
            empty_or_challenge_attempts=empty_or_challenge_attempts,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _candidate_quality(
        items: list[dict[str, str]],
        *,
        query: str,
        language: str,
    ) -> tuple[float, float, float]:
        """Rank provider candidates by contract fit, never semantic importance."""
        if not items:
            return 0.0, 0.0, 0.0
        joined = [f"{item.get('title', '')} {item.get('snippet', '')}" for item in items]
        if language.casefold().startswith("zh"):
            language_hits = sum(
                bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
                and not bool(re.search(r"[\u3040-\u30ff]", text))
                for text in joined
            )
        else:
            language_hits = len(joined)
        anchors = [
            token.casefold()
            for token in re.findall(
                r"[A-Za-z0-9][A-Za-z0-9_.+-]{1,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,}",
                query,
            )
        ]
        anchor_hits = sum(
            not anchors or any(anchor in text.casefold() for anchor in anchors) for text in joined
        )
        language_ratio = language_hits / len(joined)
        anchor_ratio = anchor_hits / len(joined)
        return (language_ratio * 0.7) + (anchor_ratio * 0.3), language_ratio, anchor_ratio

    @staticmethod
    def _anchored_items(
        items: list[dict[str, str]],
        *,
        query: str,
    ) -> list[dict[str, str]]:
        """Keep index candidates that repeat at least one concrete query anchor.

        Provider-level quality alone can hide unrelated tail results whenever a
        few good hits lift the batch score.  Search snippets remain discovery
        hints, but an item still needs a lexical connection to the request before
        it is allowed to consume an open/hydration budget.
        """
        anchors = [
            token.casefold()
            for token in re.findall(
                r"[A-Za-z0-9][A-Za-z0-9_.+-]{1,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,}",
                re.sub(r"\b(?:site|filetype|inurl|intitle):\S+", " ", query, flags=re.I),
            )
        ]
        anchors = list(dict.fromkeys(anchors))
        if not anchors:
            return items
        return [
            item
            for item in items
            if any(
                anchor
                in (f"{item.get('title', '')} {item.get('snippet', '')} {item.get('url', '')}").casefold()
                for anchor in anchors
            )
        ]

    @classmethod
    def _provider_plan(
        cls,
        index: WebSearchIndex,
        *,
        query: str,
        language: str,
        region: str,
        window_start: datetime | None,
        window_end: datetime | None,
    ) -> list[
        tuple[
            str,
            str,
            dict[str, str],
            Callable[[str, int], list[dict[str, str]]],
            str,
        ]
    ]:
        """Build at most two public-provider attempts for a logical index."""
        bing_params = cls._bing_params(
            query,
            language=language,
            region=region,
            window_start=window_start,
            window_end=window_end,
        )
        if index is WebSearchIndex.NEWS:
            return [
                (
                    "bing_news_rss",
                    BING_NEWS_RSS_URL,
                    {**bing_params, "format": "rss"},
                    cls._parse_bing_rss,
                    "coarse",
                ),
                (
                    "bing_web_rss",
                    BING_RSS_URL,
                    {**bing_params, "format": "rss"},
                    cls._parse_bing_rss,
                    "coarse",
                ),
            ]
        return [
            (
                "duckduckgo_html",
                DUCKDUCKGO_HTML_URL,
                cls._duckduckgo_params(
                    query,
                    language=language,
                    region=region,
                    window_start=window_start,
                    window_end=window_end,
                ),
                cls._parse_search_html,
                "coarse",
            ),
            (
                "bing_web_rss",
                BING_RSS_URL,
                {**bing_params, "format": "rss"},
                cls._parse_bing_rss,
                "coarse",
            ),
        ]

    @staticmethod
    def _duckduckgo_params(
        query: str,
        *,
        language: str,
        region: str,
        window_start: datetime | None,
        window_end: datetime | None,
    ) -> dict[str, str]:
        params = {
            "q": query,
            "kl": "cn-zh"
            if region.upper() == "CN" and language.casefold().startswith("zh")
            else f"{region.lower()}-{language.split('-', 1)[0].lower()}",
        }
        freshness = PublicWebSearchTool._coarse_freshness(window_start, window_end)
        if freshness:
            params["df"] = freshness
        return params

    @staticmethod
    def _bing_params(
        query: str,
        *,
        language: str,
        region: str,
        window_start: datetime | None,
        window_end: datetime | None,
    ) -> dict[str, str]:
        language_prefix = language.split("-", 1)[0].lower()
        market_language = "zh" if language_prefix == "zh" else language_prefix
        params = {
            "q": query,
            "cc": region.upper(),
            "mkt": f"{market_language}-{region.upper()}",
            "setlang": language,
        }
        if window_start is not None and window_end is not None:
            params["freshness"] = f"{window_start.date().isoformat()}..{window_end.date().isoformat()}"
        return params

    @staticmethod
    def _coarse_freshness(
        window_start: datetime | None,
        window_end: datetime | None,
    ) -> str:
        if window_start is None or window_end is None:
            return ""
        duration_days = max(0, (window_end - window_start).days)
        if duration_days <= 1:
            return "d"
        if duration_days <= 7:
            return "w"
        if duration_days <= 31:
            return "m"
        return "y"

    @staticmethod
    def _all_lines_match(payload: str, patterns: tuple[str, ...]) -> bool:
        """Return whether every non-blank line is a recognizable page-control line."""
        lines = [line.strip().casefold() for line in payload.splitlines() if line.strip()]
        return bool(lines) and all(
            any(re.fullmatch(pattern, line, re.IGNORECASE) for pattern in patterns) for line in lines
        )

    @staticmethod
    def _looks_like_search_challenge(payload: str) -> bool:
        """Detect provider challenge markup before attempting to parse search results."""
        normalized = payload.casefold()
        markers = (
            "captcha",
            "unusual traffic",
            "verify you are human",
            "验证您是真人",
            "安全验证",
        )
        return any(marker in normalized for marker in markers)

    @classmethod
    def _looks_like_challenge(cls, payload: str) -> bool:
        """Recognize a challenge page, not an article that merely discusses one."""
        return cls._all_lines_match(
            payload,
            (
                r"captcha(?::.*)?",
                r"security check(?::.*)?",
                r"unusual traffic(?::.*)?",
                r"verify you are human[.!]?",
                r"human verification(?: required)?[.!]?",
                r"请完成(?:人机|安全)?验证[！!。.]?",
                r"安全验证[！!。.]?",
                r"验证您是真人[！!。.]?",
            ),
        )

    @classmethod
    def _looks_like_login_or_paywall(cls, payload: str) -> bool:
        """Recognize an access prompt, not ordinary prose containing login vocabulary."""
        return cls._all_lines_match(
            payload,
            (
                r"(?:please )?(?:log in|login|sign in)(?: to (?:continue|read|view).*)?[.!]?",
                r"subscription required[.!]?",
                r"subscribe to continue[.!]?",
                r"paywall[.!]?",
                r"请登录(?:后.*)?[！!。.]?",
                r"登录后(?:才)?(?:可以|可|继续)?(?:查看|阅读).*[！!。.]?",
                r"(?:请)?绑定大区.*[！!。.]?",
                r"订阅后查看.*[！!。.]?",
                r"付费阅读[！!。.]?",
                r"会员专享[！!。.]?",
            ),
        )

    @classmethod
    def _looks_like_partial_content(cls, payload: str) -> bool:
        """Recognize a preview control, not prose that quotes its wording."""
        return cls._all_lines_match(
            payload,
            (
                r"read more(?: to continue reading)?[.!]?",
                r"continue reading[.!]?",
                r"preview only[.!]?",
                r"全文请见.*[！!。.]?",
                r"阅读全文[！!。.]?",
                r"展开剩余(?:内容)?[！!。.]?",
                r"剩余内容.*[！!。.]?",
                r"仅显示部分.*[！!。.]?",
            ),
        )

    @staticmethod
    def _looks_like_rendering_shell(payload: str) -> bool:
        """Return whether static markup signals JavaScript-rendered content is needed."""
        normalized = payload.casefold()
        markers = (
            "enable javascript",
            "javascript required",
            "javascript is disabled",
            "js-rendered",
            "single page application",
            "\u8bf7\u5f00\u542fjavascript",
            "\u8bf7\u542f\u7528javascript",
        )
        if any(marker in normalized for marker in markers):
            return True
        return bool(
            re.search(r'id=["\'](?:app|root)["\']', normalized)
            and re.search(r"<script\b[^>]+\bsrc=", normalized)
        )

    @staticmethod
    def _looks_like_page_shell(payload: str) -> bool:
        """Return whether visible diagnostic text carries common page-chrome signals."""
        normalized = payload.casefold()
        markers = (
            "copyright",
            "all rights reserved",
            "related articles",
            "related recommendations",
            "\u6682\u65e0\u6bb5\u4f4d",
            "\u76f8\u5173\u63a8\u8350",
            "\u7f51\u7ad9\u5bfc\u822a",
            "\u8d26\u53f7\u4e2d\u5fc3",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _shell_signal_count(payload: str) -> int:
        """Count independent chrome signals without using text length or a host rule."""
        normalized = payload.casefold()
        markers = (
            "copyright",
            "all rights reserved",
            "related articles",
            "related recommendations",
            "\u6682\u65e0\u6bb5\u4f4d",
            "\u76f8\u5173\u63a8\u8350",
            "\u7f51\u7ad9\u5bfc\u822a",
            "\u8d26\u53f7\u4e2d\u5fc3",
            "\u767b\u5f55",
            "\u7ed1\u5b9a\u5927\u533a",
        )
        return sum(marker in normalized for marker in markers)

    @classmethod
    def _is_shell_candidate(cls, text: str, diagnostic: str) -> bool:
        """Reject an extraction only when every returned line is recognizable chrome."""
        if not cls._looks_like_page_shell(diagnostic):
            return False
        shell_terms = (
            "copyright",
            "all rights reserved",
            "related articles",
            "related recommendations",
            "\u6682\u65e0\u6bb5\u4f4d",
            "\u76f8\u5173\u63a8\u8350",
            "\u7f51\u7ad9\u5bfc\u822a",
            "\u8d26\u53f7",
            "\u7ed1\u5b9a\u5927\u533a",
        )
        lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
        return (
            cls._shell_signal_count(diagnostic) >= 2
            and bool(lines)
            and all(any(term in line for term in shell_terms) for line in lines)
        )

    @classmethod
    def _is_utility_only_candidate(cls, text: str) -> bool:
        """Reject common footer/account controls only when they are the whole extraction."""
        return cls._all_lines_match(
            text,
            (
                r"(?:copyright|©).+",
                r"all rights reserved[.!]?",
                r"related (?:articles|recommendations)(?::.*)?",
                r"privacy policy[.!]?",
                r"terms (?:of use|and conditions)[.!]?",
                r"cookie (?:policy|settings|preferences)[.!]?",
                r"(?:help|account) cent(?:er|re)[.!]?",
                r"(?:back to|go to) (?:home|homepage)[.!]?",
                r"(?:open|download) (?:the )?app[.!]?",
                r"相关推荐(?::.*)?",
                r"隐私政策[！!。.]?",
                r"用户协议[！!。.]?",
                r"账号中心[！!。.]?",
                r"返回首页[！!。.]?",
                r"(?:打开|下载).{0,8}app[！!。.]?",
            ),
        )

    @staticmethod
    def _normalize_document_text(value: str) -> str:
        """Normalize extracted text without treating its length as a quality gate."""
        normalized = re.sub(r"[ \t]+", " ", html.unescape(value))
        return re.sub(r"\n{3,}", "\n\n", normalized).strip()

    @staticmethod
    def _json_ld_article_body(payload: str) -> str:
        """Return the first Article-like JSON-LD body, if present."""
        scripts = re.findall(
            r'<script\b[^>]*\btype=["\']application/ld\+json["\'][^>]*>(.*?)</script\s*>',
            payload,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for raw in scripts:
            try:
                decoded = json.loads(html.unescape(raw).strip())
            except (TypeError, ValueError):
                continue
            pending: list[object] = [decoded]
            while pending:
                candidate = pending.pop()
                if isinstance(candidate, list):
                    pending.extend(candidate)
                    continue
                if not isinstance(candidate, dict):
                    continue
                raw_type = candidate.get("@type", "")
                types = raw_type if isinstance(raw_type, list) else [raw_type]
                if any(str(item).casefold() in {"article", "newsarticle", "blogposting"} for item in types):
                    body = candidate.get("articleBody")
                    if isinstance(body, str) and body.strip():
                        return body
                graph = candidate.get("@graph")
                if isinstance(graph, list):
                    pending.extend(graph)
        return ""

    @classmethod
    def _html_document(
        cls, response_text: str, url: str, extractor: _TextExtractor
    ) -> tuple[WebDocumentOutcome, str]:
        """Extract static article text or classify a response as non-material."""
        diagnostic = "\n".join(extractor.parts)
        content_diagnostic = "\n".join(extractor.content_parts)
        if cls._looks_like_challenge(content_diagnostic):
            return WebDocumentOutcome.CHALLENGE_OR_BLOCKED, ""
        if cls._looks_like_login_or_paywall(content_diagnostic):
            return WebDocumentOutcome.LOGIN_OR_PAYWALL, ""
        if cls._looks_like_partial_content(content_diagnostic):
            return WebDocumentOutcome.PARTIAL_CONTENT, ""
        structured = cls._normalize_document_text(cls._json_ld_article_body(response_text))
        if structured:
            return WebDocumentOutcome.QUALIFIED_FULL, structured
        try:
            extracted = trafilatura.extract(
                response_text,
                url=url,
                favor_precision=True,
                include_comments=False,
                include_tables=True,
                include_links=False,
            )
        except Exception:
            extracted = None
        text = cls._normalize_document_text(extracted or "")
        shell_candidate = cls._is_shell_candidate(text, diagnostic)
        if (
            cls._is_utility_only_candidate(text)
            or shell_candidate
            or (
                not text
                and cls._looks_like_page_shell(diagnostic)
                and cls._shell_signal_count(diagnostic) >= 2
            )
        ):
            return WebDocumentOutcome.PAGE_SHELL, ""
        if text:
            return WebDocumentOutcome.QUALIFIED_FULL, text
        if cls._looks_like_rendering_shell(response_text):
            return WebDocumentOutcome.RENDERING_REQUIRED, ""
        if cls._looks_like_partial_content(response_text) or cls._looks_like_partial_content(diagnostic):
            return WebDocumentOutcome.PARTIAL_CONTENT, ""
        if cls._looks_like_login_or_paywall(response_text) or cls._looks_like_login_or_paywall(diagnostic):
            return WebDocumentOutcome.LOGIN_OR_PAYWALL, ""
        if cls._looks_like_page_shell(diagnostic):
            return WebDocumentOutcome.PAGE_SHELL, ""
        return WebDocumentOutcome.EXTRACTION_FAILED, ""

    async def fetch(self, url: str) -> WebDocument:
        """Fetch and extract a bounded public HTTP(S) document."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                response = await self._safe_get(client, url)
        except UrlPolicyViolation as error:
            return WebDocument(url=url, error=f"url rejected: {error.code}")
        except httpx.TimeoutException as error:
            raise AgentError(ErrorCode.TOOL_TRANSIENT, f"document timeout: {error}") from error
        except httpx.HTTPError as error:
            return WebDocument(url=url, error=f"document unavailable: {error}")
        content = response.content
        if len(content) > self._max_document_bytes:
            return WebDocument(url=url, error="document exceeds bounded extraction size")
        content_type = response.headers.get("content-type", "").lower()
        parsed = urlparse(str(response.url))
        response_text = ""
        extractor: _TextExtractor | None = None
        try:
            if "html" in content_type:
                response_text = self._response_text(response)
                extractor = _TextExtractor()
                extractor.feed(response_text)
                if response.status_code in {401, 402}:
                    outcome = WebDocumentOutcome.LOGIN_OR_PAYWALL
                    text = ""
                elif response.status_code in {403, 429}:
                    outcome = WebDocumentOutcome.CHALLENGE_OR_BLOCKED
                    text = ""
                else:
                    response.raise_for_status()
                    outcome, text = self._html_document(response_text, str(response.url), extractor)
                return await self._document_result(
                    response,
                    content_type,
                    extractor,
                    title=self._title(response_text),
                    text=text,
                    outcome=outcome,
                )
            response.raise_for_status()
            if "pdf" in content_type or parsed.path.lower().endswith(".pdf"):
                from pypdf import PdfReader

                text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
            else:
                response_text = self._response_text(response)
                text = response_text
        except Exception as error:
            return WebDocument(url=url, content_type=content_type, error=f"extraction failed: {error}")
        normalized = self._normalize_document_text(text)
        outcome = WebDocumentOutcome.QUALIFIED_FULL if normalized else WebDocumentOutcome.EXTRACTION_FAILED
        return await self._document_result(
            response,
            content_type,
            extractor,
            title="",
            text=normalized,
            outcome=outcome,
        )

    async def _document_result(
        self,
        response: httpx.Response,
        content_type: str,
        extractor: _TextExtractor | None,
        *,
        title: str,
        text: str,
        outcome: WebDocumentOutcome,
    ) -> WebDocument:
        """Build one qualified or typed-refusal document while retaining safe one-hop links."""
        links: list[str] = []
        for raw_link in extractor.links if extractor is not None else []:
            candidate = urljoin(str(response.url), raw_link)
            try:
                target = await asyncio.to_thread(self._url_policy.validate_url, candidate)
            except UrlPolicyViolation:
                continue
            if target.normalized_url not in links:
                links.append(target.normalized_url)
            if len(links) >= 8:
                break
        limitations = () if outcome is WebDocumentOutcome.QUALIFIED_FULL else (f"public_web_{outcome.value}",)
        return WebDocument(
            url=str(response.url),
            title=title,
            text=text[: self._max_document_chars],
            content_type=content_type,
            links=tuple(links),
            outcome=outcome,
            limitations=limitations,
        )

    async def _safe_get(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        target = await asyncio.to_thread(self._url_policy.validate_url, url)
        for redirect_count in range(self._max_redirects + 1):
            await asyncio.to_thread(self._url_policy.revalidate_target, target)
            response = await client.get(
                target.normalized_url,
                params=params,
                headers=WEB_HEADERS,
            )
            params = None
            if not response.is_redirect:
                return response
            location = response.headers.get("location", "")
            if not location:
                raise UrlPolicyViolation(
                    "redirect_location_missing",
                    "redirect response did not include a location",
                )
            if redirect_count >= self._max_redirects:
                raise UrlPolicyViolation(
                    "redirect_limit_exceeded",
                    "redirect chain exceeded the configured limit",
                )
            target = await asyncio.to_thread(
                self._url_policy.validate_redirect,
                target,
                location,
            )
        raise UrlPolicyViolation("redirect_limit_exceeded", "redirect limit exceeded")

    @staticmethod
    def _response_text(response: httpx.Response) -> str:
        """Decode HTML strictly, honoring a meta charset when headers omit it."""
        content = response.content
        prefix = content[:4096].decode("ascii", errors="ignore")
        declared = re.search(
            r"charset\s*=\s*[\"']?([a-zA-Z0-9._-]+)",
            prefix,
            re.IGNORECASE,
        )
        candidates: list[str] = []
        if declared:
            candidates.append(declared.group(1))
        content_type = response.headers.get("content-type", "")
        header = re.search(r"charset\s*=\s*([a-zA-Z0-9._-]+)", content_type, re.I)
        if header:
            candidates.append(header.group(1))
        candidates.extend(("utf-8", "gb18030"))
        for encoding in dict.fromkeys(candidates):
            try:
                text = content.decode(encoding, errors="strict")
            except (LookupError, UnicodeDecodeError):
                continue
            if "\ufffd" in text:
                # A strict decode that still contains U+FFFD means the source
                # already carried replacement characters.  Trying a permissive
                # legacy codec after that can turn the damage into plausible
                # looking mojibake, so quarantine the response immediately.
                raise UnicodeError("decoded response contains replacement character")
            return text
        raise UnicodeError("response bytes cannot be decoded without replacement")

    @staticmethod
    def _parse_search_html(payload: str, limit: int) -> list[dict[str, str]]:
        pattern = re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        matches = list(pattern.finditer(payload))
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, match in enumerate(matches):
            raw_url, raw_title = match.groups()
            url = PublicWebSearchTool._unwrap_url(html.unescape(raw_url))
            if not url or url in seen:
                continue
            seen.add(url)
            title = re.sub(r"<[^>]+>", "", html.unescape(raw_title)).strip()
            segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(payload)
            segment = payload[match.end() : segment_end]
            snippet_match = re.search(
                r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
                segment,
                re.IGNORECASE | re.DOTALL,
            )
            snippet = (
                re.sub(
                    r"\s+",
                    " ",
                    re.sub(
                        r"<[^>]+>",
                        "",
                        html.unescape(snippet_match.group(1)),
                    ),
                ).strip()
                if snippet_match
                else ""
            )
            items.append({"url": url, "title": title, "snippet": snippet})
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _unwrap_url(value: str) -> str:
        if value.startswith("//"):
            value = f"https:{value}"
        parsed = urlparse(value)
        if "duckduckgo.com" in (parsed.hostname or ""):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            return unquote(target)
        return value if parsed.scheme in {"http", "https"} else ""

    @staticmethod
    def _parse_bing_rss(payload: str, limit: int) -> list[dict[str, str]]:
        """Parse the public RSS fallback without trusting embedded markup."""
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            return []
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in root.findall(".//item"):
            url = (item.findtext("link") or "").strip()
            if not PublicWebSearchTool._unwrap_url(url) or url in seen:
                continue
            seen.add(url)
            title = re.sub(r"\s+", " ", item.findtext("title") or "").strip()
            description = item.findtext("description") or ""
            snippet = re.sub(
                r"\s+",
                " ",
                re.sub(r"<[^>]+>", "", html.unescape(description)),
            ).strip()
            result = {"url": url, "title": title, "snippet": snippet}
            published_at = (item.findtext("pubDate") or "").strip()
            if published_at:
                result["published_at"] = published_at
            items.append(result)
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _title(payload: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", payload, re.IGNORECASE | re.DOTALL)
        return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip() if match else ""


__all__ = [
    "PublicWebSearchTool",
    "WebDocument",
    "WebDocumentOutcome",
    "WebSearchIndex",
    "WebSearchResult",
]
