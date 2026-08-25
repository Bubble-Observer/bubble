from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import trafilatura

from leave_information_bubble.channels import (
    AcquisitionOutcome,
    ChannelCapabilityRole,
    HydrationDepth,
    HydrationRequest,
    PublicWebChannelAdapter,
    ScanRequest,
)
from leave_information_bubble.models.epistemics import ObservationModality
from leave_information_bubble.security import UrlSafetyPolicy
from leave_information_bubble.tools.web_search import (
    PublicWebSearchTool,
    WebDocumentOutcome,
    WebSearchIndex,
    WebSearchResult,
)
from leave_information_bubble.world import WorldStore
from leave_information_bubble.world.tools import WorldTools

NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)


class Resolver:
    def __init__(self, answers: dict[str, Sequence[str]]) -> None:
        self.answers = answers

    def __call__(self, host: str, port: int) -> Sequence[str]:
        del port
        return self.answers.get(host, ())


def _policy() -> UrlSafetyPolicy:
    return UrlSafetyPolicy(
        resolver=Resolver(
            {
                "html.duckduckgo.com": ("52.142.124.215",),
                "www.bing.com": ("13.107.21.200",),
            }
        )
    )


def test_meta_gbk_page_decodes_without_replacement_character() -> None:
    html = '<html><head><meta charset="gbk"></head><body>英雄联盟经典服</body></html>'
    response = httpx.Response(
        200,
        headers={"content-type": "text/html"},
        content=html.encode("gbk"),
    )

    decoded = PublicWebSearchTool._response_text(response)  # noqa: SLF001

    assert "英雄联盟经典服" in decoded
    assert "\ufffd" not in decoded


def test_literal_replacement_character_is_not_redecoded_as_legacy_mojibake() -> None:
    response = httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        content="<html><body>损坏\ufffd正文</body></html>".encode("utf-8"),
    )

    with pytest.raises(UnicodeError, match="replacement character"):
        PublicWebSearchTool._response_text(response)  # noqa: SLF001


@pytest.mark.asyncio
async def test_general_search_applies_chinese_locale_and_one_fallback() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(200, text="<html>安全验证 captcha</html>", request=request)
        return httpx.Response(
            200,
            text=(
                "<rss><channel><item><title>英雄联盟阵容消息</title>"
                "<link>https://club.example/roster</link>"
                "<description>俱乐部今日公布阵容</description>"
                "<pubDate>Tue, 21 Jul 2026 10:00:00 GMT</pubDate>"
                "</item></channel></rss>"
            ),
            request=request,
        )

    tool = PublicWebSearchTool(
        url_policy=_policy(),
        transport=httpx.MockTransport(handler),
    )
    result = await tool.search(
        "英雄联盟 阵容",
        language="zh-Hans",
        region="CN",
        window_start=NOW - timedelta(days=7),
        window_end=NOW,
    )

    assert result.provider == "bing_web_rss"
    assert result.provider_attempts == ("duckduckgo_html", "bing_web_rss")
    assert result.empty_or_challenge_attempts == 1
    assert requests[0].url.params["kl"] == "cn-zh"
    assert requests[0].url.params["df"] == "w"
    assert requests[1].url.params["mkt"] == "zh-CN"
    assert requests[1].url.params["setlang"] == "zh-Hans"
    assert requests[1].url.params["freshness"] == "2026-07-14..2026-07-21"
    assert result.items[0]["published_at"] == "Tue, 21 Jul 2026 10:00:00 GMT"


@pytest.mark.asyncio
async def test_general_search_keeps_ssrf_gate_and_falls_back_per_provider() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                "<rss><channel><item><title>英雄联盟 当前官方消息</title>"
                "<link>https://club.example/current</link>"
                "<description>俱乐部发布当前安排</description>"
                "</item></channel></rss>"
            ),
            request=request,
        )

    policy = UrlSafetyPolicy(
        resolver=Resolver(
            {
                "html.duckduckgo.com": ("127.0.0.1",),
                "www.bing.com": ("13.107.21.200",),
            }
        )
    )
    tool = PublicWebSearchTool(
        url_policy=policy,
        transport=httpx.MockTransport(handler),
    )

    result = await tool.search("英雄联盟 当前", language="zh-Hans", region="CN")

    assert result.provider == "bing_web_rss"
    assert result.provider_attempts == ("duckduckgo_html", "bing_web_rss")
    assert requests[0].url.host == "www.bing.com"
    assert "duckduckgo_html:url_policy:non_public_address" in result.limitations


@pytest.mark.asyncio
async def test_news_index_uses_news_surface_before_web_fallback() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                "<rss><channel><item><title>赛事官方新闻</title>"
                "<link>https://event.example/news</link>"
                "<description>官方发布赛事变更</description>"
                "</item></channel></rss>"
            ),
            request=request,
        )

    tool = PublicWebSearchTool(
        url_policy=_policy(),
        transport=httpx.MockTransport(handler),
    )
    result = await tool.search("赛事变更", index=WebSearchIndex.NEWS)

    assert result.provider == "bing_news_rss"
    assert len(requests) == 1
    assert requests[0].url.path == "/news/search"
    assert requests[0].url.params["format"] == "rss"


@pytest.mark.asyncio
async def test_facade_news_surface_uses_news_index(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                "<rss><channel><item><title>赛事官方新闻</title>"
                "<link>https://event.example/news</link>"
                "<description>官方发布赛事变更</description>"
                "</item></channel></rss>"
            ),
            request=request,
        )

    adapter = PublicWebChannelAdapter(
        PublicWebSearchTool(url_policy=_policy(), transport=httpx.MockTransport(handler)),
        now_factory=lambda: NOW,
    )
    world_tools = WorldTools(
        store=WorldStore(tmp_path / "world.sqlite3"),
        adapters={"public-web": adapter},
    )

    result = await world_tools.execute(
        "search_sources",
        {"adapter": "public-web", "query": "赛事变更", "surface_role": "news_index"},
        "facade-news",
    )

    assert result["ok"] is True
    assert len(requests) == 1
    assert requests[0].url.path == "/news/search"


@pytest.mark.asyncio
async def test_general_search_retries_when_first_nonempty_batch_breaks_chinese_contract() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(
                200,
                text=(
                    '<a class="result__a" href="https://jp.example/noise">'
                    "リーグ・オブ・レジェンド 古い動画</a>"
                    '<div class="result__snippet">日本語の再投稿コンテンツ</div>'
                ),
                request=request,
            )
        return httpx.Response(
            200,
            text=(
                "<rss><channel><item><title>英雄联盟 当前阵容官宣</title>"
                "<link>https://cn.example/current</link>"
                "<description>英雄联盟俱乐部发布当前阵容变更</description>"
                "</item></channel></rss>"
            ),
            request=request,
        )

    tool = PublicWebSearchTool(
        url_policy=_policy(),
        transport=httpx.MockTransport(handler),
    )

    result = await tool.search("英雄联盟 阵容", language="zh-Hans", region="CN")

    assert len(requests) == 2
    assert result.provider == "bing_web_rss"
    assert result.items[0]["title"] == "英雄联盟 当前阵容官宣"
    assert any(
        item.startswith("duckduckgo_html:candidate_contract_below_threshold") for item in result.limitations
    )


@pytest.mark.asyncio
async def test_general_search_drops_unanchored_tail_results_before_hydration() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "<rss><channel>"
                "<item><title>英雄联盟赛事今日赛果</title>"
                "<link>https://esports.example/lol</link>"
                "<description>League of Legends 当前赛事消息</description></item>"
                "<item><title>英雄：张艺谋执导的电影</title>"
                "<link>https://film.example/hero</link>"
                "<description>电影剧情与演员资料</description></item>"
                "</channel></rss>"
            ),
            request=request,
        )

    tool = PublicWebSearchTool(
        url_policy=_policy(),
        transport=httpx.MockTransport(handler),
    )

    result = await tool.search("英雄联盟 League of Legends", language="zh-Hans")

    assert [item["url"] for item in result.items] == ["https://esports.example/lol"]
    assert "bing_web_rss:dropped_unanchored_results:1" in result.limitations


@pytest.mark.asyncio
async def test_public_web_adapter_emits_rich_retrieval_and_contract_report() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "<rss><channel><item><title>英雄联盟俱乐部官方消息</title>"
                "<link>https://club.example/news?utm_source=index</link>"
                "<description>俱乐部公布全新阵容及后续安排</description>"
                "<pubDate>Tue, 21 Jul 2026 10:00:00 GMT</pubDate>"
                "</item></channel></rss>"
            ),
            request=request,
        )

    tool = PublicWebSearchTool(
        url_policy=_policy(),
        transport=httpx.MockTransport(handler),
    )
    adapter = PublicWebChannelAdapter(tool, now_factory=lambda: NOW)
    request = ScanRequest(
        id="scan-news",
        lane="ambient",
        query="英雄联盟 俱乐部 阵容",
        window_start=NOW - timedelta(days=30),
        window_end=NOW,
        language="zh-Hans",
        region="CN",
        timezone="Asia/Shanghai",
        capability_roles=[ChannelCapabilityRole.NEWS_INDEX],
    )

    batch = await adapter.discover(request)

    assert batch.retrieval is not None
    assert batch.retrieval.hits[0].capability_role is ChannelCapabilityRole.NEWS_INDEX
    assert batch.retrieval.hits[0].canonical_url == "https://club.example/news"
    assert batch.retrieval.contract_report.target_language_ratio == 1.0
    assert batch.retrieval.contract_report.time_filter_applied is True
    assert batch.occurrences[0].source_published_at == datetime(2026, 7, 21, 10, tzinfo=UTC)
    assert batch.occurrences[0].metadata["independence_status"] == "unknown"
    assert {descriptor.role for descriptor in adapter.capability_descriptors} >= {
        ChannelCapabilityRole.SEARCH_INDEX,
        ChannelCapabilityRole.NEWS_INDEX,
        ChannelCapabilityRole.WEB_BROWSER,
    }


@pytest.mark.asyncio
async def test_queryless_public_web_discovery_reports_unsupported() -> None:
    """A missing required query is distinct from a completed empty search."""
    adapter = PublicWebChannelAdapter(_page_tool("unused"), now_factory=lambda: NOW)

    batch = await adapter.discover(
        ScanRequest(id="queryless-web", lane="ambient", query="")
    )

    assert batch.outcome is AcquisitionOutcome.UNSUPPORTED
    assert batch.occurrences == []
    assert batch.partial is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("", AcquisitionOutcome.EMPTY),
        ("public_index_unavailable", AcquisitionOutcome.UNAVAILABLE),
    ],
)
async def test_public_web_discovery_distinguishes_empty_from_unavailable(
    error: str,
    expected: AcquisitionOutcome,
) -> None:
    """No indexed hit is not the same result as an unavailable index."""

    class EmptySearchTool:
        async def search(self, *args: object, **kwargs: object) -> WebSearchResult:
            del args, kwargs
            return WebSearchResult(items=[], query="topic", error=error)

    adapter = PublicWebChannelAdapter(EmptySearchTool(), now_factory=lambda: NOW)  # type: ignore[arg-type]
    batch = await adapter.discover(
        ScanRequest(id="empty-web", lane="ambient", query="topic")
    )

    assert batch.outcome is expected


def _page_tool(page_text: str) -> PublicWebSearchTool:
    """Build a tool whose fetch returns one bounded HTML page with ``page_text``."""

    async def handler(request: httpx.Request) -> httpx.Response:
        html = f"<html><head><title>测试页面标题</title></head><body><main>{page_text}</main></body></html>"
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": "text/html; charset=utf-8"},
            request=request,
        )

    return PublicWebSearchTool(
        url_policy=_policy(),
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_public_web_hydrate_text_full_carries_page_body() -> None:
    """TEXT depth with full=True returns the whole page text as body."""
    page_text = "转会期动态正文。"
    adapter = PublicWebChannelAdapter(_page_tool(page_text), now_factory=lambda: NOW)

    full_batch = await adapter.hydrate(
        HydrationRequest(
            id="open-full",
            source_ref="https://www.bing.com/page",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )
    short_batch = await adapter.hydrate(
        HydrationRequest(
            id="open-short",
            source_ref="https://www.bing.com/page",
            depth=HydrationDepth.TEXT,
        )
    )

    full_document = full_batch.observations[0]
    short_document = short_batch.observations[0]
    assert full_document.modality is ObservationModality.DOCUMENT_TEXT
    # the extractor folds the page title into the text; the body carries it all
    assert page_text in full_document.body
    assert full_document.excerpt == full_document.body
    assert short_document.body is None
    assert short_document.excerpt == full_document.body


@pytest.mark.asyncio
async def test_public_web_hydrate_metadata_depth_returns_metadata_only() -> None:
    """METADATA depth returns only a metadata observation (audit fix 5)."""
    page_text = "转会期动态正文。"
    adapter = PublicWebChannelAdapter(_page_tool(page_text), now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="open-metadata",
            source_ref="https://www.bing.com/page",
            depth=HydrationDepth.METADATA,
        )
    )

    assert len(batch.observations) == 1
    metadata = batch.observations[0]
    assert metadata.modality is ObservationModality.METADATA
    assert metadata.metadata["title"] == "测试页面标题"
    assert metadata.metadata["char_count"] == len(page_text)
    assert metadata.body is None
    assert "document_text_truncated" not in batch.limitations

    full_batch = await adapter.hydrate(
        HydrationRequest(
            id="open-metadata-full",
            source_ref="https://www.bing.com/page",
            depth=HydrationDepth.METADATA,
            arguments={"full": True},
        )
    )
    assert page_text in full_batch.observations[0].body


@pytest.mark.asyncio
async def test_public_web_hydrate_reactions_depth_is_typed_no_discussion() -> None:
    """REACTIONS depth states explicitly that public pages have no discussion."""
    adapter = PublicWebChannelAdapter(_page_tool("正文"), now_factory=lambda: NOW)

    batch = await adapter.hydrate(
        HydrationRequest(
            id="open-reactions",
            source_ref="https://www.bing.com/page",
            depth=HydrationDepth.REACTIONS,
        )
    )

    assert batch.partial is True
    assert batch.limitations == ["public_web_no_discussion"]
    assert all(item.modality is not ObservationModality.DOCUMENT_TEXT for item in batch.observations)
    assert all(item.modality is not ObservationModality.COMMENT for item in batch.observations)


@pytest.mark.asyncio
async def test_public_web_hydrate_fails_closed_on_url_policy_rejection() -> None:
    """Hydrate of a host the policy cannot resolve fails typed, never an ok card."""
    tool = PublicWebSearchTool(url_policy=_policy())
    adapter = PublicWebChannelAdapter(tool, now_factory=lambda: NOW)

    opened = await adapter.hydrate(
        HydrationRequest(
            id="open-rejected",
            source_ref="https://club.example/news",
            depth=HydrationDepth.TEXT,
        )
    )

    assert opened.observations == []
    assert opened.discovered_occurrences == []
    assert opened.partial is True
    assert opened.limitations == ["url rejected: resolution_empty"]
    assert opened.physical_call_count == 1


@pytest.mark.asyncio
async def test_public_web_hydrate_fails_closed_on_empty_document() -> None:
    """A fetched page with no extractable text fails typed, never an ok card."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="", request=request)

    tool = PublicWebSearchTool(url_policy=_policy(), transport=httpx.MockTransport(handler))
    adapter = PublicWebChannelAdapter(tool, now_factory=lambda: NOW)

    opened = await adapter.hydrate(
        HydrationRequest(
            id="open-empty",
            source_ref="https://www.bing.com/empty",
            depth=HydrationDepth.TEXT,
        )
    )

    assert opened.observations == []
    assert opened.partial is True
    assert opened.outcome is AcquisitionOutcome.UNAVAILABLE
    assert opened.limitations == ["public_web_extraction_failed"]
    assert opened.physical_call_count == 1


def _html_tool(page: str, *, status_code: int = 200, encoding: str = "utf-8") -> PublicWebSearchTool:
    """Return a mock public-web tool serving one static page at a public host."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=page.encode(encoding),
            headers={"content-type": f"text/html; charset={encoding}"},
            request=request,
        )

    return PublicWebSearchTool(url_policy=_policy(), transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_tencent_like_chrome_is_not_qualified_even_when_extractor_returns_text() -> None:
    """A static login/navigation shell must never become a full-material body."""
    page = """
    <html><head><title>Article details</title></head><body>
      <nav>Home Login Bind region Account center</nav>
      <section>Related recommendations</section>
      <footer>Copyright Tencent All rights reserved</footer>
    </body></html>
    """
    tool = _html_tool(page)
    document = await tool.fetch("https://www.bing.com/tencent-shell")
    adapter = PublicWebChannelAdapter(tool, now_factory=lambda: NOW)
    batch = await adapter.hydrate(
        HydrationRequest(
            id="shell",
            source_ref="https://www.bing.com/tencent-shell",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    assert document.outcome is WebDocumentOutcome.PAGE_SHELL
    assert document.text == ""
    assert batch.observations == []
    assert batch.outcome is AcquisitionOutcome.EMPTY
    assert batch.limitations == [
        "public_web_page_shell",
    ]
    assert "" not in batch.limitations


@pytest.mark.asyncio
async def test_article_with_login_navigation_remains_qualified() -> None:
    """A normal login link in chrome must not reject an otherwise readable article."""
    page = """
    <html><head><title>Match report</title></head><body>
      <nav>Home Login News</nav>
      <article><h1>Final match report</h1>
      <p>The team won the final after a careful early-game strategy.</p>
      <p>The coach said the roster will prepare for the next international event.</p>
      </article><footer>Copyright</footer></body></html>
    """
    document = await _html_tool(page).fetch("https://www.bing.com/login-navigation")

    assert document.outcome is WebDocumentOutcome.QUALIFIED_FULL
    assert "The team won the final" in document.text
    assert "Login" not in document.text
    assert document.limitations == ()


@pytest.mark.asyncio
async def test_static_article_removes_interface_and_recommendation_noise() -> None:
    """Static main-content extraction keeps article evidence instead of broad visible text."""
    page = """
    <html><body><header>Games News Esports Shop</header>
      <article><h1>League update</h1>
      <p>The league announced the schedule for the playoff stage today.</p>
      <p>Teams will play a best-of-five series under the updated tournament rules.</p>
      </article>
      <aside>Related recommendations: Buy skins, Watch more videos, Hot topics</aside>
      <footer>Copyright All rights reserved</footer>
    </body></html>
    """
    document = await _html_tool(page).fetch("https://www.bing.com/article-noise")

    assert document.outcome is WebDocumentOutcome.QUALIFIED_FULL
    assert "playoff stage" in document.text
    assert "Buy skins" not in document.text
    assert "Related recommendations" not in document.text


@pytest.mark.asyncio
async def test_div_based_article_with_footer_chrome_is_not_misclassified_as_shell() -> None:
    """A mature extractor may recover正文 without semantic article/main tags."""
    page = """
    <html><body><header>Games News Esports Shop</header>
      <div class="story"><h1>League schedule update</h1>
      <p>The organizer published the complete playoff schedule today.</p>
      <p>All qualified teams will play under the announced tournament rules.</p>
      </div>
      <aside>Related recommendations</aside>
      <footer>Copyright All rights reserved</footer>
    </body></html>
    """
    document = await _html_tool(page).fetch("https://www.bing.com/div-article")

    assert document.outcome is WebDocumentOutcome.QUALIFIED_FULL
    assert "complete playoff schedule" in document.text


@pytest.mark.asyncio
@pytest.mark.parametrize("quoted_control", ["verify you are human", "login", "read more"])
async def test_article_quoting_access_control_language_remains_qualified(
    quoted_control: str,
) -> None:
    """An article may discuss UI or anti-bot language without becoming that control page."""
    page = f"""
    <html><body><article><h1>Platform access report</h1>
      <p>The report quotes the interface text {quoted_control!r} while explaining a failed launch.</p>
      <p>The rest of the article documents what users observed and how the platform responded.</p>
    </article></body></html>
    """
    document = await _html_tool(page).fetch("https://www.bing.com/control-language-report")

    assert document.outcome is WebDocumentOutcome.QUALIFIED_FULL
    assert "Platform access report" in document.text


@pytest.mark.asyncio
async def test_utility_only_extraction_is_not_qualified() -> None:
    """A non-empty footer/privacy extraction is still page chrome, not durable evidence."""
    page = """
    <html><body><div>Privacy policy</div><div>Terms of use</div>
      <div>Account center</div><div>Back to homepage</div></body></html>
    """
    document = await _html_tool(page).fetch("https://www.bing.com/utility-only")

    assert document.outcome is WebDocumentOutcome.PAGE_SHELL
    assert document.text == ""


@pytest.mark.asyncio
async def test_short_structured_announcement_is_qualified_without_length_gate() -> None:
    """An authentic short articleBody remains durable material regardless of character count."""
    page = """
    <html><head><script type="application/ld+json">
    {
      "@context":"https://schema.org",
      "@type":"NewsArticle",
      "articleBody":"Official notice: the match starts tonight."
    }
    </script></head><body><nav>Home Login</nav></body></html>
    """
    document = await _html_tool(page).fetch("https://www.bing.com/short-notice")

    assert document.outcome is WebDocumentOutcome.QUALIFIED_FULL
    assert document.text == "Official notice: the match starts tonight."
    assert len(document.text) < 100


@pytest.mark.asyncio
async def test_gb18030_article_decodes_and_qualifies() -> None:
    """A GB18030 static article is extracted without a replacement character."""
    page = (
        "<html><body><article><p>英雄联盟赛事将在今晚举行，官方公布了完整赛程。</p></article></body></html>"
    )
    document = await _html_tool(page, encoding="gb18030").fetch("https://www.bing.com/gb18030")

    assert document.outcome is WebDocumentOutcome.QUALIFIED_FULL
    assert "英雄联盟赛事" in document.text
    assert "\ufffd" not in document.text


@pytest.mark.asyncio
async def test_extractor_failure_becomes_typed_non_material_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A static extractor exception cannot fall back to noisy visible page text."""

    def fail_extract(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("scripted extractor failure")

    monkeypatch.setattr(trafilatura, "extract", fail_extract)
    tool = _html_tool("<html><body><article><p>Readable looking text</p></article></body></html>")
    adapter = PublicWebChannelAdapter(tool, now_factory=lambda: NOW)
    batch = await adapter.hydrate(
        HydrationRequest(
            id="extractor-failure",
            source_ref="https://www.bing.com/extractor-failure",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    assert batch.observations == []
    assert batch.outcome is AcquisitionOutcome.UNAVAILABLE
    assert batch.limitations == ["public_web_extraction_failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("page", "status_code", "outcome", "targeted", "acquisition_outcome"),
    [
        (
            "<html><body>Please login to read the full article.</body></html>",
            200,
            WebDocumentOutcome.LOGIN_OR_PAYWALL,
            "public_web_no_auth_bypass",
            AcquisitionOutcome.UNSUPPORTED,
        ),
        (
            "<html><body>Captcha: verify you are human.</body></html>",
            200,
            WebDocumentOutcome.CHALLENGE_OR_BLOCKED,
            "public_web_access_challenge_active",
            AcquisitionOutcome.UNAVAILABLE,
        ),
        (
            "<html><body>Captcha: verify you are human.</body></html>",
            403,
            WebDocumentOutcome.CHALLENGE_OR_BLOCKED,
            "public_web_access_challenge_active",
            AcquisitionOutcome.UNAVAILABLE,
        ),
        (
            "<html><body>Captcha: verify you are human.</body></html>",
            429,
            WebDocumentOutcome.CHALLENGE_OR_BLOCKED,
            "public_web_access_challenge_active",
            AcquisitionOutcome.UNAVAILABLE,
        ),
        (
            '<html><body><div id="app"></div><script src="/app.js"></script></body></html>',
            200,
            WebDocumentOutcome.RENDERING_REQUIRED,
            "public_web_static_rendering_unavailable",
            AcquisitionOutcome.UNSUPPORTED,
        ),
        (
            "<html><body>Read more to continue reading.</body></html>",
            200,
            WebDocumentOutcome.PARTIAL_CONTENT,
            "public_web_only_partial_content_available",
            AcquisitionOutcome.PARTIAL,
        ),
    ],
)
async def test_nonqualified_pages_return_typed_refusals_and_no_observation(
    page: str,
    status_code: int,
    outcome: WebDocumentOutcome,
    targeted: str,
    acquisition_outcome: AcquisitionOutcome,
) -> None:
    """Blocked, partial, and rendered pages expose state without directing a next step."""
    tool = _html_tool(page, status_code=status_code)
    adapter = PublicWebChannelAdapter(tool, now_factory=lambda: NOW)
    batch = await adapter.hydrate(
        HydrationRequest(
            id=f"refusal-{outcome.value}",
            source_ref="https://www.bing.com/refusal",
            depth=HydrationDepth.TEXT,
            arguments={"full": True},
        )
    )

    assert batch.observations == []
    assert batch.partial is True
    assert batch.outcome is acquisition_outcome
    assert f"public_web_{outcome.value}" in batch.limitations
    assert targeted in batch.limitations
    assert not any(
        token in limitation
        for limitation in batch.limitations
        for token in ("pivot", "retry", "do_not_retry")
    )
    assert "" not in batch.limitations


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("control", "expected"),
    [
        ("Please login to read the full article.", WebDocumentOutcome.LOGIN_OR_PAYWALL),
        ("Read more to continue reading.", WebDocumentOutcome.PARTIAL_CONTENT),
        ("Captcha: verify you are human.", WebDocumentOutcome.CHALLENGE_OR_BLOCKED),
    ],
)
async def test_page_chrome_does_not_hide_a_content_area_refusal(
    control: str,
    expected: WebDocumentOutcome,
) -> None:
    """Title/navigation chrome cannot turn a control-only body into qualified material."""
    page = f"""
    <html><head><title>Account page</title></head><body>
      <nav>Home News Account</nav><div class="content">{control}</div>
      <footer>Copyright All rights reserved</footer>
    </body></html>
    """
    document = await _html_tool(page).fetch("https://www.bing.com/chrome-and-refusal")

    assert document.outcome is expected
    assert document.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("control", "expected"),
    [
        ("Please login to read the full article.", WebDocumentOutcome.LOGIN_OR_PAYWALL),
        ("Read more to continue reading.", WebDocumentOutcome.PARTIAL_CONTENT),
    ],
)
async def test_json_ld_body_does_not_override_page_level_access_control(
    control: str,
    expected: WebDocumentOutcome,
) -> None:
    """A stale or preview JSON-LD body cannot bypass an explicit access-control page."""
    page = f"""
    <html><head><script type="application/ld+json">
    {{"@type":"NewsArticle","articleBody":"A cached summary that is not visible on this page."}}
    </script></head><body>{control}</body></html>
    """
    document = await _html_tool(page).fetch("https://www.bing.com/blocked-json-ld")

    assert document.outcome is expected
    assert document.text == ""
