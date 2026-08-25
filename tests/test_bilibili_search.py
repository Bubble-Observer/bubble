"""Offline tests for the Bilibili platform adapter."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from leave_information_bubble.runtime.errors import AgentError, ErrorCode
from leave_information_bubble.tools import bilibili_search
from leave_information_bubble.tools.bilibili_search import (
    MAX_SUBTITLE_ATTEMPTS,
    MIN_TITLE_HAN_BIGRAMS,
    SUBTITLE_ABSENT_LIMITATION,
    SUBTITLE_IDENTITY_MISMATCH_LIMITATION,
    SUBTITLE_REPEAT_THRESHOLD,
    SUBTITLE_RETRY_INTERVAL_S,
    SUBTITLE_UNRELIABLE_LIMITATION,
    TITLE_BIGRAM_THRESHOLD,
    BilibiliAudiencePrecision,
    BilibiliSearchTool,
    _subtitle_content_hash,
    _title_bigram_hit_rate,
)
from leave_information_bubble.tools.subtitle_ledger import SubtitleLedger


class FakeResponse:
    """Minimal httpx response double."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        """Represent a successful HTTP response."""

    def json(self) -> dict[str, Any]:
        """Return the configured JSON payload."""
        return self._payload

    @property
    def content(self) -> bytes:
        """Return optional raw bytes used by the danmaku endpoint."""
        return bytes(self._payload.get("_content", b""))

    async def __aenter__(self) -> FakeResponse:
        """Allow the response to stand in for an httpx stream context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Close a fake stream context without external resources."""
        del args

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        """Yield configured audio bytes as deterministic bounded chunks."""
        content = self.content
        midpoint = max(1, len(content) // 2)
        for offset in range(0, len(content), midpoint):
            yield content[offset : offset + midpoint]


class FakeClient:
    """Async client double with an injectable get operation."""

    get_handler: Callable[..., Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        del args

    async def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
        result = self.get_handler(*args, **kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    def stream(self, method: str, *args: Any, **kwargs: Any) -> FakeResponse:
        """Return a fake streaming response for bounded audio tests."""
        result = self.get_handler(method, *args, **kwargs)
        if isinstance(result, Exception):
            raise result
        return result


def _search_payload() -> dict[str, Any]:
    return {
        "code": 0,
        "data": {
            "result": [
                {
                    "result_type": "video",
                    "data": [
                        {
                            "bvid": "BV001",
                            "title": '<em class="keyword">LPL</em> BLG复盘',
                            "author": "作者",
                            "pubdate": 1784419200,
                            "play": 1000,
                            "video_review": 20,
                            "like": 100,
                            "description": "不同视角的比赛复盘",
                        }
                    ],
                },
                {
                    "result_type": "article",
                    "data": [{"id": "cv1", "title": "should be ignored"}],
                },
            ]
        },
    }


def test_normalization_uses_type_specific_publication_time() -> None:
    payload = {
        "data": {
            "result": [
                {
                    "result_type": "media_bangumi",
                    "data": [
                        {
                            "season_id": "68619",
                            "title": "Archived second season",
                            "pubtime": "2024-11-09T00:00:00+08:00",
                        }
                    ],
                }
            ]
        }
    }

    item = BilibiliSearchTool._normalize_results(payload)[0]

    assert item["published_at"] == "2024-11-08T16:00:00+00:00"


@pytest.mark.asyncio
async def test_search_normalizes_video_and_preserves_other_result_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse(_search_payload())
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool()

    result = await tool.search("LPL", page_size=10)

    assert result.error == ""
    assert result.query == "LPL"
    assert result.total_results == 2
    assert result.items[0]["id"] == "bilibili-BV001"
    assert result.items[0]["title"] == "LPL BLG复盘"
    assert result.items[0]["text_snippet"] == "不同视角的比赛复盘"
    assert result.items[0]["published_at"].endswith("+00:00")
    assert result.items[0]["engagement"]["danmaku"] == 20
    assert result.items[1]["content_type"] == "bilibili_article"
    assert result.items[1]["id"] == "bilibili-article-cv1"
    assert tool.call_count == 1
    assert tool.call_history[0]["keyword"] == "LPL"


@pytest.mark.asyncio
async def test_search_returns_structured_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse({"code": -412, "message": "blocked"})
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    result = await BilibiliSearchTool().search("LPL")
    assert result.items == []
    assert result.error == "blocked"


def test_normalization_uses_item_type_for_mixed_live_search_cards() -> None:
    items = BilibiliSearchTool._normalize_results(
        {
            "data": {
                "result": [
                    {
                        "result_type": "video",
                        "data": [
                            {
                                "type": "live_room",
                                "roomid": 88,
                                "title": "live",
                                "online": 123,
                            }
                        ],
                    }
                ]
            }
        }
    )

    assert items[0]["content_type"] == "bilibili_live_room"
    assert items[0]["id"] == "bilibili-live_room-88"
    assert items[0]["canonical_url"] == "https://live.bilibili.com/88"


@pytest.mark.asyncio
async def test_search_maps_timeout_to_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.get_handler = lambda *args, **kwargs: httpx.TimeoutException("slow")
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    with pytest.raises(AgentError) as exc_info:
        await BilibiliSearchTool().search("LPL")
    assert exc_info.value.code is ErrorCode.TOOL_TRANSIENT


@pytest.mark.asyncio
async def test_video_info_includes_aid_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    request_kwargs: dict[str, Any] = {}

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del args
        request_kwargs.update(kwargs)
        return FakeResponse(
            {
                "code": 0,
                "data": {
                    "aid": 123,
                    "cid": 456,
                    "title": "BLG复盘",
                    "owner": {"name": "作者"},
                    "desc": "详情",
                    "pubdate": 1784419200,
                    "subtitle": {
                        "list": [
                            {
                                "id": 207911,
                                "lan": "ai-zh",
                                "lan_doc": "中文（自动生成）",
                            }
                        ]
                    },
                    "stat": {
                        "view": 1000,
                        "danmaku": 20,
                        "like": 100,
                        "coin": 50,
                        "favorite": 80,
                    },
                },
            }
        )

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    detail = await BilibiliSearchTool(sessdata="session-token").get_video_info("BV001")
    assert detail["aid"] == 123
    assert detail["cid"] == 456
    assert detail["author"] == "作者"
    assert detail["stats"]["favorite"] == 80
    assert detail["stats"]["reply"] is None
    assert detail["subtitle_tracks"] == [
        {"id": "207911", "lan": "ai-zh", "lan_doc": "中文（自动生成）"}
    ]
    assert request_kwargs["cookies"] == {"SESSDATA": "session-token"}


@pytest.mark.asyncio
async def test_video_tags_preserve_platform_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse(
        {
            "code": 0,
            "data": [
                {
                    "tag_id": 7,
                    "tag_name": "strategy",
                    "tag_type": "topic",
                    "music_id": "",
                }
            ],
        }
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    tags = await BilibiliSearchTool().get_video_tags("BV001")

    assert tags == [
        {"tag_id": 7, "name": "strategy", "type": "topic", "music_id": ""}
    ]


@pytest.mark.asyncio
async def test_concurrent_audience_preserves_exact_and_rounded_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse(
        {
            "code": 0,
            "data": {
                "total": "6000+",
                "count": "1497",
                "show_switch": {"total": True, "count": True},
            },
        }
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    result = await BilibiliSearchTool().get_concurrent_audience("BV001", 456)

    assert result.total.precision is BilibiliAudiencePrecision.ROUNDED
    assert result.total.value is None
    assert result.total.lower_bound == 6000
    assert result.total.display == "6000+"
    assert result.count.precision is BilibiliAudiencePrecision.EXACT
    assert result.count.value == 1497
    assert result.count.lower_bound is None
    assert result.error_kind == ""


@pytest.mark.asyncio
async def test_concurrent_audience_honors_hidden_show_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse(
        {
            "code": 0,
            "data": {
                "total": "99",
                "count": "12",
                "show_switch": {"total": False, "count": False},
            },
        }
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    result = await BilibiliSearchTool().get_concurrent_audience("BV001", 456)

    assert result.total.precision is BilibiliAudiencePrecision.HIDDEN
    assert result.total.value is None
    assert result.total.display == ""
    assert result.count.precision is BilibiliAudiencePrecision.HIDDEN
    assert "total_hidden_by_show_switch" in result.limitations
    assert "count_hidden_by_show_switch" in result.limitations


@pytest.mark.asyncio
async def test_concurrent_audience_isolates_412_and_schema_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", bilibili_search.ONLINE_TOTAL_URL)
    response = httpx.Response(412, request=request)
    FakeClient.get_handler = lambda *args, **kwargs: httpx.HTTPStatusError(
        "blocked",
        request=request,
        response=response,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    blocked = await BilibiliSearchTool().get_concurrent_audience("BV001", 456)

    assert blocked.error_kind == "blocked"
    assert blocked.error_code == 412
    assert blocked.total.precision is BilibiliAudiencePrecision.UNAVAILABLE

    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse(
        {
            "code": 0,
            "data": {
                "total": "10",
                "count": "3",
                "show_switch": "changed",
            },
        }
    )

    changed = await BilibiliSearchTool().get_concurrent_audience("BV001", 456)

    assert changed.error_kind == "schema_changed"
    assert changed.count.value is None


@pytest.mark.asyncio
async def test_concurrent_audience_exposes_rate_limit_and_missingness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse(
        {"code": -509, "message": "too frequent"}
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    limited = await BilibiliSearchTool().get_concurrent_audience("BV001", 456)

    assert limited.error_kind == "rate_limited"
    assert limited.error_code == -509
    assert limited.total.value is None

    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse(
        {
            "code": 0,
            "data": {
                "total": "",
                "count": "0",
            },
        }
    )

    missing = await BilibiliSearchTool().get_concurrent_audience("BV001", 456)

    assert missing.total.precision is BilibiliAudiencePrecision.UNAVAILABLE
    assert missing.total.value is None
    assert missing.count.precision is BilibiliAudiencePrecision.EXACT
    assert missing.count.value == 0
    assert missing.count.show_switch is None
    assert "show_switch_missing" in missing.limitations


@pytest.mark.asyncio
async def test_comments_resolve_bvid_and_record_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse(
        {
            "code": 0,
            "data": {
                "replies": [
                    {
                        "member": {"uname": "用户"},
                        "content": {"message": "不能只看结果"},
                        "like": 99,
                        "rcount": 4,
                        "ctime": 1,
                        "floor": 1,
                    }
                ]
            },
        }
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool()

    async def fake_info(bvid: str) -> dict[str, Any]:
        assert bvid == "BV001"
        return {"aid": 123}

    monkeypatch.setattr(tool, "get_video_info", fake_info)
    comments = await tool.get_comments("BV001", mode=3)

    assert comments == [
        {
            "user": "用户",
            "content": "不能只看结果",
            "like_count": 99,
            "reply_count": 4,
            "ctime": 1,
            "floor": 1,
            "sampling": "hot",
            "replies": [],
        }
    ]


@pytest.mark.asyncio
async def test_comments_degrade_when_bvid_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = BilibiliSearchTool()

    async def missing_info(bvid: str) -> dict[str, Any]:
        del bvid
        return {}

    monkeypatch.setattr(tool, "get_video_info", missing_info)
    assert await tool.get_comments("BV001") == []


@pytest.mark.asyncio
async def test_comment_page_discloses_total_sort_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: dict[str, Any] = {}

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del args
        requested.update(kwargs["params"])
        return FakeResponse(
            {
                "code": 0,
                "data": {
                    "cursor": {"all_count": 900, "is_end": False},
                    "replies": [
                        {
                            "member": {"uname": "viewer"},
                            "content": {"message": "new response"},
                            "like": 4,
                            "rcount": 8,
                            "ctime": 2,
                            "floor": 3,
                            "replies": [
                                {
                                    "member": {"uname": "child"},
                                    "content": {"message": "nested"},
                                    "like": 1,
                                    "ctime": 3,
                                }
                            ],
                        }
                    ],
                },
            }
        )

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    page = await BilibiliSearchTool().get_comment_page(
        123,
        limit=500,
        mode=2,
        page=1,
        reply_limit=0,
    )

    assert page.total_comments == 900
    assert page.sort == "newest"
    assert page.requested_limit == 50
    assert page.returned_count == 1
    assert page.comments[0]["replies"] == []
    assert page.has_more is True
    assert requested["ps"] == 50
    assert requested["pn"] == 1


@pytest.mark.asyncio
async def test_platform_subtitles_preserve_timestamps_and_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        if url == bilibili_search.PLAYER_INFO_URL:
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "zh-CN",
                                    "subtitle_url": "//subtitle.example/1.json",
                                }
                            ]
                        }
                    },
                }
            )
        return FakeResponse(
            {
                "body": [
                    {"from": 1.2, "to": 3.4, "content": "正文表达的是比较关系"},
                ]
            }
        )

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool(min_interval=0, subtitle_retry_interval=0)

    async def fake_info(bvid: str) -> dict[str, Any]:
        assert bvid == "BV001"
        return {"cid": 456, "title": "正文表达的是比较关系"}

    monkeypatch.setattr(tool, "get_video_info", fake_info)
    segments = await tool.get_subtitles("bilibili-BV001")

    # Title-relevant content recurs → confirmed on the second attempt; the
    # segment shape (timestamps, origin) is unchanged.
    assert segments == [
        {
            "start": 1.2,
            "end": 3.4,
            "content": "正文表达的是比较关系",
            "language": "zh-CN",
            "acquisition_method": "platform_subtitle",
            "confidence": 1.0,
            "cid": 456,
            "reliability": "confirmed",
            "attempts": 2,
        }
    ]


@pytest.mark.asyncio
async def test_subtitles_reject_player_track_not_owned_by_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched_urls: list[str] = []

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        fetched_urls.append(url)
        return FakeResponse(
            {
                "code": 0,
                "data": {
                    "subtitle": {
                        "subtitles": [
                            {
                                "id": 999,
                                "lan": "ai-zh",
                                "subtitle_url": "//subtitle.example/foreign.json",
                            }
                        ]
                    }
                },
            }
        )

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool(min_interval=0, subtitle_retry_interval=0)

    segments = await tool.get_subtitles(
        "BV001",
        video_info={
            "cid": 456,
            "title": "本视频标题内容",
            "subtitle_tracks": [{"id": "123", "lan": "ai-zh"}],
        },
    )

    assert segments == []
    assert tool.last_subtitle_limitation == SUBTITLE_IDENTITY_MISMATCH_LIMITATION
    assert "https://subtitle.example/foreign.json" not in fetched_urls


@pytest.mark.asyncio
async def test_canonical_empty_track_list_skips_player_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del args, kwargs
        nonlocal called
        called = True
        return FakeResponse({})

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool(min_interval=0, subtitle_retry_interval=0)

    segments = await tool.get_subtitles(
        "BV001",
        video_info={"cid": 456, "title": "本视频标题内容", "subtitle_tracks": []},
    )

    assert segments == []
    assert tool.last_subtitle_limitation == SUBTITLE_ABSENT_LIMITATION
    assert called is False


@pytest.mark.asyncio
async def test_subtitles_accept_matching_canonical_track_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        if url == bilibili_search.PLAYER_INFO_URL:
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "id": 123,
                                    "lan": "ai-zh",
                                    "subtitle_url": "//subtitle.example/canonical.json",
                                }
                            ]
                        }
                    },
                }
            )
        return FakeResponse(
            {"body": [{"from": 1.0, "to": 2.0, "content": "本视频标题内容"}]}
        )

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool(min_interval=0, subtitle_retry_interval=0)

    segments = await tool.get_subtitles(
        "BV001",
        video_info={
            "cid": 456,
            "title": "本视频标题内容",
            "subtitle_tracks": [{"id": "123", "lan": "ai-zh"}],
        },
    )

    assert segments[0]["content"] == "本视频标题内容"
    assert segments[0]["reliability"] == "confirmed"


@pytest.mark.asyncio
async def test_subtitles_prefer_manual_zh_cn_track_over_ai_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track selection is deterministic: manual zh-CN beats ai-zh (audit fix 2)."""
    fetched_urls: list[str] = []

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        fetched_urls.append(url)
        if url == bilibili_search.PLAYER_INFO_URL:
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "ai-zh",
                                    "lan_doc": "中文（自动生成的）",
                                    "subtitle_url": "//subtitle.example/ai.json",
                                },
                                {
                                    "lan": "zh-CN",
                                    "lan_doc": "中文（简体）",
                                    "subtitle_url": "//subtitle.example/manual.json",
                                },
                            ]
                        }
                    },
                }
            )
        if url == "https://subtitle.example/manual.json":
            return FakeResponse({"body": [{"from": 1.0, "to": 2.0, "content": "手动字幕"}]})
        return FakeResponse({"body": [{"from": 1.0, "to": 2.0, "content": "AI字幕"}]})

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool(min_interval=0, subtitle_retry_interval=0)

    async def fake_info(bvid: str) -> dict[str, Any]:
        assert bvid == "BV001"
        return {"cid": 456, "title": "手动字幕"}

    monkeypatch.setattr(tool, "get_video_info", fake_info)
    segments = await tool.get_subtitles("bilibili-BV001")

    assert "https://subtitle.example/manual.json" in fetched_urls
    assert len(segments) == 1
    assert segments[0]["content"] == "手动字幕"
    assert segments[0]["language"] == "zh-CN"
    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == 2


@pytest.mark.asyncio
async def test_subtitles_retry_after_empty_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An intermittent empty subtitle fetch is retried and can still confirm."""
    calls: list[str] = []

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        calls.append(url)
        if url == bilibili_search.PLAYER_INFO_URL:
            if len(calls) == 1:
                return FakeResponse({"code": 0, "data": {"subtitle": {"subtitles": []}}})
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "zh-CN",
                                    "lan_doc": "中文（简体）",
                                    "subtitle_url": "//subtitle.example/manual.json",
                                }
                            ]
                        }
                    },
                }
            )
        return FakeResponse({"body": [{"from": 1.0, "to": 2.0, "content": "正文"}]})

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool(min_interval=0, subtitle_retry_interval=0)

    async def fake_info(bvid: str) -> dict[str, Any]:
        assert bvid == "BV001"
        return {"cid": 456, "title": "正文内容"}

    monkeypatch.setattr(tool, "get_video_info", fake_info)
    segments = await tool.get_subtitles("BV001")

    assert len(segments) == 1
    assert segments[0]["content"] == "正文"
    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == 3
    # the empty first attempt was retried; confirmation stopped the loop
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == 3


@pytest.mark.asyncio
async def test_subtitles_retry_when_manual_track_absent_on_first_try(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty first response without the manual track gets one retry.

    The platform's track list flaps between calls and sometimes omits the
    manual zh-CN track (leaving only ai-zh); the retry must fire on that
    non-empty selection too, because a second attempt can observe the
    manual track (review finding on audit fix 2).
    """
    calls: list[str] = []

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        calls.append(url)
        if url == bilibili_search.PLAYER_INFO_URL:
            if len(calls) == 1:
                return FakeResponse(
                    {
                        "code": 0,
                        "data": {
                            "subtitle": {
                                "subtitles": [
                                    {
                                        "lan": "ai-zh",
                                        "lan_doc": "中文（自动生成的）",
                                        "subtitle_url": "//subtitle.example/ai.json",
                                    }
                                ]
                            }
                        },
                    }
                )
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "zh-CN",
                                    "lan_doc": "中文（简体）",
                                    "subtitle_url": "//subtitle.example/manual.json",
                                }
                            ]
                        }
                    },
                }
            )
        if url == "https://subtitle.example/ai.json":
            return FakeResponse({"body": [{"from": 1.0, "to": 2.0, "content": "AI字幕"}]})
        return FakeResponse({"body": [{"from": 1.0, "to": 2.0, "content": "手动字幕"}]})

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool(min_interval=0, subtitle_retry_interval=0)

    async def fake_info(bvid: str) -> dict[str, Any]:
        assert bvid == "BV001"
        return {"cid": 456, "title": "手动字幕"}

    monkeypatch.setattr(tool, "get_video_info", fake_info)
    segments = await tool.get_subtitles("BV001")

    assert len(segments) == 1
    assert segments[0]["content"] == "手动字幕"
    assert segments[0]["language"] == "zh-CN"
    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == 3
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == 3


@pytest.mark.asyncio
async def test_subtitles_keep_ai_fallback_when_manual_never_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the manual track never appears, the AI track alone can still confirm.

    Retrying must not turn a platform-side missing-manual response into an
    empty result when the AI track is the only one the platform serves and it
    matches the title.
    """
    calls: list[str] = []

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        calls.append(url)
        if url == bilibili_search.PLAYER_INFO_URL:
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "ai-zh",
                                    "lan_doc": "中文（自动生成的）",
                                    "subtitle_url": "//subtitle.example/ai.json",
                                }
                            ]
                        }
                    },
                }
            )
        return FakeResponse({"body": [{"from": 1.0, "to": 2.0, "content": "AI字幕"}]})

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool(min_interval=0, subtitle_retry_interval=0)

    async def fake_info(bvid: str) -> dict[str, Any]:
        assert bvid == "BV001"
        return {"cid": 456, "title": "AI字幕解说"}

    monkeypatch.setattr(tool, "get_video_info", fake_info)
    segments = await tool.get_subtitles("BV001")

    assert len(segments) == 1
    assert segments[0]["content"] == "AI字幕"
    assert segments[0]["language"] == "ai-zh"
    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == 2
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == 2


# ---------------------------------------------------------------------------
# Subtitle reliability judgment (Task 2: retry + double-signal + contract)
# ---------------------------------------------------------------------------

_TITLE = "英雄联盟全球总决赛"  # 9 Han chars -> 8 title bigrams
_GARBAGE = "大家好我是解说员"  # shares 0 bigrams with _TITLE
_CORRECT = "全球总决赛双方选手登场"  # shares 4/8 bigrams with _TITLE
_REAL = "欢迎来到总决赛解说现场"  # shares 2/8 bigrams with _TITLE
#: 21 distinct Han chars -> 20 distinct bigrams (no adjacent repeats).
BOUNDARY_TITLE = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌"


def _one_segment(content: str) -> list[dict[str, Any]]:
    """Return a one-segment subtitle body payload for a fetch attempt."""
    return [{"from": 1.0, "to": 2.0, "content": content}]


def _sequence_subtitle_handler(
    attempts: list[list[dict[str, Any]] | Exception],
    calls: list[str] | None = None,
) -> Callable[..., FakeResponse]:
    """Serve one subtitle attempt per PLAYER_INFO_URL hit, then repeat the last.

    Attempt ``i`` exposes an ai-zh track at ``//subtitle.example/{i}.json``
    whose body is ``attempts[i]`` (an ``Exception`` entry raises on the player
    fetch — a failed attempt). Hits beyond the sequence repeat the last entry,
    mirroring a platform that keeps serving the same (fallback) track.
    """

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        if calls is not None:
            calls.append(url)
        if url == bilibili_search.PLAYER_INFO_URL:
            player_hits = [u for u in (calls if calls is not None else []) if u == url]
            index = min(len(player_hits) - 1, len(attempts) - 1)
            entry = attempts[index]
            if isinstance(entry, Exception):
                raise entry
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "ai-zh",
                                    "lan_doc": "中文（自动生成的）",
                                    "subtitle_url": f"//subtitle.example/{index}.json",
                                }
                            ]
                        }
                    },
                }
            )
        index = int(url.rsplit("/", 1)[-1].split(".json")[0])
        entry = attempts[index]
        if isinstance(entry, Exception):
            return FakeResponse({"body": []})  # player fetch raised first
        return FakeResponse({"body": entry})

    return handler


def _fast_tool(tmp_path: Any) -> BilibiliSearchTool:
    """Construct a tool with no rate-limit/retry sleeps and a temp ledger."""
    return BilibiliSearchTool(
        min_interval=0,
        subtitle_retry_interval=0,
        subtitle_ledger_path=tmp_path / "subtitle-ledger.json",
    )


def test_subtitle_reliability_constants_match_plan() -> None:
    """Plan-bound threshold constants are named and keep their agreed values."""
    assert MAX_SUBTITLE_ATTEMPTS == 5
    assert SUBTITLE_RETRY_INTERVAL_S == 2.5
    assert TITLE_BIGRAM_THRESHOLD == 0.20
    assert SUBTITLE_REPEAT_THRESHOLD == 2
    assert MIN_TITLE_HAN_BIGRAMS == 3
    assert SUBTITLE_UNRELIABLE_LIMITATION == "platform_subtitle_unreliable_after_5_attempts"
    assert SUBTITLE_ABSENT_LIMITATION == "platform_subtitle_absent_or_unavailable_without_authentication"


@pytest.mark.asyncio
async def test_subtitles_confirm_on_repeat_and_title_relevance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Garbage pool content twice, then the real subtitle twice → confirmed.

    The double signal (same-video hash repeated >= 2 AND title-bigram hit rate
    >= 0.15) is the only path to ``confirmed``; the real content wins on the
    second occurrence and no further fetch happens.
    """
    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [
            _one_segment(_GARBAGE),
            _one_segment(_GARBAGE),
            _one_segment(_CORRECT),
            _one_segment(_CORRECT),
        ],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert len(segments) == 1
    assert segments[0]["content"] == _CORRECT
    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == 4
    assert tool.last_subtitle_limitation is None
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == 4
    # every successful fetch was recorded in the ledger
    assert (tmp_path / "subtitle-ledger.json").exists()


@pytest.mark.asyncio
async def test_subtitles_repeated_but_unrelated_never_returned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Stable fallback mode: unrelated garbage recurring is never returned.

    The platform can serve the *same* fallback-pool content on every attempt,
    so within-call repetition without title relevance is no signal at all —
    it is discarded like a blacklisted hash, and the call ends empty with the
    exhaustion limitation (b33 live finding).
    """
    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [_one_segment(_GARBAGE), _one_segment(_GARBAGE)],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert segments == []
    assert tool.last_subtitle_limitation == SUBTITLE_UNRELIABLE_LIMITATION
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == MAX_SUBTITLE_ATTEMPTS


@pytest.mark.asyncio
async def test_subtitles_best_effort_when_related_but_not_repeated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Only the relevance signal (real content once) → best_effort fallback."""
    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [_one_segment(_CORRECT), _one_segment(_GARBAGE)],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert len(segments) == 1
    assert segments[0]["content"] == _CORRECT
    assert segments[0]["reliability"] == "best_effort"
    assert segments[0]["attempts"] == MAX_SUBTITLE_ATTEMPTS


@pytest.mark.asyncio
async def test_subtitles_best_effort_prefers_relevant_content_surfaced_later(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Pool garbage first, real content later: best_effort must pick the real one.

    The platform's track list flaps between calls, so fallback-pool content
    can surface before the video's own subtitle. Among equally-ranked (ai-zh)
    candidates the higher title-bigram hit rate must win over the first-seen
    one (review I-1). The real content occurs exactly once so the call stays
    in the best_effort path instead of confirming.
    """
    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [_one_segment(_GARBAGE), _one_segment(_CORRECT), _one_segment(_GARBAGE)],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert len(segments) == 1
    assert segments[0]["content"] == _CORRECT
    assert segments[0]["reliability"] == "best_effort"
    assert segments[0]["attempts"] == MAX_SUBTITLE_ATTEMPTS


@pytest.mark.asyncio
async def test_subtitles_blacklisted_hash_never_returned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """A blacklisted hash is discarded even when repeated and title-relevant.

    The pool hash was already seen on two other bvids in this domain: it must
    never be returned, not even as best_effort, and the call ends empty with
    the exhaustion limitation.
    """
    ledger = SubtitleLedger(tmp_path / "subtitle-ledger.json")
    pool_hash = _subtitle_content_hash(_one_segment(_CORRECT))
    ledger.record("lol_cn", pool_hash, "BV1pool-a")
    ledger.record("lol_cn", pool_hash, "BV1pool-b")
    assert ledger.is_blacklisted("lol_cn", pool_hash)

    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [_one_segment(_CORRECT)] * MAX_SUBTITLE_ATTEMPTS,
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert segments == []
    assert tool.last_subtitle_limitation == SUBTITLE_UNRELIABLE_LIMITATION
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == MAX_SUBTITLE_ATTEMPTS


@pytest.mark.asyncio
async def test_subtitles_blacklist_skips_pool_then_confirms_real(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """A blacklisted pool hash is skipped without poisoning the real subtitle."""
    ledger = SubtitleLedger(tmp_path / "subtitle-ledger.json")
    pool_hash = _subtitle_content_hash(_one_segment(_CORRECT))
    ledger.record("lol_cn", pool_hash, "BV1pool-a")
    ledger.record("lol_cn", pool_hash, "BV1pool-b")

    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [
            _one_segment(_CORRECT),  # pool: blacklisted -> discarded
            _one_segment(_CORRECT),  # pool: blacklisted -> discarded
            _one_segment(_REAL),  # real, relevant, first occurrence
            _one_segment(_REAL),  # real, repeated -> confirmed
        ],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert len(segments) == 1
    assert segments[0]["content"] == _REAL
    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == 4
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == 4


@pytest.mark.asyncio
async def test_subtitles_stop_fetching_after_confirmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """命中即停: a confirmed result on attempt 2 must not trigger attempt 3."""
    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [_one_segment(_CORRECT), _one_segment(_CORRECT)],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == 2
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == 2


@pytest.mark.asyncio
async def test_subtitles_confirm_only_at_final_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Boundary: the confirm pair (repeat + relevance) lands on attempt 5.

    Pool garbage occupies the first three attempts; the real subtitle
    appears only on attempts 4 and 5, so ``confirmed`` is reachable only
    at the last of MAX_SUBTITLE_ATTEMPTS (review m-4).
    """
    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [
            _one_segment(_GARBAGE),
            _one_segment(_GARBAGE),
            _one_segment(_GARBAGE),
            _one_segment(_CORRECT),
            _one_segment(_CORRECT),
        ],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert len(segments) == 1
    assert segments[0]["content"] == _CORRECT
    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == MAX_SUBTITLE_ATTEMPTS
    assert tool.last_subtitle_limitation is None
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == MAX_SUBTITLE_ATTEMPTS


@pytest.mark.asyncio
async def test_subtitles_exhausted_attempts_return_empty_with_limitation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """All attempts fail (network error, then empty tracks) → [] + limitation."""
    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [httpx.ConnectError("network down"), []],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain="lol_cn"
    )

    assert segments == []
    assert tool.last_subtitle_limitation == SUBTITLE_UNRELIABLE_LIMITATION
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == MAX_SUBTITLE_ATTEMPTS


@pytest.mark.asyncio
async def test_subtitles_without_domain_skips_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Empty domain disables recording/blacklist checks but keeps the judgment."""
    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [_one_segment(_CORRECT), _one_segment(_CORRECT)],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": _TITLE}, domain=""
    )

    assert segments[0]["reliability"] == "confirmed"
    assert not (tmp_path / "subtitle-ledger.json").exists()


@pytest.mark.asyncio
async def test_subtitles_missing_cid_sets_absent_limitation(
    tmp_path: Any,
) -> None:
    """No subtitle track exists → [] with the existing absent limitation code."""
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles("BV001", video_info={"cid": 0})

    assert segments == []
    assert tool.last_subtitle_limitation == SUBTITLE_ABSENT_LIMITATION


@pytest.mark.asyncio
async def test_subtitles_confirm_at_exact_title_bigram_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """A 4/20 Han-bigram hit rate equals TITLE_BIGRAM_THRESHOLD exactly and confirms."""
    boundary_body = _one_segment("甲乙丙丁戊")  # shares 4 bigrams -> exactly 0.20
    assert _title_bigram_hit_rate(BOUNDARY_TITLE, "甲乙丙丁戊") == TITLE_BIGRAM_THRESHOLD

    calls: list[str] = []
    FakeClient.get_handler = _sequence_subtitle_handler(
        [boundary_body, boundary_body],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001", video_info={"cid": 456, "title": BOUNDARY_TITLE}, domain="lol_cn"
    )

    assert segments[0]["reliability"] == "confirmed"
    assert segments[0]["attempts"] == 2


@pytest.mark.asyncio
async def test_subtitles_english_fallback_content_not_flagged_relevant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """English mismatched content + letter-heavy title: never returned.

    Live false positive (BV1Xiu86aETY): letter bigrams from BLG/TES/LPL hit
    arbitrary English commentary at ~0.5, which would have flagged pool
    content as related. Han-only bigram counting yields 0.0, so the
    recurring English content is discarded as stable fallback.
    """
    calls: list[str] = []
    english_body = _one_segment("BLG TES LPL mid laner push")
    FakeClient.get_handler = _sequence_subtitle_handler(
        [english_body, english_body],
        calls,
    )
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = _fast_tool(tmp_path)

    segments = await tool.get_subtitles(
        "BV001",
        video_info={"cid": 456, "title": "BLG vs TES 赛后复盘"},
        domain="lol_cn",
    )

    assert segments == []
    assert tool.last_subtitle_limitation == SUBTITLE_UNRELIABLE_LIMITATION
    assert calls.count(bilibili_search.PLAYER_INFO_URL) == MAX_SUBTITLE_ATTEMPTS


def test_title_bigram_hit_rate_counts_shared_bigrams() -> None:
    assert _title_bigram_hit_rate(_TITLE, "全球总决赛") == 4 / 8
    assert _title_bigram_hit_rate(_TITLE, _GARBAGE) == 0.0
    assert _title_bigram_hit_rate(BOUNDARY_TITLE, "甲乙丙丁戊") == 4 / 20
    assert _title_bigram_hit_rate(BOUNDARY_TITLE, "甲乙丙丁") == 3 / 20
    # letter bigrams in a mixed title do not participate in the count
    assert _title_bigram_hit_rate("BLGvsTES赛后复盘", "赛后复盘") == 3 / 3


def test_title_bigram_hit_rate_edge_cases_are_zero() -> None:
    assert _title_bigram_hit_rate("", "任意内容") == 0.0
    assert _title_bigram_hit_rate("单", "单") == 0.0
    assert _title_bigram_hit_rate("标题内容", "") == 0.0
    assert _title_bigram_hit_rate("", "") == 0.0
    # fewer than MIN_TITLE_HAN_BIGRAMS valid Han bigrams -> relevance unusable
    assert _title_bigram_hit_rate("BLGvsTES", "任意中文内容") == 0.0
    assert _title_bigram_hit_rate("甲乙", "甲乙") == 0.0
    assert _title_bigram_hit_rate("甲乙丙", "甲乙丙") == 0.0


def test_title_bigram_hit_rate_ignores_whitespace() -> None:
    assert _title_bigram_hit_rate("英 雄 联 盟 全 球 总 决 赛", "全球 总决赛") == 4 / 8


def test_subtitle_content_hash_is_whitespace_stripped_md5_prefix() -> None:
    expected = hashlib.md5("英雄联盟全球".encode()).hexdigest()[:10]
    assert (
        _subtitle_content_hash([{"content": "英雄 联盟"}, {"content": " 全球"}])
        == expected
    )
    assert _subtitle_content_hash([]) == ""
    assert _subtitle_content_hash([{"content": "  "}]) == ""


@pytest.mark.asyncio
async def test_danmaku_is_time_aligned_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<i><d p="12.5,1,25,16777215,1,0,hash,1">reaction one</d>'
        b'<d p="13.0,1,25,16777215,2,0,hash,2">reaction two</d></i>'
    )
    FakeClient.get_handler = lambda *args, **kwargs: FakeResponse({"_content": xml})
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool()

    async def fake_info(bvid: str) -> dict[str, Any]:
        assert bvid == "BV001"
        return {"cid": 456}

    monkeypatch.setattr(tool, "get_video_info", fake_info)
    danmaku = await tool.get_danmaku("BV001", limit=1)

    assert len(danmaku) == 1
    assert danmaku[0]["video_time"] == 12.5
    assert danmaku[0]["content"] == "reaction one"
    assert danmaku[0]["sampling"] == "first_page_bounded"


@pytest.mark.asyncio
async def test_audio_fetch_reuses_detail_and_uses_lowest_bandwidth_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        requested_urls.append(url)
        if url == bilibili_search.PLAY_URL:
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "dash": {
                            "audio": [
                                {"bandwidth": 10, "baseUrl": "https://audio.example/low"},
                                {"bandwidth": 20, "baseUrl": "https://audio.example/high"},
                            ]
                        }
                    },
                }
            )
        return FakeResponse({"_content": b"encoded-audio"})

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)
    tool = BilibiliSearchTool()

    content = await tool.get_audio_bytes("BV001", video_info={"cid": 456})

    assert content == b"encoded-audio"
    assert requested_urls[-1] == "https://audio.example/low"


@pytest.mark.asyncio
async def test_audio_stream_stops_when_bounded_payload_is_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(*args: Any, **kwargs: Any) -> FakeResponse:
        del kwargs
        url = str(args[-1])
        if url == bilibili_search.PLAY_URL:
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "dash": {
                            "audio": [
                                {
                                    "bandwidth": 10,
                                    "baseUrl": "https://audio.example/oversized",
                                }
                            ]
                        }
                    },
                }
            )
        return FakeResponse({"_content": b"x" * 1_000_001})

    FakeClient.get_handler = handler
    monkeypatch.setattr(bilibili_search.httpx, "AsyncClient", FakeClient)

    content = await BilibiliSearchTool().get_audio_bytes(
        "BV001",
        max_bytes=1_000_000,
        video_info={"cid": 456},
    )

    assert content == b""
