from __future__ import annotations

from datetime import UTC, datetime, timedelta

from leave_information_bubble.channels import (
    ChannelCapabilityRole,
    IndependenceStatus,
    RetrievalHit,
    StratifiedCardReservoir,
    TimeFilterPrecision,
    build_retrieval_contract_report,
    canonicalize_public_url,
    cluster_origin_candidates,
    detect_public_text_language,
)

NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)


def _hit(
    identifier: str,
    *,
    url: str,
    title: str,
    snippet: str = "这是用于发现的足够长的中文检索摘要内容。",
    language: str = "zh-Hans",
    role: ChannelCapabilityRole = ChannelCapabilityRole.SEARCH_INDEX,
    query: str = "英雄联盟 转会",
    rank: int = 1,
) -> RetrievalHit:
    return RetrievalHit(
        id=identifier,
        request_id="scan-1",
        provider="public-index",
        capability_role=role,
        query=query,
        rank=rank,
        url=url,
        canonical_url=canonicalize_public_url(url),
        title=title,
        snippet=snippet,
        detected_language=language,
        captured_at=NOW,
    )


def test_canonical_url_removes_fragments_and_tracking_without_erasing_identity() -> None:
    assert canonicalize_public_url(
        "HTTPS://Example.COM:443/news//item/?b=2&utm_source=x&a=1#comments"
    ) == "https://example.com/news/item?a=1&b=2"
    assert canonicalize_public_url("https://example.com:not-a-port/item") == ""
    assert canonicalize_public_url("javascript:alert(1)") == ""


def test_language_detection_does_not_mistake_japanese_kana_for_chinese() -> None:
    assert detect_public_text_language("英雄联盟赛事最新消息") == "zh-Hans"
    assert detect_public_text_language("リーグ・オブ・レジェンド 最新情報") == "ja"
    assert detect_public_text_language("League of Legends roster news") == "en"


def test_origin_cluster_collapses_reprints_but_never_invents_independence() -> None:
    first = _hit(
        "hit-1",
        url="https://one.example/report",
        title="俱乐部今日公布全新阵容名单",
    )
    reprint = _hit(
        "hit-2",
        url="https://two.example/copy",
        title=first.title,
    )
    unique = _hit(
        "hit-3",
        url="https://three.example/analysis",
        title="分析师对阵容变化提出不同解释",
        snippet="另一位分析师根据公开资料给出了独立背景和不同解释。",
    )

    clustered = cluster_origin_candidates([first, reprint, unique])

    assert clustered[0].origin_cluster_id == clustered[1].origin_cluster_id
    assert clustered[0].independence_status is IndependenceStatus.SHARED_ORIGIN
    assert clustered[1].independence_status is IndependenceStatus.SHARED_ORIGIN
    assert clustered[2].independence_status is IndependenceStatus.UNKNOWN


def test_contract_report_exposes_language_anchor_time_and_host_degradation() -> None:
    hits = [
        _hit(
            f"hit-{index}",
            url=f"https://same.example/{index}",
            title="英雄联盟の最新ニュース",
            language="ja",
            rank=index,
        )
        for index in range(1, 6)
    ]

    report = build_retrieval_contract_report(
        hits,
        query="英雄联盟 转会",
        target_language="zh-Hans",
        target_region="CN",
        timezone="Asia/Shanghai",
        window_start=NOW - timedelta(days=30),
        window_end=NOW,
        time_filter_requested=True,
        time_filter_applied=False,
        time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
        provider_attempts=["provider-a", "provider-b"],
        empty_or_challenge_attempts=2,
    )

    assert report.target_language_ratio == 0.0
    assert report.entity_anchor_hit_ratio == 1.0
    assert report.dated_result_count == 0
    assert set(report.degraded_reasons) >= {
        "target_language_below_80_percent",
        "empty_or_challenge_above_50_percent",
        "single_host_above_80_percent",
        "provider_time_filter_not_applied",
    }


def test_stratified_reservoir_merges_appearances_and_preserves_language_mix() -> None:
    chinese = [
        _hit(
            f"zh-{index}",
            url=f"https://zh{index % 3}.example/{index}",
            title=f"中文来源 {index} 提供不同上下文",
            rank=index,
            role=(
                ChannelCapabilityRole.NEWS_INDEX
                if index % 2
                else ChannelCapabilityRole.SEARCH_INDEX
            ),
        )
        for index in range(1, 9)
    ]
    international = [
        _hit(
            f"en-{index}",
            url=f"https://en{index}.example/{index}",
            title=f"International primary source {index}",
            snippet="An original international source with useful context.",
            language="en",
            rank=index,
        )
        for index in range(1, 3)
    ]
    duplicate_appearance = chinese[0].model_copy(
        update={
            "id": "same-card-second-query",
            "query": "俱乐部 阵容",
            "provider": "second-index",
        }
    )
    reservoir = StratifiedCardReservoir()

    selected = reservoir.select(
        [*chinese, *international, duplicate_appearance],
        limit=10,
        target_language_ratio=0.8,
    )

    assert len(selected) == 10
    assert sum(hit.detected_language == "zh-Hans" for hit in selected) == 8
    merged = next(hit for hit in selected if hit.canonical_url == chinese[0].canonical_url)
    assert merged.metadata["appearance_count"] == 2
    assert merged.metadata["cross_query_count"] == 2
