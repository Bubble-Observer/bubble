"""Public-web Channel Adapter backed by bounded, SSRF-aware document tools."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from leave_information_bubble.models.epistemics import (
    AccessDepth,
    ObservationModality,
    SourceObservation,
)
from leave_information_bubble.tools.web_search import (
    PublicWebSearchTool,
    WebDocument,
    WebDocumentOutcome,
    WebSearchIndex,
)

from .acquisition import (
    build_retrieval_contract_report,
    canonicalize_public_url,
    cluster_origin_candidates,
    detect_public_text_language,
)
from .models import (
    AcquisitionEntryKind,
    AcquisitionOutcome,
    CapabilityDescriptor,
    ChannelCapabilityRole,
    ChannelHealth,
    ChannelHealthStatus,
    DiscoveryBatch,
    HydrationDepth,
    HydrationRequest,
    ObservationBatch,
    QuerySemantics,
    RetrievalBatch,
    RetrievalContractReport,
    RetrievalHit,
    ScanRequest,
    SourceOccurrence,
    TimeFilterPrecision,
)


class PublicWebChannelAdapter:
    """Discover public pages and hydrate bounded text without semantic conclusions."""

    adapter_id = "public-web"
    adapter_version = "2.1.0"

    def __init__(
        self,
        tool: PublicWebSearchTool,
        *,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._tool = tool
        self._now_factory = now_factory or (lambda: datetime.now(UTC))
        self._latest_content_at: datetime | None = None

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        """Declare general, news, current, and bounded public-document capabilities."""
        return (
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.SEARCH_INDEX,
                entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
                query_semantics=QuerySemantics.PLATFORM_SEARCH,
                supports_queryless=False,
                languages=["zh-Hans", "en"],
                regions=["CN", "global"],
                bounded_limit=50,
                time_filter_precision=TimeFilterPrecision.COARSE,
                limitations=["public_index_ranking_and_date_fields_are_not_guaranteed"],
            ),
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.NEWS_INDEX,
                entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
                query_semantics=QuerySemantics.PLATFORM_SEARCH,
                supports_queryless=False,
                languages=["zh-Hans", "en"],
                regions=["CN", "global"],
                bounded_limit=50,
                time_filter_precision=TimeFilterPrecision.COARSE,
                limitations=["public_news_rss_may_omit_publication_dates"],
            ),
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.CURRENT_INDEX,
                entry_kind=AcquisitionEntryKind.PLATFORM_SEARCH,
                query_semantics=QuerySemantics.PLATFORM_SEARCH,
                supports_queryless=False,
                languages=["zh-Hans", "en"],
                regions=["CN", "global"],
                bounded_limit=50,
                time_filter_precision=TimeFilterPrecision.COARSE,
                limitations=[
                    "current_index_is_broad_search_without_dedicated_topical_pipeline",
                ],
            ),
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.WEB_BROWSER,
                entry_kind=AcquisitionEntryKind.SOURCE_HYDRATION,
                query_semantics=QuerySemantics.UNSUPPORTED,
                supports_queryless=False,
                languages=["zh-Hans", "en"],
                regions=["CN", "global"],
                bounded_limit=50,
                time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
                limitations=["bounded_read_only_document_extraction"],
            ),
        )

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Search a bounded public index and return source occurrences."""
        if not request.query.strip():
            return DiscoveryBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                outcome=AcquisitionOutcome.UNSUPPORTED,
                partial=True,
                limitations=["public_web_query_required"],
            )
        language = request.language.strip() or "zh-Hans"
        region = request.region.strip() or "CN"
        timezone = request.timezone.strip() or "Asia/Shanghai"
        index = self._requested_index(request)
        result = await self._tool.search(
            request.query,
            limit=request.limit,
            language=language,
            region=region,
            timezone=timezone,
            window_start=request.window_start,
            window_end=request.window_end,
            index=index,
        )
        captured_at = self._utc_now()
        if index is WebSearchIndex.NEWS:
            role = ChannelCapabilityRole.NEWS_INDEX
        elif ChannelCapabilityRole.CURRENT_INDEX in request.capability_roles:
            role = ChannelCapabilityRole.CURRENT_INDEX
        else:
            role = ChannelCapabilityRole.SEARCH_INDEX
        raw_hits: list[RetrievalHit] = []
        for rank, item in enumerate(result.items, start=1):
            if not str(item.get("url", "")).strip():
                continue
            hit = self._retrieval_hit(
                request,
                item,
                rank=rank,
                captured_at=captured_at,
                provider=result.provider or "unknown_public_index",
                role=role,
            )
            if hit is not None:
                raw_hits.append(hit)
        hits = cluster_origin_candidates(raw_hits)
        precision = TimeFilterPrecision(result.time_filter_precision)
        report = build_retrieval_contract_report(
            hits,
            query=request.query,
            target_language=language,
            target_region=region,
            timezone=timezone,
            window_start=request.window_start,
            window_end=request.window_end,
            time_filter_requested=result.time_filter_requested,
            time_filter_applied=result.time_filter_applied,
            time_filter_precision=precision,
            provider_attempts=result.provider_attempts,
            empty_or_challenge_attempts=result.empty_or_challenge_attempts,
            cursor_supported=False,
            cursor_trustworthy=False,
            count_trustworthy=not result.error,
            blind_spots=self._blind_spots(hits, result.time_filter_applied),
        )
        occurrences = [self._occurrence(hit, report) for hit in hits]
        limitations = [
            *result.limitations,
            *([result.error] if result.error else []),
        ]
        if not result.items and not result.error:
            limitations.append("public_web_no_results")
        retrieval = RetrievalBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            hits=hits,
            contract_report=report,
            partial=bool(result.error),
            limitations=list(dict.fromkeys(limitations)),
            physical_call_count=result.physical_call_count,
        )
        return DiscoveryBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=(
                AcquisitionOutcome.PARTIAL
                if result.error and occurrences
                else AcquisitionOutcome.UNAVAILABLE
                if result.error
                else AcquisitionOutcome.SUCCESS
                if occurrences
                else AcquisitionOutcome.EMPTY
            ),
            occurrences=occurrences,
            partial=bool(result.error),
            limitations=list(dict.fromkeys(limitations)),
            physical_call_count=result.physical_call_count,
            retrieval=retrieval,
        )

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Fetch one bounded page and emit depth-appropriate observations.

        METADATA returns only a metadata observation (title, resolved URL,
        type, char count); REACTIONS states explicitly that public pages have
        no discussion (``public_web_no_discussion``); TEXT/MEDIA_TEXT/
        STRUCTURED return qualified static main text. ``full=True`` attaches
        that text as the observation body wherever a text-bearing depth or
        the metadata depth is requested, mirroring the bilibili convention.
        Typed non-material outcomes return no observation or body.
        """
        document = await self._tool.fetch(request.source_ref)
        if document.error:
            return ObservationBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                outcome=(
                    AcquisitionOutcome.UNSUPPORTED
                    if document.error.startswith("url rejected:")
                    else AcquisitionOutcome.UNAVAILABLE
                ),
                partial=True,
                limitations=[document.error],
                physical_call_count=1,
            )
        if document.outcome is not WebDocumentOutcome.QUALIFIED_FULL:
            return ObservationBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                outcome=self._document_outcome(document.outcome),
                partial=True,
                limitations=self._qualification_limitations(document.outcome, document.limitations),
                physical_call_count=1,
            )
        text = document.text.strip()
        if not text:
            return ObservationBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                outcome=AcquisitionOutcome.EMPTY,
                partial=True,
                limitations=["public_web_document_empty"],
                physical_call_count=1,
            )
        full = bool(request.arguments.get("full"))
        if request.depth is HydrationDepth.METADATA:
            observations = [
                self._metadata_observation(request, document, text=text, body=text if full else None)
            ]
            limitations = ["document_text_truncated"] if full and len(text) > 4000 else []
        elif request.depth is HydrationDepth.REACTIONS:
            observations = [self._metadata_observation(request, document, text=text)]
            limitations = ["public_web_no_discussion"]
        else:
            observations = [
                self._document_observation(request, document, text=text, body=text if full else None)
            ]
            limitations = ["document_text_truncated"] if len(text) > 4000 else []
        captured_at = self._utc_now()
        discovered_occurrences = [
            SourceOccurrence(
                id=self._stable_id("occurrence", link),
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                source_ref=link,
                canonical_url=link,
                title=link,
                content_type="public_web_link",
                language="",
                captured_at=captured_at,
                metadata={
                    "provider": "public_web_document_link",
                    "capability_role": ChannelCapabilityRole.WEB_BROWSER.value,
                    "independence_status": "unknown",
                    "field_limitations": [
                        "one_hop_link_is_discovery_only",
                        "linked_document_not_opened",
                    ],
                },
            )
            for link in document.links[:8]
            if canonicalize_public_url(link)
        ]
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=(AcquisitionOutcome.PARTIAL if limitations else AcquisitionOutcome.SUCCESS),
            observations=observations,
            discovered_occurrences=discovered_occurrences,
            limitations=limitations,
            partial=bool(limitations),
            physical_call_count=1,
        )

    @staticmethod
    def _document_outcome(outcome: WebDocumentOutcome) -> AcquisitionOutcome:
        """Map one qualified-document state to the shared acquisition vocabulary."""
        if outcome in {
            WebDocumentOutcome.LOGIN_OR_PAYWALL,
            WebDocumentOutcome.RENDERING_REQUIRED,
        }:
            return AcquisitionOutcome.UNSUPPORTED
        if outcome is WebDocumentOutcome.PARTIAL_CONTENT:
            return AcquisitionOutcome.PARTIAL
        if outcome is WebDocumentOutcome.PAGE_SHELL:
            return AcquisitionOutcome.EMPTY
        return AcquisitionOutcome.UNAVAILABLE

    @staticmethod
    def _qualification_limitations(
        outcome: WebDocumentOutcome,
        document_limitations: tuple[str, ...],
    ) -> list[str]:
        """Return bounded source-state and capability facts for non-material pages."""
        source_facts = {
            WebDocumentOutcome.LOGIN_OR_PAYWALL: "public_web_no_auth_bypass",
            WebDocumentOutcome.CHALLENGE_OR_BLOCKED: "public_web_access_challenge_active",
            WebDocumentOutcome.RENDERING_REQUIRED: "public_web_static_rendering_unavailable",
            WebDocumentOutcome.PARTIAL_CONTENT: "public_web_only_partial_content_available",
        }
        limitations = [*document_limitations, f"public_web_{outcome.value}"]
        if source_fact := source_facts.get(outcome):
            limitations.append(source_fact)
        return list(dict.fromkeys(limitations))

    def _document_observation(
        self,
        request: HydrationRequest,
        document: WebDocument,
        *,
        text: str,
        body: str | None = None,
    ) -> SourceObservation:
        """Build the full-document text observation for text-bearing depths."""
        return SourceObservation(
            id=self._stable_id("observation", request.source_ref, text),
            source_ref=request.source_ref,
            modality=ObservationModality.DOCUMENT_TEXT,
            access_depth=AccessDepth.CONTENT_TEXT,
            excerpt=text[:4000],
            location=document.url,
            acquisition_method="public_web_fetch",
            captured_at=self._utc_now(),
            metadata={
                "title": document.title,
                "content_type": document.content_type,
                "resolved_url": document.url,
            },
            body=body,
        )

    def _metadata_observation(
        self,
        request: HydrationRequest,
        document: WebDocument,
        *,
        text: str,
        body: str | None = None,
    ) -> SourceObservation:
        """Build the metadata-only observation for METADATA/REACTIONS depths."""
        metadata = {
            "title": document.title,
            "content_type": document.content_type,
            "resolved_url": document.url,
            "char_count": len(text),
        }
        return SourceObservation(
            id=self._stable_id(
                "observation",
                request.source_ref,
                "metadata",
                f"{document.url}\0{len(text)}",
            ),
            source_ref=request.source_ref,
            modality=ObservationModality.METADATA,
            access_depth=AccessDepth.METADATA,
            excerpt=document.title,
            location=document.url,
            acquisition_method="public_web_fetch_metadata",
            captured_at=self._utc_now(),
            metadata=metadata,
            body=body,
        )

    async def changes_since(self, request: ScanRequest) -> DiscoveryBatch:
        """Use bounded rediscovery until a source-specific cursor is available."""
        batch = await self.discover(request)
        return batch.model_copy(
            update={"limitations": list(dict.fromkeys([*batch.limitations, "cursor_not_supported"]))}
        )

    async def health(self) -> ChannelHealth:
        """Report configured availability with freshness tracking."""
        checked_at = self._utc_now()
        freshness: int | None = None
        status = ChannelHealthStatus.HEALTHY
        detail = ""
        if self._latest_content_at is not None:
            freshness = int((checked_at - self._latest_content_at).total_seconds())
            if freshness > 7 * 86400:  # 7 days
                status = ChannelHealthStatus.DEGRADED
                detail = (
                    f"most recent source content is {freshness // 86400}d old; "
                    "search indexes may return predominantly stale results"
                )
        return ChannelHealth(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            status=status,
            checked_at=checked_at,
            detail=detail,
            data_freshness_seconds=freshness,
        )

    def _retrieval_hit(
        self,
        request: ScanRequest,
        item: dict[str, str],
        *,
        rank: int,
        captured_at: datetime,
        provider: str,
        role: ChannelCapabilityRole,
    ) -> RetrievalHit | None:
        url = str(item.get("url", "")).strip()
        canonical_url = canonicalize_public_url(url)
        if not canonical_url:
            return None
        title = str(item.get("title", ""))[:2000]
        snippet = str(item.get("snippet", ""))[:8000]
        published_at = self._parse_published_at(str(item.get("published_at", "")))
        if published_at is not None and (
            self._latest_content_at is None or published_at > self._latest_content_at
        ):
            self._latest_content_at = published_at
        return RetrievalHit(
            id=self._stable_id("retrieval-hit", request.id, provider, str(rank), url),
            request_id=request.id,
            provider=provider,
            capability_role=role,
            query=request.query,
            rank=rank,
            url=url,
            canonical_url=canonical_url,
            title=title,
            snippet=snippet,
            detected_language=detect_public_text_language(f"{title}\n{snippet}"),
            published_at=published_at,
            captured_at=captured_at,
            metadata={
                "date_field_present": published_at is not None,
                "field_limitations": ["search_snippet_is_discovery_only"],
            },
        )

    def _occurrence(
        self,
        hit: RetrievalHit,
        report: RetrievalContractReport,
    ) -> SourceOccurrence:
        return SourceOccurrence(
            id=self._stable_id("occurrence", hit.url),
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            source_ref=hit.url,
            canonical_url=hit.canonical_url,
            title=hit.title,
            content_type="public_web_page",
            language=hit.detected_language,
            source_published_at=hit.published_at,
            captured_at=hit.captured_at,
            metadata={
                "snippet": hit.snippet,
                "provider": hit.provider,
                "capability_role": hit.capability_role.value,
                "query": hit.query,
                "query_rank": hit.rank,
                "origin_cluster_id": hit.origin_cluster_id,
                "independence_status": hit.independence_status.value,
                "target_language": report.target_language,
                "language_contract_matches": (
                    hit.detected_language.casefold().split("-", 1)[0]
                    == report.target_language.casefold().split("-", 1)[0]
                ),
                "retrieval_contract_degraded": report.degraded,
                **hit.metadata,
            },
        )

    @staticmethod
    def _requested_index(request: ScanRequest) -> WebSearchIndex:
        raw_index = str(request.arguments.get("index", "")).strip().lower()
        if (
            ChannelCapabilityRole.NEWS_INDEX in request.capability_roles
            or raw_index == WebSearchIndex.NEWS.value
        ):
            return WebSearchIndex.NEWS
        return WebSearchIndex.GENERAL

    @staticmethod
    def _parse_published_at(value: str) -> datetime | None:
        if not value.strip():
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    @staticmethod
    def _blind_spots(
        hits: list[RetrievalHit],
        time_filter_applied: bool,
    ) -> list[str]:
        blind_spots: list[str] = []
        if hits and not any(hit.published_at is not None for hit in hits):
            blind_spots.append("index_did_not_expose_publication_dates")
        if not time_filter_applied:
            blind_spots.append("provider_did_not_apply_requested_time_filter")
        else:
            blind_spots.append("provider_time_filter_effect_not_verified")
        blind_spots.append("search_result_counts_and_cursor_are_not_authoritative")
        return blind_spots

    @staticmethod
    def _stable_id(kind: str, *parts: str) -> str:
        payload = "\0".join(parts).encode("utf-8")
        return f"{kind}-{hashlib.sha256(payload).hexdigest()[:32]}"

    def _utc_now(self) -> datetime:
        value = self._now_factory()
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["PublicWebChannelAdapter"]
