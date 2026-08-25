"""Read-only NGA community channel backed by bounded public thread pages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from leave_information_bubble.models.epistemics import (
    AccessDepth,
    ObservationModality,
    SourceObservation,
)
from leave_information_bubble.tools.nga import NgaPublicTool, NgaThreadPage

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
    ScanRequest,
    SourceOccurrence,
    TimeFilterPrecision,
)


class NgaChannelAdapter:
    """Expose bounded NGA board discovery and floor hydration."""

    adapter_id = "nga"
    adapter_version = "1.0.0"

    def __init__(
        self,
        tool: NgaPublicTool,
        *,
        default_fid: str | None = "-152678",
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._tool = tool
        self._default_fid = default_fid
        self._now_factory = now_factory or (lambda: datetime.now(UTC))
        self._last_error = ""
        self._latest_content_at: datetime | None = None

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        """Declare NGA's single scoped Chinese community board surface."""
        access_limitations = [
            "public_pages_may_require_browser_established_session",
            "board_order_is_platform_controlled",
            "query_matching_is_client_side_over_bounded_board_pages",
        ]
        if self._default_fid is None:
            # no fid was configured for this domain: a queryless scan would
            # silently hit the wrong platform section, so the adapter reports
            # it cannot start without an explicit surface_key instead
            access_limitations.append(
                "no_default_fid_for_domain: provide surface_key with an "
                "explicit NGA fid (e.g. '-152678')"
            )
        return (
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.COMMUNITY_STREAM,
                entry_kind=AcquisitionEntryKind.BOUNDED_BOARD,
                query_semantics=QuerySemantics.BOUNDED_RERANK_HINT,
                supports_queryless=self._default_fid is not None,
                languages=["zh-Hans"],
                regions=["CN"],
                supports_cursor=True,
                supports_metrics=True,
                supports_authentication=True,
                bounded_limit=100,
                relative_cost=1.2,
                time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
                limitations=access_limitations,
            ),
        )

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Scan one board page and rank exact query hints without hiding order."""
        cursor_page, cursor_fid = self._cursor_options(request.cursor)
        fid = request.arguments.get("surface_key", cursor_fid or self._default_fid)
        if not fid:
            return DiscoveryBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                outcome=AcquisitionOutcome.UNSUPPORTED,
                partial=True,
                limitations=[
                    "no_default_fid_for_domain: provide surface_key with an "
                    "explicit NGA fid (e.g. '-152678')"
                ],
            )
        fid = str(fid)
        page = cursor_page
        result = await self._tool.scan_board(
            fid,
            page=page,
            limit=min(request.limit, 100),
        )
        if result.error:
            self._last_error = result.error
            return DiscoveryBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                outcome=AcquisitionOutcome.UNAVAILABLE,
                partial=True,
                limitations=[result.error],
                physical_call_count=1,
            )
        self._last_error = ""
        captured_at = self._utc_now()
        query_terms = self._query_terms(request.query)
        ranked_items = sorted(
            enumerate(result.threads, start=1),
            key=lambda pair: (
                not any(term in pair[1].title.casefold() for term in query_terms),
                pair[0],
            ),
        )
        occurrences: list[SourceOccurrence] = []
        for relevance_rank, (board_rank, item) in enumerate(ranked_items, start=1):
            freshest = item.last_reply_at or item.published_at
            if freshest is not None and (
                self._latest_content_at is None or freshest > self._latest_content_at
            ):
                self._latest_content_at = freshest
            normalized_title = item.title.casefold()
            matched_terms = [term for term in query_terms if term in normalized_title]
            occurrences.append(SourceOccurrence(
                id=self._stable_id("occurrence", item.thread_id),
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                source_ref=f"nga-thread-{item.thread_id}",
                canonical_url=item.url,
                title=item.title,
                author=item.author,
                content_type="nga_thread",
                language="zh-Hans",
                source_published_at=item.published_at,
                captured_at=captured_at,
                metadata={
                    "provider": "nga_public_board",
                    "capability_role": ChannelCapabilityRole.COMMUNITY_STREAM.value,
                    "fid": result.fid,
                    "board_page": result.page,
                    "board_rank": board_rank,
                    "query_relevance_rank": relevance_rank,
                    "query": request.query,
                    "query_matched_terms": matched_terms,
                    "query_match_is_client_side": True,
                    "engagement": {"replies": item.reply_count},
                    "last_reply_at": (
                        item.last_reply_at.isoformat()
                        if item.last_reply_at is not None else None
                    ),
                    "independence_status": "unknown",
                    "field_limitations": [
                        "thread_listing_is_discovery_only",
                        "community_thread_is_not_event_fact_authority",
                    ],
                },
            ))
        limitations = list(result.limitations)
        if request.query.strip():
            limitations.append("nga_query_applied_as_bounded_client_side_hint")
        return DiscoveryBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=(
                AcquisitionOutcome.SUCCESS
                if occurrences
                else AcquisitionOutcome.EMPTY
            ),
            occurrences=occurrences,
            next_cursor=json.dumps(
                {"page": result.page + 1, "fid": result.fid}, separators=(",", ":")
            ),
            limitations=list(dict.fromkeys(limitations)),
            physical_call_count=1,
        )

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Hydrate a discovered main post or its bounded visible floors."""
        thread_id = self._tool.thread_id(request.source_ref)
        if not thread_id:
            return self._failure(request, "invalid_nga_thread", physical_calls=0)
        sampling_raw = request.arguments.get("comment_sampling", {})
        sampling = sampling_raw if isinstance(sampling_raw, dict) else {}
        page = min(
            max(int(sampling.get("page", request.arguments.get("page", 1))), 1),
            100,
        )
        reply_limit = min(
            max(
                int(
                    sampling.get(
                        "max_total",
                        sampling.get(
                            "reply_limit", request.arguments.get("reply_limit", 40)
                        ),
                    )
                ),
                1 if request.depth is HydrationDepth.REACTIONS else 0,
            ),
            100,
        )
        thread = await self._tool.get_thread(
            thread_id,
            page=page,
            reply_limit=reply_limit,
        )
        if thread.error:
            self._last_error = thread.error
            return self._failure(request, thread.error, physical_calls=1)
        self._last_error = ""
        full = bool(request.arguments.get("full"))
        # Mirror the bilibili convention: at METADATA/REACTIONS depths the
        # main post rides as the metadata observation's full-depth body so
        # full=True is never silently ignored there; at document-bearing
        # depths the document observation carries it instead.
        metadata_body = (
            thread.content
            if full
            and request.depth in {HydrationDepth.METADATA, HydrationDepth.REACTIONS}
            and thread.content
            else None
        )
        observations = [
            self._metadata_observation(
                request.source_ref, thread, body=metadata_body
            )
        ]
        limitations = list(thread.limitations)
        if thread.published_at is None:
            limitations.append("nga_published_at_unavailable")
        if request.depth in {
            HydrationDepth.TEXT,
            HydrationDepth.MEDIA_TEXT,
            HydrationDepth.STRUCTURED,
        } and thread.content:
            observations.append(SourceObservation(
                id=self._stable_id(
                    "observation", request.source_ref, "document", thread.content
                ),
                source_ref=request.source_ref,
                modality=ObservationModality.DOCUMENT_TEXT,
                access_depth=AccessDepth.CONTENT_TEXT,
                excerpt=thread.content[:4000],
                location=f"thread:{thread.thread_id}:floor:0",
                acquisition_method="nga_public_thread_html",
                captured_at=self._utc_now(),
                sampling_scope=(
                    "single community-authored NGA main post;"
                    "not representative of the board or wider community"
                ),
                limitations=["community_post_is_author_claim_not_event_fact"],
                metadata={
                    "thread_id": thread.thread_id,
                    "title": thread.title,
                    "page": thread.page,
                    "floor": 0,
                    "author_ref": thread.author_ref,
                    "published_at": (
                        thread.published_at.isoformat()
                        if thread.published_at is not None else None
                    ),
                    "evidence_role": "community_author_claim",
                },
                body=thread.content if full else None,
            ))
            if len(thread.content) > 4000:
                limitations.append("document_text_truncated")
        if request.depth is HydrationDepth.REACTIONS:
            for reply in thread.replies:
                observations.append(SourceObservation(
                    id=self._stable_id(
                        "observation", request.source_ref, reply.reply_ref, reply.content
                    ),
                    source_ref=request.source_ref,
                    modality=ObservationModality.COMMENT,
                    access_depth=AccessDepth.REACTIONS,
                    excerpt=reply.content[:4000],
                    location=f"floor:{reply.floor}:page:{thread.page}",
                    acquisition_method="nga_public_floor_sample",
                    captured_at=self._utc_now(),
                    sampling_scope=(
                        "platform-visible bounded NGA floor page;"
                        f"page={thread.page};limit={reply_limit};"
                        f"returned={len(thread.replies)}"
                    ),
                    limitations=[
                        "floor_sample_is_not_whole_community_opinion",
                        "same_thread_floors_are_not_independent_source_origins",
                        "deleted_or_collapsed_floors_may_be_absent",
                    ],
                    metadata={
                        "reply_ref": reply.reply_ref,
                        "floor": reply.floor,
                        "author_ref": reply.author_ref,
                        "published_at": (
                            reply.published_at.isoformat()
                            if reply.published_at is not None else None
                        ),
                        "support_count": reply.support_count,
                        "quoted_reply_ref": reply.quoted_reply_ref,
                        "page": thread.page,
                        "independence_scope": f"nga-thread-{thread.thread_id}",
                    },
                ))
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            observations=observations,
            partial=bool(limitations),
            limitations=list(dict.fromkeys(limitations)),
            physical_call_count=1,
        )

    async def changes_since(self, request: ScanRequest) -> DiscoveryBatch:
        """Reuse ordered discovery because NGA exposes no public delta cursor."""
        return await self.discover(request)

    async def health(self) -> ChannelHealth:
        """Report whether anonymous/session-backed public access is usable."""
        checked_at = self._utc_now()
        retry_after = None
        if self._last_error == "nga_rate_limited":
            status = ChannelHealthStatus.RATE_LIMITED
            retry_after = checked_at + timedelta(minutes=10)
        elif self._last_error == "nga_browser_session_required":
            status = ChannelHealthStatus.AUTHENTICATION_REQUIRED
        elif self._last_error in {"nga_board_closed", "nga_redirect_not_allowed"}:
            status = ChannelHealthStatus.BLOCKED
        elif self._last_error.endswith("schema_changed"):
            status = ChannelHealthStatus.SCHEMA_CHANGED
        elif self._last_error:
            status = ChannelHealthStatus.DEGRADED
        else:
            status = ChannelHealthStatus.HEALTHY
        freshness = (
            int((checked_at - self._latest_content_at).total_seconds())
            if self._latest_content_at is not None else None
        )
        return ChannelHealth(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            status=status,
            checked_at=checked_at,
            retry_after=retry_after,
            failure_code=self._last_error,
            detail=(
                "Configure an authorized browser-established NGA session cookie."
                if status is ChannelHealthStatus.AUTHENTICATION_REQUIRED else ""
            ),
            data_freshness_seconds=max(0, freshness) if freshness is not None else None,
        )

    def _metadata_observation(
        self,
        source_ref: str,
        thread: NgaThreadPage,
        *,
        body: str | None = None,
    ) -> SourceObservation:
        metadata = {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "canonical_url": thread.url,
            "page": thread.page,
            "platform": "nga",
            "visible_reply_count": len(thread.replies),
        }
        return SourceObservation(
            id=self._stable_id(
                "observation",
                source_ref,
                "metadata",
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
            source_ref=source_ref,
            modality=ObservationModality.METADATA,
            access_depth=AccessDepth.METADATA,
            excerpt=thread.title,
            location="thread_metadata",
            acquisition_method="nga_public_thread_metadata",
            captured_at=self._utc_now(),
            limitations=["metadata_supports_attention_and_identity_only"],
            metadata=metadata,
            body=body,
        )

    def _failure(
        self,
        request: HydrationRequest,
        code: str,
        *,
        physical_calls: int,
    ) -> ObservationBatch:
        return ObservationBatch(
            request_id=request.id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            outcome=(
                AcquisitionOutcome.UNSUPPORTED
                if physical_calls == 0
                else AcquisitionOutcome.UNAVAILABLE
            ),
            partial=True,
            limitations=[code],
            physical_call_count=physical_calls,
        )

    @staticmethod
    def _cursor_options(cursor: str) -> tuple[int, str]:
        if not cursor.strip():
            return 1, ""
        try:
            value = json.loads(cursor)
            if isinstance(value, dict):
                page = min(max(int(value.get("page", 1)), 1), 100)
                fid = str(value.get("fid", "")).strip()
                return page, fid
            return min(max(int(value), 1), 100), ""
        except (ValueError, TypeError, json.JSONDecodeError):
            return 1, ""

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        return [
            term.casefold()
            for term in query.replace("/", " ").split()
            if len(term.strip()) >= 2
        ][:12]

    @staticmethod
    def _stable_id(kind: str, *parts: str) -> str:
        payload = "\0".join(parts).encode("utf-8")
        return f"{kind}-{hashlib.sha256(payload).hexdigest()[:32]}"

    def _utc_now(self) -> datetime:
        value = self._now_factory()
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["NgaChannelAdapter"]
