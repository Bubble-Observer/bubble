"""Read-only Hupu community channel backed by public forum HTML."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime

from leave_information_bubble.models.epistemics import (
    AccessDepth,
    ObservationModality,
    SourceObservation,
)
from leave_information_bubble.tools.hupu import HupuPublicTool, HupuThreadPage

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


class HupuChannelAdapter:
    """Expose bounded Hupu board discovery and thread/reply hydration."""

    adapter_id = "hupu"
    adapter_version = "1.0.0"

    def __init__(
        self,
        tool: HupuPublicTool,
        *,
        default_board: str | None = "lol",
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._tool = tool
        self._default_board = default_board
        self._now_factory = now_factory or (lambda: datetime.now(UTC))
        self._last_error = ""
        self._latest_content_at: datetime | None = None

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        """Describe the single bounded public community surface exposed here."""
        limitations = [
            "public_anonymous_html_only",
            "board_order_is_platform_controlled",
            "query_matching_is_client_side_over_bounded_board_pages",
        ]
        if self._default_board is None:
            # no board was configured for this domain: a queryless scan would
            # silently hit the wrong platform section, so the adapter reports
            # it cannot start without an explicit surface_key instead
            limitations.append(
                "no_default_board_for_domain: provide surface_key with an "
                "explicit hupu board name (e.g. 'lol', 'csgo')"
            )
        return (
            CapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                role=ChannelCapabilityRole.COMMUNITY_STREAM,
                entry_kind=AcquisitionEntryKind.BOUNDED_BOARD,
                query_semantics=QuerySemantics.BOUNDED_RERANK_HINT,
                supports_queryless=self._default_board is not None,
                languages=["zh-Hans"],
                regions=["CN"],
                supports_cursor=True,
                supports_metrics=True,
                bounded_limit=100,
                relative_cost=0.8,
                time_filter_precision=TimeFilterPrecision.UNSUPPORTED,
                limitations=limitations,
            ),
        )

    async def discover(self, request: ScanRequest) -> DiscoveryBatch:
        """Scan one public board page and rank exact query hints first."""
        cursor_page, cursor_board = self._cursor_options(request.cursor)
        board = request.arguments.get("surface_key", cursor_board or self._default_board)
        if not board:
            return DiscoveryBatch(
                request_id=request.id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                outcome=AcquisitionOutcome.UNSUPPORTED,
                partial=True,
                limitations=[
                    "no_default_board_for_domain: provide surface_key with an "
                    "explicit hupu board name (e.g. 'lol', 'csgo')"
                ],
            )
        board = str(board)
        page = cursor_page
        result = await self._tool.scan_board(
            board,
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
        captured_at = self._utc_now()
        query_terms = self._query_terms(request.query)
        occurrences: list[SourceOccurrence] = []
        ranked_items = sorted(
            enumerate(result.threads, start=1),
            key=lambda pair: (
                not any(term in pair[1].title.casefold() for term in query_terms),
                pair[0],
            ),
        )
        for relevance_rank, (board_rank, item) in enumerate(ranked_items, start=1):
            if item.published_at is not None and (
                self._latest_content_at is None
                or item.published_at > self._latest_content_at
            ):
                self._latest_content_at = item.published_at
            normalized_title = item.title.casefold()
            matched_terms = [term for term in query_terms if term in normalized_title]
            occurrences.append(SourceOccurrence(
                id=self._stable_id("occurrence", item.thread_id),
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                source_ref=f"hupu-thread-{item.thread_id}",
                canonical_url=item.url,
                title=item.title,
                author=item.author,
                content_type="hupu_thread",
                language="zh-Hans",
                source_published_at=item.published_at,
                captured_at=captured_at,
                metadata={
                    "provider": "hupu_public_board",
                    "capability_role": ChannelCapabilityRole.COMMUNITY_STREAM.value,
                    "board": result.board,
                    "board_page": result.page,
                    "board_rank": board_rank,
                    "query_relevance_rank": relevance_rank,
                    "query": request.query,
                    "query_matched_terms": matched_terms,
                    "query_match_is_client_side": True,
                    "engagement": {
                        "views": item.view_count,
                        "replies": item.reply_count,
                    },
                    "independence_status": "unknown",
                    "field_limitations": [
                        "thread_listing_is_discovery_only",
                        "community_thread_is_not_event_fact_authority",
                    ],
                },
            ))
        limitations = list(result.limitations)
        if request.query.strip():
            limitations.append("hupu_query_applied_as_bounded_client_side_hint")
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
                {"page": result.page + 1, "board": result.board},
                separators=(",", ":"),
            ),
            limitations=list(dict.fromkeys(limitations)),
            physical_call_count=1,
        )

    async def hydrate(self, request: HydrationRequest) -> ObservationBatch:
        """Hydrate a previously discovered thread or its visible replies."""
        thread_id = self._tool.thread_id(request.source_ref)
        if not thread_id:
            return self._failure(request, "invalid_hupu_thread", physical_calls=0)
        sampling_raw = request.arguments.get("comment_sampling", {})
        sampling = sampling_raw if isinstance(sampling_raw, dict) else {}
        page = min(max(int(sampling.get("page", request.arguments.get("page", 1))), 1), 100)
        reply_limit = min(
            max(int(sampling.get("reply_limit", request.arguments.get("reply_limit", 40))), 0),
            100,
        )
        if request.depth is HydrationDepth.REACTIONS:
            reply_limit = min(
                max(int(sampling.get("max_total", request.arguments.get("max_total", reply_limit))), 1),
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
        full = bool(request.arguments.get("full"))
        # Mirror the bilibili convention: at METADATA/REACTIONS depths the
        # thread main text rides as the metadata observation's full-depth
        # body so full=True is never silently ignored there; at
        # document-bearing depths the document observation carries it.
        metadata_body = (
            thread.content
            if full
            and request.depth in {HydrationDepth.METADATA, HydrationDepth.REACTIONS}
            and thread.content
            else None
        )
        observations = [
            self._metadata_observation(
                request.source_ref, thread, page, body=metadata_body
            )
        ]
        limitations = list(thread.limitations)
        if request.depth in {
            HydrationDepth.TEXT,
            HydrationDepth.MEDIA_TEXT,
            HydrationDepth.STRUCTURED,
        } and thread.content:
            observations.append(SourceObservation(
                id=self._stable_id("observation", request.source_ref, "document", thread.content),
                source_ref=request.source_ref,
                modality=ObservationModality.DOCUMENT_TEXT,
                access_depth=AccessDepth.CONTENT_TEXT,
                excerpt=thread.content[:4000],
                location=f"thread:{thread.thread_id}:main",
                acquisition_method="hupu_public_thread_html",
                captured_at=self._utc_now(),
                sampling_scope=(
                    "single community-authored thread main post;"
                    "not representative of the whole community"
                ),
                limitations=["community_post_is_author_claim_not_event_fact"],
                metadata={
                    "thread_id": thread.thread_id,
                    "title": thread.title,
                    "page": page,
                    "evidence_role": "community_author_claim",
                },
                body=thread.content if full else None,
            ))
            if len(thread.content) > 4000:
                limitations.append("document_text_truncated")
        if request.depth is HydrationDepth.REACTIONS:
            for reply in thread.replies:
                observations.append(SourceObservation(
                    id=self._stable_id("observation", request.source_ref, reply.reply_ref),
                    source_ref=request.source_ref,
                    modality=ObservationModality.COMMENT,
                    access_depth=AccessDepth.REACTIONS,
                    excerpt=reply.content[:4000],
                    location=f"reply:{reply.reply_ref}:page:{page}",
                    acquisition_method="hupu_public_reply_sample",
                    captured_at=self._utc_now(),
                    sampling_scope=(
                        "platform-visible bounded reply page;"
                        f"page={page};limit={reply_limit};"
                        f"returned={len(thread.replies)}"
                    ),
                    limitations=[
                        "reply_sample_is_not_whole_community_opinion",
                        "nested_reply_dialogs_not_expanded",
                    ],
                    metadata={
                        "reply_ref": reply.reply_ref,
                        "author_ref": self._author_ref(reply.author),
                        "published_at": (
                            reply.published_at.isoformat()
                            if reply.published_at is not None else None
                        ),
                        "highlighted": reply.highlighted,
                        "page": page,
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
        """Reuse bounded board discovery because Hupu exposes no public delta feed."""
        return await self.discover(request)

    async def health(self) -> ChannelHealth:
        """Return operational health inferred from the last public request."""
        checked_at = self._utc_now()
        if self._last_error == "hupu_rate_limited":
            status = ChannelHealthStatus.RATE_LIMITED
            from datetime import timedelta

            retry_after = checked_at + timedelta(minutes=5)
        elif self._last_error in {"hupu_access_blocked", "hupu_challenge_page"}:
            status = ChannelHealthStatus.BLOCKED
            retry_after = None
        elif self._last_error.endswith("schema_changed"):
            status = ChannelHealthStatus.SCHEMA_CHANGED
            retry_after = None
        elif self._last_error == "hupu_thread_no_public_text":
            # a no-text thread (image-only / deleted-body / joke posts) is a
            # platform answer, not an adapter fault: the channel stays healthy
            status = ChannelHealthStatus.HEALTHY
            retry_after = None
        elif self._last_error:
            status = ChannelHealthStatus.DEGRADED
            retry_after = None
        else:
            status = ChannelHealthStatus.HEALTHY
            retry_after = None
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
            data_freshness_seconds=max(0, freshness) if freshness is not None else None,
        )

    def _metadata_observation(
        self,
        source_ref: str,
        thread: HupuThreadPage,
        page: int,
        *,
        body: str | None = None,
    ) -> SourceObservation:
        metadata = {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "canonical_url": thread.url,
            "page": page,
            "stats": {
                "reply": thread.reply_count,
                "light": thread.light_count,
                "view": thread.view_count,
            },
            "platform": "hupu",
        }
        return SourceObservation(
            id=self._stable_id(
                "observation", source_ref, "metadata",
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
            source_ref=source_ref,
            modality=ObservationModality.METADATA,
            access_depth=AccessDepth.METADATA,
            excerpt=thread.title,
            location="thread_metadata",
            acquisition_method="hupu_public_thread_metadata",
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
                board = str(value.get("board", "")).strip().casefold()
                return page, board
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

    @staticmethod
    def _author_ref(author: str) -> str:
        return HupuChannelAdapter._stable_id("hupu-author", author) if author else ""

    def _utc_now(self) -> datetime:
        value = self._now_factory()
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["HupuChannelAdapter"]
