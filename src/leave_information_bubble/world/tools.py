"""Bounded native-model tools for durable world memory and source acquisition."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import sqlite3
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypeVar, cast

from pydantic import ValidationError

from leave_information_bubble.channels import (
    AcquisitionEntryKind,
    AcquisitionOutcome,
    ChannelAdapter,
    ChannelCapabilityRole,
    DiscoveryBatch,
    HydrationDepth,
    HydrationRequest,
    ObservationBatch,
    QuerySemantics,
    ScanRequest,
    SourceOccurrence,
    TimeFilterPrecision,
)
from leave_information_bubble.models.epistemics import AccessDepth, ObservationModality, SourceObservation
from leave_information_bubble.runtime.errors import AgentError, ErrorCode
from leave_information_bubble.runtime.inquiry_lease import InquiryLeaseStore

from .contracts import CognitiveDelta, MemoryOverview, ObjectKind, ObservationDepth, ObservationInput
from .finalize import wake_mutation_lock
from .graph_contract_text import EVENT_REIFICATION_RULE, IDENTITY_MODEL_SENTENCE
from .graph_patch_contract import graph_patch_parameters_schema
from .materials import (
    BodyEnvelope,
    LegacyBody,
    MaterialPart,
    MaterialPartInput,
    build_body_envelope,
    parse_stored_body,
)
from .preflight import diff_working_graph, inspect_working_graph, staged_item_history
from .proposal import GraphRef, NewInquiryProposal
from .recall import _READ_VIEWS, MemoryBundle, WorldRecall
from .staging import _payload_hash
from .staging import apply_patch as _staging_apply_patch
from .store import DEPTH_LEVELS, WorldStore, observation_id
from .tool_capabilities import (
    render_scan_capabilities,
    scan_schema_contracts,
    supports_targeted_search,
)

_T = TypeVar("_T")

_SCAN_HISTORY_CAP = 500

_PATCH_BATCH_CAP = 20  # bounded graph_patch batch (Core §6.3)
_DIFF_PAGE_CAP = 200  # bounded graph_diff page (plan §7.7)

_NAMES = (
    "memory_recent",
    "memory_search",
    "memory_read",
    "memory_compare",
    "memory_expand",
    "memory_evidence",
    "memory_inquiries",
    "memory_changes",
    "memory_overview",
    "claim_inquiry",
    "release_inquiry",
    "discover_sources",
    "search_sources",
    "open_source",
    "sample_discussion",
    "inspect_media",
    "follow_related",
    "digest_observation",
    "propose_inquiry",
    "log_inquiry_point",
    "graph_patch",
    "graph_inspect",
    "graph_diff",
)
# Fallback adapter ids for schema-only callers.  A live WorldTools instance
# advertises only its registered adapters; this list preserves the historical
# platform-visible schema when no adapters have been supplied at all.
_ADAPTER_IDS = ("bilibili", "nga", "hupu", "public-web")
_SCAN_SURFACE_ROLES = (
    ChannelCapabilityRole.PLATFORM_DISCOVERY,
    ChannelCapabilityRole.ATTENTION_RANKING,
    ChannelCapabilityRole.RECENT_STREAM,
    ChannelCapabilityRole.SEARCH_INDEX,
    ChannelCapabilityRole.NEWS_INDEX,
    ChannelCapabilityRole.CURRENT_INDEX,
    ChannelCapabilityRole.COMMUNITY_STREAM,
)
_DISCOVERY_ENTRY_KINDS = frozenset(
    {
        AcquisitionEntryKind.PLATFORM_SEARCH,
        AcquisitionEntryKind.RANKED_SURFACE,
        AcquisitionEntryKind.BOUNDED_BOARD,
    }
)

# C4 differentiated descriptions (spec §3): each tool states its capability,
# information boundary, and how it differs from adjacent tools without imposing
# a memory-before-acquisition workflow or a search scheduler. Identity reuse and
# changes to remembered cognition still require relevant memory reads. Scan tools close with an
# availability-only sentence naming the registered adapters — no content
# labels, per the platform visibility decision. Descriptions are the model's
# only tool-selection signal, so every claim here matches the facade's actual
# behavior (recall.py / tools.py execute paths).
_DESCRIPTIONS: dict[str, str] = {
    "memory_recent": (
        "Return a bounded, newest-first slice of recent revisable durable judgments "
        "and their subjects and objects, with no query. It is not all memory, "
        "not a relevance search, and an absent item may still exist outside the "
        "returned slice; open inquiries live in memory_inquiries. It can provide "
        "limited temporal orientation after current material makes older context "
        "relevant, but is not a required wake-opening step or an agenda. Unlike memory_search it "
        "returns recent items by time rather than query matches; unlike "
        "memory_changes it is a snapshot, not a diff since a point in time."
    ),
    "memory_search": (
        "Return full-text query matches from prior durable cognition: revisable "
        "judgments, their subjects and objects, and open inquiries. A specific "
        "object, event, or claim in current material can make related remembered "
        "judgments useful for recognizing an update, duplicate, conflict, or "
        "reusable durable identity. It is optional, not a required step "
        "before every external search. Unlike memory_recent (no query, bounded "
        "latest slice) it matches your query text; unlike memory_expand it "
        "returns matches, not a neighborhood graph around chosen objects. "
        "Each matched object carries one candidates entry labeled with how "
        "it matched, ordered most-identity-specific first: identity_alias_exact "
        "(the query text is one of its active identity aliases), canonical_exact "
        "(the query text equals its canonical name; several objects may share a "
        "name), name_usage (the query text is an asserted community or period "
        "usage, carrying the assertion id, qualifiers, time, and evidence "
        "counts), legacy_name (a historical alias, marked "
        "identity_authority=false — never an identity claim), possible_match "
        "(the query text resembles a stored name but is not one), or text_match "
        "(a full-text hit only), plus the object's domain hints for "
        "disambiguation. An empty result does not establish that no "
        "other memory exists. "
        "Structured filters narrow a query: kind, an event-span time window "
        "(time_from/time_to), predicate, has_participants, and the "
        "assertion_count_min/max range on an object's revisable judgments. "
        "mode=count reports per-bucket totals and page_size first, so you can "
        "decide whether paging is worth it; page mode returns bounded pages "
        "and mints next_cursor (valid only within the same wake) — a page "
        "call may pass only the cursor. " + IDENTITY_MODEL_SENTENCE
    ),
    "memory_read": (
        "Return one object's complete current portrait (identity summary plus "
        'three-direction views) — "对象 X 的完整当前画像". The summary is '
        "always present: every name surface (canonical name, active aliases, "
        "removed and legacy forms, asserted community usages), kind and "
        "type_key, the event span, participants, and per-direction counts of "
        "your revisable judgments. The views themselves stay folded until you "
        "request them via views (self_attributes / out_edges / in_edges / "
        "correction_chain); each is bounded and reports its cut. "
        "correction_chain shows how claims were corrected over time: supersede "
        "never deletes, the chain tail is the current cognition. "
        "object_id may be omitted to read the most recently committed object. "
        "Pass assertion_id instead for one judgment's evidence view (its "
        "evidence links plus what it superseded and what replaced it). "
        "Use it after memory_search or memory_expand to inspect a concrete id "
        "you intend to reference or compare. Unlike memory_recent (a latest "
        "slice) and memory_expand (a neighborhood walk) this reads one "
        "referent's full current state. It reads the formal graph; pass "
        "include_working=true to overlay this wake's staged working graph "
        "(graph_patch): staged-only items are marked source=staged, overlays "
        "over formal rows source=merged. For the whole working graph's "
        "blockers, warnings and readiness use graph_inspect."
    ),
    "memory_compare": (
        'Compare two concrete ids side by side — "这两个是不是同一 referent？" '
        "— without deciding. Use it when memory_search returns several "
        "candidates that may name the same referent: the payload lines up the "
        "identity fields (canonical name, active aliases, kind/type_key, "
        "event span, participants, asserted usages, key assertions) or, for "
        "two assertion ids, the signature fields (subject, predicate, "
        "object/literal, event window, epistemic role, evidence, supersede "
        "relations). Scalar fields carry left/right/equal; set fields carry "
        "shared/left_only/right_only splits. It states facts and differences "
        "only — it never says whether to merge; you decide. Copy ids verbatim "
        "from a search hit or an earlier read; an unresolvable id is reported "
        "as missing. It reads only the formal graph — for staged work use "
        "graph_inspect."
    ),
    "memory_expand": (
        "Return a bounded local slice of revisable judgments anchored on one or more "
        "objects — literal claims and object edges alike — plus neighbor "
        "objects reached through those edges. The payload also carries an "
        "ego-graph zoom: direct neighbors with their edge predicates "
        "(ego_neighbors), the subject's current status assertions "
        "(status_assertions), a recent events timeline (event_timeline), the "
        "has_participant edges of an event root (event_edges, with role, "
        "qualifiers and evidence summaries), and the recent events an entity "
        "root participates in (participated_events, sorted event_time_start "
        "DESC, id ASC). participated_events pages through its own "
        "event_limit and the optional before cursor: copy the returned "
        "next_cursor into a later call to page backward; sort_basis "
        "names the order and omitted_counts reports how many qualifying items "
        "each capped section cut, so a cap cut is never silent. Set "
        "include_history to also surface retired (superseded) claims beside "
        "the current ones — the graph as recorded; the default current-only "
        "state keeps retired claims in the memory_read correction chain "
        "view. Use it after a memory map pinpoints a center: map → ego zoom "
        "→ detail. Items outside the applied limit may still exist. It "
        "exposes part of a known center's cognition web and immediate "
        "neighbors, not a complete object history. Unlike memory_search it "
        "takes object_ids, not a query, and walks the graph instead of "
        "matching text."
    ),
    "memory_evidence": (
        "Return one remembered judgment with its bounded evidence-link observation "
        "IDs, evidence roles, depth, source kind, times, and available reliability "
        "or limitation metadata; it does not return source bodies. An empty result "
        "does not establish that no other memory exists. It is relevant when the "
        "basis details of an already-surfaced remembered judgment matter to the "
        "current interpretation or correction; it is not a default prerequisite and "
        "need not be called for every judgment. Unlike memory_search and "
        "memory_expand, which surface judgments, this tool focuses on the evidence "
        "links of one assertion id."
    ),
    "memory_inquiries": (
        "Return a bounded list of open inquiries: unresolved questions retained "
        "from prior cognition. A current stimulus, a continuing Deep pull, or a "
        "known center may make some older unresolved edges relevant, but "
        "it is not a backlog or default startup agenda. Unlike memory_recent and "
        "memory_search, which return remembered judgments, inquiries are open "
        "questions, not answers."
    ),
    "memory_changes": (
        "Report what changed in your memory since a point in time — new "
        "objects, new revisable judgments, new inquiries, resolved inquiries. "
        "When since is omitted, the boundary defaults to the latest proposal attempt; "
        "if no proposal attempt exists, changes are computed from all world audit rows. "
        "since must be a tz-aware ISO-8601 instant (naive values are rejected). "
        "Each returned list remains capped. This is a bounded diff, while memory_recent "
        "is a bounded snapshot of the latest items."
    ),
    "memory_overview": (
        "Return a small, read-only map of what your durable memory currently contains: "
        "active changes, dormant inquiries reactivated by later evidence, direct "
        "cold bridges, and inquiry-local gaps. Each entry names durable memory ids "
        "and reason codes. This is a compact, limited orientation rather than a "
        "complete memory dump. These optional memory flashes are not a task list. "
        "Like memory_inquiries, this is a pure read: it never "
        "demotes, promotes, or reopens an inquiry."
    ),
    "claim_inquiry": (
        "Acquire a time-limited lease on an open or dormant unresolved inquiry so you own it "
        "exclusively. Use when committing to answer that specific inquiry "
        "and to keep other agents from claiming it concurrently. Unlike the "
        "memory_* read tools, this is a coordination action: it changes lease "
        "state and returns a lease token, not knowledge."
    ),
    "release_inquiry": (
        "Release a lease token you previously claimed, returning the inquiry "
        "to claimable state. Use when you finish or abandon an owned inquiry. "
        "Unlike claim_inquiry it only releases; the two form the inquiry "
        "ownership protocol."
    ),
    "discover_sources": (
        "Return candidate source cards from one platform's current surface — a breadth survey "
        "whose query is optional when the registered adapter supports a "
        "targetless scan, and otherwise guides its search. A query anchors the "
        "returned candidates to that topic, so those results do not represent "
        "the platform's whole current surface. Unlike search_sources, which "
        "requires a query, discover may scan without "
        "one; both call the same platform surfaces with the adapter's default "
        "ordering, including relevance-ranked search where supported. Unlike open_source it "
        "returns candidate cards and does not enter one source; a returned candidate "
        "is material, not an assertion. Failures report bounded access or content "
        "limitations rather than silently broadening the request. "
        "Adapter scan capabilities:\n{scan_capabilities}"
    ),
    "search_sources": (
        "Return external candidate source cards matching a specific query. The adapter's "
        "retrieval behavior must support platform search; the query is required. "
        "Unlike discover_sources (a breadth survey, "
        "query optional) this requires a query; both scan the same platform "
        "surfaces with the adapter's default ordering. Unlike memory_search "
        "it searches external platforms, not your memory, and it does not by "
        "itself provide a broad view of the domain. Returned candidates are "
        "material, not assertions; failures expose bounded access or content "
        "limitations. Targeted-search adapters allowed by this schema: "
        "{targeted_adapters}."
    ),
    "open_source": (
        "Open one known source, identified by an observation id returned from "
        "a memory read or scan, at a chosen depth: "
        "preview (metadata and excerpt), full (full-text body), discussion "
        "(comments), media (transcript, or the remaining body text where the "
        "source has no media). A candidate need not "
        "be opened beyond the depth needed for your present understanding or "
        "evidence. Unlike "
        "sample_discussion and inspect_media, which fix the depth, open_source "
        "lets you choose it per call; an already-read dimension returns the "
        "stored card instead of re-fetching. Set refresh=true only with "
        "depth=full when you explicitly need to revalidate stored material. "
        "Failures return access or content limitations; they do not imply that "
        "another depth exists or that the source supports a judgment."
    ),
    "sample_discussion": (
        "Enter the discussion layer (comments and reactions) of one source "
        "identified by an observation id returned from a memory read or scan. "
        "It returns community reaction, opinion, or debate rather than a verified "
        "account of the underlying event. Unlike inspect_media, which reads transcripts and "
        "media, this reads comments; open_source with depth=discussion is the "
        "equivalent choice at the first page — only sample_discussion can "
        "paginate (page)."
    ),
    "inspect_media": (
        "Read the media text of a known source — the video transcript "
        "on bilibili, the thread or page body text elsewhere; no adapter "
        "returns images or OCR. The returned text remains source material, not an "
        "automatically accepted judgment. Unlike sample_discussion, "
        "which reads comments, this reads transcript/media content; "
        "open_source with depth=media is the equivalent explicit choice."
    ),
    "follow_related": (
        "Return related candidate cards around a known source: the bilibili "
        "adapter returns the platform's "
        "related-video recommendations; public-web returns outbound links "
        "from the page; hupu and nga do not support this and will fail with "
        "related_occurrences_unavailable. Unlike open_source, which enters the source itself, this "
        "surfaces the related set as candidate cards, not opened content."
    ),
    "digest_observation": (
        "Accept an already-returned full-text body or non-empty preview/discussion "
        "excerpt and record digest points, "
        "assertion candidates with predicate/literal/epistemic_role/"
        "confidence, and evidence_refs citing exact observation ids. Set material_id by "
        "copying the returned card's digest_material_id verbatim; never use its observation_id. "
        "Optionally record what you read but still do not understand as "
        "unresolved items (what + why); an excerpt that "
        "lacks context must not be presented as unread context or host-certified truth; "
        "it may shape honestly attributed, confidence-calibrated revisable cognition or "
        "remain unresolved. Use "
        "only content already returned by open_source and never to request more "
        "source content. A digest compresses material; it does not require every "
        "candidate or point to become durable cognition. Undigested returned content "
        "may gate finalization. Unlike the "
        "memory_* and acquisition tools, this writes no store rows: it "
        "captures cognition into the graph for the terminal proposal."
    ),
    "propose_inquiry": (
        "Propose a new open inquiry — a worth-deepening question anchored to "
        "one object. The validated proposal is carried in state and "
        "merged into the terminal submission, never written to the store "
        "directly. The subject must be a memory_id of an object already in "
        "memory, or a local_ref you declare in this proposal's new_objects; "
        "questions about undeclared objects are dropped at merge. Unlike "
        "memory_inquiries (reads your open questions) this only proposes a new "
        "question for terminal submission; unlike claim_inquiry it takes no lease and reserves "
        "nothing."
    ),
    "log_inquiry_point": (
        "Record one lightweight inquiry point: something in the input "
        "stream you noticed you do not understand — an unfamiliar meme or "
        "term, a contradiction with what you know, an unexpected community "
        "reaction, or a door a discovery opened. Inquiry points are one-line "
        "runtime traces: they never enter your proposal or durable memory. Unlike "
        "propose_inquiry (a formal cognitive question) and digest "
        "unresolved (a committed gap) this records only a non-durable trace."
    ),
    "graph_patch": (
        "Write to the current wake's durable working graph when you need to "
        "stage new or corrected cognition. Unlike the read-only memory tools "
        "(memory_search / memory_read / memory_compare / memory_expand), this "
        "is the write tool: a batch of object / assertion / inquiry items "
        "that stage for an explicit finalize_graph decision, isolated to this wake and "
        "visible to reads only through memory_read include_working=true. "
        "Use it only for items eligible under the Domain Lens persistence scope; material "
        "being discovered or associated is not by itself eligible to enter the graph. "
        "The exact supported variants are object create/update/drop, assertion "
        "create/update/drop/supersede, and inquiry create/resolve/drop; each "
        "variant has a closed payload schema and fields from another action are "
        "rejected before dispatch. Objects carry one of six kinds — person, "
        "organization, place, event, concept, or entity as the fallback, "
        "never the default — and a type_key as the domain refinement (team, "
        "player, match, tournament, season, version); prefer domain-neutral "
        "type_key words over prefixed ones. Events are time anchors. "
        + EVENT_REIFICATION_RULE
        + " Carry an event's time in event_time_start/end; a momentary "
        "attribute with no durable meaning stays an assertion with event_time, "
        "not an event node. "
        "Every item has its own op_id "
        "idempotency key and returns its own verdict — a replayed op_id with "
        "the same payload replays the original result, a replayed op_id with "
        "a different payload is an explicit conflict. When the occurrence time "
        "of what you stage is known from the material, carry it in the item's "
        "event_time_start / event_time_end fields (UTC ISO); leave them out "
        "when timing is unknown. Creating an object runs "
        "an identity pre-check: an exact canonical/alias candidate returns "
        "needs_identity_resolution with the candidates until you confirm the "
        "new referent is distinct (decision confirm_distinct with the "
        "candidate id and basis) or target an existing id instead. References "
        "must resolve to formal or currently active staged ids; dropped items "
        "cannot be referenced, and staged ids are host-issued (never invent "
        "them). A correction supersedes the target assertion explicitly "
        "(supersedes_ref) — the old judgment stays as history, never "
        "deleted. Re-stating the same subject-predicate-object relation with "
        "the same role and qualifiers over an overlapping or unknown time span "
        "is a revision and must supersede the existing relation; genuinely "
        "non-overlapping recurring episodes may coexist. This is the only way "
        "to change the working graph; formal "
        "commit happens at finalize."
    ),
    "graph_inspect": (
        "Report the current working graph's state and what stands between it "
        'and a publishable finalize — "当前工作图离可发布还缺什么？". '
        "Returns per-kind active/abandoned/finalized statistics, a bounded "
        "staged-item summary, and two separated finding tiers: blockers "
        "(dangling or abandoned references, unresolved identity collisions, "
        "exact assertion duplicates, overlapping parallel relations, supersede "
        "cycles, dangling evidence "
        "refs, isolated new non-provisional objects, unanchored new events) "
        "and warnings (missing evidence, duplicate inquiries) that never "
        "block. Structural readiness cannot infer an event the Agent omitted "
        "by choosing a direct edge; make the event-versus-direct decision "
        "before patching and review that abstraction yourself. "
        "readiness=blocked until every blocker is gone — fix by "
        "patching (add a relation, drop the item, confirm distinct) and "
        "re-inspecting. The report is read from durable staging, so on "
        "resume it is the authoritative working-graph view. Pass item_id "
        "for one staged item's current state plus its patch history. Unlike "
        "memory_read (one referent's formal portrait) this is the whole "
        "wake's uncommitted surface."
    ),
    "graph_diff": (
        "Show what this wake changed relative to the formal graph — "
        '"本 wake 相对正式图改了什么？" — as bounded before/after entries: '
        "created objects, formal-target updates (formal row before, staged "
        "row after), supersedes (the superseded assertion before, the "
        "correction after), drops, and inquiries. Every response is marked "
        "published=false: nothing here is committed until finalize_graph. "
        "Pass limit/offset to page; by_action/by_kind summarize the page "
        "population. Use it when you need to review the delta before "
        "finalizing; unlike memory_changes (committed formal history), this "
        "tool always focuses on the current wake's uncommitted staging."
    ),
}
_READ_TOOLS = frozenset(
    (
        "memory_recent",
        "memory_search",
        "memory_read",
        "memory_compare",
        "memory_expand",
        "memory_evidence",
        "memory_inquiries",
    )
)
_SCANS = {"discover_sources", "search_sources"}
_DEPTHS = {
    "sample_discussion": HydrationDepth.REACTIONS,
    "inspect_media": HydrationDepth.MEDIA_TEXT,
    "follow_related": HydrationDepth.TEXT,
}
_ACQUISITION_TOOLS = frozenset({*_SCANS, *_DEPTHS, "open_source"})
# Adapters whose hydrate fills discovered_occurrences (public-web emits the
# page's outbound links), so follow_related can still answer from a plain
# hydrate even without a native related(). Adapters without related() that
# are not listed here short-circuit before any hydrate: their hydrate never
# fills discovered_occurrences, so the old fallback burned a full TEXT
# hydrate for a guaranteed related_occurrences_unavailable (audit item 2).
_RELATED_FROM_HYDRATE = frozenset({"public-web"})
_CARD_CAP = 5
_READ_CAP = 15  # agent-initiated recall window (spec B3); injection stays at _CARD_CAP
# (except inquiries, see _INQUIRY_CAP).
_INQUIRY_CAP = 8  # injection widening for the inquiries blind-zone experiment
# (bootstrap 8 / per-round 5); stepped up gradually.
_NAV_EXCERPT = 150  # display-level navigation excerpt (spec A1); storage stays 800
_EXPAND_CAP = 30  # graph BFS imagery needs a wider window than the 5-card tool cap
_EVENT_LIMIT_DEFAULT = 5  # entity-centric participated-events page size
_FULL_BODY_CAP = 32_000  # tools-layer re-cap after cross-observation body assembly
_LIMITATIONS_CARD_MAX_ITEMS = 8
_LIMITATIONS_ITEM_MAX_CHARS = 160
_LIMITATIONS_CARD_MAX_CHARS = 640
_LIMITATIONS_TRUNCATED = "limitations_truncated"

# Failure semantics live in one table so call sites report observations rather
# than each inventing a next-step policy.
_FAILURE_FACTS: dict[str, tuple[bool, str, str]] = {
    ErrorCode.TOOL_TRANSIENT.value: (
        True,
        "provider_or_network_conditions_change",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    ErrorCode.SOURCE_UNAVAILABLE.value: (
        True,
        "source_access_or_provider_availability_changes",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    ErrorCode.INPUT_INVALID.value: (
        False,
        "request_arguments_change",
        AcquisitionOutcome.UNSUPPORTED.value,
    ),
    ErrorCode.TOOL_PERMANENT.value: (
        False,
        "adapter_capability_or_request_changes",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    "invalid_arguments": (
        False,
        "request_arguments_change",
        AcquisitionOutcome.UNSUPPORTED.value,
    ),
    "wake_deadline_exceeded": (
        False,
        "new_wake_dispatch_cutoff",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    "storage_unavailable": (
        True,
        "storage_or_lock_conditions_change",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    "tool_timeout": (
        True,
        "provider_response_before_configured_timeout",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    "unknown_adapter": (
        True,
        "registered_adapter_set_changes",
        AcquisitionOutcome.UNSUPPORTED.value,
    ),
    "unknown_observation": (
        True,
        "observation_exists_in_world_store",
        AcquisitionOutcome.UNSUPPORTED.value,
    ),
    "adapter_provenance_mismatch": (
        True,
        "adapter_response_identity_changes",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    "related_occurrences_unavailable": (
        True,
        "source_recommendations_or_adapter_capability_changes",
        AcquisitionOutcome.EMPTY.value,
    ),
    "no_content_observed": (
        True,
        "source_content_or_public_access_changes",
        AcquisitionOutcome.EMPTY.value,
    ),
    "no_full_content": (
        True,
        "source_content_or_public_access_changes",
        AcquisitionOutcome.EMPTY.value,
    ),
    "stored_body_invalid": (
        True,
        "stored_body_is_repaired_or_source_is_refreshed",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    "material_write_failed": (
        True,
        "world_store_write_succeeds",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    "material_replay_missing": (
        True,
        "matching_material_transaction_exists",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
    "material_persisted_missing": (
        True,
        "persisted_observation_is_readable",
        AcquisitionOutcome.UNAVAILABLE.value,
    ),
}

_TARGET_DEPTH = {
    "sample_discussion": ObservationDepth.DISCUSSION,
    "inspect_media": ObservationDepth.MEDIA,
}
# The inquiry kinds propose_inquiry accepts; kept in sync with the
# NewInquiryProposal.kind Literal and the schema enum.
_INQUIRY_KINDS = ("factual", "semantic", "stateful")
# "full" is a depth FLAG, not an ObservationDepth rung; its already_opened
# comparison is body presence in observation_bodies (P2 Task 3): a stored
# body short-circuits _open_full to a zero-network serve.
_TARGET_FULL = {"open_source"}
_OPEN_SOURCE_TARGETS = {
    "preview": ObservationDepth.CONTENT,
    "discussion": ObservationDepth.DISCUSSION,
    "media": ObservationDepth.MEDIA,
}
_OPEN_SOURCE_HYDRATE = {
    "preview": HydrationDepth.TEXT,
    "full": HydrationDepth.TEXT,
    "discussion": HydrationDepth.REACTIONS,
    "media": HydrationDepth.MEDIA_TEXT,
}


def digest_material_id(observation_id: str, material_kind: str, text: str) -> str:
    """Return the immutable Digest v2 identity for exact model-visible text."""
    if not observation_id or material_kind not in {"full", "excerpt"} or not text:
        raise ValueError("digest material identity fields must be non-empty")
    payload = f"{observation_id}\0{material_kind}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()


# Channel-wide engagement vocabulary the bilibili adapter already normalizes
# into (bilibili.py _normalized_engagement); hydrated observations carry the
# same counters under the raw "stats" keys, mapped here so discovery and
# hydrate cards speak one vocabulary.
_STATS_TO_ENGAGEMENT = {
    "play": "views",
    "like": "likes",
    "reply": "comments",
    "danmaku": "realtime_reactions",
    "favorite": "saves",
    "coin": "supports",
    "share": "shares",
}
# Sub-content modalities keep their own rows under content-derived sub-ids
# (task T13 fix F2): a comment, a transcript segment, or a danmaku item is
# content *under* the source, not a property of the source row itself, so
# the main row never absorbs them and re-fetched identical content hits the
# same sub-row (content-hash idempotency).
_SUB_CONTENT_KIND = {
    ObservationModality.COMMENT: "comment",
    ObservationModality.DANMAKU: "danmaku",
    ObservationModality.TRANSCRIPT: "transcript",
    ObservationModality.IMAGE: "media",
    ObservationModality.OCR: "media",
}
# Modality -> observed dimension recorded in every row a hydration writes
# (task T13 fix F1). already_opened short-circuits per dimension, never
# across them: reading subtitles must not count as reading comments.
_OBSERVED_KIND = {
    ObservationModality.METADATA: "metadata",
    ObservationModality.DOCUMENT_TEXT: "content",
    ObservationModality.COMMENT: "comments",
    ObservationModality.DANMAKU: "danmaku",
    ObservationModality.TRANSCRIPT: "transcript",
    ObservationModality.IMAGE: "media",
    ObservationModality.OCR: "media",
}
# Observed dimensions that already cover one tool call (fix F1): the linear
# depth ladder cannot order orthogonal dimensions (DISCUSSION vs MEDIA), so
# each tool only short-circuits when its own dimension was read before.
_OBSERVED_REQUIREMENTS = {
    "open_source:discussion": frozenset({"comments"}),
    "open_source:media": frozenset({"transcript", "media"}),
    "sample_discussion": frozenset({"comments"}),
    "inspect_media": frozenset({"transcript", "media"}),
}


class WorldTools:
    """Expose one typed, bounded facade over world memory and channel adapters."""

    def __init__(
        self,
        *,
        store: WorldStore,
        adapters: Mapping[str, ChannelAdapter],
        leases: InquiryLeaseStore | None = None,
        digested_ids: set[str] | None = None,
        domain: str = "",
        thread_id: str = "",
        wake_id: str | None = None,
        mode: str = "",
        discovery_timeout_seconds: float = 45.0,
        hydration_timeout_seconds: float = 90.0,
        full_hydration_timeout_seconds: float = 900.0,
        wake_deadline_at: float | None = None,
        closed_wake_guard: bool = False,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        """Build a facade with named adapters and optional runtime inquiry leases.

        Args:
            store: The durable world store validated against on every digest.
            adapters: Named channel adapters for source acquisition.
            leases: Optional runtime inquiry leases for claim/release tools.
            digested_ids: Optional shared set of already-digested observation
                ids (the graph supplies its own in the digest loop). The tool
                checks membership to short-circuit repeat digests and records
                each successfully digested id into this same set. When None
                no dedup happens: every digest call validates and echoes.
            domain: Domain key forwarded on every hydration request arguments
                so channel adapters can scope per-domain ledger bookkeeping
                (e.g. the subtitle-reliability blacklist keyed by domain).
                Passed through verbatim, including the empty default.
            thread_id: Run/thread key stamped on every curiosity-log row so
                the shared append-only log stays filterable per thread.
                Passed through verbatim, including the empty default.
            wake_id: Durable working-graph owner and cursor scope. Defaults
                to ``thread_id`` for legacy callers; Graph Shell passes the
                separately minted wake identity explicitly.
            mode: Wake posture. Broad discovery receives a transparent
                24-hour default window; Deep and targeted search stay unfiltered.
            discovery_timeout_seconds: Whole-call limit for source scans.
            hydration_timeout_seconds: Whole-call limit for ordinary hydrates.
            full_hydration_timeout_seconds: Whole-call limit for full material.
            wake_deadline_at: Optional shared monotonic cutoff for this wake.
            closed_wake_guard: Serialize graph_patch with finalize and reject
                new work after a durable finalize receipt. Graph Shell enables
                this; legacy/G5a raw staging keeps its reviewed semantics.
            progress: Optional best-effort operational event callback.

        """
        self._store = store
        self._recall = WorldRecall(store)
        self._adapters = dict(adapters)
        self._leases = leases
        self._digested_ids = digested_ids
        self._scan_observation_ids: list[str] = []
        self._scan_history_truncated = False
        # The graph sets this transient per-wake allow-list before executing a
        # digest. It distinguishes an observation merely present in the long-
        # term store from text actually returned to this agent transcript.
        self._digest_context: dict[str, dict[str, object]] | None = None
        self._domain = domain
        self._thread_id = thread_id
        self._wake_id = thread_id if wake_id is None else wake_id
        self._mode = mode
        self._discovery_timeout_seconds = max(0.01, discovery_timeout_seconds)
        self._hydration_timeout_seconds = max(0.01, hydration_timeout_seconds)
        self._full_hydration_timeout_seconds = max(0.01, full_hydration_timeout_seconds)
        self._wake_deadline_at = wake_deadline_at
        self._closed_wake_guard = closed_wake_guard
        self._progress = progress

    def set_digest_context(self, materials: Mapping[str, Mapping[str, object]] | None) -> None:
        """Set exact material identities and evidence allow-lists for this wake.

        ``None`` clears a graph-local context. Direct facade callers then
        resolve an exact material from the local store; graph calls remain
        constrained to the materials returned in the current wake. A graph
        supplies each exact model-visible material, its primary id, and its
        allowed evidence refs.
        """
        self._digest_context = (
            {str(key): dict(value) for key, value in materials.items()} if materials is not None else None
        )

    def set_scan_history(self, ids: Sequence[str]) -> None:
        """Restore the ordered, bounded source identities observed in this wake."""
        distinct = list(dict.fromkeys(str(identifier) for identifier in ids if str(identifier)))
        self._scan_history_truncated = len(distinct) > _SCAN_HISTORY_CAP
        self._scan_observation_ids = distinct[-_SCAN_HISTORY_CAP:]

    def set_scan_history_truncated(self, truncated: bool) -> None:
        """Restore whether an earlier checkpoint already exceeded the history bound."""
        self._scan_history_truncated = bool(truncated) or self._scan_history_truncated

    def scan_history(self) -> list[str]:
        """Return a copy of the ordered source identities retained for scan novelty."""
        return list(self._scan_observation_ids)

    def scan_history_truncated(self) -> bool:
        """Return whether distinct scan identities have exceeded the retained bound."""
        return self._scan_history_truncated

    def schemas(self, *, include_memory_overview: bool = True) -> list[dict[str, Any]]:
        """Return complete OpenAI-compatible schemas for exactly the supported tools.

        Every description is differentiated (spec §3 C4): purpose, when to
        use the tool, and how it differs from adjacent tools, so the model can
        pick the right tool at the right moment. The external-acquisition
        tools do not impose an order between memory and acquisition.
        """
        adapter_ids = tuple(self._adapters) or _ADAPTER_IDS
        capability_adapters = self._adapters or {
            adapter_id: cast(ChannelAdapter, object()) for adapter_id in adapter_ids
        }
        scan_capabilities = render_scan_capabilities(capability_adapters)
        scan_contracts = scan_schema_contracts(self._adapters) if self._adapters else None
        targeted_adapters = (
            ", ".join(
                adapter_id
                for adapter_id in adapter_ids
                if scan_contracts is None
                or scan_contracts.get(adapter_id, {}).get("classified") is not True
                or scan_contracts.get(adapter_id, {}).get("targeted_search") is True
            )
            or "none"
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        _DESCRIPTIONS[name]
                        .replace("{scan_capabilities}", scan_capabilities)
                        .replace("{targeted_adapters}", targeted_adapters)
                        if name in _SCANS
                        else _DESCRIPTIONS[name]
                    ),
                    "parameters": _parameters(
                        name,
                        adapter_ids=adapter_ids,
                        scan_contracts=scan_contracts,
                    ),
                },
            }
            for name in _NAMES
            if include_memory_overview or name != "memory_overview"
        ]

    async def execute(self, name: str, arguments: Mapping[str, Any], call_id: str) -> dict[str, Any]:
        """Run one model call and return bounded JSON data or a typed limitation."""
        if name not in _NAMES:
            return _limited(f"unknown_tool:{name}")
        try:
            if name == "memory_changes":
                since = arguments.get("since")
                try:
                    parsed = (
                        None if since is None else datetime.fromisoformat(str(since).replace("Z", "+00:00"))
                    )
                except ValueError:
                    return _limited("invalid_arguments")
                if parsed is not None and parsed.tzinfo is None:
                    # A naive instant is interpreted in the server's local zone
                    # on astimezone, silently shifting the change window by the
                    # UTC offset; the schema advertises tz-aware ISO-8601, so
                    # reject the ambiguity at the boundary (memory_search's
                    # time_from/time_to policy).
                    return _limited("invalid_arguments")
                limit = _read_limit(arguments, 15)
                changes = self._recall.changes(since=parsed, limit=limit)
                return _memory(
                    changes,
                    scope={
                        "since": changes.get("since"),
                        "defaulted": parsed is None,
                    },
                    limit=limit,
                )
            if name == "memory_overview":
                raw_as_of = arguments.get("as_of")
                if raw_as_of is None:
                    moment = None
                else:
                    moment = datetime.fromisoformat(str(raw_as_of).replace("Z", "+00:00"))
                    if moment.tzinfo is None:
                        return _limited("invalid_arguments")
                overview_limit = min(int(arguments.get("limit", 3)), 3)
                return _memory_overview(
                    self._recall.overview(as_of=moment, limit=overview_limit),
                    limit=overview_limit,
                    requested_as_of=moment,
                )
            if name == "memory_search":
                return self._search(arguments, call_id)
            if name == "memory_expand":
                return self._expand(arguments)
            if name == "graph_patch":
                return self._graph_patch(arguments)
            if name == "graph_inspect":
                return self._graph_inspect(arguments)
            if name == "graph_diff":
                return self._graph_diff(arguments)
            if name in _READ_TOOLS:
                # bootstrap call ids keep the 5-card injection budget; agent reads
                # route through the wider _read_limit window
                injection = call_id.startswith("bootstrap-")
                bundle = _read(self._recall, name, arguments, injection, self._wake_id)
                return _memory(
                    bundle,
                    scope=_memory_scope(name, arguments),
                    limit=_memory_limit(name, arguments, injection),
                    requested_ids=_memory_requested_ids(name, arguments),
                    ego=(name == "memory_expand"),
                )
            if name == "claim_inquiry":
                return self._claim(arguments)
            if name == "release_inquiry":
                return self._release(arguments)
            if name == "digest_observation":
                return self._digest(arguments)
            if name == "propose_inquiry":
                return self._propose_inquiry(arguments)
            if name == "log_inquiry_point":
                return self._log_inquiry_point(arguments, call_id)
            return await self._execute_acquisition(name, arguments, call_id)
        except (TypeError, ValueError, ValidationError):
            if name in _ACQUISITION_TOOLS:
                return _operational_failure(
                    code="invalid_arguments",
                    stage=_acquisition_stage(name, arguments),
                    attempts=0,
                )
            return _limited("invalid_arguments")
        except sqlite3.OperationalError as error:
            # Storage faults (most commonly ``database is locked`` while
            # another process holds the writer) are typed, not crashes: the
            # Agent keeps control of the wake and can retry or wind down.
            if name in _ACQUISITION_TOOLS:
                return _operational_failure(
                    code="storage_unavailable",
                    stage=_acquisition_stage(name, arguments),
                    attempts=0,
                    limitations=[f"storage_unavailable: {type(error).__name__}"],
                )
            return _limited(f"storage_unavailable: {type(error).__name__}")
        except sqlite3.IntegrityError as error:
            # Constraint violations are contract/program errors, not storage
            # faults: report the typed name without pretending storage broke.
            # (OperationalError and IntegrityError are siblings, not a
            # subclass pair, so the narrower catch is deliberate.)
            return _limited(f"constraint_violation: {type(error).__name__}")

    def _graph_patch(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Apply one bounded batch of graph_patch items to this wake's working graph.

        Each item carries its own op_id idempotency key and verdict; a
        replayed op_id with an identical payload returns the original result,
        a replayed op_id with a different payload is an explicit conflict
        (Core §6.2). Staged ids are host-issued. The identity pre-check runs
        on create; a candidate hit returns needs_identity_resolution unless
        the item carries a confirm_distinct decision (Core §6.4).
        """
        items = arguments.get("items")
        if not isinstance(items, list) or not items or len(items) > _PATCH_BATCH_CAP:
            return _limited("invalid_arguments")
        if not all(isinstance(item, dict) for item in items):
            return _limited("invalid_arguments")
        op_ids = [str(item.get("op_id") or "") for item in items]

        def apply_batch() -> tuple[list[dict[str, Any]], set[str]]:
            with self._store.read_connection() as connection:
                receipts = {
                    str(row["op_id"]): row
                    for row in connection.execute(
                        "SELECT op_id, payload_hash, result_json "
                        "FROM staged_patch_receipts WHERE wake_id = ?",
                        (self._wake_id,),
                    ).fetchall()
                }
                closed = (
                    self._closed_wake_guard
                    and connection.execute(
                        "SELECT 1 FROM finalize_receipts WHERE wake_id = ?",
                        (self._wake_id,),
                    ).fetchone()
                    is not None
                )
            replayed = {op_id for op_id in op_ids if op_id in receipts}
            if not closed:
                return _staging_apply_patch(self._store, self._wake_id, items), replayed
            closed_results: list[dict[str, Any]] = []
            for item, op_id in zip(items, op_ids, strict=True):
                if not op_id:
                    closed_results.append({"op_id": "", "status": "rejected", "error_code": "missing_op_id"})
                    continue
                receipt = receipts.get(op_id)
                if receipt is not None:
                    if receipt["payload_hash"] != _payload_hash(item):
                        closed_results.append(
                            {
                                "op_id": op_id,
                                "status": "conflict",
                                "error_code": "op_id_reused",
                                "message": (f"op_id {op_id} already applied with a different payload"),
                            }
                        )
                    else:
                        closed_results.append(dict(json.loads(receipt["result_json"])))
                    continue
                closed_results.append(
                    {
                        "op_id": op_id,
                        "status": "rejected",
                        "error_code": "wake_closed",
                        "message": (f"wake {self._wake_id} already has a durable finalize receipt"),
                    }
                )
            return closed_results, replayed

        if self._closed_wake_guard:
            with wake_mutation_lock(self._store, self._wake_id):
                results, replayed_ops = apply_batch()
        else:
            results, replayed_ops = apply_batch()
        results = [
            {
                **result,
                **(
                    {"replayed": True}
                    if op_ids[index] in replayed_ops and result.get("status") != "conflict"
                    else {}
                ),
            }
            for index, result in enumerate(results)
        ]
        ok_count = sum(1 for result in results if result.get("status") == "ok")
        replay_count = sum(bool(result.get("replayed")) for result in results)
        applied_count = sum(
            1 for result in results if result.get("status") == "ok" and not result.get("replayed")
        )
        problematic_count = len(results) - ok_count
        outcome = "success" if problematic_count == 0 else "partial" if ok_count else "rejected"
        return {
            "ok": True,
            "payload": {"results": results},
            "outcome": outcome,
            "counts": {
                "applied": applied_count,
                "replayed": replay_count,
                "problematic": problematic_count,
            },
            "scope": {
                "wake_id": self._wake_id,
                "item_count": len(results),
                "replay_count": replay_count,
            },
            "summary": f"Applied {ok_count} of {len(results)} patch items to the working graph.",
        }

    def _graph_inspect(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Report working-graph state, blockers, warnings, and readiness.

        Read directly from durable staging (survives restarts), so on resume
        it is the authoritative working-graph view (plan §7.6). Pass
        ``item_id`` for one staged item's current state plus patch history.
        """
        item_id = str(arguments.get("item_id") or "").strip()
        if item_id:
            detail = staged_item_history(self._store, self._wake_id, item_id)
            if detail is None:
                return _limited("invalid_arguments")
            report = inspect_working_graph(self._store, self._wake_id)
            payload = {
                "wake_id": report["wake_id"],
                "readiness": report["readiness"],
                "stats": report["stats"],
                "active_total": report["active_total"],
                "item": detail,
            }
            summary = (
                f"Staged item {item_id}: {detail['item'].get('status', 'unknown')}, "
                f"{detail['ops_total']} patch op(s); working graph {report['readiness']}."
            )
            return {"ok": True, "payload": payload, "scope": {"wake_id": self._wake_id}, "summary": summary}
        report = inspect_working_graph(self._store, self._wake_id)
        scope = {"wake_id": self._wake_id, "readiness": report["readiness"]}
        summary = (
            f"Working graph: {report['active_total']} active item(s), "
            f"{len(report['blockers'])} blocker(s), {len(report['warnings'])} warning(s) — "
            f"{'ready' if report['readiness'] == 'ready' else 'not ready'} to finalize."
        )
        return {"ok": True, "payload": report, "scope": scope, "summary": summary}

    def _graph_diff(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Page this wake's staged changes relative to the formal graph.

        Entries carry formal before / staged after; every response is marked
        unpublished (plan §7.7). Committed changes belong to memory_changes.
        """
        diff = diff_working_graph(
            self._store,
            self._wake_id,
            limit=int(arguments.get("limit", 50)),
            offset=int(arguments.get("offset", 0)),
        )
        summary = (
            f"{diff['summary']['total']} staged change(s): "
            + ", ".join(f"{action} {count}" for action, count in diff["summary"]["by_action"].items())
            + " — not yet published (finalize_graph commits them)."
        )
        return {
            "ok": True,
            "payload": diff,
            "scope": {"wake_id": self._wake_id, "published": False},
            "summary": summary,
        }

    def _search(self, arguments: Mapping[str, Any], call_id: str) -> dict[str, Any]:
        """Run memory_search page/count with wake-bound cursors (design §4.2).

        Page mode serves one bounded page and mints a thread-bound
        ``next_cursor`` when more rows follow; count mode reports per-bucket
        totals and the page size instead of rows. A malformed, foreign, or
        forged cursor degrades to the first page of the current call's query,
        flagged ``scope.cursor_status = "invalid_cursor"`` (P14); a
        cursor-only call whose token is dead fails closed with a typed
        limitation — there is no query to fall back to. A page resumed
        through a valid cursor pages against the cursor's stored query and
        filters, so a stale or re-argumented call cannot silently page a
        different query.
        """
        limit = _memory_limit("memory_search", arguments, call_id.startswith("bootstrap-"))
        mode = str(arguments.get("mode", "page"))
        if mode not in ("page", "count"):
            return _limited("invalid_arguments")
        cursor_raw = arguments.get("cursor")
        if cursor_raw is not None and mode == "count":
            return _limited("invalid_arguments")
        filters = _search_filters(arguments)
        if filters is None:
            return _limited("invalid_arguments")
        query = str(arguments.get("query", ""))[:240]
        if mode == "count":
            counts = self._recall.search_count(
                query,
                kind=filters.get("kind"),
                time_from=_published(filters.get("time_from")),
                time_to=_published(filters.get("time_to")),
                predicate=filters.get("predicate"),
                has_participants=filters.get("has_participants"),
                assertion_count_min=filters.get("assertion_count_min"),
                assertion_count_max=filters.get("assertion_count_max"),
            )
            return _search_counts(
                counts,
                limit=limit,
                scope=_search_scope(query, mode, filters, offset=0, cursor_status=None),
                query=query,
            )
        offset = 0
        cursor_status: str | None = None
        if cursor_raw is not None:
            decoded = _decode_search_cursor(str(cursor_raw)[:2048], self._wake_id)
            if decoded is None:
                if not query:
                    # a cursor-only call whose token is dead has nothing to
                    # fall back to: fail closed with a typed limitation
                    return _limited("invalid_cursor")
                cursor_status = "invalid_cursor"
            else:
                limit = max(1, min(int(decoded.get("limit", limit)), _READ_CAP))
                offset = max(0, min(int(decoded.get("offset", 0)), _SEARCH_OFFSET_LIMIT))
                query = str(decoded.get("query", query))[:240]
                filters = {key: decoded[key] for key in _SEARCH_CURSOR_KEYS if key in decoded}
        bundle = self._recall.search(
            query,
            limit=limit,
            kind=filters.get("kind"),
            time_from=_published(filters.get("time_from")),
            time_to=_published(filters.get("time_to")),
            predicate=filters.get("predicate"),
            has_participants=filters.get("has_participants"),
            assertion_count_min=filters.get("assertion_count_min"),
            assertion_count_max=filters.get("assertion_count_max"),
            offset=offset,
        )
        envelope = _memory(
            bundle,
            scope=_search_scope(query, "page", filters, offset=offset, cursor_status=cursor_status),
            limit=limit,
            requested_ids=_memory_requested_ids("memory_search", arguments),
        )
        if bundle.truncated:
            envelope["next_cursor"] = _encode_search_cursor(
                self._wake_id,
                {**filters, "query": query, "offset": offset + limit, "limit": limit},
            )
        return envelope

    def _expand(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run memory_expand with wake-bound event cursors (design §4.5).

        The envelope-level ``next_cursor`` is a thread-bound token; the
        payload keeps the recall layer's raw ``event_next_cursor`` position
        marker. A malformed, foreign, or forged cursor degrades to the first
        page of the current call's object_ids, flagged
        ``scope.cursor_status = "invalid_cursor"`` (P14); a cursor-only call
        whose token is dead fails closed with a typed limitation — there is
        no object set to fall back to. ``include_history`` passes through to
        the recall layer's historical-state switch.
        """
        limit = _memory_limit("memory_expand", arguments, False)
        object_ids = list(
            dict.fromkeys(str(value)[:160] for value in arguments.get("object_ids") or [] if value)
        )
        roots = object_ids[:_CARD_CAP]
        cursor_status: str | None = None
        before: str | None = None
        raw_before = arguments.get("before")
        if raw_before is not None:
            decoded = _decode_event_cursor(str(raw_before)[:2048], self._wake_id)
            if decoded is None:
                if not roots:
                    # a cursor-only call whose token is dead has nothing to
                    # fall back to: fail closed with a typed limitation
                    return _limited("invalid_cursor")
                cursor_status = "invalid_cursor"
            else:
                before = json.dumps({"t": decoded[0], "id": decoded[1]})
        bundle = self._recall.expand(
            roots,
            depth=min(int(arguments.get("depth", 1)), _CARD_CAP),
            limit=limit,
            event_limit=min(int(arguments.get("event_limit", _EVENT_LIMIT_DEFAULT)), _EXPAND_CAP),
            before=before,
            include_history=bool(arguments.get("include_history", False)),
        )
        envelope = _memory(
            bundle,
            scope=_memory_scope("memory_expand", arguments),
            limit=limit,
            requested_ids=_memory_requested_ids("memory_expand", arguments),
            ego=True,
        )
        if cursor_status is not None:
            envelope["scope"]["cursor_status"] = cursor_status
        if bundle.event_next_cursor is not None:
            raw = json.loads(bundle.event_next_cursor)
            envelope["next_cursor"] = _encode_event_cursor(self._wake_id, str(raw["t"]), str(raw["id"]))
        return envelope

    async def _execute_acquisition(
        self,
        name: str,
        arguments: Mapping[str, Any],
        call_id: str,
    ) -> dict[str, Any]:
        """Bound one external acquisition and expose its operational outcome."""
        configured = self._acquisition_timeout(name, arguments)
        remaining = None if self._wake_deadline_at is None else self._wake_deadline_at - time.monotonic()
        if remaining is not None and remaining <= 0:
            return _tool_stopped(
                code="wake_deadline_exceeded",
                stage="before_dispatch",
                attempts=0,
            )
        # The wake cutoff governs dispatch only. Once a call has started, its
        # own configured timeout settles it; the wake cutoff never cancels an
        # in-flight source request.
        timeout = configured
        attempts = [0]
        started = time.monotonic()
        self._emit_progress(
            "tool_started",
            tool=name,
            call_id=call_id,
            adapter=str(arguments.get("adapter", "")),
            observation_id=str(arguments.get("observation_id", "")),
            timeout_seconds=round(timeout, 3),
        )
        try:
            async with asyncio.timeout(timeout):
                result = await self._acquire(name, arguments, call_id, attempts)
        except TimeoutError:
            elapsed = time.monotonic() - started
            self._emit_progress(
                "tool_timed_out",
                tool=name,
                call_id=call_id,
                elapsed_seconds=round(elapsed, 3),
                code="tool_timeout",
            )
            return _tool_stopped(
                code="tool_timeout",
                stage=_acquisition_stage(name, arguments),
                attempts=max(attempts[0], 1),
                elapsed_seconds=elapsed,
            )
        elapsed = time.monotonic() - started
        self._emit_progress(
            "tool_completed",
            tool=name,
            call_id=call_id,
            ok=bool(result.get("ok")),
            elapsed_seconds=round(elapsed, 3),
        )
        return result

    def _acquisition_timeout(self, name: str, arguments: Mapping[str, Any]) -> float:
        """Return the configured whole-call limit for one acquisition."""
        if name in _SCANS:
            return self._discovery_timeout_seconds
        if name == "open_source" and str(arguments.get("depth", "preview")) == "full":
            return self._full_hydration_timeout_seconds
        return self._hydration_timeout_seconds

    def _emit_progress(self, event: str, **details: object) -> None:
        """Publish one best-effort event without coupling tools to a UI."""
        if self._progress is None:
            return
        try:
            self._progress({"component": "world_tool", "event": event, **details})
        except Exception:  # noqa: BLE001 - observability must not break cognition
            return

    def _claim(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self._leases is None:
            return _inquiry_failure(
                "inquiry_leases_unavailable",
                field="inquiry_id",
                same_request_can_change=True,
                change_condition="runtime_lease_store_is_available",
            )
        inquiry_id = str(arguments.get("inquiry_id", "")).strip()
        owner_id = str(arguments.get("owner_id", "")).strip()
        if not inquiry_id or not owner_id:
            return _inquiry_failure(
                "invalid_lease_field",
                field="inquiry_id" if not inquiry_id else "owner_id",
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        try:
            ttl_seconds = int(arguments.get("ttl_seconds", 900))
        except (TypeError, ValueError):
            ttl_seconds = 0
        if ttl_seconds <= 0:
            return _inquiry_failure(
                "invalid_lease_field",
                field="ttl_seconds",
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        with self._store.read_connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM inquiries WHERE id = ? AND status IN ('open', 'dormant')",
                (inquiry_id,),
            ).fetchone()
        if exists is None:
            return _inquiry_failure(
                "unknown_or_closed_inquiry_id",
                field="inquiry_id",
                same_request_can_change=True,
                change_condition="an_open_or_dormant_inquiry_with_this_id_exists",
            )
        lease = self._leases.claim(
            inquiry_id,
            owner_id,
            timedelta(seconds=ttl_seconds),
        )
        if lease is None:
            return _inquiry_failure(
                "inquiry_lease_occupied",
                field="inquiry_id",
                same_request_can_change=True,
                change_condition="current_lease_expires_or_is_released",
            )
        return {"ok": True, "lease": lease.model_dump(mode="json"), "status": "claimed"}

    def _release(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self._leases is None:
            return _inquiry_failure(
                "inquiry_leases_unavailable",
                field="lease_token",
                same_request_can_change=True,
                change_condition="runtime_lease_store_is_available",
            )
        released = self._leases.release(str(arguments.get("lease_token", "")))
        if not released:
            return _inquiry_failure(
                "invalid_lease_token",
                field="lease_token",
                same_request_can_change=False,
                change_condition="a_valid_active_lease_token_is_supplied",
            )
        return {"ok": True, "released": True, "status": "released"}

    def _digest(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Record a model-side digest of one exact material without writing to the store.

        Backend validation follows the verify-then-cite discipline: the
        observation id must exist and must carry either a stored full body or
        a non-empty returned excerpt; every evidence ref must exist. This
        accepts both C2 input kinds while refusing empty navigation echoes.
        Success echoes the exact ids back unchanged plus the digest content.
        The digest itself is ephemeral — the graph captures it into state (P2
        Task 5); this tool never writes to the world store.

        Idempotency: when a shared ``digested_ids`` set was injected at
        construction (the graph supplies its own in Task 5), an id already
        present short-circuits the call with ``already_digested: True`` and
        each successfully digested id is recorded into that same set. With
        no injected set (None) no dedup happens.

        Args:
            arguments: Raw tool call arguments (observation_id, points,
                assertion_candidates, evidence_refs, optional unresolved).

        Returns:
            The digest payload echoing the exact ids, or a typed limitation.

        """
        material_id = str(arguments.get("material_id", ""))
        raw_points = arguments.get("points", [])
        raw_candidates = arguments.get("assertion_candidates", [])
        raw_refs = arguments.get("evidence_refs", [])
        if (
            not material_id
            or not isinstance(raw_points, list)
            or not isinstance(raw_candidates, list)
            or not isinstance(raw_refs, list)
        ):
            return _limited("invalid_arguments")
        # Optional meta-knowledge items are filtered, never fatal: garbage
        # unresolved entries are dropped instead of rejecting the whole digest.
        unresolved = _unresolved(arguments.get("unresolved", []))
        points = list(raw_points)
        refs = list(raw_refs)
        context = (
            self._digest_context.get(material_id)
            if self._digest_context is not None
            else self._standalone_digest_context(material_id)
        )
        if not isinstance(context, dict):
            with self._store.read_connection() as connection:
                wrong_kind = connection.execute(
                    "SELECT 1 FROM observations WHERE id = ?", (material_id,)
                ).fetchone()
            return _digest_failure(
                "digest_material_id_required" if wrong_kind is not None else "unknown_material_id",
                field="material_id",
            )
        observation_id = str(context.get("observation_id", ""))
        material_kind = str(context.get("material_kind", ""))
        text = context.get("text")
        allowed_refs = context.get("allowed_evidence_refs", [])
        if not isinstance(text, str) or not text:
            return _digest_failure("digest_material_content_unavailable", field="material_id")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            expected_material_id = digest_material_id(observation_id, material_kind, text)
        except ValueError:
            return _digest_failure("digest_material_context_invalid", field="material_id")
        if expected_material_id != material_id:
            return _digest_failure("digest_material_identity_mismatch", field="material_id")
        if context.get("content_hash") != content_hash:
            return _digest_failure("stale_material_hash", field="material_id")
        if not isinstance(allowed_refs, list):
            return _digest_failure("digest_evidence_scope_unavailable", field="evidence_refs")
        outside_refs = [str(ref) for ref in refs if ref not in allowed_refs]
        if outside_refs:
            return _digest_failure(
                "evidence_ref_outside_material",
                field="evidence_refs",
                rejected_ids=outside_refs,
            )
        # The transcript boundary applies even to an idempotent repeat.  A
        # previously digested source cannot be cited again merely because it
        # still exists in SQLite; the current wake must have returned it.
        if self._digested_ids is not None and material_id in self._digested_ids:
            return {"ok": True, "already_digested": True, "material_id": material_id}
        with self._store.read_connection() as connection:
            observation = connection.execute(
                "SELECT excerpt FROM observations WHERE id = ?", (observation_id,)
            ).fetchone()
            if observation is None:
                return _digest_failure("unknown_observation_id", field="material_id")
            body = connection.execute(
                "SELECT 1 FROM observation_bodies WHERE observation_id = ?", (observation_id,)
            ).fetchone()
            if body is None and not str(observation["excerpt"] or "").strip():
                return _digest_failure("digest_material_content_unavailable", field="material_id")
            missing_refs: list[str] = []
            for ref in refs:
                if connection.execute("SELECT 1 FROM observations WHERE id = ?", (ref,)).fetchone() is None:
                    missing_refs.append(str(ref))
            if missing_refs:
                return _digest_failure(
                    "unknown_evidence_id",
                    field="evidence_refs",
                    rejected_ids=missing_refs,
                )
        if self._digested_ids is not None:
            self._digested_ids.add(material_id)
        return {
            "ok": True,
            "material_id": material_id,
            "observation_id": observation_id,
            "material_kind": material_kind,
            "content_hash": content_hash,
            "already_digested": False,
            "digest": {
                "points": points,
                "assertion_candidates": list(raw_candidates),
                "evidence_refs": refs,
                "unresolved": unresolved,
            },
        }

    def _standalone_digest_context(self, material_id: str) -> dict[str, object] | None:
        """Resolve a material from local storage for the public standalone facade.

        The graph always supplies an in-wake context and must never reach this
        path.  Direct deterministic callers have no transcript boundary, but
        still get the same exact-id and primary/part evidence restriction.
        """
        with self._store.read_connection() as connection:
            observations = connection.execute(
                "SELECT id, excerpt FROM observations WHERE excerpt IS NOT NULL"
            ).fetchall()
            bodies = connection.execute("SELECT observation_id, body_json FROM observation_bodies").fetchall()
        for row in observations:
            observation_id = str(row["id"])
            text = str(row["excerpt"] or "")
            if text and digest_material_id(observation_id, "excerpt", text) == material_id:
                return {
                    "observation_id": observation_id,
                    "material_kind": "excerpt",
                    "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text,
                    "allowed_evidence_refs": [observation_id],
                }
        for row in bodies:
            observation_id = str(row["observation_id"])
            try:
                body = parse_stored_body(str(row["body_json"]))
            except ValueError:
                continue
            text = body.text
            if not text or digest_material_id(observation_id, "full", text) != material_id:
                continue
            refs = [observation_id]
            if isinstance(body, BodyEnvelope):
                refs.extend(part.observation_id for part in body.parts)
            return {
                "observation_id": observation_id,
                "material_kind": "full",
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
                "allowed_evidence_refs": list(dict.fromkeys(refs)),
            }
        return None

    def _propose_inquiry(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a mid-exploration new-inquiry proposal and return its payload.

        The tool only validates and returns; it never writes to the store
        (dedup and commit happen at submit time). The subject must be a
        GraphRef with exactly one identity and, when it is a memory_id, that
        object must exist in the store — the same verify-then-cite discipline
        as _digest's observation check. A local_ref subject is not checked
        here: it can only be declared and resolved inside the terminal
        proposal. The returned payload carries no local_ref — the graph
        assigns the deterministic "round-N" reference so the terminal
        NewInquiryProposal validates (spec §1).

        Args:
            arguments: Raw tool call arguments (subject, prompt, rationale,
                kind, deepens_inquiry_id, answers_inquiry_id).

        Returns:
            The validated inquiry payload, or a typed limitation.

        """
        subject_raw = arguments.get("subject")
        if not isinstance(subject_raw, dict) or not subject_raw:
            return _inquiry_failure(
                "invalid_inquiry_field",
                field="subject",
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        kind = _text(arguments.get("kind", "factual"))
        if kind not in _INQUIRY_KINDS:
            return _inquiry_failure(
                "invalid_inquiry_field",
                field="kind",
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        prompt = _text(arguments.get("prompt"))
        rationale = _text(arguments.get("rationale"))
        if not prompt or not rationale:
            return _inquiry_failure(
                "invalid_inquiry_field",
                field="prompt" if not prompt else "rationale",
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        try:
            subject = GraphRef(**subject_raw)
        except ValidationError as error:
            location = error.errors()[0].get("loc", ()) if error.errors() else ()
            suffix = ".".join(str(part) for part in location)
            return _inquiry_failure(
                "invalid_inquiry_field",
                field=f"subject.{suffix}" if suffix else "subject",
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        try:
            proposal = NewInquiryProposal(
                local_ref="pending",
                subject=subject,
                prompt=prompt,
                rationale=rationale,
                kind=cast(Literal["factual", "semantic", "stateful"], kind),
                deepens_inquiry_id=_text(arguments.get("deepens_inquiry_id")) or None,
                answers_inquiry_id=_text(arguments.get("answers_inquiry_id")) or None,
            )
        except ValidationError as error:
            location = error.errors()[0].get("loc", ()) if error.errors() else ()
            field = ".".join(str(part) for part in location) or "subject"
            return _inquiry_failure(
                "invalid_inquiry_field",
                field=field,
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        if proposal.subject.memory_id is not None:
            with self._store.read_connection() as connection:
                if (
                    connection.execute(
                        "SELECT 1 FROM objects WHERE id = ?", (proposal.subject.memory_id,)
                    ).fetchone()
                    is None
                ):
                    return _inquiry_failure(
                        "unknown_memory_id",
                        field="subject.memory_id",
                        same_request_can_change=True,
                        change_condition="the_subject_object_exists_in_durable_memory",
                    )
        payload = proposal.model_dump(mode="json", exclude_none=True)
        payload.pop("local_ref")
        return {
            "ok": True,
            "inquiry": payload,
            "status": (
                "pending_terminal_subject_resolution"
                if proposal.subject.local_ref is not None
                else "validated_existing_subject"
            ),
            "writes_store": False,
        }

    def _log_inquiry_point(self, arguments: Mapping[str, Any], call_id: str) -> dict[str, Any]:
        """Record one inquiry point to the curiosity log."""
        topic = str(arguments.get("topic", "") or "").strip()
        source_ref = str(arguments.get("source_ref", "") or "").strip()
        reason = str(arguments.get("reason", "") or "").strip()
        if not topic:
            return _inquiry_failure(
                "missing_required_inquiry_point_field",
                field="topic",
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        if not source_ref or not reason:
            missing = [f for f in ("source_ref", "reason") if not str(arguments.get(f, "") or "").strip()]
            return _inquiry_failure(
                "missing_required_inquiry_point_field",
                field=",".join(missing),
                same_request_can_change=False,
                change_condition="request_field_is_corrected",
            )
        from leave_information_bubble.runtime.curiosity_log import (
            CURIOSITY_LOG_DEFAULT,
            append_curiosity,
        )

        append_curiosity(
            CURIOSITY_LOG_DEFAULT,
            self._thread_id,
            int(arguments.get("_round", 0) or 0),
            "unknown",
            source_ref,
            topic[:80],
            reason,
        )
        return {
            "ok": True,
            "logged": True,
            "status": "runtime_trace_recorded",
            "writes_world": False,
        }

    async def _acquire(
        self,
        name: str,
        arguments: Mapping[str, Any],
        call_id: str,
        attempts: list[int],
    ) -> dict[str, Any]:
        if name in _SCANS:
            adapter_id = str(arguments.get("adapter", ""))
            adapter = self._adapters.get(adapter_id)
            if adapter is None:
                return _operational_failure(
                    code="unknown_adapter",
                    stage="discover",
                    attempts=0,
                    limitations=[f"unknown_adapter:{adapter_id}"],
                )
            if name == "search_sources" and supports_targeted_search(adapter) is False:
                failure = _operational_failure(
                    code="adapter_targeted_search_unsupported",
                    stage="discover",
                    attempts=0,
                    same_request_can_change=False,
                    change_condition="request_arguments_or_adapter_capability_changes",
                    outcome=AcquisitionOutcome.UNSUPPORTED.value,
                )
                failure["completeness"] = _batch_completeness(
                    returned=0,
                    available=0,
                    partial=False,
                    physical_calls=0,
                )
                return failure
            try:
                request, window_mode = _scan(arguments, name, call_id, mode=self._mode)
                declared_surface_roles = _declared_surface_roles(adapter)
                if request.capability_roles and request.capability_roles[0] not in declared_surface_roles:
                    failure = _operational_failure(
                        code="adapter_surface_role_unsupported",
                        stage="discover",
                        attempts=0,
                        same_request_can_change=False,
                        change_condition="request_arguments_or_adapter_capability_changes",
                        outcome=AcquisitionOutcome.UNSUPPORTED.value,
                    )
                    failure["error"]["declared_surface_roles"] = [
                        role.value for role in declared_surface_roles
                    ]
                    return failure
                batch = await self._retry_transient(lambda: adapter.discover(request), attempts=attempts)
            except (TypeError, ValueError, ValidationError):
                return _operational_failure(
                    code="invalid_arguments",
                    stage="discover",
                    attempts=0,
                )
            except AgentError as exc:
                return _adapter_failure(exc, stage="discover", attempts=attempts[0])
            except Exception as exc:
                return _adapter_exception(exc, stage="discover", attempts=attempts[0])
            if not _matches(batch, adapter, f"tool:{call_id}"):
                return _operational_failure(
                    code="adapter_provenance_mismatch",
                    stage="discover",
                    attempts=attempts[0],
                )
            items = batch.occurrences[:_CARD_CAP]
            return self._commit_cards(
                _merge_observations([_seen(item) for item in items]),
                batch.limitations,
                batch.next_cursor,
                call_id,
                scope=_scan_scope(
                    request,
                    batch=batch,
                    adapter=adapter,
                    window_mode=window_mode,
                ),
                outcome=batch.outcome.value,
                completeness={
                    "returned": len(items),
                    "limit": request.limit,
                    "partial": batch.partial,
                    "truncated": len(batch.occurrences) > len(items) or bool(batch.next_cursor),
                    "next_cursor_usable": bool(batch.next_cursor),
                    "physical_calls": batch.physical_call_count,
                },
                track_scan_novelty=True,
            )
        row = self._stored_observation(str(arguments.get("observation_id", "")))
        if row is None:
            return _operational_failure(
                code="unknown_observation",
                stage="lookup",
                attempts=0,
            )
        observed = self._observed_dimensions(row)
        if name == "open_source":
            requested = str(arguments.get("depth", "preview"))
            refresh = arguments.get("refresh", False)
            if not isinstance(refresh, bool) or (refresh and requested != "full"):
                return _operational_failure(
                    code="invalid_arguments",
                    stage="hydrate",
                    attempts=0,
                )
            target = _OPEN_SOURCE_TARGETS.get(requested)
            if target is None and requested != "full":
                return _operational_failure(
                    code="invalid_arguments",
                    stage="hydrate",
                    attempts=0,
                )
            if _served_from_store(name, requested, row, observed):
                return self._already_opened(row)
            # below-target observations fall through to hydrate; "full" is wired in _open_full
            hydrate_depth = _OPEN_SOURCE_HYDRATE[requested]
        else:
            hydrate_depth = _DEPTHS[name]
            if _served_from_store(name, None, row, observed):
                return self._already_opened(row)
        adapter_id = str(row["source_kind"])
        adapter = self._adapters.get(adapter_id)
        if adapter is None:
            return _operational_failure(
                code="unknown_adapter",
                stage=_acquisition_stage(name, arguments),
                attempts=0,
                limitations=[f"unknown_adapter:{adapter_id}"],
            )
        if name == "open_source" and requested == "full":
            return await self._open_full(row, adapter, arguments, call_id, attempts, refresh=refresh)
        if name == "follow_related":
            related_call = getattr(adapter, "related", None)
            if related_call is not None:
                return await self._follow_related(related_call, adapter, row, call_id, attempts)
            if adapter.adapter_id not in _RELATED_FROM_HYDRATE:
                # No native related() and no hydrate-discovered related set:
                # the old fallback burned a full TEXT hydrate whose
                # discovered_occurrences stay empty, for a guaranteed
                # related_occurrences_unavailable (audit item 2). Fail fast
                # instead of wasting the network call and the turn.
                return _no_related([], attempts=0)
        request_arguments: dict[str, Any] = {}
        if name == "sample_discussion":
            page = int(arguments.get("page", 1))
            if page > 1:
                request_arguments = {"page": page}
        try:
            request = HydrationRequest(
                id=f"tool:hydrate:{arguments.get('observation_id', '')}",
                source_ref=str(row["source_uri"]),
                depth=hydrate_depth,
                arguments={**request_arguments, "domain": self._domain},
            )
            batch = await self._retry_transient(lambda: adapter.hydrate(request), attempts=attempts)
        except AgentError as exc:
            return _adapter_failure(exc, stage=_acquisition_stage(name, arguments), attempts=attempts[0])
        except Exception as exc:
            return _adapter_exception(exc, stage=_acquisition_stage(name, arguments), attempts=attempts[0])
        if not _matches(batch, adapter, f"tool:hydrate:{arguments.get('observation_id', '')}"):
            return _operational_failure(
                code="adapter_provenance_mismatch",
                stage=_acquisition_stage(name, arguments),
                attempts=attempts[0],
            )
        if name == "follow_related":
            related = batch.discovered_occurrences[:_CARD_CAP]
            if not related:
                return _no_related(batch.limitations, attempts=attempts[0])
            return self._commit_cards(
                _merge_observations([_seen(item) for item in related]),
                batch.limitations,
                "",
                call_id,
                preview=True,
                outcome=_effective_outcome(batch).value,
                completeness=_batch_completeness(
                    returned=len(related),
                    available=len(batch.discovered_occurrences),
                    partial=batch.partial,
                    physical_calls=batch.physical_call_count,
                ),
            )
        content = batch.observations[:_CARD_CAP]
        related = batch.discovered_occurrences[: max(0, _CARD_CAP - len(content))]
        if not content and not related:
            # A hydrate batch with neither observations nor discovered
            # occurrences is a platform answer (e.g. a hupu thread whose main
            # post has no public text), not a successful acquisition: an empty
            # ok card would teach the model "this platform returns nothing",
            # directly undermining platform visibility. Fail with the
            # adapter's own reasons plus the no_content_observed marker.
            return _no_content(
                batch.limitations,
                stage=_acquisition_stage(name, arguments),
                attempts=attempts[0],
                outcome=batch.outcome,
            )
        observations = [_hydrated(item, adapter_id, adapter.adapter_version) for item in content]
        observations.extend(_seen(item) for item in related)
        # Main observations (metadata, description) share the source-derived
        # main id and merge into one row; sub-content (comments, transcript
        # segments, danmaku) already carries distinct content-derived ids and
        # passes through. Every row of the hydrated source records which
        # dimensions this hydration observed (fix F1).
        merged = _merge_observations(observations)
        if content:
            merged = _mark_observed(merged, content[0].source_ref, _observed_kinds(content))
        return self._commit_cards(
            merged,
            batch.limitations,
            "",
            call_id,
            preview=True,
            outcome=_effective_outcome(batch).value,
            completeness=_batch_completeness(
                returned=len(content) + len(related),
                available=len(batch.observations) + len(batch.discovered_occurrences),
                partial=batch.partial,
                physical_calls=batch.physical_call_count,
            ),
        )

    async def _follow_related(
        self,
        related_call: Callable[..., Awaitable[DiscoveryBatch]],
        adapter: ChannelAdapter,
        row: Mapping[str, Any],
        call_id: str,
        attempts: list[int],
    ) -> dict[str, Any]:
        """Answer follow_related from the adapter's native related() batch.

        Adapters that implement related() (bilibili platform
        recommendations) answer the tool directly instead of re-hydrating
        the source and reading discovered_occurrences the hydrate never
        fills — the old path returned empty ok cards while burning full
        hydrate API calls (audit A3-2). Provenance is checked like every
        other adapter batch, and an empty recommendation set fails with an
        explicit limitation instead of a masking ok card.

        Args:
            related_call: The adapter's related() implementation.
            adapter: The registered adapter owning the anchor source.
            row: The stored anchor observation row.
            call_id: The model call identifier for provenance and commit.
            attempts: Mutable counter of adapter dispatch attempts for feedback.

        Returns:
            Bounded related cards, or a typed limitation.

        """
        request_id = f"tool:follow_related:{call_id}"
        try:
            batch = await self._retry_transient(
                lambda: related_call(
                    request_id=request_id,
                    source_ref=str(row["source_uri"]),
                    limit=_CARD_CAP,
                ),
                attempts=attempts,
            )
        except AgentError as exc:
            return _adapter_failure(exc, stage="related", attempts=attempts[0])
        except Exception as exc:
            return _adapter_exception(exc, stage="related", attempts=attempts[0])
        if not _matches(batch, adapter, request_id):
            return _operational_failure(
                code="adapter_provenance_mismatch",
                stage="related",
                attempts=attempts[0],
            )
        related = batch.occurrences[:_CARD_CAP]
        if not related:
            return _no_related(batch.limitations, attempts=attempts[0])
        return self._commit_cards(
            _merge_observations([_seen(item) for item in related]),
            batch.limitations,
            "",
            call_id,
            preview=True,
            outcome=_effective_outcome(batch).value,
            completeness=_batch_completeness(
                returned=len(related),
                available=len(batch.occurrences),
                partial=batch.partial,
                physical_calls=batch.physical_call_count,
            ),
        )

    def _stored_observation(self, observation_id: str) -> Mapping[str, Any] | None:
        with self._store.read_connection() as connection:
            return connection.execute("SELECT * FROM observations WHERE id = ?", (observation_id,)).fetchone()

    def _already_opened(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Return the stored card without re-hydrating an opened observation.

        The store-served card is a snapshot, not current truth: the source may
        have changed since it was observed (audit A6, the "S16 标题" case where
        Bilibili retitled a video and the store kept the old title). Every card
        therefore carries a soft freshness signal — ``stored_days_ago`` — and
        observations at least a day old also gain a limitation naming the age
        and the re-scan path: ``discover_sources``/``search_sources`` re-fetch
        the source and the store's same-id upsert refreshes the stored title
        and excerpt, so the model can decide to re-verify without a hard gate.
        """
        metadata = _stored_metadata(row)
        card: dict[str, Any] = {
            "id": row["id"],
            "title": row["title"],
            "source_uri": row["source_uri"],
            "source_kind": row["source_kind"],
            "depth": row["depth"],
            "published_at": row["source_published_at"],
            "observed_at": row["observed_at"],
            "excerpt": row["excerpt"],
            **_material_card_fields(metadata),
            "persisted": True,
            "adapter": {
                "id": row["source_kind"],
                "version": str(metadata.get("adapter_version", "")),
            },
        }
        days_ago = _stored_days_ago(row["observed_at"])
        if days_ago is not None:
            card["stored_days_ago"] = days_ago
        limitations = [f"observation already opened at depth={row['depth']}"]
        if days_ago is not None and days_ago >= 1:
            unit = "day" if days_ago == 1 else "days"
            limitations.append(f"stored {days_ago} {unit} ago; source content may have changed")
        engagement = _engagement(metadata)
        if engagement:
            card["engagement"] = engagement
        return {
            "ok": True,
            "outcome": AcquisitionOutcome.SUCCESS.value,
            "already_opened": True,
            "cards": [card],
            "limitations": limitations,
            "completeness": {
                "returned": 1,
                "limit": 1,
                "partial": False,
                "truncated": False,
                "next_cursor_usable": False,
                "physical_calls": 0,
            },
        }

    @staticmethod
    def _observed_dimensions(row: Mapping[str, Any]) -> set[str]:
        """Return the observed dimensions recorded on one stored observation row."""
        metadata = _stored_metadata(row)
        raw = metadata.get("observed")
        if not isinstance(raw, list):
            return set()
        return {str(kind) for kind in raw}

    async def _open_full(
        self,
        row: Mapping[str, Any],
        adapter: ChannelAdapter,
        arguments: Mapping[str, Any],
        call_id: str,
        attempts: list[int],
        *,
        refresh: bool,
    ) -> dict[str, Any]:
        """Serve an observation's full-text body, writing it on first acquisition.

        A stored body short-circuits hydration entirely (zero network). New
        full material first persists every body-bearing hydrated observation,
        then writes a typed v2 envelope whose exact text, deterministic part
        separators, offsets, and provenance can be read without reacquiring.
        Length is advisory only: every non-blank body is durable material.

        Args:
            row: The stored observation row being opened.
            adapter: The channel adapter owning this observation's source.
            arguments: The raw tool call arguments (observation_id).
            call_id: The model call id used for the idempotent part commit.
            attempts: Mutable counter of adapter dispatch attempts for feedback.
            refresh: Whether this caller explicitly requests a new hydration.

        Returns:
            A body card, or a typed limitation when the source has no full
            text. These refusals carry adapter reasons plus bounded source-state facts.

        """
        existing = self._store.read_observation_body(str(row["id"]))
        stored: BodyEnvelope | LegacyBody | None = None
        if existing is not None:
            try:
                stored = parse_stored_body(str(existing["body_json"]))
            except ValueError:
                return _full_unavailable([], "stored_body_invalid", attempts=0)
            if not stored.text.strip():
                return _full_unavailable([], "no_full_content", attempts=0)
            if not refresh and not _revision_mismatch(stored, row):
                return self._serve_stored_body(row, stored, str(existing["content_type"]))
        request = HydrationRequest(
            id=f"tool:hydrate:{arguments.get('observation_id', '')}",
            source_ref=str(row["source_uri"]),
            depth=HydrationDepth.TEXT,
            arguments={"full": True, "domain": self._domain},
        )
        try:
            batch = await self._retry_transient(lambda: adapter.hydrate(request), attempts=attempts)
        except AgentError as exc:
            if stored is not None and existing is not None:
                return self._serve_stale_body(
                    row, stored, str(existing["content_type"]), f"adapter_failure:{exc.code.value}"
                )
            return _adapter_failure(exc, stage="full_hydrate", attempts=attempts[0])
        except Exception as exc:
            if stored is not None and existing is not None:
                return self._serve_stale_body(
                    row, stored, str(existing["content_type"]), f"adapter_failure:{type(exc).__name__}"
                )
            return _adapter_exception(exc, stage="full_hydrate", attempts=attempts[0])
        if not _matches(batch, adapter, f"tool:hydrate:{arguments.get('observation_id', '')}"):
            if stored is not None and existing is not None:
                return self._serve_stale_body(
                    row, stored, str(existing["content_type"]), "adapter_provenance_mismatch"
                )
            return _operational_failure(
                code="adapter_provenance_mismatch",
                stage="full_hydrate",
                attempts=attempts[0],
            )
        if any(str(item.source_ref) != str(row["source_uri"]) for item in batch.observations):
            if stored is not None and existing is not None:
                return self._serve_stale_body(
                    row, stored, str(existing["content_type"]), "adapter_provenance_mismatch"
                )
            return _operational_failure(
                code="adapter_provenance_mismatch",
                stage="full_hydrate",
                attempts=attempts[0],
            )
        body_observations = [item for item in batch.observations if item.body and item.body.strip()]
        if not body_observations:
            if stored is not None and existing is not None:
                return self._serve_stale_body(
                    row, stored, str(existing["content_type"]), *batch.limitations, "no_full_content"
                )
            return _full_unavailable(batch.limitations, "no_full_content", attempts=attempts[0])

        durable_parts = [
            _body_part_observation(item, adapter.adapter_id, adapter.adapter_version)
            for item in body_observations
        ]
        envelope = build_body_envelope(
            [
                _material_part_input(item, durable.id)
                for item, durable in zip(body_observations, durable_parts, strict=True)
            ],
            max_chars=_FULL_BODY_CAP,
            **_source_revision(row),
        )
        if envelope is None:
            return _full_unavailable(batch.limitations, "no_full_content", attempts=attempts[0])
        body_json = envelope.to_json()
        try:
            written = self._store.replace_observation_body_with_parts(
                str(row["id"]),
                durable_parts,
                "mixed",
                body_json,
                commit_id=f"tool:{call_id}:body-parts",
            )
        except (ValueError, sqlite3.Error):
            if stored is not None and existing is not None:
                return self._serve_stale_body(
                    row, stored, str(existing["content_type"]), "refresh_material_write_failed"
                )
            return _operational_failure(
                code="material_write_failed",
                stage="persist",
                attempts=attempts[0],
            )
        if not written:
            replayed = self._store.read_observation_body(str(row["id"]))
            if replayed is None:
                return _operational_failure(
                    code="material_replay_missing",
                    stage="persist",
                    attempts=attempts[0],
                )
            try:
                replayed_body = parse_stored_body(str(replayed["body_json"]))
            except ValueError:
                return _operational_failure(
                    code="stored_body_invalid",
                    stage="persist",
                    attempts=attempts[0],
                )
            return self._serve_stored_body(
                row,
                replayed_body,
                str(replayed["content_type"]),
            )
        fixed_limitations = ["body_truncated"] if envelope.truncated else []
        persisted_row = self._stored_observation(str(row["id"]))
        if persisted_row is None:
            return _operational_failure(
                code="material_persisted_missing",
                stage="persist",
                attempts=attempts[0],
            )
        return {
            "ok": True,
            "outcome": _effective_outcome(batch).value,
            "cards": [_body_card(persisted_row, envelope, "mixed")],
            "limitations": _bounded_response_limitations(batch.limitations, fixed=fixed_limitations),
            "completeness": {
                "returned": 1,
                "limit": 1,
                "partial": batch.partial,
                "truncated": False,
                "next_cursor_usable": False,
                "physical_calls": batch.physical_call_count,
            },
        }

    def _serve_stored_body(
        self, row: Mapping[str, Any], body: BodyEnvelope | LegacyBody, content_type: str
    ) -> dict[str, Any]:
        """Return a zero-adapter stored body, upgrading only verified v2 material."""
        limitations = ["served_from_store"]
        if isinstance(body, LegacyBody):
            limitations.append("body_provenance_unknown")
        elif self._store.body_parts_are_verifiable(str(row["id"]), list(body.parts)):
            self._store.upgrade_observation_depth(str(row["id"]))
        else:
            limitations.append("body_provenance_unverified")
        if body.truncated:
            limitations.append("body_truncated")
        persisted_row = self._stored_observation(str(row["id"]))
        if persisted_row is None:
            return _limited("material_persisted_missing")
        return {
            "ok": True,
            "outcome": AcquisitionOutcome.SUCCESS.value,
            "cards": [_body_card(persisted_row, body, content_type)],
            "limitations": limitations,
            "completeness": {
                "returned": 1,
                "limit": 1,
                "partial": False,
                "truncated": False,
                "next_cursor_usable": False,
                "physical_calls": 0,
            },
        }

    def _serve_stale_body(
        self,
        row: Mapping[str, Any],
        body: BodyEnvelope | LegacyBody,
        content_type: str,
        *reasons: str,
    ) -> dict[str, Any]:
        """Keep an old body readable when a requested refresh cannot replace it."""
        served = self._serve_stored_body(row, body, content_type)
        served["limitations"] = _bounded_response_limitations(
            list(reasons), fixed=[*served["limitations"], "served_stale_after_refresh_failure"]
        )
        served["outcome"] = AcquisitionOutcome.PARTIAL.value
        served["completeness"]["partial"] = True
        return served

    def _commit_cards(
        self,
        observations: list[ObservationInput],
        limitations: list[str],
        next_cursor: str,
        call_id: str,
        preview: bool = False,
        scope: Mapping[str, object] | None = None,
        outcome: str | None = None,
        completeness: Mapping[str, object] | None = None,
        track_scan_novelty: bool = False,
    ) -> dict[str, Any]:
        persisted = False
        if observations:
            self._store.memory_commit(CognitiveDelta(observations=observations), f"tool:{call_id}")
            persisted = True
        builder = _preview_card if preview else _card
        result = {
            "ok": True,
            "cards": [builder(item, persisted=persisted) for item in observations],
            "limitations": _bounded_response_limitations(limitations),
            "next_cursor": next_cursor,
        }
        if scope is not None:
            result["scope"] = dict(scope)
        if outcome is not None:
            result["outcome"] = outcome
            result["ok"] = outcome not in {
                AcquisitionOutcome.UNSUPPORTED.value,
                AcquisitionOutcome.UNAVAILABLE.value,
            }
        if completeness is not None:
            result["completeness"] = dict(completeness)
        if track_scan_novelty:
            observation_ids = list(dict.fromkeys(item.id for item in observations))
            retained = set(self._scan_observation_ids)
            repeated = sum(identifier in retained for identifier in observation_ids)
            new_ids = [identifier for identifier in observation_ids if identifier not in retained]
            combined = [*self._scan_observation_ids, *new_ids]
            if len(combined) > _SCAN_HISTORY_CAP:
                self._scan_history_truncated = True
            self._scan_observation_ids = combined[-_SCAN_HISTORY_CAP:]
            result["novelty"] = {
                "new_in_wake": len(new_ids),
                "repeated_in_wake": repeated,
                "retained_history": len(self._scan_observation_ids),
                "history_truncated": self._scan_history_truncated,
            }
        if outcome in {
            AcquisitionOutcome.UNSUPPORTED.value,
            AcquisitionOutcome.UNAVAILABLE.value,
        }:
            first_reason = next(
                (str(item).strip() for item in limitations if str(item).strip()),
                f"acquisition_{outcome}",
            )
            unsupported = outcome == AcquisitionOutcome.UNSUPPORTED.value
            result["error"] = {
                "code": first_reason[:160],
                "stage": "discover",
                "attempts": max(
                    0,
                    int((completeness or {}).get("physical_calls", 0) or 0),
                ),
                "same_request_can_change": not unsupported,
                "change_condition": (
                    "request_arguments_or_adapter_capability_changes"
                    if unsupported
                    else "source_access_or_provider_availability_changes"
                ),
            }
        return result

    async def _retry_transient(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        attempts: list[int],
    ) -> _T:
        for attempt in range(2):
            attempts[0] += 1
            try:
                return await operation()
            except AgentError as error:
                if error.code is not ErrorCode.TOOL_TRANSIENT or attempt == 1:
                    raise
                self._emit_progress("tool_retrying", reason=error.code.value, attempt=attempt + 2)
        raise RuntimeError("unreachable transient retry state")


def _memory(
    bundle: MemoryBundle | dict[str, Any],
    *,
    scope: Mapping[str, Any] | None = None,
    limit: int | None = None,
    requested_ids: list[str] | None = None,
    ego: bool = False,
) -> dict[str, Any]:
    """Wrap a recall payload in the ok/memory/limitations envelope plus a first-person summary.

    The machine-readable payload stays unchanged under ``memory`` so every
    existing parse path keeps working; ``summary`` renders it in the first
    person (spec B4): "You remember N revisable judgments: <ids>" and bounded
    open-question context. Ids are repeated verbatim in the summary text so
    the model can extract and cite them (verify-then-cite) without parsing
    the full payload.
    """
    if isinstance(bundle, MemoryBundle):
        payload = bundle.model_dump(mode="json")
        truncated = bundle.truncated
        if bundle.paths:
            payload["paths_text"] = [" → ".join(path) for path in bundle.paths][
                : len(payload.get("paths", []))
            ]
        summary = _memory_summary(payload)
    else:
        payload = dict(bundle)
        truncated = bool(payload.pop("truncated", False))
        summary = _changes_summary(payload)
    if not payload.get("candidates"):
        # candidates is a search-only, additive envelope field: other memory
        # reads and object-less searches keep the pre-task-8 payload shape
        payload.pop("candidates", None)
    if not ego:
        # the ego-graph sections are expand-only additive fields: other memory
        # reads keep the pre-task-9 payload shape
        for ego_field in (
            "ego_neighbors",
            "status_assertions",
            "event_timeline",
            "event_edges",
            "participated_events",
            "omitted_counts",
            "sort_basis",
            "event_next_cursor",
        ):
            if not payload.get(ego_field):
                payload.pop(ego_field, None)
    returned_fields = {
        "anchor_objects",
        "assertions",
        "neighboring_objects",
        "inquiries",
        "evidence_refs",
        "candidate_observation_refs",
        "paths",
        "new_objects",
        "new_assertions",
        "new_inquiries",
        "resolved_inquiries",
        "candidates",
        "ego_neighbors",
        "status_assertions",
        "event_timeline",
        "event_edges",
        "participated_events",
    }
    returned = {
        key: len(value)
        for key, value in payload.items()
        if key in returned_fields and isinstance(value, list)
    }
    available_ids = {
        str(item["id"])
        for value in payload.values()
        if isinstance(value, list)
        for item in value
        if isinstance(item, dict) and item.get("id")
    }
    # memory_compare resolves existence through its side descriptors, not a
    # list field: a resolved side counts as found so a successful compare is
    # never misreported as two missing ids.
    compare = payload.get("compare")
    if isinstance(compare, dict):
        for side in (compare.get("left"), compare.get("right")):
            if isinstance(side, dict) and side.get("status") == "ok" and side.get("id"):
                available_ids.add(str(side["id"]))
    requested = list(dict.fromkeys(requested_ids or []))
    effective_scope = dict(scope or {})
    return {
        "ok": True,
        "memory": payload,
        "limitations": [],
        "summary": summary,
        "scope": effective_scope,
        "returned": returned,
        "limit": limit,
        "truncated": truncated or bool(effective_scope.get("object_ids_truncated")),
        "found_ids": [identifier for identifier in requested if identifier in available_ids],
        "missing_ids": [identifier for identifier in requested if identifier not in available_ids],
    }


def _memory_overview(
    overview: MemoryOverview,
    *,
    limit: int,
    requested_as_of: datetime | None,
) -> dict[str, Any]:
    """Wrap the P3 map without forcing it into legacy MemoryBundle fields."""
    payload = overview.model_dump(mode="json")
    counts = payload["counts"]
    return {
        "ok": True,
        "memory": {"overview": payload},
        "limitations": [],
        "scope": {
            "as_of": (
                overview.as_of.isoformat()
                if requested_as_of is None
                else requested_as_of.astimezone(UTC).isoformat()
            )
        },
        "returned": {
            key: len(payload[key])
            for key in ("active_fronts", "reactivated_fronts", "cold_bridges", "coverage_gaps")
        },
        "limit": limit,
        "truncated": overview.truncated,
        "found_ids": [],
        "missing_ids": [],
        "summary": (
            "Your memory map has "
            f"{counts['objects']} objects, {counts['assertions']} revisable judgments, and "
            f"{counts['open_inquiries']} open inquiries."
        ),
    }


def _memory_summary(payload: dict[str, Any]) -> str:
    """Render a memory bundle as a first-person cognition summary (spec B4)."""
    parts: list[str] = []
    compare = payload.get("compare")
    if isinstance(compare, dict) and compare.get("left") and compare.get("right"):
        left, right = compare["left"], compare["right"]
        label_key = "predicate" if compare.get("mode") == "assertion" else "canonical_name"
        fields = compare.get("fields", [])
        equal = sum(1 for entry in fields if entry.get("equal"))
        parts.append(
            f"You compared {left.get('id')} ({left.get(label_key)}) with "
            f"{right.get('id')} ({right.get(label_key)}): {equal} of "
            f"{len(fields)} fields equal."
        )
    assertions = payload.get("assertions") or []
    if assertions:
        ids = ", ".join(str(item.get("id", "")) for item in assertions)
        parts.append(
            f"You remember {_counted(len(assertions), 'revisable judgment', 'revisable judgments')}: {ids}"
        )
    inquiries = payload.get("inquiries") or []
    if inquiries:
        shown = inquiries[:3]
        items = "; ".join(
            f"{item.get('id', '')} ({item.get('subject_id', '')}): {str(item.get('prompt', ''))[:80]}"
            for item in shown
        )
        parts.append(f"Open remembered questions include: {items}")
        hidden = len(inquiries) - len(shown)
        if hidden:
            # honesty marker: truncated ids stay retrievable from the payload
            parts.append(f"and {hidden} more (see memory)")
    return " ".join(parts) or "This bounded memory result contains no revisable judgments."


def _changes_summary(payload: dict[str, Any]) -> str:
    """Render a memory_changes payload as a first-person change summary (spec B4).

    The since boundary renders as a readable date (or a relative description
    when absent or unparsable) so the first-person sentence never exposes a
    raw ISO timestamp.
    """
    since = payload.get("since")
    if since is None:
        since_label = "your last attempt"
    else:
        try:
            parsed = datetime.fromisoformat(str(since).replace("Z", "+00:00"))
            since_label = parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            since_label = "your last check"
    new_objects = payload.get("new_objects") or []
    new_assertions = payload.get("new_assertions") or []
    new_inquiries = payload.get("new_inquiries") or []
    resolved = payload.get("resolved_inquiries") or []
    return (
        f"Your memory changed since {since_label}: "
        f"{_counted(len(new_objects), 'new object', 'new objects')}, "
        f"{_counted(len(new_assertions), 'new revisable judgment', 'new revisable judgments')}, "
        f"{_counted(len(new_inquiries), 'new inquiry', 'new inquiries')}, "
        f"{_counted(len(resolved), 'resolved inquiry', 'resolved inquiries')}."
    )


def _counted(count: int, singular: str, plural: str) -> str:
    """Render a count with the singular or plural noun form."""
    return f"{count} {singular if count == 1 else plural}"


def _unresolved(raw: object) -> list[dict[str, str]]:
    """Return only well-formed unresolved items, dropping any garbage.

    The unresolved field is optional and the model may fill it loosely, so
    the facade tolerates garbage instead of rejecting the whole digest: a
    non-list value yields nothing, and list items survive only when they
    carry non-blank string ``what`` and ``why`` values.
    """
    if not isinstance(raw, list):
        return []
    items: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        what = item.get("what")
        why = item.get("why")
        if isinstance(what, str) and isinstance(why, str) and what.strip() and why.strip():
            items.append({"what": what, "why": why})
    return items


def _limited(limitation: str) -> dict[str, Any]:
    return {"ok": False, "limitations": [limitation]}


def _digest_failure(
    code: str,
    *,
    field: str,
    rejected_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return bounded digest validation facts without material text or advice."""
    error: dict[str, Any] = {"code": code[:160], "field": field[:80]}
    if rejected_ids:
        unique_ids = list(dict.fromkeys(identifier[:160] for identifier in rejected_ids if identifier))
        error["rejected_ids"] = unique_ids[:8]
        error["rejected_count"] = len(unique_ids)
        error["rejected_ids_truncated"] = len(unique_ids) > 8
    return {"ok": False, "limitations": [code], "error": error}


def _inquiry_failure(
    code: str,
    *,
    field: str,
    same_request_can_change: bool,
    change_condition: str,
) -> dict[str, Any]:
    """Return one factual inquiry/lease failure without prescribing an action."""
    return {
        "ok": False,
        "limitations": [code],
        "error": {
            "code": code[:160],
            "field": field[:80],
            "same_request_can_change": same_request_can_change,
            "change_condition": change_condition[:160],
        },
    }


def _acquisition_stage(name: str, arguments: Mapping[str, Any]) -> str:
    """Return the factual stage label used by acquisition feedback."""
    if name in _SCANS:
        return "discover"
    if name == "open_source" and str(arguments.get("depth", "preview")) == "full":
        return "full_hydrate"
    return "hydrate"


def _tool_stopped(
    *,
    code: str,
    stage: str,
    attempts: int,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Return stable stop facts without telling the Agent what to do next."""
    result = _operational_failure(
        code=code,
        stage=stage,
        attempts=attempts,
    )
    if elapsed_seconds is not None:
        result["elapsed_seconds"] = round(elapsed_seconds, 3)
    return result


def _operational_failure(
    *,
    code: str,
    stage: str,
    attempts: int,
    same_request_can_change: bool | None = None,
    change_condition: str | None = None,
    outcome: str | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Describe what failed and the condition that could change that fact."""
    default_same, default_condition, default_outcome = _FAILURE_FACTS.get(
        code,
        (False, "runtime_or_request_conditions_change", AcquisitionOutcome.UNAVAILABLE.value),
    )
    return {
        "ok": False,
        "outcome": outcome or default_outcome,
        "error": {
            "code": code[:160],
            "stage": stage[:80],
            "attempts": max(0, attempts),
            "same_request_can_change": (
                default_same if same_request_can_change is None else same_request_can_change
            ),
            "change_condition": (change_condition or default_condition)[:160],
        },
        "limitations": _bounded_response_limitations(limitations or [code]),
    }


def _adapter_failure(error: AgentError, *, stage: str, attempts: int) -> dict[str, Any]:
    """Project a classified adapter exception with its observed attempt count."""
    return _operational_failure(
        code=error.code.value,
        stage=stage,
        attempts=attempts,
        limitations=[f"adapter_failure:{error.code.value}"],
    )


def _adapter_exception(error: Exception, *, stage: str, attempts: int) -> dict[str, Any]:
    """Expose an unclassified adapter failure without leaking exception text."""
    error_type = type(error).__name__
    return _operational_failure(
        code=f"adapter_exception:{error_type}",
        stage=stage,
        attempts=attempts,
        same_request_can_change=True,
        change_condition="adapter_or_external_conditions_change",
        limitations=[f"adapter_failure:{error_type}"],
    )


def _full_unavailable(limitations: list[str], marker: str, *, attempts: int) -> dict[str, Any]:
    """Return the typed no-full-text envelope for _open_full failures.

    Every no-usable-full-text refusal reports adapter-provided reasons first,
    followed by fixed facts about unavailable full material. Retryable
    refusals (adapter_failure, provenance mismatch) bypass this helper.
    """
    return _operational_failure(
        code=marker,
        stage="full_hydrate",
        attempts=attempts,
        limitations=_bounded_response_limitations(
            [item for item in limitations if item != "use_sample_discussion_or_preview_instead"],
            fixed=[marker, "full_content_unavailable"],
        ),
    )


def _no_related(limitations: list[str], *, attempts: int) -> dict[str, Any]:
    """Return the typed no-related-data envelope for follow_related.

    follow_related must never mask a missing recommendation set with an
    empty ok card (audit A3-2): when an adapter returns no related
    occurrences the tool fails with an explicit limitation, keeping any
    adapter-provided reasons alongside the marker.
    """
    return _operational_failure(
        code="related_occurrences_unavailable",
        stage="related",
        attempts=attempts,
        limitations=_bounded_response_limitations(limitations, fixed=["related_occurrences_unavailable"]),
    )


def _no_content(
    limitations: list[str],
    *,
    stage: str,
    attempts: int,
    outcome: AcquisitionOutcome,
) -> dict[str, Any]:
    """Return the typed no-content envelope for preview/discussion opens.

    A hydrate batch yielding neither observations nor discovered occurrences
    is a platform answer (e.g. a hupu thread whose main post has no public
    text), never an adapter failure: an empty ok card would teach the model
    "this platform returns nothing", directly undermining platform
    visibility. The adapter-provided reasons ride first, then the marker,
    deduplicated via dict.fromkeys like _no_related.
    """
    unavailable = outcome is AcquisitionOutcome.UNAVAILABLE
    unsupported = outcome is AcquisitionOutcome.UNSUPPORTED
    partial = outcome is AcquisitionOutcome.PARTIAL
    specific_code = next(
        (str(item).strip() for item in limitations if str(item).strip()),
        "no_content_observed",
    )
    return _operational_failure(
        code=(specific_code if unavailable or unsupported or partial else "no_content_observed"),
        stage=stage,
        attempts=attempts,
        same_request_can_change=not unsupported,
        change_condition=(
            "request_or_adapter_capability_changes"
            if unsupported
            else "source_content_or_public_access_changes"
        ),
        outcome=(outcome.value if unavailable or unsupported or partial else AcquisitionOutcome.EMPTY.value),
        limitations=_bounded_response_limitations(limitations, fixed=["no_content_observed"]),
    )


def _stored_days_ago(observed_at: object) -> int | None:
    """Whole days between an observation's capture and now, or None when unparseable.

    The freshness signal is informational (audit A6): a stored card is a
    snapshot, and the model decides whether the age matters. Whole-day
    granularity matches the S16 title-change timescale; sub-day captures
    report 0. A missing or unparseable timestamp yields None so legacy rows
    never crash the store-served card.

    Args:
        observed_at: The stored capture timestamp (UTC ISO string or None).

    Returns:
        Whole days (floored, clamped at 0), or None when the value is absent
        or cannot be parsed.

    """
    if not observed_at:
        return None
    try:
        parsed = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0, (datetime.now(UTC) - parsed.astimezone(UTC)).days)


def _scan(
    arguments: Mapping[str, Any],
    name: str,
    call_id: str,
    *,
    mode: str,
) -> tuple[ScanRequest, str]:
    raw_start = arguments.get("window_start")
    raw_end = arguments.get("window_end")
    if (raw_start is None) != (raw_end is None):
        raise ValueError("window_start and window_end must be supplied together")
    if raw_start is not None:
        window_start = raw_start
        window_end = raw_end
        window_mode = "explicit"
    elif name == "discover_sources" and mode == "broad":
        window_end = datetime.now(UTC)
        window_start = window_end - timedelta(hours=24)
        window_mode = "broad_default_24h"
    else:
        window_start = None
        window_end = None
        window_mode = "unspecified"
    request = ScanRequest.model_validate(
        {
            "id": f"tool:{call_id}",
            "lane": f"world_{name}",
            "query": arguments.get("query", ""),
            "limit": min(int(arguments.get("limit", _CARD_CAP)), _CARD_CAP),
            "window_start": window_start,
            "window_end": window_end,
            "cursor": arguments.get("cursor", ""),
            "capability_roles": (
                [ChannelCapabilityRole(str(arguments["surface_role"]))]
                if arguments.get("surface_role") is not None
                else []
            ),
            "arguments": (
                {"surface_key": str(arguments["surface_key"])}
                if arguments.get("surface_key") is not None
                else {}
            ),
        }
    )
    return request, window_mode


def _declared_surface_roles(adapter: ChannelAdapter) -> tuple[ChannelCapabilityRole, ...]:
    """Return bounded, unique discovery roles in declaration order."""
    declared: list[ChannelCapabilityRole] = []
    for descriptor in getattr(adapter, "capability_descriptors", ()):
        role = getattr(descriptor, "role", None)
        entry_kind = getattr(descriptor, "entry_kind", None)
        if (
            not isinstance(role, ChannelCapabilityRole)
            or role not in _SCAN_SURFACE_ROLES
            or entry_kind not in _DISCOVERY_ENTRY_KINDS
            or role in declared
        ):
            continue
        declared.append(role)
    return tuple(declared)


def _scan_scope(
    request: ScanRequest,
    *,
    batch: DiscoveryBatch,
    adapter: ChannelAdapter,
    window_mode: str,
) -> dict[str, object]:
    """Echo the effective bounded scan identity without source content."""
    if request.window_start is None:
        applied: bool | None = False
        precision = "not_requested"
    elif batch.retrieval is not None:
        report = batch.retrieval.contract_report
        applied = report.time_filter_applied
        precision = report.time_filter_precision.value
    else:
        descriptors = tuple(getattr(adapter, "capability_descriptors", ()))
        declared = {item.time_filter_precision for item in descriptors}
        unsupported = any("time_filter_unsupported" in item for item in batch.limitations) or (
            bool(declared) and declared == {TimeFilterPrecision.UNSUPPORTED}
        )
        applied = False if unsupported else None
        precision = "unsupported" if unsupported else "unknown"
    return {
        "cursor": request.cursor,
        "query_semantics": _scan_query_semantics(adapter, request.capability_roles),
        "time_window": {
            "mode": window_mode,
            "start": request.window_start.isoformat() if request.window_start is not None else None,
            "end": request.window_end.isoformat() if request.window_end is not None else None,
            "applied": applied,
            "precision": precision,
        },
    }


def _scan_query_semantics(
    adapter: ChannelAdapter,
    requested_roles: list[ChannelCapabilityRole],
) -> str:
    """Return declared query behavior for the effective scan surface."""
    semantics: list[QuerySemantics] = []
    for descriptor in getattr(adapter, "capability_descriptors", ()):
        if (
            descriptor.entry_kind not in _DISCOVERY_ENTRY_KINDS
            or (requested_roles and descriptor.role not in requested_roles)
            or descriptor.query_semantics in semantics
        ):
            continue
        semantics.append(descriptor.query_semantics)
    if not semantics:
        return "capability_unclassified"
    if len(semantics) == 1:
        return semantics[0].value
    return "mixed:" + ",".join(item.value for item in semantics[:3])


def _effective_outcome(
    batch: DiscoveryBatch | ObservationBatch,
) -> AcquisitionOutcome:
    """Preserve explicit outcomes and lift legacy partial batches compatibly."""
    if batch.outcome is AcquisitionOutcome.SUCCESS and batch.partial:
        return AcquisitionOutcome.PARTIAL
    return batch.outcome


def _batch_completeness(
    *,
    returned: int,
    available: int,
    partial: bool,
    physical_calls: int,
) -> dict[str, object]:
    """Describe the bounded projection shared by open and related tools."""
    return {
        "returned": returned,
        "limit": _CARD_CAP,
        "partial": partial,
        "truncated": available > returned,
        "next_cursor_usable": False,
        "physical_calls": physical_calls,
    }


def _matches(batch: DiscoveryBatch | ObservationBatch, adapter: ChannelAdapter, request_id: str) -> bool:
    return (
        batch.request_id == request_id
        and batch.adapter_id == adapter.adapter_id
        and batch.adapter_version == adapter.adapter_version
        and all(
            item.adapter_id == adapter.adapter_id and item.adapter_version == adapter.adapter_version
            for item in (
                batch.occurrences if isinstance(batch, DiscoveryBatch) else batch.discovered_occurrences
            )
        )
    )  # noqa: E501


def _served_from_store(
    name: str,
    requested: str | None,
    row: Mapping[str, Any],
    observed: set[str],
) -> bool:
    """Decide whether a stored row already covers one tool call's dimension.

    already_opened short-circuits per observation dimension (task T13 fix
    F1): the linear depth ladder cannot order orthogonal dimensions, so
    reading subtitles must never count as reading comments. sample_discussion
    and open_source(discussion) short-circuit only when comments were
    observed; inspect_media and open_source(media) only when transcripts or
    media were observed. preview is the main row's own dimension — an
    observed ``content`` marker or a stored CONTENT depth means the source's
    body was read. Legacy rows written before the observed marker fall back
    to their stored depth for the discussion/media dimensions.

    Args:
        name: The tool being called.
        requested: The requested open_source depth, or None for other tools.
        row: The stored observation row the call references.
        observed: The row's recorded observed dimensions.

    Returns:
        True when the stored row already covers this call's dimension.

    """
    stored_depth = ObservationDepth(str(row["depth"]))
    if name == "open_source":
        if requested == "preview":
            return (
                "content" in observed or DEPTH_LEVELS[stored_depth] >= DEPTH_LEVELS[ObservationDepth.CONTENT]
            )
        if requested in {"discussion", "media"}:
            if _OBSERVED_REQUIREMENTS[f"open_source:{requested}"] & observed:
                return True
            return DEPTH_LEVELS[stored_depth] >= DEPTH_LEVELS[_OPEN_SOURCE_TARGETS[requested]]
        return False
    if name in {"sample_discussion", "inspect_media"}:
        if _OBSERVED_REQUIREMENTS[name] & observed:
            return True
        return DEPTH_LEVELS[stored_depth] >= DEPTH_LEVELS[_TARGET_DEPTH[name]]
    return False


def _seen_id(item: SourceOccurrence) -> str:
    return observation_id(item.adapter_id, item.source_ref)


def _selection_from_occurrence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Project bounded adapter selection facts without inventing a ranking model."""
    selection: dict[str, Any] = {}
    string_fields = {
        "provider": "provider",
        "capability_role": "capability_role",
        "provider_order": "provider_order",
        "origin_cluster_id": "origin_cluster_id",
        "independence_status": "independence_status",
    }
    for source, target in string_fields.items():
        value = metadata.get(source)
        if isinstance(value, str) and value.strip():
            selection[target] = value.strip()[:160]
    surface = metadata.get("board", metadata.get("fid"))
    if isinstance(surface, (str, int)) and str(surface).strip():
        selection["surface"] = str(surface).strip()[:160]
    integer_fields = {
        "board_page": "page",
        "board_rank": "original_rank",
        "query_relevance_rank": "relevance_rank",
        "query_rank": "relevance_rank",
        "search_rank": "search_rank",
    }
    for source, target in integer_fields.items():
        value = metadata.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            selection[target] = value
    matched = metadata.get("query_matched_terms")
    if isinstance(matched, list):
        terms = [str(item).strip()[:80] for item in matched[:8] if str(item).strip()]
        if terms:
            selection["matched_terms"] = terms
    if metadata.get("query_match_is_client_side") is True:
        selection["match_method"] = "bounded_client_side"
    return selection


def _seen(item: SourceOccurrence) -> ObservationInput:
    metadata: dict[str, Any] = {
        "adapter_version": item.adapter_version,
        "content_type": item.content_type,
        "language": item.language,
    }
    if item.content_hash_verified and item.content_hash:
        metadata["content_hash"] = item.content_hash
        metadata["content_hash_verified"] = True
        metadata["source_revision"] = item.content_hash
        metadata["source_revision_kind"] = "verified_content_hash"
    engagement = _engagement(item.metadata)
    if engagement:
        metadata["engagement"] = engagement
    selection = _selection_from_occurrence(item.metadata)
    if selection:
        metadata["selection"] = selection
    field_limitations = item.metadata.get("field_limitations")
    if isinstance(field_limitations, list):
        metadata["limitations"] = _bounded_limitations(field_limitations)[0]
    return ObservationInput(
        id=_seen_id(item),
        # source_ref alone is the source identity: the store's same-id
        # identity check compares source_uri, so discovery and hydration
        # must agree on it (canonical_url is display sugar, not identity).
        source_uri=item.source_ref,
        source_kind=item.adapter_id,
        title=item.title[:500],
        excerpt=(_text(item.metadata.get("snippet", "")) or item.title)[:800],
        depth=ObservationDepth.SEEN,
        source_published_at=item.source_published_at,
        observed_at=item.captured_at,
        metadata=metadata,
    )


def _hydrated(item: SourceObservation, adapter_id: str, version: str) -> ObservationInput:
    metadata: dict[str, Any] = {
        "adapter_version": version,
        "modality": item.modality.value,
        "location": item.location,
        "acquisition_method": item.acquisition_method,
        "confidence": item.confidence,
        "sampling_scope": item.sampling_scope,
        "limitations": item.limitations[:12],
    }
    for field in (
        "subtitle_reliability",
        "transcript_acquisition_method",
        "source_revision",
        "source_revision_kind",
    ):
        value = item.metadata.get(field)
        if isinstance(value, str) and value:
            metadata[field] = value
    engagement = _observation_engagement(item.metadata)
    if engagement:
        metadata["engagement"] = engagement
    kind = _SUB_CONTENT_KIND.get(item.modality)
    identifier = (
        _sub_content_id(adapter_id, item.source_ref, kind, item.excerpt)
        if kind is not None
        else observation_id(adapter_id, item.source_ref)
    )
    return ObservationInput(
        id=identifier,
        source_uri=item.source_ref,
        source_kind=adapter_id,
        title=_text(item.metadata.get("title", ""))[:500],
        excerpt=item.excerpt[:800],
        depth=_depth(item),
        source_published_at=_published(item.metadata.get("published_at")),
        observed_at=item.captured_at,
        metadata=metadata,
    )


def _body_part_observation(
    item: SourceObservation,
    adapter_id: str,
    version: str,
) -> ObservationInput:
    """Return the durable observation referenced by one body-envelope part."""
    hydrated = _hydrated(item, adapter_id, version)
    body = item.body or ""
    kind = _material_kind(item)
    reliability = _material_reliability(item)
    metadata = {
        **hydrated.metadata,
        "body_part": True,
        "material_kind": kind,
        "material_reliability": reliability,
    }
    return hydrated.model_copy(
        update={
            "id": _sub_content_id(adapter_id, item.source_ref, f"body-{kind}", body),
            "depth": ObservationDepth.CONTENT,
            "metadata": metadata,
        }
    )


def _material_part_input(item: SourceObservation, durable_id: str) -> MaterialPartInput:
    """Project adapter observation provenance into the body-envelope contract."""
    return MaterialPartInput(
        observation_id=durable_id,
        text=item.body or "",
        kind=_material_kind(item),
        location=item.location,
        acquisition_method=item.acquisition_method,
        confidence=item.confidence,
        reliability=_material_reliability(item),
        sampling_scope=item.sampling_scope,
        limitations=tuple(item.limitations),
        captured_at=item.captured_at,
    )


def _material_kind(item: SourceObservation) -> str:
    """Return a compact material kind without widening the adapter protocol."""
    if item.location == "video_description":
        return "description"
    if item.location == "video_transcript:full":
        return "transcript"
    return item.modality.value


def _material_reliability(item: SourceObservation) -> str:
    """Classify acquisition reliability from existing adapter-observable fields."""
    declared = item.metadata.get("subtitle_reliability")
    if declared in {"confirmed", "best_effort"}:
        return str(declared)
    method = item.acquisition_method.casefold()
    if method.startswith("faster_whisper:") or "asr" in method:
        return "automatic"
    if item.modality is ObservationModality.DOCUMENT_TEXT:
        return "source_direct"
    return "unknown"


def _source_revision(row: Mapping[str, Any]) -> dict[str, str | None]:
    """Extract only adapter-supplied, comparable revision metadata."""
    try:
        metadata = json.loads(str(row["metadata_json"] or "{}"))
    except (TypeError, ValueError):
        return {"source_revision": None, "source_revision_kind": None}
    revision = metadata.get("source_revision")
    kind = metadata.get("source_revision_kind")
    if isinstance(revision, str) and revision and isinstance(kind, str) and kind:
        return {"source_revision": revision, "source_revision_kind": kind}
    return {"source_revision": None, "source_revision_kind": None}


def _revision_mismatch(body: BodyEnvelope | LegacyBody, row: Mapping[str, Any]) -> bool:
    """Return whether verified, comparable source revisions disagree.

    Unknown or unverified discovery metadata is deliberately not freshness
    evidence, so it cannot trigger an implicit adapter call.
    """
    if not isinstance(body, BodyEnvelope) or body.source_revision_kind != "verified_content_hash":
        return False
    try:
        metadata = json.loads(str(row["metadata_json"] or "{}"))
    except (TypeError, ValueError):
        return False
    revision = metadata.get("source_revision")
    return (
        metadata.get("content_hash_verified") is True
        and metadata.get("source_revision_kind") == "verified_content_hash"
        and isinstance(revision, str)
        and bool(revision)
        and revision != body.source_revision
    )


def _sub_content_id(adapter_id: str, source_ref: str, kind: str, content: str) -> str:
    """Derive the id of one sub-content observation (comment/segment/danmaku item).

    Sub-content rows belong to a source without being properties of it, so
    each keeps its own row whose id extends the source id with a content
    hash: ``f"{adapter}:observation-<hash(source)>-{kind}-<hash(content)>"``.
    The content hash makes the id idempotent — the same comment or segment
    re-fetched later hits the same row and upgrades it (task T13 fix F2).

    Args:
        adapter_id: The channel adapter that owns the source.
        source_ref: The adapter's canonical source reference.
        kind: The sub-content kind (``comment``, ``transcript``, ...).
        content: The sub-content text the id is derived from.

    Returns:
        The stable sub-content observation id.

    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
    return f"{observation_id(adapter_id, source_ref)}-{kind}-{digest}"


def _observed_kinds(items: list[SourceObservation]) -> list[str]:
    """Return the distinct observed dimensions one hydration batch covers."""
    kinds: list[str] = []
    for item in items:
        kind = _OBSERVED_KIND.get(item.modality)
        if kind is not None and kind not in kinds:
            kinds.append(kind)
    return kinds


def _mark_observed(
    observations: list[ObservationInput],
    source_ref: str,
    kinds: list[str],
) -> list[ObservationInput]:
    """Attach the batch's observed dimensions to every row of one source.

    Rows of the hydrated source (main row and sub-content rows) record which
    dimensions this hydration observed, so later already_opened checks can
    short-circuit per dimension (fix F1). Rows of other sources (related
    occurrences) are left untouched. The store unions the marker across
    same-id re-writes, so the main row accumulates every dimension ever read.

    Args:
        observations: Merged observations from one hydrate batch.
        source_ref: The hydrated source's reference.
        kinds: Observed dimensions of this hydration, in first-seen order.

    Returns:
        The same observations with ``metadata["observed"]`` set on rows of
        the hydrated source when any dimension was observed.

    """
    if not kinds:
        return observations
    updated: list[ObservationInput] = []
    for item in observations:
        if item.source_uri != source_ref:
            updated.append(item)
            continue
        metadata = dict(item.metadata)
        metadata["observed"] = kinds
        updated.append(item.model_copy(update={"metadata": metadata}))
    return updated


def _merge_observations(observations: list[ObservationInput]) -> list[ObservationInput]:
    """Collapse same-id observations from one batch into one merged observation.

    Main observations of one source (video metadata, description) share the
    source-derived main id and fold into the richest single view: the
    longest excerpt decides the primary, the deepest observation in the
    group decides the row depth (monotone per DEPTH_LEVELS, mirroring the
    store's depth guard — a preview-opened source with a body read at
    CONTENT depth never collapses back to SEEN), the first non-empty title
    wins, engagement counters are unioned, and the primary's capture time
    is kept. Sub-content rows (comments, transcript segments, danmaku)
    already carry distinct content-derived ids, so they pass through
    untouched — only a duplicate of identical content within one batch
    (same id) collapses to its first copy. A discovery batch may also
    surface the same source twice; the merge folds those duplicates too.

    Args:
        observations: Observations from one adapter batch, in batch order.

    Returns:
        One observation per distinct id, in first-seen id order.

    """
    groups: dict[str, list[ObservationInput]] = {}
    for item in observations:
        groups.setdefault(item.id, []).append(item)
    return [_merge_group(items) for items in groups.values()]


def _merge_group(items: list[ObservationInput]) -> ObservationInput:
    """Fold one same-id group into its richest single observation.

    The primary (longest excerpt) supplies excerpt, title, metadata, and
    capture time, but the merged depth is the *deepest* observation in the
    group per DEPTH_LEVELS, not the primary's: a preview hydration returns
    a metadata row (SEEN) whose title excerpt often outlengthens the body
    row's description, and taking the primary's depth there would leave a
    CONTENT-body observation mislabeled SEEN while the observed marker
    already claims content was read. Depth rises with the deepest read in
    the batch and never falls below any member (monotone semantics, same
    ordering as the store's depth guard in _write_delta).

    Args:
        items: One id's observations from a single adapter batch; non-empty
            (grouping guarantees at least one item).

    Returns:
        The merged observation for the group's id.

    """
    primary = items[0]
    for item in items[1:]:
        if len(item.excerpt) > len(primary.excerpt):
            primary = item
    title = next((item.title for item in items if item.title), "")
    engagement: dict[str, Any] = {}
    for item in items:
        for key, value in _engagement(item.metadata).items():
            engagement.setdefault(key, value)
    metadata = dict(primary.metadata)
    if engagement:
        metadata["engagement"] = engagement
    return ObservationInput(
        id=primary.id,
        source_uri=primary.source_uri,
        source_kind=primary.source_kind,
        title=title,
        excerpt=primary.excerpt,
        depth=max(items, key=lambda item: DEPTH_LEVELS[item.depth]).depth,
        source_published_at=next(
            (item.source_published_at for item in items if item.source_published_at is not None),
            None,
        ),
        observed_at=primary.observed_at,
        metadata=metadata,
    )


def _depth(item: SourceObservation) -> ObservationDepth:
    if item.modality in {ObservationModality.TRANSCRIPT, ObservationModality.IMAGE, ObservationModality.OCR}:
        return ObservationDepth.MEDIA
    if (
        item.modality in {ObservationModality.COMMENT, ObservationModality.DANMAKU}
        or item.access_depth >= AccessDepth.REACTIONS
    ):
        return ObservationDepth.DISCUSSION
    return ObservationDepth.SEEN if item.access_depth is AccessDepth.METADATA else ObservationDepth.CONTENT


def _engagement(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return adapter engagement counters with missing values omitted.

    The adapter already writes its channel-wide engagement vocabulary (bilibili
    normalizes play/like/reply/danmaku into views/likes/comments/
    realtime_reactions; hupu and nga write views/replies) — this helper only
    drops null entries so a card never presents a missing counter as zero.
    """
    raw = metadata.get("engagement")
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if value is not None}


def _observation_engagement(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Map raw hydrated-observation counters into the shared engagement vocabulary.

    Hydrated observations carry the same bilibili counters under the raw
    ``stats`` keys (play/danmaku/reply/...) or an already-normalized
    ``engagement`` dict; comment aggregates additionally write a sampled
    ``comment_count``. Every present, non-null counter is surfaced so the
    hydrated card speaks the same vocabulary as discovery cards.
    """
    merged: dict[str, Any] = {}
    stats = metadata.get("stats")
    if isinstance(stats, dict):
        for source_key, target_key in _STATS_TO_ENGAGEMENT.items():
            value = stats.get(source_key)
            if value is not None:
                merged[target_key] = value
    raw = metadata.get("engagement")
    if isinstance(raw, dict):
        for key, value in raw.items():
            if value is not None:
                merged[key] = value
    sampled = metadata.get("comment_count")
    if sampled is not None:
        merged["comment_count"] = sampled
    return merged


def _card(item: ObservationInput, *, persisted: bool) -> dict[str, Any]:
    card: dict[str, Any] = {
        "id": item.id,
        "title": item.title,
        "source_uri": item.source_uri,
        "source_kind": item.source_kind,
        "depth": item.depth.value,
        "published_at": item.source_published_at and item.source_published_at.isoformat(),
        "observed_at": item.observed_at.isoformat(),
        "excerpt": item.excerpt[:_NAV_EXCERPT],
        **_material_card_fields(item.metadata),
        "persisted": persisted,
        "adapter": {"id": item.source_kind, "version": str(item.metadata["adapter_version"])},
    }
    engagement = _engagement(item.metadata)
    if engagement:
        card["engagement"] = engagement
    return card


def _preview_card(item: ObservationInput, *, persisted: bool) -> dict[str, Any]:
    """Build a preview card retaining the full stored excerpt for hydrate paths."""
    card = _card(item, persisted=persisted)
    card["excerpt"] = item.excerpt
    return card


def _body_card(
    row: Mapping[str, Any],
    body: BodyEnvelope | LegacyBody,
    content_type: str,
) -> dict[str, Any]:
    """Build a body card with exact material identity and provenance."""
    card: dict[str, Any] = {
        "id": row["id"],
        "primary_observation_id": row["id"],
        "title": row["title"],
        "source_uri": row["source_uri"],
        "source_kind": row["source_kind"],
        "depth": row["depth"],
        "published_at": row["source_published_at"],
        "observed_at": row["observed_at"],
        "body": body.text,
        "content_type": content_type,
        "refresh_available": True,
        "persisted": True,
    }
    if isinstance(body, LegacyBody):
        card.update(_material_card_fields(_stored_metadata(row)))
        card["provenance"] = "unknown"
        return card
    reliabilities = {part.reliability for part in body.parts}
    body_reliability = next(iter(reliabilities)) if len(reliabilities) == 1 else "mixed"
    body_limitations, limitations_truncated = _bounded_limitations(
        list(dict.fromkeys(item for part in body.parts for item in part.limitations))
    )
    card.update(
        {
            "captured_at": body.captured_at.isoformat(),
            "material_hash": body.material_hash,
            "quality_flags": list(body.quality_flags),
            "source_revision": body.source_revision,
            "source_revision_kind": body.source_revision_kind,
            "material_reliability": body_reliability,
            "limitations": body_limitations,
            "limitations_truncated": limitations_truncated,
            "parts": [_body_part_card(part) for part in body.parts],
        }
    )
    return card


def _body_part_card(part: MaterialPart) -> dict[str, Any]:
    """Project a durable body part into a bounded card without changing stored provenance."""
    card = part.as_dict()
    limitations, truncated = _bounded_limitations(list(part.limitations))
    card["limitations"] = limitations
    card["limitations_truncated"] = truncated
    return card


def _stored_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return stored observation metadata without letting malformed legacy JSON break recall."""
    try:
        metadata = json.loads(str(row["metadata_json"] or "{}"))
    except (TypeError, ValueError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _metadata_limitations(metadata: Mapping[str, Any]) -> list[str]:
    """Return observation-local limitations, never the enclosing tool limitations."""
    raw = metadata.get("limitations")
    if not isinstance(raw, list):
        return []
    return _bounded_limitations(raw)[0]


def _bounded_limitations(raw: list[Any]) -> tuple[list[str], bool]:
    """Bound card-local limitations without silently dropping a truncation signal."""
    limitations: list[str] = []
    total_chars = 0
    truncated = False
    for item in raw:
        if not isinstance(item, str) or not item:
            truncated = True
            continue
        value = item[:_LIMITATIONS_ITEM_MAX_CHARS]
        truncated = truncated or value != item
        if (
            len(limitations) >= _LIMITATIONS_CARD_MAX_ITEMS
            or total_chars + len(value) > _LIMITATIONS_CARD_MAX_CHARS
        ):
            truncated = True
            continue
        limitations.append(value)
        total_chars += len(value)
    if not truncated:
        return limitations, False
    while limitations and (
        len(limitations) >= _LIMITATIONS_CARD_MAX_ITEMS
        or total_chars + len(_LIMITATIONS_TRUNCATED) > _LIMITATIONS_CARD_MAX_CHARS
    ):
        total_chars -= len(limitations.pop())
    if len(_LIMITATIONS_TRUNCATED) <= _LIMITATIONS_CARD_MAX_CHARS:
        limitations.append(_LIMITATIONS_TRUNCATED)
    return limitations, True


def _bounded_response_limitations(limitations: list[str], *, fixed: list[str] | None = None) -> list[str]:
    """Bound adapter reasons while preserving any tool-owned factual markers."""
    trailing = list(dict.fromkeys(fixed or []))
    adapter = [item for item in dict.fromkeys(limitations) if item not in trailing]
    bounded, truncated = _bounded_limitations(adapter)
    if truncated:
        bounded.pop()
    marker_needed = truncated
    trailing_chars = sum(map(len, trailing))
    while bounded and (
        len(bounded) + len(trailing) + int(marker_needed) > _LIMITATIONS_CARD_MAX_ITEMS
        or sum(map(len, bounded)) + trailing_chars + (len(_LIMITATIONS_TRUNCATED) if marker_needed else 0)
        > _LIMITATIONS_CARD_MAX_CHARS
    ):
        bounded.pop()
        marker_needed = True
    result = [*bounded, *trailing]
    if marker_needed:
        result.append(_LIMITATIONS_TRUNCATED)
    return result


def _metadata_reliability(metadata: Mapping[str, Any]) -> str:
    """Return the persisted material reliability vocabulary, preserving unknowns honestly."""
    value = metadata.get("material_reliability")
    if value in {"source_direct", "confirmed", "best_effort", "automatic", "mixed", "unknown"}:
        return str(value)
    if metadata.get("subtitle_reliability") in {"confirmed", "best_effort"}:
        return str(metadata["subtitle_reliability"])
    method = str(metadata.get("acquisition_method", "")).casefold()
    if method.startswith("faster_whisper:") or "asr" in method:
        return "automatic"
    if metadata.get("modality") == ObservationModality.DOCUMENT_TEXT.value:
        return "source_direct"
    return "unknown"


def _material_card_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "material_reliability": _metadata_reliability(metadata),
        "limitations": _metadata_limitations(metadata),
    }
    selection = metadata.get("selection")
    if isinstance(selection, dict):
        bounded = _selection_from_occurrence(selection)
        # Persisted selection already uses display names rather than raw
        # adapter names, so retain those bounded values directly.
        for key in (
            "provider",
            "capability_role",
            "surface",
            "page",
            "original_rank",
            "relevance_rank",
            "search_rank",
            "matched_terms",
            "match_method",
            "provider_order",
            "origin_cluster_id",
            "independence_status",
        ):
            value = selection.get(key)
            if isinstance(value, str) and value.strip():
                bounded[key] = value.strip()[:160]
            elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                bounded[key] = value
            elif key == "matched_terms" and isinstance(value, list):
                terms = [str(item).strip()[:80] for item in value[:8] if str(item).strip()]
                if terms:
                    bounded[key] = terms
        if bounded:
            fields["selection"] = bounded
    return fields


def _published(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else None
    return None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parameters(
    name: str,
    *,
    adapter_ids: tuple[str, ...] = _ADAPTER_IDS,
    scan_contracts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    integer = {"type": "integer", "minimum": 1, "maximum": _CARD_CAP}
    read_integer = {"type": "integer", "minimum": 1, "maximum": _READ_CAP}
    string = {"type": "string", "minLength": 1}
    properties: dict[str, Any] = {}
    required: list[str] = []
    if name == "memory_recent":
        properties = {"limit": read_integer}
    elif name == "memory_inquiries":
        properties = {
            "limit": {"type": "integer", "minimum": 1, "maximum": _INQUIRY_CAP},
            "object_id": string,
            "inquiry_id": string,
        }
    elif name == "memory_search":
        properties = {
            "query": {
                **string,
                "description": (
                    "Required for the first page; a subsequent page may pass only the "
                    "cursor returned by the previous response."
                ),
            },
            "limit": read_integer,
            "mode": {
                "type": "string",
                "enum": ["page", "count"],
                "default": "page",
                "description": (
                    "count reports per-bucket totals and page_size without rows; "
                    "page returns one bounded page plus next_cursor."
                ),
            },
            "cursor": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Opaque page token from a previous memory_search; valid only within "
                    "the same wake. A page call may pass only the cursor."
                ),
            },
            "kind": {"type": "string", "enum": [member.value for member in ObjectKind]},
            "time_from": {
                "type": "string",
                "format": "date-time",
                "description": (
                    "ISO-8601 window start (tz-aware); keeps results whose event span "
                    "overlaps [time_from, time_to]."
                ),
            },
            "time_to": {
                "type": "string",
                "format": "date-time",
                "description": (
                    "ISO-8601 window end (tz-aware); keeps results whose event span "
                    "overlaps [time_from, time_to]."
                ),
            },
            "predicate": {"type": "string", "description": "Restricts the assertion bucket."},
            "has_participants": {
                "type": "boolean",
                "description": "Keep events with (true) or without (false) a current participant edge.",
            },
            "assertion_count_min": {
                "type": "integer",
                "minimum": 0,
                "description": "Lower bound; must not exceed assertion_count_max when both are present.",
            },
            "assertion_count_max": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Upper bound; must not be lower than assertion_count_min when both are present."
                ),
            },
        }
        required = []
    elif name == "memory_read":
        properties = {
            "object_id": {
                **string,
                "description": (
                    "Formal-graph object id, copied verbatim from a search hit or an "
                    "earlier read. Omit to read the most recently committed object."
                ),
            },
            "assertion_id": {
                **string,
                "description": (
                    "Evidence view: one judgment id with its evidence links and "
                    "correction chain. Mutually exclusive with object_id."
                ),
            },
            "include_working": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Overlay the current wake's staged working graph (graph_patch "
                    "items) over the formal graph: staged creates resolve by "
                    "staged_id, staged updates merge over the formal portrait, "
                    "staged drops hide the item. Default false reads the formal "
                    "graph only."
                ),
            },
            "views": {
                "type": "array",
                "items": {"type": "string", "enum": list(_READ_VIEWS)},
                "maxItems": 4,
                "description": (
                    "Which detail sections to expand: self_attributes (literal claims), "
                    "out_edges (judgments pointing at other objects), in_edges (who "
                    "references this object), correction_chain (how claims were "
                    "corrected, tail = current). Nothing expands by default."
                ),
            },
        }
        required = []
    elif name == "memory_compare":
        properties = {
            "left_id": {
                **string,
                "description": (
                    "First id, copied verbatim from a search hit or an earlier read. "
                    "Object ids compare identity fields; assertion ids compare "
                    "signature fields; the two sides must be the same kind."
                ),
            },
            "right_id": {
                **string,
                "description": "Second id, copied verbatim. Same kind as left_id.",
            },
        }
        required = ["left_id", "right_id"]
    elif name == "memory_expand":
        properties, required = (
            {
                "object_ids": {
                    "type": "array",
                    "items": string,
                    "minItems": 1,
                    "maxItems": _CARD_CAP,
                    "description": (
                        "Root objects to expand from. Required for a first page; "
                        "a paging call may send before alone."
                    ),
                },
                "depth": {"type": "integer", "minimum": 1, "maximum": _CARD_CAP},
                "limit": {"type": "integer", "minimum": 1, "maximum": _EXPAND_CAP},
                "event_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _EXPAND_CAP,
                    "default": _EVENT_LIMIT_DEFAULT,
                    "description": (
                        "Independent cap on participated_events; events beyond it "
                        "are counted in omitted_counts, never silently dropped."
                    ),
                },
                "include_history": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Set true to include retired (superseded) claims beside "
                        "current ones — the graph as recorded. Defaults to false "
                        "(current knowledge state)."
                    ),
                },
                "before": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Copy the next_cursor returned by an earlier memory_expand "
                        "call to page backward through participated events."
                    ),
                },
            },
            [],
        )
    elif name == "memory_evidence":
        properties, required = {"assertion_id": string}, ["assertion_id"]
    elif name == "memory_changes":
        properties = {
            "since": {
                **string,
                "format": "date-time",
                "description": (
                    "ISO-8601 instant (tz-aware); lists graph changes committed "
                    "after it. Naive datetimes are rejected; an offset or 'Z' "
                    "suffix is interpreted in its own zone."
                ),
            },
            "limit": read_integer,
        }
    elif name == "memory_overview":
        properties = {
            "as_of": {**string, "format": "date-time"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 3},
        }
    elif name == "claim_inquiry":
        properties, required = (
            {"inquiry_id": string, "owner_id": string, "ttl_seconds": {"type": "integer", "minimum": 1}},
            ["inquiry_id", "owner_id"],
        )
    elif name == "release_inquiry":
        properties, required = {"lease_token": string}, ["lease_token"]
    elif name == "digest_observation":
        properties, required = (
            {
                "material_id": {
                    **string,
                    "description": (
                        "Copy the digest_material_id of the returned card verbatim; "
                        "do not use its observation_id."
                    ),
                },
                "points": {"type": "array", "items": {"type": "string"}},
                "assertion_candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "predicate": {"type": "string"},
                            "literal": {"type": "string"},
                            "epistemic_role": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["predicate", "literal", "epistemic_role", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "unresolved": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "what": {"type": "string"},
                            "why": {"type": "string"},
                        },
                        "required": ["what", "why"],
                        "additionalProperties": False,
                    },
                },
            },
            ["material_id", "points", "assertion_candidates", "evidence_refs"],
        )
    elif name == "propose_inquiry":
        properties, required = (
            {
                "subject": {
                    "type": "object",
                    "properties": {
                        "local_ref": {"type": "string", "minLength": 1},
                        "memory_id": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
                "prompt": string,
                "rationale": string,
                "kind": {"type": "string", "enum": ["factual", "semantic", "stateful"]},
                "deepens_inquiry_id": string,
                "answers_inquiry_id": string,
            },
            ["subject", "prompt", "rationale"],
        )
    elif name == "log_inquiry_point":
        properties, required = (
            {
                "topic": string,
                "source_ref": string,
                "reason": string,
            },
            ["topic", "source_ref", "reason"],
        )
    elif name == "graph_patch":
        return graph_patch_parameters_schema(batch_cap=_PATCH_BATCH_CAP)
    elif name == "graph_inspect":
        properties, required = (
            {
                "item_id": {
                    **string,
                    "description": (
                        "Optional staged id (host-issued, e.g. wake:o1) to inspect one "
                        "staged item's current state plus its patch history."
                    ),
                },
            },
            [],
        )
    elif name == "graph_diff":
        properties, required = (
            {
                "limit": {"type": "integer", "minimum": 1, "maximum": _DIFF_PAGE_CAP, "default": 50},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            [],
        )
    elif name in _SCANS:
        contracts = scan_contracts or {}
        effective_adapter_ids = list(adapter_ids)
        if name == "search_sources" and scan_contracts is not None:
            effective_adapter_ids = [
                adapter_id
                for adapter_id in adapter_ids
                if contracts.get(adapter_id, {}).get("classified") is not True
                or contracts.get(adapter_id, {}).get("targeted_search") is True
            ]
        properties = {
            # A configured instance exposes precisely its registered adapters;
            # schema-only callers retain the global fallback above.
            "adapter": {"type": "string", "enum": effective_adapter_ids},
            "query": {"type": "string", "minLength": 1},
            "limit": integer,
            "window_start": {"type": "string", "format": "date-time"},
            "window_end": {"type": "string", "format": "date-time"},
            "surface_role": {
                "type": "string",
                "enum": [item.value for item in _SCAN_SURFACE_ROLES],
                "description": "Choose only a role declared for this adapter in the capability summary.",
            },
            "surface_key": {
                "type": "string",
                "description": (
                    "Adapter-specific surface key: for hupu the board name "
                    "(e.g. 'lol', 'csgo'), for nga the fid (e.g. '-152678'). "
                    "Required when the adapter's capability summary says "
                    "queryless=no for lack of a configured default board."
                ),
            },
            "cursor": {
                "type": "string",
                "description": "Copy a next_cursor returned by an earlier call to continue that surface.",
            },
        }
        # discover_sources is a targetless breadth survey: the query is
        # optional (hupu/nga scan their board order without one; bilibili and
        # public-web answer with a typed query-required limitation). Only
        # search_sources stays query-required — it is the targeted retrieval.
        required = ["adapter"] if name == "discover_sources" else ["adapter", "query"]
    elif name in _DEPTHS or name == "open_source":
        properties = {"observation_id": string}
        if name == "sample_discussion":
            properties["page"] = {"type": "integer", "minimum": 1}
        if name == "open_source":
            properties["depth"] = {"type": "string", "enum": ["preview", "full", "discussion", "media"]}
            properties["refresh"] = {"type": "boolean", "default": False}
        required = ["observation_id"]
    else:
        raise ValueError(f"unsupported tool schema: {name}")
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if name == "open_source":
        schema["allOf"] = [
            {
                "if": {
                    "properties": {"refresh": {"const": True}},
                    "required": ["refresh"],
                },
                "then": {
                    "properties": {"depth": {"const": "full"}},
                    "required": ["depth"],
                },
            }
        ]
    elif name == "memory_search":
        cursor_forbidden = [
            "query",
            "limit",
            "mode",
            "kind",
            "time_from",
            "time_to",
            "predicate",
            "has_participants",
            "assertion_count_min",
            "assertion_count_max",
        ]
        schema["anyOf"] = [{"required": ["query"]}, {"required": ["cursor"]}]
        schema["allOf"] = [
            {
                "if": {"required": ["cursor"]},
                "then": {"not": {"anyOf": [{"required": [field]} for field in cursor_forbidden]}},
            }
        ]
    elif name == "memory_read":
        schema["not"] = {"required": ["object_id", "assertion_id"]}
    elif name == "memory_expand":
        first_page_fields = [
            "object_ids",
            "depth",
            "limit",
            "event_limit",
            "include_history",
        ]
        schema["oneOf"] = [
            {
                "required": ["object_ids"],
                "not": {"required": ["before"]},
            },
            {
                "required": ["before"],
                "not": {"anyOf": [{"required": [field]} for field in first_page_fields]},
            },
        ]
    elif name in _SCANS:
        schema["dependentRequired"] = {
            "window_start": ["window_end"],
            "window_end": ["window_start"],
        }
        adapter_rules: list[dict[str, Any]] = []
        for adapter_id in schema["properties"]["adapter"]["enum"]:
            contract = (scan_contracts or {}).get(adapter_id, {})
            if contract.get("classified") is not True:
                continue
            then: dict[str, Any] = {}
            raw_roles = contract.get("roles")
            roles = [str(role) for role in raw_roles] if isinstance(raw_roles, (list, tuple)) else []
            if roles:
                then.setdefault("properties", {})["surface_role"] = {
                    "type": "string",
                    "enum": roles,
                }
            forbidden: list[dict[str, list[str]]] = []
            if contract.get("supports_cursor") is not True:
                forbidden.append({"required": ["cursor"]})
            if contract.get("supports_surface_key") is not True:
                forbidden.append({"required": ["surface_key"]})
            if forbidden:
                then["not"] = {"anyOf": forbidden}
            if name == "discover_sources" and contract.get("query_required") is True:
                if contract.get("supports_surface_key") is True:
                    then["anyOf"] = [
                        {"required": ["query"]},
                        {"required": ["surface_key"]},
                    ]
                else:
                    then["required"] = ["query"]
            adapter_rules.append(
                {
                    "if": {
                        "properties": {"adapter": {"const": adapter_id}},
                        "required": ["adapter"],
                    },
                    "then": then,
                }
            )
        if adapter_rules:
            schema["allOf"] = adapter_rules
    return schema


# --- memory_search structured pagination (Slice 2a, design §4.2 / P14) ---

_SEARCH_CURSOR_VERSION = 1
# Page positions beyond this are refused: a forged or drifting cursor can
# never push a structured read into an unbounded scan.
_SEARCH_OFFSET_LIMIT = 10_000
_SEARCH_CURSOR_KEYS = (
    "query",
    "kind",
    "time_from",
    "time_to",
    "predicate",
    "has_participants",
    "assertion_count_min",
    "assertion_count_max",
    "offset",
    "limit",
)


def _search_filters(arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize structured memory_search filters, or None when malformed.

    Times must be tz-aware ISO-8601 (stored UTC-normalized); kind must be a
    live ObjectKind value; has_participants must be a real boolean; the
    assertion count range must be non-negative and not inverted. Any
    violation fails closed with ``invalid_arguments``.
    """
    filters: dict[str, Any] = {}
    raw_kind = arguments.get("kind")
    if raw_kind is not None:
        kind = str(raw_kind)
        if kind not in {member.value for member in ObjectKind}:
            return None
        filters["kind"] = kind
    for key in ("time_from", "time_to"):
        raw = arguments.get(key)
        if raw is None:
            continue
        if not isinstance(raw, str):
            return None
        moment = _published(raw)
        if moment is None:
            return None
        filters[key] = moment.astimezone(UTC).isoformat()
    raw_predicate = arguments.get("predicate")
    if raw_predicate is not None:
        predicate = str(raw_predicate).strip()
        if not predicate:
            return None
        filters["predicate"] = predicate[:160]
    raw_participants = arguments.get("has_participants")
    if raw_participants is not None:
        if not isinstance(raw_participants, bool):
            return None
        filters["has_participants"] = raw_participants
    for key in ("assertion_count_min", "assertion_count_max"):
        raw = arguments.get(key)
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            return None
        filters[key] = raw
    if (
        filters.get("assertion_count_min") is not None
        and filters.get("assertion_count_max") is not None
        and filters["assertion_count_min"] > filters["assertion_count_max"]
    ):
        return None
    return filters


def _encode_search_cursor(thread_id: str, state: Mapping[str, Any]) -> str:
    """Mint an opaque, thread-bound page token for one structured search."""
    payload = {
        "v": _SEARCH_CURSOR_VERSION,
        "thread": thread_id,
        **{key: state[key] for key in _SEARCH_CURSOR_KEYS if key in state},
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii")


def _decode_search_cursor(token: str, thread_id: str) -> dict[str, Any] | None:
    """Decode a page token, refusing anything not minted for this thread.

    Any parse failure, version mismatch, or thread mismatch returns None —
    the caller falls back to the first page and flags ``invalid_cursor``
    (P14: cursors are valid only within the wake that minted them).
    """
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        state = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    if state.get("v") != _SEARCH_CURSOR_VERSION:
        return None
    if state.get("thread") != thread_id:
        return None
    return state


# Participated-events page tokens (design §4.5 unified pagination): the
# recall layer speaks raw {"t", "id"} position JSON; the facade wraps it in
# the same thread-bound base64url token machinery as memory_search so one
# degradation story serves both tools (P14, D-008).
_EVENT_CURSOR_VERSION = 1


def _encode_event_cursor(thread_id: str, moment: str, event_id: str) -> str:
    """Mint an opaque, thread-bound event page token for memory_expand."""
    payload = {
        "v": _EVENT_CURSOR_VERSION,
        "thread": thread_id,
        "t": moment,
        "id": event_id,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii")


def _decode_event_cursor(token: str, thread_id: str) -> tuple[str, str] | None:
    """Decode an event page token into its (t, id) position, or None when foreign/dead.

    Any parse failure, version mismatch, or thread mismatch returns None so
    the caller degrades to the first page and flags ``invalid_cursor``.
    """
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        state = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    if state.get("v") != _EVENT_CURSOR_VERSION:
        return None
    if state.get("thread") != thread_id:
        return None
    if not isinstance(state.get("t"), str) or not isinstance(state.get("id"), str):
        return None
    return state["t"], state["id"]


def _search_scope(
    query: str,
    mode: str,
    filters: Mapping[str, Any],
    *,
    offset: int,
    cursor_status: str | None,
) -> dict[str, Any]:
    """Project the effective memory_search scope (query + filters + page state)."""
    scope: dict[str, Any] = {"query": query, "mode": mode}
    for key in ("kind", "predicate"):
        if filters.get(key) is not None:
            scope[key] = str(filters[key])[:160]
    for key in ("time_from", "time_to"):
        if filters.get(key) is not None:
            scope[key] = str(filters[key])[:64]
    if filters.get("has_participants") is not None:
        scope["has_participants"] = bool(filters["has_participants"])
    for key in ("assertion_count_min", "assertion_count_max"):
        if filters.get(key) is not None:
            scope[key] = int(filters[key])
    if offset or cursor_status is not None:
        scope["offset"] = offset
    if cursor_status is not None:
        scope["cursor_status"] = cursor_status
    return scope


def _search_counts(
    counts: Mapping[str, int],
    *,
    limit: int,
    scope: Mapping[str, Any],
    query: str,
) -> dict[str, Any]:
    """Wrap count-mode totals in the standard envelope (no rows returned)."""
    return {
        "ok": True,
        "memory": {
            "counts": {key: int(counts[key]) for key in ("objects", "assertions", "inquiries")},
            "page_size": limit,
        },
        "limitations": [],
        "summary": (
            f"You remember {int(counts['objects'])} objects, {int(counts['assertions'])} "
            f"revisable judgments, and {int(counts['inquiries'])} open inquiries matching "
            f"the query."
        ),
        "scope": dict(scope),
        "returned": {},
        "limit": limit,
        "truncated": False,
        "found_ids": [],
        "missing_ids": [],
    }


def _read(
    recall: WorldRecall,
    name: str,
    arguments: Mapping[str, Any],
    injection: bool,
    thread_id: str = "",
) -> MemoryBundle:
    """Dispatch one memory read; injection stays at the card cap (inquiries at _INQUIRY_CAP)."""
    if name == "memory_recent":
        limit = _card_limit(arguments, 12) if injection else _read_limit(arguments, 12)
        return recall.recent(limit=limit)
    # memory_search runs through WorldTools._search (structured filters,
    # count mode, wake-bound cursors) and is intercepted before this dispatch.
    if name == "memory_read":
        include_working = bool(arguments.get("include_working", False))
        return recall.read(
            arguments.get("object_id"),
            arguments.get("assertion_id"),
            views=list(arguments.get("views", [])),
            include_working=include_working,
            wake_id=thread_id if include_working else "",
        )
    if name == "memory_compare":
        return recall.compare(
            str(arguments.get("left_id", "") or ""),
            str(arguments.get("right_id", "") or ""),
        )
    # memory_expand runs through WorldTools._expand (wake-bound event
    # cursors, include_history) and is intercepted before this dispatch.
    if name == "memory_evidence":
        return recall.evidence(str(arguments.get("assertion_id", "")))
    return recall.inquiries(
        arguments.get("object_id"),
        limit=_inquiry_limit(arguments, 20),
        inquiry_id=arguments.get("inquiry_id"),
    )


def _memory_limit(name: str, arguments: Mapping[str, Any], injection: bool) -> int:
    """Return the exact facade cap applied to one memory read."""
    if name in {"memory_recent", "memory_search"}:
        return _card_limit(arguments, 12) if injection else _read_limit(arguments, 12)
    if name == "memory_expand":
        return min(int(arguments.get("limit", _EXPAND_CAP)), _EXPAND_CAP)
    if name in {"memory_evidence", "memory_read", "memory_compare"}:
        return _EXPAND_CAP
    return _inquiry_limit(arguments, 20)


def _memory_scope(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Project the effective read scope without inventing recall advice."""
    if name == "memory_recent":
        return {"order": "newest_first"}
    if name == "memory_search":
        return _search_scope(
            str(arguments.get("query", ""))[:240],
            str(arguments.get("mode", "page")),
            _search_filters(arguments) or {},
            offset=0,
            cursor_status=None,
        )
    if name == "memory_expand":
        requested = list(
            dict.fromkeys(str(value)[:160] for value in arguments.get("object_ids", []) if value)
        )
        root_cap = min(_CARD_CAP, _memory_limit(name, arguments, False))
        raw_before = arguments.get("before")
        return {
            "object_ids": requested[:root_cap],
            "requested_object_count": len(requested),
            "object_ids_truncated": len(requested) > root_cap,
            "depth": min(int(arguments.get("depth", 1)), _CARD_CAP),
            "event_limit": min(int(arguments.get("event_limit", _EVENT_LIMIT_DEFAULT)), _EXPAND_CAP),
            "before": str(raw_before)[:200] if raw_before is not None else None,
            "include_history": bool(arguments.get("include_history", False)),
        }
    if name == "memory_evidence":
        return {"assertion_id": str(arguments.get("assertion_id", ""))[:160]}
    if name == "memory_read":
        return {
            "object_id": str(arguments["object_id"])[:160] if arguments.get("object_id") else None,
            "assertion_id": (str(arguments["assertion_id"])[:160] if arguments.get("assertion_id") else None),
            "defaulted": not arguments.get("object_id") and not arguments.get("assertion_id"),
            "views": [str(view)[:32] for view in arguments.get("views", [])][:4],
            "include_working": bool(arguments.get("include_working", False)),
        }
    if name == "memory_compare":
        return {
            "left_id": str(arguments.get("left_id", "") or "")[:160],
            "right_id": str(arguments.get("right_id", "") or "")[:160],
        }
    scope: dict[str, Any] = {}
    if arguments.get("object_id") is not None:
        scope["object_id"] = str(arguments["object_id"])[:160]
    if arguments.get("inquiry_id") is not None:
        scope["inquiry_id"] = str(arguments["inquiry_id"])[:160]
    if not scope:
        scope["statuses"] = ["open", "dormant"]
    return scope


def _memory_requested_ids(name: str, arguments: Mapping[str, Any]) -> list[str]:
    """Return exact ids whose existence this read can truthfully resolve."""
    if name == "memory_expand":
        requested = list(
            dict.fromkeys(str(value)[:160] for value in arguments.get("object_ids", []) if value)
        )
        return requested[: min(_CARD_CAP, _memory_limit(name, arguments, False))]
    if name == "memory_evidence":
        return [str(arguments.get("assertion_id", ""))[:160]]
    if name == "memory_read":
        requested = str(arguments.get("object_id", "") or "")[:160]
        if requested:
            return [requested]
        return [str(arguments.get("assertion_id", "") or "")[:160]]
    if name == "memory_compare":
        return [
            str(arguments.get("left_id", "") or "")[:160],
            str(arguments.get("right_id", "") or "")[:160],
        ]
    if name == "memory_inquiries" and arguments.get("inquiry_id") is not None:
        return [str(arguments["inquiry_id"])[:160]]
    return []


def _card_limit(arguments: Mapping[str, Any], default: int) -> int:
    return min(int(arguments.get("limit", default)), _CARD_CAP)


def _inquiry_limit(arguments: Mapping[str, Any], default: int) -> int:
    """Cap the inquiries injection at the dedicated widened window."""
    return min(int(arguments.get("limit", default)), _INQUIRY_CAP)


def _read_limit(arguments: Mapping[str, Any], default: int) -> int:
    """Cap agent-initiated recall at the wider read window."""
    return min(int(arguments.get("limit", default)), _READ_CAP)


__all__ = ["WorldTools"]
