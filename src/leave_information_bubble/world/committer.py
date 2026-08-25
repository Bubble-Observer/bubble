"""Deterministically compile Agent proposals into one atomic world commit."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from .contracts import (
    AliasAction,
    AliasOperation,
    AssertionInput,
    CognitiveDelta,
    CommitReceipt,
    EpistemicRole,
    EvidenceInput,
    InquiryInput,
    InquiryResolution,
    JsonValue,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationLinkInput,
)
from .graph_contract import normalize_identity_alias
from .materials import BodyEnvelope, parse_stored_body
from .proposal import (
    AttemptContext,
    CognitionDeltaProposal,
    GraphRef,
    NewObjectProposal,
    ObjectUpdateProposal,
    ProposalCommitReceipt,
    ReviewIssue,
    ReviewIssueCode,
    ReviewOutcome,
)
from .similarity import bigrams, jaccard, normalize
from .store import (
    AmbiguousLegacyTouchState,
    CommitReplayConflict,
    ObjectAliasCollision,
    WorldStore,
    _assertion_signature,
    _identity_alias_form,
    alias_lookup_forms,
)


class ProposalValidationError(ValueError):
    """A repairable proposal error with optional durable identity candidates.

    ``answerable_inquiries`` carries the receipt's exact answerable set for
    the rejected proposal (open/dormant inquiries on the assertions'
    subjects, attached by :meth:`ProposalCommitter.commit`'s rejection
    paths) so the repair feedback can name it — the one moment the proposal
    exists, which the transcript-approximating finalization injection cannot
    reach.
    """

    def __init__(
        self,
        errors: list[str],
        candidate_object_ids: list[str] | None = None,
        candidate_inquiry_ids: list[str] | None = None,
    ) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
        self.candidate_object_ids = candidate_object_ids or []
        self.candidate_inquiry_ids = candidate_inquiry_ids or []
        self.review_issues: list[ReviewIssue] = []
        self.answerable_inquiries: list[str] = []


# Calibrated 2026-08-04 against a stability-audit world snapshot (18 open
# inquiries). Plain bigram Jaccard cannot separate the known duplicate
# groups (lowest intra-group pair 0.035, highest cross-group pair 0.262), so
# the documented fallback applies: store-side interception only compares rows
# whose subject_id equals the proposed inquiry's subject. Under subject-match
# every same-subject duplicate pair scores >= 0.241 while every other
# same-subject pair scores <= 0.1; 0.2 separates them.
_INQUIRY_SIMILARITY_THRESHOLD = 0.2

# Task 2.1 matrix row 4: a fuzzy-only name hit against an existing object is
# a duplicate-object candidate warning. The line is the shared 0.20 bigram
# jaccard the graph's ghost-memory-id candidate search
# (_GHOST_CANDIDATE_SIMILARITY_THRESHOLD) and the read-side recall bucketing
# (_MATCH_SIMILARITY_THRESHOLD) use.
_FUZZY_SIMILARITY_THRESHOLD = 0.2

# Durable candidate-list cap for ghost-memory-id issues
# (_missing_memory_issue): the graph's model-visible feedback renders at
# most _FEEDBACK_CANDIDATE_CAP candidates per issue with a counted tail, so
# the durable issues_json record only carries a small bounded list per
# issue — kept above the feedback cap so the counted tail stays truthful.
_ISSUE_CANDIDATE_CAP = 8

# Legacy public kinds fold into the three core graph kinds at the proposal
# boundary, each with a default type_key unless the proposal declares an
# explicit legal one (which always wins). The 'source' kind is deliberately
# absent: it is rejected with an ``unsupported_object_kind`` review issue and
# steered toward observation/evidence material.
_LEGACY_KIND_ROUTES: dict[str, tuple[ObjectKind, str]] = {
    ObjectKind.ORGANIZATION.value: (ObjectKind.ENTITY, "organization"),
    ObjectKind.PLACE.value: (ObjectKind.ENTITY, "place"),
    ObjectKind.METHOD.value: (ObjectKind.CONCEPT, "method"),
    ObjectKind.RULE.value: (ObjectKind.CONCEPT, "rule"),
}


def _route_new_object_kinds(
    proposal: CognitionDeltaProposal,
) -> tuple[dict[int, tuple[ObjectKind, str | None]], list[int]]:
    """Map every new-object proposal index to its routed core kind + type_key.

    The routing is a pure function of the proposal (replay-safe): legacy
    kinds organization/place fold to entity and method/rule to concept, each
    with a default type_key unless the proposal declares an explicit legal
    one; the three core kinds pass through unchanged. The unsupported
    ``source`` kind yields no routing at all, so the caller never mints a
    durable id for it.

    Returns:
        A ``(routed, unsupported)`` pair: index -> effective (kind, type_key)
        for every supported item, and the sorted indexes of unsupported items.

    """
    routed: dict[int, tuple[ObjectKind, str | None]] = {}
    unsupported: list[int] = []
    for index, item in enumerate(proposal.new_objects):
        legacy = _LEGACY_KIND_ROUTES.get(item.kind.value)
        if legacy is None:
            if item.kind is ObjectKind.SOURCE:
                unsupported.append(index)
            else:
                routed[index] = (item.kind, item.type_key)
            continue
        core_kind, default_type_key = legacy
        routed[index] = (
            core_kind,
            item.type_key if item.type_key is not None else default_type_key,
        )
    return routed, unsupported


@dataclass(frozen=True)
class CandidateWindow:
    """A capped, deduplicated durable candidate list with the counted tail (F6)."""

    ids: tuple[str, ...]
    omitted_count: int


def _candidate_window(raw_ids: Iterable[str], cap: int = _ISSUE_CANDIDATE_CAP) -> CandidateWindow:
    """Cap a durable candidate list and count the omitted tail.

    Every candidate issue uses the same window for ``candidate_ids`` and its
    corresponding ``actual_value`` ID list, plus ``candidate_omitted_count``
    when the tail is non-empty — one issue never carries an unbounded
    payload under any key.
    """
    unique = tuple(dict.fromkeys(str(value) for value in raw_ids))
    return CandidateWindow(ids=unique[:cap], omitted_count=max(0, len(unique) - cap))


def _candidate_payload(
    raw_ids: Iterable[str] | None,
    *,
    sampled: bool = False,
    truncated: bool = False,
) -> tuple[list[str], dict[str, int | bool]]:
    """Return the windowed candidate list and its honest tail keys.

    Exact sources keep ``candidate_omitted_count`` — the exact number of
    candidates beyond the visible cap. A sampled source (one whose SQL
    pre-selection can hit its LIMIT, so the true set is unbounded) never
    claims an exact tail. A truncated pre-selection leaves two distinct
    facts, and the emitted key must match the one actually known
    (closeout 2026-08-18 correction):

    * the final (post-classification) window itself was cut at the cap —
      ``candidate_has_more: true``: at least one further candidate is
      *confirmed* to exist, the total is not countable;
    * the final candidates fit under the cap but the pre-selection was
      truncated — rows beyond the SQL LIMIT were never classified, so the
      total is unknown: ``candidate_sample_truncated: true``, never
      ``candidate_has_more`` (which would claim certainty the caller does
      not have).

    When the pre-selection was not truncated the count is exact.
    ``candidate_has_more``, ``candidate_sample_truncated`` and
    ``candidate_omitted_count`` never co-occur on one issue, and the tail
    keys are conditional, so under-cap issues keep their exact payloads
    unchanged (replay-deterministic).
    """
    window = _candidate_window(raw_ids or ())
    if sampled and truncated:
        if window.omitted_count:
            return list(window.ids), {"candidate_has_more": True}
        return list(window.ids), {"candidate_sample_truncated": True}
    tail = {"candidate_omitted_count": window.omitted_count} if window.omitted_count else {}
    return list(window.ids), tail


# Identical to recall._CJK_RUN: a pure CJK run renders as an FTS5 prefix
# phrase (the read side's _phrase behavior) so CJK probes keep their recall.
_CJK_RUN = re.compile(r"[一-鿿]+")


def _fts_phrase(term: str) -> str:
    """Render one fuzzy-name probe as an escaped FTS5 phrase (F6).

    An embedded double quote is doubled inside the phrase; a pure CJK run
    keeps the prefix-star behavior, everything else is an exact phrase.
    """
    escaped = str(term).replace('"', '""')
    if _CJK_RUN.fullmatch(str(term)):
        return f'"{escaped}"*'
    return f'"{escaped}"'


@dataclass(frozen=True)
class _RecordedAttemptReview:
    """The append-only review result of one durable proposal attempt."""

    outcome: ReviewOutcome
    issues: list[ReviewIssue]


@dataclass(frozen=True)
class NewObjectCollisionDecision:
    """One new object's collision fate (Task 2.1): omit or allow, never merge.

    An ``omit`` decision carries exactly one ``identity_alias_claim_conflict``
    issue plus the colliding (index, alias, owner) pairs for fail-closed
    bookkeeping. An ``allow`` decision carries warning issues only
    (``ambiguous_name_candidates``, ``duplicate_object_candidate``) and never
    rewrites the local ref.
    """

    omit: bool
    issue: ReviewIssue | None
    warning_issues: tuple[ReviewIssue, ...]
    collision_pairs: tuple[tuple[int, str, str], ...]


@dataclass(frozen=True)
class ObjectUpdateAliasDecision:
    """One object update's alias ADD fate (F3): omit only the conflicting ADDs.

    ``accepted_adds`` are the raw ADD strings that passed the active-owner
    check (proposal-local claims checked before stored owners); ``conflicts``
    are ``(raw_alias, owner_id)`` pairs for the claimed forms. The safe
    update fields and non-conflicting aliases still compile; the active
    identity unique index remains the race-time fail-closed backstop.
    """

    accepted_adds: tuple[str, ...]
    conflicts: tuple[tuple[str, str], ...]


def _multi_action_alias_conflicts(
    add_aliases: Sequence[str],
    remove_aliases: Sequence[str],
    demote_aliases: Sequence[str],
) -> list[str]:
    """Return normalized forms declared in more than one alias action.

    One update may declare a normalized form in at most one action; a form
    declared in two or more is dropped from every action with a single
    ``alias_operation_invalid`` issue. This is the single decision point
    shared by the alias-operation compiler and the object-update rewrite, so
    a dropped form can never be re-activated by the compiled alias list.
    """
    seen: dict[str, set[AliasAction]] = {}
    for action, raws in (
        (AliasAction.ADD, add_aliases),
        (AliasAction.REMOVE, remove_aliases),
        (AliasAction.DEMOTE, demote_aliases),
    ):
        for raw in raws:
            normalized_form = _identity_alias_form(raw)
            if normalized_form:
                seen.setdefault(normalized_form, set()).add(action)
    return sorted(form for form, actions in seen.items() if len(actions) > 1)


# F4: one ordered signature tracker for the whole proposal — explicit
# assertions register as they compile, and compiled participant edges run the
# same announcement path afterward, so a later same-signature entry always
# sees the first owner (the store keeps the first compiled assertion's id).
PendingAssertionSignatures = dict[str, AssertionInput]


def _stored_assertion_metadata(row: sqlite3.Row) -> dict[str, JsonValue]:
    """Return the first-owner metadata of a stored assertion as comparable values.

    Fields are the ones the store keeps on first write and never replaces;
    the announcement compares them against the proposed values and discloses
    every difference (F4). Key order is shared with
    :func:`_proposed_assertion_metadata` so both sides zip field by field.
    """
    return {
        "confidence": row["confidence"],
        "event_time_precision": row["event_time_precision"],
        "supersedes_id": row["supersedes_id"],
        "supersede_reason": row["supersede_reason"],
        "answers_inquiry_id": row["answers_inquiry_id"],
    }


def _proposed_assertion_metadata(assertion: AssertionInput) -> dict[str, JsonValue]:
    """Return the proposed metadata of a compiled assertion, same field order."""
    return {
        "confidence": assertion.confidence,
        "event_time_precision": assertion.event_time_precision,
        "supersedes_id": assertion.supersedes_id,
        "supersede_reason": assertion.supersede_reason,
        "answers_inquiry_id": assertion.answers_inquiry_id,
    }


class ProposalCommitter:
    """Resolve proposal-local identities and delegate one atomic store transaction."""

    def __init__(
        self,
        store: WorldStore,
        *,
        inquiry_similarity_threshold: float = _INQUIRY_SIMILARITY_THRESHOLD,
        thread_id: str = "",
    ) -> None:
        """Build a proposal committer bound to one world store and thread."""
        self._store = store
        self._inquiry_similarity_threshold = inquiry_similarity_threshold
        self._thread_id = thread_id

    def commit(
        self,
        proposal: CognitionDeltaProposal,
        commit_id: str,
        *,
        attempt: AttemptContext | None = None,
        missing_memory_candidates: dict[str, list[str]] | None = None,
    ) -> ProposalCommitReceipt:
        """Validate, compile, and atomically persist one Agent proposal."""
        if not commit_id.strip():
            raise ProposalValidationError(["commit id must be non-empty"])
        attempt = attempt or self._legacy_attempt_context(proposal, commit_id)
        attempt_recorded = self._assert_attempt_available(attempt, proposal)
        committed = self._store.committed_delta(commit_id)
        replay_receipt = committed[0] if committed is not None else None
        replay_delta = committed[1] if committed is not None else None
        replay_inquiry_ids = {item.id for item in replay_delta.inquiries} if replay_delta else set()
        object_ids: dict[str, str] = {}
        resolved_object_refs: list[dict[str, str]] = []
        collisions: list[tuple[int, str, str]] = []
        omitted_new_objects: dict[int, tuple[list[str], list[str]]] = {}
        alias_add_decisions: dict[int, ObjectUpdateAliasDecision] = {}
        # The legacy-kind routing is a pure function of the proposal and runs
        # BEFORE any durable-id minting: unsupported items (source) get no id,
        # and every collision decision compares the effective routed kind.
        routed_object_kinds, unsupported_object_indexes = _route_new_object_kinds(proposal)
        unsupported_indexes = set(unsupported_object_indexes)
        claimed_identity_aliases: dict[str, str] = {}
        collision_issues: dict[int, list[ReviewIssue]] = {}
        with self._store.read_connection() as connection:
            for index, item in enumerate(proposal.new_objects):
                if index in unsupported_indexes:
                    # unsupported kinds never mint a durable id nor match
                    # aliases; they surface as review issues in _compile
                    continue
                own_id = durable_id("object", commit_id, item.local_ref)
                decision = self._decide_new_object_collision(
                    index,
                    item,
                    kind=routed_object_kinds[index][0],
                    own_id=own_id,
                    connection=connection,
                    claimed_identity_aliases=claimed_identity_aliases,
                )
                if decision.omit:
                    # Task 2.1 matrix row 1: the item declares an identity
                    # alias actively owned by another object; the item itself
                    # is omitted with the owner as repair candidate — never
                    # auto-merged into the owner. Assertions/inquiries/links
                    # referencing its local ref are omitted as its declared
                    # dependencies.
                    collisions.extend(decision.collision_pairs)
                    omitted_new_objects[index] = (
                        list(dict.fromkeys(alias for _, alias, _ in decision.collision_pairs)),
                        list(dict.fromkeys(candidate for _, _, candidate in decision.collision_pairs)),
                    )
                    issue = decision.issue
                    if issue is not None:
                        collision_issues[index] = [
                            issue.model_copy(
                                update={
                                    "omitted_dependencies": self._omitted_dependencies(
                                        proposal, item.local_ref
                                    )
                                }
                            )
                        ]
                else:
                    # Rows 2-4: same-name / legacy / domain-overlap hits are
                    # bounded warnings; the object is still created under its
                    # own deterministic durable id, and its identity aliases
                    # claim the proposal-local alias map for later items.
                    object_ids[item.local_ref] = own_id
                    for alias in item.aliases:
                        normalized = _identity_alias_form(alias)
                        if normalized:
                            claimed_identity_aliases[normalized] = own_id
                    if decision.warning_issues:
                        collision_issues[index] = list(decision.warning_issues)
            # ── Task 2.3: event identity is candidate-only, never auto-dedup ──
            # A new event is never rewritten to an existing event and never
            # merged silently. The host compares it against every stored event
            # (objects(kind='event') rows plus their has_participant assertions
            # — no separate events table exists) and returns tiered
            # duplicate_event_candidate issues. A strong match (same non-empty
            # type_key, identical complete (role, object_id) participant pairs,
            # overlapping time at the same non-unknown precision, and consistent
            # part_of/granularity declarations) rejects the whole proposal into
            # one bounded repair with an error issue; a partial match commits
            # with a warning and both events coexist. The decision is a pure
            # function of the proposal and the read-only object and assertion
            # tables — no store mutation, replay-safe. Items already omitted by
            # alias collision (§5.2) keep that decision and are never
            # candidates, and an event never matches its own durable id
            # (same-commit replay must not self-collide).
            strong_event_issues: dict[int, ReviewIssue] = {}
            for index, item in enumerate(proposal.new_objects):
                if item.kind is not ObjectKind.EVENT:
                    continue
                if index in omitted_new_objects or index in unsupported_indexes:
                    continue
                strong_issue, partial_issue = self._decide_event_candidates(
                    index,
                    item,
                    proposal,
                    object_ids,
                    commit_id,
                    connection=connection,
                )
                if strong_issue is not None:
                    strong_event_issues[index] = strong_issue
                elif partial_issue is not None:
                    collision_issues.setdefault(index, []).append(partial_issue)
            # F3: an identity alias claimed by a different durable object
            # makes that one ADD unsafe, not the whole update. The decision
            # per update keeps global identity-alias uniqueness by omitting
            # only the claimed ADDs; the safe update fields and unclaimed
            # aliases still compile (field-level omission, spec §7.2). The
            # check reads ACTIVE identity_aliases owners plus the
            # proposal-local claims made by surviving new objects and earlier
            # updates, so the store's raw unique-index failure is unreachable
            # from the compiler. Stale or missing targets never apply and
            # never claim aliases; _compile surfaces them on its own.
            for index, item in enumerate(proposal.object_updates):
                target_id = str(item.target.memory_id)
                row = connection.execute(
                    "SELECT version FROM objects WHERE id = ?", (target_id,)
                ).fetchone()
                if row is None or row["version"] != item.expected_version:
                    continue
                accepted_adds: list[str] = []
                conflicts: list[tuple[str, str]] = []
                seen_forms: set[str] = set()
                for alias in item.add_aliases:
                    normalized = _identity_alias_form(alias)
                    if not normalized or normalized in seen_forms:
                        # empty and duplicate forms are ignored deterministically
                        continue
                    seen_forms.add(normalized)
                    owner = claimed_identity_aliases.get(normalized)
                    if owner is None:
                        row = connection.execute(
                            "SELECT object_id FROM identity_aliases"
                            " WHERE normalized_alias = ? AND status = 'active' AND object_id != ?"
                            " ORDER BY object_id LIMIT 1",
                            (normalized, target_id),
                        ).fetchone()
                        owner = str(row["object_id"]) if row is not None else None
                    if owner is not None:
                        conflicts.append((alias, owner))
                    else:
                        accepted_adds.append(alias)
                if accepted_adds or conflicts:
                    alias_add_decisions[index] = ObjectUpdateAliasDecision(
                        accepted_adds=tuple(accepted_adds),
                        conflicts=tuple(conflicts),
                    )
                for alias in accepted_adds:
                    normalized = _identity_alias_form(alias)
                    if normalized:
                        claimed_identity_aliases[normalized] = target_id
            inquiry_ids = {
                item.local_ref: durable_id("inquiry", commit_id, item.local_ref)
                for item in proposal.new_inquiries
            }
            # ── E2: similar inquiries are omitted, never reject the proposal ──────
            open_rows = connection.execute(
                "SELECT id, subject_id, prompt FROM inquiries WHERE status IN ('open','dormant')"
            ).fetchall()
            omitted_inquiries: list[int] = []
            deepened: list[str] = []
            omitted_local_refs: set[str] = set()
            omitted_object_refs = {
                proposal.new_objects[omitted_index].local_ref
                for omitted_index in omitted_new_objects
            } | {
                proposal.new_objects[index].local_ref
                for index in unsupported_object_indexes
            }
            for index, item in enumerate(proposal.new_inquiries):
                if item.subject.local_ref in omitted_object_refs:
                    # the inquiry anchors to a new object that was omitted for
                    # an alias collision; a compiled inquiry cannot reference
                    # an object that will not exist, so it is omitted with it
                    omitted_inquiries.append(index)
                    omitted_local_refs.add(item.local_ref)
                    continue
                prompt_bigrams = bigrams(item.prompt)
                subject_id = self._subject_id(item.subject, object_ids)
                hits = [
                    str(row["id"])
                    for row in open_rows
                    if str(row["id"]) not in replay_inquiry_ids
                    and str(row["subject_id"]) == subject_id
                    and jaccard(prompt_bigrams, bigrams(str(row["prompt"])))
                    >= self._inquiry_similarity_threshold
                ]
                # intra-delta comparison against other new inquiries
                hits += [
                    inquiry_ids[other.local_ref]
                    for other in proposal.new_inquiries
                    if other is not item
                    and jaccard(prompt_bigrams, bigrams(other.prompt)) >= self._inquiry_similarity_threshold
                ]
                hits = list(dict.fromkeys(hits))
                if not hits:
                    continue
                if item.deepens_inquiry_id is not None and item.deepens_inquiry_id in hits:
                    deepened.append(item.deepens_inquiry_id)
                    continue  # declared and matched — allowed
                omitted_inquiries.append(index)
                omitted_local_refs.add(item.local_ref)
        try:
            if strong_event_issues:
                # Task 2.3 fail-closed: a strong duplicate event is an error,
                # never a silent omission and never a reuse — the whole
                # proposal is rejected into one bounded repair carrying every
                # candidate id, and sibling collision issues ride the same
                # feedback at error severity (an error issue can never ride a
                # committed receipt, so rejection is the only shape that keeps
                # the replay-verification invariant intact).
                candidate_ids = list(
                    dict.fromkeys(
                        candidate
                        for index in sorted(strong_event_issues)
                        for candidate in strong_event_issues[index].candidate_ids
                    )
                )
                error = ProposalValidationError(
                    [
                        "new event duplicates an existing event: "
                        + ", ".join(
                            f"{proposal.new_objects[index].local_ref} -> {candidate}"
                            for index in sorted(strong_event_issues)
                            for candidate in strong_event_issues[index].candidate_ids
                        )
                    ],
                    candidate_ids,
                )
                error.review_issues = [
                    strong_event_issues[index] for index in sorted(strong_event_issues)
                ]
                for index in sorted(collision_issues):
                    issue = collision_issues.get(index, [None])[0]
                    if issue is not None:
                        error.review_issues.append(
                            issue.model_copy(update={"severity": "error"})
                        )
                raise error
            if omitted_new_objects and len(omitted_new_objects) == len(proposal.new_objects):
                # §5.2 fail-closed: every new object collides, so the omission
                # discipline has nothing safe to keep from the object section;
                # the whole proposal is rejected instead of committing a hollow
                # delta (this also keeps the d3 double-collision shape intact).
                # The pre-built identity-conflict issues carry the repair
                # candidates with error severity for the rejected attempt.
                candidate_ids = list(dict.fromkeys(candidate for _, _, candidate in collisions))
                error = ProposalValidationError(
                    [
                        "object aliases already exist: "
                        + ", ".join(f"{alias} -> {candidate}" for _, alias, candidate in collisions)
                    ],
                    candidate_ids,
                )
                error.review_issues = []
                for index in sorted(omitted_new_objects):
                    issue = collision_issues.get(index, [None])[0]
                    if issue is not None:
                        error.review_issues.append(issue.model_copy(update={"severity": "error"}))
                raise error
            (
                delta,
                omitted_assertions,
                omitted_resolutions,
                answerable,
                evidence_missing_assertions,
                dropped_evidence,
                review_issues,
                missing_resolution_ids,
            ) = self._compile(
                proposal,
                commit_id,
                object_ids,
                inquiry_ids,
                omitted_local_refs,
                omitted_inquiries,
                alias_add_decisions,
                omitted_new_objects,
                collision_issues,
                committed_at=(replay_receipt.committed_at if replay_receipt else datetime.now(UTC)),
                replay_delta=replay_delta,
                replay_attempt_recorded=attempt_recorded,
            )
            recorded_review = (
                self._recorded_attempt_review(attempt, commit_id)
                if attempt_recorded and replay_delta is not None
                else None
            )
            if recorded_review is not None:
                review_issues = recorded_review.issues
            touched = {
                ref
                for new_inquiry in proposal.new_inquiries
                for ref in (new_inquiry.deepens_inquiry_id, new_inquiry.answers_inquiry_id)
                if ref
            }
            # spec I4-5: 提案引用该 inquiry（deepens/answers/resolve）时 +1。
            # answers 同时覆盖 new inquiry 与 assertion；assertion 的引用以
            # 编译保留项（delta.assertions）为准——保留断言的 answers 目标
            # 经 store 存在性校验保证可达，省略断言（重复/冲突）的引用不计。
            touched.update(
                assertion.answers_inquiry_id
                for assertion in delta.assertions
                if assertion.answers_inquiry_id is not None
            )
            if replay_delta is not None:
                # replay/backfill 与 legacy 恢复只认存储 delta 的权威记录，
                # 绝不用当前数据库状态或调用方当前提供的 proposal 重新解释
                # 历史：① new inquiry 的 deepens/answers 是 proposal-local
                # 字段，不进持久化 delta——调用方重放时任意添加都无法被
                # delta 相等性校验发现，backfill/恢复一律不写其 ledger、不
                # 递增；② assertion answers 在 6d 之前的首次执行从未 touch
                # 过，存储状态无法区分新旧提交，重放一律不补（不制造
                # attempt_count=0 而 ledger=1 的不一致）；③ 仅保留 resolution
                # 是各版本首次执行都 touch 过的部分，是唯一可证明的历史
                # touch 来源。被省略的 resolution 不在 delta 中，重放也不得
                # 为其补造 ledger 行（首次 MISSING 的 inquiry 后来实体化时
                # 同样不触碰）。
                touched = {resolution.id for resolution in delta.resolve_inquiries}
            else:
                # 首次执行：编译后省略但有效的 resolution（stale version /
                # 已终状态 / 无 answering assertion）仍是一次真实尝试，须
                # 计入 touched；幻觉 inquiry ID 不在库中，绝不 touch。
                touched.update(
                    item.memory_id
                    for item in proposal.resolve_inquiries
                    if item.memory_id not in missing_resolution_ids
                )
            outcome = (
                recorded_review.outcome
                if recorded_review is not None
                else (
                    ReviewOutcome.COMMIT_WITH_WARNINGS
                    if review_issues
                    else ReviewOutcome.ACCEPT
                )
            )

            def finalize_commit(
                connection: sqlite3.Connection, durable_receipt: CommitReceipt
            ) -> None:
                inserted = self._record_attempt(
                    attempt=attempt,
                    durable_commit_id=durable_receipt.commit_id,
                    outcome=outcome.value,
                    proposal=proposal,
                    omitted_assertions=len(omitted_assertions),
                    omitted_inquiries=len(omitted_inquiries),
                    omitted_resolutions=len(omitted_resolutions),
                    resolved_inquiries=len(proposal.resolve_inquiries),
                    evidence_missing_assertions=len(evidence_missing_assertions),
                    error=None,
                    issues=review_issues,
                    connection=connection,
                )
                if inserted and durable_receipt.replayed:
                    self._store.assert_legacy_touches_not_applied(
                        touched,
                        durable_receipt.committed_at,
                        connection=connection,
                    )
                for ref in sorted(touched):
                    if inserted:
                        self._store.touch_inquiry(
                            ref,
                            durable_receipt.committed_at,
                            run_commit_id=attempt.run_commit_id,
                            connection=connection,
                        )
                    else:
                        self._store.record_inquiry_touch(
                            ref,
                            durable_receipt.committed_at,
                            run_commit_id=attempt.run_commit_id,
                            connection=connection,
                        )

            receipt = self._store.finalized_memory_commit(delta, commit_id, finalize_commit)
        except ObjectAliasCollision as error:
            issues = self._store_alias_collision_issues(proposal, error)
            if replay_receipt is None:
                self._record_rejection(attempt, proposal, error, issues)
            raised = ProposalValidationError([str(error)], [error.existing_object_id])
            raised.review_issues = issues
            raised.answerable_inquiries = self._rejected_proposal_answerable(proposal, object_ids)
            raise raised from error
        except ProposalValidationError as error:
            if not error.review_issues:
                # Task 2.2: ghost durable references map to an indexed
                # invalid_reference issue whose candidate_ids (recovered from
                # the transcript at the shared 0.20 line) only decorate the
                # repair feedback — they never rewrite the reference.
                error.review_issues = [
                    self._issue_from_error(error, proposal, missing_memory_candidates)
                ]
            if replay_receipt is None:
                self._record_rejection(attempt, proposal, error, error.review_issues)
            error.answerable_inquiries = self._rejected_proposal_answerable(proposal, object_ids)
            raise
        except (AmbiguousLegacyTouchState, CommitReplayConflict) as error:
            issue = self._issue_from_error(error, proposal)
            raised = ProposalValidationError([str(error)])
            raised.review_issues = [issue]
            raised.answerable_inquiries = self._rejected_proposal_answerable(proposal, object_ids)
            raise raised from error
        except ValueError as error:
            issue = self._issue_from_error(error, proposal)
            if replay_receipt is None:
                self._record_rejection(attempt, proposal, error, [issue])
            raised = ProposalValidationError([str(error)])
            raised.review_issues = [issue]
            raised.answerable_inquiries = self._rejected_proposal_answerable(proposal, object_ids)
            raise raised from error
        return ProposalCommitReceipt(
            commit=receipt,
            review_outcome=outcome,
            review_issues=review_issues,
            attempt=attempt,
            object_ids_by_local_ref=object_ids,
            inquiry_ids_by_local_ref=inquiry_ids,
            omitted_assertion_indexes=omitted_assertions,
            evidence_missing_assertion_indexes=evidence_missing_assertions,
            omitted_resolution_ids=omitted_resolutions,
            omitted_inquiry_indexes=omitted_inquiries,
            deepened_inquiry_ids=deepened,
            answerable_inquiries=answerable,
            dropped_evidence=dropped_evidence,
            resolved_object_refs=resolved_object_refs,
        )

    def _rejected_proposal_answerable(
        self, proposal: CognitionDeltaProposal, object_ids: dict[str, str]
    ) -> list[str]:
        """Compute the receipt's exact answerable set for one rejected proposal.

        The receipt's rule (see :meth:`_compile`): every assertion that does
        not declare ``answers_inquiry_id`` surfaces the open/dormant
        inquiries on its subject. A rejection happens before compile, so the
        same rule is applied to the rejected proposal's own assertions —
        the one moment the proposal exists — letting the repair feedback
        name the exact answerable set that the finalization-time
        approximation (transcript-surfaced subjects only) cannot. Closed
        inquiries are excluded by the same ``status IN ('open', 'dormant')``
        predicate the receipt uses. Subject resolution is the offline
        ``_subject_id`` form: a proposal-local subject (a new object) never
        matches any stored inquiry, and an existing subject is its durable
        id verbatim.

        Args:
            proposal: The rejected proposal whose assertions define the set.
            object_ids: Proposal-local object refs to their durable ids.

        Returns:
            The deduplicated answerable inquiry ids, query order preserved.

        """
        answerable: list[str] = []
        with self._store.read_connection() as connection:
            for item in proposal.assertions:
                if item.answers_inquiry_id is not None:
                    continue
                subject_id = self._subject_id(item.subject, object_ids)
                rows = connection.execute(
                    "SELECT id FROM inquiries"
                    " WHERE status IN ('open', 'dormant') AND subject_id = ?",
                    (subject_id,),
                ).fetchall()
                answerable.extend(str(row["id"]) for row in rows)
        return list(dict.fromkeys(answerable))

    def _record_rejection(
        self,
        attempt: AttemptContext,
        proposal: CognitionDeltaProposal,
        error: Exception,
        issues: list[ReviewIssue] | None = None,
    ) -> None:
        """Record one rejected attempt with zero omitted counts (compile never ran)."""
        self._record_attempt(
            attempt=attempt,
            durable_commit_id=None,
            outcome=ReviewOutcome.REJECT_AND_REPAIR.value,
            proposal=proposal,
            omitted_assertions=0,
            omitted_inquiries=0,
            omitted_resolutions=0,
            resolved_inquiries=len(proposal.resolve_inquiries),
            evidence_missing_assertions=0,
            error=str(error)[:2000],
            issues=issues or [],
        )

    def reject_attempt(
        self,
        proposal: CognitionDeltaProposal,
        *,
        attempt: AttemptContext,
        error: ProposalValidationError,
        issue: ReviewIssue,
    ) -> None:
        """Append a graph-boundary rejection that never reached compilation."""
        error.review_issues = [issue]
        self._assert_attempt_available(attempt, proposal)
        self._record_rejection(attempt, proposal, error, [issue])

    def record_identical_resubmission(
        self,
        proposal: CognitionDeltaProposal,
        *,
        attempt: AttemptContext,
        previous_attempt_id: str,
    ) -> None:
        """Append the intercepted resubmission to the append-only attempt ledger.

        Spec §5.1: a repair-round resubmission byte-identical (after
        normalization) to the previously rejected proposal is never offered to
        the model again; this row records that interception with an explicit
        ``identical_resubmission`` marker so the audit trail stays continuous
        with the rejected attempt it repeats.
        """
        self._assert_attempt_available(attempt, proposal)
        self._record_attempt(
            attempt=attempt,
            durable_commit_id=None,
            outcome=ReviewOutcome.REJECT_AND_REPAIR.value,
            proposal=proposal,
            omitted_assertions=0,
            omitted_inquiries=0,
            omitted_resolutions=0,
            resolved_inquiries=len(proposal.resolve_inquiries),
            evidence_missing_assertions=0,
            error=(
                "identical_resubmission: proposal byte-identical after normalization "
                f"to rejected attempt {previous_attempt_id}"
            ),
            issues=[],
        )

    def _record_attempt(
        self,
        *,
        attempt: AttemptContext,
        durable_commit_id: str | None,
        outcome: str,
        proposal: CognitionDeltaProposal,
        omitted_assertions: int,
        omitted_inquiries: int,
        omitted_resolutions: int,
        resolved_inquiries: int,
        evidence_missing_assertions: int,
        error: str | None,
        issues: list[ReviewIssue],
        connection: sqlite3.Connection | None = None,
    ) -> bool:
        """Append one attempt, returning whether this call inserted its audit row.

        A successful proposal passes its world-commit connection so the audit
        row is the idempotency gate for inquiry touches in the same transaction.
        Rejections retain their standalone audit transaction.
        """
        payload = proposal.model_dump_json()
        serialized_issues = json.dumps(
            [issue.model_dump(mode="json") for issue in issues],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        def write(target: sqlite3.Connection) -> bool:
            existing = target.execute(
                "SELECT run_commit_id, durable_commit_id, attempt_no, parent_attempt_id, outcome, "
                "delta_json, "
                "issues_json FROM proposal_attempts WHERE attempt_id = ?",
                (attempt.attempt_id,),
            ).fetchone()
            if existing is not None:
                same_identity_and_payload = (
                    existing["run_commit_id"] == attempt.run_commit_id
                    and existing["attempt_no"] == attempt.attempt_no
                    and existing["parent_attempt_id"] == attempt.parent_attempt_id
                    and existing["delta_json"] == payload
                )
                if (
                    same_identity_and_payload
                    and existing["durable_commit_id"] == durable_commit_id
                ):
                    return False
                raise ProposalValidationError(
                    [f"attempt id already has different payload: {attempt.attempt_id}"]
                )
            target.execute(
                "INSERT INTO proposal_attempts"
                "(attempt_id, commit_id, run_commit_id, durable_commit_id, attempt_no, parent_attempt_id,"
                " thread_id, attempted_at, outcome, new_objects, assertions,"
                " inquiries, omitted_assertions, omitted_inquiries, omitted_resolutions,"
                " resolved_inquiries, evidence_missing_assertions, error, delta_json, issues_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt.attempt_id,
                    durable_commit_id or attempt.run_commit_id,
                    attempt.run_commit_id,
                    durable_commit_id,
                    attempt.attempt_no,
                    attempt.parent_attempt_id,
                    self._thread_id if hasattr(self, "_thread_id") else "",
                    datetime.now(UTC).isoformat(),
                    outcome,
                    len(proposal.new_objects),
                    len(proposal.assertions),
                    len(proposal.new_inquiries),
                    omitted_assertions,
                    omitted_inquiries,
                    omitted_resolutions,
                    resolved_inquiries,
                    evidence_missing_assertions,
                    error,
                    payload,
                    serialized_issues,
                ),
            )
            return True

        if connection is not None:
            return write(connection)
        with self._store.write_connection() as owned_connection:
            return write(owned_connection)

    def _legacy_attempt_context(self, proposal: CognitionDeltaProposal, commit_id: str) -> AttemptContext:
        """Give unchanged callers replay-safe attempt identity without reusing a commit id."""
        digest = hashlib.sha256(proposal.model_dump_json().encode()).hexdigest()[:16]
        return AttemptContext(
            run_commit_id=commit_id,
            attempt_id=f"{commit_id}:legacy:{digest}",
            attempt_no=1,
        )

    def _assert_attempt_available(
        self, attempt: AttemptContext, proposal: CognitionDeltaProposal
    ) -> bool:
        """Reject a conflicting replay before it can write another durable commit."""
        with self._store.read_connection() as connection:
            row = connection.execute(
                "SELECT run_commit_id, attempt_no, delta_json FROM proposal_attempts WHERE attempt_id = ?",
                (attempt.attempt_id,),
            ).fetchone()
            if row is not None and (
                row["run_commit_id"] != attempt.run_commit_id
                or row["attempt_no"] != attempt.attempt_no
                or row["delta_json"] != proposal.model_dump_json()
            ):
                raise ProposalValidationError(
                    [f"attempt id already has different payload: {attempt.attempt_id}"]
                )
            number = connection.execute(
                "SELECT attempt_id FROM proposal_attempts WHERE run_commit_id = ? AND attempt_no = ?",
                (attempt.run_commit_id, attempt.attempt_no),
            ).fetchone()
            if number is not None and number["attempt_id"] != attempt.attempt_id:
                raise ProposalValidationError(
                    [f"attempt number already exists for run: {attempt.run_commit_id}:{attempt.attempt_no}"]
                )
            return row is not None

    def _recorded_attempt_review(
        self,
        attempt: AttemptContext,
        commit_id: str,
    ) -> _RecordedAttemptReview | None:
        """Restore review fields that can drift while the durable delta stays equal.

        Material depth and provenance can legitimately improve after a commit.
        Recomputing warnings against that newer state would make an idempotent
        replay disagree with the append-only attempt audit. Omission and
        evidence-drop fields remain compile-derived: changing any of them also
        changes the CognitiveDelta and is rejected by Store replay equality.
        """
        with self._store.read_connection() as connection:
            row = connection.execute(
                "SELECT durable_commit_id, outcome, issues_json "
                "FROM proposal_attempts WHERE attempt_id = ?",
                (attempt.attempt_id,),
            ).fetchone()
        if row is None or row["durable_commit_id"] != commit_id:
            return None
        try:
            outcome = ReviewOutcome(str(row["outcome"]))
            raw_issues = json.loads(str(row["issues_json"]))
            if not isinstance(raw_issues, list):
                raise ValueError("issues_json is not a list")
            issues = [ReviewIssue.model_validate(issue) for issue in raw_issues]
            valid_accept = outcome is ReviewOutcome.ACCEPT and not issues
            valid_warnings = (
                outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
                and bool(issues)
                and all(issue.severity == "warning" for issue in issues)
            )
            if not (valid_accept or valid_warnings):
                raise ValueError("invalid successful review outcome/issues combination")
        except (TypeError, ValueError) as error:
            raise CommitReplayConflict(
                f"proposal attempt has invalid durable review audit: {attempt.attempt_id}"
            ) from error
        return _RecordedAttemptReview(outcome=outcome, issues=issues)

    @staticmethod
    def _issue(
        code: ReviewIssueCode,
        severity: str,
        item_kind: str,
        item_index: int | None,
        message: str,
        actions: list[str],
        actual_value: JsonValue,
        candidate_ids: list[str] | None = None,
        dropped_evidence_ids: list[str] | None = None,
        durable_id: str | None = None,
        match_basis: list[str] | None = None,
        omitted_dependencies: list[dict[str, JsonValue]] | None = None,
    ) -> ReviewIssue:
        marker = f"{item_kind}:{item_index if item_index is not None else durable_id or 'proposal'}"
        return ReviewIssue(
            issue_id=f"{code.value}:{marker}",
            code=code,
            severity=severity,  # type: ignore[arg-type]
            failed_rule=code.value,
            actual_value=actual_value,
            item_kind=item_kind,
            item_index=item_index,
            durable_id=durable_id,
            message=message,
            suggested_actions=actions,
            candidate_ids=list(dict.fromkeys(candidate_ids or [])),
            dropped_evidence_ids=list(dict.fromkeys(dropped_evidence_ids or [])),
            match_basis=list(dict.fromkeys(match_basis or [])),
            omitted_dependencies=list(omitted_dependencies or []),
        )

    def _issue_from_error(
        self,
        error: Exception,
        proposal: CognitionDeltaProposal | None = None,
        missing_memory_candidates: dict[str, list[str]] | None = None,
    ) -> ReviewIssue:
        message = str(error)
        # F1 accessory (spec §5): a proposal-level rejection is never
        # anonymous — when the message names a proposal item, the issue
        # points at it (resolve_inquiry index + inquiry id, or the declaring
        # assertion) with a concrete repair hint instead of item_index=None.
        if proposal is not None:
            for index, item in enumerate(proposal.resolve_inquiries):
                if item.memory_id in message:
                    if "status mismatch" in message:
                        actions = ["remove_resolution_but_keep_assertions"]
                    elif "stale" in message:
                        actions = ["refresh_expected_version", "remove_resolution_but_keep_assertions"]
                    else:
                        actions = [
                            "add_answering_assertion_declaring_answers_inquiry_id",
                            "keep_inquiry_open",
                        ]
                    return self._issue(
                        ReviewIssueCode.STALE_RESOLUTION,
                        "error",
                        "resolve_inquiry",
                        index,
                        message,
                        actions,
                        actual_value={"inquiry_id": item.memory_id, "error": message},
                        durable_id=item.memory_id,
                    )
            for index, item in enumerate(proposal.assertions):
                if (
                    item.answers_inquiry_id is not None
                    and item.answers_inquiry_id in message
                ):
                    return self._issue(
                        ReviewIssueCode.UNKNOWN_MEMORY_ID,
                        "error",
                        "assertion",
                        index,
                        message,
                        ["remove_answers_inquiry_id", "use_real_inquiry_id"],
                        actual_value={"answers_inquiry_id": item.answers_inquiry_id, "error": message},
                        durable_id=item.answers_inquiry_id,
                    )
        if "local reference" in message:
            return self._issue(
                ReviewIssueCode.UNKNOWN_LOCAL_REFERENCE,
                "error",
                "proposal",
                None,
                message,
                ["use_real_store_id", "remove_item"],
                actual_value={"error": message},
            )
        if "memory does not exist" in message:
            # Task 2.2: a durable memory id that names no stored object is an
            # invalid_reference (basis missing_memory_id) — never a fuzzy
            # rewrite; the item-indexed scan below attributes the ghost id to
            # its proposal item and may decorate it with transcript-recovered
            # stored-object candidates.
            return self._missing_memory_issue(message, proposal, missing_memory_candidates)
        if "stale" in message:
            return self._issue(
                ReviewIssueCode.STALE_VERSION,
                "error",
                "proposal",
                None,
                message,
                ["refresh_expected_version", "remove_item"],
                actual_value={"error": message},
            )
        return self._issue(
            ReviewIssueCode.UNKNOWN_MEMORY_ID,
            "error",
            "proposal",
            None,
            message,
            ["use_real_store_id", "remove_item"],
            actual_value={"error": message},
        )

    def _missing_memory_issue(
        self,
        message: str,
        proposal: CognitionDeltaProposal | None,
        missing_memory_candidates: dict[str, list[str]] | None,
    ) -> ReviewIssue:
        """Map one ghost durable reference to an indexed invalid_reference issue.

        The offending proposal item is located by scanning the same object
        refs the graph's candidate search covers (assertion subject/object,
        inquiry subject, object-update target, observation-link object
        targets); the issue names the ghost id as ``durable_id`` and may
        carry transcript-recovered stored-object candidates — candidates
        only, the reference itself stays unresolved. A message attributable
        to no item falls back to a proposal-level issue parsing the ghost id
        out of the error text.
        """
        if proposal is not None:
            for index, item in enumerate(proposal.assertions):
                for ref in (item.subject, item.object):
                    if ref is not None and ref.memory_id is not None and ref.memory_id in message:
                        missing_candidate_ids, missing_candidate_tail = _candidate_payload(
                            (missing_memory_candidates or {}).get(ref.memory_id)
                        )
                        return self._issue(
                            ReviewIssueCode.INVALID_REFERENCE,
                            "error",
                            "assertion",
                            index,
                            message,
                            ["use_real_store_id", "declare_item_with_local_ref", "remove_item"],
                            actual_value={
                                "memory_id": ref.memory_id,
                                "error": message,
                                **missing_candidate_tail,
                            },
                            candidate_ids=missing_candidate_ids,
                            durable_id=ref.memory_id,
                            match_basis=["missing_memory_id"],
                        )
            for index, item in enumerate(proposal.new_inquiries):
                ref = item.subject
                if ref.memory_id is not None and ref.memory_id in message:
                    missing_candidate_ids, missing_candidate_tail = _candidate_payload(
                        (missing_memory_candidates or {}).get(ref.memory_id)
                    )
                    return self._issue(
                        ReviewIssueCode.INVALID_REFERENCE,
                        "error",
                        "new_inquiry",
                        index,
                        message,
                        ["use_real_store_id", "declare_item_with_local_ref", "remove_item"],
                        actual_value={
                            "memory_id": ref.memory_id,
                            "error": message,
                            **missing_candidate_tail,
                        },
                        candidate_ids=missing_candidate_ids,
                        durable_id=ref.memory_id,
                        match_basis=["missing_memory_id"],
                    )
            for index, item in enumerate(proposal.object_updates):
                ref = item.target
                if ref.memory_id is not None and ref.memory_id in message:
                    missing_candidate_ids, missing_candidate_tail = _candidate_payload(
                        (missing_memory_candidates or {}).get(ref.memory_id)
                    )
                    return self._issue(
                        ReviewIssueCode.INVALID_REFERENCE,
                        "error",
                        "object_update",
                        index,
                        message,
                        ["use_real_store_id", "declare_item_with_local_ref", "remove_item"],
                        actual_value={
                            "memory_id": ref.memory_id,
                            "error": message,
                            **missing_candidate_tail,
                        },
                        candidate_ids=missing_candidate_ids,
                        durable_id=ref.memory_id,
                        match_basis=["missing_memory_id"],
                    )
            for index, item in enumerate(proposal.observation_links):
                if item.target_kind != "object":
                    continue
                ref = item.target
                if ref.memory_id is not None and ref.memory_id in message:
                    missing_candidate_ids, missing_candidate_tail = _candidate_payload(
                        (missing_memory_candidates or {}).get(ref.memory_id)
                    )
                    return self._issue(
                        ReviewIssueCode.INVALID_REFERENCE,
                        "error",
                        "observation_link",
                        index,
                        message,
                        ["use_real_store_id", "declare_item_with_local_ref", "remove_item"],
                        actual_value={
                            "memory_id": ref.memory_id,
                            "error": message,
                            **missing_candidate_tail,
                        },
                        candidate_ids=missing_candidate_ids,
                        durable_id=ref.memory_id,
                        match_basis=["missing_memory_id"],
                    )
        ghost = message.rsplit("memory does not exist: ", 1)[-1].strip()
        ghost_candidate_ids, ghost_candidate_tail = _candidate_payload(
            (missing_memory_candidates or {}).get(ghost)
        )
        return self._issue(
            ReviewIssueCode.INVALID_REFERENCE,
            "error",
            "proposal",
            None,
            message,
            ["use_real_store_id", "declare_item_with_local_ref", "remove_item"],
            actual_value={"error": message, **ghost_candidate_tail},
            candidate_ids=ghost_candidate_ids,
            durable_id=ghost if ghost and ghost != message else None,
            match_basis=["missing_memory_id"],
        )

    def _store_alias_collision_issues(
        self, proposal: CognitionDeltaProposal, error: ObjectAliasCollision
    ) -> list[ReviewIssue]:
        """Attach store-raised identity alias conflicts to their proposal item.

        The compiler omits every proposal-reachable cross-object identity
        conflict, so this path is a backstop — e.g. a conflict introduced by
        a competing writer between the compiler's read and the store's write
        (race test), or a delta compiled outside this committer.
        """
        actual_value: JsonValue = {
            "alias": error.normalized_alias,
            "existing_object_id": error.existing_object_id,
        }
        issues: list[ReviewIssue] = []
        for index, item in enumerate(proposal.new_objects):
            if any(_identity_alias_form(alias) == error.normalized_alias for alias in item.aliases):
                issues.append(
                    self._issue(
                        ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                        "error",
                        "new_object",
                        index,
                        (
                            "The new object declares an identity alias already claimed by "
                            "an existing object; it was omitted without reusing the owner."
                        ),
                        [
                            "use_existing_object_id",
                            "remove_duplicate_new_object",
                            "choose_unclaimed_alias",
                        ],
                        actual_value=actual_value,
                        candidate_ids=[error.existing_object_id],
                        match_basis=["identity_alias_active_owner"],
                    )
                )
        for index, item in enumerate(proposal.object_updates):
            if any(_identity_alias_form(alias) == error.normalized_alias for alias in item.add_aliases):
                issues.append(
                    self._issue(
                        ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                        "error",
                        "object_update",
                        index,
                        "The object update adds an identity alias claimed by another stored object.",
                        [
                            "omit_conflicting_object_update",
                            "keep_other_safe_items",
                            "use_unclaimed_alias",
                        ],
                        durable_id=str(item.target.memory_id),
                        actual_value=actual_value,
                        candidate_ids=[error.existing_object_id],
                        match_basis=["identity_alias_active_owner"],
                    )
                )
        if issues:
            return issues
        return [
            self._issue(
                ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                "error",
                "proposal",
                None,
                "A store identity alias conflict could not be mapped to a proposal item.",
                ["remove_conflicting_item"],
                actual_value=actual_value,
                candidate_ids=[error.existing_object_id],
            )
        ]

    @staticmethod
    def _fuzzy_object_candidate_ids(
        connection: sqlite3.Connection,
        names: Iterable[str],
        own_id: str,
        limit: int = 9,
    ) -> tuple[list[str], bool]:
        """Bound the advisory fuzzy-name candidate set through the FTS index (F6).

        Renders every probe name as escaped FTS5 phrases (see
        :func:`_fts_phrase`): each whitespace token as an exact phrase, plus
        the two-character grams of the normalized name as prefix phrases —
        the same sliding grams the 0.20 bigram/Jaccard gate compares, so a
        stored token that merely overlaps a probe gram ("Bing" vs "Bin")
        still reaches the bounded set, mirroring recall's CJK expansion.
        Retrieves at most ``limit`` object ids from ``objects_fts`` in rank
        order — replacing the former full-table scans of
        ``objects``/``identity_aliases``/``object_aliases``. The bigram/
        Jaccard check runs only over the bounded ids; an advisory fuzzy miss
        never authorizes a merge or rejection, so bounded recall is the
        correct failure direction.

        Malformed model-provided probes degrade safely: NUL bytes are
        stripped (FTS5 parses them as end-of-string and would raise
        "unterminated string"), and a residual FTS5 parser error falls back
        to no candidates — a raw parser failure can never abort an
        otherwise valid commit.

        Returns:
            A ``(ids, limit_hit)`` pair: the bounded object ids in FTS rank
            order (own id excluded), and whether the pre-selection hit its
            LIMIT — the signal that rows beyond the sample were never
            classified, so the caller reports ``candidate_sample_truncated``
            (final set fits the cap, total unknown) or ``candidate_has_more``
            (final set was cut too, a confirmed lower bound) instead of an
            exact omitted count.

        """
        phrases: list[str] = []
        for name in dict.fromkeys(str(value) for value in names if value):
            cleaned = str(name).replace("\x00", "")
            tokens = [token for token in cleaned.split() if token]
            phrases.extend(_fts_phrase(token) for token in tokens)
            folded = normalize(cleaned)
            phrases.extend(
                f'"{folded[index : index + 2]}"*' for index in range(len(folded) - 1)
            )
        if not phrases:
            return [], False
        phrases = list(dict.fromkeys(phrases))
        try:
            rows = connection.execute(
                "SELECT id FROM objects_fts WHERE objects_fts MATCH ? ORDER BY rank, id LIMIT ?",
                (" OR ".join(phrases), limit),
            ).fetchall()
        except sqlite3.Error:
            # Fail closed on a malformed probe: the advisory fuzzy miss
            # degrades to no candidates and never rejects or merges.
            return [], False
        candidates: list[str] = []
        for row in rows:
            candidate = str(row["id"])
            if candidate != own_id and candidate not in candidates:
                candidates.append(candidate)
        return candidates, len(rows) == limit

    def _decide_new_object_collision(
        self,
        index: int,
        item: NewObjectProposal,
        *,
        kind: ObjectKind,
        own_id: str,
        connection: sqlite3.Connection,
        claimed_identity_aliases: dict[str, str],
    ) -> NewObjectCollisionDecision:
        """Decide one new object's collision fate (Task 2.1 matrix).

        Matrix rows, evaluated in order:

        1. An ACTIVE ``identity_aliases`` owner of any declared alias (the
           item's own replay identity excluded; the proposal-local claimed
           map consulted first so intra-delta claims behave like stored
           ones) → omit with one ``identity_alias_claim_conflict`` issue;
           the owner is returned as the repair candidate and the local ref
           is never rewritten to it.
        2. A canonical-name exact match (case-insensitive) → allow with an
           ``ambiguous_name_candidates`` warning returning every same-name
           object.
        3. A legacy ``object_aliases`` exact match → folded into the same
           ``ambiguous_name_candidates`` warning with basis
           ``legacy_name_read_only``; never a strong conflict.
        4. Domain-hint overlap between the item and its name-matched
           candidates, or a fuzzy-only name hit (no exact anchor) against
           any stored object at the shared 0.20 bigram-jaccard line → a
           ``duplicate_object_candidate`` warning; still allowed, never a
           rewrite.

        The decision is replay-deterministic by construction: a pure
        function of the proposal item, the read-only alias/object tables,
        ``commit_id``, and the caller-owned proposal-local claim map.
        """
        identity_conflicts: list[tuple[str, str]] = []
        seen_identity_forms: set[str] = set()
        for alias in item.aliases:
            normalized = _identity_alias_form(alias)
            if not normalized or normalized in seen_identity_forms:
                continue
            seen_identity_forms.add(normalized)
            owner = claimed_identity_aliases.get(normalized)
            if owner is None:
                row = connection.execute(
                    "SELECT object_id FROM identity_aliases"
                    " WHERE normalized_alias = ? AND status = 'active' ORDER BY object_id LIMIT 1",
                    (normalized,),
                ).fetchone()
                owner = str(row["object_id"]) if row is not None else None
            if owner is not None and owner != own_id:
                identity_conflicts.append((alias, owner))
        if identity_conflicts:
            owner_ids, owner_tail = _candidate_payload(
                owner for _, owner in identity_conflicts
            )
            aliases = list(dict.fromkeys(alias for alias, _ in identity_conflicts))
            issue = self._issue(
                ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                "warning",
                "new_object",
                index,
                (
                    "The new object declares an identity alias already claimed by an "
                    "existing object; it was omitted without reusing the owner."
                ),
                [
                    "use_existing_object_id",
                    "remove_duplicate_new_object",
                    "choose_unclaimed_alias",
                ],
                actual_value={
                    "aliases": aliases,
                    "existing_object_ids": owner_ids,
                    **owner_tail,
                },
                candidate_ids=owner_ids,
                match_basis=["identity_alias_active_owner"],
            )
            return NewObjectCollisionDecision(
                omit=True,
                issue=issue,
                warning_issues=(),
                collision_pairs=tuple((index, alias, owner) for alias, owner in identity_conflicts),
            )
        # Rows 2-3: canonical exact and legacy exact hits are candidates for
        # a warning, never for a rewrite.
        candidates: list[str] = []
        match_basis: list[str] = []
        for row in connection.execute(
            "SELECT id FROM objects WHERE canonical_name = ? COLLATE NOCASE",
            (item.canonical_name,),
        ).fetchall():
            candidate = str(row["id"])
            if candidate != own_id and candidate not in candidates:
                candidates.append(candidate)
        if candidates:
            match_basis.append("canonical_name_exact")
        legacy_owners: list[str] = []
        for alias in dict.fromkeys([item.canonical_name, *item.aliases]):
            forms = alias_lookup_forms(alias)
            if not forms[0]:
                continue
            marks = ", ".join("?" for _ in forms)
            for row in connection.execute(
                f"SELECT object_id FROM object_aliases WHERE normalized_alias IN ({marks})",
                forms,
            ).fetchall():
                owner = str(row["object_id"])
                if owner != own_id and owner not in legacy_owners:
                    legacy_owners.append(owner)
        if legacy_owners:
            candidates.extend(legacy_owners)
            match_basis.append("legacy_name_read_only")
        issues: list[ReviewIssue] = []
        if candidates:
            candidate_ids, candidate_tail = _candidate_payload(candidates)
            issues.append(
                self._issue(
                    ReviewIssueCode.AMBIGUOUS_NAME_CANDIDATES,
                    "warning",
                    "new_object",
                    index,
                    (
                        "The new object shares a name with existing stored objects; it "
                        "was created with the same-name candidates for review."
                    ),
                    [
                        "keep_new_object",
                        "reference_existing_object_id",
                        "choose_distinct_canonical_name",
                    ],
                    actual_value={
                        "canonical_name": item.canonical_name,
                        "existing_object_ids": candidate_ids,
                        **candidate_tail,
                    },
                    candidate_ids=candidate_ids,
                    match_basis=match_basis,
                )
            )
        # Row 4: domain overlap with the name-matched candidates only.
        overlap_owners: list[str] = []
        if item.domain_hints:
            for candidate in dict.fromkeys(candidates):
                row = connection.execute(
                    "SELECT domain_hints_json FROM objects WHERE id = ?", (candidate,)
                ).fetchone()
                if row is None:
                    continue
                if set(item.domain_hints) & set(json.loads(row["domain_hints_json"])):
                    overlap_owners.append(candidate)
        if overlap_owners:
            overlap_ids, overlap_tail = _candidate_payload(overlap_owners)
            issues.append(
                self._issue(
                    ReviewIssueCode.DUPLICATE_OBJECT_CANDIDATE,
                    "warning",
                    "new_object",
                    index,
                    (
                        "The new object's domain hints overlap an existing same-name "
                        "object; it was created without rewriting its reference."
                    ),
                    [
                        "keep_new_object",
                        "reference_existing_object_id",
                        "narrow_domain_hints",
                    ],
                    actual_value={
                        "domain_hints": item.domain_hints,
                        "existing_object_ids": overlap_ids,
                        **overlap_tail,
                    },
                    candidate_ids=overlap_ids,
                    match_basis=["domain_hints_overlap"],
                )
            )
        # Row 4, fuzzy arm: with no canonical/legacy anchor, a name that
        # resembles any stored object (canonical name or stored alias form,
        # best pair at the shared 0.20 bigram-jaccard line) is a duplicate
        # candidate warning. It never blocks creation and never rewrites the
        # local ref.
        fuzzy_owners: list[str] = []
        if not candidates:
            # F6: the FTS5 index replaces the three full-table name scans —
            # bounded recall is the correct failure direction for an advisory
            # fuzzy warning, and the bigram/Jaccard check runs only over the
            # bounded ids and their stored active/legacy names.
            names = [name for name in [item.canonical_name, *item.aliases] if name]
            if names:
                bounded_ids, fuzzy_limit_hit = self._fuzzy_object_candidate_ids(
                    connection, names, own_id
                )
                probes = [probe for name in names if (probe := bigrams(name))]
                if bounded_ids:
                    marks = ", ".join("?" for _ in bounded_ids)
                    stored_names: dict[str, list[str]] = {}
                    for row in connection.execute(
                        f"SELECT id, canonical_name FROM objects WHERE id IN ({marks})",
                        bounded_ids,
                    ).fetchall():
                        stored_names.setdefault(str(row["id"]), [str(row["canonical_name"])])
                    for row in connection.execute(
                        f"SELECT object_id, normalized_alias FROM identity_aliases"
                        f" WHERE status = 'active' AND object_id IN ({marks}) ORDER BY object_id",
                        bounded_ids,
                    ).fetchall():
                        stored_names.setdefault(str(row["object_id"]), []).append(
                            str(row["normalized_alias"])
                        )
                    for row in connection.execute(
                        f"SELECT object_id, normalized_alias FROM object_aliases"
                        f" WHERE object_id IN ({marks}) ORDER BY object_id",
                        bounded_ids,
                    ).fetchall():
                        stored_names.setdefault(str(row["object_id"]), []).append(
                            str(row["normalized_alias"])
                        )
                    for object_id, names in stored_names.items():
                        if object_id == own_id:
                            continue
                        if any(
                            jaccard(probe, bigrams(name)) >= _FUZZY_SIMILARITY_THRESHOLD
                            for probe in probes
                            for name in names
                        ):
                            fuzzy_owners.append(object_id)
        if fuzzy_owners:
            # F6 + Task 4: the fuzzy source is pre-capped by its SQL LIMIT.
            # A truncated pre-selection is only ever the *sample* being cut —
            # the Jaccard classification runs after it, so the final set may
            # still fit under the cap (total unknown -> candidate_sample_
            # truncated) or itself be cut (one more confirmed ->
            # candidate_has_more). Either way the payload never looks exact.
            fuzzy_ids, fuzzy_tail = _candidate_payload(
                fuzzy_owners, sampled=True, truncated=fuzzy_limit_hit
            )
            issues.append(
                self._issue(
                    ReviewIssueCode.DUPLICATE_OBJECT_CANDIDATE,
                    "warning",
                    "new_object",
                    index,
                    (
                        "The new object's name resembles an existing stored object; it "
                        "was created without rewriting its reference."
                    ),
                    [
                        "keep_new_object",
                        "reference_existing_object_id",
                        "narrow_domain_hints",
                    ],
                    actual_value={
                        "canonical_name": item.canonical_name,
                        "existing_object_ids": fuzzy_ids,
                        **fuzzy_tail,
                    },
                    candidate_ids=fuzzy_ids,
                    match_basis=["fuzzy_name_match"],
                )
            )
        return NewObjectCollisionDecision(
            omit=False,
            issue=None,
            warning_issues=tuple(issues),
            collision_pairs=(),
        )

    @staticmethod
    def _omitted_dependencies(
        proposal: CognitionDeltaProposal, local_ref: str
    ) -> list[dict[str, JsonValue]]:
        """Enumerate the proposal items referencing one omitted new object.

        Mirrors the compile-time omission rules: an assertion is a dependency
        when it references the local ref as subject or object; an inquiry
        when the ref is its subject; an observation link when it targets the
        ref as an object. Each record carries the dependent's kind, its
        proposal index, the local ref it referenced and the reason it was
        omitted: ``{"item_kind": ..., "item_index": <index>, "local_ref": <ref>,
        "reason": "references_omitted_object"}``.
        """
        dependencies: list[dict[str, JsonValue]] = []
        for index, assertion in enumerate(proposal.assertions):
            referenced = [assertion.subject.local_ref]
            if assertion.object is not None:
                referenced.append(assertion.object.local_ref)
            if local_ref in referenced:
                dependencies.append(
                    {
                        "item_kind": "assertion",
                        "item_index": index,
                        "local_ref": local_ref,
                        "reason": "references_omitted_object",
                    }
                )
        for index, inquiry in enumerate(proposal.new_inquiries):
            if inquiry.subject.local_ref == local_ref:
                dependencies.append(
                    {
                        "item_kind": "inquiry",
                        "item_index": index,
                        "local_ref": local_ref,
                        "reason": "references_omitted_object",
                    }
                )
        for index, link in enumerate(proposal.observation_links):
            if link.target_kind == "object" and link.target.local_ref == local_ref:
                dependencies.append(
                    {
                        "item_kind": "observation_link",
                        "item_index": index,
                        "local_ref": local_ref,
                        "reason": "references_omitted_object",
                    }
                )
        return dependencies

    @staticmethod
    def _event_participant_pairs(
        proposal: CognitionDeltaProposal,
        item: NewObjectProposal,
        object_ids: dict[str, str],
        commit_id: str,
    ) -> tuple[set[tuple[str, str]], bool]:
        """Resolve one new event's declared (role, object_id) participant pairs.

        Task 2.3: the pair set comes from the event's explicit
        ``participants`` declarations — the same declarations Task 1.6
        compiles into has_participant assertions. Each pair carries the
        participant's role (``role`` or ``qualifiers["role"]``); a missing
        role marks the set incomplete, so the event can only ever be a
        partial candidate. A proposal-local participant object resolves
        through the object identity map (falling back to the deterministic
        id a previous commit of the same proposal would have written when the
        ref was omitted), and an existing object's memory_id is its durable
        id verbatim. Returns ``(pairs, complete)``.
        """
        pairs: set[tuple[str, str]] = set()
        complete = True
        for participant in item.participants:
            role = participant.role or participant.qualifiers.get("role") or ""
            if not role:
                complete = False
            target = participant.object
            if target.local_ref is not None:
                object_id = object_ids.get(
                    target.local_ref, durable_id("object", commit_id, target.local_ref)
                )
            else:
                object_id = str(target.memory_id)
            pairs.add((role, object_id))
        return pairs, complete

    @staticmethod
    def _stored_event_participant_pairs(
        connection: sqlite3.Connection, event_id: str
    ) -> tuple[set[tuple[str, str]], bool]:
        """Read one stored event's has_participant (role, object_id) pairs.

        Task 2.3: participants are read from the assertions table's current
        has_participant edges (roles from ``qualifiers["role"]``) — no
        separate events table exists. Edges with any other predicate or a
        literal target never count; a missing role marks the set incomplete.
        Returns ``(pairs, complete)``.
        """
        pairs: set[tuple[str, str]] = set()
        complete = True
        for row in connection.execute(
            "SELECT object_id, qualifiers_json FROM current_assertions"
            " WHERE subject_id = ? AND predicate = 'has_participant' AND object_id IS NOT NULL",
            (event_id,),
        ).fetchall():
            qualifiers = json.loads(str(row["qualifiers_json"] or "{}"))
            role = ""
            if isinstance(qualifiers, dict):
                role = str(qualifiers.get("role") or "")
            if not role:
                complete = False
            pairs.add((role, str(row["object_id"])))
        return pairs, complete

    @staticmethod
    def _event_part_of_parents(
        proposal: CognitionDeltaProposal,
        item: NewObjectProposal,
        object_ids: dict[str, str],
        commit_id: str,
    ) -> set[str]:
        """Resolve the proposal event's explicitly declared part_of parent ids.

        Task 2.3: only assertions anchored on the event itself with predicate
        ``part_of`` and an object target declare a parent; the target
        resolves through the object identity map like a participant does.
        """
        parents: set[str] = set()
        for assertion in proposal.assertions:
            if assertion.subject.local_ref != item.local_ref:
                continue
            if assertion.predicate != "part_of" or assertion.object is None:
                continue
            target = assertion.object
            if target.local_ref is not None:
                parents.add(
                    object_ids.get(
                        target.local_ref, durable_id("object", commit_id, target.local_ref)
                    )
                )
            else:
                parents.add(str(target.memory_id))
        return parents

    @staticmethod
    def _stored_event_part_of_parents(
        connection: sqlite3.Connection, event_id: str
    ) -> set[str]:
        """One stored event's current part_of parent ids (Task 2.3)."""
        return {
            str(row["object_id"])
            for row in connection.execute(
                "SELECT object_id FROM current_assertions"
                " WHERE subject_id = ? AND predicate = 'part_of' AND object_id IS NOT NULL",
                (event_id,),
            ).fetchall()
        }

    @staticmethod
    def _event_granularity(
        proposal: CognitionDeltaProposal, item: NewObjectProposal
    ) -> set[str]:
        """Resolve the proposal event's explicitly declared granularity values.

        Task 2.3: a granularity declaration is any assertion anchored on the
        event that carries the ``granularity`` qualifier (the bounded
        qualifier vocabulary); its value is the declaration.
        """
        return {
            str(assertion.qualifiers["granularity"])
            for assertion in proposal.assertions
            if assertion.subject.local_ref == item.local_ref
            and assertion.qualifiers.get("granularity")
        }

    @staticmethod
    def _stored_event_granularity(
        connection: sqlite3.Connection, event_id: str
    ) -> set[str]:
        """One stored event's declared granularity values (Task 2.3)."""
        values: set[str] = set()
        for row in connection.execute(
            "SELECT qualifiers_json FROM current_assertions WHERE subject_id = ?",
            (event_id,),
        ).fetchall():
            raw = row["qualifiers_json"]
            if not raw:
                continue
            qualifiers = json.loads(str(raw))
            if isinstance(qualifiers, dict) and qualifiers.get("granularity"):
                values.add(str(qualifiers["granularity"]))
        return values

    @staticmethod
    def _time_ranges_overlap(
        a_start: datetime | None,
        a_end: datetime | None,
        b_start: datetime | None,
        b_end: datetime | None,
    ) -> bool:
        """Whether two possibly-unbounded time windows intersect (Task 2.3).

        A missing bound is unbounded on that side, so two events without any
        temporal information overlap trivially (the precision gate then
        decides the tier). Naive datetimes are read as UTC.
        """
        time_min = datetime.min.replace(tzinfo=UTC)
        time_max = datetime.max.replace(tzinfo=UTC)
        left = max(
            _as_utc(a_start) if a_start is not None else time_min,
            _as_utc(b_start) if b_start is not None else time_min,
        )
        right = min(
            _as_utc(a_end) if a_end is not None else time_max,
            _as_utc(b_end) if b_end is not None else time_max,
        )
        return left <= right

    @staticmethod
    def _event_candidate_weakness(
        item: NewObjectProposal,
        proposal_complete: bool,
        stored_complete: bool,
        pairs_equal: bool,
        stored_type_key: str | None,
        stored_precision: str,
        parents_equal: bool,
        granularity_equal: bool,
        *,
        stored_provisional: bool,
    ) -> list[str]:
        """Name every strong criterion the comparison fails (Task 2.3, F2).

        Returns an empty list when all strong criteria hold — the candidate
        is strong. Provisional events on either side, incomplete or differing
        participant sets, missing/differing type keys, unknown/differing
        precision, and missing/differing part_of or granularity declarations
        each yield one stable token in declaration order.
        """
        weakness: list[str] = []
        if item.provisional or stored_provisional:
            weakness.append("provisional")
        if not (item.type_key and stored_type_key):
            weakness.append("type_key_missing")
        elif item.type_key != stored_type_key:
            weakness.append("type_key_mismatch")
        if not (proposal_complete and stored_complete):
            weakness.append("participant_roles_missing")
        elif not pairs_equal:
            weakness.append("participant_set_different")
        if item.event_time_precision == "unknown" or stored_precision == "unknown":
            weakness.append("time_precision_unknown")
        elif item.event_time_precision != stored_precision:
            weakness.append("time_precision_mismatch")
        if not parents_equal:
            weakness.append("parent_missing_or_different")
        if not granularity_equal:
            weakness.append("granularity_missing_or_different")
        return weakness

    @staticmethod
    def _strong_event_candidate_ids(
        connection: sqlite3.Connection,
        proposal_pairs: set[tuple[str, str]],
        item: NewObjectProposal,
        own_id: str,
    ) -> list[str]:
        """Preselect stored events that can still match strongly (F6, Step 4).

        The first candidate step is never ``SELECT ... FROM objects WHERE
        kind='event'``: this query narrows authority through the
        ``assertions.object_id`` index (``idx_assertions_object``) over the
        proposed participants' object ids, then applies the stored-side
        filters that any strong candidate must already satisfy — kind,
        non-provisional, non-own id, equal type key, and the same
        possibly-unbounded time overlap the Python check enforces (a stored
        NULL bound means unbounded on that side; ``NULL`` start is the
        minimum bound, ``NULL`` end the maximum).

        Only runs when the proposal has a type_key, a known time precision
        and at least one bounded time edge — without those, no stored event
        can pass the strong criteria, so the query is skipped entirely and
        the result is empty (fail closed). This SQL only narrows authority;
        the exact participant-pair, time-overlap, part-of and granularity
        criteria are re-applied in Python on the returned set.
        """
        if not item.type_key or item.event_time_precision == "unknown":
            return []
        if item.event_time_start is None and item.event_time_end is None:
            return []
        participant_ids = list(dict.fromkeys(object_id for _, object_id in proposal_pairs))
        if not participant_ids:
            return []
        marks = ", ".join("?" for _ in participant_ids)
        predicates: list[str] = []
        time_values: list[object] = []
        if item.event_time_start is not None:
            predicates.append("(o.event_time_end IS NULL OR o.event_time_end >= ?)")
            time_values.append(_as_utc(item.event_time_start).astimezone(UTC).isoformat())
        if item.event_time_end is not None:
            predicates.append("(o.event_time_start IS NULL OR o.event_time_start <= ?)")
            time_values.append(_as_utc(item.event_time_end).astimezone(UTC).isoformat())
        values: list[object] = [
            *participant_ids,
            *time_values,
            ObjectKind.EVENT.value,
            own_id,
            item.type_key,
        ]
        sql = (
            "SELECT DISTINCT a.subject_id AS event_id"
            " FROM current_assertions AS a"
            " JOIN objects AS o ON o.id = a.subject_id"
            " WHERE a.predicate = 'has_participant'"
            f" AND a.object_id IN ({marks})"
            + (" AND " + " AND ".join(predicates) if predicates else "")
            + " AND o.kind = ? AND o.provisional = 0 AND o.id != ? AND o.type_key = ?"
            + " ORDER BY a.subject_id"
        )
        return [str(row["event_id"]) for row in connection.execute(sql, values).fetchall()]

    @staticmethod
    def _partial_event_candidate_ids(
        connection: sqlite3.Connection,
        proposal_pairs: set[tuple[str, str]],
        own_id: str,
        limit: int = 9,
    ) -> tuple[list[str], bool]:
        """Bound the advisory partial-event candidate set (F6, Step 5).

        The same ``assertions.object_id`` participant lookup as the strong
        path but without type/time/provisional authority, ``LIMIT 9``
        (cap + 1) in stable event id order. Exact weakness classification
        still runs on the returned rows; an advisory partial miss never
        rejects or merges, so bounded recall is the correct failure
        direction.

        Returns:
            A ``(ids, limit_hit)`` pair: the bounded event ids in stable
            order, and whether the pre-selection hit its LIMIT — the signal
            that rows beyond the sample were never weakness-classified, so
            the caller reports ``candidate_sample_truncated`` (final set
            fits the cap, total unknown) or ``candidate_has_more`` (final
            set was cut too) instead of an exact omitted count.

        """
        participant_ids = list(dict.fromkeys(object_id for _, object_id in proposal_pairs))
        if not participant_ids:
            return [], False
        marks = ", ".join("?" for _ in participant_ids)
        rows = connection.execute(
            "SELECT DISTINCT a.subject_id AS event_id"
            " FROM current_assertions AS a"
            f" WHERE a.predicate = 'has_participant' AND a.object_id IN ({marks})"
            " AND a.subject_id != ? ORDER BY a.subject_id LIMIT ?",
            [*participant_ids, own_id, limit],
        ).fetchall()
        return [str(row["event_id"]) for row in rows], len(rows) == limit

    def _decide_event_candidates(
        self,
        index: int,
        item: NewObjectProposal,
        proposal: CognitionDeltaProposal,
        object_ids: dict[str, str],
        commit_id: str,
        *,
        connection: sqlite3.Connection,
    ) -> tuple[ReviewIssue | None, ReviewIssue | None]:
        """Decide one new event's tiered duplicate candidates (Task 2.3).

        Compares the proposal event against every stored event
        (objects(kind='event')) by (role, object_id) participant pairs,
        type_key, time window and precision, and part_of/granularity
        declarations:

        * strong (error): the proposal event is non-provisional, both
          participant sets are non-empty AND complete and equal as
          (role, object_id) pairs, the type_keys are equal and non-empty, the
          time windows overlap at the same non-unknown precision, and any
          explicitly declared part_of parent or granularity exists on both
          sides and agrees;
        * partial (warning): both participant sets are non-empty with at
          least one shared participant object and the time windows overlap
          (or are both unknown), but at least one strong criterion fails;
        * no candidate: either participant set is empty, the participant
          object sets are disjoint, or the known time windows do not overlap
          — such events are different events and coexist silently.

        The decision is replay-deterministic by construction: it is a pure
        function of the proposal (pairs and declarations resolved through the
        final object map) and the read-only object and assertion tables, with
        the commit's own event id excluded so a same-commit replay never
        self-collides. No store mutation happens here.

        Returns:
            An ``(strong, partial)`` issue pair; at most one is not None,
            and both are None when the event has no candidate.

        """
        own_id = durable_id("object", commit_id, item.local_ref)
        proposal_pairs, proposal_complete = ProposalCommitter._event_participant_pairs(
            proposal, item, object_ids, commit_id
        )
        if not proposal_pairs:
            return None, None
        proposal_parents = ProposalCommitter._event_part_of_parents(
            proposal, item, object_ids, commit_id
        )
        proposal_granularity = ProposalCommitter._event_granularity(proposal, item)
        strong_candidates: list[str] = []
        partial_candidates: list[str] = []
        partial_weakness: list[str] = []
        for stored_id in self._strong_event_candidate_ids(connection, proposal_pairs, item, own_id):
            stored_pairs, stored_complete = ProposalCommitter._stored_event_participant_pairs(
                connection, stored_id
            )
            if not stored_pairs:
                continue
            proposal_objects = {object_id for _, object_id in proposal_pairs}
            stored_objects = {object_id for _, object_id in stored_pairs}
            if not (proposal_objects & stored_objects):
                continue
            stored_row = connection.execute(
                "SELECT type_key, provisional, event_time_start, event_time_end,"
                " event_time_precision FROM objects WHERE id = ?",
                (stored_id,),
            ).fetchone()
            if stored_row is None:
                continue
            if not self._time_ranges_overlap(
                item.event_time_start,
                item.event_time_end,
                _stored_datetime(stored_row["event_time_start"]),
                _stored_datetime(stored_row["event_time_end"]),
            ):
                # known non-overlapping windows are a hard discriminator: the
                # events are different, no candidate at all
                continue
            stored_type_key = str(stored_row["type_key"]) if stored_row["type_key"] else None
            stored_precision = str(stored_row["event_time_precision"] or "unknown")
            stored_parents = ProposalCommitter._stored_event_part_of_parents(
                connection, stored_id
            )
            stored_granularity = ProposalCommitter._stored_event_granularity(
                connection, stored_id
            )
            weakness = ProposalCommitter._event_candidate_weakness(
                item,
                proposal_complete,
                stored_complete,
                proposal_pairs == stored_pairs,
                stored_type_key,
                stored_precision,
                proposal_parents == stored_parents,
                proposal_granularity == stored_granularity,
                stored_provisional=bool(stored_row["provisional"]),
            )
            # a preselected event that fails the exact strong re-check is a
            # partial-tier candidate; the bounded partial lookup below may
            # still cut it, which is the accepted advisory-recall trade-off
            if not weakness:
                strong_candidates.append(stored_id)
        if not strong_candidates:
            partial_preselected, partial_limit_hit = self._partial_event_candidate_ids(
                connection, proposal_pairs, own_id
            )
            for stored_id in partial_preselected:
                stored_pairs, stored_complete = ProposalCommitter._stored_event_participant_pairs(
                    connection, stored_id
                )
                if not stored_pairs:
                    continue
                proposal_objects = {object_id for _, object_id in proposal_pairs}
                stored_objects = {object_id for _, object_id in stored_pairs}
                if not (proposal_objects & stored_objects):
                    continue
                stored_row = connection.execute(
                    "SELECT type_key, provisional, event_time_start, event_time_end,"
                    " event_time_precision FROM objects WHERE id = ?",
                    (stored_id,),
                ).fetchone()
                if stored_row is None:
                    continue
                if not self._time_ranges_overlap(
                    item.event_time_start,
                    item.event_time_end,
                    _stored_datetime(stored_row["event_time_start"]),
                    _stored_datetime(stored_row["event_time_end"]),
                ):
                    # known non-overlapping windows are a hard discriminator:
                    # the events are different, no candidate at all
                    continue
                stored_type_key = str(stored_row["type_key"]) if stored_row["type_key"] else None
                stored_precision = str(stored_row["event_time_precision"] or "unknown")
                stored_parents = ProposalCommitter._stored_event_part_of_parents(
                    connection, stored_id
                )
                stored_granularity = ProposalCommitter._stored_event_granularity(
                    connection, stored_id
                )
                weakness = ProposalCommitter._event_candidate_weakness(
                    item,
                    proposal_complete,
                    stored_complete,
                    proposal_pairs == stored_pairs,
                    stored_type_key,
                    stored_precision,
                    proposal_parents == stored_parents,
                    proposal_granularity == stored_granularity,
                    stored_provisional=bool(stored_row["provisional"]),
                )
                if weakness:
                    partial_candidates.append(stored_id)
                    partial_weakness.extend(weakness)
        if strong_candidates:
            strong_ids, strong_tail = _candidate_payload(strong_candidates)
            return (
                self._issue(
                    ReviewIssueCode.DUPLICATE_EVENT_CANDIDATE,
                    "error",
                    "new_object",
                    index,
                    (
                        "The new event duplicates an existing event with the "
                        "same participant roles, type key and overlapping "
                        "time; it was rejected into one bounded repair instead "
                        "of being reused."
                    ),
                    [
                        "use_existing_event_id",
                        "remove_duplicate_new_event",
                        "differentiate_event_identity",
                    ],
                    actual_value={
                        "local_ref": item.local_ref,
                        "existing_event_ids": strong_ids,
                        **strong_tail,
                    },
                    candidate_ids=strong_ids,
                    match_basis=["participant_pairs+type_key+time_overlap"],
                    omitted_dependencies=self._omitted_dependencies(proposal, item.local_ref),
                ),
                None,
            )
        if partial_candidates:
            # F6 + Task 4: the partial source is pre-capped by its SQL LIMIT.
            # A truncated pre-selection only proves the *sample* was cut; the
            # exact weakness classification runs after it, so the final set
            # may fit under the cap (candidate_sample_truncated — total
            # unknown) or itself be cut (candidate_has_more — confirmed
            # lower bound). Never an exact-looking count.
            partial_ids, partial_tail = _candidate_payload(
                partial_candidates, sampled=True, truncated=partial_limit_hit
            )
            return (
                None,
                self._issue(
                    ReviewIssueCode.DUPLICATE_EVENT_CANDIDATE,
                    "warning",
                    "new_object",
                    index,
                    (
                        "The new event shares participants with an existing "
                        "event but does not strongly match its type, time or "
                        "granularity; both events coexist as a partial "
                        "duplicate candidate."
                    ),
                    [
                        "complete_event_type_or_time",
                        "declare_participant_roles",
                        "declare_parent_or_granularity",
                        "raise_inquiry_for_duplicate_event",
                    ],
                    actual_value={
                        "local_ref": item.local_ref,
                        "existing_event_ids": partial_ids,
                        **partial_tail,
                        "weakness": list(dict.fromkeys(partial_weakness)),
                    },
                    candidate_ids=partial_ids,
                    match_basis=["participant_set_overlap"],
                ),
            )
        return None, None

    def _compile(
        self,
        proposal: CognitionDeltaProposal,
        commit_id: str,
        object_ids: dict[str, str],
        inquiry_ids: dict[str, str],
        omitted_local_refs: set[str],
        omitted_inquiries: list[int],
        alias_add_decisions: dict[int, ObjectUpdateAliasDecision],
        omitted_new_objects: dict[int, tuple[list[str], list[str]]],
        collision_issues: dict[int, list[ReviewIssue]],
        committed_at: datetime,
        replay_delta: CognitiveDelta | None = None,
        replay_attempt_recorded: bool = False,
    ) -> tuple[
        CognitiveDelta,
        list[int],
        list[str],
        list[str],
        list[int],
        dict[int, list[str]],
        list[ReviewIssue],
        set[str],
    ]:
        replay_objects = {item.id: item for item in replay_delta.objects} if replay_delta else {}
        replay_assertions = {item.id: item for item in replay_delta.assertions} if replay_delta else {}
        replay_resolutions = (
            {item.id: item for item in replay_delta.resolve_inquiries} if replay_delta else {}
        )
        # The routing is a pure function of the proposal, so compiling the
        # replay recomputes exactly the kinds/type_keys the original commit
        # stored; unsupported items are omitted with a review issue.
        routed_object_kinds, unsupported_object_indexes = _route_new_object_kinds(proposal)
        unsupported_refs = {
            proposal.new_objects[index].local_ref for index in unsupported_object_indexes
        }
        omitted_object_refs = {
            proposal.new_objects[omitted_index].local_ref for omitted_index in omitted_new_objects
        } | unsupported_refs
        with self._store.read_connection() as connection:
            objects = [
                ObjectInput(
                    id=object_ids[item.local_ref],
                    kind=routed_object_kinds[index][0],
                    type_key=routed_object_kinds[index][1],
                    canonical_name=item.canonical_name,
                    aliases=item.aliases,
                    domain_hints=item.domain_hints,
                    provisional=item.provisional,
                    event_time_start=item.event_time_start,
                    event_time_end=item.event_time_end,
                    event_time_precision=item.event_time_precision,
                )
                for index, item in enumerate(proposal.new_objects)
                if item.local_ref not in omitted_object_refs
            ]
            review_issues: list[ReviewIssue] = []
            alias_operations: list[AliasOperation] = []
            for index in sorted(unsupported_object_indexes):
                item = proposal.new_objects[index]
                review_issues.append(
                    self._issue(
                        ReviewIssueCode.UNSUPPORTED_OBJECT_KIND,
                        "warning",
                        "new_object",
                        index,
                        (
                            "The new object was omitted because 'source' is not a "
                            "supported world graph kind; record the material as an "
                            "observation and link it as evidence instead."
                        ),
                        [
                            "record_as_observation",
                            "link_observation_as_evidence",
                            "drop_source_new_object",
                        ],
                        actual_value={"kind": item.kind.value, "local_ref": item.local_ref},
                    )
                )
            # Task 2.1: the collision issues are pre-built by the commit loop
            # (identity conflicts are decided there, not here), so every
            # omitted item's issue — with its candidates, basis, and omitted
            # dependencies — is one source of truth for both the committed
            # warning and the fail-closed rejection.
            for index in sorted(collision_issues):
                review_issues.extend(collision_issues[index])
            for index, item in enumerate(proposal.object_updates):
                decision = alias_add_decisions.get(index)
                accepted_adds: Sequence[str] = (
                    decision.accepted_adds if decision is not None else item.add_aliases
                )
                if decision is not None and decision.conflicts:
                    # F3: only the claimed ADDs are omitted; the safe update
                    # fields and unclaimed aliases still compile.
                    conflict_ids, conflict_tail = _candidate_payload(
                        owner for _, owner in decision.conflicts
                    )
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                            "warning",
                            "object_update",
                            index,
                            (
                                "The conflicting alias additions were omitted because "
                                "each adds an identity alias already claimed by another "
                                "stored object; the safe update fields and unclaimed "
                                "aliases were kept."
                            ),
                            [
                                "omit_conflicting_alias_add",
                                "keep_safe_update_fields",
                                "do_not_merge_distinct_objects",
                            ],
                            durable_id=str(item.target.memory_id),
                            actual_value={
                                "object_id": str(item.target.memory_id),
                                "conflicting_adds": [
                                    {"alias": alias, "existing_object_id": owner}
                                    for alias, owner in decision.conflicts
                                ],
                                **conflict_tail,
                            },
                            candidate_ids=conflict_ids,
                            match_basis=["identity_alias_active_owner"],
                        )
                    )
                replay_object = replay_objects.get(str(item.target.memory_id))
                if (
                    replay_object is not None
                    and replay_object.expected_version == item.expected_version
                    and replay_attempt_recorded
                ):
                    updated = replay_object
                else:
                    # A form declared in more than one action is dropped from
                    # every action (one alias_operation_invalid issue); the
                    # compiled update rewrite must not re-activate it.
                    updated = self._object_update(
                        connection,
                        item,
                        accepted_adds=accepted_adds,
                        dropped_forms=_multi_action_alias_conflicts(
                            accepted_adds, item.remove_aliases, item.demote_aliases
                        ),
                    )
                if updated is not None:
                    objects.append(updated)
                    if (
                        replay_delta is not None
                        and replay_object is not None
                        and replay_object.expected_version == item.expected_version
                        and replay_attempt_recorded
                    ):
                        # Same-commit replay: the stored delta already carries
                        # this object's alias operations; reusing them keeps the
                        # replayed delta byte-identical (no second transition).
                        alias_operations.extend(
                            operation
                            for operation in replay_delta.alias_operations
                            if operation.object_id == str(item.target.memory_id)
                        )
                    else:
                        self._compile_alias_operations(
                            connection,
                            proposal,
                            item,
                            str(item.target.memory_id),
                            index,
                            object_ids,
                            review_issues,
                            alias_operations,
                            accepted_adds=accepted_adds,
                        )
                else:
                    row = connection.execute(
                        "SELECT version FROM objects WHERE id = ?", (item.target.memory_id,)
                    ).fetchone()
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.STALE_OBJECT_UPDATE,
                            "warning",
                            "object_update",
                            index,
                            "The object update did not match the current object version.",
                            ["refresh_expected_version", "remove_update"],
                            actual_value={
                                "object_id": str(item.target.memory_id),
                                "expected_version": item.expected_version,
                                "current_version": row["version"] if row is not None else None,
                            },
                            durable_id=str(item.target.memory_id),
                        ).model_copy(
                            update={
                                "message": (
                                    "The object update expected an old version; current version is "
                                    f"{row['version'] if row is not None else 'missing'}."
                                )
                            }
                        )
                    )
            assertions: list[AssertionInput] = []
            omitted_assertions: list[int] = []
            evidence_missing_assertions: list[int] = []
            answerable: list[str] = []
            dropped_evidence: dict[int, list[str]] = {}
            # F4: the first compiled owner per signature. Explicit assertions
            # register here; participant edges run the same path afterward.
            pending_by_signature: PendingAssertionSignatures = {}
            for index, item in enumerate(proposal.assertions):
                referenced_refs = [item.subject.local_ref]
                if item.object is not None:
                    referenced_refs.append(item.object.local_ref)
                if any(ref in omitted_object_refs for ref in referenced_refs if ref is not None):
                    # dependency: the assertion references a new object that
                    # will not exist (omitted for an alias collision, or
                    # rejected as an unsupported kind); a compiled assertion
                    # cannot reference an object that will not exist
                    omitted_assertions.append(index)
                    unsupported_dependency = any(
                        ref is not None and ref in unsupported_refs for ref in referenced_refs
                    )
                    if unsupported_dependency:
                        review_issues.append(
                            self._issue(
                                ReviewIssueCode.UNSUPPORTED_OBJECT_KIND,
                                "warning",
                                "assertion",
                                index,
                                (
                                    "The assertion was omitted because it references a new "
                                    "object of the unsupported 'source' kind; record the "
                                    "material as an observation and link it as evidence "
                                    "instead."
                                ),
                                [
                                    "record_as_observation",
                                    "link_observation_as_evidence",
                                    "drop_source_new_object",
                                ],
                                actual_value={
                                    "assertion": item.model_dump(mode="json"),
                                    "omitted_object_local_refs": sorted(omitted_object_refs),
                                },
                            )
                        )
                    else:
                        review_issues.append(
                            self._issue(
                                ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                                "warning",
                                "assertion",
                                index,
                                (
                                    "The assertion was omitted because it references a new "
                                    "object that was omitted for an identity alias claim conflict."
                                ),
                                [
                                    "use_existing_object_id",
                                    "remove_duplicate_new_object",
                                    "replace_local_refs",
                                ],
                                actual_value={
                                    "assertion": item.model_dump(mode="json"),
                                    "omitted_object_local_refs": sorted(omitted_object_refs),
                                },
                            )
                        )
                    continue
                evidence: list[EvidenceInput] = []
                dropped: list[str] = []
                declared_evidence = bool(item.evidence)
                seen_support_ids: list[str] = []
                comment_evidence_ids: list[str] = []
                low_reliability_evidence_ids: list[str] = []
                reliable_exact_part_ids: list[str] = []
                reliable_exact_part_sources: set[str] = set()
                mixed_primary_sources: dict[str, str] = {}
                evidence_reliabilities: list[str] = []
                for link in item.evidence:
                    row = connection.execute(
                        "SELECT observations.source_uri, observations.depth, "
                        "observations.metadata_json, observation_bodies.body_json "
                        "FROM observations LEFT JOIN observation_bodies "
                        "ON observation_bodies.observation_id = observations.id WHERE observations.id = ?",
                        (link.observation_id,),
                    ).fetchone()
                    if row is None:
                        # hallucinated observation id: sanitized at the proposal
                        # boundary so one forged id cannot kill the whole delta;
                        # the drop is recorded per assertion so the receipt and
                        # the terminal summary can surface it (b36b Task 4 H3)
                        dropped.append(link.observation_id)
                        continue
                    if link.role == "supports" and row["depth"] == ObservationDepth.SEEN.value:
                        seen_support_ids.append(link.observation_id)
                    if item.epistemic_role is EpistemicRole.FACT:
                        if _is_comment_observation(link.observation_id, row["metadata_json"]):
                            comment_evidence_ids.append(link.observation_id)
                        reliability, is_mixed_primary, is_exact_part = _evidence_reliability(row)
                        if is_mixed_primary:
                            mixed_primary_sources[link.observation_id] = str(row["source_uri"])
                        elif reliability in {"best_effort", "automatic"}:
                            low_reliability_evidence_ids.append(link.observation_id)
                            evidence_reliabilities.append(reliability)
                        elif reliability is not None:
                            evidence_reliabilities.append(reliability)
                            if is_exact_part:
                                reliable_exact_part_ids.append(link.observation_id)
                                reliable_exact_part_sources.add(str(row["source_uri"]))
                    evidence.append(link)
                if dropped:
                    dropped_evidence[index] = dropped
                    # Unknown observation ids never cross the durable link
                    # boundary. The Agent's assertion and every valid link
                    # remain independently expressible, while the dropped
                    # ids stay visible in the receipt and attempt audit.
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.MISSING_EVIDENCE_ID,
                            "warning",
                            "assertion",
                            index,
                            (
                                "Unknown evidence links were dropped; the assertion was retained "
                                "without inventing replacement observations."
                            ),
                            ["use_existing_observation_ids_in_future_proposals"],
                            actual_value={
                                "declared_evidence_ids": [link.observation_id for link in item.evidence],
                                "missing_evidence_ids": dropped,
                            },
                            dropped_evidence_ids=dropped,
                        )
                    )
                if declared_evidence and not evidence:
                    evidence_missing_assertions.append(index)
                if not evidence:
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.NO_VALID_EVIDENCE,
                            "warning",
                            "assertion",
                            index,
                            "The assertion was committed with no valid evidence links.",
                            ["preserve_no_basis_status"],
                            actual_value={
                                "declared_evidence_count": len(item.evidence),
                                "dropped_evidence_ids": dropped,
                                "assertion": item.model_dump(mode="json"),
                            },
                            dropped_evidence_ids=dropped,
                        )
                    )
                mixed_without_exact_part = [
                    observation_id
                    for observation_id, source_uri in mixed_primary_sources.items()
                    if source_uri not in reliable_exact_part_sources
                ]
                if mixed_without_exact_part:
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.MIXED_RELIABILITY_PRIMARY,
                            "warning",
                            "assertion",
                            index,
                            (
                                "The fact judgment cites a mixed-reliability body without an exact "
                                "reliable part; the assertion and evidence roles were preserved."
                            ),
                            ["preserve_mixed_body_context"],
                            actual_value={
                                "mixed_primary_evidence_ids": mixed_without_exact_part,
                                "reliable_exact_evidence_ids": reliable_exact_part_ids,
                            },
                        )
                    )
                if (
                    evidence
                    and evidence_reliabilities
                    and len(low_reliability_evidence_ids) == len(evidence)
                ):
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.LOW_RELIABILITY_FACT,
                            "warning",
                            "assertion",
                            index,
                            (
                                "The fact judgment is based only on best-effort or automatic "
                                "transcription; the assertion and evidence roles were preserved."
                            ),
                            ["preserve_reliability_context"],
                            actual_value={
                                "low_reliability_evidence_ids": low_reliability_evidence_ids,
                                "reliabilities": evidence_reliabilities,
                            },
                        )
                    )
                if evidence and len(comment_evidence_ids) == len(evidence):
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.COMMENT_ONLY_FACT,
                            "warning",
                            "assertion",
                            index,
                            (
                                "The fact judgment is based only on comment or discussion material; "
                                "the assertion and evidence roles were preserved."
                            ),
                            ["preserve_discussion_basis_context"],
                            actual_value={
                                "comment_evidence_ids": comment_evidence_ids,
                                "assertion": item.model_dump(mode="json"),
                            },
                        )
                    )
                if seen_support_ids:
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.SEEN_ONLY_SUPPORT,
                            "warning",
                            "assertion",
                            index,
                            (
                                "The assertion uses seen-depth material as supports; the assertion "
                                "and Agent-selected evidence role were preserved."
                            ),
                            ["preserve_seen_depth_context"],
                            actual_value={
                                "seen_evidence_ids": seen_support_ids,
                                "assertion": item.model_dump(mode="json"),
                            },
                        )
                    )
                subject_id = self._resolve_ref(connection, item.subject, object_ids, "objects")
                if item.answers_inquiry_id is None:
                    rows = connection.execute(
                        "SELECT id FROM inquiries WHERE status IN ('open', 'dormant') AND subject_id = ?",
                        (subject_id,),
                    ).fetchall()
                    answerable.extend(str(r["id"]) for r in rows)
                assertion_id = durable_id("assertion", commit_id, str(index))
                replay_assertion = replay_assertions.get(assertion_id)
                # The signature deliberately excludes confidence,
                # event_time_precision, evidence and the supersedes /
                # answers-inquiry links (see store._assertion_signature):
                # replay of old-format commits must stay intact. A proposal
                # assertion that collides on the signature is announced by
                # _announce_assertion_dedup instead of being silently dropped:
                # the store keeps the existing row and links the valid new
                # evidence to it, and every un-replaced metadata difference is
                # surfaced in the warning. Without a collision, the bounded
                # C-class preflight recalls similar literals as candidates
                # (possible_cognition_conflict) without ever judging a
                # contradiction or rewriting supersedes.
                assertion = AssertionInput(
                    id=assertion_id,
                    subject_id=subject_id,
                    predicate=item.predicate,
                    object_id=(
                        self._resolve_ref(connection, item.object, object_ids, "objects")
                        if item.object is not None
                        else None
                    ),
                    literal=item.literal,
                    epistemic_role=item.epistemic_role,
                    confidence=item.confidence,
                    # spec §11.1-4: proposal qualifiers (community/language)
                    # must survive the compile into the stored assertion
                    qualifiers=item.qualifiers,
                    evidence=evidence,
                    event_time_start=item.event_time_start,
                    event_time_end=item.event_time_end,
                    event_time_precision=item.event_time_precision,
                    supersedes_id=item.supersedes_id,
                    superseded_at=(
                        replay_assertion.superseded_at
                        if replay_assertion is not None
                        and replay_assertion.supersedes_id == item.supersedes_id
                        else (committed_at if item.supersedes_id is not None else None)
                    ),
                    supersede_reason=item.supersede_reason,
                    answers_inquiry_id=item.answers_inquiry_id,
                )
                assertions.append(assertion)
                if not self._announce_assertion_dedup(
                    connection, index, assertion, review_issues, pending_by_signature
                ):
                    self._preflight_similar_literal(connection, index, assertion, review_issues)
            # ── Task 1.6: explicit event participants compile to edges ────────
            # The compiled edges carry the Agent's own epistemic role,
            # confidence and evidence; only the explicit declaration counts,
            # so a non-provisional event without participants warns instead of
            # deriving edges from text.
            compiled_edges = self._compile_participant_edges(
                proposal,
                commit_id,
                connection,
                object_ids,
                omitted_object_refs,
                unsupported_refs,
                review_issues,
            )
            for edge_index, edge in compiled_edges:
                # F4: participant edges run the same signature-collision path
                # as explicit assertions (shared pending tracker, first owner
                # wins); the store still links the edge's evidence to the
                # retained id, and replay stays deterministic.
                self._announce_assertion_dedup(
                    connection, edge_index, edge, review_issues, pending_by_signature
                )
                assertions.append(edge)
            inquiries = [
                InquiryInput(
                    id=inquiry_ids[item.local_ref],
                    subject_id=self._resolve_ref(connection, item.subject, object_ids, "objects"),
                    prompt=item.prompt,
                    rationale=item.rationale,
                    kind=item.kind,
                )
                for item in proposal.new_inquiries
                if item.local_ref not in omitted_local_refs
            ]
            # F2b (spec §5): a resolution is answerable only by an assertion
            # that declares answers_inquiry_id == the inquiry — substantive
            # assertions on the subject no longer qualify on their own, and
            # an UNCERTAINTY link never answers (matches the store predicate
            # in _validate_delta, so the committer never compiles a delta the
            # store would hard-reject)
            # Task 2 (closeout): the resolution gate reads the metadata that
            # WILL persist. Signature-collision dedup (store._write_assertion)
            # keeps the stored row — else the first same-signature owner in
            # compile order — and discards later proposals' answers_inquiry_id;
            # a resolution must not pass on a link the write throws away, and
            # must not fail on a link the stored row still carries. The
            # effective owner mirrors _announce_assertion_dedup.
            effective_answer_links: dict[str, str | None] = {}
            for assertion in assertions:
                signature = _assertion_signature(assertion)
                if signature in effective_answer_links:
                    continue
                stored = WorldStore._assertion_by_signature(connection, signature)
                if stored is not None:
                    effective_answer_links[signature] = stored["answers_inquiry_id"]
                else:
                    pending = pending_by_signature.get(signature, assertion)
                    effective_answer_links[signature] = pending.answers_inquiry_id
            answering_inquiry_ids: set[str] = set()
            for assertion in assertions:
                if assertion.epistemic_role is EpistemicRole.UNCERTAINTY:
                    continue
                link = effective_answer_links[_assertion_signature(assertion)]
                if link is not None:
                    answering_inquiry_ids.add(link)
            resolutions: list[InquiryResolution] = []
            omitted_resolutions: list[str] = []
            # hallucinated resolution targets (row absent) are excluded from
            # the attempt-touch set: touching them would raise, and a proposal
            # never "attempted" a memory that does not exist
            missing_resolution_ids: set[str] = set()
            for index, item in enumerate(proposal.resolve_inquiries):
                row = connection.execute(
                    "SELECT subject_id, status, version FROM inquiries WHERE id = ?", (item.memory_id,)
                ).fetchone()
                current_status = str(row["status"]) if row is not None else None
                current_version = int(row["version"]) if row is not None else None
                replay_resolution = replay_resolutions.get(item.memory_id)
                if (
                    replay_resolution is not None
                    and replay_resolution.expected_version == item.expected_version
                ):
                    resolutions.append(replay_resolution)
                    continue
                if replay_delta is not None and replay_resolution is None:
                    # F2a replay guard: the stored delta omitted this
                    # resolution (its answer links were preserved); recompiling
                    # must keep it omitted or stored-delta equality breaks.
                    # Old-format stored deltas replay unchanged; only new
                    # proposals validate differently. The replay attempt-touch
                    # set derives from the stored delta alone, so an omitted
                    # resolution is never re-touched even when a later commit
                    # created the same inquiry id (missing_resolution_ids is
                    # only consulted on first execution).
                    omitted_resolutions.append(item.memory_id)
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.STALE_RESOLUTION,
                            "warning",
                            "resolve_inquiry",
                            index,
                            "The resolution is not part of the stored delta; it stays omitted on replay.",
                            ["keep_omitted_resolution"],
                            actual_value={
                                "inquiry_id": item.memory_id,
                                "current_status": current_status,
                                "current_version": current_version,
                                "expected_version": item.expected_version,
                                "action": "keep_omitted_resolution",
                            },
                            durable_id=item.memory_id,
                        )
                    )
                    continue
                if row is None:
                    missing_resolution_ids.add(item.memory_id)
                    omitted_resolutions.append(item.memory_id)
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.MISSING_RESOLUTION,
                            "warning",
                            "resolve_inquiry",
                            index,
                            (
                                "The inquiry does not exist; its resolution was omitted while "
                                "other cognition remains eligible."
                            ),
                            ["use_real_inquiry_id", "remove_resolution_but_keep_assertions"],
                            actual_value={
                                "inquiry_id": item.memory_id,
                                "current_status": current_status,
                                "current_version": current_version,
                                "expected_version": item.expected_version,
                                "action": "remove_resolution_but_keep_assertions",
                            },
                            durable_id=item.memory_id,
                        )
                    )
                    continue
                if current_version != item.expected_version:
                    omitted_resolutions.append(item.memory_id)
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.STALE_RESOLUTION,
                            "warning",
                            "resolve_inquiry",
                            index,
                            f"The inquiry version is stale; current version is {current_version}.",
                            ["refresh_expected_version", "remove_resolution_but_keep_assertions"],
                            actual_value={
                                "inquiry_id": item.memory_id,
                                "current_status": current_status,
                                "current_version": current_version,
                                "expected_version": item.expected_version,
                                "action": "refresh_expected_version",
                            },
                            durable_id=item.memory_id,
                        )
                    )
                    continue
                if current_status not in {"open", "dormant"}:
                    omitted_resolutions.append(item.memory_id)
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.STALE_RESOLUTION,
                            "warning",
                            "resolve_inquiry",
                            index,
                            f"The inquiry is already {current_status}; its resolution was omitted.",
                            ["remove_resolution_but_keep_assertions", "keep_inquiry_resolved"],
                            actual_value={
                                "inquiry_id": item.memory_id,
                                "current_status": current_status,
                                "current_version": current_version,
                                "expected_version": item.expected_version,
                                "action": "remove_resolution_but_keep_assertions",
                            },
                            durable_id=item.memory_id,
                        )
                    )
                    continue
                if item.memory_id not in answering_inquiry_ids:
                    omitted_resolutions.append(item.memory_id)
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.STALE_RESOLUTION,
                            "warning",
                            "resolve_inquiry",
                            index,
                            "The inquiry has no answering assertion declaring "
                            "answers_inquiry_id in this proposal.",
                            ["add_answering_assertion", "keep_inquiry_open"],
                            actual_value={
                                "inquiry_id": item.memory_id,
                                "current_status": current_status,
                                "current_version": current_version,
                                "expected_version": item.expected_version,
                                "answering_assertion_present": False,
                                "action": "add_answering_assertion",
                            },
                            durable_id=item.memory_id,
                        )
                    )
                    continue
                resolutions.append(
                    InquiryResolution(id=item.memory_id, expected_version=item.expected_version)
                )
            # links targeting an E2-omitted inquiry or a hallucinated observation
            # are dropped, never rejected: the compiled delta must not reference
            # a memory that will not exist
            links: list[ObservationLinkInput] = []
            for index, item in enumerate(proposal.observation_links):
                if item.target_kind == "object" and item.target.local_ref in omitted_object_refs:
                    # Task 2.1 dependency: the link targets a new object that
                    # was omitted for an identity alias claim conflict (or an
                    # unsupported kind); drop it with a warning instead of
                    # letting the compiled link dangle
                    unsupported_dependency = item.target.local_ref in unsupported_refs
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.UNSUPPORTED_OBJECT_KIND
                            if unsupported_dependency
                            else ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                            "warning",
                            "observation_link",
                            index,
                            (
                                "The link targets a new object that was omitted for the "
                                "unsupported 'source' kind."
                                if unsupported_dependency
                                else (
                                    "The link targets a new object that was omitted for an "
                                    "identity alias claim conflict."
                                )
                            ),
                            (
                                [
                                    "record_as_observation",
                                    "link_observation_as_evidence",
                                    "drop_source_new_object",
                                ]
                                if unsupported_dependency
                                else ["remove_link"]
                            ),
                            actual_value=item.model_dump(mode="json"),
                        )
                    )
                    continue
                if item.target_kind == "inquiry" and item.target.local_ref in omitted_local_refs:
                    target_subject_omitted = any(
                        inquiry.subject.local_ref in omitted_object_refs
                        for inquiry in proposal.new_inquiries
                        if inquiry.local_ref == item.target.local_ref
                    )
                    target_subject_unsupported = any(
                        inquiry.subject.local_ref in unsupported_refs
                        for inquiry in proposal.new_inquiries
                        if inquiry.local_ref == item.target.local_ref
                    )
                    if target_subject_unsupported:
                        code = ReviewIssueCode.UNSUPPORTED_OBJECT_KIND
                        message = (
                            "The link targets an inquiry whose subject new object was "
                            "omitted for the unsupported 'source' kind."
                        )
                        actions = [
                            "record_as_observation",
                            "link_observation_as_evidence",
                            "drop_source_new_object",
                        ]
                    elif target_subject_omitted:
                        code = ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT
                        message = (
                            "The link targets an inquiry that was omitted because its "
                            "subject new object was omitted for an identity alias claim conflict."
                        )
                        actions = [
                            "use_existing_object_id",
                            "remove_duplicate_new_object",
                            "replace_local_refs",
                        ]
                    else:
                        code = ReviewIssueCode.DUPLICATE_INQUIRY
                        message = "The link targets an inquiry that was omitted as a duplicate."
                        actions = ["reuse_existing_inquiry", "remove_link"]
                    review_issues.append(
                        self._issue(
                            code,
                            "warning",
                            "observation_link",
                            index,
                            message,
                            actions,
                            actual_value=item.model_dump(mode="json"),
                        )
                    )
                    continue
                if (
                    connection.execute(
                        "SELECT 1 FROM observations WHERE id = ?", (item.observation_id,)
                    ).fetchone()
                    is None
                ):
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.MISSING_OBSERVATION_LINK,
                            "warning",
                            "observation_link",
                            index,
                            "The observation link names no stored observation.",
                            ["replace_with_real_observation_id", "remove_link"],
                            actual_value=item.model_dump(mode="json"),
                            dropped_evidence_ids=[item.observation_id],
                        )
                    )
                    continue
                links.append(
                    ObservationLinkInput(
                        target_kind=item.target_kind,
                        target_id=self._resolve_ref(
                            connection,
                            item.target,
                            object_ids if item.target_kind == "object" else inquiry_ids,
                            "objects" if item.target_kind == "object" else "inquiries",
                        ),
                        observation_id=item.observation_id,
                        role=item.role,
                    )
                )
            open_rows = connection.execute(
                "SELECT id, subject_id, prompt FROM inquiries WHERE status IN ('open', 'dormant')"
            ).fetchall()
            for index in omitted_inquiries:
                item = proposal.new_inquiries[index]
                if item.subject.local_ref in unsupported_refs:
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.UNSUPPORTED_OBJECT_KIND,
                            "warning",
                            "new_inquiry",
                            index,
                            (
                                "The inquiry was omitted because its subject new object was "
                                "omitted for the unsupported 'source' kind."
                            ),
                            [
                                "record_as_observation",
                                "link_observation_as_evidence",
                                "drop_source_new_object",
                            ],
                            actual_value={
                                "prompt": item.prompt,
                                "subject": item.subject.model_dump(mode="json"),
                            },
                        )
                    )
                    continue
                if item.subject.local_ref in omitted_object_refs:
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                            "warning",
                            "new_inquiry",
                            index,
                            (
                                "The inquiry was omitted because its subject new object was "
                                "omitted for an identity alias claim conflict."
                            ),
                            [
                                "use_existing_object_id",
                                "remove_duplicate_new_object",
                                "replace_local_refs",
                            ],
                            actual_value={
                                "prompt": item.prompt,
                                "subject": item.subject.model_dump(mode="json"),
                            },
                        )
                    )
                    continue
                candidates = [
                    str(row["id"])
                    for row in open_rows
                    if str(row["subject_id"]) == self._subject_id(item.subject, object_ids)
                    and jaccard(bigrams(item.prompt), bigrams(str(row["prompt"])))
                    >= self._inquiry_similarity_threshold
                ]
                inquiry_ids, inquiry_tail = _candidate_payload(candidates)
                review_issues.append(
                    self._issue(
                        ReviewIssueCode.DUPLICATE_INQUIRY,
                        "warning",
                        "new_inquiry",
                        index,
                        "The inquiry duplicates an open or dormant inquiry.",
                        ["reuse_existing_inquiry", "declare_deepens_inquiry_id", "remove_inquiry"],
                        actual_value={
                            "prompt": item.prompt,
                            "subject": item.subject.model_dump(mode="json"),
                            "candidate_ids": inquiry_ids,
                            **inquiry_tail,
                        },
                        candidate_ids=inquiry_ids,
                    )
                )
        return (
            CognitiveDelta(
                objects=objects,
                assertions=assertions,
                inquiries=inquiries,
                resolve_inquiries=resolutions,
                observation_links=links,
                alias_operations=alias_operations,
            ),
            omitted_assertions,
            omitted_resolutions,
            list(dict.fromkeys(answerable)),
            evidence_missing_assertions,
            dropped_evidence,
            review_issues,
            missing_resolution_ids,
        )

    def _announce_assertion_dedup(
        self,
        connection: sqlite3.Connection,
        index: int,
        assertion: AssertionInput,
        review_issues: list[ReviewIssue],
        pending_by_signature: PendingAssertionSignatures,
    ) -> bool:
        """Announce a v1/v2 signature collision instead of letting it be silent.

        The store's structural dedup (see store._write_assertion) keeps the
        existing assertion id, links the valid new evidence to it, and never
        replaces the stored metadata. This read-only preflight makes that
        behavior explicit on the receipt: the warning carries the retained id,
        the linked evidence ids, and every proposed-but-un-replaced difference
        (confidence, event time precision, supersedes/answers-inquiry links)
        in ``actual_value["unreplaced"]`` — the commit never silently claims
        the new metadata was saved. The compiled assertion still rides the
        delta (idempotent replay needs the exact same items), and the store
        maps it to the existing row at write time.

        F4: the retained assertion resolves against the stored row first, then
        against the earlier same-proposal entries in deterministic compile
        order (``pending_by_signature``). Explicit assertions register as they
        compile; compiled participant edges run the same path afterward, so an
        intra-proposal collision is announced exactly like a stored one.

        Returns:
            True when a collision was announced (the caller then skips the
            similar-literal preflight — the signature collision already names
            the identity story).

        """
        signature = _assertion_signature(assertion)
        stored = WorldStore._assertion_by_signature(connection, signature)
        pending = pending_by_signature.get(signature)
        if stored is not None and str(stored["id"]) != assertion.id:
            existing_id = str(stored["id"])
            retained_metadata = _stored_assertion_metadata(stored)
        elif pending is not None and pending.id != assertion.id:
            existing_id = pending.id
            retained_metadata = _proposed_assertion_metadata(pending)
        else:
            # no stored twin and no earlier same-proposal twin (or the stored
            # row is this assertion's own replay — never a spurious
            # self-collision): register the first owner in compile order
            pending_by_signature.setdefault(signature, assertion)
            return False
        proposed_metadata = _proposed_assertion_metadata(assertion)
        unreplaced: dict[str, JsonValue] = {}
        for field, retained_value in retained_metadata.items():
            if retained_value != proposed_metadata[field]:
                unreplaced[field] = {"stored": retained_value, "proposed": proposed_metadata[field]}
        review_issues.append(
            self._issue(
                ReviewIssueCode.POSSIBLE_COGNITION_CONFLICT,
                "warning",
                "assertion",
                index,
                (
                    "A structurally identical assertion already exists in the world graph "
                    "or earlier in this proposal; its id was retained and any valid new "
                    "evidence was linked to it. The proposed metadata differences were "
                    "NOT saved."
                ),
                [
                    "keep_existing_assertion",
                    "declare_supersedes_to_correct",
                    "raise_inquiry_for_conflict",
                ],
                actual_value={
                    "existing_assertion_id": existing_id,
                    "linked_evidence_ids": [item.observation_id for item in assertion.evidence],
                    "unreplaced": unreplaced,
                },
                candidate_ids=[existing_id],
                durable_id=existing_id,
                match_basis=["assertion_signature_collision"],
            )
        )
        return True

    def _preflight_similar_literal(
        self,
        connection: sqlite3.Connection,
        index: int,
        assertion: AssertionInput,
        review_issues: list[ReviewIssue],
    ) -> None:
        """Bounded C-class recall: similar literals are candidates, never verdicts.

        When a literal assertion's text resembles a stored assertion about the
        same subject and predicate, the host only recalls the candidate with a
        ``possible_cognition_conflict`` warning: it never judges a logical
        contradiction and never rewrites ``supersedes``. The proposed
        assertion still commits as-is. Byte-identical stored literals are
        explicitly skipped by the scan guard — they are not "similar"
        candidates: a structural twin is announced upstream by
        :meth:`_announce_assertion_dedup` (signature collision), and a
        byte-identical literal whose signature differs (different epistemic
        role, event window or qualifiers) is its own assertion, never a
        conflict warning.
        """
        if assertion.literal is None:
            return
        literal = str(assertion.literal)
        probe = bigrams(literal)
        if not probe:
            return
        rows = connection.execute(
            "SELECT id, literal_json FROM assertions"
            " WHERE subject_id = ? AND lower(predicate) = ?"
            " AND literal_json IS NOT NULL AND id != ? ORDER BY id",
            (assertion.subject_id, assertion.predicate.casefold(), assertion.id),
        ).fetchall()
        similar: list[dict[str, JsonValue]] = []
        for row in rows:
            stored_literal = str(json.loads(str(row["literal_json"])))
            if stored_literal == literal:
                continue
            if jaccard(probe, bigrams(stored_literal)) >= _FUZZY_SIMILARITY_THRESHOLD:
                similar.append({"assertion_id": str(row["id"]), "literal": stored_literal})
        if not similar:
            return
        similar_ids, similar_tail = _candidate_payload(
            str(entry["assertion_id"]) for entry in similar
        )
        bounded = {
            str(entry["assertion_id"]): entry
            for entry in similar
            if str(entry["assertion_id"]) in similar_ids
        }
        review_issues.append(
            self._issue(
                ReviewIssueCode.POSSIBLE_COGNITION_CONFLICT,
                "warning",
                "assertion",
                index,
                (
                    "The assertion's literal resembles an existing assertion about the same "
                    "subject and predicate; it was committed as-is with no contradiction "
                    "judgment and no supersedes rewrite."
                ),
                [
                    "keep_both_assertions",
                    "declare_supersedes_to_correct",
                    "raise_inquiry_for_conflict",
                ],
                actual_value={
                    "similar_assertions": [
                        bounded[entry_id] for entry_id in similar_ids if entry_id in bounded
                    ],
                    **similar_tail,
                },
                candidate_ids=similar_ids,
                durable_id=str(similar[0]["assertion_id"]),
                match_basis=["similar_literal"],
            )
        )

    def _compile_participant_edges(
        self,
        proposal: CognitionDeltaProposal,
        commit_id: str,
        connection: sqlite3.Connection,
        object_ids: dict[str, str],
        omitted_object_refs: set[str],
        unsupported_refs: set[str],
        review_issues: list[ReviewIssue],
    ) -> list[tuple[int, AssertionInput]]:
        """Compile each new event's explicit participants into participant edges.

        The compiler expands only the Agent's explicit declaration: subject is
        the event's durable id, predicate is ``has_participant``, object_id is
        the resolved participant GraphRef, and the qualifiers are the
        participant's qualifiers with a non-empty ``role`` folded into
        ``qualifiers["role"]`` (a role conflicting with ``qualifiers.role`` is
        rejected at the proposal boundary). Epistemic role, confidence and
        evidence pass through verbatim; unknown evidence observation ids are
        dropped with a warning (the same sanitization the assertion path
        applies) so one forged id cannot kill the whole delta. Assertion ids
        follow the durable id discipline (kind/commit/local_ref hash), so
        replaying the same commit mints identical ids. A non-provisional event
        with no participants gets an ``event_participant_incomplete`` warning
        with an inquiry suggestion; the host never derives edges from titles,
        literals or other assertions. Edges whose event or participant new
        object was omitted (identity alias claim conflict, or unsupported
        kind) are dropped with the object; the dropped edge is reported with
        the same code/action vocabulary as the sibling dangling-dependency
        paths (assertion, inquiry, observation link).

        F4: the compiled edges are returned paired with their source event
        indexes in declaration order instead of being appended directly; the
        caller runs the same signature-collision announcement as explicit
        assertions before appending them to the delta.
        """
        compiled: list[tuple[int, AssertionInput]] = []
        for event_index, item in enumerate(proposal.new_objects):
            if item.kind is not ObjectKind.EVENT:
                continue
            if item.local_ref in omitted_object_refs:
                # the event itself was omitted (alias collision): a compiled
                # edge cannot reference an object that will not exist
                continue
            event_id = object_ids[item.local_ref]
            if not item.participants:
                if not item.provisional:
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.EVENT_PARTICIPANT_INCOMPLETE,
                            "warning",
                            "new_object",
                            event_index,
                            (
                                "The event was committed without declared participants; "
                                "the host never infers participant edges from titles, "
                                "literals or other assertions."
                            ),
                            [
                                "declare_participants_in_future_proposal",
                                "raise_inquiry_for_participants",
                            ],
                            actual_value={
                                "local_ref": item.local_ref,
                                "durable_id": event_id,
                                "provisional": item.provisional,
                            },
                            durable_id=event_id,
                        )
                    )
                continue
            for participant_index, participant in enumerate(item.participants):
                if (
                    participant.object.local_ref is not None
                    and participant.object.local_ref in omitted_object_refs
                ):
                    # the participant is a new object omitted for an identity
                    # alias claim conflict (or an unsupported kind): this one
                    # edge would dangle, so it is omitted with a warning while
                    # the rest of the delta stays eligible
                    unsupported_dependency = participant.object.local_ref in unsupported_refs
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.UNSUPPORTED_OBJECT_KIND
                            if unsupported_dependency
                            else ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT,
                            "warning",
                            "new_object",
                            event_index,
                            (
                                "The participant edge was omitted because the participant "
                                "new object was omitted for the unsupported 'source' kind."
                                if unsupported_dependency
                                else (
                                    "The participant edge was omitted because the participant "
                                    "new object was omitted for an identity alias claim conflict."
                                )
                            ),
                            (
                                [
                                    "record_as_observation",
                                    "link_observation_as_evidence",
                                    "drop_source_new_object",
                                ]
                                if unsupported_dependency
                                else [
                                    "use_existing_object_id",
                                    "remove_duplicate_new_object",
                                    "choose_unclaimed_alias",
                                ]
                            ),
                            actual_value={
                                "event_local_ref": item.local_ref,
                                "participant_local_ref": participant.object.local_ref,
                            },
                            durable_id=event_id,
                        )
                    )
                    continue
                evidence: list[EvidenceInput] = []
                dropped: list[str] = []
                for link in participant.evidence:
                    if (
                        connection.execute(
                            "SELECT 1 FROM observations WHERE id = ?", (link.observation_id,)
                        ).fetchone()
                        is None
                    ):
                        dropped.append(link.observation_id)
                        continue
                    evidence.append(link)
                if dropped:
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.MISSING_EVIDENCE_ID,
                            "warning",
                            "new_object",
                            event_index,
                            (
                                "Unknown participant evidence ids were dropped; the "
                                "participant edge was retained without inventing "
                                "replacement observations."
                            ),
                            ["use_existing_observation_ids_in_future_proposals"],
                            actual_value={
                                "declared_evidence_ids": [
                                    link.observation_id for link in participant.evidence
                                ],
                                "missing_evidence_ids": dropped,
                            },
                            durable_id=event_id,
                            dropped_evidence_ids=dropped,
                        )
                    )
                qualifiers = dict(participant.qualifiers)
                if participant.role is not None:
                    qualifiers["role"] = participant.role
                compiled.append(
                    (
                        event_index,
                        AssertionInput(
                            id=durable_id(
                                "assertion",
                                commit_id,
                                f"{item.local_ref}:participant:{participant_index}",
                            ),
                            subject_id=event_id,
                            predicate="has_participant",
                            object_id=self._resolve_ref(
                                connection, participant.object, object_ids, "objects"
                            ),
                            literal=None,
                            epistemic_role=participant.epistemic_role,
                            confidence=participant.confidence,
                            evidence=evidence,
                            qualifiers=qualifiers,
                        ),
                    )
                )
        return compiled

    @staticmethod
    def _subject_id(reference: GraphRef, object_ids: dict[str, str]) -> str:
        r"""Resolve an inquiry subject to its durable object id for similarity checks.

        A ``local_ref`` resolves through the proposal's object identity map; an
        unknown mapping yields a ``"\0missing"`` sentinel that matches no stored
        subject, leaving compile-time reference validation (which raises) as the
        only authority on unknown local refs.

        Args:
            reference: The proposed inquiry's subject reference.
            object_ids: Proposal-local object refs to their durable ids.

        Returns:
            The durable subject object id, or the missing-subject sentinel.

        """
        if reference.local_ref is not None:
            return object_ids.get(reference.local_ref, "\0missing")
        return str(reference.memory_id)

    @staticmethod
    def _resolve_ref(
        connection: sqlite3.Connection,
        reference: GraphRef,
        local_ids: dict[str, str],
        table: str,
    ) -> str:
        if reference.local_ref is not None:
            resolved = local_ids.get(reference.local_ref)
            if resolved is None:
                raise ProposalValidationError(
                    [f"unknown {table[:-1]} local reference: {reference.local_ref}"]
                )
            return resolved
        memory_id = str(reference.memory_id)
        if connection.execute(f"SELECT 1 FROM {table} WHERE id = ?", (memory_id,)).fetchone() is None:
            raise ProposalValidationError([f"referenced {table[:-1]} memory does not exist: {memory_id}"])
        return memory_id

    @staticmethod
    def _object_update(
        connection: sqlite3.Connection,
        item: ObjectUpdateProposal,
        *,
        accepted_adds: Sequence[str],
        dropped_forms: Sequence[str] = (),
    ) -> ObjectInput | None:
        object_id = str(item.target.memory_id)
        row = connection.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
        if row is None:
            raise ProposalValidationError([f"referenced object memory does not exist: {object_id}"])
        if row["version"] != item.expected_version:
            # stale expected_version: the object changed since the model last
            # saw it, so applying the update could clobber newer knowledge;
            # skip the update rather than rejecting the whole delta
            return None
        # F1: the update's compiled alias list feeds the active identity
        # index (_persist_identity_aliases with skip_existing=True), so only
        # ACTIVE identity aliases may enter it; legacy object_aliases rows
        # are read-only history and must never be promoted (F1). New writes
        # never touch object_aliases (Task 2.1 Ruling A). F3: only the
        # preflight-accepted ADDs are compiled — claimed forms were omitted.
        aliases = WorldStore._stored_active_identity_aliases(connection, object_id)
        compiled_aliases = list(dict.fromkeys([*aliases, *accepted_adds]))
        if dropped_forms:
            # The same form declared in two actions is dropped from every
            # action (one alias_operation_invalid issue); it must not re-enter
            # the compiled alias list either, or the update-rewrite path would
            # activate an alias the receipt announces as omitted.
            dropped = set(dropped_forms)
            compiled_aliases = [
                alias
                for alias in compiled_aliases
                if _identity_alias_form(alias) not in dropped
            ]
        return ObjectInput(
            id=object_id,
            kind=ObjectKind(row["kind"]),
            type_key=str(row["type_key"]) if row["type_key"] is not None else None,
            canonical_name=str(row["canonical_name"]),
            aliases=compiled_aliases,
            domain_hints=json.loads(row["domain_hints_json"]),
            provisional=(bool(row["provisional"]) if item.provisional is None else item.provisional),
            event_time_start=_datetime(row["event_time_start"]),
            event_time_end=_datetime(row["event_time_end"]),
            event_time_precision=str(row["event_time_precision"] or "unknown"),
            expected_version=item.expected_version,
        )

    def _compile_alias_operations(
        self,
        connection: sqlite3.Connection,
        proposal: CognitionDeltaProposal,
        item: ObjectUpdateProposal,
        object_id: str,
        item_index: int,
        object_ids: dict[str, str],
        review_issues: list[ReviewIssue],
        alias_operations: list[AliasOperation],
        *,
        accepted_adds: Sequence[str],
    ) -> None:
        """Compile identity alias corrections into durable AliasOperations.

        Task 1.5 contract enforcement:

        - Every alias uses the shared identity normalizer; forms that normalize
          to empty are silently skipped.
        - The same normalized alias may not appear in more than one action of
          one update: the conflicting form is dropped from every action with a
          single ``alias_operation_invalid`` issue.
        - remove/demote apply only to the target's currently ACTIVE alias;
          otherwise the operation is omitted with ``alias_not_active``.
        - demote additionally requires, in the same proposal, an assertion
          naming the raw alias as ``name_usage`` for this object; otherwise the
          demote is omitted with ``demote_requires_name_usage``.
        - Omitted operations never block the other safe alias operations or the
          ``provisional`` update in the same object update.
        """
        # One shared decision point: a normalized form declared in more than
        # one action is dropped from every action. The object-update rewrite
        # consumes the same decision, so an omitted form can never be
        # re-activated through the compiled alias list.
        conflicts = _multi_action_alias_conflicts(
            accepted_adds, item.remove_aliases, item.demote_aliases
        )
        declared: list[tuple[AliasAction, str]] = [
            # F3: only preflight-accepted ADDs are compiled; claimed forms were
            # omitted with an IDENTITY_ALIAS_CLAIM_CONFLICT issue.
            *((AliasAction.ADD, raw) for raw in accepted_adds),
            *((AliasAction.REMOVE, raw) for raw in item.remove_aliases),
            *((AliasAction.DEMOTE, raw) for raw in item.demote_aliases),
        ]
        normalized: list[tuple[AliasAction, str, str] | None] = []
        for action, raw in declared:
            try:
                normalized_form = normalize_identity_alias(raw)
            except ValueError:
                normalized_form = None
            if normalized_form is None:
                normalized.append(None)
            else:
                normalized.append((action, raw, normalized_form))
        if conflicts:
            review_issues.append(
                self._issue(
                    ReviewIssueCode.ALIAS_OPERATION_INVALID,
                    "warning",
                    "object_update",
                    item_index,
                    (
                        "The same normalized alias appears in more than one identity "
                        "alias action; the conflicting forms were dropped from all actions."
                    ),
                    ["declare_each_normalized_alias_in_exactly_one_action"],
                    actual_value={
                        "object_id": object_id,
                        "normalized_aliases": conflicts,
                    },
                    durable_id=object_id,
                    match_basis=["single_action_per_normalized_alias"],
                )
            )
        conflicting = set(conflicts)
        for entry in normalized:
            if entry is None or entry[2] in conflicting:
                continue
            action, raw, normalized_form = entry
            if action is AliasAction.REMOVE or action is AliasAction.DEMOTE:
                active = connection.execute(
                    "SELECT 1 FROM identity_aliases"
                    " WHERE object_id = ? AND normalized_alias = ? AND status = 'active'",
                    (object_id, normalized_form),
                ).fetchone()
                if active is None:
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.ALIAS_OPERATION_INVALID,
                            "warning",
                            "object_update",
                            item_index,
                            (
                                "The alias is not a currently active identity alias of "
                                "the target object, so the remove/demote was omitted."
                            ),
                            ["add_the_alias_first", "check_the_alias_owner"],
                            actual_value={
                                "object_id": object_id,
                                "raw_alias": raw,
                                "normalized_alias": normalized_form,
                                "action": action.value,
                            },
                            durable_id=object_id,
                            match_basis=["alias_not_active"],
                        )
                    )
                    continue
                if action is AliasAction.DEMOTE and not self._proposal_declares_name_usage(
                    proposal, object_id, raw, object_ids
                ):
                    review_issues.append(
                        self._issue(
                            ReviewIssueCode.ALIAS_OPERATION_INVALID,
                            "warning",
                            "object_update",
                            item_index,
                            (
                                "The demote is omitted because the same proposal does not "
                                "declare a name_usage assertion naming the raw alias for "
                                "this object."
                            ),
                            ["declare_name_usage_assertion", "use_remove_instead"],
                            actual_value={
                                "object_id": object_id,
                                "raw_alias": raw,
                                "normalized_alias": normalized_form,
                                "action": action.value,
                            },
                            durable_id=object_id,
                            match_basis=["demote_requires_name_usage"],
                        )
                    )
                    continue
            alias_operations.append(
                AliasOperation(
                    object_id=object_id,
                    raw_alias=raw,
                    normalized_alias=normalized_form,
                    action=action,
                )
            )

    @staticmethod
    def _proposal_declares_name_usage(
        proposal: CognitionDeltaProposal,
        object_id: str,
        raw_alias: str,
        object_ids: dict[str, str],
    ) -> bool:
        """Return whether the same proposal asserts ``name_usage`` for the alias.

        The demote contract requires, in the SAME proposal, an assertion whose
        subject resolves to the updated object, ``predicate == "name_usage"``
        and ``literal == the raw pre-normalization alias``.
        """
        return any(
            assertion.predicate == "name_usage"
            and assertion.literal == raw_alias
            and ProposalCommitter._subject_id(assertion.subject, object_ids) == object_id
            for assertion in proposal.assertions
        )


_COMMENT_OBSERVATION_ID = re.compile(r".+-(?:comment|danmaku)-[0-9a-f]{32}\Z")
_COMMENT_MATERIAL_KINDS = frozenset({"comment", "danmaku"})


def _is_comment_observation(observation_id: str, metadata_json: str | None) -> bool:
    """Return whether durable provenance identifies comment-like material.

    Main document rows can reach DISCUSSION depth after reactions are read, so
    depth is not a material-kind discriminator. Hydrated rows carry modality
    and body-part rows carry material_kind; legacy comment/danmaku sub-content
    ids retain their content-derived kind token as the compatibility fallback.

    Args:
        observation_id: The stored observation id to classify.
        metadata_json: The observation's durable provenance metadata.

    Returns:
        True only for comment or danmaku material.

    """
    try:
        metadata = json.loads(metadata_json or "{}")
    except (TypeError, ValueError):
        metadata = {}
    if isinstance(metadata, dict) and (
        metadata.get("modality") in _COMMENT_MATERIAL_KINDS
        or metadata.get("material_kind") in _COMMENT_MATERIAL_KINDS
    ):
        return True
    return _COMMENT_OBSERVATION_ID.fullmatch(observation_id) is not None


def _evidence_reliability(row: sqlite3.Row) -> tuple[str | None, bool, bool]:
    """Return reliability plus mixed-primary and exact-part provenance flags."""
    try:
        metadata = json.loads(str(row["metadata_json"] or "{}"))
    except (TypeError, ValueError):
        metadata = {}
    reliability = metadata.get("material_reliability") if isinstance(metadata, dict) else None
    if isinstance(reliability, str):
        return reliability, False, bool(metadata.get("body_part"))
    body_json = row["body_json"]
    if not isinstance(body_json, str):
        return None, False, False
    try:
        body = parse_stored_body(body_json)
    except ValueError:
        return None, False, False
    if not isinstance(body, BodyEnvelope):
        return None, False, False
    reliabilities = {part.reliability for part in body.parts}
    if len(reliabilities) > 1:
        return None, True, False
    if len(reliabilities) == 1:
        return next(iter(reliabilities)), False, False
    return None, False, False


def durable_id(kind: str, commit_id: str, local_ref: str) -> str:
    """Derive the deterministic durable id of one delta item under a commit.

    The same (kind, commit_id, local_ref) triple always maps to the same id,
    so replaying a commit under its original id is idempotent.

    Args:
        kind: The delta item kind, e.g. ``"object"`` or ``"inquiry"``.
        commit_id: The commit the item belongs to.
        local_ref: The item's proposal-local reference within that commit.

    Returns:
        A stable ``f"{kind}-"`` prefixed id.

    """
    payload = f"{commit_id}:{kind}:{local_ref}".encode()
    return f"{kind}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _as_utc(value: datetime) -> datetime:
    """Normalize one datetime to an aware UTC value for comparisons."""
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _stored_datetime(value: object) -> datetime | None:
    """Parse one stored ISO event-time column back to an aware datetime.

    Task 2.3: stored times may be naive (the interval validator does not
    enforce timezone awareness), so naive values are read as UTC to make
    interval comparisons across proposal and stored rows well-defined.
    """
    if value is None:
        return None
    return _as_utc(datetime.fromisoformat(str(value)))
