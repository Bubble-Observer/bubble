"""finalize_graph: the deterministic formal-publish bridge for one wake.

Plan §6.10 / §6.11, D-007 / D-014 (slice 5 Part 1). Active staging compiles
deterministically into the existing formal cognitive delta and commits through
the Store's idempotent formal chain within one SQLite transaction (D-007
preferred route): the transaction that writes ``commit_receipts`` /
``world_audit`` also converges the wake's staging to ``finalized`` and records
the per-wake finalize receipt, so the split states the §6.11 crash points must
exclude are unreachable by construction.

Contract:

- Readiness reuses ``inspect_working_graph`` (D-013 single surface): blocked
  returns the structured blockers, staging is preserved, and a later patch +
  re-finalize converges.
- The compile is a pure function of the staging rows and the formal tables.
  Formal ids equal the host-issued staged ids (``{wake_id}:s{n}`` etc.),
  falling back to the committer's ``durable_id`` minting only when a legacy
  row already occupies the id. Update overlays compile to the covered formal
  id with the current formal version as ``expected_version``.
- Drops compile to nothing: staging has no formal retirement concept, and a
  dropped overlay is exactly "withdraw the update".
- Idempotency: a stored finalize receipt wins (repeat finalize returns it,
  never recompiles); an empty or all-abandoned working graph still publishes
  an honest empty formal commit.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field

from .committer import _INQUIRY_SIMILARITY_THRESHOLD, durable_id
from .contracts import (
    AssertionInput,
    CognitiveDelta,
    CommitReceipt,
    EpistemicRole,
    EvidenceInput,
    InquiryInput,
    InquiryResolution,
    ObjectInput,
    ObjectKind,
    WorldModel,
)
from .graph_contract import QUALIFIER_KEYS
from .preflight import inspect_working_graph
from .similarity import bigrams, jaccard
from .staging import _formal_assertion_exists, _formal_object_exists, read_active_staged
from .store import CommitReplayConflict, ObjectAliasCollision, WorldStore, _assertion_signature

#: Staged kinds the formal enum cannot represent directly, routed like the
#: legacy proposal boundary (committer._LEGACY_KIND_ROUTES). Empty since the
#: 2026-08-23 six-kind alignment: graph_patch kinds (person included) now
#: publish verbatim; kept as a table so any future legacy value folds the
#: same way the historical boundary did.
_LEGACY_STAGED_KIND_ROUTES: dict[str, ObjectKind] = {}

#: The assertion-evidence role every staged evidence id compiles to. Staging
#: stores plain observation id lists; the formal link needs a typed role.
_STAGED_EVIDENCE_ROLE = "supports"

#: The 0.20 bigram-jaccard line for the similar-open-inquiry report is
#: imported from the committer (E2, proposal link) so a recalibration never
#: drifts between the two links. Formal compile only reports on the F-B
#: tolerance channel, it never omits — the deterministic finalizer must not
#: drop staged work, and an operator can collapse the line later.

# G5b-1 same-wake near-concurrency scope is deliberately local-process: one
# finalizer serializes calls for the same world/wake so the receipt-first
# replay check observes the first completed transaction. SQLite remains the
# durable transaction authority; this lock does not claim distributed
# multi-writer support.
_FINALIZE_LOCKS_GUARD = threading.Lock()
_FINALIZE_LOCKS: dict[tuple[str, str], threading.RLock] = {}


class FinalizeStats(WorldModel):
    """Mechanical counts of one wake's staging shape at finalize time."""

    objects_created: int = 0
    objects_updated: int = 0
    assertions_created: int = 0
    inquiries_created: int = 0
    abandoned: int = 0
    total_items: int = 0


class FinalizeReceipt(WorldModel):
    """The durable outcome of one finalize_graph call for one wake.

    Only ``published`` receipts are persisted (``finalize_receipts``); the
    other statuses are transient reports that keep staging preserved and are
    never replayed as authority.
    """

    wake_id: str
    status: Literal[
        "published",
        "already_published",
        "blocked",
        "compile_failed",
        "commit_rejected",
        "wake_closed",
    ]
    commit_id: str | None = None
    committed_at: datetime | None = None
    stats: FinalizeStats = Field(default_factory=FinalizeStats)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[dict[str, Any]] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    #: staged_id -> formal id for every compiled item (traceability, §6.10.7).
    item_ids: dict[str, str] = Field(default_factory=dict)
    all_work_abandoned: bool = False
    replayed: bool = False


class FinalizeCompileError(ValueError):
    """Staged content is not deterministically compilable to a formal delta."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _finalize_commit_id(wake_id: str) -> str:
    """Return the one deterministic formal commit id a wake may publish under."""
    return f"{wake_id}:finalize"


@contextmanager
def wake_mutation_lock(store: WorldStore, wake_id: str) -> Iterator[None]:
    """Serialize Graph Shell patch/finalize mutations for one local-process wake."""
    key = (str(store.path.resolve()), wake_id)
    with _FINALIZE_LOCKS_GUARD:
        lock = _FINALIZE_LOCKS.setdefault(key, threading.RLock())
    with lock:
        yield


def finalize_graph(store: WorldStore, wake_id: str) -> FinalizeReceipt:
    """Serialize one process's same-wake callers, then publish/replay."""
    with wake_mutation_lock(store, wake_id):
        return _finalize_graph_serialized(store, wake_id)


def _finalize_graph_serialized(store: WorldStore, wake_id: str) -> FinalizeReceipt:
    """Publish the wake's working graph through the unique formal chain.

    Returns a ``FinalizeReceipt``; a published receipt is durable and a
    repeat call returns it unchanged (``already_published``). Blocked,
    compile-failed and commit-rejected calls leave staging active for
    further patching; ``wake_closed`` means the wake already published and
    items staged afterwards must move to a new wake.
    """
    if not wake_id.strip():
        raise ValueError("wake id must be non-empty")
    commit_id = _finalize_commit_id(wake_id)

    stored = _stored_receipt(store, wake_id)
    if stored is not None:
        active = read_active_staged(store, wake_id)
        stranded = {
            kind: [str(row["staged_id"]) for row in rows]
            for kind, rows in active.items()
            if rows
        }
        if stranded:
            # A wake publishes exactly once (I1): active items staged after
            # the stored receipt can never publish under this wake. Reject
            # them explicitly with the current staging shape instead of
            # replaying the old receipt over new work (G5a F-3).
            shape = _staging_shape(store, wake_id)
            stranded_ids = [item for kind_ids in stranded.values() for item in kind_ids]
            return FinalizeReceipt(
                wake_id=wake_id,
                status="wake_closed",
                commit_id=stored.commit_id,
                committed_at=stored.committed_at,
                stats=shape,
                all_work_abandoned=shape.total_items == 0 and shape.abandoned > 0,
                blockers=[
                    {
                        "code": "wake_closed",
                        "ref": wake_id,
                        "message": (
                            f"wake {wake_id} already finalized at {stored.committed_at}; "
                            f"{len(stranded_ids)} active item(s) staged afterwards will "
                            "never publish — start a new wake"
                        ),
                        "stranded": stranded_ids,
                    }
                ],
            )
        return stored.model_copy(update={"status": "already_published", "replayed": True})

    shape = _staging_shape(store, wake_id)
    all_abandoned = shape.total_items == 0 and shape.abandoned > 0
    report = inspect_working_graph(store, wake_id)
    if report["readiness"] != "ready":
        return FinalizeReceipt(
            wake_id=wake_id,
            status="blocked",
            stats=shape,
            warnings=list(report["warnings"]),
            blockers=list(report["blockers"]),
            all_work_abandoned=all_abandoned,
        )
    tolerance_warnings: list[str] = []
    try:
        delta, item_ids = compile_final_delta(
            store, wake_id, commit_id, warnings=tolerance_warnings
        )
    except FinalizeCompileError as error:
        return FinalizeReceipt(
            wake_id=wake_id,
            status="compile_failed",
            stats=shape,
            warnings=list(report["warnings"]),
            problems=list(error.problems),
            all_work_abandoned=all_abandoned,
        )

    warnings = list(report["warnings"]) + tolerance_warnings
    try:
        store.finalized_memory_commit(
            delta,
            commit_id,
            _make_finalizer(wake_id, shape, warnings, item_ids, all_abandoned),
        )
    except CommitReplayConflict:
        # a replay guard firing for the finalize commit id means the durable
        # commit/receipt states diverged — an anomaly that must stay loud
        raise
    except (ObjectAliasCollision, ValueError) as error:
        # A Store hard gate refused the compiled delta (alias collision,
        # stale expected version, ...). The transaction rolled back cleanly
        # and the staging stays active: the agent gets a structured
        # rejection it can act on (drop / re-patch) instead of a raw
        # exception (G5a F-2).
        message = f"store rejected the compiled delta: {error}"
        text = str(error)
        if "stale object version:" in text or "stale inquiry version:" in text:
            # The overlay's durable patch-time base no longer matches the
            # formal version: the gate fired, nothing was overwritten, and
            # only drop -> reread -> re-patch can converge (G5b-1 F-01).
            message += (
                "; the formal base changed after this overlay was patched — "
                "memory_read the target, then drop and re-patch this overlay"
            )
        return FinalizeReceipt(
            wake_id=wake_id,
            status="commit_rejected",
            stats=shape,
            warnings=warnings,
            problems=[message],
            all_work_abandoned=all_abandoned,
        )
    published = _stored_receipt(store, wake_id)
    if published is None:  # pragma: no cover — the finalizer just inserted it
        raise RuntimeError(f"finalize committed but receipt is missing for wake {wake_id}")
    return published


def compile_final_delta(
    store: WorldStore,
    wake_id: str,
    commit_id: str,
    warnings: list[str] | None = None,
) -> tuple[CognitiveDelta, dict[str, str]]:
    """Deterministically compile active staging into a formal cognitive delta.

    A pure function of the staging rows and the formal tables: the same
    staging always yields the same delta, so a re-finalize after a crash or a
    replayed commit converges byte-identically.

    ``warnings`` opens the publish-tolerance channel (F-B): content that
    compiles only with loss (for example, an invented qualifier key written
    before the write path enforced the closed set) is dropped and reported
    there instead of failing the whole delta. When ``warnings`` is None the
    strict behavior is unchanged and such content fails the compile.

    Returns:
        The ``(delta, item_ids)`` pair; ``item_ids`` maps every compiled
        staged_id to its formal id.

    Raises:
        FinalizeCompileError: When any active row is not compilable; nothing
            has been written and staging is preserved.

    """
    active = read_active_staged(store, wake_id)
    problems: list[str] = []
    object_ids: dict[str, str] = {}
    item_ids: dict[str, str] = {}
    objects: list[ObjectInput] = []

    with store.read_connection() as connection:
        for row in active["objects"]:
            staged_id = str(row["staged_id"])
            target_ref = row.get("target_ref")
            try:
                if target_ref:
                    target = str(target_ref)
                    formal = connection.execute(
                        "SELECT 1 FROM objects WHERE id = ?", (target,)
                    ).fetchone()
                    if formal is None:
                        problems.append(
                            f"object update overlay targets a missing formal object: {target}"
                        )
                        continue
                    # The overlay compares against its durable patch-time
                    # base, never against the version re-read now (G5b-1
                    # F-01): a formal change between preflight and compile
                    # must fire the Store's expected_version hard gate
                    # (commit_rejected), never silently overwrite the newer
                    # content with the stale overlay.
                    base_version = row.get("base_version")
                    if base_version is None:
                        problems.append(
                            f"object update overlay {staged_id} has no trustworthy "
                            "patch-time base version — memory_read the target, then "
                            "drop and re-patch"
                        )
                        continue
                    object_ids[staged_id] = target
                    item_ids[staged_id] = target
                    objects.append(
                        _object_update_input(row, target, int(base_version), problems)
                    )
                else:
                    formal_id = (
                        durable_id("object", commit_id, staged_id)
                        if _formal_object_exists(connection, staged_id)
                        else staged_id
                    )
                    object_ids[staged_id] = formal_id
                    item_ids[staged_id] = formal_id
                    objects.append(_object_create_input(row, formal_id, problems))
            except (TypeError, ValueError) as error:
                problems.append(f"object {staged_id}: {error}")

        inquiries: list[InquiryInput] = []
        # Same-wake staged_id -> formal inquiry id map (Wave B / D-019): the
        # answering-assertion and deepens refs are resolved against the FULL
        # map in later passes. Like assertions, staged inquiry id string order
        # is not staging sequence (i12 sorts before i2), so deepens links are
        # resolved only after every create row has claimed its formal id.
        staged_inquiry_ids: dict[str, str] = {}
        pending_inquiries: list[tuple[str, InquiryInput, object]] = []
        for row in active["inquiries"]:
            staged_id = str(row["staged_id"])
            # Resolve rows (kind 'resolution') are not inquiry creates; they
            # compile to InquiryResolution entries in the resolve pass below.
            if str(row["kind"]) == "resolution":
                continue
            try:
                subject_id = _resolve_object_ref(
                    str(row["subject_ref"]), object_ids, connection, problems, staged_id
                )
                if subject_id is None:
                    continue
                formal_id = (
                    durable_id("inquiry", commit_id, staged_id)
                    if _formal_inquiry_exists(connection, staged_id)
                    else staged_id
                )
                item_ids[staged_id] = formal_id
                staged_inquiry_ids[staged_id] = formal_id
                pending_inquiries.append(
                    (
                        staged_id,
                        InquiryInput(
                            id=formal_id,
                            subject_id=subject_id,
                            prompt=str(row["prompt"]),
                            rationale=str(row["rationale"]),
                            kind=str(row["kind"]),
                        ),
                        row.get("deepens_ref"),
                    )
                )
            except (TypeError, ValueError) as error:
                problems.append(f"inquiry {staged_id}: {error}")
        # T2-2 similar-open-inquiry report (observability, not semantics):
        # the formal write path has no prompt-level dedup (store._write_inquiry
        # dedups by id only), so a wake can publish a factual repeat of an
        # open/dormant line. Compile reports the collision on the F-B warnings
        # channel and publishes anyway — the deterministic finalizer never
        # drops staged work, and the operator can collapse the line later.
        # A declared, resolvable deepens link is exempt: it is the model's
        # intent to deepen that very inquiry (committer E2's rule).
        open_inquiry_rows = connection.execute(
            "SELECT id, subject_id, prompt FROM inquiries"
            " WHERE status IN ('open','dormant')"
        ).fetchall()
        compiled_inquiries: list[tuple[str, str, str]] = []
        for staged_id, inquiry_input, deepens_ref in pending_inquiries:
            deepens_id = None
            if deepens_ref is not None:
                deepens_id = _resolve_inquiry_ref(
                    str(deepens_ref), staged_inquiry_ids, connection, problems, staged_id
                )
            if deepens_id is not None and deepens_id != inquiry_input.id:
                inquiries.append(
                    inquiry_input.model_copy(update={"deepens_id": deepens_id})
                )
                compiled_inquiries.append(
                    (staged_id, inquiry_input.subject_id, inquiry_input.prompt)
                )
                continue
            if warnings is not None:
                prompt_bigrams = bigrams(inquiry_input.prompt)
                hits = [
                    str(row["id"])
                    for row in open_inquiry_rows
                    if str(row["subject_id"]) == inquiry_input.subject_id
                    and jaccard(prompt_bigrams, bigrams(str(row["prompt"])))
                    >= _INQUIRY_SIMILARITY_THRESHOLD
                ]
                # intra-delta: already compiled inquiries of this very delta
                hits += [
                    other_staged
                    for other_staged, other_subject, other_prompt in compiled_inquiries
                    if other_subject == inquiry_input.subject_id
                    and jaccard(prompt_bigrams, bigrams(other_prompt))
                    >= _INQUIRY_SIMILARITY_THRESHOLD
                ]
                if hits:
                    warnings.append(
                        {
                            "code": "similar_open_inquiry",
                            "ref": staged_id,
                            "message": (
                                f"inquiry {staged_id} repeats an open/dormant line "
                                f"(bigram jaccard >= {_INQUIRY_SIMILARITY_THRESHOLD}): "
                                + ", ".join(dict.fromkeys(hits))
                            ),
                        }
                    )
            inquiries.append(inquiry_input)
            compiled_inquiries.append(
                (staged_id, inquiry_input.subject_id, inquiry_input.prompt)
            )

        assertions: list[AssertionInput] = []
        # Formal signature dedup mirror (store._write_assertion): two staged
        # assertions with the same identity signature collapse onto the first
        # row that carries it — stored or in this very delta. The compile
        # resolves the same target so the receipt's item_ids is truthful.
        signature_ids: dict[str, str] = {}
        carrier_indices: dict[str, int] = {}
        # Same-wake staged_id -> formal id map (D-014 ③ / Core §6.7): a
        # supersedes ref may name a live staged assertion of this wake, whose
        # compiled formal id it resolves to. The map is completed for every
        # row before any supersedes ref is resolved (pass 2): staged_id string
        # order is not staging sequence (a11 sorts before a2), so a two-digit
        # superseder can precede its single-digit target.
        staged_assertion_ids: dict[str, str] = {}
        prepared: list[tuple[str, AssertionInput, object]] = []
        for row in active["assertions"]:
            staged_id = str(row["staged_id"])
            try:
                subject_id = _resolve_object_ref(
                    str(row["subject_ref"]), object_ids, connection, problems, staged_id
                )
                if subject_id is None:
                    continue
                object_ref = row.get("object_ref")
                object_id = (
                    _resolve_object_ref(
                        str(object_ref), object_ids, connection, problems, staged_id
                    )
                    if object_ref is not None
                    else None
                )
                literal = row.get("literal")
                if object_id is None and literal is None:
                    problems.append(
                        f"assertion {staged_id} has neither object_ref nor literal"
                    )
                    continue
                if object_id is not None and literal is not None:
                    problems.append(
                        f"assertion {staged_id} declares both object_ref and literal"
                    )
                    continue
                evidence = [
                    EvidenceInput(observation_id=str(observation_id), role=_STAGED_EVIDENCE_ROLE)
                    for observation_id in (row.get("evidence") or [])
                ]
                tentative_id = (
                    durable_id("assertion", commit_id, staged_id)
                    if _formal_assertion_exists(connection, staged_id)
                    else staged_id
                )
                # Wave B: a staged answer link (answers_ref) compiles to the
                # formal answers_inquiry_id; a dangling ref fails the delta.
                answers_ref = row.get("answers_ref")
                answers_inquiry_id = (
                    _resolve_inquiry_ref(
                        str(answers_ref), staged_inquiry_ids, connection, problems, staged_id
                    )
                    if answers_ref is not None
                    else None
                )
                staged_qualifiers = dict(row.get("qualifiers") or {})
                unknown_qualifier_keys = sorted(set(staged_qualifiers) - QUALIFIER_KEYS)
                if unknown_qualifier_keys:
                    # F-B publish tolerance: staging written before the write
                    # path enforced the closed key set may hold invented keys.
                    # Drop them and report, never fail the whole wake on one
                    # bad key; the write path now prevents new ones. Without
                    # the tolerance channel this stays a fatal problem.
                    dropped = (
                        f"assertion {staged_id} qualifier keys dropped (not in the "
                        f"closed set): {', '.join(unknown_qualifier_keys)}"
                    )
                    if warnings is not None:
                        warnings.append(
                            {
                                "code": "qualifier_keys_dropped",
                                "ref": staged_id,
                                "message": dropped,
                            }
                        )
                    else:
                        problems.append(dropped)
                    staged_qualifiers = {
                        key: value
                        for key, value in staged_qualifiers.items()
                        if key in QUALIFIER_KEYS
                    }
                assertion_input = AssertionInput(
                    id=tentative_id,
                    subject_id=subject_id,
                    predicate=str(row["predicate"]),
                    object_id=object_id,
                    literal=literal,
                    epistemic_role=EpistemicRole(str(row["epistemic_role"])),
                    confidence=float(row["confidence"]),
                    evidence=evidence,
                    qualifiers=staged_qualifiers,
                    event_time_start=_event_time(
                        row.get("event_time_start"), problems, staged_id
                    ),
                    event_time_end=_event_time(
                        row.get("event_time_end"), problems, staged_id
                    ),
                    supersedes_id=None,  # resolved in pass 2 against the full map
                    answers_inquiry_id=answers_inquiry_id,
                )
                signature = _assertion_signature(assertion_input)
                existing = connection.execute(
                    "SELECT id FROM assertions WHERE signature = ?", (signature,)
                ).fetchone()
                if signature in signature_ids:
                    # a later twin of a signature already claimed in this
                    # delta: the store collapses it onto the first row
                    # carrying the signature and links its evidence there
                    # (store._write_assertion) — mirror that by merging the
                    # evidence (and any supersede target the carrier lacks)
                    # and skipping the duplicate entry, which the unique-id
                    # delta contract forbids (store._unique_ids)
                    formal_id = signature_ids[signature]
                    carrier_index = carrier_indices[signature]
                    carrier_staged_id, carried_input, carried_ref = prepared[carrier_index]
                    merged_ref = (
                        carried_ref if carried_ref is not None else row.get("supersedes_ref")
                    )
                    prepared[carrier_index] = (
                        carrier_staged_id,
                        carried_input.model_copy(
                            update={
                                "evidence": _merge_evidence(
                                    carried_input.evidence, assertion_input.evidence
                                )
                            }
                        ),
                        merged_ref,
                    )
                    item_ids[staged_id] = formal_id
                    staged_assertion_ids[staged_id] = formal_id
                    continue
                formal_id = (
                    str(existing["id"]) if existing is not None else tentative_id
                )
                if formal_id != tentative_id:
                    assertion_input = assertion_input.model_copy(update={"id": formal_id})
                signature_ids[signature] = formal_id
                carrier_indices[signature] = len(prepared)
                item_ids[staged_id] = formal_id
                staged_assertion_ids[staged_id] = formal_id
                prepared.append((staged_id, assertion_input, row.get("supersedes_ref")))
            except (TypeError, ValueError) as error:
                problems.append(f"assertion {staged_id}: {error}")
        for staged_id, assertion_input, supersedes_ref in prepared:
            if supersedes_ref is None:
                assertions.append(assertion_input)
                continue
            supersedes_id = _resolve_assertion_ref(
                str(supersedes_ref),
                staged_assertion_ids,
                connection,
                problems,
                staged_id,
            )
            if supersedes_id is None:
                assertions.append(assertion_input)
                continue
            if supersedes_id == assertion_input.id:
                # the correction collapses onto the very row it supersedes
                # (signature dedup, store._write_assertion): the edge is never
                # materialized, so encoding it would self-reference a fresh
                # row and trip the supersedes_id foreign key (G5a F-6)
                assertions.append(assertion_input)
                continue
            assertions.append(
                assertion_input.model_copy(update={"supersedes_id": supersedes_id})
            )
        # The store writes delta assertions in list order under an immediate
        # supersedes_id foreign key: a superseder must follow its in-delta
        # target, which staged_id string order does not guarantee (G5a F-6).
        assertions = _supersede_write_order(assertions, problems)

        # Wave B resolve pass: staged resolve rows (kind 'resolution') compile
        # to version-checked InquiryResolution entries. This pass runs AFTER
        # the assertion pass so the delta-answer gate can see the compiled
        # answering assertions — the store requires the answer to be
        # co-published in the same delta (same-delta create+resolve, D-019).
        delta_answers: set[str] = {
            str(assertion.answers_inquiry_id)
            for assertion in assertions
            if assertion.answers_inquiry_id is not None
            and assertion.epistemic_role is not EpistemicRole.UNCERTAINTY
        }
        staged_inquiry_versions = {
            str(row["staged_id"]): int(row["version"])
            for row in active["inquiries"]
            if str(row["kind"]) != "resolution"
        }
        resolve_inquiries: list[InquiryResolution] = []
        seen_targets: dict[str, str] = {}
        for row in active["inquiries"]:
            if str(row["kind"]) != "resolution":
                continue
            staged_id = str(row["staged_id"])
            target_ref = row.get("target_ref")
            try:
                expected_version = int(row["version"] or 0)
                formal_id = (
                    staged_inquiry_ids.get(str(target_ref)) if target_ref is not None else None
                )
                if formal_id is None:
                    formal_row = (
                        connection.execute(
                            "SELECT status, version, resolved_at FROM inquiries WHERE id = ?",
                            (target_ref,),
                        ).fetchone()
                        if target_ref is not None
                        else None
                    )
                    if formal_row is None:
                        problems.append(
                            f"resolve {staged_id}: target {target_ref} compiles to "
                            "neither a staged inquiry of this wake nor a formal inquiry"
                        )
                        continue
                    formal_id = str(target_ref)
                    if formal_row["resolved_at"] is not None:
                        problems.append(
                            f"resolve {staged_id}: inquiry {target_ref} is already resolved"
                        )
                        continue
                    if str(formal_row["status"]) not in {"open", "dormant"}:
                        problems.append(
                            f"resolve {staged_id}: inquiry {target_ref} is "
                            f"{formal_row['status']}, not open or dormant"
                        )
                        continue
                    if int(formal_row["version"]) != expected_version:
                        problems.append(
                            f"resolve {staged_id}: stale expected_version "
                            f"{expected_version} for inquiry {target_ref} at version "
                            f"{formal_row['version']}"
                        )
                        continue
                else:
                    # The store inserts a delta-created inquiry at version 1 in
                    # this very commit, and the staged target row must still be
                    # at the version the resolve froze. A target updated after
                    # the resolve went stale (same-delta gate, D-019).
                    if int(staged_inquiry_versions.get(str(target_ref), 0)) != expected_version:
                        problems.append(
                            f"resolve {staged_id}: stale expected_version "
                            f"{expected_version} for inquiry {target_ref} at version "
                            f"{staged_inquiry_versions.get(str(target_ref), '?')}"
                        )
                        continue
                    if expected_version != 1:
                        problems.append(
                            f"resolve {staged_id}: inquiry {target_ref} is created by "
                            "this same commit; expected_version must be 1"
                        )
                        continue
                if formal_id in seen_targets:
                    problems.append(
                        f"resolve {staged_id}: inquiry {formal_id} already has a "
                        f"resolve in this wake ({seen_targets[formal_id]})"
                    )
                    continue
                if formal_id not in delta_answers:
                    problems.append(
                        f"resolve {staged_id}: no non-uncertainty answering assertion "
                        f"in this delta for inquiry {formal_id}"
                    )
                    continue
                seen_targets[formal_id] = staged_id
                resolve_inquiries.append(
                    InquiryResolution(id=formal_id, expected_version=expected_version)
                )
                item_ids[staged_id] = formal_id
            except (TypeError, ValueError) as error:
                problems.append(f"resolve {staged_id}: {error}")

    if problems:
        raise FinalizeCompileError(problems)
    return (
        CognitiveDelta(
            objects=objects,
            assertions=assertions,
            inquiries=inquiries,
            resolve_inquiries=resolve_inquiries,
        ),
        item_ids,
    )


def _object_create_input(
    row: Mapping[str, Any], formal_id: str, problems: list[str]
) -> ObjectInput:
    kind, type_key = _formal_kind(row)
    return ObjectInput(
        id=formal_id,
        kind=kind,
        type_key=type_key,
        canonical_name=str(row["canonical_name"]),
        aliases=list(row.get("aliases") or []),
        domain_hints=json.loads(row.get("domain_hints_json") or "[]"),
        provisional=bool(row.get("provisional")),
        event_time_start=_event_time(row.get("event_time_start"), problems, str(row["staged_id"])),
        event_time_end=_event_time(row.get("event_time_end"), problems, str(row["staged_id"])),
    )


def _object_update_input(
    row: Mapping[str, Any], formal_id: str, expected_version: int, problems: list[str]
) -> ObjectInput:
    kind, type_key = _formal_kind(row)
    return ObjectInput(
        id=formal_id,
        kind=kind,
        type_key=type_key,
        canonical_name=str(row["canonical_name"]),
        aliases=list(row.get("aliases") or []),
        domain_hints=json.loads(row.get("domain_hints_json") or "[]"),
        provisional=bool(row.get("provisional")),
        event_time_start=_event_time(row.get("event_time_start"), problems, str(row["staged_id"])),
        event_time_end=_event_time(row.get("event_time_end"), problems, str(row["staged_id"])),
        expected_version=expected_version,
    )


def _formal_kind(row: Mapping[str, Any]) -> tuple[ObjectKind, str | None]:
    """Route a staged kind to a formal ObjectKind (verbatim since six kinds)."""
    value = str(row["kind"])
    routed = _LEGACY_STAGED_KIND_ROUTES.get(value)
    if routed is not None:
        type_key = row.get("type_key")
        return routed, (type_key if type_key is not None else value)
    return ObjectKind(value), row.get("type_key")


def _event_time(
    value: object, problems: list[str] | None, ref: str
) -> datetime | None:
    """Parse a staged event-time column ('' or None means no bound)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        if problems is not None:
            problems.append(f"{ref}: invalid event_time value {value!r}")
        return None


def _resolve_object_ref(
    reference: str,
    staged_object_ids: Mapping[str, str],
    connection: sqlite3.Connection,
    problems: list[str],
    ref_owner: str,
) -> str | None:
    """Map a subject/object ref to its formal object id (staged or formal)."""
    formal = staged_object_ids.get(reference)
    if formal is not None:
        return formal
    if _formal_object_exists(connection, reference):
        return reference
    problems.append(
        f"{ref_owner}: object ref {reference} resolves to neither staged nor formal"
    )
    return None


def _resolve_inquiry_ref(
    reference: str,
    staged_inquiry_ids: Mapping[str, str],
    connection: sqlite3.Connection,
    problems: list[str],
    ref_owner: str,
) -> str | None:
    """Map a deepens/answer ref to a formal inquiry id (staged or formal).

    The ref may name a live staged inquiry of this wake (its compiled formal
    id, Wave B / D-019) or a stored formal inquiry; anything else fails closed
    as a compile problem.
    """
    formal = staged_inquiry_ids.get(reference)
    if formal is not None:
        return formal
    if _formal_inquiry_exists(connection, reference):
        return reference
    problems.append(
        f"{ref_owner}: inquiry ref {reference} is neither a live staged "
        "inquiry of this wake nor a formal inquiry"
    )
    return None


def _resolve_assertion_ref(
    reference: str,
    staged_assertion_ids: Mapping[str, str],
    connection: sqlite3.Connection,
    problems: list[str],
    ref_owner: str,
) -> str | None:
    """Map a supersedes ref to a formal assertion id.

    The ref may name a live staged assertion of this wake (its compiled
    formal id, Core §6.7 / D-014 ③) or a stored formal assertion; anything
    else fails closed as a compile problem.
    """
    formal = staged_assertion_ids.get(reference)
    if formal is not None:
        return formal
    if _formal_assertion_exists(connection, reference):
        return reference
    problems.append(
        f"{ref_owner}: supersedes ref {reference} is neither a live staged "
        "assertion of this wake nor a formal assertion"
    )
    return None


def _supersede_write_order(
    assertions: list[AssertionInput], problems: list[str]
) -> list[AssertionInput]:
    """Order the delta so each assertion follows the member it supersedes.

    The store checks the ``supersedes_id`` foreign key immediately per write,
    and the compile's row order (staged_id string) is not staging sequence
    (``a11`` sorts before ``a2``) — a superseder must therefore not be written
    before its in-delta target. Entries are tracked by position because
    signature-dedup pairs share a formal id, and a supersede cycle fails
    closed as a compile problem (the delta order is then left unchanged).
    """
    target_first: dict[str, int] = {}
    for index, assertion in enumerate(assertions):
        target_first.setdefault(assertion.id, index)
    ordered: list[AssertionInput] = []
    emitted: set[int] = set()
    in_progress: set[int] = set()

    def emit(index: int) -> None:
        if index in emitted:
            return
        in_progress.add(index)
        assertion = assertions[index]
        target = assertion.supersedes_id
        if target is not None and target != assertion.id:
            target_index = target_first.get(target)
            if target_index is not None:
                if target_index in in_progress:
                    problems.append(
                        f"assertion {assertion.id}: supersede cycle at {target}"
                    )
                else:
                    emit(target_index)
        in_progress.discard(index)
        emitted.add(index)
        ordered.append(assertion)

    for index in range(len(assertions)):
        emit(index)
    return assertions if problems else ordered


def _merge_evidence(
    base: list[EvidenceInput], extra: list[EvidenceInput]
) -> list[EvidenceInput]:
    """Merge two evidence lists, deduplicated by observation id (order preserved).

    Mirrors the store's evidence linking for a collapsed twin pair: the
    store links every entry's evidence to the surviving row
    (``INSERT OR IGNORE`` keyed by observation id).
    """
    seen: set[str] = set()
    merged: list[EvidenceInput] = []
    for evidence in [*base, *extra]:
        if evidence.observation_id in seen:
            continue
        seen.add(evidence.observation_id)
        merged.append(evidence)
    return merged


def _formal_inquiry_exists(connection: sqlite3.Connection, identifier: str) -> bool:
    return (
        connection.execute("SELECT 1 FROM inquiries WHERE id = ?", (identifier,)).fetchone()
        is not None
    )


def _staging_shape(store: WorldStore, wake_id: str) -> FinalizeStats:
    """Mechanical counts of the wake's staging (active shapes + abandoned)."""
    active = read_active_staged(store, wake_id)
    objects_created = sum(1 for row in active["objects"] if not row.get("target_ref"))
    # resolution staging rows carry kind='resolution' (the same row-class
    # marker the finalize layer uses to compile resolve rows); only actual
    # create rows count as created (M-EXT-01).
    inquiries_created = sum(1 for row in active["inquiries"] if row.get("kind") != "resolution")
    with store.read_connection() as connection:
        abandoned = 0
        for table in ("staged_objects", "staged_assertions", "staged_inquiries"):
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM {table}"
                " WHERE wake_id = ? AND status = 'abandoned'",
                (wake_id,),
            ).fetchone()
            abandoned += int(row["count"])
    return FinalizeStats(
        objects_created=objects_created,
        objects_updated=len(active["objects"]) - objects_created,
        assertions_created=len(active["assertions"]),
        inquiries_created=inquiries_created,
        abandoned=abandoned,
        total_items=(
            len(active["objects"]) + len(active["assertions"]) + len(active["inquiries"])
        ),
    )


def _stored_receipt(store: WorldStore, wake_id: str) -> FinalizeReceipt | None:
    """Return the wake's durable finalize receipt, if any."""
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT receipt_json FROM finalize_receipts WHERE wake_id = ?", (wake_id,)
        ).fetchone()
    if row is None:
        return None
    return FinalizeReceipt.model_validate_json(row["receipt_json"])


def _finalize_staging_rows(connection: sqlite3.Connection, wake_id: str, now: str) -> None:
    """Converge the wake's active staging to the finalized terminal state."""
    for table in ("staged_objects", "staged_assertions", "staged_inquiries"):
        connection.execute(
            f"UPDATE {table} SET status = 'finalized', updated_at = ?"
            " WHERE wake_id = ? AND status = 'active'",
            (now, wake_id),
        )


def _make_finalizer(
    wake_id: str,
    stats: FinalizeStats,
    warnings: list[dict[str, Any]],
    item_ids: dict[str, str],
    all_work_abandoned: bool,
) -> Callable[[sqlite3.Connection, CommitReceipt], None]:
    """Build the same-transaction finalizer for the formal commit.

    Runs inside the Store's commit transaction (D-007 preferred route): the
    staging finalization and the receipt insert either commit with the formal
    delta or roll back with it. The receipt insert is ``INSERT OR IGNORE`` so
    a deterministic-compile race between two finalizes of the same wake
    converges on the winner's (identical) receipt instead of erroring.
    """

    def finalize(connection: sqlite3.Connection, receipt: CommitReceipt) -> None:
        now = _now()
        _finalize_staging_rows(connection, wake_id, now)
        payload = FinalizeReceipt(
            wake_id=wake_id,
            status="published",
            commit_id=receipt.commit_id,
            committed_at=receipt.committed_at,
            stats=stats,
            warnings=warnings,
            item_ids=item_ids,
            all_work_abandoned=all_work_abandoned,
        ).model_dump_json()
        connection.execute(
            "INSERT OR IGNORE INTO finalize_receipts"
            "(wake_id, commit_id, receipt_json, created_at) VALUES (?, ?, ?, ?)",
            (wake_id, receipt.commit_id, payload, now),
        )

    return finalize
