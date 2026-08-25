"""Atomic SQLite persistence for durable world cognition."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .contracts import (
    AliasAction,
    AliasOperation,
    AssertionInput,
    CognitiveDelta,
    CommitReceipt,
    EpistemicRole,
    InquiryInput,
    ObjectInput,
    ObservationDepth,
    ObservationInput,
)
from .graph_contract import normalize_identity_alias
from .materials import MaterialPart
from .schema import initialize

# Depth rungs ordered shallow-to-deep; the only supported depth *change*
# after a row exists is an upgrade (SEEN -> CONTENT when a full body is
# read), so comparisons here are monotone.
DEPTH_LEVELS = {
    ObservationDepth.SEEN: 0,
    ObservationDepth.CONTENT: 1,
    ObservationDepth.DISCUSSION: 2,
    ObservationDepth.MEDIA: 3,
}

_CommitFinalizer = Callable[[sqlite3.Connection, CommitReceipt], None]


def observation_id(adapter_id: str, source_ref: str) -> str:
    """Derive the one durable observation id for a source across discover and hydrate.

    The id is a pure function of the adapter and the source reference (audit
    Audit3-6 / task T13): a discovery card and a hydration row for the same
    source must collide on one id so the store holds a single row per source
    and already_opened can short-circuit to it. Modality, location, and
    excerpt never participate in the identity — they describe the content of
    a read, not what was read — so they travel in the observation metadata
    instead. The digest covers the source reference alone and matches the
    bilibili adapter's internal ``_stable_id("observation", source_ref)`` so
    both layers derive the same suffix.

    Args:
        adapter_id: The channel adapter that owns the source.
        source_ref: The adapter's canonical source reference (e.g. a bvid).

    Returns:
        A stable ``f"{adapter_id}:observation-<hash>"`` id.

    """
    digest = hashlib.sha256(source_ref.encode("utf-8")).hexdigest()[:32]
    return f"{adapter_id}:observation-{digest}"


def normalize_alias(alias: str) -> str:
    """LEGACY-ONLY: fold one alias for the read-only ``object_aliases`` table.

    Legacy whitespace-collapsing normalization (casefolded, no whitespace) for
    the old single-column ``object_aliases`` table. The identity alias index
    (``identity_aliases``) uses ``normalize_identity_alias`` instead, which
    collapses whitespace runs to a single space and preserves internal spaces;
    the two rules intentionally differ and must never be swapped. New identity
    alias writes never call this function.

    Whitespace variants of an alias (``"2026 EWC 英雄联盟项目"`` vs
    ``"2026 EWC英雄联盟项目"``) must identify the same Object, so every
    legacy alias comparison and persistence site uses this one function
    instead of its own ``strip().casefold()`` (audit F3/Audit4-F3,
    Audit5-F2). Only whitespace is folded — no punctuation, pinyin, or
    full/half-width transformations.

    Args:
        alias: The raw alias text.

    Returns:
        The casefolded alias with every whitespace run removed. The result
        may be empty for an all-whitespace alias; callers must skip or
        reject empty identities rather than persisting them.

    """
    return "".join(alias.strip().casefold().split())


def alias_lookup_forms(alias: str) -> list[str]:
    """LEGACY-ONLY: lookup forms for the read-only ``object_aliases`` table.

    Serves legacy preflight/collision reads over ``object_aliases`` and is
    never used for ``identity_aliases`` lookups or writes; the identity alias
    index stores and queries a single ``normalize_identity_alias`` form.

    New writes persist only the whitespace-collapsed form (see
    ``normalize_alias``), but legacy rows written before whitespace folding
    still hold the ``strip().casefold()`` form with internal whitespace
    preserved (e.g. ``"t1 stars 男团出道"``). Every legacy ``normalized_alias``
    lookup must match both forms so a proposal hits legacy data instead of
    minting a twin object (T7 dual-form query); writes still persist only
    the collapsed form. An alias with no internal whitespace collapses the
    two forms into one value.

    Args:
        alias: The raw alias text.

    Returns:
        The deduplicated lookup values: the collapsed form first, then the
        legacy space-preserving form when it differs. An all-whitespace
        alias yields ``[""]``; callers skip empty identities.

    """
    return list(dict.fromkeys([normalize_alias(alias), alias.strip().casefold()]))


def _identity_alias_form(alias: str) -> str | None:
    """Return the identity-normalized form of one raw alias, or ``None``.

    Degenerate aliases (all-whitespace) never persist and must not
    participate in collision or unchanged comparisons; callers skip ``None``.
    """
    try:
        return normalize_identity_alias(alias)
    except ValueError:
        return None


class ObjectAliasCollision(ValueError):
    """An exact normalized alias already identifies another Object."""

    def __init__(self, normalized_alias: str, existing_object_id: str) -> None:
        super().__init__(f"object alias already belongs to {existing_object_id}: {normalized_alias}")
        self.normalized_alias = normalized_alias
        self.existing_object_id = existing_object_id


class CommitReplayConflict(ValueError):
    """One durable commit id was replayed with unverifiable or different cognition."""


class AmbiguousLegacyTouchState(ValueError):
    """A pre-ledger receipt cannot prove which inquiry touches completed."""


class WorldStore:
    """The sole write authority for durable world cognition."""

    def __init__(self, path: str | Path, *, initialize_schema: bool = True) -> None:
        """Open a world database, optionally initializing or migrating its schema."""
        self.path = Path(path)
        if initialize_schema:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                initialize(connection)
        elif not self.path.is_file():
            raise FileNotFoundError(f"world database does not exist: {self.path}")
        self._commit_finalizer: ContextVar[tuple[str, _CommitFinalizer] | None] = ContextVar(
            f"world_store_commit_finalizer_{id(self)}",
            default=None,
        )

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a bounded read-only connection for world recall."""
        connection = self._connect()
        connection.execute("PRAGMA query_only = ON")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def write_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a bounded write-capable connection that commits on clean exit."""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def memory_commit(self, delta: CognitiveDelta, commit_id: str) -> CommitReceipt:
        """Validate and atomically persist *delta* under an idempotent commit id."""
        if not commit_id.strip():
            raise ValueError("commit id must be non-empty")
        pending_finalizer = self._commit_finalizer.get()
        finalize = (
            pending_finalizer[1]
            if pending_finalizer is not None and pending_finalizer[0] == commit_id
            else None
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = connection.execute(
                "SELECT r.receipt_json, a.delta_json "
                "FROM commit_receipts AS r "
                "LEFT JOIN world_audit AS a ON a.commit_id = r.commit_id "
                "WHERE r.commit_id = ?",
                (commit_id,),
            ).fetchone()
            if replay is not None:
                self._assert_replay_delta_matches(replay["delta_json"], delta, commit_id)
                receipt = CommitReceipt.model_validate_json(replay["receipt_json"]).model_copy(
                    update={"replayed": True}
                )
                if finalize is None:
                    connection.rollback()
                else:
                    finalize(connection, receipt)
                    connection.commit()
                return receipt
            self._validate_delta(connection, delta)
            receipt = self._write_delta(connection, delta, commit_id)
            if finalize is not None:
                finalize(connection, receipt)
            connection.commit()
            return receipt
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _assert_replay_delta_matches(
        stored_delta_json: str | None,
        delta: CognitiveDelta,
        commit_id: str,
    ) -> None:
        """Fail closed when one idempotency key is reused for other cognition."""
        if stored_delta_json is None:
            raise CommitReplayConflict(f"commit id has no world audit delta: {commit_id}")
        try:
            stored = CognitiveDelta.model_validate_json(stored_delta_json)
        except (TypeError, ValueError) as error:
            raise CommitReplayConflict(
                f"commit id has an invalid world audit delta: {commit_id}"
            ) from error
        if stored != delta:
            raise CommitReplayConflict(
                f"commit id already has different cognitive delta: {commit_id}"
            )

    def finalized_memory_commit(
        self,
        delta: CognitiveDelta,
        commit_id: str,
        finalize: _CommitFinalizer,
    ) -> CommitReceipt:
        """Commit a delta and caller finalizer within one SQLite transaction.

        The scoped finalizer is keyed by ``commit_id`` and isolated with a
        context variable.  Calling the ordinary two-argument ``memory_commit``
        keeps compatibility with existing wrappers while unrelated nested or
        concurrent commits cannot consume this finalizer.
        """
        if self._commit_finalizer.get() is not None:
            raise RuntimeError("a memory commit finalizer is already active")
        token = self._commit_finalizer.set((commit_id, finalize))
        try:
            return self.memory_commit(delta, commit_id)
        finally:
            self._commit_finalizer.reset(token)

    def committed_delta(self, commit_id: str) -> tuple[CommitReceipt, CognitiveDelta] | None:
        """Return the receipt and normalized audited delta for an existing commit."""
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT r.receipt_json, a.delta_json "
                "FROM commit_receipts AS r "
                "LEFT JOIN world_audit AS a ON a.commit_id = r.commit_id "
                "WHERE r.commit_id = ?",
                (commit_id,),
            ).fetchone()
        if row is None:
            return None
        if row["delta_json"] is None:
            raise CommitReplayConflict(f"commit id has no world audit delta: {commit_id}")
        try:
            receipt = CommitReceipt.model_validate_json(row["receipt_json"])
            delta = CognitiveDelta.model_validate_json(row["delta_json"])
        except (TypeError, ValueError) as error:
            raise CommitReplayConflict(
                f"commit id has invalid durable audit data: {commit_id}"
            ) from error
        return receipt, delta

    def demote_stale_inquiries(self, now: datetime, max_age_days: int = 7) -> int:
        """Demote open inquiries untouched for max_age_days to 'dormant'."""
        cutoff = _iso(now - timedelta(days=max_age_days))
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE inquiries SET status = 'dormant' WHERE status = 'open'"
                " AND kind != 'stateful' AND last_attempted_at IS NOT NULL"
                " AND last_attempted_at < ?",
                (cutoff,),
            )
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def close_inquiry(self, inquiry_id: str, reason: str) -> None:
        """Explicitly close an open or dormant inquiry (open→dormant→closed).

        Closed inquiries drop out of every live query: the E2 similar-query
        dedup, the answerable rule, and the resolution gate all match
        ``status IN ('open', 'dormant')``, so a closed row is neither
        offered back to the model nor resolvable. Closing bumps
        ``version`` (the version protocol guards concurrent stale reads);
        the durable ``closed_reason`` is stored for the audit trail. A
        closed inquiry can be reopened with :meth:`reopen_inquiry`.

        Args:
            inquiry_id: The inquiry row to close.
            reason: A non-empty, durable reason for the closure.

        Raises:
            ValueError: The inquiry is missing, or is not open or dormant.

        """
        if not reason.strip():
            raise ValueError("inquiry close reason must be non-empty")
        with self.write_connection() as connection:
            row = connection.execute(
                "SELECT status, version FROM inquiries WHERE id = ?", (inquiry_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"inquiry does not exist: {inquiry_id}")
            if str(row["status"]) not in {"open", "dormant"}:
                raise ValueError(f"inquiry is not open or dormant: {inquiry_id}")
            connection.execute(
                "UPDATE inquiries SET status = 'closed', closed_reason = ?,"
                " version = version + 1 WHERE id = ?",
                (reason, inquiry_id),
            )

    def reopen_inquiry(self, inquiry_id: str) -> None:
        """Reopen a closed inquiry to 'open' (closed→open), restoring resolvability.

        Reopening bumps ``version`` once more and clears the durable
        ``closed_reason``; the inquiry becomes answerable and E2-visible
        again. Note the open-inquiry dedup index is still enforced: a
        reopen that would collide with another open ``(subject_id, prompt)``
        row fails with an IntegrityError rather than silently doubling the
        live inquiry.

        Args:
            inquiry_id: The closed inquiry row to reopen.

        Raises:
            ValueError: The inquiry is missing, or is not closed.

        """
        with self.write_connection() as connection:
            row = connection.execute(
                "SELECT status FROM inquiries WHERE id = ?", (inquiry_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"inquiry does not exist: {inquiry_id}")
            if str(row["status"]) != "closed":
                raise ValueError(f"inquiry is not closed: {inquiry_id}")
            connection.execute(
                "UPDATE inquiries SET status = 'open', closed_reason = NULL,"
                " version = version + 1 WHERE id = ?",
                (inquiry_id,),
            )

    def write_observation_body(
        self,
        observation_id: str,
        content_type: str,
        body: str,
        size_bytes: int | None = None,
    ) -> None:
        """Upsert one observation body with its serialized UTF-8 byte size.

        ``size_bytes`` remains accepted for legacy callers, but the persisted
        value is always derived from the exact serialized payload.
        """
        del size_bytes
        serialized_size = len(body.encode("utf-8"))
        with self.write_connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO observation_bodies"
                "(observation_id, content_type, body_json, size_bytes) VALUES (?, ?, ?, ?)",
                (observation_id, content_type, body, serialized_size),
            )

    def replace_observation_body_with_parts(
        self,
        observation_id: str,
        parts: list[ObservationInput],
        content_type: str,
        body: str,
        *,
        commit_id: str,
    ) -> bool:
        """Atomically persist body-part anchors, body payload, and CONTENT depth.

        This is deliberately a narrow material operation: it reuses the
        existing observation and receipt tables rather than introducing a
        migration, while ensuring a refresh cannot leave new part anchors
        paired with an old body (or vice versa). A replayed commit id is an
        idempotent replay of the complete material transaction: it never
        overwrites the body and returns ``False``.
        """
        if not commit_id.strip():
            raise ValueError("commit id must be non-empty")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            primary = connection.execute(
                "SELECT 1 FROM observations WHERE id = ?", (observation_id,)
            ).fetchone()
            if primary is None:
                raise ValueError("observation body requires an existing observation")
            replay = connection.execute(
                "SELECT 1 FROM commit_receipts WHERE commit_id = ?", (commit_id,)
            ).fetchone()
            if replay is not None:
                existing_body = connection.execute(
                    "SELECT 1 FROM observation_bodies WHERE observation_id = ?",
                    (observation_id,),
                ).fetchone()
                if existing_body is None:
                    raise ValueError("material replay has no committed body")
                connection.rollback()
                return False
            delta = CognitiveDelta(observations=parts)
            self._validate_delta(connection, delta)
            self._write_delta(connection, delta, commit_id)
            serialized_size = len(body.encode("utf-8"))
            connection.execute(
                "INSERT OR REPLACE INTO observation_bodies"
                "(observation_id, content_type, body_json, size_bytes) VALUES (?, ?, ?, ?)",
                (observation_id, content_type, body, serialized_size),
            )
            connection.execute(
                "UPDATE observations SET depth = CASE WHEN depth = ? THEN ? ELSE depth END WHERE id = ?",
                (ObservationDepth.SEEN.value, ObservationDepth.CONTENT.value, observation_id),
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def read_observation_body(self, observation_id: str) -> dict[str, Any] | None:
        """Return the stored body row or None."""
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT content_type, body_json, size_bytes FROM observation_bodies WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def body_parts_are_verifiable(
        self,
        primary_observation_id: str,
        parts: list[MaterialPart],
    ) -> bool:
        """Return whether body parts are durable, same-source CONTENT anchors."""
        if not parts:
            return False
        ids = list(dict.fromkeys(part.observation_id for part in parts))
        placeholders = ", ".join("?" for _ in ids)
        with self.read_connection() as connection:
            primary = connection.execute(
                "SELECT source_uri, source_kind FROM observations WHERE id = ?",
                (primary_observation_id,),
            ).fetchone()
            if primary is None:
                return False
            rows = connection.execute(
                f"SELECT id, source_uri, source_kind, depth, metadata_json "
                f"FROM observations WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        if len(rows) != len(ids):
            return False
        rows_by_id = {str(row["id"]): row for row in rows}
        for part in parts:
            row = rows_by_id[part.observation_id]
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except (TypeError, ValueError):
                return False
            if not isinstance(metadata, dict):
                return False
            if (
                str(row["source_uri"]) != str(primary["source_uri"])
                or str(row["source_kind"]) != str(primary["source_kind"])
                or DEPTH_LEVELS[ObservationDepth(str(row["depth"]))] < DEPTH_LEVELS[ObservationDepth.CONTENT]
                or metadata.get("body_part") is not True
                or metadata.get("material_kind") != part.kind
                or metadata.get("material_reliability") != part.reliability
                or metadata.get("location") != part.location
                or metadata.get("acquisition_method") != part.acquisition_method
            ):
                return False
        return True

    def upgrade_observation_depth(self, observation_id: str) -> None:
        """Raise one observation's stored depth to CONTENT (spec D1).

        Reading a full-text body means the observation was deeply inspected,
        so its evidence depth rises from SEEN to CONTENT and the observation
        may back formal assertions (committer's seen_support rule reads this
        stored depth). The upgrade is monotone: an observation already at
        CONTENT/DISCUSSION/MEDIA is left untouched and a missing row is a
        silent no-op, keeping the call idempotent for the store-served full
        re-read path. The change goes through the write connection so it is
        committed like every other store write.

        Args:
            observation_id: The observation row to upgrade.

        """
        with self.write_connection() as connection:
            row = connection.execute(
                "SELECT depth FROM observations WHERE id = ?", (observation_id,)
            ).fetchone()
            if row is None:
                return
            if DEPTH_LEVELS[ObservationDepth(str(row["depth"]))] < DEPTH_LEVELS[ObservationDepth.CONTENT]:
                connection.execute(
                    "UPDATE observations SET depth = ? WHERE id = ?",
                    (ObservationDepth.CONTENT.value, observation_id),
                )

    def touch_inquiry(
        self,
        inquiry_id: str,
        now: datetime,
        *,
        run_commit_id: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> bool:
        """Bump one inquiry, optionally once per durable run inside a transaction."""
        if connection is not None:
            return self._touch_inquiry(
                connection,
                inquiry_id,
                now,
                run_commit_id=run_commit_id,
            )
        with self.write_connection() as owned_connection:
            return self._touch_inquiry(
                owned_connection,
                inquiry_id,
                now,
                run_commit_id=run_commit_id,
            )

    def record_inquiry_touch(
        self,
        inquiry_id: str,
        now: datetime,
        *,
        run_commit_id: str,
        connection: sqlite3.Connection,
    ) -> bool:
        """Backfill a proven-complete touch without incrementing its inquiry."""
        if not run_commit_id.strip():
            raise ValueError("run commit id must be non-empty")
        if connection.execute(
            "SELECT 1 FROM inquiries WHERE id = ?", (inquiry_id,)
        ).fetchone() is None:
            raise ValueError(f"inquiry does not exist: {inquiry_id}")
        inserted = connection.execute(
            "INSERT OR IGNORE INTO inquiry_attempt_touches"
            "(run_commit_id, inquiry_id, touched_at) VALUES (?, ?, ?)",
            (run_commit_id, inquiry_id, _iso(now)),
        )
        return inserted.rowcount == 1

    def assert_legacy_touches_not_applied(
        self,
        inquiry_ids: set[str],
        committed_at: datetime,
        *,
        connection: sqlite3.Connection,
    ) -> None:
        """Fail closed unless a receipt-only commit provably touched no target.

        The pre-atomic implementation touched inquiries only after its durable
        receipt was written.  Therefore a target timestamp before the receipt
        proves that target was not touched by the interrupted finalizer.  An
        equal or later timestamp is ambiguous (partial finalizer or unrelated
        activity), so recovery must not risk a duplicate increment.
        """
        if not inquiry_ids:
            return
        ordered = sorted(inquiry_ids)
        marks = ", ".join("?" for _ in ordered)
        rows = connection.execute(
            f"SELECT id, last_attempted_at FROM inquiries WHERE id IN ({marks})",
            ordered,
        ).fetchall()
        found = {str(row["id"]): row["last_attempted_at"] for row in rows}
        missing = [inquiry_id for inquiry_id in ordered if inquiry_id not in found]
        if missing:
            raise ValueError(f"inquiries do not exist: {', '.join(missing)}")
        ambiguous = [
            inquiry_id
            for inquiry_id, last_attempted_at in found.items()
            if last_attempted_at is not None
            and datetime.fromisoformat(str(last_attempted_at)).astimezone(UTC) >= committed_at
        ]
        if ambiguous:
            raise AmbiguousLegacyTouchState(
                "cannot safely recover legacy inquiry touches; prior touch state is ambiguous: "
                + ", ".join(sorted(ambiguous))
            )

    @staticmethod
    def _touch_inquiry(
        connection: sqlite3.Connection,
        inquiry_id: str,
        now: datetime,
        *,
        run_commit_id: str | None = None,
    ) -> bool:
        if run_commit_id is not None:
            if not run_commit_id.strip():
                raise ValueError("run commit id must be non-empty")
            claimed = connection.execute(
                "INSERT OR IGNORE INTO inquiry_attempt_touches"
                "(run_commit_id, inquiry_id, touched_at) VALUES (?, ?, ?)",
                (run_commit_id, inquiry_id, _iso(now)),
            )
            if claimed.rowcount != 1:
                return False
        updated = connection.execute(
            "UPDATE inquiries SET attempt_count = attempt_count + 1, last_attempted_at = ? WHERE id = ?",
            (_iso(now), inquiry_id),
        )
        if updated.rowcount != 1:
            raise ValueError(f"inquiry does not exist: {inquiry_id}")
        return True

    def _validate_delta(self, connection: sqlite3.Connection, delta: CognitiveDelta) -> None:
        self._unique_ids(delta.objects, "object")
        self._unique_ids(delta.observations, "observation")
        self._unique_ids(delta.assertions, "assertion")
        self._unique_ids(delta.inquiries, "inquiry")
        self._validate_object_aliases(connection, delta.objects)
        object_ids = {item.id for item in delta.objects}
        inquiry_ids = {item.id for item in delta.inquiries}
        for operation in delta.alias_operations:
            self._require_object(connection, operation.object_id, object_ids)
        stale_object_ids: list[str] = []
        for item in delta.objects:
            row = connection.execute("SELECT * FROM objects WHERE id = ?", (item.id,)).fetchone()
            if row is None and item.expected_version is not None:
                stale_object_ids.append(item.id)
            elif (
                row is not None
                and item.expected_version is None
                and not self._object_is_unchanged(connection, row, item)
            ):
                raise ValueError(f"object mutation requires expected version: {item.id}")
            elif (
                row is not None
                and item.expected_version is not None
                and (row["version"] != item.expected_version)
            ):
                stale_object_ids.append(item.id)
        observation_depths = self._effective_observation_depths(connection, delta)
        assertion_ids = self._prospective_assertion_ids(connection, delta)
        missing_evidence_ids: list[str] = []
        missing_superseded_ids: list[str] = []
        for assertion in delta.assertions:
            self._require_object(connection, assertion.subject_id, object_ids)
            if assertion.object_id is not None:
                self._require_object(connection, assertion.object_id, object_ids)
            for evidence in assertion.evidence:
                depth = observation_depths.get(evidence.observation_id)
                if depth is None:
                    row = connection.execute(
                        "SELECT depth FROM observations WHERE id = ?", (evidence.observation_id,)
                    ).fetchone()
                    if row is None:
                        missing_evidence_ids.append(evidence.observation_id)
                        continue
            if assertion.supersedes_id is not None and not self._assertion_exists(
                connection, assertion.supersedes_id, assertion_ids
            ):
                missing_superseded_ids.append(assertion.supersedes_id)
        stale_inquiry_ids: list[str] = []
        unresolvable_inquiries: list[tuple[str, str, int, int]] = []
        for inquiry in delta.inquiries:
            self._require_object(connection, inquiry.subject_id, object_ids)
            if inquiry.deepens_id is not None and (
                inquiry.deepens_id == inquiry.id
                or inquiry.deepens_id not in inquiry_ids
                and connection.execute(
                    "SELECT 1 FROM inquiries WHERE id = ?", (inquiry.deepens_id,)
                ).fetchone()
                is None
            ):
                raise ValueError(
                    "deepens_id references a nonexistent inquiry: " + inquiry.deepens_id
                )
            row = connection.execute("SELECT version FROM inquiries WHERE id = ?", (inquiry.id,)).fetchone()
            if inquiry.expected_version is not None and (
                row is None or row["version"] != inquiry.expected_version
            ):
                stale_inquiry_ids.append(inquiry.id)
        missing_link_ids: list[str] = []
        for link in delta.observation_links:
            if link.target_kind == "object":
                self._require_object(connection, link.target_id, object_ids)
            elif (
                link.target_id not in inquiry_ids
                and connection.execute("SELECT 1 FROM inquiries WHERE id = ?", (link.target_id,)).fetchone()
                is None
            ):
                # Candidate link targets are defensive, not contractual: a
                # link whose inquiry target exists nowhere is dropped by
                # _write_delta instead of failing the commit (recovery can
                # hand over such links; the delta must not die with them).
                continue
            if (
                link.observation_id not in observation_depths
                and connection.execute(
                    "SELECT 1 FROM observations WHERE id = ?", (link.observation_id,)
                ).fetchone()
                is None
            ):
                missing_link_ids.append(link.observation_id)
        declared_answer_ids = {
            str(assertion.answers_inquiry_id)
            for assertion in delta.assertions
            if assertion.answers_inquiry_id is not None
        }
        known_inquiry_ids = inquiry_ids | {
            str(row["id"]) for row in connection.execute("SELECT id FROM inquiries").fetchall()
        }
        for declared in sorted(declared_answer_ids):
            if declared not in known_inquiry_ids:
                raise ValueError(f"answers_inquiry_id references a nonexistent inquiry: {declared}")
        for resolution in delta.resolve_inquiries:
            row = connection.execute(
                "SELECT subject_id, status, version FROM inquiries WHERE id = ?", (resolution.id,)
            ).fetchone()
            if row is None:
                # Same-delta create+resolve (Wave B): the inquiry row does not
                # exist yet because this delta inserts it later in the same
                # commit. The INSERT writes version 1 with status 'open', so
                # the staged expected_version must name exactly that — anything
                # else is a version guess against a row that does not exist
                # and fails closed like any other stale resolution.
                if resolution.id not in inquiry_ids:
                    stale_inquiry_ids.append(resolution.id)
                    continue
                if resolution.expected_version != 1:
                    stale_inquiry_ids.append(resolution.id)
                    continue
                status = "open"
            else:
                if row["version"] != resolution.expected_version:
                    stale_inquiry_ids.append(resolution.id)
                    continue
                # Dormant means deprioritized, not abandoned. A version-matched
                # inquiry with a relevant substantive answer may close directly;
                # forcing an artificial dormant -> open transition would make a
                # valid answer depend on a second state-changing proposal.
                status = str(row["status"])
            if status not in {"open", "dormant"}:
                unresolvable_inquiries.append(
                    (
                        resolution.id,
                        str(row["status"]),
                        int(row["version"]),
                        resolution.expected_version,
                    )
                )
                continue
            # F1/F2b (spec §5): resolution validity is scoped per resolution —
            # at least one non-uncertainty assertion must declare
            # answers_inquiry_id == this inquiry. The single-valued link
            # naturally forbids one assertion answering two inquiries, so the
            # old cross-assertion "all same-subject answers must match" check
            # is gone: two resolutions on one subject are both valid (d2).
            #
            # Task 2 (closeout): the check reads the metadata that WILL
            # persist, mirroring the committer's effective-owner gate —
            # signature-collision dedup keeps the stored row (else the first
            # same-signature delta item) and discards later answers_inquiry_id,
            # so a discarded proposal link never resolves an inquiry and a
            # stored answer link keeps resolving its inquiry.
            effective_links: dict[str, str | None] = {}
            for assertion in delta.assertions:
                signature = _assertion_signature(assertion)
                if signature in effective_links:
                    continue
                stored = self._assertion_by_signature(connection, signature)
                effective_links[signature] = (
                    stored["answers_inquiry_id"]
                    if stored is not None
                    else assertion.answers_inquiry_id
                )
            answering_assertions = [
                assertion
                for assertion in delta.assertions
                if effective_links[_assertion_signature(assertion)] == resolution.id
                and assertion.epistemic_role is not EpistemicRole.UNCERTAINTY
            ]
            if not answering_assertions:
                raise ValueError(
                    "inquiry resolution requires an answering assertion declaring "
                    f"answers_inquiry_id={resolution.id}"
                )
        problems: list[str] = []
        if missing_evidence_ids:
            problems.append(
                "assertion evidence observation does not exist: "
                + ", ".join(sorted(set(missing_evidence_ids)))
            )
        if missing_link_ids:
            problems.append(
                "observation link observation does not exist: " + ", ".join(sorted(set(missing_link_ids)))
            )
        if missing_superseded_ids:
            problems.append(
                "supersedes assertion does not exist: " + ", ".join(sorted(set(missing_superseded_ids)))
            )
        if stale_object_ids:
            problems.append("stale object version: " + ", ".join(sorted(set(stale_object_ids))))
        if stale_inquiry_ids:
            problems.append("stale inquiry version: " + ", ".join(sorted(set(stale_inquiry_ids))))
        if unresolvable_inquiries:
            problems.append(
                "inquiry resolution status mismatch: "
                + ", ".join(
                    f"{identifier} (current_status: {status}, current_version: {current}, "
                    f"expected_version: {expected})"
                    for identifier, status, current, expected in unresolvable_inquiries
                )
            )
        if problems:
            raise ValueError(_truncate("; ".join(problems)))

    @staticmethod
    def _validate_object_aliases(connection: sqlite3.Connection, objects: list[ObjectInput]) -> None:
        """Enforce active identity-alias uniqueness across one delta (Task 2.1).

        Only ``identity_aliases`` rows with status 'active' block: an item
        declaring an identity alias whose active owner is a different object
        (stored, or another object in the same delta) is a strong conflict
        (matrix row 1 backstop). Legacy ``object_aliases`` rows are read-only
        history and never block — the compiler surfaces them as
        ``ambiguous_name_candidates`` warnings instead. The canonical name is
        NOT an implicit identity alias, so two objects may share a canonical
        name (the identity alias index is untouched by it).
        """
        proposed_owners: dict[str, str] = {}
        for item in objects:
            for alias in _distinct(item.aliases):
                normalized = _identity_alias_form(alias)
                if not normalized:
                    # a degenerate alias (all-whitespace) never persists, so
                    # it cannot collide
                    continue
                row = connection.execute(
                    "SELECT object_id FROM identity_aliases"
                    " WHERE normalized_alias = ? AND status = 'active' AND object_id != ?"
                    " ORDER BY object_id LIMIT 1",
                    (normalized, item.id),
                ).fetchone()
                if row is not None:
                    raise ObjectAliasCollision(normalized, str(row["object_id"]))
                proposed = proposed_owners.get(normalized)
                if proposed is not None and proposed != item.id:
                    raise ObjectAliasCollision(normalized, proposed)
                proposed_owners[normalized] = item.id

    def _effective_observation_depths(
        self, connection: sqlite3.Connection, delta: CognitiveDelta
    ) -> dict[str, ObservationDepth]:
        depths: dict[str, ObservationDepth] = {}
        for item in delta.observations:
            row = connection.execute(
                "SELECT source_uri, source_kind, depth FROM observations WHERE id = ?", (item.id,)
            ).fetchone()
            if row is not None:
                stored_depth = ObservationDepth(str(row["depth"]))
                if row["source_uri"] != item.source_uri or row["source_kind"] != item.source_kind:
                    raise ValueError("observation id identifies a different observation")
                # Depth is mutable in one direction only. A deeper proposal
                # used to be an identity conflict, but the unified id space
                # (observation_id) makes a hydrate of a discovery card the
                # same row at a deeper depth — that is the upgrade path, not
                # a relabel. Re-scans may also re-see a deep row at SEEN and
                # must never reject the batch or pull the upgrade back, so
                # the effective depth is the deeper of the two; evidence
                # checks read the stored depth whenever it is not shallower.
                depths[item.id] = (
                    stored_depth if DEPTH_LEVELS[stored_depth] >= DEPTH_LEVELS[item.depth] else item.depth
                )
            else:
                depths[item.id] = item.depth
        return depths

    def _prospective_assertion_ids(self, connection: sqlite3.Connection, delta: CognitiveDelta) -> list[str]:
        assertion_ids: list[str] = []
        for item in delta.assertions:
            signature = _assertion_signature(item)
            by_id = connection.execute("SELECT signature FROM assertions WHERE id = ?", (item.id,)).fetchone()
            if by_id is not None and by_id["signature"] != signature:
                raise ValueError("assertion id conflicts with a different signature")
            by_signature = connection.execute(
                "SELECT id FROM assertions WHERE signature = ?", (signature,)
            ).fetchone()
            assertion_ids.append(str(by_signature["id"]) if by_signature is not None else item.id)
        return assertion_ids

    @staticmethod
    def _assertion_exists(connection: sqlite3.Connection, assertion_id: str, proposed_ids: list[str]) -> bool:
        if assertion_id in proposed_ids:
            return True
        row = connection.execute("SELECT 1 FROM assertions WHERE id = ?", (assertion_id,)).fetchone()
        return row is not None

    def _object_is_unchanged(
        self, connection: sqlite3.Connection, row: sqlite3.Row, item: ObjectInput
    ) -> bool:
        # Task 2.1: the alias dimension compares ACTIVE identity alias forms
        # only. The canonical name is a row field compared separately above
        # and is never an implicit identity alias; legacy object_aliases rows
        # are read-only history that no longer participates in identity.
        stored_identity_aliases = {
            str(alias["normalized_alias"])
            for alias in connection.execute(
                "SELECT normalized_alias FROM identity_aliases"
                " WHERE object_id = ? AND status = 'active'",
                (item.id,),
            ).fetchall()
        }
        return (
            row["kind"] == item.kind.value
            and row["type_key"] == item.type_key
            and row["canonical_name"] == item.canonical_name
            and row["domain_hints_json"] == _json(item.domain_hints)
            and row["provisional"] == int(item.provisional)
            and row["event_time_start"] == _iso(item.event_time_start)
            and row["event_time_end"] == _iso(item.event_time_end)
            and row["event_time_precision"] == item.event_time_precision
            and stored_identity_aliases
            == {
                normalized
                for alias in item.aliases
                if (normalized := _identity_alias_form(alias))
            }
        )

    def _write_delta(
        self, connection: sqlite3.Connection, delta: CognitiveDelta, commit_id: str
    ) -> CommitReceipt:
        committed_at = datetime.now(UTC)
        for item in delta.objects:
            self._write_object(connection, item, commit_id)
        # Identity alias corrections run in the same BEGIN IMMEDIATE
        # transaction as the object/assertion writes: the world audit delta
        # and the current identity_aliases index can never diverge.
        for operation in delta.alias_operations:
            self._apply_alias_operation(connection, operation, commit_id)
        for item in delta.observations:
            stored = connection.execute(
                "SELECT depth, content_ref, metadata_json FROM observations WHERE id = ?",
                (item.id,),
            ).fetchone()
            values = (
                item.id,
                item.source_uri,
                item.source_kind,
                item.title,
                item.excerpt,
                item.content_ref,
                item.depth.value,
                _iso(item.source_published_at),
                _iso(item.observed_at),
                _json(item.metadata),
            )
            if stored is None:
                connection.execute(
                    """INSERT INTO observations
                    (id, source_uri, source_kind, title, excerpt, content_ref, depth,
                     source_published_at, observed_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values,
                )
            else:
                # Same-id re-writes update content instead of being silently
                # ignored (audit Audit3-6 / task T13): discover re-sees and
                # hydrate upgrades the one row per source. Title, excerpt,
                # and metadata refresh; depth only ever rises, so a re-scan
                # of a fully read observation never pulls the upgrade back
                # (monotone guard f6f89b7). content_ref is only ever set,
                # never cleared by an empty proposal. observed_at is pinned
                # to the first observation time and never refreshed here
                # (spec §7): a re-read of old content must not masquerade as
                # a new observation, so the UPDATE leaves the stored
                # observed_at untouched. The audit delta still records the
                # proposed timestamp; the overview reads the stored one.
                stored_depth = ObservationDepth(str(stored["depth"]))
                depth = item.depth if DEPTH_LEVELS[item.depth] > DEPTH_LEVELS[stored_depth] else stored_depth
                content_ref = item.content_ref or str(stored["content_ref"] or "")
                metadata = _merge_observed_metadata(
                    json.loads(stored["metadata_json"] or "{}"), item.metadata
                )
                connection.execute(
                    """UPDATE observations SET source_uri = ?, source_kind = ?,
                    title = ?, excerpt = ?, content_ref = ?, depth = ?,
                    source_published_at = ?, metadata_json = ?
                    WHERE id = ?""",
                    (
                        item.source_uri,
                        item.source_kind,
                        item.title,
                        item.excerpt,
                        content_ref,
                        depth.value,
                        _iso(item.source_published_at),
                        _json(metadata),
                        item.id,
                    ),
                )
        for item in delta.inquiries:
            self._write_inquiry(connection, item)
        assertion_ids = [self._write_assertion(connection, item, committed_at) for item in delta.assertions]
        # Candidate link targets are defensive, mirroring the observation-side
        # existence check: a link whose inquiry target is not actually present
        # in the store — never declared, or declared but silently swallowed by
        # the open-inquiry dedup index — is skipped instead of firing the
        # inquiry_observations foreign key. Existence is read after the delta's
        # own inquiry inserts, batched into one same-transaction query, never
        # re-queried per link.
        known_inquiry_ids: set[str] = set()
        if any(link.target_kind != "object" for link in delta.observation_links):
            known_inquiry_ids = {
                str(row["id"]) for row in connection.execute("SELECT id FROM inquiries").fetchall()
            }
        for link in delta.observation_links:
            if link.target_kind != "object" and link.target_id not in known_inquiry_ids:
                continue
            table, column = (
                ("object_observations", "object_id")
                if link.target_kind == "object"
                else ("inquiry_observations", "inquiry_id")
            )
            connection.execute(
                f"INSERT OR IGNORE INTO {table}"
                f"({column}, observation_id, role, linked_at) VALUES (?, ?, ?, ?)",
                (link.target_id, link.observation_id, link.role, _iso(committed_at)),
            )
        for resolution in delta.resolve_inquiries:
            connection.execute(
                "UPDATE inquiries SET status = 'resolved', version = version + 1,"
                " resolved_at = ? WHERE id = ?",
                (_iso(committed_at), resolution.id),
            )
        receipt = CommitReceipt(
            commit_id=commit_id,
            committed_at=committed_at,
            object_ids=[item.id for item in delta.objects],
            observation_ids=[item.id for item in delta.observations],
            assertion_ids=_distinct(assertion_ids),
            inquiry_ids=[item.id for item in delta.inquiries],
            resolved_inquiry_ids=[item.id for item in delta.resolve_inquiries],
        )
        payload = receipt.model_dump_json()
        connection.execute(
            "INSERT INTO commit_receipts(commit_id, committed_at, receipt_json) VALUES (?, ?, ?)",
            (commit_id, _iso(committed_at), payload),
        )
        connection.execute(
            "INSERT INTO world_audit(commit_id, committed_at, delta_json) VALUES (?, ?, ?)",
            (commit_id, _iso(committed_at), delta.model_dump_json()),
        )
        return receipt

    def _write_object(
        self, connection: sqlite3.Connection, item: ObjectInput, commit_id: str
    ) -> None:
        row = connection.execute("SELECT version FROM objects WHERE id = ?", (item.id,)).fetchone()
        if row is None:
            connection.execute(
                """INSERT INTO objects
                (id, kind, type_key, canonical_name, domain_hints_json, provisional,
                 event_time_start, event_time_end, event_time_precision, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    item.id,
                    item.kind.value,
                    item.type_key,
                    item.canonical_name,
                    _json(item.domain_hints),
                    int(item.provisional),
                    _iso(item.event_time_start),
                    _iso(item.event_time_end),
                    item.event_time_precision,
                ),
            )
            # Task 1.5: a new object's initial identity aliases come ONLY from
            # ObjectInput.aliases — the canonical name is never an implicit
            # identity alias. Alias forms normalize through the shared
            # identity normalizer (internal spaces preserved); degenerate
            # forms that normalize to empty are skipped. A plain INSERT makes
            # a cross-object active-alias conflict fail the whole transaction
            # closed (partial unique index on active normalized_alias).
            self._persist_identity_aliases(connection, item, commit_id)
        elif item.expected_version is not None:
            connection.execute(
                """UPDATE objects SET kind = ?, type_key = ?, canonical_name = ?,
                domain_hints_json = ?, provisional = ?,
                event_time_start = ?, event_time_end = ?, event_time_precision = ?,
                version = version + 1 WHERE id = ?""",
                (
                    item.kind.value,
                    item.type_key,
                    item.canonical_name,
                    _json(item.domain_hints),
                    int(item.provisional),
                    _iso(item.event_time_start),
                    _iso(item.event_time_end),
                    item.event_time_precision,
                    item.id,
                ),
            )
            # Task 2.1 (Ruling A): a versioned rewrite re-asserts the current
            # alias list (stored ∪ adds, as compiled by the committer); the
            # identity index persists only the genuinely new forms (see
            # _persist_identity_aliases) so the update never loses existing
            # identity aliases and never clobbers their history columns.
            self._persist_identity_aliases(connection, item, commit_id, skip_existing=True)
        else:
            return
        # Task 2.1 (Ruling A): new writes never touch the legacy
        # object_aliases table; the FTS aliases column refreshes from the
        # identity alias index (plus any pre-existing legacy rows). The FTS
        # view is a read/search surface, so it combines both sources (F1).
        persisted_aliases = self._stored_search_aliases(connection, item.id)
        connection.execute("DELETE FROM objects_fts WHERE id = ?", (item.id,))
        connection.execute(
            "INSERT INTO objects_fts(id, canonical_name, aliases) VALUES (?, ?, ?)",
            (item.id, item.canonical_name, " ".join(persisted_aliases)),
        )

    @classmethod
    def _persist_identity_aliases(
        cls,
        connection: sqlite3.Connection,
        item: ObjectInput,
        commit_id: str,
        *,
        skip_existing: bool = False,
    ) -> None:
        """Persist one object's declared identity aliases (both write branches).

        Task 2.1 (Ruling A) removal of the legacy write loop must not cost the
        update branch its identity index: a versioned rewrite re-asserts the
        full current alias list (stored ∪ adds, as compiled by the committer).
        With ``skip_existing`` (update branch) only the genuinely new forms
        are inserted — existing rows are never re-touched, so their
        ``added_commit_id``/``removed_commit_id`` history survives and the
        add/remove/demote alias operations own the transitions. The new-object
        branch keeps a plain INSERT: a cross-object claim on an ACTIVE form
        violates the partial unique index and fails the whole transaction
        closed (a backstop after the preflight already raised
        ``ObjectAliasCollision`` for it).
        """
        existing: set[str] = set()
        if skip_existing:
            existing = {
                str(row["normalized_alias"])
                for row in connection.execute(
                    "SELECT normalized_alias FROM identity_aliases WHERE object_id = ?",
                    (item.id,),
                ).fetchall()
            }
        seen_identity_aliases: set[str] = set()
        for alias in item.aliases:
            try:
                normalized = normalize_identity_alias(alias)
            except ValueError:
                continue
            if normalized in seen_identity_aliases or normalized in existing:
                continue
            seen_identity_aliases.add(normalized)
            connection.execute(
                """INSERT INTO identity_aliases
                (object_id, raw_alias, normalized_alias, status, added_commit_id, removed_commit_id)
                VALUES (?, ?, ?, 'active', ?, NULL)""",
                (item.id, alias, normalized, commit_id),
            )

    @staticmethod
    def _apply_alias_operation(
        connection: sqlite3.Connection, operation: AliasOperation, commit_id: str
    ) -> None:
        """Apply one identity alias operation to the current index.

        ADD is a deterministic UPSERT: re-adding an alias after a remove
        restores the row to ``active`` and clears ``removed_commit_id`` while
        recording the re-add's ``added_commit_id``; the original raw_alias is
        preserved. REMOVE and DEMOTE share the same state transition — the
        action distinction lives only in the delta audit; both write
        ``status='removed'`` with ``removed_commit_id`` and only affect a
        currently ACTIVE row. New writes never touch the legacy
        ``object_aliases`` table (read-only).
        """
        if operation.action is AliasAction.ADD:
            connection.execute(
                """INSERT INTO identity_aliases
                (object_id, raw_alias, normalized_alias, status, added_commit_id, removed_commit_id)
                VALUES (?, ?, ?, 'active', ?, NULL)
                ON CONFLICT(object_id, normalized_alias) DO UPDATE SET
                    status = 'active',
                    removed_commit_id = NULL,
                    added_commit_id = excluded.added_commit_id""",
                (operation.object_id, operation.raw_alias, operation.normalized_alias, commit_id),
            )
        else:
            connection.execute(
                "UPDATE identity_aliases SET status = 'removed', removed_commit_id = ?"
                " WHERE object_id = ? AND normalized_alias = ? AND status = 'active'",
                (commit_id, operation.object_id, operation.normalized_alias),
            )

    def _write_inquiry(self, connection: sqlite3.Connection, item: InquiryInput) -> None:
        row = connection.execute("SELECT version FROM inquiries WHERE id = ?", (item.id,)).fetchone()
        if row is None:
            now = datetime.now(UTC)
            connection.execute(
                "INSERT OR IGNORE INTO inquiries"
                "(id, subject_id, prompt, rationale, kind, created_at, last_attempted_at,"
                " attempt_count, status, version, deepens_id) VALUES (?, ?, ?, ?, ?, ?, ?, 0,"
                " 'open', 1, ?)",
                (
                    item.id,
                    item.subject_id,
                    item.prompt,
                    item.rationale,
                    item.kind,
                    _iso(item.created_at or now),
                    _iso(item.last_attempted_at or item.created_at or now),
                    item.deepens_id,
                ),
            )
        elif item.expected_version is not None:
            connection.execute(
                "UPDATE inquiries SET prompt = ?, rationale = ?, kind = ?,"
                " version = version + 1 WHERE id = ?",
                (item.prompt, item.rationale, item.kind, item.id),
            )

    @staticmethod
    def _assertion_by_signature(
        connection: sqlite3.Connection, signature: str
    ) -> sqlite3.Row | None:
        """Return the stored assertion sharing one signature, if any.

        Single source of truth for the structural dedup lookup:
        :meth:`_write_assertion` reuses the found row's id and links the new
        evidence there, and the committer's dedup announcement reads the same
        row — so the announced un-replaced metadata differences always match
        what the write actually keeps.
        """
        return connection.execute(
            "SELECT id, confidence, event_time_precision, supersedes_id,"
            " supersede_reason, answers_inquiry_id FROM assertions WHERE signature = ?",
            (signature,),
        ).fetchone()

    def _write_assertion(
        self, connection: sqlite3.Connection, item: AssertionInput, committed_at: datetime
    ) -> str:
        connection.execute("INSERT OR IGNORE INTO predicates(name) VALUES (?)", (item.predicate,))
        signature = _assertion_signature(item)
        row = self._assertion_by_signature(connection, signature)
        assertion_id = str(row["id"]) if row is not None else item.id
        if row is None:
            connection.execute(
                """INSERT INTO assertions
                (id, signature, subject_id, predicate, object_id, literal_json, epistemic_role, confidence,
                 qualifiers_json,
                 event_time_start, event_time_end, event_time_precision, supersedes_id, supersede_reason,
                 answers_inquiry_id, superseded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.id,
                    signature,
                    item.subject_id,
                    item.predicate,
                    item.object_id,
                    _json(item.literal) if item.literal is not None else None,
                    item.epistemic_role.value,
                    item.confidence,
                    _json(item.qualifiers) if item.qualifiers else None,
                    _iso(item.event_time_start),
                    _iso(item.event_time_end),
                    item.event_time_precision,
                    item.supersedes_id,
                    item.supersede_reason,
                    item.answers_inquiry_id,
                    _iso(item.superseded_at),
                ),
            )
            connection.execute(
                "INSERT INTO assertions_fts(id, subject_id, predicate, object_id, literal) "
                "VALUES (?, ?, ?, ?, ?)",
                (item.id, item.subject_id, item.predicate, item.object_id or "", _json(item.literal)),
            )
        for evidence in item.evidence:
            connection.execute(
                "INSERT OR IGNORE INTO assertion_evidence"
                "(assertion_id, observation_id, role, linked_at) VALUES (?, ?, ?, ?)",
                (assertion_id, evidence.observation_id, evidence.role, _iso(committed_at)),
            )
        return assertion_id

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _unique_ids(items: list[object], label: str) -> None:
        ids = [str(item.id) for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {label} id in delta")

    @staticmethod
    def _require_object(connection: sqlite3.Connection, object_id: str, proposed_ids: set[str]) -> None:
        if object_id in proposed_ids:
            return
        if connection.execute("SELECT 1 FROM objects WHERE id = ?", (object_id,)).fetchone() is None:
            raise ValueError("referenced object does not exist")

    @staticmethod
    def _stored_active_identity_aliases(
        connection: sqlite3.Connection, object_id: str
    ) -> list[str]:
        """Return the object's ACTIVE identity aliases (write-path source).

        Write paths use this view exclusively: legacy ``object_aliases`` rows
        are read-only history and must never be re-inserted as active
        identity (F1). Search/FTS use :meth:`_stored_search_aliases` instead.
        """
        rows = connection.execute(
            "SELECT normalized_alias FROM identity_aliases"
            " WHERE object_id = ? AND status = 'active'"
            " ORDER BY normalized_alias",
            (object_id,),
        ).fetchall()
        return [str(row["normalized_alias"]) for row in rows]

    @classmethod
    def _stored_search_aliases(
        cls, connection: sqlite3.Connection, object_id: str
    ) -> list[str]:
        """Return the combined read/search alias view for one object.

        ACTIVE identity aliases plus any pre-existing legacy ``object_aliases``
        rows (read-only history). Feeds FTS refresh and recall compatibility
        only; never feeds an identity write (F1).
        """
        active = set(cls._stored_active_identity_aliases(connection, object_id))
        legacy = {
            str(row["normalized_alias"])
            for row in connection.execute(
                "SELECT normalized_alias FROM object_aliases WHERE object_id = ?",
                (object_id,),
            ).fetchall()
        }
        return sorted(active | legacy)


def _merge_observed_metadata(
    stored: dict[str, Any],
    proposed: dict[str, Any],
) -> dict[str, Any]:
    """Union the observed-dimension marker across same-id re-writes.

    The main row's metadata records which observation dimensions a source
    has been read in (task T13 fix F1: comments, transcript, content, ...).
    Re-writes accumulate the marker set instead of replacing it, so one
    hydration must never erase a dimension an earlier hydration observed;
    every other metadata key keeps the proposed (freshest) value.

    Args:
        stored: The stored row's parsed metadata.
        proposed: The delta item's metadata.

    Returns:
        The metadata to persist: *proposed* values, with the ``observed``
        marker unioned when either side carries one.

    """
    stored_observed = stored.get("observed")
    proposed_observed = proposed.get("observed")
    if not isinstance(stored_observed, list) and not isinstance(proposed_observed, list):
        return proposed
    merged = list(
        dict.fromkeys(
            [
                *(str(kind) for kind in (stored_observed if isinstance(stored_observed, list) else [])),
                *(str(kind) for kind in (proposed_observed if isinstance(proposed_observed, list) else [])),
            ]
        )
    )
    return {**proposed, "observed": merged}


def _assertion_signature(item: AssertionInput) -> str:
    """Hash the identity inputs of an assertion (deliberately NOT its metadata).

    The signature covers [subject, predicate, object-or-literal, epistemic
    role, event window] only. Empty qualifiers keep the v1 payload
    byte-identical so pre-v10 rows keep matching idempotent replay; non-empty
    qualifiers switch to a version-marked v2 payload that appends the
    normalized qualifier object (keys already lexicographic via the shared
    ``_json`` sort, values normalized by ``normalize_qualifiers``), so
    role/community/granularity differences split identities. The ``"v2:"``
    prefix lives INSIDE the hashed payload and cannot byte-collide with the
    v1 payload, which always starts with ``[`` (a JSON array).
    ``event_time_precision``, ``supersede_reason``, confidence, evidence and
    supersedes must NOT enter the signature: they are interpretation or
    provenance metadata, not identity, and adding them would change the
    signature of every stored row and break idempotent replay of old-format
    commits. Documented consequence: two assertions differing ONLY in those
    fields collide on the signature UNIQUE. The store then keeps the existing
    row, links the new evidence to it and never replaces the metadata; the
    committer announces the collision (``_announce_assertion_dedup``) so the
    dedup is never silent.
    """
    identity = [
        item.subject_id.strip(),
        item.predicate.strip().casefold(),
        item.object_id.strip() if item.object_id is not None else item.literal,
        item.epistemic_role.value,
        _iso(item.event_time_start),
        _iso(item.event_time_end),
    ]
    payload = "v2:" + _json([*identity, item.qualifiers]) if item.qualifiers else _json(identity)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _distinct(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _truncate(message: str, limit: int = 2000) -> str:
    """Cut an aggregated error message to a sane length for repair feedback.

    The proposal ledger caps stored errors at 2000 characters (see
    ``ProposalCommitter._record_attempt``), so the store keeps its own raised
    message under the same bound instead of letting id lists explode it.

    Args:
        message: The full aggregated validation message.
        limit: Maximum message length; 2000 matches the ledger cap.

    Returns:
        *message* unchanged when short enough, otherwise a truncated copy
        ending with an ellipsis.

    """
    if len(message) <= limit:
        return message
    return f"{message[: limit - 3]}..."


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
