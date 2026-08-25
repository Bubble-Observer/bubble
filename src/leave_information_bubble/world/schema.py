"""SQLite schema owned by the durable world store."""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

SCHEMA_V1_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS objects (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    domain_hints_json TEXT NOT NULL,
    provisional INTEGER NOT NULL,
    event_time_start TEXT,
    event_time_end TEXT,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS object_aliases (
    normalized_alias TEXT PRIMARY KEY,
    object_id TEXT NOT NULL REFERENCES objects(id)
);
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    title TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    content_ref TEXT NOT NULL,
    depth TEXT NOT NULL,
    source_published_at TEXT,
    observed_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predicates (
    name TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS assertions (
    id TEXT PRIMARY KEY,
    signature TEXT NOT NULL UNIQUE,
    subject_id TEXT NOT NULL REFERENCES objects(id),
    predicate TEXT NOT NULL REFERENCES predicates(name),
    object_id TEXT REFERENCES objects(id),
    literal_json TEXT,
    epistemic_role TEXT NOT NULL,
    confidence REAL NOT NULL,
    event_time_start TEXT,
    event_time_end TEXT,
    supersedes_id TEXT REFERENCES assertions(id)
);
CREATE TABLE IF NOT EXISTS assertion_evidence (
    assertion_id TEXT NOT NULL REFERENCES assertions(id),
    observation_id TEXT NOT NULL REFERENCES observations(id),
    role TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    PRIMARY KEY(assertion_id, observation_id, role)
);
CREATE TABLE IF NOT EXISTS inquiries (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES objects(id),
    prompt TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS object_observations (
    object_id TEXT NOT NULL REFERENCES objects(id),
    observation_id TEXT NOT NULL REFERENCES observations(id),
    role TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    PRIMARY KEY(object_id, observation_id, role)
);
CREATE TABLE IF NOT EXISTS inquiry_observations (
    inquiry_id TEXT NOT NULL REFERENCES inquiries(id),
    observation_id TEXT NOT NULL REFERENCES observations(id),
    role TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    PRIMARY KEY(inquiry_id, observation_id, role)
);
CREATE TABLE IF NOT EXISTS commit_receipts (
    commit_id TEXT PRIMARY KEY,
    committed_at TEXT NOT NULL,
    receipt_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS world_audit (
    commit_id TEXT PRIMARY KEY REFERENCES commit_receipts(commit_id),
    committed_at TEXT NOT NULL,
    delta_json TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS objects_fts USING fts5(
    id UNINDEXED,
    canonical_name,
    aliases
);
CREATE VIRTUAL TABLE IF NOT EXISTS assertions_fts USING fts5(
    id UNINDEXED,
    subject_id,
    predicate,
    object_id,
    literal
);
"""

# ── v2 migration ──────────────────────────────────────────────────────────────
# Applied only while user_version is 0. Every ALTER is guarded by a
# column-existence check and the dedup-index creation is best-effort, so a
# mid-script failure (e.g. an IntegrityError from legacy duplicate open
# (subject_id, prompt) rows) can never wedge the database: re-runs are safe
# regardless of what state a previous attempt left behind.
_INQUIRY_V2_COLUMNS: tuple[tuple[str, str], ...] = (
    ("kind", "TEXT NOT NULL DEFAULT 'factual'"),
    ("created_at", "TEXT"),
    ("last_attempted_at", "TEXT"),
    ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
    ("resolved_at", "TEXT"),
)

_ASSERTION_V2_COLUMNS: tuple[tuple[str, str], ...] = (("answers_inquiry_id", "TEXT"),)

# Legacy rows get their timestamp columns backfilled (UTC ISO, same shape as
# store._iso) so demote_stale_inquiries can age pre-v2 frontiers.
_MIGRATION_V2_BACKFILL_SQL = (
    "UPDATE inquiries SET"
    " created_at = COALESCE(created_at, strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')),"
    " last_attempted_at = COALESCE(last_attempted_at, strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now'))"
)

_MIGRATION_V2_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS proposal_attempts (
    commit_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('committed','rejected','recovery')),
    new_objects INTEGER NOT NULL,
    assertions INTEGER NOT NULL,
    inquiries INTEGER NOT NULL,
    omitted_assertions INTEGER NOT NULL,
    omitted_inquiries INTEGER NOT NULL,
    omitted_resolutions INTEGER NOT NULL,
    resolved_inquiries INTEGER NOT NULL,
    error TEXT,
    delta_json TEXT NOT NULL
);
"""

MIGRATION_V2_SQL = (
    "".join(
        f"ALTER TABLE inquiries ADD COLUMN {name} {definition};\n" for name, definition in _INQUIRY_V2_COLUMNS
    )
    + "".join(
        f"ALTER TABLE assertions ADD COLUMN {name} {definition};\n"
        for name, definition in _ASSERTION_V2_COLUMNS
    )
    + "CREATE UNIQUE INDEX IF NOT EXISTS idx_open_inquiry_dedup"
    " ON inquiries(subject_id, prompt) WHERE status = 'open';\n"
    + _MIGRATION_V2_BACKFILL_SQL
    + ";\n"
    + _MIGRATION_V2_TABLES_SQL
)

# ── v3 migration ──────────────────────────────────────────────────────────────
# Applied only while user_version is 1 (after v2). The ALTER is guarded by a
# column-existence check like v2's, so a mid-script failure can never wedge
# the database: re-runs are safe regardless of what state a previous attempt
# left behind.
_MIGRATION_V3_ALTER_SQL = (
    "ALTER TABLE proposal_attempts ADD COLUMN"
    " evidence_missing_assertions INTEGER NOT NULL DEFAULT 0"
)


def _apply_migration_v3(connection: sqlite3.Connection) -> None:
    """Apply the v3 migration idempotently: ledger evidence-drop observability.

    Adds ``proposal_attempts.evidence_missing_assertions`` so every proposal
    attempt records how many declared assertions were omitted because their
    evidence was sanitized away (hallucinated observation ids). Legacy ledger
    rows get the NOT NULL DEFAULT 0 backfill automatically.

    Args:
        connection: The database connection being migrated (row_factory must
            be ``sqlite3.Row``); user_version stays 1 until the whole step
            commits.

    """
    connection.execute("BEGIN")
    try:
        attempt_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(proposal_attempts)").fetchall()
        }
        if "evidence_missing_assertions" not in attempt_columns:
            connection.execute(_MIGRATION_V3_ALTER_SQL)
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v4 migration ──────────────────────────────────────────────────────────────
# Applied only while user_version is 2 (after v3). Both statements carry
# IF NOT EXISTS, so a mid-script failure can never wedge the database:
# re-runs are safe regardless of what state a previous attempt left behind.
MIGRATION_V4_SQL = """
CREATE TABLE IF NOT EXISTS observation_bodies (
    observation_id TEXT PRIMARY KEY REFERENCES observations(id),
    content_type TEXT NOT NULL CHECK (content_type IN ('article','subtitle','comments','mixed')),
    body_json TEXT NOT NULL,
    size_bytes INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assertions_subject ON assertions(subject_id);
"""


# ── v5 migration ──────────────────────────────────────────────────────────────
# Applied only while user_version is 3 (after v4). The retired-assertion
# filter (`NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)`)
# full-scans at assertion scale without this index. IF NOT EXISTS keeps
# re-runs safe regardless of what state a previous attempt left behind.
MIGRATION_V5_SQL = """
CREATE INDEX IF NOT EXISTS idx_assertions_supersedes ON assertions(supersedes_id);
"""


# ── v6 migration ──────────────────────────────────────────────────────────────
# Applied only while user_version is 4 (after v5). Performance indexes (task
# second-batch #2): changes()/search() filter world_audit on committed_at and
# expand() matches assertions on the object_id side of its
# `subject_id = ? OR object_id = ?` condition (idx_assertions_subject already
# covers the subject side from v4). Indexes never change query semantics.
# IF NOT EXISTS keeps re-runs safe regardless of what state a previous
# attempt left behind.
MIGRATION_V6_SQL = """
CREATE INDEX IF NOT EXISTS idx_world_audit_committed_at ON world_audit(committed_at);
CREATE INDEX IF NOT EXISTS idx_assertions_object ON assertions(object_id);
"""


# ── v7 migration ──────────────────────────────────────────────────────────────
# Applied only while user_version is 5 (after v6). Adds the correction-history
# column ``superseded_at`` (timestamps the moment a new assertion supersedes an
# old one; it lives on the SUPERSEDING row because the store is pure-INSERT —
# old rows are never touched, so idempotent replay and the signature UNIQUE
# hold) and the ``current_assertions`` view (recall.py's canonical
# retired-assertion filter, exposed as a queryable object for direct SQL
# consumers). The ALTER is guarded by a column-existence check like v2/v3's,
# so a mid-script failure can never wedge the database: re-runs are safe
# regardless of what state a previous attempt left behind.
#
# The view's retired-assertion filter uses ``NOT IN`` over the non-null
# ``supersedes_id`` set — semantically identical to the original correlated
# ``NOT EXISTS`` (an id is retired exactly when some row's ``supersedes_id``
# points at it; null supersedes_id values are excluded so three-valued NOT IN
# semantics cannot drop rows) — but the SQLite planner materializes the set
# once (LIST SUBQUERY) instead of re-scanning the supersedes covering index
# per outer candidate row. Measured on a skewed 10k commit-candidate seed:
# 359.6 ms -> 1.24 ms median (closeout 2026-08-18 correction; databases
# created before the rewrite are rebuilt by the v11 migration; no new index).
_MIGRATION_V7_ALTER_SQL = "ALTER TABLE assertions ADD COLUMN superseded_at TEXT"

MIGRATION_V7_SQL = """
CREATE VIEW IF NOT EXISTS current_assertions AS
SELECT a.* FROM assertions a
WHERE a.id NOT IN (SELECT supersedes_id FROM assertions WHERE supersedes_id IS NOT NULL);
"""


def _apply_migration_v7(connection: sqlite3.Connection) -> None:
    """Apply the v7 migration idempotently: superseded_at column and the current_assertions view.

    Adds ``assertions.superseded_at`` (nullable TEXT) and creates the
    ``current_assertions`` view. The ALTER runs only while the column is
    absent, and the view uses CREATE VIEW IF NOT EXISTS, so a partial previous
    attempt re-applies cleanly.

    Args:
        connection: The database connection being migrated (row_factory must
            be ``sqlite3.Row``); user_version stays 5 until the whole step
            commits.

    """
    connection.execute("BEGIN")
    try:
        assertion_columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(assertions)").fetchall()
        }
        if "superseded_at" not in assertion_columns:
            connection.execute(_MIGRATION_V7_ALTER_SQL)
        # Recreate the view on every pass: the definition is executable
        # contract (closeout correction measured the NOT IN rewrite), and a
        # database arriving at user_version 5 may still carry the
        # correlated-NOT-EXISTS definition (pre-rewrite v7 pass). DROP before
        # CREATE so the new text wins; both are in one transaction, so a
        # failure rolls back to the old view. Databases already at v6–v10 are
        # outside this step — the v11 migration rebuilds their view instead.
        connection.execute("DROP VIEW IF EXISTS current_assertions")
        connection.execute(MIGRATION_V7_SQL)
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── P1 attempt-ledger migration (user_version 6 → 7) ─────────────────────────
# P1 separates a logical run from its durable commits and review attempts.  The
# old table cannot be ALTERed because ``commit_id`` was its primary key, so this
# migration rebuilds it inside one transaction.  It is intentionally exercised
# only against synthetic user_version 6 fixtures; production b36b content is
# never input.
_PROPOSAL_ATTEMPTS_V7_SQL = """
CREATE TABLE proposal_attempts_v7 (
    attempt_id TEXT PRIMARY KEY,
    -- Kept for legacy read-only SQL consumers; new code uses the explicit
    -- run/durable columns below.
    commit_id TEXT,
    run_commit_id TEXT NOT NULL DEFAULT '',
    durable_commit_id TEXT,
    attempt_no INTEGER NOT NULL DEFAULT 1 CHECK (attempt_no >= 1),
    parent_attempt_id TEXT,
    thread_id TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    outcome TEXT NOT NULL,
    new_objects INTEGER NOT NULL,
    assertions INTEGER NOT NULL,
    inquiries INTEGER NOT NULL,
    omitted_assertions INTEGER NOT NULL,
    omitted_inquiries INTEGER NOT NULL,
    omitted_resolutions INTEGER NOT NULL,
    resolved_inquiries INTEGER NOT NULL,
    evidence_missing_assertions INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    delta_json TEXT NOT NULL,
    issues_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE(run_commit_id, attempt_no)
);
"""

_PROPOSAL_ATTEMPTS_V7_TRIGGER = """
CREATE TRIGGER proposal_attempts_legacy_identity
AFTER INSERT ON proposal_attempts_v7
WHEN NEW.attempt_id IS NULL OR NEW.run_commit_id = ''
BEGIN
    UPDATE proposal_attempts_v7
    SET attempt_id = COALESCE(NEW.attempt_id, NEW.commit_id || ':legacy:' || NEW.rowid),
        run_commit_id = CASE WHEN NEW.run_commit_id = '' THEN NEW.commit_id ELSE NEW.run_commit_id END,
        durable_commit_id = COALESCE(
            NEW.durable_commit_id,
            CASE WHEN NEW.outcome = 'committed' THEN NEW.commit_id ELSE NULL END
        )
    WHERE rowid = NEW.rowid;
END;
"""


def _apply_migration_v7_attempt_ledger(connection: sqlite3.Connection) -> None:
    """Atomically rebuild the old ledger and advance user_version from 6 to 7."""
    connection.execute("BEGIN")
    try:
        connection.execute(_PROPOSAL_ATTEMPTS_V7_SQL)
        connection.execute(
            "INSERT INTO proposal_attempts_v7("
            "attempt_id, commit_id, run_commit_id, durable_commit_id, attempt_no, parent_attempt_id,"
            " thread_id, attempted_at, outcome, new_objects, assertions, inquiries,"
            " omitted_assertions, omitted_inquiries, omitted_resolutions, resolved_inquiries,"
            " evidence_missing_assertions, error, delta_json, issues_json) "
            "SELECT commit_id || ':legacy:1', commit_id, commit_id, "
            "CASE WHEN outcome = 'committed' THEN commit_id ELSE NULL END, 1, NULL, "
            "thread_id, attempted_at, outcome, new_objects, assertions, inquiries, "
            "omitted_assertions, omitted_inquiries, omitted_resolutions, resolved_inquiries, "
            "evidence_missing_assertions, error, delta_json, '[]' FROM proposal_attempts"
        )
        connection.execute("DROP TABLE proposal_attempts")
        connection.execute("ALTER TABLE proposal_attempts_v7 RENAME TO proposal_attempts")
        connection.execute(
            _PROPOSAL_ATTEMPTS_V7_TRIGGER.replace(
                "proposal_attempts_v7", "proposal_attempts"
            )
        )
        connection.execute("PRAGMA user_version = 7")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v8 inquiry-touch ledger migration (user_version 7 → 8) ───────────────────
_INQUIRY_ATTEMPT_TOUCH_LEDGER_SQL = """
CREATE TABLE inquiry_attempt_touches (
    run_commit_id TEXT NOT NULL,
    inquiry_id TEXT NOT NULL REFERENCES inquiries(id),
    touched_at TEXT NOT NULL,
    PRIMARY KEY(run_commit_id, inquiry_id)
)
"""

_INQUIRY_ATTEMPT_TOUCH_LEDGER_INDEX_SQL = """
CREATE INDEX idx_inquiry_attempt_touches_inquiry
ON inquiry_attempt_touches(inquiry_id)
"""


def _apply_migration_v8_inquiry_touch_ledger(connection: sqlite3.Connection) -> None:
    """Atomically add or verify per-run inquiry-touch idempotency and advance to v8."""
    connection.execute("BEGIN")
    try:
        table = connection.execute(
            "SELECT type FROM sqlite_master WHERE name = 'inquiry_attempt_touches'"
        ).fetchone()
        if table is None:
            connection.execute(_INQUIRY_ATTEMPT_TOUCH_LEDGER_SQL)
        elif table["type"] != "table":
            raise sqlite3.OperationalError(
                "inquiry_attempt_touches exists but is not a v8 ledger table"
            )
        else:
            columns = [
                (str(row["name"]), str(row["type"]).upper(), int(row["notnull"]), int(row["pk"]))
                for row in connection.execute(
                    "PRAGMA table_info(inquiry_attempt_touches)"
                ).fetchall()
            ]
            expected = [
                ("run_commit_id", "TEXT", 1, 1),
                ("inquiry_id", "TEXT", 1, 2),
                ("touched_at", "TEXT", 1, 0),
            ]
            foreign_keys = [
                (str(row["table"]), str(row["from"]), str(row["to"]))
                for row in connection.execute(
                    "PRAGMA foreign_key_list(inquiry_attempt_touches)"
                ).fetchall()
            ]
            if columns != expected or foreign_keys != [("inquiries", "inquiry_id", "id")]:
                raise sqlite3.OperationalError(
                    "existing inquiry_attempt_touches schema is incompatible with v8"
                )

        index = connection.execute(
            "SELECT type, tbl_name FROM sqlite_master "
            "WHERE name = 'idx_inquiry_attempt_touches_inquiry'"
        ).fetchone()
        if index is None:
            connection.execute(_INQUIRY_ATTEMPT_TOUCH_LEDGER_INDEX_SQL)
        else:
            index_columns = [
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA index_info(idx_inquiry_attempt_touches_inquiry)"
                ).fetchall()
            ]
            if (
                index["type"] != "index"
                or index["tbl_name"] != "inquiry_attempt_touches"
                or index_columns != ["inquiry_id"]
            ):
                raise sqlite3.OperationalError(
                    "existing inquiry touch ledger index is incompatible with v8"
                )
        connection.execute("PRAGMA user_version = 8")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v9 additive migration (user_version 8 → 9) ───────────────────────────────
# Adds event-time precision metadata and explicit supersede reasons, plus the
# inquiry ``closed`` state support (spec §7/§8). All three are additive:
#   - ``assertions.event_time_precision`` TEXT NOT NULL DEFAULT 'unknown'
#     (exact/interval/period/unknown) — value validation lives in the pydantic
#     layer (proposal.py / contracts.py); the default backfills every legacy
#     row in one ALTER, so replay of old-format deltas still matches.
#   - ``assertions.supersede_reason`` TEXT NULL — deliberately NOT part of the
#     assertion signature: the signature hashes [subject, predicate, object,
#     role, event window], and adding a hash input would break replay of every
#     stored commit. Consequence (documented): two assertions differing ONLY
#     in supersede_reason collide on the signature UNIQUE; the store keeps the
#     existing row (linking the new evidence) and the committer announces the
#     collision with the un-replaced differences, so the dedup is never silent.
#   - ``objects.event_time_precision`` TEXT NOT NULL DEFAULT 'unknown'.
#   - ``inquiries.closed_reason`` TEXT NULL, durable reason for the closed
#     state (open→dormant→closed→open(reopen)).
# The ALTERs are guarded by per-table column-existence checks like v2/v3/v7,
# so a mid-script failure can never wedge the database: re-runs are safe
# regardless of what state a previous attempt left behind.
_MIGRATION_V9_COLUMNS = (
    ("assertions", "event_time_precision", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("assertions", "supersede_reason", "TEXT"),
    ("objects", "event_time_precision", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("inquiries", "closed_reason", "TEXT"),
)


def _apply_migration_v9(connection: sqlite3.Connection) -> None:
    """Apply the v9 migration idempotently and advance user_version to 9.

    Adds the event_time_precision, supersede_reason and closed_reason columns
    (each ALTER runs only while its column is absent) inside one transaction.
    ``NOT NULL DEFAULT 'unknown'`` backfills legacy rows in-place, so
    old-format stored deltas keep replaying with equal values.

    Args:
        connection: The database connection being migrated (row_factory must
            be ``sqlite3.Row``); user_version stays 8 until the whole step
            commits.

    """
    connection.execute("BEGIN")
    try:
        for table, name, definition in _MIGRATION_V9_COLUMNS:
            columns = {
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if name not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        connection.execute("PRAGMA user_version = 9")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v10 graph-contract migration (user_version 9 → 10) ───────────────────────
# Adds the durable-cognition graph contract vocabulary in one atomic step:
#   - ``objects.type_key`` TEXT NULL — domain-independent sub-kind (None or
#     [a-z][a-z0-9._-]{0,63} after strip+casefold; no enum table maintained);
#     NULL on legacy rows, matching their absent type.
#   - ``assertions.qualifiers_json`` TEXT NULL — compact JSON object of
#     qualifiers (role/language/community/scope/granularity, at most 5 entries)
#     that don't belong in the assertion signature.
#   - ``identity_aliases`` table + two indexes — the world-scope strong-identity
#     alias store consumed by later contract slices. Deliberately plain CREATE
#     TABLE/INDEX (not IF NOT EXISTS) and NOT a copy/backfill of the legacy
#     ``object_aliases`` table: the whole step runs inside ONE transaction, so
#     a mid-step failure (e.g. a name collision) rolls back the ALTERs and the
#     version together, and re-runs happen only through the user_version guard
#     in ``initialize``.
# The ALTERs are guarded by per-table column-existence checks like v2/v3/v7/v9,
# and ``PRAGMA user_version = 10`` is the LAST statement so the version only
# advances when the whole step commits.
_MIGRATION_V10_COLUMNS = (
    ("objects", "type_key", "TEXT"),
    ("assertions", "qualifiers_json", "TEXT"),
)

_MIGRATION_V10_IDENTITY_ALIASES_SQL = """
CREATE TABLE identity_aliases (
    object_id TEXT NOT NULL REFERENCES objects(id),
    raw_alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'removed')),
    added_commit_id TEXT NOT NULL,
    removed_commit_id TEXT,
    PRIMARY KEY (object_id, normalized_alias)
)
"""

_MIGRATION_V10_IDENTITY_ALIASES_ACTIVE_INDEX_SQL = """
CREATE UNIQUE INDEX idx_identity_aliases_active_normalized
    ON identity_aliases(normalized_alias) WHERE status = 'active'
"""

_MIGRATION_V10_IDENTITY_ALIASES_OBJECT_INDEX_SQL = """
CREATE INDEX idx_identity_aliases_object
    ON identity_aliases(object_id, status)
"""


def _apply_migration_v10(connection: sqlite3.Connection) -> None:
    """Apply the v10 migration atomically and advance user_version to 10.

    Adds the graph-contract columns (each ALTER runs only while its column is
    absent) and the ``identity_aliases`` table with its two indexes inside ONE
    transaction; ``PRAGMA user_version = 10`` is the last statement so the
    version advances only when the whole step commits. Legacy rows are never
    copied: ``identity_aliases`` starts empty and the old ``object_aliases``
    table is left untouched.

    Args:
        connection: The database connection being migrated (row_factory must
            be ``sqlite3.Row``); user_version stays 9 until the whole step
            commits.

    """
    connection.execute("BEGIN")
    try:
        for table, name, definition in _MIGRATION_V10_COLUMNS:
            columns = {
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if name not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        connection.execute(_MIGRATION_V10_IDENTITY_ALIASES_SQL)
        connection.execute(_MIGRATION_V10_IDENTITY_ALIASES_ACTIVE_INDEX_SQL)
        connection.execute(_MIGRATION_V10_IDENTITY_ALIASES_OBJECT_INDEX_SQL)
        connection.execute("PRAGMA user_version = 10")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v11 migration (user_version 10 → 11) ────────────────────────────────────
# Rebuilds ``current_assertions`` with the NOT IN definition for databases
# whose view predates the closeout rewrite. The v7 step only runs at
# user_version 5, so any database at version 6–10 created before the rewrite
# keeps the correlated-NOT-EXISTS definition forever: ``initialize`` re-runs
# the chain but no step touches the view, and a contract canary cloned from
# such a snapshot would exercise the pre-fix plan. The view text is
# executable contract, so the rewrite must be delivered as an auditable
# migration rather than a silent startup check — one user_version maps to
# exactly one view definition. A single step at the end of the chain covers
# the whole affected range: every such database reaches version 10 on its
# next initialize and triggers this step. ``PRAGMA user_version = 11`` is the
# LAST statement so the version only advances when the whole step commits; a
# failure rolls back to the old view text and version 10 together.


def _apply_migration_v11(connection: sqlite3.Connection) -> None:
    """Apply the v11 migration atomically and advance user_version to 11.

    Drops and recreates ``current_assertions`` with the NOT IN retired-filter
    text (the same definition the v7 step writes for fresh chains) inside ONE
    transaction; ``PRAGMA user_version = 11`` is the last statement so the
    version advances only when the whole step commits. A failure rolls back
    to the previous view definition and version together.

    Args:
        connection: The database connection being migrated (row_factory must
            be ``sqlite3.Row``); user_version stays 10 until the whole step
            commits.

    """
    connection.execute("BEGIN")
    try:
        connection.execute("DROP VIEW IF EXISTS current_assertions")
        connection.execute(MIGRATION_V7_SQL)
        connection.execute("PRAGMA user_version = 11")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v12 staging migration (user_version 11 → 12) ────────────────────────────
# Durable working graph (plan §6.2, D-006): typed staging tables isolated by
# wake_id, mirrored on the formal tables' column shapes, plus the patch
# idempotency ledger (wake_id + op_id keyed). Status lifecycle: active →
# finalized / abandoned; drop and cancel are status changes, never deletes.
MIGRATION_V12_SQL = """
CREATE TABLE IF NOT EXISTS staged_objects (
    staged_id TEXT PRIMARY KEY,
    wake_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    target_ref TEXT,
    kind TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    type_key TEXT,
    domain_hints_json TEXT NOT NULL,
    event_time_start TEXT,
    event_time_end TEXT,
    aliases_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_staged_objects_wake_status
    ON staged_objects(wake_id, status);
CREATE TABLE IF NOT EXISTS staged_assertions (
    staged_id TEXT PRIMARY KEY,
    wake_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    target_ref TEXT,
    subject_ref TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_ref TEXT,
    literal_json TEXT,
    epistemic_role TEXT NOT NULL,
    confidence REAL NOT NULL,
    event_time_start TEXT,
    event_time_end TEXT,
    supersedes_ref TEXT,
    qualifiers_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_staged_assertions_wake_status
    ON staged_assertions(wake_id, status);
CREATE TABLE IF NOT EXISTS staged_inquiries (
    staged_id TEXT PRIMARY KEY,
    wake_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    target_ref TEXT,
    subject_ref TEXT NOT NULL,
    prompt TEXT NOT NULL,
    rationale TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'factual',
    deepens_ref TEXT,
    answers_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_staged_inquiries_wake_status
    ON staged_inquiries(wake_id, status);
CREATE TABLE IF NOT EXISTS staged_patch_receipts (
    wake_id TEXT NOT NULL,
    op_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (wake_id, op_id)
);
"""


def _apply_migration_v12(connection: sqlite3.Connection) -> None:
    """Apply the v12 staging migration atomically and advance user_version to 12."""
    connection.execute("BEGIN")
    try:
        connection.executescript(MIGRATION_V12_SQL)
        connection.execute("PRAGMA user_version = 12")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v13 staging columns (user_version 12 → 13) ──────────────────────────────
# Additive columns that close the staged-object mirroring gap with the formal
# objects table (slice 4, D-013): ``provisional`` feeds the zero-connection /
# event-anchor readiness rules (plan §6.9), and ``identity_basis_json``
# records confirm_distinct decisions so graph_inspect can tell a resolved
# identity from an unresolved one without contradicting an accepted patch.
# Each ALTER is guarded by a column check: SQLite has no ADD COLUMN IF NOT
# EXISTS, and the synthetic-version tests rewind a current database to an old
# user_version then re-run the chain, where the columns already exist (the
# same tolerance CREATE TABLE IF NOT EXISTS gives the v12 staging migration).


# ── v14 finalize-receipt ledger (user_version 13 → 14) ─────────────────────
# Durable receipt store for the Graph Shell finalize bridge (plan §6.10,
# slice 5): one row per wake that successfully finalized, keyed by wake_id so
# a repeat finalize can return the original receipt without recompiling. The
# commit_id references the formal commit_receipts row written in the same
# SQLite transaction (D-007 same-transaction route), so a receipt row implies
# the formal commit exists and vice versa. CREATE TABLE IF NOT EXISTS keeps
# the synthetic-version rewind tests idempotent (same tolerance as v12).
MIGRATION_V14_SQL = """
CREATE TABLE IF NOT EXISTS finalize_receipts (
    wake_id TEXT PRIMARY KEY,
    commit_id TEXT NOT NULL REFERENCES commit_receipts(commit_id),
    receipt_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _apply_migration_v14(connection: sqlite3.Connection) -> None:
    """Apply the v14 finalize-receipt ledger atomically and advance to user_version 14."""
    connection.execute("BEGIN")
    try:
        connection.executescript(MIGRATION_V14_SQL)
        connection.execute("PRAGMA user_version = 14")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v15 patch-time formal base (user_version 14 → 15) ─────────────────────
# A formal object overlay must retain the version it was based on at patch
# time. Finalize compares this durable base with the current formal version
# and fails closed instead of silently rebasing an old draft (G5b-1 I8).
# Existing overlays upgraded from v14 remain NULL because their historical
# base is unknowable. Preflight reports that uncertainty and requires an
# explicit reread/repatch instead of fabricating a trustworthy baseline.


def _apply_migration_v15(connection: sqlite3.Connection) -> None:
    """Add the durable formal-base version and advance to user_version 15."""
    connection.execute("BEGIN")
    try:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(staged_objects)").fetchall()
        }
        if "base_version" not in columns:
            connection.execute("ALTER TABLE staged_objects ADD COLUMN base_version INTEGER")
        connection.execute("PRAGMA user_version = 15")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v16 Graph Shell wake identity claims (user_version 15 → 16) ──────────
# A fresh wake can have no staging or finalize receipt, so its single-use
# identity cannot live only in a runtime/checkpoint database that callers may
# replace.  The world-scoped claim is a small control ledger, not cognition.

MIGRATION_V16_SQL = """
CREATE TABLE IF NOT EXISTS graph_shell_wake_claims (
    wake_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    runtime_store_identity TEXT NOT NULL,
    claimed_at REAL NOT NULL
);
"""


def _apply_migration_v16(connection: sqlite3.Connection) -> None:
    """Create the world-scoped single-use wake ledger and advance to v16."""
    connection.execute("BEGIN")
    try:
        connection.executescript(MIGRATION_V16_SQL)
        connection.execute("PRAGMA user_version = 16")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


# ── v17 Graph Shell singleton writer lease (user_version 16 → 17) ──────────
# The world's single active Graph Shell writer is arbitrated by one row in
# ``graph_shell_writer_leases`` (singleton_id = 1 PK CHECK). The table is a
# schema citizen like every other world table: it comes from this migration
# (and therefore from ``initialize`` / ``initialize_blank_world``), not from
# a lazy ``CREATE TABLE IF NOT EXISTS`` in the runner. Pre-v17 worlds acquire
# it automatically the first time a production ``WorldStore`` opens them.

MIGRATION_V17_SQL = """
CREATE TABLE IF NOT EXISTS graph_shell_writer_leases (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    owner_wake_id TEXT NOT NULL,
    owner_thread_id TEXT NOT NULL,
    claimed_at REAL NOT NULL
);
"""


def _apply_migration_v17(connection: sqlite3.Connection) -> None:
    """Create the singleton writer-lease table and advance to v17."""
    connection.execute("BEGIN")
    try:
        connection.executescript(MIGRATION_V17_SQL)
        connection.execute("PRAGMA user_version = 17")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _apply_migration_v18(connection: sqlite3.Connection) -> None:
    """Add the v18 inquiry columns atomically and advance to user_version 18.

    ``inquiries.deepens_id`` is the formal deepen parent link (validated in
    ``store._validate_delta``, no FK — same discipline as
    ``assertions.answers_inquiry_id``); ``staged_assertions.answers_ref`` is
    the staging-layer answer reference, so a staged resolve can name the exact
    answering assertion it is grounded in. Both ALTERs are guarded so a
    partially applied migration re-runs safely.
    """
    connection.execute("BEGIN")
    try:
        for table, column in (
            ("inquiries", "deepens_id"),
            ("staged_assertions", "answers_ref"),
        ):
            columns = {
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        connection.execute("PRAGMA user_version = 18")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _apply_migration_v13(connection: sqlite3.Connection) -> None:
    """Apply the v13 staged-object columns atomically and advance to user_version 13."""
    connection.execute("BEGIN")
    try:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(staged_objects)").fetchall()
        }
        if "provisional" not in columns:
            connection.execute(
                "ALTER TABLE staged_objects"
                " ADD COLUMN provisional INTEGER NOT NULL DEFAULT 0"
            )
        if "identity_basis_json" not in columns:
            connection.execute(
                "ALTER TABLE staged_objects"
                " ADD COLUMN identity_basis_json TEXT NOT NULL DEFAULT '[]'"
            )
        connection.execute("PRAGMA user_version = 13")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


PREDICATES = (
    "related_to",
    "participated_in",
    "occurred_at",
    "has_result",
    "expresses",
    "originated_from",
    "used_in",
    "explains",
    "supersedes",
)


def initialize(connection: sqlite3.Connection) -> None:
    """Create or migrate the world schema, then seed predicates and set user_version."""
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_V1_SQL)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 0:
        _apply_migration_v2(connection)
        connection.execute("PRAGMA user_version = 1")
    if connection.execute("PRAGMA user_version").fetchone()[0] == 1:
        _apply_migration_v3(connection)
        connection.execute("PRAGMA user_version = 2")
    if connection.execute("PRAGMA user_version").fetchone()[0] == 2:
        connection.executescript(MIGRATION_V4_SQL)
        connection.execute("PRAGMA user_version = 3")
    if connection.execute("PRAGMA user_version").fetchone()[0] == 3:
        connection.executescript(MIGRATION_V5_SQL)
        connection.execute("PRAGMA user_version = 4")
    if connection.execute("PRAGMA user_version").fetchone()[0] == 4:
        connection.executescript(MIGRATION_V6_SQL)
        connection.execute("PRAGMA user_version = 5")
    if connection.execute("PRAGMA user_version").fetchone()[0] == 5:
        _apply_migration_v7(connection)
        connection.execute("PRAGMA user_version = 6")
    if connection.execute("PRAGMA user_version").fetchone()[0] == 6:
        _apply_migration_v7_attempt_ledger(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 7:
        _apply_migration_v8_inquiry_touch_ledger(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 8:
        _apply_migration_v9(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 9:
        _apply_migration_v10(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 10:
        _apply_migration_v11(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 11:
        _apply_migration_v12(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 12:
        _apply_migration_v13(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 13:
        _apply_migration_v14(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 14:
        _apply_migration_v15(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 15:
        _apply_migration_v16(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 16:
        _apply_migration_v17(connection)
    if connection.execute("PRAGMA user_version").fetchone()[0] == 17:
        _apply_migration_v18(connection)
    connection.executemany(
        "INSERT OR IGNORE INTO predicates(name) VALUES (?)", ((name,) for name in PREDICATES)
    )
    connection.commit()


def _apply_migration_v2(connection: sqlite3.Connection) -> None:
    """Apply the v2 migration idempotently so a failure never wedges the DB.

    Each ALTER runs only when its column is absent, and the dedup-index
    creation is wrapped in an IntegrityError guard: legacy duplicate open
    ``(subject_id, prompt)`` rows log the offenders and skip the index while
    the rest of the migration (and every subsequent open) continues — E2
    soft-omission and INSERT OR IGNORE remain the operational backstop.

    Args:
        connection: The database connection being migrated (row_factory must
            be ``sqlite3.Row``); version stays 0 until the whole step commits.

    """
    connection.executescript(_MIGRATION_V2_TABLES_SQL)  # idempotent IF NOT EXISTS
    connection.execute("BEGIN")
    try:
        inquiry_columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(inquiries)").fetchall()
        }
        for name, definition in _INQUIRY_V2_COLUMNS:
            if name not in inquiry_columns:
                connection.execute(f"ALTER TABLE inquiries ADD COLUMN {name} {definition}")
        assertion_columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(assertions)").fetchall()
        }
        for name, definition in _ASSERTION_V2_COLUMNS:
            if name not in assertion_columns:
                connection.execute(f"ALTER TABLE assertions ADD COLUMN {name} {definition}")
        try:
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_open_inquiry_dedup"
                " ON inquiries(subject_id, prompt) WHERE status = 'open'"
            )
        except sqlite3.IntegrityError:
            duplicates = connection.execute(
                "SELECT subject_id, prompt FROM inquiries WHERE status = 'open'"
                " GROUP BY subject_id, prompt HAVING COUNT(*) > 1"
            ).fetchall()
            logger.warning(
                "idx_open_inquiry_dedup skipped: %d legacy duplicate open inquiry"
                " (subject_id, prompt) rows; E2 soft-omission and INSERT OR IGNORE"
                " remain the operational backstop: %s",
                len(duplicates),
                [dict(row) for row in duplicates],
            )
        connection.execute(_MIGRATION_V2_BACKFILL_SQL)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
