"""Schema v2 migration and proposal-attempts ledger tests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    InquiryInput,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)
from leave_information_bubble.world.schema import (
    MIGRATION_V4_SQL,
    MIGRATION_V5_SQL,
    MIGRATION_V6_SQL,
    SCHEMA_V1_SQL,
    _apply_migration_v2,
    _apply_migration_v3,
    _apply_migration_v7,
    _apply_migration_v7_attempt_ledger,
    _apply_migration_v8_inquiry_touch_ledger,
    _apply_migration_v9,
    initialize,
)


def _v1_database(tmp_path) -> sqlite3.Connection:
    """Build a database with the pre-v2 schema (user_version=0, no new columns)."""
    conn = sqlite3.connect(tmp_path / "v1.sqlite3")
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE inquiries (
            id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, prompt TEXT NOT NULL,
            rationale TEXT NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL
        );
        CREATE TABLE assertions (
            id TEXT PRIMARY KEY, signature TEXT NOT NULL UNIQUE, subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL, object_id TEXT, literal_json TEXT, epistemic_role TEXT NOT NULL,
            confidence REAL NOT NULL, event_time_start TEXT, event_time_end TEXT, supersedes_id TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO inquiries VALUES ('inq-1','obj-1','old prompt','old rationale','open',1)"
    )
    conn.commit()
    return conn


def test_initialize_migrates_v1_to_v2(tmp_path):
    conn = _v1_database(tmp_path)
    initialize(conn)
    # v2..v16 all apply in one initialize pass: v2 columns, then the v3 ledger
    # column, then the v4 observation_bodies table, then the v5 supersedes
    # index, then the v6 performance indexes, then the v7 correction-history
    # column and view, then the v8 inquiry-touch idempotency ledger, then
    # v9..v11 columns, then the v12 staging tables, then the v13 staged
    # provisional/identity-basis columns, then the v14 finalize-receipt ledger
    # and the v15 patch-time formal-base column plus v16 wake-claim ledger
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    # new columns exist with defaults
    row = conn.execute(
        "SELECT kind, created_at, last_attempted_at, attempt_count, resolved_at"
        " FROM inquiries WHERE id='inq-1'"
    ).fetchone()
    assert row["kind"] == "factual"
    assert row["attempt_count"] == 0
    # legacy rows are timestamp-backfilled so stale aging applies to them too
    assert row["created_at"] is not None
    assert row["last_attempted_at"] is not None
    # assertions column added
    assert "answers_inquiry_id" in [
        r[1] for r in conn.execute("PRAGMA table_info(assertions)")
    ]
    # dedup index + ledger table exist
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_open_inquiry_dedup'"
    ).fetchone() is not None
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='proposal_attempts'"
    ).fetchone() is not None
    assert conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='graph_shell_wake_claims'"
    ).fetchone() is not None


def test_initialize_is_idempotent(tmp_path):
    conn = _v1_database(tmp_path)
    initialize(conn)
    initialize(conn)  # second run must not fail or duplicate
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assert conn.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0] == 1


def test_open_inquiry_dedup_index_rejects_duplicate(tmp_path):
    conn = _v1_database(tmp_path)
    initialize(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO inquiries(id, subject_id, prompt, rationale, status, version) "
            "VALUES ('inq-2','obj-1','old prompt','x','open',1)"
        )


def test_initialize_survives_legacy_duplicate_open_inquiries(tmp_path):
    # a pre-v2 database can already hold duplicate open (subject_id, prompt)
    # rows; the migration must skip the unique index gracefully instead of
    # wedging the DB with user_version 0 and ALTERs half-applied
    conn = _v1_database(tmp_path)
    conn.execute(
        "INSERT INTO inquiries VALUES ('inq-2','obj-1','old prompt','old rationale','open',1)"
    )
    conn.commit()
    initialize(conn)  # must succeed: index creation is skipped, not fatal
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assert conn.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0] == 2
    initialize(conn)  # a subsequent run must also succeed (no wedge)


def _v2_database(tmp_path) -> sqlite3.Connection:
    """Build a database migrated only to v2 (user_version=1, no v3 column).

    ``initialize`` now applies v2 and v3 in one pass, so the v2 state is
    produced by calling the v2 migration step directly.
    """
    conn = _v1_database(tmp_path)
    conn.row_factory = sqlite3.Row
    _apply_migration_v2(conn)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    return conn


def test_initialize_migrates_v2_to_v3(tmp_path):
    conn = _v2_database(tmp_path)
    conn.execute(
        "INSERT INTO proposal_attempts(commit_id, thread_id, attempted_at, outcome,"
        " new_objects, assertions, inquiries, omitted_assertions, omitted_inquiries,"
        " omitted_resolutions, resolved_inquiries, error, delta_json)"
        " VALUES ('c-1','t-1','2026-08-04T00:00:00+00:00','committed',1,1,0,0,0,0,0,NULL,'{}')"
    )
    conn.commit()

    initialize(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    columns = [r[1] for r in conn.execute("PRAGMA table_info(proposal_attempts)")]
    assert "evidence_missing_assertions" in columns
    # legacy ledger rows get the DEFAULT 0 backfill
    row = conn.execute(
        "SELECT evidence_missing_assertions FROM proposal_attempts WHERE commit_id='c-1'"
    ).fetchone()
    assert row[0] == 0


def test_fresh_database_reaches_latest_schema(tmp_path):
    conn = sqlite3.connect(tmp_path / "fresh.sqlite3")
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    columns = [r[1] for r in conn.execute("PRAGMA table_info(proposal_attempts)")]
    assert "evidence_missing_assertions" in columns
    staged_columns = [r[1] for r in conn.execute("PRAGMA table_info(staged_objects)")]
    assert staged_columns.count("base_version") == 1


def test_v15_formal_base_migration_is_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "v14.sqlite3")
    initialize(conn)
    conn.execute("PRAGMA user_version = 14")
    conn.commit()

    initialize(conn)
    initialize(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    staged_columns = [r[1] for r in conn.execute("PRAGMA table_info(staged_objects)")]
    assert staged_columns.count("base_version") == 1


def test_v15_does_not_invent_base_for_existing_active_overlay(tmp_path):
    conn = sqlite3.connect(tmp_path / "v14-active.sqlite3")
    conn.row_factory = sqlite3.Row
    initialize(conn)
    conn.execute(
        "INSERT INTO objects (id, kind, canonical_name, domain_hints_json, provisional, version)"
        " VALUES ('obj-1', 'entity', 'Formal', '[]', 0, 7)"
    )
    conn.execute(
        "INSERT INTO staged_objects (staged_id, wake_id, status, target_ref, kind,"
        " canonical_name, domain_hints_json, aliases_json, created_at, updated_at, version,"
        " base_version) VALUES ('wake:s1', 'wake', 'active', 'obj-1', 'entity',"
        " 'Old Draft', '[]', '[]', '2026-08-20', '2026-08-20', 1, NULL)"
    )
    conn.execute("PRAGMA user_version = 14")
    conn.commit()

    initialize(conn)

    row = conn.execute(
        "SELECT base_version FROM staged_objects WHERE staged_id = 'wake:s1'"
    ).fetchone()
    assert row["base_version"] is None


def test_attempt_ledger_migrates_synthetic_user_version_6_rows_without_overwriting_history(tmp_path):
    """P1 upgrades only a synthetic old fixture and preserves its ledger projection."""
    conn = sqlite3.connect(tmp_path / "synthetic-user-version-6.sqlite3")
    initialize(conn)
    conn.execute(
        "INSERT INTO proposal_attempts(commit_id, thread_id, attempted_at, outcome,"
        " new_objects, assertions, inquiries, omitted_assertions, omitted_inquiries,"
        " omitted_resolutions, resolved_inquiries, evidence_missing_assertions, error, delta_json)"
        " VALUES ('legacy-accepted', 'thread', '2026-08-09T00:00:00+00:00', 'committed',"
        " 1, 2, 3, 0, 0, 0, 0, 0, NULL, '{}')"
    )
    conn.execute(
        "INSERT INTO proposal_attempts(commit_id, thread_id, attempted_at, outcome,"
        " new_objects, assertions, inquiries, omitted_assertions, omitted_inquiries,"
        " omitted_resolutions, resolved_inquiries, evidence_missing_assertions, error, delta_json)"
        " VALUES ('legacy-rejected', 'thread', '2026-08-09T00:01:00+00:00', 'rejected',"
        " 1, 0, 0, 0, 0, 0, 0, 0, 'bad id', '{}')"
    )
    _drop_v10_artifacts(conn)
    conn.execute("PRAGMA user_version = 6")
    conn.commit()

    initialize(conn)
    initialize(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    rows = conn.execute(
        "SELECT attempt_id, run_commit_id, durable_commit_id, attempt_no, outcome "
        "FROM proposal_attempts ORDER BY run_commit_id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("legacy-accepted:legacy:1", "legacy-accepted", "legacy-accepted", 1, "committed"),
        ("legacy-rejected:legacy:1", "legacy-rejected", None, 1, "rejected"),
    ]


def test_attempt_ledger_rebuild_commits_user_version_7_in_the_same_transaction(tmp_path):
    """The direct rebuild cannot leave a new table behind with user_version still 6."""
    conn = sqlite3.connect(tmp_path / "synthetic-atomic-user-version.sqlite3")
    initialize(conn)
    conn.execute(
        "INSERT INTO proposal_attempts(commit_id, thread_id, attempted_at, outcome,"
        " new_objects, assertions, inquiries, omitted_assertions, omitted_inquiries,"
        " omitted_resolutions, resolved_inquiries, evidence_missing_assertions, error, delta_json)"
        " VALUES ('legacy', 'thread', '2026-08-09T00:00:00+00:00', 'committed',"
        " 0, 0, 0, 0, 0, 0, 0, 0, NULL, '{}')"
    )
    conn.execute("PRAGMA user_version = 6")
    conn.commit()

    _apply_migration_v7_attempt_ledger(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 7
    assert "attempt_id" in [row[1] for row in conn.execute("PRAGMA table_info(proposal_attempts)")]


def test_initialize_migrates_v7_to_v8_inquiry_touch_ledger(tmp_path):
    conn = sqlite3.connect(tmp_path / "v7-to-v8.sqlite3")
    initialize(conn)
    conn.execute("DROP INDEX idx_inquiry_attempt_touches_inquiry")
    conn.execute("DROP TABLE inquiry_attempt_touches")
    _drop_v10_artifacts(conn)
    conn.execute("PRAGMA user_version = 7")
    conn.commit()

    initialize(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    columns = [row[1] for row in conn.execute("PRAGMA table_info(inquiry_attempt_touches)")]
    assert columns == ["run_commit_id", "inquiry_id", "touched_at"]
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name='idx_inquiry_attempt_touches_inquiry'"
    ).fetchone() is not None


def test_v8_touch_ledger_migration_is_atomic_on_failure(tmp_path):
    conn = sqlite3.connect(tmp_path / "v8-rollback.sqlite3")
    initialize(conn)
    conn.execute("DROP INDEX idx_inquiry_attempt_touches_inquiry")
    conn.execute("DROP TABLE inquiry_attempt_touches")
    conn.execute("CREATE INDEX idx_inquiry_attempt_touches_inquiry ON inquiries(id)")
    conn.execute("PRAGMA user_version = 7")
    conn.commit()

    with pytest.raises(sqlite3.OperationalError, match="index is incompatible"):
        _apply_migration_v8_inquiry_touch_ledger(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 7
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='inquiry_attempt_touches'"
    ).fetchone() is None


def test_v8_touch_ledger_rejects_an_incompatible_existing_table(tmp_path):
    conn = sqlite3.connect(tmp_path / "v8-incompatible-table.sqlite3")
    initialize(conn)
    conn.execute("DROP INDEX idx_inquiry_attempt_touches_inquiry")
    conn.execute("DROP TABLE inquiry_attempt_touches")
    conn.execute(
        "CREATE TABLE inquiry_attempt_touches "
        "(run_commit_id TEXT PRIMARY KEY, inquiry_id TEXT, touched_at TEXT)"
    )
    conn.execute("PRAGMA user_version = 7")
    conn.commit()

    with pytest.raises(sqlite3.OperationalError, match="schema is incompatible"):
        _apply_migration_v8_inquiry_touch_ledger(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 7
    columns = [row[1] for row in conn.execute("PRAGMA table_info(inquiry_attempt_touches)")]
    assert columns == ["run_commit_id", "inquiry_id", "touched_at"]
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name='idx_inquiry_attempt_touches_inquiry'"
    ).fetchone() is None


def test_attempt_ledger_migration_preserves_synthetic_world_snapshot(tmp_path):
    """Only proposal_attempts changes when a synthetic user_version 6 DB upgrades."""
    path = tmp_path / "synthetic-snapshot.sqlite3"
    store = WorldStore(path)
    now = datetime(2026, 8, 9, tzinfo=UTC)
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="obs-1",
                    source_uri="https://example.test/one",
                    source_kind="web",
                    depth=ObservationDepth.CONTENT,
                    observed_at=now,
                )
            ],
            objects=[
                ObjectInput(id="object-1", kind=ObjectKind.EVENT, canonical_name="One")
            ],
            assertions=[
                AssertionInput(
                    id="assertion-1",
                    subject_id="object-1",
                    predicate="has_result",
                    literal="1-0",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                )
            ],
            inquiries=[
                InquiryInput(
                    id="inquiry-1",
                    subject_id="object-1",
                    prompt="What happened next?",
                    rationale="Synthetic snapshot coverage.",
                )
            ],
        ),
        "snapshot-seed",
    )
    conn = sqlite3.connect(path)
    tables = ("objects", "assertions", "inquiries", "observations", "world_audit", "commit_receipts")
    before = {
        table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        for table in tables
    }
    _drop_v10_artifacts(conn)
    conn.execute("PRAGMA user_version = 6")
    conn.commit()

    initialize(conn)

    after = {
        table: [
            tuple(row)
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        ]
        for table in tables
    }
    assert after == before


def test_attempt_ledger_migration_rolls_back_on_synthetic_failure(tmp_path):
    """A failed synthetic user_version 6 rebuild leaves its source ledger untouched."""
    conn = sqlite3.connect(tmp_path / "synthetic-rollback.sqlite3")
    initialize(conn)
    conn.execute(
        "INSERT INTO proposal_attempts(commit_id, thread_id, attempted_at, outcome,"
        " new_objects, assertions, inquiries, omitted_assertions, omitted_inquiries,"
        " omitted_resolutions, resolved_inquiries, evidence_missing_assertions, error, delta_json)"
        " VALUES ('legacy', 'thread', '2026-08-09T00:00:00+00:00', 'committed',"
        " 0, 0, 0, 0, 0, 0, 0, 0, NULL, '{}')"
    )
    conn.execute("CREATE TABLE proposal_attempts_v7(blocker TEXT)")
    conn.execute("PRAGMA user_version = 6")
    conn.commit()

    with pytest.raises(sqlite3.OperationalError):
        initialize(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
    assert [
        tuple(row)
        for row in conn.execute("SELECT commit_id, outcome FROM proposal_attempts").fetchall()
    ] == [("legacy", "committed")]

def test_v3_migration_is_idempotent(tmp_path):
    conn = _v2_database(tmp_path)
    initialize(conn)
    initialize(conn)  # the guarded ALTER must no-op on the second run
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    columns = [r[1] for r in conn.execute("PRAGMA table_info(proposal_attempts)")]
    assert columns.count("evidence_missing_assertions") == 1


def test_v4_migration_creates_observation_bodies_and_index(tmp_path):
    conn = _v1_database(tmp_path)  # existing helper — builds user_version 0 schema
    initialize(conn)  # runs the historical migrations and the P1 ledger migration
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    cols = [r[1] for r in conn.execute("PRAGMA table_info(observation_bodies)")]
    assert cols == ["observation_id", "content_type", "body_json", "size_bytes"]
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_assertions_subject'"
    ).fetchone() is not None


def test_v4_migration_preserves_existing_data_and_is_idempotent(tmp_path):
    conn = _v1_database(tmp_path)
    initialize(conn)
    initialize(conn)  # second run must be a no-op
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assert conn.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='observation_bodies'"
    ).fetchone()[0] == 1


def test_observation_bodies_content_type_check(tmp_path):
    """The CHECK on content_type must reject invalid values on its own.

    observation_bodies.observation_id is a foreign key to observations, and
    this connection has PRAGMA foreign_keys = ON, so inserting a body for a
    nonexistent observation would raise IntegrityError from the FK even if
    the CHECK were dropped. Seed a valid observation first so only the CHECK
    can reject the invalid-content_type insert.
    """
    conn = _v1_database(tmp_path)
    initialize(conn)
    conn.execute(
        "INSERT INTO observations"
        " (id, source_uri, source_kind, title, excerpt, content_ref, depth,"
        "  source_published_at, observed_at, metadata_json)"
        " VALUES ('o1', 'https://example.com/obs/1', 'bilibili', 'title',"
        "         'excerpt', 'ref://1', 'medium', '2026-08-01T00:00:00+00:00',"
        "         '2026-08-01T01:00:00+00:00', '{}')"
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO observation_bodies VALUES ('o1', 'nonsense', '{}', 10)")


def test_v5_migration_creates_supersedes_index(tmp_path):
    """Fresh initialize applies the v5 supersedes index, user_version=6."""
    conn = sqlite3.connect(tmp_path / "fresh-v5.sqlite3")
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
        " AND name='idx_assertions_supersedes'"
    ).fetchone() is not None


def test_v5_migration_applies_to_existing_v3_database(tmp_path):
    """A database left at user_version=3 (pre-v5) picks up the index on initialize.

    Mirrors the accumulation DBs' state today: schema through v4 applied, no
    supersedes index. The guarded v5 step must re-apply it on the next run.
    """
    conn = _v1_database(tmp_path)
    initialize(conn)
    # Rewind to the pre-v5 state: version 3 and no supersedes index.
    conn.execute("DROP INDEX idx_assertions_supersedes")
    _drop_v10_artifacts(conn)
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
        " AND name='idx_assertions_supersedes'"
    ).fetchone() is not None


def test_v5_migration_is_idempotent(tmp_path):
    """Second initialize must no-op: version stays 6, index not duplicated."""
    conn = _v1_database(tmp_path)
    initialize(conn)
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"
        " AND name='idx_assertions_supersedes'"
    ).fetchone()[0] == 1


_V6_INDEXES = ("idx_world_audit_committed_at", "idx_assertions_object")


def _assert_indexes(conn: sqlite3.Connection, names: tuple[str, ...]) -> None:
    for name in names:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name = ?", (name,)
        ).fetchone() is not None, f"missing index {name}"


def test_v6_migration_creates_both_performance_indexes(tmp_path):
    """Fresh initialize applies the v6 indexes: both present, user_version=6."""
    conn = sqlite3.connect(tmp_path / "fresh-v6.sqlite3")
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    _assert_indexes(conn, _V6_INDEXES)


def test_v6_migration_applies_to_existing_v4_database(tmp_path):
    """A database left at user_version=4 (pre-v6) picks up both indexes.

    Mirrors the accumulation DBs' state before this migration: schema through
    v5 applied, no performance indexes. The guarded v6 step must re-apply
    both on the next initialize.
    """
    conn = _v1_database(tmp_path)
    initialize(conn)
    # Rewind to the pre-v6 state: version 4 and no performance indexes.
    conn.execute("DROP INDEX idx_world_audit_committed_at")
    conn.execute("DROP INDEX idx_assertions_object")
    _drop_v10_artifacts(conn)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    _assert_indexes(conn, _V6_INDEXES)


def test_v6_migration_is_idempotent(tmp_path):
    """Second initialize must no-op: version stays 6, indexes not duplicated."""
    conn = _v1_database(tmp_path)
    initialize(conn)
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    for name in _V6_INDEXES:
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name = ?", (name,)
        ).fetchone()[0] == 1


def test_v7_migration_adds_superseded_at_and_current_assertions_view(tmp_path):
    """Fresh initialize lands on v7: superseded_at column and the current_assertions view."""
    conn = sqlite3.connect(tmp_path / "fresh-v7.sqlite3")
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    columns = [r[1] for r in conn.execute("PRAGMA table_info(assertions)")]
    assert "superseded_at" in columns
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='current_assertions'"
    ).fetchone() is not None


def test_v7_migration_applies_to_existing_v6_database(tmp_path):
    """A database left at user_version=5 (pre-v7) picks up the column and view.

    Mirrors the accumulation DBs' state before this migration: schema through
    v6 applied, no superseded_at column, no current_assertions view. The
    guarded v7 step must add both on the next initialize.
    """
    conn = _v1_database(tmp_path)
    initialize(conn)
    # Rewind to the pre-v7 state: version 5, no superseded_at, no view.
    conn.execute("DROP VIEW current_assertions")
    conn.execute("ALTER TABLE assertions DROP COLUMN superseded_at")
    _drop_v10_artifacts(conn)
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    columns = [r[1] for r in conn.execute("PRAGMA table_info(assertions)")]
    assert "superseded_at" in columns
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='current_assertions'"
    ).fetchone() is not None


def test_v7_migration_is_idempotent(tmp_path):
    """Second initialize must no-op: version stays 6, column and view not duplicated."""
    conn = _v1_database(tmp_path)
    initialize(conn)
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    columns = [r[1] for r in conn.execute("PRAGMA table_info(assertions)")]
    assert columns.count("superseded_at") == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name='current_assertions'"
    ).fetchone()[0] == 1


def test_current_assertions_view_filters_superseded(tmp_path):
    """The view exposes only assertions no newer row supersedes, history intact."""
    conn = _v1_database(tmp_path)
    initialize(conn)
    conn.execute(
        "INSERT INTO objects (id, kind, canonical_name, domain_hints_json, provisional, version)"
        " VALUES ('obj-1', 'entity', 'Alpha', '[]', 0, 1)"
    )
    conn.execute(
        "INSERT INTO assertions (id, signature, subject_id, predicate, object_id, literal_json,"
        " epistemic_role, confidence, event_time_start, event_time_end, supersedes_id, superseded_at)"
        " VALUES ('a-1', 'sig-1', 'obj-1', 'related_to', NULL, '\"old\"', 'fact', 0.5,"
        "         NULL, NULL, NULL, NULL)"
    )
    conn.execute(
        "INSERT INTO assertions (id, signature, subject_id, predicate, object_id, literal_json,"
        " epistemic_role, confidence, event_time_start, event_time_end, supersedes_id, superseded_at)"
        " VALUES ('a-2', 'sig-2', 'obj-1', 'related_to', NULL, '\"new\"', 'fact', 0.9,"
        "         NULL, NULL, 'a-1', '2026-08-06T00:00:00+00:00')"
    )
    conn.commit()
    # the superseded row is retired from the current view but stays in history
    rows = [row[0] for row in conn.execute("SELECT id FROM current_assertions ORDER BY id")]
    assert rows == ["a-2"]
    assert conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 2
    # the superseding row carries the correction-history timestamp
    row = conn.execute(
        "SELECT superseded_at FROM current_assertions WHERE id = 'a-2'"
    ).fetchone()
    assert row[0] == "2026-08-06T00:00:00+00:00"


def test_current_assertions_view_rewrite_matches_not_exists_semantics(tmp_path):
    """Closeout correction: the NOT IN view is semantically identical to the
    retired correlated NOT EXISTS definition on real candidate paths.

    The rewrite is only a planner change (materialize the supersedes set
    once instead of re-scanning it per outer row). This test pins the
    equivalence on the two production consumers — the strong event candidate
    query (with type/time authority) and the partial candidate query
    (without) — against a literal NOT EXISTS oracle, including the
    in-window / superseded-chain distinctions the naive rewrite could break:
    a superseded row must drop out of candidates, a superseding row whose
    participant changed must not leak into the old participant's candidate
    set, and unrelated supersedes chains must not suppress their targets.
    """
    conn = sqlite3.connect(tmp_path / "equiv.sqlite3")
    initialize(conn)
    conn.execute("INSERT OR IGNORE INTO predicates(name) VALUES ('has_participant')")
    window = ("2026-08-01T00:00:00+00:00", "2026-08-02T00:00:00+00:00")
    outside = ("2026-07-01T00:00:00+00:00", "2026-07-02T00:00:00+00:00")
    for oid, kind, tkey, bounds in (
        ("evt-1", "event", "match", window),
        ("evt-2", "event", "match", window),
        ("evt-retired", "event", "match", window),
        ("evt-outside", "event", "match", outside),
        ("ent-1", "entity", None, (None, None)),
        ("ent-2", "entity", None, (None, None)),
    ):
        conn.execute(
            "INSERT INTO objects (id, kind, canonical_name, domain_hints_json,"
            " provisional, event_time_start, event_time_end, version, type_key)"
            " VALUES (?, ?, ?, '[]', 0, ?, ?, 1, ?)",
            (oid, kind, oid, *bounds, tkey),
        )
    # a-1/a-2: current participants of evt-1/evt-2; a-3: evt-retired's old
    # participant claim, superseded by a-4 whose participant changed to
    # ent-2 — so evt-retired must vanish from the ent-1 candidate set;
    # a-5: an out-of-window participant (partial keeps it, strong drops it);
    # a-6/a-7: an unrelated supersedes chain that must not suppress its
    # target from the view.
    for aid, sig, subject, pred, object_id, supersedes in (
        ("a-1", "sig-1", "evt-1", "has_participant", "ent-1", None),
        ("a-2", "sig-2", "evt-2", "has_participant", "ent-1", None),
        ("a-3", "sig-3", "evt-retired", "has_participant", "ent-1", None),
        ("a-4", "sig-4", "evt-retired", "has_participant", "ent-2", "a-3"),
        ("a-5", "sig-5", "evt-outside", "has_participant", "ent-1", None),
        ("a-6", "sig-6", "ent-1", "related_to", "ent-2", None),
        ("a-7", "sig-7", "ent-1", "related_to", "ent-2", "a-6"),
    ):
        conn.execute(
            "INSERT INTO assertions (id, signature, subject_id, predicate, object_id,"
            " literal_json, epistemic_role, confidence, event_time_start, event_time_end,"
            " supersedes_id, superseded_at, qualifiers_json)"
            " VALUES (?, ?, ?, ?, ?, NULL, 'fact', 0.5, NULL, NULL, ?,"
            " CASE WHEN ? IS NULL THEN NULL ELSE '2026-08-06T00:00:00+00:00' END, NULL)",
            (aid, sig, subject, pred, object_id, supersedes, supersedes),
        )
    conn.commit()

    def run(sql: str, values: list[object]) -> list[str]:
        return [str(row[0]) for row in conn.execute(sql, values).fetchall()]

    # View-level oracle: the retired NOT EXISTS definition the view replaced.
    oracle_view = (
        "SELECT a.id FROM assertions a"
        " WHERE NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)"
        " ORDER BY a.id"
    )
    assert run(oracle_view, []) == run("SELECT id FROM current_assertions ORDER BY id", [])

    # Strong path with a time window, mirrored with the literal oracle.
    window_start, window_end = (
        "2026-08-01T01:00:00+00:00",
        "2026-08-01T23:00:00+00:00",
    )
    strong_sql = (
        "SELECT DISTINCT a.subject_id AS event_id FROM assertions AS a"
        " JOIN objects AS o ON o.id = a.subject_id"
        " WHERE NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)"
        " AND a.predicate = 'has_participant' AND a.object_id IN (?)"
        " AND (o.event_time_end IS NULL OR o.event_time_end >= ?)"
        " AND (o.event_time_start IS NULL OR o.event_time_start <= ?)"
        " AND o.kind = ? AND o.provisional = 0 AND o.id != ? AND o.type_key = ?"
        " ORDER BY a.subject_id"
    )
    strong_values = ["ent-1", window_start, window_end, "event", "evt-self", "match"]
    oracle_strong = run(strong_sql, strong_values)
    view_strong = run(
        "SELECT DISTINCT a.subject_id AS event_id FROM current_assertions AS a"
        " JOIN objects AS o ON o.id = a.subject_id"
        " WHERE a.predicate = 'has_participant' AND a.object_id IN (?)"
        " AND (o.event_time_end IS NULL OR o.event_time_end >= ?)"
        " AND (o.event_time_start IS NULL OR o.event_time_start <= ?)"
        " AND o.kind = ? AND o.provisional = 0 AND o.id != ? AND o.type_key = ?"
        " ORDER BY a.subject_id",
        strong_values,
    )
    assert view_strong == oracle_strong
    # window keeps evt-1/evt-2; evt-outside is dropped by the window
    # predicates; evt-retired vanishes because its superseding row changed
    # participants (view must not leak the old participant claim)
    assert view_strong == ["evt-1", "evt-2"]

    # Partial path (no type/time authority, LIMIT 9), mirrored likewise.
    partial_sql = (
        "SELECT DISTINCT a.subject_id AS event_id FROM assertions AS a"
        " WHERE NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)"
        " AND a.predicate = 'has_participant' AND a.object_id IN (?)"
        " AND a.subject_id != ? ORDER BY a.subject_id LIMIT ?"
    )
    partial_values = ["ent-1", "evt-self", 9]
    oracle_partial = run(partial_sql, partial_values)
    view_partial = run(
        "SELECT DISTINCT a.subject_id AS event_id FROM current_assertions AS a"
        " WHERE a.predicate = 'has_participant' AND a.object_id IN (?)"
        " AND a.subject_id != ? ORDER BY a.subject_id LIMIT ?",
        partial_values,
    )
    assert view_partial == oracle_partial
    # partial has no window authority: evt-outside stays, but the superseded
    # participant claim is still gone
    assert view_partial == ["evt-1", "evt-2", "evt-outside"]


def _v8_database(tmp_path) -> sqlite3.Connection:
    """Build a synthetic v8-era database (user_version=8, no v9 columns).

    The v9 migration touches only the assertions/objects/inquiries columns, so
    this minimal v8 shape (the v1 tables plus a representative legacy row per
    table, with the version pinned to 8 so no earlier migration re-runs) is
    the faithful fixture for the v9 ALTERs and their backfill guarantees.
    """
    conn = sqlite3.connect(tmp_path / "v8.sqlite3")
    conn.executescript(SCHEMA_V1_SQL)
    conn.execute("PRAGMA user_version = 8")
    conn.execute(
        "INSERT INTO objects(id, kind, canonical_name, domain_hints_json, provisional,"
        " event_time_start, event_time_end, version)"
        " VALUES ('obj-1', 'entity', 'Bin', '[]', 0, NULL, NULL, 1)"
    )
    conn.execute("INSERT INTO predicates(name) VALUES ('has_status')")
    conn.execute(
        "INSERT INTO assertions(id, signature, subject_id, predicate, object_id, literal_json,"
        " epistemic_role, confidence, event_time_start, event_time_end, supersedes_id)"
        " VALUES ('assertion-1', 'sig-1', 'obj-1', 'has_status', NULL, '\"playing\"', 'fact',"
        " 0.9, NULL, NULL, NULL)"
    )
    conn.execute(
        "INSERT INTO inquiries(id, subject_id, prompt, rationale, status, version)"
        " VALUES ('inq-1', 'obj-1', 'old prompt', 'old rationale', 'open', 1)"
    )
    conn.commit()
    return conn


def _v9_database(tmp_path) -> sqlite3.Connection:
    """Build a real v9-schema database by driving each migration step directly.

    ``initialize`` applies the whole chain through v11 in one pass, so the v9
    state is produced by calling the migration steps in the same order as
    ``initialize``'s dispatch, pinning the intermediate user_version after
    every step that does not set it itself, and committing no v10 artifacts.
    """
    conn = sqlite3.connect(tmp_path / "v9.sqlite3")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_V1_SQL)
    _apply_migration_v2(conn)
    conn.execute("PRAGMA user_version = 1")
    _apply_migration_v3(conn)
    conn.execute("PRAGMA user_version = 2")
    conn.executescript(MIGRATION_V4_SQL)
    conn.execute("PRAGMA user_version = 3")
    conn.executescript(MIGRATION_V5_SQL)
    conn.execute("PRAGMA user_version = 4")
    conn.executescript(MIGRATION_V6_SQL)
    conn.execute("PRAGMA user_version = 5")
    _apply_migration_v7(conn)
    conn.execute("PRAGMA user_version = 6")
    _apply_migration_v7_attempt_ledger(conn)
    _apply_migration_v8_inquiry_touch_ledger(conn)
    _apply_migration_v9(conn)
    conn.commit()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 9
    return conn


def _drop_v10_artifacts(conn: sqlite3.Connection) -> None:
    """Undo the v10 migration so a user_version rewind emulates a pre-v9 DB.

    The v10 DDL is deliberately plain CREATE TABLE (no IF NOT EXISTS), so
    re-running initialize from a rewound user_version collides if the v10
    artifacts are still present. A real pre-v9 database never had them, so the
    rewind tests remove them alongside the artifact under test.
    """
    conn.execute("DROP TABLE identity_aliases")
    conn.execute("ALTER TABLE objects DROP COLUMN type_key")
    conn.execute("ALTER TABLE assertions DROP COLUMN qualifiers_json")


def test_initialize_migrates_v8_to_v9(tmp_path):
    """v9 adds event_time_precision, supersede_reason and inquiry closed support."""
    conn = _v8_database(tmp_path)
    initialize(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assertion_columns = {r[1] for r in conn.execute("PRAGMA table_info(assertions)")}
    object_columns = {r[1] for r in conn.execute("PRAGMA table_info(objects)")}
    inquiry_columns = {r[1] for r in conn.execute("PRAGMA table_info(inquiries)")}
    assert "event_time_precision" in assertion_columns
    assert "supersede_reason" in assertion_columns
    assert "event_time_precision" in object_columns
    assert "closed_reason" in inquiry_columns
    # legacy rows backfill: NOT NULL DEFAULT 'unknown', supersede_reason stays NULL
    row = conn.execute(
        "SELECT event_time_precision, supersede_reason FROM assertions WHERE id='assertion-1'"
    ).fetchone()
    assert row["event_time_precision"] == "unknown"
    assert row["supersede_reason"] is None
    row = conn.execute(
        "SELECT event_time_precision FROM objects WHERE id='obj-1'"
    ).fetchone()
    assert row["event_time_precision"] == "unknown"
    assert conn.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0] == 1


def test_v9_migration_is_idempotent(tmp_path):
    """A second initialize run must not fail or duplicate any v9 artifact."""
    conn = _v8_database(tmp_path)
    initialize(conn)
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assertion_columns = {r[1] for r in conn.execute("PRAGMA table_info(assertions)")}
    assert "event_time_precision" in assertion_columns
    assert conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 1


def test_v8_store_read_only_open_never_migrates(tmp_path):
    """Read-only opens (baseline/verify scripts) must never add v9 columns."""
    conn = _v8_database(tmp_path)
    conn.close()
    store = WorldStore(tmp_path / "v8.sqlite3", initialize_schema=False)
    with store.read_connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 8
        assert "event_time_precision" not in [
            row[1] for row in connection.execute("PRAGMA table_info(assertions)")
        ]
    # a truly read-only connection rejects the v9 ALTER outright
    with sqlite3.connect(f"file:{tmp_path / 'v8.sqlite3'}?mode=ro", uri=True) as ro, pytest.raises(
        sqlite3.OperationalError
    ):
        ro.execute("ALTER TABLE assertions ADD COLUMN event_time_precision TEXT")


def test_initialize_fresh_database_reaches_v11(tmp_path):
    """Fresh initialize lands on v11: graph-contract columns, identity_aliases, NOT IN view."""
    conn = sqlite3.connect(tmp_path / "fresh-v11.sqlite3")
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    object_columns = {r[1] for r in conn.execute("PRAGMA table_info(objects)")}
    assertion_columns = {r[1] for r in conn.execute("PRAGMA table_info(assertions)")}
    assert "type_key" in object_columns
    assert "qualifiers_json" in assertion_columns
    columns = [r[1] for r in conn.execute("PRAGMA table_info(identity_aliases)")]
    assert columns == [
        "object_id",
        "raw_alias",
        "normalized_alias",
        "status",
        "added_commit_id",
        "removed_commit_id",
    ]
    for name in (
        "idx_identity_aliases_active_normalized",
        "idx_identity_aliases_object",
    ):
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name = ?", (name,)
        ).fetchone() is not None, f"missing index {name}"
    # fresh databases land directly on the NOT IN view definition
    view_text = _view_sql(conn)
    assert "NOT IN" in view_text
    assert "NOT EXISTS" not in view_text


def test_v9_to_v11_adds_columns_and_identity_alias_table_without_copying_legacy_rows(tmp_path):
    """The v10 step adds both columns and identity_aliases (chain lands on v11)."""
    conn = _v9_database(tmp_path)
    conn.execute(
        "INSERT INTO objects (id, kind, canonical_name, domain_hints_json, provisional, version)"
        " VALUES ('obj-1', 'entity', 'Alpha', '[]', 0, 1)"
    )
    conn.execute(
        "INSERT INTO object_aliases (normalized_alias, object_id) VALUES ('alpha', 'obj-1')"
    )
    conn.commit()

    initialize(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    object_columns = {r[1] for r in conn.execute("PRAGMA table_info(objects)")}
    assertion_columns = {r[1] for r in conn.execute("PRAGMA table_info(assertions)")}
    assert "type_key" in object_columns
    assert "qualifiers_json" in assertion_columns
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='identity_aliases'"
    ).fetchone() is not None
    # the legacy alias row survives in its old table and is NOT backfilled:
    # identity_aliases starts empty (no copy/backfill on migration)
    assert conn.execute("SELECT COUNT(*) FROM object_aliases").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM identity_aliases").fetchone()[0] == 0


def test_v11_initialize_is_idempotent(tmp_path):
    """A second initialize run sees user_version == 11 and skips the v11 step."""
    conn = _v9_database(tmp_path)
    initialize(conn)
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    assert conn.execute("SELECT COUNT(*) FROM identity_aliases").fetchone()[0] == 0
    object_columns = [r[1] for r in conn.execute("PRAGMA table_info(objects)")]
    assert object_columns.count("type_key") == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='identity_aliases'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
        "AND name='idx_identity_aliases_active_normalized'"
    ).fetchone()[0] == 1
    # a second initialize sees user_version 11 and skips the v11 step
    # (idempotent by skip, not rebuild); the view is untouched
    assert "NOT IN" in _view_sql(conn)


def test_v10_migration_rolls_back_columns_and_version_on_mid_migration_failure(tmp_path):
    """A real v9 DB with a blocking identity_aliases view rolls back the whole step.

    The migration fails at CREATE TABLE identity_aliases — AFTER both ALTERs —
    and the single transaction must undo the ALTERs and keep user_version 9.
    """
    conn = _v9_database(tmp_path)
    conn.execute(
        "INSERT INTO objects (id, kind, canonical_name, domain_hints_json, provisional, version)"
        " VALUES ('obj-1', 'entity', 'Alpha', '[]', 0, 1)"
    )
    conn.execute(
        "INSERT INTO object_aliases (normalized_alias, object_id) VALUES ('alpha', 'obj-1')"
    )
    conn.execute("CREATE VIEW identity_aliases AS SELECT id AS object_id FROM objects")
    conn.commit()

    with pytest.raises(sqlite3.OperationalError, match="identity_aliases already exists"):
        initialize(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 9
    object_columns = {r[1] for r in conn.execute("PRAGMA table_info(objects)")}
    assertion_columns = {r[1] for r in conn.execute("PRAGMA table_info(assertions)")}
    assert "type_key" not in object_columns
    assert "qualifiers_json" not in assertion_columns
    # the blocker view and the legacy tables/rows are untouched
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='identity_aliases'"
    ).fetchone() is not None
    assert conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM object_aliases").fetchone()[0] == 1


# The v7 step only runs at user_version 5, so databases created before the
# NOT IN rewrite keep the correlated-NOT-EXISTS view forever unless the v11
# migration rebuilds it. This is the pre-rewrite definition, byte-faithful at
# commit 8621886.
_OLD_NOT_EXISTS_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS current_assertions AS
SELECT a.* FROM assertions a
WHERE NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id);
"""


def _view_sql(conn: sqlite3.Connection) -> str:
    """Return the ``current_assertions`` view definition text of *conn*."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='view' AND name='current_assertions'"
    ).fetchone()
    assert row is not None, "current_assertions view missing"
    return str(row[0])


def _v10_old_view_database(tmp_path) -> sqlite3.Connection:
    """Build a v10 database carrying the pre-rewrite NOT EXISTS view.

    The real pre-fix population is exactly this: full v10 schema, the
    correlated-NOT-EXISTS view text, and user_version 10 — initialize() must
    have no step that touches the view for this state to persist.
    """
    conn = sqlite3.connect(tmp_path / "v10-old-view.sqlite3")
    conn.row_factory = sqlite3.Row
    initialize(conn)
    conn.execute("DROP VIEW current_assertions")
    conn.execute(_OLD_NOT_EXISTS_VIEW_SQL)
    conn.execute("PRAGMA user_version = 10")
    conn.commit()
    assert "NOT EXISTS" in _view_sql(conn)
    return conn


def test_v10_old_view_upgrades_to_v11_with_not_in(tmp_path):
    """A v10 DB with the pre-rewrite view gets the NOT IN view via the v11 step."""
    conn = _v10_old_view_database(tmp_path)
    initialize(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
    view_text = _view_sql(conn)
    assert "NOT IN" in view_text
    assert "NOT EXISTS" not in view_text


def test_v11_migration_rolls_back_version_and_old_view_on_failure(tmp_path, monkeypatch):
    """A failed v11 rebuild keeps user_version 10 AND the old view text together."""
    conn = _v10_old_view_database(tmp_path)
    from leave_information_bubble.world import schema as schema_module

    monkeypatch.setattr(schema_module, "MIGRATION_V7_SQL", "CREATE VIEW current_assertions AS (")
    with pytest.raises(sqlite3.OperationalError):
        initialize(conn)

    # the whole step rolled back: version AND the previous view text
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 10
    view_text = _view_sql(conn)
    assert "NOT EXISTS" in view_text
    assert "NOT IN" not in view_text
