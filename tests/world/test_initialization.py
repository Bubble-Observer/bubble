"""Blank long-term database initialization and isolated canary preparation tests."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from leave_information_bubble.world import WorldStore
from leave_information_bubble.world.initialization import (
    RUNTIME_TABLES,
    WORLD_TABLES,
    InitializationError,
    init_blank_database_pair,
    initialize_blank_world,
    prepare_contract_canary,
)

_CORE_WORLD_TABLES = frozenset(
    {
        "objects",
        "observations",
        "assertions",
        "inquiries",
        "commit_receipts",
        "world_audit",
        "identity_aliases",
        "graph_shell_wake_claims",
        "graph_shell_writer_leases",
    }
)


def _user_version(path: Path) -> int:
    """Return the SQLite user_version of *path*."""
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


def _table_names(path: Path) -> set[str]:
    """Return the user table names present in the SQLite master of *path*."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        return {str(row[0]) for row in rows}
    finally:
        connection.close()


def _count(path: Path, table: str) -> int:
    """Return the row count of *table* in *path*."""
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def _predicate_names(path: Path) -> set[str]:
    """Return every seeded or inserted predicate name of *path*."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT name FROM predicates").fetchall()
        return {str(row[0]) for row in rows}
    finally:
        connection.close()


def _marker_world(path: Path, marker: str = "x-canary-marker") -> None:
    """Create a small current-version world snapshot carrying one row."""
    initialize_blank_world(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("INSERT INTO predicates(name) VALUES (?)", (marker,))
        connection.commit()
    finally:
        connection.close()


def _alias_under(parent: Path, sub_name: str, name: str) -> Path:
    """Return a relative alias of *parent/name* that walks through *sub_name*."""
    (parent / sub_name).mkdir()
    return parent / sub_name / ".." / name


async def test_blank_initializer_creates_current_world_and_runtime_tables(tmp_path):
    world_path, runtime_path = await init_blank_database_pair(
        tmp_path / "longterm.sqlite3", tmp_path / "runtime.sqlite3"
    )

    assert world_path == (tmp_path / "longterm.sqlite3").resolve()
    assert runtime_path == (tmp_path / "runtime.sqlite3").resolve()
    assert world_path != runtime_path

    # the world went through the official schema.initialize path: v17 with
    # the identity aliases, Graph Shell wake-claim ledger, singleton writer
    # lease table and no data rows
    assert _user_version(world_path) == 18
    assert _table_names(world_path) >= _CORE_WORLD_TABLES
    for table in (
        "objects",
        "assertions",
        "inquiries",
        "graph_shell_wake_claims",
        "graph_shell_writer_leases",
    ):
        assert _count(world_path, table) == 0

    # the runtime is a real database with every required table, never a
    # zero-byte fake
    assert runtime_path.stat().st_size > 0
    assert set(RUNTIME_TABLES) <= _table_names(runtime_path)
    for table in RUNTIME_TABLES:
        assert _count(runtime_path, table) == 0


async def test_blank_initializer_refuses_existing_targets_without_partial_output(tmp_path):
    world = tmp_path / "world.sqlite3"
    runtime = tmp_path / "runtime.sqlite3"
    original_world = b"pre-existing world database"
    world.write_bytes(original_world)

    with pytest.raises(InitializationError) as error:
        await init_blank_database_pair(world, runtime)
    assert str(world.resolve()) in str(error.value)
    assert not runtime.exists()  # no partial output
    assert world.read_bytes() == original_world  # untouched

    runtime.write_bytes(b"pre-existing runtime database")
    fresh = tmp_path / "fresh.sqlite3"
    with pytest.raises(InitializationError):
        await init_blank_database_pair(fresh, runtime)
    assert not fresh.exists()
    assert runtime.read_bytes() == b"pre-existing runtime database"

    with pytest.raises(InitializationError) as error:
        await init_blank_database_pair(world, runtime)
    message = str(error.value)
    assert str(world.resolve()) in message
    assert str(runtime.resolve()) in message

    # a relative alias that resolves onto an existing target is refused too
    alias = _alias_under(tmp_path, "sub", "world.sqlite3")
    other = tmp_path / "other.sqlite3"
    with pytest.raises(InitializationError):
        await init_blank_database_pair(alias, other)
    assert not other.exists()
    assert world.read_bytes() == original_world

    # the two targets must be different files even when both are new
    with pytest.raises(InitializationError):
        await init_blank_database_pair(tmp_path / "same.sqlite3", tmp_path / "same.sqlite3")
    assert not (tmp_path / "same.sqlite3").exists()


async def test_canary_preparer_clones_world_but_starts_with_empty_runtime(tmp_path):
    source = tmp_path / "snapshot.sqlite3"
    _marker_world(source, marker="x-canary-marker")
    source_bytes = source.read_bytes()

    world_dest, runtime_dest = await prepare_contract_canary(
        source, tmp_path / "canary" / "world.sqlite3", tmp_path / "canary" / "runtime.sqlite3"
    )

    # byte-identical clone of the snapshot: schema, version, and the marker row
    assert world_dest.read_bytes() == source_bytes
    assert _user_version(world_dest) == 18
    assert "identity_aliases" in _table_names(world_dest)
    assert "x-canary-marker" in _predicate_names(world_dest)

    # empty runtime with every required table
    assert runtime_dest.stat().st_size > 0
    assert set(RUNTIME_TABLES) <= _table_names(runtime_dest)

    # the source snapshot is untouched
    assert source.read_bytes() == source_bytes


# The v7 step only runs at user_version 5, so a snapshot created before the
# NOT IN rewrite keeps the correlated-NOT-EXISTS view at v10. Byte-faithful
# pre-rewrite definition (commit 8621886); the v11 migration rebuilds it.
_OLD_NOT_EXISTS_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS current_assertions AS
SELECT a.* FROM assertions a
WHERE NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id);
"""


def _rewind_to_v10_old_view(path: Path) -> None:
    """Turn *path* into a pre-rewrite v10 snapshot: full schema, old view text."""
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP VIEW current_assertions")
        connection.execute(_OLD_NOT_EXISTS_VIEW_SQL)
        connection.execute("PRAGMA user_version = 10")
        connection.commit()
    finally:
        connection.close()


async def test_canary_clone_rebuilds_old_view_through_production_open(tmp_path):
    """A v10 old-view snapshot cloned by the canary gets the NOT IN view on open.

    ``prepare_contract_canary`` byte-copies the snapshot, so the view text
    (executable contract) only reaches the clone when the production open
    path runs ``initialize``: the v11 step rebuilds it. The source snapshot
    stays byte-identical.
    """
    source = tmp_path / "snapshot.sqlite3"
    initialize_blank_world(source)
    _rewind_to_v10_old_view(source)
    assert _user_version(source) == 10
    source_bytes = source.read_bytes()

    world_dest, runtime_dest = await prepare_contract_canary(
        source, tmp_path / "canary" / "world.sqlite3", tmp_path / "canary" / "runtime.sqlite3"
    )
    # byte-identical clone of the pre-rewrite snapshot
    assert world_dest.read_bytes() == source_bytes

    # the production open path (WorldStore, initialize_schema=True) applies
    # the full migration chain to the clone
    store = WorldStore(world_dest)
    with store.read_connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 18
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='current_assertions'"
        ).fetchone()
    assert "NOT IN" in row[0]
    assert "NOT EXISTS" not in row[0]

    # the source snapshot is untouched
    assert source.read_bytes() == source_bytes


async def test_canary_preparer_refuses_same_source_and_destination(tmp_path):
    source = tmp_path / "snapshot.sqlite3"
    _marker_world(source)
    source_bytes = source.read_bytes()

    # literal same source and destination
    with pytest.raises(InitializationError):
        await prepare_contract_canary(source, source, tmp_path / "runtime.sqlite3")

    # a relative alias resolving onto the source is refused too
    alias = _alias_under(tmp_path, "sub", "snapshot.sqlite3")
    with pytest.raises(InitializationError):
        await prepare_contract_canary(alias, alias, tmp_path / "runtime.sqlite3")

    # destinations must not exist
    existing = tmp_path / "existing.sqlite3"
    existing.write_bytes(b"pre-existing destination")
    with pytest.raises(InitializationError):
        await prepare_contract_canary(source, existing, tmp_path / "runtime.sqlite3")
    with pytest.raises(InitializationError):
        await prepare_contract_canary(source, tmp_path / "world.sqlite3", existing)

    # world and runtime destinations must stay separate files
    with pytest.raises(InitializationError):
        await prepare_contract_canary(source, tmp_path / "same.sqlite3", tmp_path / "same.sqlite3")

    # nothing was created or modified by any refused attempt
    assert source.read_bytes() == source_bytes
    assert not (tmp_path / "runtime.sqlite3").exists()
    assert not (tmp_path / "world.sqlite3").exists()
    assert not (tmp_path / "same.sqlite3").exists()
    assert existing.read_bytes() == b"pre-existing destination"


async def test_initializers_never_resolve_to_configured_real_database_fixture(tmp_path):
    """A target that resolves onto a configured live database is always refused.

    The fixture mirrors the ops run configuration, which names its live
    databases world_db and runtime_db; the test never references real user
    database paths.
    """
    configured_dir = tmp_path / "configured"
    configured_dir.mkdir()
    world_db = configured_dir / "world.sqlite3"
    runtime_db = configured_dir / "runtime.sqlite3"
    world_bytes = b"configured live world database"
    runtime_bytes = b"configured live runtime database"
    world_db.write_bytes(world_bytes)
    runtime_db.write_bytes(runtime_bytes)
    config_path = configured_dir / "run-config.json"
    config_path.write_text(
        json.dumps({"world_db": str(world_db), "runtime_db": str(runtime_db)}), encoding="utf-8"
    )
    configured = json.loads(config_path.read_text(encoding="utf-8"))
    configured_world = Path(configured["world_db"])
    configured_runtime = Path(configured["runtime_db"])

    # direct configured paths are refused by the blank initializer
    with pytest.raises(InitializationError):
        await init_blank_database_pair(configured_world, tmp_path / "fresh-runtime.sqlite3")
    with pytest.raises(InitializationError):
        await init_blank_database_pair(tmp_path / "fresh-world.sqlite3", configured_runtime)
    assert not (tmp_path / "fresh-runtime.sqlite3").exists()
    assert not (tmp_path / "fresh-world.sqlite3").exists()

    # a relative alias resolving onto a configured path is refused too
    alias = _alias_under(tmp_path, "sub", "configured/world.sqlite3")
    with pytest.raises(InitializationError):
        await init_blank_database_pair(alias, tmp_path / "fresh-runtime.sqlite3")
    assert not (tmp_path / "fresh-runtime.sqlite3").exists()

    # the canary preparer refuses configured paths as destinations
    source = tmp_path / "snapshot.sqlite3"
    _marker_world(source)
    with pytest.raises(InitializationError):
        await prepare_contract_canary(source, configured_world, tmp_path / "canary-runtime.sqlite3")
    with pytest.raises(InitializationError):
        await prepare_contract_canary(source, tmp_path / "canary-world.sqlite3", configured_runtime)
    assert not (tmp_path / "canary-runtime.sqlite3").exists()
    assert not (tmp_path / "canary-world.sqlite3").exists()

    # nothing ever wrote through: every fixture file keeps its original bytes
    assert world_db.read_bytes() == world_bytes
    assert runtime_db.read_bytes() == runtime_bytes


async def test_initializers_refuse_symlink_aliases_of_configured_database_fixture(tmp_path):
    """A symlink alias that resolves onto a configured live database is refused.

    Platform-gated: symlink creation needs privileges most Windows sessions
    lack, so this test skips there; the alias guard itself is covered without
    symlinks by ``test_initializers_never_resolve_to_configured_real_database_fixture``.
    """
    configured_dir = tmp_path / "configured"
    configured_dir.mkdir()
    world_db = configured_dir / "world.sqlite3"
    world_bytes = b"configured live world database"
    world_db.write_bytes(world_bytes)
    link = tmp_path / "link-world.sqlite3"
    try:
        os.symlink(world_db, link)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(InitializationError):
        await init_blank_database_pair(link, tmp_path / "fresh-runtime.sqlite3")
    assert not (tmp_path / "fresh-runtime.sqlite3").exists()
    assert world_db.read_bytes() == world_bytes


async def test_graph_shell_run_creates_no_runtime_tables_outside_registry(tmp_path) -> None:
    """F-08: the fresh-runtime registry is the complete contract for Graph Shell.

    ``RUNTIME_TABLES`` documents every table a fresh runtime must contain
    after initialization. A real Graph Shell CLI wake must therefore need no
    table the registry does not list: the wake-claim ledger is created eagerly
    (not lazily by the CLI) and the checkpointer's ``writes`` table is
    registered alongside ``checkpoints``. Run a full publish cycle on an
    ``init_blank_database_pair`` runtime and prove the registry is a superset
    of what the wake actually used.
    """
    from tests.world_agent.test_vertical_slice import _load_runner, _write_replay_fixture

    world_path, runtime_path = await init_blank_database_pair(
        tmp_path / "contract-world.sqlite3", tmp_path / "contract-runtime.sqlite3"
    )
    replay_path = tmp_path / "contract-replay.json"
    model_path = tmp_path / "contract-model.json"
    _write_replay_fixture(replay_path)
    model_path.write_text(
        json.dumps(
            [
                {
                    "content": "Stage a provisional concept.",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "graph_patch",
                            "arguments": {
                                "items": [
                                    {
                                        "op_id": "op-1",
                                        "kind": "object",
                                        "action": "create",
                                        "payload": {
                                            "canonical_name": "Registry Probe",
                                            "kind": "concept",
                                            "provisional": True,
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                },
                {
                    "content": "Publish the draft.",
                    "tool_calls": [{"id": "c2", "name": "finalize_graph", "arguments": {}}],
                },
            ]
        ),
        encoding="utf-8",
    )
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--perspective",
            "Registry contract probe.",
            "--world-db",
            str(world_path),
            "--runtime-db",
            str(runtime_path),
            "--thread-id",
            "contract-thread",
            "--replay-fixture",
            str(replay_path),
            "--scripted-model-fixture",
            str(model_path),
        ]
    )
    result = await runner.run(args, print_report=False)
    assert result["terminal_status"] == "published"
    # the registry is the complete contract: a real wake invents no table
    assert set(RUNTIME_TABLES) <= _table_names(runtime_path)
    assert {
        name
        for name in _table_names(runtime_path)
        if not name.startswith("sqlite_")
    } <= set(RUNTIME_TABLES)


async def test_graph_shell_run_creates_no_world_tables_outside_registry(tmp_path) -> None:
    """C-1: the WORLD table registry is the complete contract for Graph Shell.

    The writer-lease table must come from the official schema migration, never
    from a lazy ``CREATE TABLE IF NOT EXISTS`` in the runner. A real Graph
    Shell CLI publish cycle therefore needs no world table the registry does
    not list. FTS shadow tables (``objects_fts*`` / ``assertions_fts*``) are
    SQLite-internal machinery, not application tables, and are excluded the
    same way the runtime registry check excludes ``sqlite_*`` names.
    """
    from tests.world_agent.test_vertical_slice import _load_runner, _write_replay_fixture

    world_path, runtime_path = await init_blank_database_pair(
        tmp_path / "world-contract.sqlite3", tmp_path / "runtime-contract.sqlite3"
    )
    replay_path = tmp_path / "world-contract-replay.json"
    model_path = tmp_path / "world-contract-model.json"
    _write_replay_fixture(replay_path)
    model_path.write_text(
        json.dumps(
            [
                {
                    "content": "Stage a provisional concept.",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "graph_patch",
                            "arguments": {
                                "items": [
                                    {
                                        "op_id": "op-1",
                                        "kind": "object",
                                        "action": "create",
                                        "payload": {
                                            "canonical_name": "World Registry Probe",
                                            "kind": "concept",
                                            "provisional": True,
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                },
                {
                    "content": "Publish the draft.",
                    "tool_calls": [{"id": "c2", "name": "finalize_graph", "arguments": {}}],
                },
            ]
        ),
        encoding="utf-8",
    )
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--perspective",
            "World registry contract probe.",
            "--world-db",
            str(world_path),
            "--runtime-db",
            str(runtime_path),
            "--thread-id",
            "world-contract-thread",
            "--replay-fixture",
            str(replay_path),
            "--scripted-model-fixture",
            str(model_path),
        ]
    )
    result = await runner.run(args, print_report=False)
    assert result["terminal_status"] == "published"
    # the world registry is the complete contract: a real wake invents no table
    assert set(WORLD_TABLES) <= _table_names(world_path)
    assert {
        name
        for name in _table_names(world_path)
        if not name.startswith("sqlite_") and not name.endswith("_fts")
        and "_fts_" not in name
    } <= set(WORLD_TABLES)


async def test_legacy_v16_world_auto_migrates_to_v17_writer_lease_on_open(tmp_path) -> None:
    """C-1: pre-v17 worlds gain the writer-lease table through the migration chain.

    A world created before the singleton writer lease existed (v16, no
    ``graph_shell_writer_leases``) must acquire the table the moment a
    production ``WorldStore`` opens it — no manual SQL, no lazy runner-side
    DDL. This keeps old databases operable after a crash window instead of
    depending on the runner to invent the table.
    """
    world_path = tmp_path / "legacy-v16.sqlite3"
    connection = sqlite3.connect(world_path)
    try:
        from leave_information_bubble.world.schema import initialize

        initialize(connection)
        # rewind exactly to the pre-v17 state: drop the lease table (if the
        # chain created it) and mark the world as v16
        connection.execute("DROP TABLE IF EXISTS graph_shell_writer_leases")
        connection.execute("PRAGMA user_version = 16")
        connection.commit()
    finally:
        connection.close()
    assert "graph_shell_writer_leases" not in _table_names(world_path)
    assert _user_version(world_path) == 16

    store = WorldStore(world_path)  # production open path migrates
    with store.read_connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 18
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'graph_shell_writer_leases'"
        ).fetchone() is not None


async def test_blank_initializer_fails_closed_when_world_registry_incomplete(tmp_path, monkeypatch):
    """C-1: initialize_blank_world verifies the whole world registry, not just the version.

    If the schema chain ever stops producing a registry table, the blank
    initializer must refuse to hand out a half-built world instead of trusting
    the version number alone.
    """
    import leave_information_bubble.world.initialization as initialization_module

    real_initialize = initialization_module.initialize

    def incomplete_initialize(connection):
        real_initialize(connection)
        connection.execute("DROP TABLE graph_shell_writer_leases")
        connection.execute("PRAGMA user_version = 16")
        connection.commit()

    monkeypatch.setattr(initialization_module, "initialize", incomplete_initialize)
    with pytest.raises(InitializationError):
        initialize_blank_world(tmp_path / "incomplete.sqlite3")
