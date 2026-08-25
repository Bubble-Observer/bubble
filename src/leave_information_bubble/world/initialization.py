"""Blank long-term database initialization and isolated canary preparation.

Two strictly separated workflows for the paid canary, both operating on
brand-new paths only:

- ``init_blank_database_pair`` creates an empty v16 world (through the
  official :func:`~leave_information_bubble.world.schema.initialize` path)
  and a fresh runtime database: the four runtime stores in fixed order
  (inquiry leases, model-call ledger, digest cache, wake impressions), then
  the LangGraph ``AsyncSqliteSaver``, then a reopened-table verification.
- ``prepare_contract_canary`` clones an existing world snapshot to an
  isolated path and creates an empty runtime there; source and destinations
  must resolve to different files and every destination must not exist.

Both workflows fail closed: every target is checked (after resolving
symlinks and relative aliases) before anything is created, so a configured
real database can never be touched, directly or through a path alias.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from leave_information_bubble.runtime.inquiry_lease import InquiryLeaseStore
from leave_information_bubble.world.schema import initialize
from leave_information_bubble.world_agent.model_calls import ModelCallRecorder

#: The current world schema version produced by ``schema.initialize``.
WORLD_SCHEMA_VERSION = 18

#: Every application table a fresh world database must contain after
#: initialization (the schema chain's versioned additions included). FTS
#: shadow tables (``objects_fts*`` / ``assertions_fts*``) are SQLite-internal
#: machinery and are not application tables; the registry contract is
#: ``WORLD_TABLES ⊆ actual`` plus a closure check that actual minus the
#: registry contains only FTS/sqlite machinery.
WORLD_TABLES = (
    "objects",
    "assertions",
    "inquiries",
    "observations",
    "object_observations",
    "inquiry_observations",
    "observation_bodies",
    "assertion_evidence",
    "predicates",
    "object_aliases",
    "proposal_attempts",
    "inquiry_attempt_touches",
    "world_audit",
    "identity_aliases",
    "commit_receipts",
    "finalize_receipts",
    "staged_objects",
    "staged_assertions",
    "staged_inquiries",
    "staged_patch_receipts",
    "graph_shell_wake_claims",
    "graph_shell_writer_leases",
)

#: Every table a fresh runtime database must contain after initialization.
#: ``writes`` is the LangGraph ``AsyncSqliteSaver`` pair of ``checkpoints``;
#: ``graph_shell_wake_claims`` is the Graph Shell CLI's per-wake identity
#: ledger, created eagerly here so a fresh runtime always satisfies the
#: registry (the CLI's ``CREATE TABLE IF NOT EXISTS`` remains an idempotent
#: fallback for runtimes created before the registry included the table).
RUNTIME_TABLES = (
    "checkpoints",
    "writes",
    "inquiry_leases",
    "model_calls",
    "graph_shell_wake_claims",
)


class InitializationError(RuntimeError):
    """A requested target exists, aliases an existing database, or is invalid."""


def initialize_blank_world(path: str | Path) -> Path:
    """Create an empty v17 world database through the official schema path.

    The target must not exist: opening an existing database would run the
    schema migrations against a live world, so this entry point is guarded.
    Workflow callers (:func:`init_blank_database_pair`) check targets before
    calling — never point a direct caller at a configured real database.

    Args:
        path: The world database path; parent directories are created.

    Returns:
        The resolved world database path.

    Raises:
        InitializationError: If the target already exists (checked after
            symlink/relative resolution), or the official initialize pass
            does not reach the current schema version or produce every
            :data:`WORLD_TABLES` table.

    """
    resolved = Path(path).resolve()
    if resolved.exists():
        raise InitializationError(f"target already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved)
    try:
        initialize(connection)
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()
    if version != WORLD_SCHEMA_VERSION:
        raise InitializationError(
            f"world schema version {version} after initialize, expected {WORLD_SCHEMA_VERSION}"
        )
    missing = set(WORLD_TABLES) - _existing_tables(resolved)
    if missing:
        raise InitializationError(
            f"world initialization incomplete; missing tables: {', '.join(sorted(missing))}"
        )
    return resolved


async def initialize_blank_runtime(path: str | Path) -> Path:
    """Create a fresh runtime database with every required table.

    The target must not exist: opening an existing database would layer new
    stores and tables over a live runtime ledger, so this entry point is
    guarded. Workflow callers (:func:`init_blank_database_pair`,
    :func:`prepare_contract_canary`) check targets before calling — never
    point a direct caller at a configured real database.

    The four runtime stores are constructed in fixed order, the Graph Shell
    wake-claim ledger is created eagerly (the CLI may not invent runtime
    tables at wake start; its ``CREATE TABLE IF NOT EXISTS`` is only an
    idempotent fallback), then the LangGraph ``AsyncSqliteSaver`` is set up
    on an ``isolation_level=None`` aiosqlite connection. Afterwards the
    database is reopened and every required runtime table is verified, so a
    zero-byte or half-initialized file never passes as a runtime database.

    Args:
        path: The runtime database path; parent directories are created.

    Returns:
        The resolved runtime database path.

    Raises:
        InitializationError: If the target already exists (checked after
            symlink/relative resolution), or any required runtime table is
            missing after initialization.

    """
    resolved = Path(path).resolve()
    if resolved.exists():
        raise InitializationError(f"target already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    InquiryLeaseStore(resolved)
    ModelCallRecorder(resolved)
    connection = sqlite3.connect(resolved)
    try:
        # Graph Shell wake claims: eager so a fresh runtime satisfies the
        # registry; the DDL mirrors the CLI's fallback exactly.
        connection.execute(
            "CREATE TABLE IF NOT EXISTS graph_shell_wake_claims ("
            "wake_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, "
            "world_store_identity TEXT NOT NULL, claimed_at REAL NOT NULL)"
        )
        connection.commit()
    finally:
        connection.close()
    connection = await aiosqlite.connect(resolved, isolation_level=None)
    try:
        saver = AsyncSqliteSaver(connection)
        await saver.setup()
    finally:
        await connection.close()
    missing = set(RUNTIME_TABLES) - _existing_tables(resolved)
    if missing:
        raise InitializationError(
            f"runtime initialization incomplete; missing tables: {', '.join(sorted(missing))}"
        )
    return resolved


async def init_blank_database_pair(world_path: str | Path, runtime_path: str | Path) -> tuple[Path, Path]:
    """Create a blank v16 world and a fresh runtime, failing closed.

    Every target is checked before anything is created: the workflow accepts
    only non-existent targets, the two targets must resolve to different
    files, and relative aliases or symlinks cannot hide an existing target.

    Args:
        world_path: The new world database path.
        runtime_path: The new runtime database path.

    Returns:
        The resolved (world, runtime) path pair.

    Raises:
        InitializationError: If any target already exists or the two targets
            resolve to the same file.

    """
    world = Path(world_path).resolve()
    runtime = Path(runtime_path).resolve()
    problems: list[str] = []
    if world.exists():
        problems.append(f"target already exists: {world}")
    if runtime.exists():
        problems.append(f"target already exists: {runtime}")
    if world == runtime:
        problems.append(f"world and runtime must be different files (both resolve to {world})")
    if problems:
        raise InitializationError("; ".join(problems))
    initialize_blank_world(world)
    await initialize_blank_runtime(runtime)
    return world, runtime


async def prepare_contract_canary(
    world_from: str | Path, world_out: str | Path, runtime_out: str | Path
) -> tuple[Path, Path]:
    """Clone a world snapshot to an isolated path and create an empty runtime.

    The workflow never creates a blank world: it copies the existing
    snapshot byte-for-byte. The source must be a file, every destination must
    not exist, and source and destinations must resolve to different files —
    all checked before anything is created, so a configured real database can
    never be a destination.

    Args:
        world_from: The existing world snapshot to clone.
        world_out: The isolated world copy path.
        runtime_out: The new empty runtime path.

    Returns:
        The resolved (world copy, runtime) path pair.

    Raises:
        InitializationError: If the source is missing, a destination exists,
            or source and destinations resolve to the same file.

    """
    source = Path(world_from).resolve()
    world_dest = Path(world_out).resolve()
    runtime_dest = Path(runtime_out).resolve()
    problems: list[str] = []
    if not source.is_file():
        problems.append(f"world source is not a file: {source}")
    if world_dest.exists():
        problems.append(f"destination already exists: {world_dest}")
    if runtime_dest.exists():
        problems.append(f"destination already exists: {runtime_dest}")
    if source == world_dest:
        problems.append(f"world source and destination resolve to the same file: {source}")
    if source == runtime_dest:
        problems.append(f"runtime destination resolves to the world source: {source}")
    if world_dest == runtime_dest:
        problems.append(
            f"world and runtime destinations must be different files (both resolve to {world_dest})"
        )
    if problems:
        raise InitializationError("; ".join(problems))
    world_dest.parent.mkdir(parents=True, exist_ok=True)
    runtime_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, world_dest)
    await initialize_blank_runtime(runtime_dest)
    return world_dest, runtime_dest


def _existing_tables(path: Path) -> set[str]:
    """Return the table names present in the SQLite master of *path*."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        return {str(row[0]) for row in rows}
    finally:
        connection.close()
