"""Singleton Graph Shell writer lease helpers (world DB).

The world's single active Graph Shell writer is arbitrated by one row in
``graph_shell_writer_leases`` (singleton_id = 1 PK CHECK; v17 schema
migration). Acquisition stays in the CLI composition root because it must
race the wake-claim INSERT atomically in one write transaction; this module
holds the read-only inspection and release helpers shared by the CLI, the
read-only status entry and the graph-level lease re-validation.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from typing import Literal

from leave_information_bubble.world.store import WorldStore

#: The three staging ledgers whose ``status = 'active'`` rows are a wake's
#: live working graph. A wake owns active staging even after its run ends,
#: and active staging by any wake blocks other writers.
STAGED_TABLES = ("staged_objects", "staged_assertions", "staged_inquiries")

#: F-05 frozen lease lifecycle for a finished run. ``published`` and
#: ``already_published`` always release the gate; ``wake_closed`` releases it
#: only when nothing active remains staged for the wake. Everything else
#: (``blocked`` / ``compile_failed`` / ``commit_rejected`` /
#: ``staged_unpublished``) keeps the gate because the same wake may resume
#: and keep working; reconciliation / provider failures and
#: ``CommitReplayConflict`` never return a terminal and keep the gate as
#: exception paths. ``abandon_writer_lease`` is the only release-by-abandon
#: path (the ``--graph-shell-abandon`` management entry); its reverse,
#: ``restore_writer_lease`` (``--graph-shell-restore``), re-activates an
#: abandoned wake's staging so the same wake can resume and publish it.
RELEASE_TERMINALS = frozenset({"published", "already_published"})


def writer_lease_action(
    terminal_status: str, has_active_staging: bool
) -> Literal["release", "keep", "keep_recovery_action"]:
    """Return the frozen lease action for a finished run's terminal state.

    ``keep_recovery_action`` means the gate stays AND the operator needs an
    explicit action: a ``wake_closed`` wake with stranded active staging can
    never publish those items under its own identity, and only the explicit
    abandon entry (or same-wake resume after a manual reset) can move on.
    """
    if terminal_status in RELEASE_TERMINALS:
        return "release"
    if terminal_status == "wake_closed":
        return "release" if not has_active_staging else "keep_recovery_action"
    return "keep"


def read_writer_lease(connection: sqlite3.Connection) -> sqlite3.Row | None:
    """Return the world's singleton writer-lease row, or None when unleased.

    The lease table is a v17 schema citizen; callers opening a world through
    ``WorldStore`` always have it. A missing row means no wake owns the write
    gate — either no Graph Shell wake ever ran, or the previous owner
    published or was explicitly abandoned.
    """
    return connection.execute(
        "SELECT owner_wake_id, owner_thread_id, claimed_at "
        "FROM graph_shell_writer_leases WHERE singleton_id = 1"
    ).fetchone()


def lease_owner_is(store: WorldStore, wake_id: str) -> bool:
    """Return whether *wake_id* still owns the world's writer lease.

    The lease can disappear mid-run only through an operator's explicit
    abandon (or a manual DELETE): the graph re-validates this before every
    world mutation so an abandoned wake stops writing instead of continuing
    after the gate was released.
    """
    with store.read_connection() as connection:
        row = read_writer_lease(connection)
    return row is not None and str(row["owner_wake_id"]) == wake_id


def wake_active_staging(store: WorldStore, wake_id: str) -> dict[str, list[str]]:
    """Return each staging ledger's active ``staged_id`` list for *wake_id*."""
    active: dict[str, list[str]] = {}
    with store.read_connection() as connection:
        for table in STAGED_TABLES:
            rows = connection.execute(
                f"SELECT staged_id FROM {table} "
                "WHERE wake_id = ? AND status = 'active' ORDER BY staged_id",
                (wake_id,),
            ).fetchall()
            active[table] = [str(row["staged_id"]) for row in rows]
    return active


def wake_has_active_staging(store: WorldStore, wake_id: str) -> bool:
    """Return whether *wake_id* still holds any active staged working graph."""
    return any(wake_active_staging(store, wake_id).values())


def release_writer_lease(store: WorldStore, wake_id: str) -> None:
    """Close the world's singleton writer lease after a wake publishes.

    Tolerant of a missing table only as defense: the lease table is a v17
    schema citizen and every WorldStore open migrates it in, but a world
    opened without migration (read-only or pre-v17 tooling) still releases
    nothing.
    """
    try:
        with store.write_connection() as world_connection:
            world_connection.execute(
                "DELETE FROM graph_shell_writer_leases WHERE owner_wake_id = ?",
                (wake_id,),
            )
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error):
            raise


def abandon_writer_lease(
    store: WorldStore, wake_id: str
) -> tuple[dict[str, int], bool]:
    """Mark *wake_id*'s active staging abandoned and release the world lease.

    Runs inside one world write transaction, so the staging statuses and the
    lease row either land together or not at all — there is no intermediate
    durable state where the gate is free while stranded active staging still
    blocks other writers, or vice versa. Returns the abandoned row count per
    ledger and whether the lease row was released. Tolerant of a missing
    lease table only as defense, like ``release_writer_lease``.
    """
    abandoned: dict[str, int] = {}
    released = False
    try:
        with store.write_connection() as connection:
            now = datetime.now(UTC).isoformat()
            for table in STAGED_TABLES:
                cursor = connection.execute(
                    f"UPDATE {table} SET status = 'abandoned', updated_at = ? "
                    "WHERE wake_id = ? AND status = 'active'",
                    (now, wake_id),
                )
                abandoned[table] = cursor.rowcount
            cursor = connection.execute(
                "DELETE FROM graph_shell_writer_leases WHERE owner_wake_id = ?",
                (wake_id,),
            )
            released = cursor.rowcount == 1
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error):
            raise
    return abandoned, released


def restore_writer_lease(
    store: WorldStore, wake_id: str, thread_id: str
) -> tuple[dict[str, int], bool]:
    """Re-activate *wake_id*'s abandoned staging and re-acquire the world lease.

    The explicit reverse of ``abandon_writer_lease`` (used by the
    ``--graph-shell-restore`` management entry): marks every abandoned staged
    row of *wake_id* back to ``active`` and inserts the singleton lease row
    for the same wake/thread in one world transaction, so the documented
    resume workflow can continue the wake and publish the recovered working
    graph. The caller must already have validated ownership and traceability
    (every abandoned row backed by a successful patch receipt); a leaked
    IntegrityError here means another process claimed the gate concurrently
    and must fail closed.
    """
    restored: dict[str, int] = {}
    now = datetime.now(UTC).isoformat()
    with store.write_connection() as connection:
        for table in STAGED_TABLES:
            cursor = connection.execute(
                f"UPDATE {table} SET status = 'active', updated_at = ? "
                "WHERE wake_id = ? AND status = 'abandoned'",
                (now, wake_id),
            )
            restored[table] = cursor.rowcount
        connection.execute(
            "INSERT INTO graph_shell_writer_leases "
            "(singleton_id, owner_wake_id, owner_thread_id, claimed_at) "
            "VALUES (1, ?, ?, ?)",
            (wake_id, thread_id, time.time()),
        )
    return restored, True
