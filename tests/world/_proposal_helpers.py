# Split from tests/world/test_proposal.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

import importlib
import importlib.util
import sqlite3
from datetime import UTC, datetime

from leave_information_bubble.world import (
    ObservationDepth,
    ObservationInput,
)


def _proposal_module():
    spec = importlib.util.find_spec("leave_information_bubble.world.proposal")
    assert spec is not None, "world.proposal must define the Agent output contract"
    return importlib.import_module("leave_information_bubble.world.proposal")


def _committer_module():
    spec = importlib.util.find_spec("leave_information_bubble.world.committer")
    assert spec is not None, "world.committer must import Agent proposals deterministically"
    return importlib.import_module("leave_information_bubble.world.committer")


def _observation(identifier: str, depth: ObservationDepth) -> ObservationInput:
    return ObservationInput(
        id=identifier,
        source_uri=f"https://example.test/{identifier}",
        source_kind="web",
        title=identifier,
        depth=depth,
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )


def _world_schema_snapshot(path) -> tuple[set[str], list[tuple[str, str]]]:
    """Return the world tables plus objects columns to detect schema drift.

    Routing legacy kinds must not create subtype tables or dynamic columns,
    so tests compare this snapshot before and after a routed commit.
    """
    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        columns = [
            (str(row[1]), str(row[2]))
            for row in connection.execute("PRAGMA table_info(objects)").fetchall()
        ]
    return tables, columns


