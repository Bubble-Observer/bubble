"""Cross-domain shared helpers for world recall (strict leaf)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

_MAX_RESULTS = 30


def _usage_evidence_counts(
    connection: sqlite3.Connection, assertion_ids: list[str]
) -> dict[str, dict[str, int]]:
    """Count each name_usage assertion's evidence links by role, zero-filled."""
    counts: dict[str, dict[str, int]] = {}
    if not assertion_ids:
        return counts
    marks = ", ".join("?" for _ in assertion_ids)
    for row in connection.execute(
        "SELECT assertion_id, role, COUNT(*) AS count FROM assertion_evidence"
        f" WHERE assertion_id IN ({marks}) GROUP BY assertion_id, role",
        assertion_ids,
    ).fetchall():
        counts.setdefault(str(row["assertion_id"]), {})[str(row["role"])] = int(row["count"])
    for identifier in assertion_ids:
        by_role = counts.setdefault(identifier, {})
        for role in ("supports", "context", "contradicts"):
            by_role.setdefault(role, 0)
    return counts


def _objects_by_id(connection: sqlite3.Connection, identifiers: list[str]) -> list[dict[str, Any]]:
    if not identifiers:
        return []
    marks = ", ".join("?" for _ in identifiers)
    statement = f"SELECT * FROM objects WHERE id IN ({marks}) ORDER BY id"
    return _rows(connection.execute(statement, identifiers).fetchall())


def _rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be positive")
    return min(limit, _MAX_RESULTS)


def _parse_iso(value: str | None) -> datetime | None:
    """Parse a stored UTC ISO timestamp string into an aware datetime (None passes through)."""
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
