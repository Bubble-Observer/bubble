"""Bounded, read-only inspection queries for the local Agent console."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from leave_information_bubble.world.graph_contract import normalize_identity_alias
from leave_information_bubble.world.recall_search import identity_alias_forms, legacy_alias_forms
from leave_information_bubble.world.writer_lease import STAGED_TABLES

_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_INQUIRY_STATUSES = frozenset({"open", "dormant", "resolved"})
_DETAIL_ASSERTION_LIMIT = 40
_DETAIL_INQUIRY_LIMIT = 40
_DETAIL_OBSERVATION_LIMIT = 100
_EVIDENCE_PER_ASSERTION_LIMIT = 12
_DETAIL_ALIAS_LIMIT = 40
_DETAIL_NAME_USAGE_LIMIT = 40
_TEXT_LIMIT = 500
_LIMITATIONS_TRUNCATED = "limitations_truncated"
_GRAPH_NODE_MAX = 32
_GRAPH_EDGE_MAX = 96

# Task 4.3 basis labels reuse the task 4.1 graded name-candidate vocabulary
# (identity_alias_exact / canonical_exact / name_usage / legacy_name /
# possible): the console search labels each hit with its strongest graded
# claim. name_usage is not a search surface — usage assertions surface in
# object detail only.
_BASIS_IDENTITY_ALIAS_EXACT = "identity_alias_exact"
_BASIS_CANONICAL_EXACT = "canonical_exact"
_BASIS_LEGACY_NAME = "legacy_name"
_BASIS_POSSIBLE = "possible"
_BASIS_LAYER_ORDER = (_BASIS_IDENTITY_ALIAS_EXACT, _BASIS_CANONICAL_EXACT, _BASIS_LEGACY_NAME)
_BASIS_RANK = {kind: index for index, kind in enumerate(_BASIS_LAYER_ORDER)}

# Identity-term derivation mirrors world.recall._identity_terms so an exact
# claim means exactly what memory_search's candidate layers mean.
_MAX_QUERY_TERMS = 12
_LATIN_TOKEN = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[一-鿿]+")


class ReadOnlyInspection:
    """Read one Agent's world and runtime stores without creating or migrating them."""

    def __init__(self, world_db: str | Path, runtime_db: str | Path | None = None) -> None:
        self.world_db = Path(world_db)
        self.runtime_db = Path(runtime_db) if runtime_db is not None else None

    def summary(self) -> dict[str, Any]:
        """Return headline memory counts, or an uninitialized result for a missing store."""
        counts = {
            "objects": 0,
            "assertions": 0,
            "current_assertions": 0,
            "observations": 0,
            "inquiries": 0,
            "open_inquiries": 0,
            "commits": 0,
            "cognition_commits": 0,
        }
        with _read_connection(self.world_db) as connection:
            if connection is None:
                return {"initialized": False, "available": False, "counts": counts}
            tables = _tables(connection)
            for key, table in (
                ("objects", "objects"),
                ("assertions", "assertions"),
                ("observations", "observations"),
                ("inquiries", "inquiries"),
                ("commits", "world_audit"),
            ):
                if table in tables:
                    counts[key] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            if "assertions" in tables:
                counts["current_assertions"] = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM assertions a WHERE NOT EXISTS "
                        "(SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)"
                    ).fetchone()[0]
                )
            if "inquiries" in tables:
                counts["open_inquiries"] = int(
                    connection.execute("SELECT COUNT(*) FROM inquiries WHERE status = 'open'").fetchone()[0]
                )
            if "world_audit" in tables:
                counts["cognition_commits"] = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM world_audit WHERE commit_id LIKE 'agent:%'"
                    ).fetchone()[0]
                )
            return {"initialized": True, "available": True, "counts": counts}

    def pending_wakes(self) -> list[dict[str, Any]]:
        """List wakes whose active staging a finalize attempt can still publish.

        A wake is listed exactly when it has a world-scoped writer claim, at
        least one active staged row, and no published receipt — the same
        fail-closed criteria the deterministic finalize entry applies.  Every
        listed wake is therefore a publish candidate and nothing else is:
        wakes whose rows were abandoned (no claim) belong to restore, and
        wakes that published already (stored receipt) are excluded even when
        later staging exists (I1: post-publish work belongs to a new wake).
        Missing or uninitialized stores list as empty.
        """
        with _read_connection(self.world_db) as connection:
            if connection is None:
                return []
            tables = _tables(connection)
            if "graph_shell_wake_claims" not in tables:
                return []
            staged_tables = [table for table in STAGED_TABLES if table in tables]
            if not staged_tables:
                return []
            rows_by_wake: dict[str, dict[str, int]] = {}
            for table in staged_tables:
                for row in connection.execute(
                    f"SELECT wake_id, COUNT(*) AS n FROM {table} WHERE status = 'active' GROUP BY wake_id"
                ):
                    counts = rows_by_wake.setdefault(str(row["wake_id"]), {})
                    counts[table] = int(row["n"])
            if not rows_by_wake:
                return []
            published = (
                {str(row["wake_id"]) for row in connection.execute("SELECT wake_id FROM finalize_receipts")}
                if "finalize_receipts" in tables
                else set()
            )
            claims = {
                str(row["wake_id"]): str(row["thread_id"])
                for row in connection.execute("SELECT wake_id, thread_id FROM graph_shell_wake_claims")
            }
        return [
            {
                "wake_id": wake_id,
                "staging": {name: counts.get(name, 0) for name in STAGED_TABLES},
                "staging_total": sum(counts.values()),
                "claimed_by": claims.get(wake_id),
            }
            for wake_id, counts in sorted(rows_by_wake.items())
            if wake_id in claims and wake_id not in published
        ]

    def search_objects(
        self,
        query: str = "",
        *,
        kind: str | None = None,
        limit: int = _DEFAULT_LIMIT,
        after_name: str | None = None,
        after_id: str | None = None,
    ) -> dict[str, Any]:
        """Search objects and aliases with stable keyset pagination.

        Results carry a ``basis`` label naming the strongest graded name
        claim of the query per object (task 4.1 vocabulary): an
        identity-normalized term equal to an ACTIVE identity alias is
        ``identity_alias_exact``; equal to the canonical name is
        ``canonical_exact`` (canonical names are not unique, so several
        objects may share the label); equal to a legacy ``object_aliases``
        row via the legacy dual forms is ``legacy_name``. An object that
        only matched the substring without any exact graded claim is
        ``possible``. An empty query lists everything and carries no basis.
        """
        cap = _bounded_limit(limit)
        needle = query.strip()[:500]
        with _read_connection(self.world_db) as connection:
            if connection is None:
                return _page([], cap, available=False)
            tables = _tables(connection)
            has_identity_aliases = "identity_aliases" in tables
            has_object_aliases = "object_aliases" in tables
            # Task 2.1 (Ruling A): alias matches search the ACTIVE identity
            # alias index plus any pre-existing legacy rows; a schema without
            # either alias table (or without objects) is too partial to answer
            if "objects" not in tables or not (has_identity_aliases or has_object_aliases):
                return _page([], cap, available=False)
            conditions: list[str] = []
            parameters: list[object] = []
            if needle:
                alias_clauses: list[str] = []
                if has_identity_aliases:
                    alias_clauses.append(
                        "EXISTS (SELECT 1 FROM identity_aliases ia WHERE ia.object_id = o.id "
                        "AND ia.status = 'active' AND ia.normalized_alias LIKE ? "
                        "ESCAPE '\\' COLLATE NOCASE)"
                    )
                if has_object_aliases:
                    alias_clauses.append(
                        "EXISTS (SELECT 1 FROM object_aliases oa WHERE oa.object_id = o.id "
                        "AND oa.normalized_alias LIKE ? ESCAPE '\\' COLLATE NOCASE)"
                    )
                conditions.append(
                    "(o.canonical_name LIKE ? ESCAPE '\\' COLLATE NOCASE OR "
                    + " OR ".join(alias_clauses)
                    + ")"
                )
                pattern = f"%{_escape_like(needle)}%"
                parameters.extend((pattern,) * (1 + len(alias_clauses)))
            if kind:
                conditions.append("o.kind = ?")
                parameters.append(kind)
            if after_name is not None and after_id is not None:
                conditions.append(
                    "(o.canonical_name COLLATE NOCASE > ? OR "
                    "(o.canonical_name COLLATE NOCASE = ? AND o.id > ?))"
                )
                parameters.extend((after_name, after_name, after_id))
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            rows = connection.execute(
                "SELECT o.id, o.kind, o.canonical_name, o.provisional, o.event_time_start, "
                "o.event_time_end, o.version, "
                "(SELECT COUNT(*) FROM assertions a WHERE a.subject_id = o.id AND NOT EXISTS "
                "(SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)) AS assertion_count, "
                "(SELECT COUNT(*) FROM inquiries i WHERE i.subject_id = o.id "
                "AND i.status IN ('open','dormant')) AS inquiry_count "
                f"FROM objects o{where} ORDER BY o.canonical_name COLLATE NOCASE, o.id LIMIT ?",
                (*parameters, cap + 1),
            ).fetchall()
            bases = _match_bases(connection, rows[:cap], needle) if needle else None
        items = [_bounded_row(dict(row)) for row in rows[:cap]]
        if bases is not None:
            for item, label in zip(items, bases, strict=True):
                item["basis"] = label
        next_cursor = None
        if len(rows) > cap and rows:
            next_cursor = {"after_name": rows[cap - 1]["canonical_name"], "after_id": rows[cap - 1]["id"]}
        return {"available": True, "items": items, "next_cursor": next_cursor}

    def graph_snapshot(self, *, limit: int = 24, window: int = 0) -> dict[str, Any]:
        """Return one deterministic window of a relationship-first projection.

        The console must stay useful when literal-heavy objects outrank the few
        objects that actually form a graph.  Relationship endpoints lead a stable
        ranking, followed by the remaining objects in activity order.  ``window``
        advances through that complete ranking so lower-ranked objects are not
        permanently hidden.  All queries remain read-only and the returned totals
        make the bounded projection explicit to the UI.
        """
        cap = _bounded_limit(limit)
        if cap > _GRAPH_NODE_MAX:
            raise ValueError(f"graph limit must be between 1 and {_GRAPH_NODE_MAX}")
        if window < 0:
            raise ValueError("graph window must be non-negative")
        empty = {"available": False, "nodes": [], "edges": [], "truncated": False}
        with _read_connection(self.world_db) as connection:
            if connection is None:
                return empty
            tables = _tables(connection)
            if not {"objects", "assertions"}.issubset(tables):
                return empty
            object_columns = _columns(connection, "objects")
            assertion_columns = _columns(connection, "assertions")
            if not {"id", "kind", "canonical_name", "provisional"}.issubset(object_columns):
                return empty
            if not {
                "id",
                "subject_id",
                "object_id",
                "predicate",
                "epistemic_role",
                "confidence",
                "supersedes_id",
            }.issubset(assertion_columns):
                return empty
            inquiry_columns = _columns(connection, "inquiries") if "inquiries" in tables else set()
            inquiry_count = (
                "(SELECT COUNT(*) FROM inquiries i WHERE i.subject_id = o.id "
                "AND i.status IN ('open','dormant'))"
                if {"subject_id", "status"}.issubset(inquiry_columns)
                else "0"
            )
            total_nodes = int(connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0])
            assertion_totals = connection.execute(
                "SELECT COUNT(*) AS current_assertions, "
                "COALESCE(SUM(CASE WHEN a.object_id IS NOT NULL THEN 1 ELSE 0 END), 0) "
                "AS object_relations, "
                "COUNT(DISTINCT CASE WHEN a.object_id IS NOT NULL THEN "
                "a.subject_id || CHAR(31) || LOWER(TRIM(a.predicate)) || CHAR(31) || a.object_id "
                "END) AS object_connections "
                "FROM assertions a WHERE NOT EXISTS "
                "(SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)"
            ).fetchone()
            total_assertions = int(assertion_totals["current_assertions"])
            total_relations = int(assertion_totals["object_relations"])
            total_connections = int(assertion_totals["object_connections"])

            # Rank relation clusters before high-volume literal attributes.  The
            # endpoint loop below guarantees that a selected edge keeps both ends
            # whenever the requested node cap has room for them.
            ranked_edges = connection.execute(
                "WITH current_edges AS ("
                "SELECT MIN(a.id) AS id, a.subject_id, a.object_id, MIN(a.predicate) AS predicate, "
                "MAX(a.confidence) AS confidence, COUNT(*) AS assertion_count FROM assertions a "
                "WHERE a.object_id IS NOT NULL AND NOT EXISTS "
                "(SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id) "
                "GROUP BY a.subject_id, a.object_id, LOWER(TRIM(a.predicate))), "
                "edge_nodes AS ("
                "SELECT id, subject_id AS node_id FROM current_edges "
                "UNION SELECT id, object_id AS node_id FROM current_edges), "
                "degrees AS (SELECT node_id, COUNT(*) AS degree FROM edge_nodes GROUP BY node_id) "
                "SELECT e.*, (source_degree.degree + target_degree.degree) AS relation_weight "
                "FROM current_edges e "
                "JOIN degrees source_degree ON source_degree.node_id = e.subject_id "
                "JOIN degrees target_degree ON target_degree.node_id = e.object_id "
                "ORDER BY relation_weight DESC, e.confidence DESC, e.id LIMIT ?",
                (_GRAPH_EDGE_MAX + 1,),
            ).fetchall()
            ranked_ids: list[str] = []
            ranked_set: set[str] = set()
            for edge in ranked_edges:
                for column in ("subject_id", "object_id"):
                    object_id = str(edge[column])
                    if object_id not in ranked_set:
                        ranked_ids.append(object_id)
                        ranked_set.add(object_id)

            # Complete the stable rotation ranking with every remaining object.
            # The aggregate CTE scans current assertions once instead of running a
            # correlated count for every object as the database grows.
            active_rows = connection.execute(
                "WITH current_assertions AS ("
                "SELECT a.id, a.subject_id, a.object_id FROM assertions a WHERE NOT EXISTS "
                "(SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)), "
                "assertion_nodes AS ("
                "SELECT id, subject_id AS node_id FROM current_assertions "
                "UNION SELECT id, object_id AS node_id FROM current_assertions "
                "WHERE object_id IS NOT NULL), "
                "activity AS (SELECT node_id, COUNT(*) AS assertion_count "
                "FROM assertion_nodes GROUP BY node_id) "
                "SELECT o.id FROM objects o LEFT JOIN activity ON activity.node_id = o.id "
                "ORDER BY COALESCE(activity.assertion_count, 0) DESC, "
                "o.canonical_name COLLATE NOCASE, o.id",
            ).fetchall()
            for row in active_rows:
                object_id = str(row["id"])
                if object_id not in ranked_set:
                    ranked_ids.append(object_id)
                    ranked_set.add(object_id)

            if not ranked_ids:
                return {
                    "available": True,
                    "nodes": [],
                    "edges": [],
                    "truncated": False,
                    "selection": "relationship_first",
                    "schema_version": int(connection.execute("PRAGMA user_version").fetchone()[0]),
                    "latest_commit": None,
                    "stats": {
                        "total_nodes": 0,
                        "total_current_assertions": total_assertions,
                        "total_object_relations": total_relations,
                        "total_object_connections": total_connections,
                        "total_literal_assertions": total_assertions - total_relations,
                        "displayed_nodes": 0,
                        "displayed_relations": 0,
                        "displayed_connections": 0,
                        "collapsed_relations": 0,
                    },
                }
            window_count = max(1, (len(ranked_ids) + cap - 1) // cap)
            window_index = window % window_count
            window_start = window_index * cap
            selected_ids = ranked_ids[window_start : window_start + cap]
            if len(selected_ids) < cap and len(ranked_ids) > cap:
                selected_ids.extend(ranked_ids[: cap - len(selected_ids)])
            placeholders = ",".join("?" for _ in selected_ids)
            node_rows = connection.execute(
                "SELECT o.id, o.kind, o.canonical_name, o.provisional, "
                "(SELECT COUNT(*) FROM assertions a WHERE "
                "(a.subject_id = o.id OR a.object_id = o.id) AND NOT EXISTS "
                "(SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)) AS assertion_count, "
                f"{inquiry_count} AS inquiry_count "
                f"FROM objects o WHERE o.id IN ({placeholders})",
                selected_ids,
            ).fetchall()
            rows_by_id = {str(row["id"]): row for row in node_rows}
            selected_rows = [rows_by_id[object_id] for object_id in selected_ids if object_id in rows_by_id]
            edge_rows = connection.execute(
                "SELECT MIN(a.id) AS id, a.subject_id, a.object_id, MIN(a.predicate) AS predicate, "
                "CASE WHEN COUNT(DISTINCT a.epistemic_role) = 1 THEN MIN(a.epistemic_role) "
                "ELSE 'mixed' END AS epistemic_role, MAX(a.confidence) AS confidence, "
                "COUNT(*) AS assertion_count, GROUP_CONCAT(a.id, CHAR(31)) AS assertion_ids "
                "FROM assertions a "
                f"WHERE a.subject_id IN ({placeholders}) "
                f"AND a.object_id IN ({placeholders}) "
                "AND NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id) "
                "GROUP BY a.subject_id, a.object_id, LOWER(TRIM(a.predicate)) "
                "ORDER BY MIN(a.id) LIMIT ?",
                (*selected_ids, *selected_ids, _GRAPH_EDGE_MAX + 1),
            ).fetchall()
            latest_commit = None
            if "commit_receipts" in tables and {
                "commit_id",
                "committed_at",
            }.issubset(_columns(connection, "commit_receipts")):
                latest_row = connection.execute(
                    "SELECT commit_id, committed_at FROM commit_receipts "
                    "WHERE commit_id LIKE 'agent:%' OR commit_id LIKE '%:finalize' "
                    "ORDER BY committed_at DESC LIMIT 1"
                ).fetchone()
                if latest_row is not None:
                    latest_commit = {
                        "commit_id": _text(latest_row["commit_id"]),
                        "committed_at": _text(latest_row["committed_at"]),
                    }
            schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        nodes = [_bounded_row(dict(row)) for row in selected_rows]
        edges = [
            {
                "id": str(row["id"]),
                "source": str(row["subject_id"]),
                "target": str(row["object_id"]),
                "predicate": _text(row["predicate"]),
                "epistemic_role": _text(row["epistemic_role"]),
                "confidence": row["confidence"],
                "assertion_count": int(row["assertion_count"]),
                "assertion_ids": sorted(str(row["assertion_ids"]).split("\x1f")),
            }
            for row in edge_rows[:_GRAPH_EDGE_MAX]
        ]
        displayed_relations = sum(int(edge["assertion_count"]) for edge in edges)
        truncated = total_nodes > len(nodes) or total_connections > len(edges)
        return {
            "available": True,
            "nodes": nodes,
            "edges": edges,
            "truncated": truncated,
            "selection": "relationship_first",
            "window": {
                "index": window_index,
                "count": window_count,
                "size": len(selected_ids),
                "rotating": window_count > 1,
            },
            "schema_version": schema_version,
            "latest_commit": latest_commit,
            "stats": {
                "total_nodes": total_nodes,
                "total_current_assertions": total_assertions,
                "total_object_relations": total_relations,
                "total_object_connections": total_connections,
                "total_literal_assertions": total_assertions - total_relations,
                "displayed_nodes": len(nodes),
                "displayed_relations": displayed_relations,
                "displayed_connections": len(edges),
                "collapsed_relations": displayed_relations - len(edges),
            },
        }

    def object_detail(self, object_id: str) -> dict[str, Any] | None:
        """Return one object with current assertions, inquiries and linked evidence summaries.

        Name surfaces are kept apart (task 4.3): ``aliases`` holds only
        ACTIVE identity alias forms (world-unique identity claims);
        ``legacy_names`` is the read-only history of ``object_aliases`` rows
        (no identity authority, and never merged into ``aliases``);
        ``name_usages`` summarizes the object's current
        ``predicate='name_usage'`` assertions (literal, qualifiers, time and
        evidence counts). On an unmigrated v9 snapshot the identity table is
        absent and ``aliases`` degrades to ``[]`` — inspection never migrates
        or writes. Old databases persisted the canonical name in
        ``object_aliases``, which both name lists exclude.
        """
        with _read_connection(self.world_db) as connection:
            if connection is None or not _has_tables(connection, "objects"):
                return None
            row = connection.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
            if row is None:
                return None
            tables = _tables(connection)
            canonical_name = row["canonical_name"]
            aliases, aliases_truncated = _identity_aliases(connection, object_id, tables, canonical_name)
            legacy_names, legacy_names_truncated = _legacy_names(
                connection, object_id, tables, canonical_name
            )
            name_usages, name_usages_truncated = _name_usages(connection, object_id, tables)
            assertions, assertions_truncated = _object_assertions(connection, object_id, tables)
            inquiries, inquiries_truncated = _object_inquiries(connection, object_id, tables)
            observations, observations_truncated = _object_observations(connection, object_id, tables)
            return {
                "available": True,
                "object": _bounded_row(_json_columns(dict(row))),
                "aliases": aliases,
                "aliases_truncated": aliases_truncated,
                "legacy_names": legacy_names,
                "legacy_names_truncated": legacy_names_truncated,
                "name_usages": name_usages,
                "name_usages_truncated": name_usages_truncated,
                "assertions": assertions,
                "assertions_truncated": assertions_truncated,
                "inquiries": inquiries,
                "inquiries_truncated": inquiries_truncated,
                "observations": observations,
                "observations_truncated": observations_truncated,
            }

    def inquiries(
        self,
        *,
        statuses: Sequence[str] = ("open", "dormant"),
        object_id: str | None = None,
        limit: int = _DEFAULT_LIMIT,
        after_id: str | None = None,
    ) -> dict[str, Any]:
        """List bounded inquiries, optionally restricted to one subject."""
        cap = _bounded_limit(limit)
        selected = tuple(dict.fromkeys(statuses))
        if not selected or any(status not in _INQUIRY_STATUSES for status in selected):
            raise ValueError("unsupported inquiry status")
        with _read_connection(self.world_db) as connection:
            if connection is None or not _has_tables(connection, "inquiries", "objects"):
                return _page([], cap, available=False)
            marks = ",".join("?" for _ in selected)
            conditions = [f"i.status IN ({marks})"]
            parameters: list[object] = list(selected)
            if object_id is not None:
                conditions.append("i.subject_id = ?")
                parameters.append(object_id)
            if after_id is not None:
                conditions.append("i.id > ?")
                parameters.append(after_id)
            rows = connection.execute(
                "SELECT i.*, o.canonical_name AS subject_name FROM inquiries i "
                "JOIN objects o ON o.id = i.subject_id WHERE "
                + " AND ".join(conditions)
                + " ORDER BY i.id LIMIT ?",
                (*parameters, cap + 1),
            ).fetchall()
        items = [_bounded_row(_json_columns(dict(row))) for row in rows[:cap]]
        cursor = {"after_id": items[-1]["id"]} if len(rows) > cap and items else None
        return {"available": True, "items": items, "next_cursor": cursor}

    def recent_commits(
        self,
        *,
        limit: int = 20,
        before_at: str | None = None,
        before_id: str | None = None,
    ) -> dict[str, Any]:
        """List recent Agent cognition commits, excluding tool-material commits."""
        cap = _bounded_limit(limit)
        with _read_connection(self.world_db) as connection:
            if connection is None or not _has_tables(connection, "commit_receipts", "world_audit"):
                return _page([], cap, available=False)
            parameters: list[object] = ["agent:%"]
            boundary = ""
            if before_at is not None and before_id is not None:
                boundary = " AND (r.committed_at < ? OR (r.committed_at = ? AND r.commit_id < ?))"
                parameters.extend((before_at, before_at, before_id))
            rows = connection.execute(
                "SELECT r.commit_id, r.committed_at, r.receipt_json, a.delta_json "
                "FROM commit_receipts r JOIN world_audit a ON a.commit_id = r.commit_id "
                "WHERE r.commit_id LIKE ?"
                + boundary
                + " ORDER BY r.committed_at DESC, r.commit_id DESC LIMIT ?",
                (*parameters, cap + 1),
            ).fetchall()
        items = [_commit_view(row) for row in rows[:cap]]
        cursor = None
        if len(rows) > cap and items:
            cursor = {"before_at": items[-1]["committed_at"], "before_id": items[-1]["commit_id"]}
        return {"available": True, "items": items, "next_cursor": cursor}

    def run_inspection(self, thread_id: str, *, wake_id: str | None = None) -> dict[str, Any]:
        """Join one thread/wake to explicit durable-commit and model ledgers."""
        if not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        if wake_id is not None and not wake_id.strip():
            raise ValueError("wake_id must be non-empty when provided")
        run_commit_id = f"agent:{thread_id}"
        attempts = self._attempts(run_commit_id)
        durable_ids = [str(item["durable_commit_id"]) for item in attempts if item.get("durable_commit_id")]
        # Exact root and recovery ids cover successful legacy runs and the recovery
        # path, which currently has no proposal_attempts row. No prefix/time scan is used.
        durable_ids.extend((run_commit_id, f"{run_commit_id}:recovery"))
        # Graph Shell publishes exactly once under the wake-scoped finalize id.
        # The caller recovers the wake identity from the run checkpoint/report;
        # keeping it explicit avoids an unsafe prefix or timestamp attribution.
        if wake_id is not None:
            durable_ids.append(f"{wake_id.strip()}:finalize")
        commits = self._commits_by_id(list(dict.fromkeys(durable_ids)))
        models = self._model_summary(thread_id)
        omissions = {
            "assertions": sum(int(item.get("omitted_assertions") or 0) for item in attempts),
            "inquiries": sum(int(item.get("omitted_inquiries") or 0) for item in attempts),
            "resolutions": sum(int(item.get("omitted_resolutions") or 0) for item in attempts),
            "evidence_missing": sum(int(item.get("evidence_missing_assertions") or 0) for item in attempts),
        }
        durable_diff = _durable_diff(commits)
        return {
            "thread_id": thread_id,
            "wake_id": wake_id,
            "run_commit_id": run_commit_id,
            "model": models,
            "proposal_review": {"attempts": attempts, "omissions": omissions},
            "durable_diff": durable_diff,
            "writes": self._exact_writes(commits, durable_diff),
            "limitations": [
                "tool observation commits are not attributed to a thread by this view",
                "in-memory console events and full run configuration are not reconstructed here",
            ],
        }

    def _exact_writes(
        self,
        commits: Sequence[Mapping[str, Any]],
        durable_diff: Mapping[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Hydrate only ids named by exact durable receipts and deltas."""
        ids: dict[str, list[str]] = {
            "objects": [],
            "assertions": [],
            "inquiries": [],
            "observations": [],
            "resolved_inquiries": [],
        }
        for commit in commits:
            receipt = commit.get("receipt")
            if not isinstance(receipt, Mapping):
                continue
            for target, key in (
                ("objects", "object_ids"),
                ("assertions", "assertion_ids"),
                ("inquiries", "inquiry_ids"),
                ("resolved_inquiries", "resolved_inquiry_ids"),
            ):
                values = receipt.get(key, [])
                if isinstance(values, list):
                    ids[target].extend(str(value) for value in values)
        observations = durable_diff.get("observations", [])
        if isinstance(observations, list):
            ids["observations"].extend(
                str(item["id"]) for item in observations if isinstance(item, Mapping) and item.get("id")
            )
        ids = {key: list(dict.fromkeys(values)) for key, values in ids.items()}
        output: dict[str, list[dict[str, Any]]] = {
            "objects": [],
            "assertions": [],
            "inquiries": [],
            "observations": [],
            "resolved_inquiries": [],
            "evidence_links": [],
            "evidence_links_truncated": False,
        }
        with _read_connection(self.world_db) as connection:
            if connection is None:
                return output
            tables = _tables(connection)
            for key, table in (
                ("objects", "objects"),
                ("assertions", "assertions"),
                ("inquiries", "inquiries"),
                ("observations", "observations"),
            ):
                if table not in tables or not ids[key]:
                    continue
                marks = ",".join("?" for _ in ids[key])
                rows = connection.execute(
                    f"SELECT * FROM {table} WHERE id IN ({marks}) ORDER BY id",
                    tuple(ids[key]),
                ).fetchall()
                output[key] = [_json_columns(dict(row)) for row in rows]
            assertion_object_ids = {
                str(object_id)
                for item in output["assertions"]
                for object_id in (item.get("subject_id"), item.get("object_id"))
                if object_id
            }
            if "objects" in tables and assertion_object_ids:
                marks = ",".join("?" for _ in assertion_object_ids)
                rows = connection.execute(
                    f"SELECT id, canonical_name FROM objects WHERE id IN ({marks})",
                    tuple(sorted(assertion_object_ids)),
                ).fetchall()
                names = {str(row["id"]): str(row["canonical_name"]) for row in rows}
                for item in output["assertions"]:
                    subject_id = str(item.get("subject_id") or "")
                    object_id = str(item.get("object_id") or "")
                    item["subject_name"] = names.get(subject_id, subject_id)
                    if object_id:
                        item["object_name"] = names.get(object_id, object_id)
            if "assertion_evidence" in tables and "observations" in tables and ids["assertions"]:
                marks = ",".join("?" for _ in ids["assertions"])
                rows = connection.execute(
                    "WITH ranked AS ("
                    "SELECT ae.assertion_id, ae.role, o.id, o.title, o.source_uri, "
                    "o.source_kind, o.excerpt, o.depth, o.source_published_at, o.observed_at, "
                    "o.metadata_json, ROW_NUMBER() OVER (PARTITION BY ae.assertion_id "
                    "ORDER BY ae.linked_at, ae.observation_id) AS row_no "
                    "FROM assertion_evidence ae JOIN observations o ON o.id = ae.observation_id "
                    f"WHERE ae.assertion_id IN ({marks})"
                    ") SELECT * FROM ranked WHERE row_no <= ? ORDER BY assertion_id, row_no",
                    (*ids["assertions"], _EVIDENCE_PER_ASSERTION_LIMIT + 1),
                ).fetchall()
                output["evidence_links_truncated"] = any(
                    int(row["row_no"]) > _EVIDENCE_PER_ASSERTION_LIMIT for row in rows
                )
                output["evidence_links"] = [
                    {
                        "assertion_id": str(row["assertion_id"]),
                        "role": str(row["role"]),
                        "observation": _observation_view(row),
                    }
                    for row in rows
                    if int(row["row_no"]) <= _EVIDENCE_PER_ASSERTION_LIMIT
                ]
            if "inquiries" in tables and ids["resolved_inquiries"]:
                marks = ",".join("?" for _ in ids["resolved_inquiries"])
                rows = connection.execute(
                    f"SELECT * FROM inquiries WHERE id IN ({marks}) ORDER BY id",
                    tuple(ids["resolved_inquiries"]),
                ).fetchall()
                output["resolved_inquiries"] = [_json_columns(dict(row)) for row in rows]
        return output

    def _attempts(self, run_commit_id: str) -> list[dict[str, Any]]:
        with _read_connection(self.world_db) as connection:
            if connection is None or not _has_tables(connection, "proposal_attempts"):
                return []
            columns = _columns(connection, "proposal_attempts")
            required = {"attempt_id", "run_commit_id", "attempt_no", "outcome"}
            if not required.issubset(columns):
                return []
            wanted = [
                name
                for name in (
                    "attempt_id",
                    "attempt_no",
                    "parent_attempt_id",
                    "attempted_at",
                    "outcome",
                    "durable_commit_id",
                    "new_objects",
                    "assertions",
                    "inquiries",
                    "omitted_assertions",
                    "omitted_inquiries",
                    "omitted_resolutions",
                    "resolved_inquiries",
                    "evidence_missing_assertions",
                    "error",
                    "issues_json",
                )
                if name in columns
            ]
            rows = connection.execute(
                f"SELECT {', '.join(wanted)} FROM proposal_attempts "
                "WHERE run_commit_id = ? ORDER BY attempt_no, attempt_id",
                (run_commit_id,),
            ).fetchall()
        return [_json_columns(dict(row)) for row in rows]

    def _commits_by_id(self, commit_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not commit_ids:
            return []
        with _read_connection(self.world_db) as connection:
            if connection is None or not _has_tables(connection, "commit_receipts", "world_audit"):
                return []
            marks = ",".join("?" for _ in commit_ids)
            rows = connection.execute(
                "SELECT r.commit_id, r.committed_at, r.receipt_json, a.delta_json "
                "FROM commit_receipts r JOIN world_audit a ON a.commit_id = r.commit_id "
                f"WHERE r.commit_id IN ({marks}) ORDER BY r.committed_at, r.commit_id",
                tuple(commit_ids),
            ).fetchall()
        return [_commit_view(row) for row in rows]

    def _model_summary(self, thread_id: str) -> dict[str, Any]:
        empty = {
            "available": False,
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_input_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "by_purpose": {},
            "by_phase": {},
        }
        if self.runtime_db is None:
            return empty
        with _read_connection(self.runtime_db) as connection:
            if connection is None or not _has_tables(connection, "model_calls"):
                return empty
            columns = _columns(connection, "model_calls")
            if not {"id", "thread_id", "purpose"}.issubset(columns):
                return empty
            names = [
                name
                for name in (
                    "purpose",
                    "phase",
                    "prompt_tokens",
                    "completion_tokens",
                    "cached_input_tokens",
                    "cost_usd",
                    "latency_ms",
                )
                if name in columns
            ]
            rows = connection.execute(
                f"SELECT {', '.join(names)} FROM model_calls WHERE thread_id = ? ORDER BY id",
                (thread_id,),
            ).fetchall()
        purposes = Counter(str(row["purpose"] or "unknown") for row in rows)
        phases = Counter(str(row["phase"] or "unknown") for row in rows) if "phase" in names else Counter()
        return {
            "available": True,
            "calls": len(rows),
            "prompt_tokens": _sum_rows(rows, "prompt_tokens"),
            "completion_tokens": _sum_rows(rows, "completion_tokens"),
            "cached_input_tokens": _sum_rows(rows, "cached_input_tokens"),
            "cost_usd": _sum_rows(rows, "cost_usd"),
            "latency_ms": _sum_rows(rows, "latency_ms"),
            "by_purpose": dict(purposes),
            "by_phase": dict(phases),
        }


@contextmanager
def _read_connection(path: Path) -> Iterator[sqlite3.Connection | None]:
    """Open an existing SQLite file in query-only mode without creating it."""
    if not path.is_file():
        yield None
        return
    try:
        connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
    except (OSError, sqlite3.Error):
        yield None
        return
    try:
        yield connection
    finally:
        connection.close()


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _has_tables(connection: sqlite3.Connection, *names: str) -> bool:
    return set(names).issubset(_tables(connection))


def _match_bases(
    connection: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
    needle: str,
) -> list[str]:
    """Label each matched object with its strongest graded name claim.

    Mirrors the task 4.1 layer semantics over the console's substring
    search: a term that identity-normalizes equal to an ACTIVE identity
    alias is ``identity_alias_exact``; equal to the canonical name is
    ``canonical_exact``; equal to a legacy ``object_aliases`` row (via the
    legacy dual forms) is ``legacy_name``. An object that only matched the
    substring without any exact graded claim is ``possible`` — the console
    has no FTS similarity line, so ``possible`` here means "surfaced, no
    exact name claim". ``name_usage`` is not a search surface.
    """
    if not rows:
        return []
    object_ids = [str(row["id"]) for row in rows]
    tables = _tables(connection)
    # unmigrated v9 snapshots lack the identity table: degrade, never migrate
    identity_forms = identity_alias_forms(connection, object_ids) if "identity_aliases" in tables else {}
    legacy_forms = legacy_alias_forms(connection, object_ids) if "object_aliases" in tables else {}
    terms = _identity_terms(needle)
    output: list[str] = []
    for row in rows:
        identifier = str(row["id"])
        canonical_norm: str | None = None
        if row["canonical_name"]:
            try:
                canonical_norm = normalize_identity_alias(str(row["canonical_name"]))
            except ValueError:
                canonical_norm = None
        best: str | None = None
        best_rank = len(_BASIS_LAYER_ORDER)
        for term in terms:
            try:
                normalized = normalize_identity_alias(term)
            except ValueError:
                continue
            rank: int | None = None
            if normalized in identity_forms.get(identifier, ()):
                rank = _BASIS_RANK[_BASIS_IDENTITY_ALIAS_EXACT]
            elif canonical_norm is not None and normalized == canonical_norm:
                rank = _BASIS_RANK[_BASIS_CANONICAL_EXACT]
            elif set(_legacy_lookup_forms(term)) & legacy_forms.get(identifier, set()):
                rank = _BASIS_RANK[_BASIS_LEGACY_NAME]
            if rank is not None and rank < best_rank:
                best_rank = rank
                best = _BASIS_LAYER_ORDER[rank]
        output.append(best or _BASIS_POSSIBLE)
    return output


def _identity_terms(query: str) -> list[str]:
    """Split a query into identity-comparison terms, keeping CJK runs whole.

    Mirrors ``world.recall._identity_terms`` so the console's exact claims
    use the same term surface as memory_search's candidate layers.
    """
    folded = query.casefold()
    tokens = list(_LATIN_TOKEN.findall(folded))
    tokens.extend(run for run in _CJK_RUN.findall(folded))
    terms = list(dict.fromkeys(tokens))[:_MAX_QUERY_TERMS]
    whole = query.strip()
    if whole:
        terms.append(whole)
    return list(dict.fromkeys(terms))


def _legacy_lookup_forms(alias: str) -> list[str]:
    """LEGACY-ONLY dual lookup forms, mirroring ``world.store.alias_lookup_forms``.

    Legacy ``object_aliases`` rows were persisted with the old
    whitespace-collapsed normalizer; rows written before whitespace folding
    may still hold the space-preserving form. The identity alias index never
    uses these forms.
    """
    collapsed = "".join(alias.strip().casefold().split())
    return list(dict.fromkeys([collapsed, alias.strip().casefold()]))


def _identity_aliases(
    connection: sqlite3.Connection,
    object_id: str,
    tables: set[str],
    canonical_name: object,
) -> tuple[list[str], bool]:
    """ACTIVE identity alias forms for one object; ``[]`` when the table is absent."""
    if "identity_aliases" not in tables:
        return [], False
    rows = connection.execute(
        "SELECT normalized_alias FROM identity_aliases WHERE object_id = ?"
        " AND status = 'active' ORDER BY normalized_alias LIMIT ?",
        (object_id, _DETAIL_ALIAS_LIMIT + 1),
    ).fetchall()
    selected = _exclude_canonical([str(item[0]) for item in rows], canonical_name)
    return [_text(item) for item in selected[:_DETAIL_ALIAS_LIMIT]], (
        len(rows) > _DETAIL_ALIAS_LIMIT
        or any(len(item) > _TEXT_LIMIT for item in selected[:_DETAIL_ALIAS_LIMIT])
    )


def _legacy_names(
    connection: sqlite3.Connection,
    object_id: str,
    tables: set[str],
    canonical_name: object,
) -> tuple[list[str], bool]:
    """Read-only legacy ``object_aliases`` rows for one object; ``[]`` when absent."""
    if "object_aliases" not in tables:
        return [], False
    rows = connection.execute(
        "SELECT normalized_alias FROM object_aliases WHERE object_id = ? ORDER BY normalized_alias LIMIT ?",
        (object_id, _DETAIL_ALIAS_LIMIT + 1),
    ).fetchall()
    selected = _exclude_canonical([str(item[0]) for item in rows], canonical_name)
    return [_text(item) for item in selected[:_DETAIL_ALIAS_LIMIT]], (
        len(rows) > _DETAIL_ALIAS_LIMIT
        or any(len(item) > _TEXT_LIMIT for item in selected[:_DETAIL_ALIAS_LIMIT])
    )


def _exclude_canonical(values: Sequence[str], canonical_name: object) -> list[str]:
    """Drop canonical-name duplicates old databases persisted in name tables."""
    folded = str(canonical_name or "").casefold()
    return [value for value in dict.fromkeys(values) if value.casefold() != folded]


def _name_usages(
    connection: sqlite3.Connection,
    object_id: str,
    tables: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Summarize the object's current ``name_usage`` assertions.

    Each summary mirrors the recall layer's explainable context block:
    assertion id, literal, qualifiers (absent on unmigrated v9 snapshots),
    event time and role-wise evidence counts (zero-filled). Only current
    (non-superseded) assertions are shown.
    """
    if "assertions" not in tables:
        return [], False
    rows = connection.execute(
        "SELECT a.* FROM assertions a WHERE a.subject_id = ? AND a.predicate = 'name_usage'"
        " AND NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)"
        " ORDER BY a.id LIMIT ?",
        (object_id, _DETAIL_NAME_USAGE_LIMIT + 1),
    ).fetchall()
    selected = rows[:_DETAIL_NAME_USAGE_LIMIT]
    if not selected:
        return [], len(rows) > _DETAIL_NAME_USAGE_LIMIT
    counts: dict[str, dict[str, int]] = {}
    if "assertion_evidence" in tables:
        marks = ",".join("?" for _ in selected)
        for row in connection.execute(
            "SELECT assertion_id, role, COUNT(*) AS count FROM assertion_evidence"
            f" WHERE assertion_id IN ({marks}) GROUP BY assertion_id, role",
            [str(item["id"]) for item in selected],
        ).fetchall():
            counts.setdefault(str(row["assertion_id"]), {})[str(row["role"])] = int(row["count"])
    output: list[dict[str, Any]] = []
    for item in selected:
        view = _bounded_row(_json_columns(dict(item)))
        by_role = counts.setdefault(str(item["id"]), {})
        for role in ("supports", "context", "contradicts"):
            by_role.setdefault(role, 0)
        literal = view.get("literal")
        qualifiers = view.get("qualifiers")
        output.append(
            {
                "assertion_id": str(item["id"]),
                "literal": literal if isinstance(literal, str) else "",
                "qualifiers": qualifiers if isinstance(qualifiers, dict) else {},
                "time": view.get("event_time_start"),
                "evidence_counts": by_role,
            }
        )
    return output, len(rows) > _DETAIL_NAME_USAGE_LIMIT


def _bounded_limit(limit: int) -> int:
    if limit < 1 or limit > _MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
    return limit


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _page(items: list[dict[str, Any]], limit: int, *, available: bool) -> dict[str, Any]:
    del limit
    return {"available": available, "items": items, "next_cursor": None}


def _safe_json(value: object, fallback: object) -> object:
    if not isinstance(value, str):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _json_columns(row: dict[str, Any]) -> dict[str, Any]:
    for key in tuple(row):
        if key.endswith("_json"):
            row[key.removesuffix("_json")] = _safe_json(row.pop(key), [] if key == "issues_json" else {})
    return row


def _object_assertions(
    connection: sqlite3.Connection, object_id: str, tables: set[str]
) -> tuple[list[dict[str, Any]], bool]:
    if not {"assertions", "objects"}.issubset(tables):
        return [], False
    rows = connection.execute(
        "SELECT a.*, s.canonical_name AS subject_name, o.canonical_name AS object_name "
        "FROM assertions a JOIN objects s ON s.id = a.subject_id "
        "LEFT JOIN objects o ON o.id = a.object_id "
        "WHERE (a.subject_id = ? OR a.object_id = ?) AND NOT EXISTS "
        "(SELECT 1 FROM assertions newer WHERE newer.supersedes_id = a.id) ORDER BY a.id LIMIT ?",
        (object_id, object_id, _DETAIL_ASSERTION_LIMIT + 1),
    ).fetchall()
    selected = rows[:_DETAIL_ASSERTION_LIMIT]
    output = [_assertion_view(row, object_id) for row in selected]
    if not output or not {"assertion_evidence", "observations"}.issubset(tables):
        return output, len(rows) > _DETAIL_ASSERTION_LIMIT
    marks = ",".join("?" for _ in output)
    evidence_rows = connection.execute(
        "WITH ranked AS ("
        "SELECT ae.assertion_id, ae.observation_id, ae.role, o.title, o.source_uri, "
        "o.source_kind, o.depth, o.source_published_at, o.observed_at, o.metadata_json, "
        "ROW_NUMBER() OVER (PARTITION BY ae.assertion_id ORDER BY ae.linked_at, ae.observation_id) AS row_no "
        "FROM assertion_evidence ae JOIN observations o ON o.id = ae.observation_id "
        f"WHERE ae.assertion_id IN ({marks})"
        ") SELECT * FROM ranked WHERE row_no <= ? ORDER BY assertion_id, row_no",
        (*[item["id"] for item in output], _EVIDENCE_PER_ASSERTION_LIMIT + 1),
    ).fetchall()
    evidence_by_assertion: dict[str, list[dict[str, Any]]] = {item["id"]: [] for item in output}
    evidence_truncated: set[str] = set()
    for evidence in evidence_rows:
        assertion_id = str(evidence["assertion_id"])
        if int(evidence["row_no"]) > _EVIDENCE_PER_ASSERTION_LIMIT:
            evidence_truncated.add(assertion_id)
            continue
        evidence_by_assertion[assertion_id].append(_evidence_view(evidence))
    for item in output:
        item["evidence"] = evidence_by_assertion[item["id"]]
        item["evidence_truncated"] = item["id"] in evidence_truncated
    return output, len(rows) > _DETAIL_ASSERTION_LIMIT


def _assertion_view(row: sqlite3.Row, object_id: str) -> dict[str, Any]:
    item = _bounded_row(_json_columns(dict(row)))
    item["direction"] = "subject" if item.get("subject_id") == object_id else "object"
    return item


def _evidence_view(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "observation_id": _text(row["observation_id"]),
        "role": _text(row["role"]),
        "observation": _observation_view(row),
    }


def _observation_view(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = dict(row)
    metadata = _metadata_object(fields.get("metadata_json"))
    raw_values = {
        "id": fields.get("observation_id", fields.get("id", "")),
        "title": fields.get("title"),
        "source_uri": fields.get("source_uri"),
        "source_kind": fields.get("source_kind"),
        "excerpt": fields.get("excerpt"),
        "depth": fields.get("depth"),
        "source_published_at": fields.get("source_published_at"),
        "observed_at": fields.get("observed_at"),
    }
    truncated_fields = [key for key, value in raw_values.items() if len(str(value or "")) > _TEXT_LIMIT]
    output = {
        **{key: _text(value) for key, value in raw_values.items()},
        "material_reliability": _material_reliability(metadata),
        "limitations": _limitations(metadata),
    }
    if truncated_fields:
        output["text_truncated_fields"] = truncated_fields
    return output


def _object_inquiries(
    connection: sqlite3.Connection, object_id: str, tables: set[str]
) -> tuple[list[dict[str, Any]], bool]:
    if "inquiries" not in tables:
        return [], False
    rows = connection.execute(
        "SELECT * FROM inquiries WHERE subject_id = ? "
        "ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'dormant' THEN 1 ELSE 2 END, id LIMIT ?",
        (object_id, _DETAIL_INQUIRY_LIMIT + 1),
    ).fetchall()
    return [_bounded_row(dict(row)) for row in rows[:_DETAIL_INQUIRY_LIMIT]], len(
        rows
    ) > _DETAIL_INQUIRY_LIMIT


def _object_observations(
    connection: sqlite3.Connection, object_id: str, tables: set[str]
) -> tuple[list[dict[str, Any]], bool]:
    if "observations" not in tables:
        return [], False
    links: list[str] = []
    parameters: list[str] = []
    if "object_observations" in tables:
        links.append("SELECT oo.observation_id, oo.role FROM object_observations oo WHERE oo.object_id = ?")
        parameters.append(object_id)
    if {"assertion_evidence", "assertions"}.issubset(tables):
        links.append(
            "SELECT ae.observation_id, 'evidence' AS role FROM assertion_evidence ae "
            "JOIN assertions a ON a.id = ae.assertion_id "
            "WHERE a.subject_id = ? OR a.object_id = ?"
        )
        parameters.extend((object_id, object_id))
    if not links:
        return [], False
    rows = connection.execute(
        "SELECT o.id, o.title, o.source_uri, o.source_kind, o.excerpt, o.depth, "
        "o.source_published_at, o.observed_at, o.metadata_json, links.role FROM ("
        + " UNION ".join(links)
        + ") links JOIN observations o ON o.id = links.observation_id "
        "ORDER BY o.observed_at DESC, o.id LIMIT ?",
        (*parameters, _DETAIL_OBSERVATION_LIMIT + 1),
    ).fetchall()
    return [
        _observation_view(row) | {"role": _text(row["role"])} for row in rows[:_DETAIL_OBSERVATION_LIMIT]
    ], len(rows) > _DETAIL_OBSERVATION_LIMIT


def _metadata_object(value: object) -> dict[str, Any]:
    parsed = _safe_json(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _material_reliability(metadata: Mapping[str, Any]) -> str:
    value = metadata.get("material_reliability")
    if value in {"source_direct", "confirmed", "best_effort", "automatic", "mixed", "unknown"}:
        return str(value)
    if metadata.get("subtitle_reliability") in {"confirmed", "best_effort"}:
        return str(metadata["subtitle_reliability"])
    method = str(metadata.get("acquisition_method", "")).casefold()
    if method.startswith("faster_whisper:") or "asr" in method:
        return "automatic"
    if metadata.get("modality") == "document_text":
        return "source_direct"
    return "unknown"


def _limitations(metadata: Mapping[str, Any]) -> list[str]:
    raw = metadata.get("limitations")
    if not isinstance(raw, list):
        return []
    output: list[str] = []
    total = 0
    truncated = False
    for value in raw:
        if not isinstance(value, str) or not value:
            truncated = True
            continue
        bounded = value[:160]
        truncated = truncated or bounded != value
        if len(output) >= 8 or total + len(bounded) > 640:
            truncated = True
            continue
        output.append(bounded)
        total += len(bounded)
    if truncated:
        while output and (len(output) >= 8 or total + len(_LIMITATIONS_TRUNCATED) > 640):
            total -= len(output.pop())
        output.append(_LIMITATIONS_TRUNCATED)
    return output


def _bounded_row(row: dict[str, Any]) -> dict[str, Any]:
    truncated = False
    for key, value in tuple(row.items()):
        row[key], value_truncated = _bounded_json(value)
        truncated = truncated or value_truncated
    if truncated:
        row["text_truncated"] = True
    return row


def _bounded_json(value: object, *, depth: int = 0) -> tuple[object, bool]:
    if isinstance(value, str):
        return value[:_TEXT_LIMIT], len(value) > _TEXT_LIMIT
    if depth >= 3 and isinstance(value, list):
        return [], True
    if depth >= 3 and isinstance(value, dict):
        return {}, True
    if isinstance(value, list):
        truncated = len(value) > 40
        bounded: list[Any] = []
        for item in value[:40]:
            projected, item_truncated = _bounded_json(item, depth=depth + 1)
            bounded.append(projected)
            truncated = truncated or item_truncated
        return bounded, truncated
    if isinstance(value, dict):
        truncated = len(value) > 40
        bounded_dict: dict[str, Any] = {}
        for key, item in list(value.items())[:40]:
            projected, item_truncated = _bounded_json(item, depth=depth + 1)
            bounded_dict[_text(key)] = projected
            truncated = truncated or item_truncated or len(str(key)) > _TEXT_LIMIT
        return bounded_dict, truncated
    return value, False


def _text(value: object, limit: int = _TEXT_LIMIT) -> str:
    return str(value or "")[:limit]


def _commit_view(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "commit_id": str(row["commit_id"]),
        "committed_at": str(row["committed_at"]),
        "receipt": _safe_json(row["receipt_json"], {}),
        "delta": _safe_json(row["delta_json"], {}),
    }


def _durable_diff(commits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = {
        "objects": [],
        "observations": [],
        "assertions": [],
        "inquiries": [],
        "resolve_inquiries": [],
        "observation_links": [],
    }
    for commit in commits:
        delta = commit.get("delta")
        if not isinstance(delta, Mapping):
            continue
        for key in categories:
            values = delta.get(key, [])
            if isinstance(values, list):
                categories[key].extend(item for item in values if isinstance(item, Mapping))
    return {"commits": list(commits), **categories}


def _sum_rows(rows: Sequence[sqlite3.Row], key: str) -> int | float:
    values = [row[key] for row in rows if key in tuple(row.keys()) and row[key] is not None]
    total = sum(values, 0)
    return float(total) if any(isinstance(value, float) for value in values) else int(total)


__all__ = ["ReadOnlyInspection"]
