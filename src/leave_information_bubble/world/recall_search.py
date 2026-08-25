"""Read-side name recall and candidate layering for WorldRecall.search (strict leaf)."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from typing import Any

from .contracts import RecallClock
from .recall_common import _objects_by_id, _rows, _usage_evidence_counts
from .similarity import bigrams, jaccard
from .store import alias_lookup_forms, normalize_identity_alias

_MAX_QUERY_TERMS = 12
# Read-side identity bucketing line: a query term whose bigram-jaccard against
# a stored object name reaches this value (without being an exact alias) is a
# possible_match candidate. G5b-2 retired the world-agent ghost-candidate
# search and its graph.py _GHOST_CANDIDATE_SIMILARITY_THRESHOLD pin target, so
# this is now the single recall threshold (0.20), pinned by
# test_search_possible_match_threshold_is_the_single_recall_line so it cannot
# drift.
_MATCH_SIMILARITY_THRESHOLD = 0.20

# memory_search candidate layers (read-side name recall, contract §7.3 /
# task 4.1). Each matched object gets ONE entry labeled with its highest
# layer; entries sort by layer priority, and every entry is a candidate —
# exact identity never triggers writes or merges. No confidence float is
# produced anywhere: the agent disambiguates from the layer plus the
# object's domain hints.
CANDIDATE_IDENTITY_ALIAS_EXACT = "identity_alias_exact"
CANDIDATE_CANONICAL_EXACT = "canonical_exact"
CANDIDATE_NAME_USAGE = "name_usage"
CANDIDATE_LEGACY_NAME = "legacy_name"
CANDIDATE_POSSIBLE_MATCH = "possible_match"
CANDIDATE_TEXT_MATCH = "text_match"
_CANDIDATE_LAYER_ORDER = (
    CANDIDATE_IDENTITY_ALIAS_EXACT,
    CANDIDATE_CANONICAL_EXACT,
    CANDIDATE_NAME_USAGE,
    CANDIDATE_LEGACY_NAME,
    CANDIDATE_POSSIBLE_MATCH,
    CANDIDATE_TEXT_MATCH,
)
_LAYER_RANK = {kind: index for index, kind in enumerate(_CANDIDATE_LAYER_ORDER)}

# Bounded pool for the name_usage assertion scan: rows are pre-filtered in
# SQL and exact-normalized in Python, so the pool only needs to hold the
# query-shaped rows plus headroom for several objects' usage assertions.
_NAME_USAGE_POOL = 64

# R7 candidate payload bound: the matched name-surface text a candidate
# reports inline stays at 80 characters (with an explicit truncation flag
# when cut), so the disambiguation hint can never balloon the read envelope.
_MATCH_SURFACE_LIMIT = 80

# FTS5's unicode61 tokenizer keeps every consecutive CJK run as ONE token, so a
# phrase query built from a long CJK run can only match text whose run is byte
# identical (audit B-1: mission_relevant never hit). Queries are therefore
# expanded to sliding two-character grams (pure Python, offline) and matched as
# FTS5 prefix phrases: "英雄"* hits stored tokens like "英雄联盟转会市场".
_LATIN_TOKEN = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[一-鿿]+")


def _identity_terms(query: str) -> list[str]:
    """Split a query into identity-comparison terms, keeping CJK runs whole.

    FTS matching gram-expands CJK runs (``_terms``), but identity bucketing
    compares against stored aliases, so runs stay whole here: the query
    "英雄联盟" is one term that can equal the alias "英雄联盟", never one of
    its grams. The whole query is appended as the most specific term so a
    multiword alias — including the legacy space-preserving form
    "t1 stars 男团出道" that ``store.alias_lookup_forms`` also serves — is
    reachable exactly the way the committer reaches it.
    """
    folded = query.casefold()
    tokens = list(_LATIN_TOKEN.findall(folded))
    tokens.extend(run for run in _CJK_RUN.findall(folded))
    terms = list(dict.fromkeys(tokens))[:_MAX_QUERY_TERMS]
    whole = query.strip()
    if whole:
        # the literal query text (not casefolded): normalization happens
        # inside alias_lookup_forms / bigrams, so match_term stays the
        # user-visible spelling
        terms.append(whole)
    return list(dict.fromkeys(terms))


def _object_alias_sets(
    connection: sqlite3.Connection, object_ids: list[str]
) -> dict[str, set[str]]:
    """Map each object id to the name forms used ONLY for fuzzy similarity.

    Consistent with the identity model, this merged set is the similarity
    name pool for ``possible_match`` only: ACTIVE ``identity_aliases``
    forms, any pre-existing legacy ``object_aliases`` rows (read-only
    history, stored exactly as written), and the canonical name in its
    identity-normalized form. Exact identity claims never flow through this
    set — the ``identity_alias_exact`` / ``canonical_exact`` /
    ``name_usage`` / ``legacy_name`` layers each compare the query against
    its own surface with its own normalizer, and the canonical name is
    covered exactly by ``canonical_exact`` (the normalized comparison), not
    through this pool.
    """
    if not object_ids:
        return {}
    aliases = identity_alias_forms(connection, object_ids)
    legacy = legacy_alias_forms(connection, object_ids)
    for identifier, forms in legacy.items():
        aliases[identifier].update(forms)
    marks = ", ".join("?" for _ in object_ids)
    for row in connection.execute(
        f"SELECT id, canonical_name FROM objects WHERE id IN ({marks})",
        object_ids,
    ).fetchall():
        try:
            aliases[str(row["id"])].add(normalize_identity_alias(str(row["canonical_name"])))
        except ValueError:
            # a degenerate canonical name never participates in identity
            continue
    return aliases


def identity_alias_forms(
    connection: sqlite3.Connection, object_ids: list[str]
) -> dict[str, set[str]]:
    """Map each object id to its ACTIVE identity alias forms (identity layer)."""
    if not object_ids:
        return {}
    marks = ", ".join("?" for _ in object_ids)
    forms: dict[str, set[str]] = {identifier: set() for identifier in object_ids}
    for row in connection.execute(
        "SELECT object_id, normalized_alias FROM identity_aliases"
        f" WHERE status = 'active' AND object_id IN ({marks})",
        object_ids,
    ).fetchall():
        forms[str(row["object_id"])].add(str(row["normalized_alias"]))
    return forms


def legacy_alias_forms(
    connection: sqlite3.Connection, object_ids: list[str]
) -> dict[str, set[str]]:
    """Map each object id to its legacy ``object_aliases`` rows (legacy layer)."""
    if not object_ids:
        return {}
    marks = ", ".join("?" for _ in object_ids)
    forms: dict[str, set[str]] = {identifier: set() for identifier in object_ids}
    for row in connection.execute(
        f"SELECT object_id, normalized_alias FROM object_aliases WHERE object_id IN ({marks})",
        object_ids,
    ).fetchall():
        forms[str(row["object_id"])].add(str(row["normalized_alias"]))
    return forms


def _normalized_identity_forms(terms: list[str]) -> set[str]:
    """Normalize query terms with the identity normalizer, skipping degenerate ones."""
    forms: set[str] = set()
    for term in terms:
        try:
            forms.add(normalize_identity_alias(term))
        except ValueError:
            continue
    return forms


def _matching_name_usage(
    connection: sqlite3.Connection,
    identity_terms: list[str],
    limit: int,
    moment: str | None,
    clock: RecallClock,
) -> list[dict[str, Any]]:
    """Return current name_usage assertions whose literal is an exact query term.

    A name_usage assertion is an explicit Agent-submitted claim (it has an
    assertion id, qualifiers, time, and evidence), so it is recalled as its
    own candidate layer, never conflated with an identity alias. SQL
    pre-filters rows whose literal text contains one of the query's legacy
    or identity-normalized forms (the same dual-form discipline
    ``alias_lookup_forms`` applies to the legacy table); Python then
    requires exact equality under ``normalize_identity_alias``, so a
    whitespace variant can never collapse into a match. The pool is bounded
    by ``_NAME_USAGE_POOL``; as-of and supersession filters mirror
    ``_matching_assertions``.
    """
    patterns: set[str] = set()
    for term in identity_terms:
        for form in alias_lookup_forms(term):
            if form:
                patterns.add(form)
        try:
            patterns.add(normalize_identity_alias(term))
        except ValueError:
            continue
    if not patterns:
        return []
    like = " OR ".join("json_extract(a.literal_json, '$') LIKE ?" for _ in patterns)
    values: list[Any] = [f"%{pattern}%" for pattern in patterns]
    time_filter = ""
    if moment is not None:
        if clock is RecallClock.KNOWLEDGE:
            time_filter = " AND EXISTS (SELECT 1 FROM world_audit, json_each(delta_json, '$.assertions') item WHERE json_extract(item.value, '$.id') = a.id AND committed_at <= ?)"  # noqa: E501
            values.append(moment)
        else:
            time_filter = " AND (a.event_time_start IS NULL OR a.event_time_start <= ?) AND (a.event_time_end IS NULL OR a.event_time_end >= ?)"  # noqa: E501
            values.extend((moment, moment))
    retire_sql, retire_values = _supersedes_filter(moment, clock)
    values.extend(retire_values)
    statement = (
        "SELECT a.* FROM assertions a "
        "WHERE a.predicate = 'name_usage' AND json_type(a.literal_json, '$') = 'text'"
        f" AND ({like}){time_filter}{retire_sql} ORDER BY a.id LIMIT ?"
    )
    values.append(max(limit, _NAME_USAGE_POOL))
    rows = _rows(connection.execute(statement, values).fetchall())
    forms = _normalized_identity_forms(identity_terms)
    matched: list[dict[str, Any]] = []
    for row in rows:
        try:
            literal = json.loads(row["literal_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(literal, str):
            continue
        try:
            normalized = normalize_identity_alias(literal)
        except ValueError:
            continue
        if normalized in forms:
            matched.append(row)
    return matched


def _merge_usage_subjects(
    connection: sqlite3.Connection,
    objects: list[dict[str, Any]],
    usage_assertions: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Append objects that surfaced only through a name_usage assertion."""
    present = {str(row["id"]) for row in objects}
    subjects = [
        str(row["subject_id"])
        for row in usage_assertions
        if row.get("subject_id") and str(row["subject_id"]) not in present
    ]
    if not subjects:
        return objects
    extras = _objects_by_id(connection, list(dict.fromkeys(subjects)))
    return (objects + extras)[:limit]


def _candidate_surface(
    kind: str,
    row: Mapping[str, Any],
    best_term: str,
    *,
    normalized_term: str | None,
    canonical_norm: str | None,
    usage_literals: Mapping[str, dict[str, Any]],
    legacy_set: set[str],
    raw_identity: Mapping[str, str],
    names: list[str],
) -> str | None:
    """Return the raw name-surface text where the winning layer hit (R7).

    Each exact layer surfaces the stored spelling it matched against: the
    identity layer the raw alias as written, the canonical layer the
    canonical name, the name_usage layer the assertion literal, and the
    legacy layer its stored (normalized) alias form. The fuzzy layer
    surfaces the stored form that reached the similarity line (the raw
    spelling when one exists). A ``text_match`` has no name hit, so it
    surfaces nothing.
    """
    if kind == CANDIDATE_IDENTITY_ALIAS_EXACT and normalized_term:
        return raw_identity.get(normalized_term, best_term)
    if kind == CANDIDATE_CANONICAL_EXACT:
        return str(row["canonical_name"]) if row.get("canonical_name") else best_term
    if kind == CANDIDATE_NAME_USAGE and normalized_term:
        usage_row = usage_literals.get(normalized_term)
        if usage_row is not None:
            try:
                literal = json.loads(usage_row["literal_json"])
            except (TypeError, ValueError):
                literal = None
            if isinstance(literal, str):
                return literal
        return best_term
    if kind == CANDIDATE_LEGACY_NAME:
        matched_form = next(
            (form for form in alias_lookup_forms(best_term) if form in legacy_set),
            normalized_term,
        )
        return matched_form or best_term
    if kind == CANDIDATE_POSSIBLE_MATCH:
        probe = bigrams(best_term)
        if probe:
            best_name = max(names, key=lambda name: jaccard(probe, bigrams(name)))
            if jaccard(probe, bigrams(best_name)) >= _MATCH_SIMILARITY_THRESHOLD:
                return raw_identity.get(best_name, best_name)
        return best_term
    return None


def _match_candidates(
    connection: sqlite3.Connection,
    objects: list[dict[str, Any]],
    terms: list[str],
    usage_assertions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Label each matched object with its highest name-recall layer.

    Layer precedence per object, most specific first (contract §7.3 / task
    4.1):

    - ``identity_alias_exact``: the whole query or one of its terms,
      folded by ``normalize_identity_alias`` (NFC → casefold → strip →
      whitespace collapse), equals one of the object's ACTIVE
      ``identity_aliases`` forms.
    - ``canonical_exact``: the normalized term equals the object's
      identity-normalized canonical name. Multiple objects may share one
      canonical name, so this layer can return several entries.
    - ``name_usage``: the normalized term equals the literal of one of the
      object's current name_usage assertions; the entry carries the
      assertion id, literal, qualifiers, time, and evidence counts.
    - ``legacy_name``: one of ``alias_lookup_forms(term)`` (the legacy
      dual-form lookup) equals a legacy ``object_aliases`` row. These
      entries explicitly carry ``identity_authority=false`` — legacy rows
      never claim identity.
    - ``possible_match``: no exact layer, but a term's bigram-jaccard
      against the canonical name or a stored form reaches
      ``_MATCH_SIMILARITY_THRESHOLD`` (0.20, the graph's ghost-candidate
      line).
    - ``text_match``: the object's FTS row matched, but no term reached
      any other layer; the query's first term is reported as context.

    Multi-term queries pick the best matching term per object: a higher
    layer always beats a lower one regardless of term length, and within a
    layer the longest (most specific) term wins, ties to the first in query
    order. All layers normalize the query with the identity normalizer; the
    legacy layer is the only one additionally served by
    ``alias_lookup_forms``. Entries sort by layer priority (then result
    order), and an exact identity hit is only a candidate — never a write or
    merge authorization.

    Returns:
        One entry per object, in layer-priority order:
        ``{"id", "kind", "match_term", "domain_hints"}``, plus
        ``identity_authority=false`` on ``legacy_name`` entries and a
        ``name_usage`` context block on ``name_usage`` entries. Every layer
        except ``text_match`` additionally carries ``match_surface`` — the
        raw name-surface text where the hit landed, bounded to 80 characters
        with an explicit ``match_surface_truncated`` flag when cut (R7) —
        and objects with an event span carry ``time_range``
        (``{"start", "end"}``).

    """
    if not objects:
        return []
    object_ids = [str(row["id"]) for row in objects]
    present = set(object_ids)
    identity_forms = identity_alias_forms(connection, object_ids)
    legacy_forms = legacy_alias_forms(connection, object_ids)
    fuzzy_names = _object_alias_sets(connection, object_ids)
    marks = ", ".join("?" for _ in object_ids)
    # R7: the candidate surfaces the RAW stored name text where the hit
    # landed (the identity layer stores raw_alias as written), so the model
    # sees the user-visible spelling instead of a normalized fold. The
    # legacy layer stores only the normalized form, so its surface is that
    # form.
    raw_identity: dict[str, str] = {}
    for row in connection.execute(
        "SELECT raw_alias, normalized_alias FROM identity_aliases"
        f" WHERE status = 'active' AND object_id IN ({marks})",
        object_ids,
    ).fetchall():
        raw_identity.setdefault(str(row["normalized_alias"]), str(row["raw_alias"]))
    usage_literals_by_subject: dict[str, dict[str, dict[str, Any]]] = {}
    usage_ids: list[str] = []
    for row in usage_assertions:
        subject = str(row.get("subject_id") or "")
        if subject not in present:
            continue
        try:
            literal = json.loads(row["literal_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(literal, str):
            continue
        try:
            normalized = normalize_identity_alias(literal)
        except ValueError:
            continue
        usage_literals_by_subject.setdefault(subject, {}).setdefault(normalized, row)
        if row["id"] not in usage_ids:
            usage_ids.append(row["id"])
    evidence_counts = _usage_evidence_counts(connection, usage_ids)
    entries: list[tuple[int, int, dict[str, Any]]] = []
    for position, row in enumerate(objects):
        identifier = str(row["id"])
        canonical_norm: str | None = None
        if row.get("canonical_name"):
            try:
                canonical_norm = normalize_identity_alias(str(row["canonical_name"]))
            except ValueError:
                canonical_norm = None
        identity_set = identity_forms.get(identifier, set())
        legacy_set = legacy_forms.get(identifier, set())
        usage_literals = usage_literals_by_subject.get(identifier, {})
        names = [str(row["canonical_name"])] if row.get("canonical_name") else []
        names.extend(fuzzy_names.get(identifier, set()))
        best_rank = _LAYER_RANK[CANDIDATE_TEXT_MATCH]
        best_term = terms[0] if terms else ""
        for term in terms:
            normalized = normalize_identity_alias(term) if term else None
            rank: int | None = None
            if normalized and normalized in identity_set:
                rank = _LAYER_RANK[CANDIDATE_IDENTITY_ALIAS_EXACT]
            elif (
                rank is None
                and normalized is not None
                and canonical_norm is not None
                and normalized == canonical_norm
            ):
                rank = _LAYER_RANK[CANDIDATE_CANONICAL_EXACT]
            elif rank is None and normalized and normalized in usage_literals:
                rank = _LAYER_RANK[CANDIDATE_NAME_USAGE]
            elif rank is None and set(alias_lookup_forms(term)) & legacy_set:
                rank = _LAYER_RANK[CANDIDATE_LEGACY_NAME]
            elif rank is None:
                probe = bigrams(term)
                if probe and any(
                    jaccard(probe, bigrams(name)) >= _MATCH_SIMILARITY_THRESHOLD for name in names
                ):
                    rank = _LAYER_RANK[CANDIDATE_POSSIBLE_MATCH]
            if rank is not None and (rank < best_rank or (rank == best_rank and len(term) > len(best_term))):
                best_rank = rank
                best_term = term
        kind = _CANDIDATE_LAYER_ORDER[best_rank]
        entry: dict[str, Any] = {
            "id": identifier,
            "kind": kind,
            "match_term": best_term,
            "domain_hints": _domain_hints(row),
        }
        if kind == CANDIDATE_NAME_USAGE:
            entry["name_usage"] = _name_usage_context(
                usage_literals.get(normalize_identity_alias(best_term)),
                evidence_counts,
            )
        elif kind == CANDIDATE_LEGACY_NAME:
            entry["identity_authority"] = False
        surface = _candidate_surface(
            kind,
            row,
            best_term,
            normalized_term=normalize_identity_alias(best_term) if best_term else None,
            canonical_norm=canonical_norm,
            usage_literals=usage_literals,
            legacy_set=legacy_set,
            raw_identity=raw_identity,
            names=names,
        )
        if surface is not None:
            if len(surface) > _MATCH_SURFACE_LIMIT:
                surface = surface[:_MATCH_SURFACE_LIMIT - 1] + "…"
                entry["match_surface_truncated"] = True
            entry["match_surface"] = surface
        if row.get("event_time_start") or row.get("event_time_end"):
            entry["time_range"] = {
                "start": row.get("event_time_start"),
                "end": row.get("event_time_end"),
            }
        entries.append((best_rank, position, entry))
    entries.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in entries]


def _name_usage_context(
    row: dict[str, Any] | None,
    evidence_counts: Mapping[str, dict[str, int]],
) -> dict[str, Any]:
    """Render the explainable context block for one name_usage candidate."""
    if row is None:
        return {}
    literal = json.loads(row["literal_json"]) if row.get("literal_json") else None
    qualifiers = json.loads(row["qualifiers_json"]) if row.get("qualifiers_json") else {}
    return {
        "assertion_id": row["id"],
        "literal": literal if isinstance(literal, str) else "",
        "qualifiers": qualifiers if isinstance(qualifiers, dict) else {},
        "time": row.get("event_time_start"),
        "evidence_counts": evidence_counts.get(str(row["id"]), {}),
    }


def _domain_hints(row: Mapping[str, Any]) -> list[str]:
    """Parse an objects row's domain_hints_json into a list, tolerating garbage."""
    raw = row.get("domain_hints_json")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _object_filters(
    *,
    kind: str | None,
    time_from: str | None,
    time_to: str | None,
    has_participants: bool | None,
    assertion_count_min: int | None,
    assertion_count_max: int | None,
) -> tuple[str, list[object]]:
    """Build the structured filter SQL fragment shared by search and count.

    R9: the same fragment serves ``WorldRecall.search`` today and
    ``graph_inspect`` structured queries later, so both surfaces enumerate
    identical dimensions. NULL event bounds are unbounded, and the time
    window is an overlap test against the object's ``[event_time_start,
    event_time_end]`` span. ``has_participants`` tests for a current
    (non-superseded) ``has_participant`` edge anchored on the object, and
    the assertion count counts current subject-anchored assertions — the
    same aggregation ``expand`` uses.
    """
    clauses: list[str] = []
    values: list[object] = []
    if kind is not None:
        clauses.append("o.kind = ?")
        values.append(kind)
    if time_from is not None:
        clauses.append("(o.event_time_end IS NULL OR o.event_time_end >= ?)")
        values.append(time_from)
    if time_to is not None:
        clauses.append("(o.event_time_start IS NULL OR o.event_time_start <= ?)")
        values.append(time_to)
    if has_participants is not None:
        edge = (
            "EXISTS (SELECT 1 FROM assertions p WHERE p.subject_id = o.id"
            " AND p.predicate = 'has_participant' AND NOT EXISTS"
            " (SELECT 1 FROM assertions s WHERE s.supersedes_id = p.id))"
        )
        clauses.append(edge if has_participants else f"NOT {edge}")
    if assertion_count_min is not None or assertion_count_max is not None:
        count = (
            "(SELECT COUNT(*) FROM assertions a WHERE a.subject_id = o.id"
            " AND NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id))"
        )
        if assertion_count_min is not None:
            clauses.append(f"{count} >= ?")
            values.append(assertion_count_min)
        if assertion_count_max is not None:
            clauses.append(f"{count} <= ?")
            values.append(assertion_count_max)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), values


def _assertion_filters(
    *,
    predicate: str | None,
    time_from: str | None,
    time_to: str | None,
) -> tuple[str, list[object]]:
    """Build the structured assertion filter fragment (search and count)."""
    clauses: list[str] = []
    values: list[object] = []
    if predicate is not None:
        clauses.append("a.predicate = ?")
        values.append(predicate)
    if time_from is not None:
        clauses.append("(a.event_time_end IS NULL OR a.event_time_end >= ?)")
        values.append(time_from)
    if time_to is not None:
        clauses.append("(a.event_time_start IS NULL OR a.event_time_start <= ?)")
        values.append(time_to)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), values


def _matching_objects(
    connection: sqlite3.Connection,
    terms: list[str],
    limit: int,
    moment: str | None,
    clock: RecallClock,
    *,
    kind: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    has_participants: bool | None = None,
    assertion_count_min: int | None = None,
    assertion_count_max: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    match = " OR ".join(_phrase(term) for term in terms)
    hints = " OR ".join("o.domain_hints_json LIKE ?" for _ in terms)
    time_filter = ""
    values = [match]
    if moment is not None and clock is RecallClock.KNOWLEDGE:
        time_filter = " AND EXISTS (SELECT 1 FROM world_audit, json_each(delta_json, '$.objects') item WHERE json_extract(item.value, '$.id') = o.id AND committed_at <= ?)"  # noqa: E501
        values.append(moment)
    filter_sql, filter_values = _object_filters(
        kind=kind,
        time_from=time_from,
        time_to=time_to,
        has_participants=has_participants,
        assertion_count_min=assertion_count_min,
        assertion_count_max=assertion_count_max,
    )
    values.extend(filter_values)
    statement = (
        "SELECT DISTINCT o.* FROM objects o JOIN objects_fts f ON f.id = o.id "
        f"WHERE objects_fts MATCH ?{time_filter}{filter_sql} "
        f"ORDER BY CASE WHEN {hints} THEN 0 ELSE 1 END, o.id LIMIT ? OFFSET ?"
    )
    values.extend(f"%{term}%" for term in terms)
    values.extend((limit, offset))
    return _rows(connection.execute(statement, values).fetchall())


def _matching_objects_count(
    connection: sqlite3.Connection,
    terms: list[str],
    moment: str | None,
    clock: RecallClock,
    *,
    kind: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    has_participants: bool | None = None,
    assertion_count_min: int | None = None,
    assertion_count_max: int | None = None,
) -> int:
    """Total object matches for the same query + filters (count mode, no rows)."""
    match = " OR ".join(_phrase(term) for term in terms)
    time_filter = ""
    values = [match]
    if moment is not None and clock is RecallClock.KNOWLEDGE:
        time_filter = " AND EXISTS (SELECT 1 FROM world_audit, json_each(delta_json, '$.objects') item WHERE json_extract(item.value, '$.id') = o.id AND committed_at <= ?)"  # noqa: E501
        values.append(moment)
    filter_sql, filter_values = _object_filters(
        kind=kind,
        time_from=time_from,
        time_to=time_to,
        has_participants=has_participants,
        assertion_count_min=assertion_count_min,
        assertion_count_max=assertion_count_max,
    )
    values.extend(filter_values)
    statement = (
        "SELECT COUNT(DISTINCT o.id) FROM objects o JOIN objects_fts f ON f.id = o.id "
        f"WHERE objects_fts MATCH ?{time_filter}{filter_sql}"
    )
    row = connection.execute(statement, values).fetchone()
    return int(row[0]) if row else 0


def _matching_assertions(
    connection: sqlite3.Connection,
    terms: list[str],
    limit: int,
    moment: str | None,
    clock: RecallClock,
    *,
    predicate: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    match = " OR ".join(_phrase(term) for term in terms)
    time_filter = ""
    values = [match]
    if moment is not None:
        if clock is RecallClock.WORLD:
            time_filter = " AND (a.event_time_start IS NULL OR a.event_time_start <= ?) AND (a.event_time_end IS NULL OR a.event_time_end >= ?)"  # noqa: E501
            values.extend((moment, moment))
        else:
            time_filter = " AND EXISTS (SELECT 1 FROM world_audit, json_each(delta_json, '$.assertions') item WHERE json_extract(item.value, '$.id') = a.id AND committed_at <= ?)"  # noqa: E501
            values.append(moment)
    retire_sql, retire_values = _supersedes_filter(moment, clock)
    values.extend(retire_values)
    filter_sql, filter_values = _assertion_filters(
        predicate=predicate, time_from=time_from, time_to=time_to
    )
    values.extend(filter_values)
    statement = (
        "SELECT DISTINCT a.* FROM assertions a JOIN assertions_fts f ON f.id = a.id "
        f"WHERE assertions_fts MATCH ?{time_filter}{retire_sql}{filter_sql} "
        "ORDER BY a.id LIMIT ? OFFSET ?"
    )
    values.extend((limit, offset))
    return _rows(connection.execute(statement, values).fetchall())


def _matching_assertions_count(
    connection: sqlite3.Connection,
    terms: list[str],
    moment: str | None,
    clock: RecallClock,
    *,
    predicate: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
) -> int:
    """Total assertion matches for the same query + filters (count mode)."""
    match = " OR ".join(_phrase(term) for term in terms)
    time_filter = ""
    values = [match]
    if moment is not None:
        if clock is RecallClock.WORLD:
            time_filter = " AND (a.event_time_start IS NULL OR a.event_time_start <= ?) AND (a.event_time_end IS NULL OR a.event_time_end >= ?)"  # noqa: E501
            values.extend((moment, moment))
        else:
            time_filter = " AND EXISTS (SELECT 1 FROM world_audit, json_each(delta_json, '$.assertions') item WHERE json_extract(item.value, '$.id') = a.id AND committed_at <= ?)"  # noqa: E501
            values.append(moment)
    retire_sql, retire_values = _supersedes_filter(moment, clock)
    values.extend(retire_values)
    filter_sql, filter_values = _assertion_filters(
        predicate=predicate, time_from=time_from, time_to=time_to
    )
    values.extend(filter_values)
    statement = (
        "SELECT COUNT(DISTINCT a.id) FROM assertions a JOIN assertions_fts f ON f.id = a.id "
        f"WHERE assertions_fts MATCH ?{time_filter}{retire_sql}{filter_sql}"
    )
    row = connection.execute(statement, values).fetchone()
    return int(row[0]) if row else 0


def _supersedes_filter(moment: str | None, clock: RecallClock) -> tuple[str, list[object]]:
    """Return SQL and parameters that retire assertions superseded by a committed one.

    An assertion leaves the recall surface when a later assertion names it in
    ``supersedes_id`` (the new assertion owns the pointer; the old row is
    never touched). Recalled cards then reflect the current knowledge graph
    instead of silently accumulating retired claims (audit A3/A5-F1: the
    cognitive graph "膨胀"). History stays reachable through the ``changes``
    delta log and by-id ``evidence`` lookups, which are deliberately unfiltered.

    As-of knowledge queries stay precise: a superseding assertion only retires
    the old one when the replacement itself was committed by the moment, so a
    snapshot before the replacement commit still shows the assertion that was
    current then. World-clock as-of queries are event-time snapshots, so
    knowledge retirement does not apply — what the world was like at an event
    time does not change when a later claim supersedes the knowledge of it.

    Args:
        moment: The as-of moment (UTC ISO) or None for current-state recall.
        clock: Which clock semantics the query uses.

    Returns:
        A SQL fragment (starting with `` AND ...``) and its bound parameters.

    """
    if clock is RecallClock.WORLD and moment is not None:
        return "", []
    if moment is None:
        return " AND NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id)", []
    return (
        " AND NOT EXISTS (SELECT 1 FROM assertions s WHERE s.supersedes_id = a.id"
        " AND EXISTS (SELECT 1 FROM world_audit, json_each(delta_json, '$.assertions') item"
        " WHERE json_extract(item.value, '$.id') = s.id AND committed_at <= ?))",
        [moment],
    )


def _matching_inquiries(
    connection: sqlite3.Connection, terms: list[str], limit: int, moment: str | None, clock: RecallClock
) -> list[dict[str, Any]]:
    matches = " OR ".join("prompt LIKE ? OR rationale LIKE ?" for _ in terms)
    values = [value for term in terms for value in (f"%{term}%", f"%{term}%")]
    if moment is not None and clock is RecallClock.KNOWLEDGE:
        time_filter = " AND EXISTS (SELECT 1 FROM world_audit, json_each(delta_json, '$.inquiries') item WHERE json_extract(item.value, '$.id') = inquiries.id AND committed_at <= ?)"  # noqa: E501
        values.append(moment)
    else:
        time_filter = ""
    statement = (
        f"SELECT * FROM inquiries WHERE status = 'open' AND ({matches}){time_filter} ORDER BY id LIMIT ?"  # noqa: E501
    )
    values.append(limit)
    return _rows(connection.execute(statement, values).fetchall())


def _matching_inquiries_count(
    connection: sqlite3.Connection,
    terms: list[str],
    moment: str | None,
    clock: RecallClock,
) -> int:
    """Total open-inquiry matches for the same query (count mode)."""
    matches = " OR ".join("prompt LIKE ? OR rationale LIKE ?" for _ in terms)
    values = [value for term in terms for value in (f"%{term}%", f"%{term}%")]
    if moment is not None and clock is RecallClock.KNOWLEDGE:
        time_filter = " AND EXISTS (SELECT 1 FROM world_audit, json_each(delta_json, '$.inquiries') item WHERE json_extract(item.value, '$.id') = inquiries.id AND committed_at <= ?)"  # noqa: E501
        values.append(moment)
    else:
        time_filter = ""
    statement = (
        f"SELECT COUNT(*) FROM inquiries WHERE status = 'open' AND ({matches}){time_filter}"  # noqa: E501
    )
    row = connection.execute(statement, values).fetchone()
    return int(row[0]) if row else 0


def _terms(query: str) -> list[str]:
    """Split a query into FTS5-matchable terms, gram-expanding CJK runs.

    The FTS5 unicode61 tokenizer stores each consecutive CJK run as one
    token, so a raw long run (e.g. the mission text) can never hit stored
    content (audit B-1). Latin/digit runs stay whole; every CJK run is
    expanded to its sliding two-character grams (pure Python, offline), and
    the caller matches those grams as prefix phrases. Mixed tokens such as
    "2026EWC英雄联盟总决赛DK夺冠" split into their latin and CJK parts so
    neither side is polluted by the other's grams.

    Args:
        query: The raw user query text.

    Returns:
        Up to ``_MAX_QUERY_TERMS`` unique terms, whole for non-CJK text and
        two-character grams for CJK runs (single-char runs kept as-is).

    """
    folded = query.casefold()
    terms: list[str] = []
    for token in _LATIN_TOKEN.findall(folded):
        terms.append(token)
    for run in _CJK_RUN.findall(folded):
        if len(run) > 1:
            terms.extend(_bigrams(run))
        else:
            terms.append(run)
    return list(dict.fromkeys(terms))[:_MAX_QUERY_TERMS]


def _bigrams(run: str) -> list[str]:
    """Yield the sliding two-character grams of a CJK run."""
    return [run[index : index + 2] for index in range(len(run) - 1)]


def _phrase(term: str) -> str:
    """Render one term as an FTS5 phrase, prefix-matched for CJK grams.

    A two-character gram matches stored tokens exactly when the stored CJK
    run is itself two characters, and as a token prefix (``"英雄"*``) when the
    run is longer ("英雄联盟转会市场"). Latin terms stay exact phrases.
    """
    if _CJK_RUN.fullmatch(term):
        return f'"{term}"*'
    return f'"{term}"'
