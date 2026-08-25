# Split from tests/world/test_world_recall.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    InquiryInput,
    ObjectInput,
    ObjectKind,
    RecallClock,
    WorldRecall,
    WorldStore,
)
from tests.world._recall_helpers import (
    NOW,
    _assertion,
    _event_object,
    _object,
    _observation,
    _participant_edge,
    _TracingWorldStore,
)


def test_search_finds_alias_assertion_text_and_active_inquiry(tmp_path: pytest.TempPathFactory) -> None:
    """Catch retrieval that misses one of its object, assertion, or inquiry sources."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("orbital-project", aliases=["skybridge"])],
            observations=[_observation("observation-1")],
            assertions=[_assertion("assertion-1", "orbital-project", literal="payload reaches orbit")],
            inquiries=[
                InquiryInput(
                    id="inquiry-1",
                    subject_id="orbital-project",
                    prompt="Who funds the skybridge?",
                    rationale="Funding is unknown",
                )
            ],
        ),
        "searchable-world",
    )

    bundle = WorldRecall(store).search("skybridge orbit funding")

    assert [row["id"] for row in bundle.anchor_objects] == ["orbital-project"]
    assert [row["id"] for row in bundle.assertions] == ["assertion-1"]
    assert [row["id"] for row in bundle.inquiries] == ["inquiry-1"]


def test_search_uses_object_fts_for_partial_multiword_canonical_name(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch object recall that requires a convenience alias for a name fragment."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(objects=[_object("ig-vs-lng-playoff-series")]),
        "multiword-object",
    )

    bundle = WorldRecall(store).search("IG vs LNG")

    assert [row["id"] for row in bundle.anchor_objects] == ["ig-vs-lng-playoff-series"]


def test_legacy_only_name_remains_searchable_but_never_active_identity(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """F1/Task 1: legacy-only object_aliases stay searchable without identity power.

    A pre-existing legacy ``object_aliases`` row survives unrelated writes,
    keeps feeding the FTS search surface through the read-only search-alias
    view, and never appears in the active identity index.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(objects=[_object("legacy-searchable")]),
        "seed-legacy-search",
    )
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO object_aliases (normalized_alias, object_id) VALUES (?, ?)",
            ("old-form", "legacy-searchable"),
        )
    # Any later write refreshes the FTS aliases column from the combined
    # read/search view (active identity aliases ∪ legacy rows).
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="legacy-searchable",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Legacy Searchable",
                    aliases=["new-form"],
                    expected_version=1,
                )
            ]
        ),
        "touch-legacy-search",
    )

    bundle = WorldRecall(store).search("old-form")

    assert [row["id"] for row in bundle.anchor_objects] == ["legacy-searchable"]
    with store.read_connection() as connection:
        active = connection.execute(
            "SELECT normalized_alias FROM identity_aliases WHERE object_id = 'legacy-searchable'"
        ).fetchall()
        legacy = connection.execute(
            "SELECT normalized_alias FROM object_aliases WHERE object_id = 'legacy-searchable'"
        ).fetchall()
    assert [row["normalized_alias"] for row in active] == ["new-form"]
    # object_aliases is read-only history: the new alias never lands there.
    assert [row["normalized_alias"] for row in legacy] == ["old-form"]


def test_domain_hints_rank_matching_objects_without_filtering_them(tmp_path: pytest.TempPathFactory) -> None:
    """Catch domain hints that either suppress matching objects or do not affect result order."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("apple", domain_hints=["food"]),
                _object("markets", domain_hints=["markets"]),
            ]
        ),
        "ranked-hints",
    )

    objects = WorldRecall(store).search("apple markets").anchor_objects

    assert [row["id"] for row in objects] == ["markets", "apple"]


def test_search_candidates_bucket_identity_alias_exact(tmp_path: pytest.TempPathFactory) -> None:
    """Catch identity_alias_exact bucketing that misses a stored identity alias hit."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("team-a", aliases=["lng"], domain_hints=["lol"]),
                _object("team-b", aliases=["blg"], domain_hints=["lol"]),
            ]
        ),
        "seed",
    )

    candidates = WorldRecall(store).search("lng").candidates

    assert candidates == [
        {
            "id": "team-a",
            "kind": "identity_alias_exact",
            "match_term": "lng",
            "domain_hints": ["lol"],
            "match_surface": "lng",
        }
    ]


def test_search_candidates_bucket_possible_match(tmp_path: pytest.TempPathFactory) -> None:
    """Catch possible_match bucketing for a query that resembles but is not a stored name."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="lpl-final",
                    kind=ObjectKind.ENTITY,
                    canonical_name="英雄联盟总决赛",
                    domain_hints=["lol"],
                )
            ]
        ),
        "seed",
    )

    candidates = WorldRecall(store).search("英雄联盟").candidates

    assert candidates == [
        {
            "id": "lpl-final",
            "kind": "possible_match",
            "match_term": "英雄联盟",
            "domain_hints": ["lol"],
            "match_surface": "英雄联盟总决赛",
        }
    ]


def test_search_candidates_bucket_text_match(tmp_path: pytest.TempPathFactory) -> None:
    """Catch text_match bucketing when the FTS hit falls below the similarity line."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="lpl-transfer",
                    kind=ObjectKind.ENTITY,
                    canonical_name="英雄联盟转会市场",
                    domain_hints=["lol"],
                )
            ]
        ),
        "seed",
    )

    candidates = WorldRecall(store).search("英雄").candidates

    assert candidates == [
        {
            "id": "lpl-transfer",
            "kind": "text_match",
            "match_term": "英雄",
            "domain_hints": ["lol"],
        }
    ]


def test_search_candidates_possible_match_line_at_0_20(tmp_path: pytest.TempPathFactory) -> None:
    """The possible_match line sits at the same 0.20 jaccard the graph remap uses."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="numbers",
                    kind=ObjectKind.ENTITY,
                    canonical_name="一二三四五六",
                    domain_hints=["math"],
                )
            ]
        ),
        "seed",
    )
    recall = WorldRecall(store)

    at_line = recall.search("一二").candidates
    below_line = recall.search("一").candidates

    # 一二 shares exactly one of five bigrams (jaccard 0.20): possible_match.
    # 一 shares none: text_match.
    assert at_line[0]["kind"] == "possible_match"
    assert below_line[0]["kind"] == "text_match"


def test_search_candidates_most_specific_term_wins(tmp_path: pytest.TempPathFactory) -> None:
    """Multi-term queries bucket by the most specific matching term."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(objects=[_object("team-b", aliases=["bins", "lng"], domain_hints=["lol"])]),
        "seed",
    )

    candidates = WorldRecall(store).search("bins lng").candidates

    assert candidates == [
        {
            "id": "team-b",
            "kind": "identity_alias_exact",
            "match_term": "bins",
            "domain_hints": ["lol"],
            "match_surface": "bins",
        }
    ]


def test_search_candidates_empty_when_only_assertion_text_matches(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Candidates cover FTS-matched objects only; assertion-text hits add none."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("orbital-project")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("assertion-1", "orbital-project", literal="orbital funding secured")
            ],
        ),
        "seed",
    )

    bundle = WorldRecall(store).search("funding")

    assert [row["id"] for row in bundle.assertions] == ["assertion-1"]
    assert bundle.candidates == []


def test_search_candidates_canonical_exact_uses_identity_normalizer(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The canonical_exact layer compares through the identity normalizer."""
    from leave_information_bubble.world.recall import alias_lookup_forms as recall_forms
    from leave_information_bubble.world.store import alias_lookup_forms as store_forms

    # the same function object, never a forked copy
    assert recall_forms is store_forms

    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="t1",
                    kind=ObjectKind.ENTITY,
                    canonical_name="T1 Stars 男团出道",
                    domain_hints=["lol"],
                )
            ]
        ),
        "seed",
    )

    # The canonical name is not an identity alias (Task 1.5), so the exact
    # claim lives in the canonical_exact layer and is reached through the
    # whitespace-preserving identity normalizer.
    spaced = WorldRecall(store).search("T1 Stars 男团出道").candidates

    assert spaced == [
        {
            "id": "t1",
            "kind": "canonical_exact",
            "match_term": "T1 Stars 男团出道",
            "domain_hints": ["lol"],
            "match_surface": "T1 Stars 男团出道",
        }
    ]


def test_search_candidates_hit_identity_alias_with_internal_space(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """An identity alias with an internal space answers the space-preserving query text."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="t1",
                    kind=ObjectKind.ENTITY,
                    canonical_name="T1 Stars 男团出道",
                    aliases=["T1 Stars 男团出道"],
                    domain_hints=["lol"],
                )
            ]
        ),
        "seed",
    )
    recall = WorldRecall(store)

    spaced = recall.search("T1 Stars 男团出道").candidates
    folded = recall.search("T1 Stars男团出道").candidates

    assert spaced == [
        {
            "id": "t1",
            "kind": "identity_alias_exact",
            "match_term": "T1 Stars 男团出道",
            "domain_hints": ["lol"],
            "match_surface": "T1 Stars 男团出道",
        }
    ]
    # The whitespace-collapsed spelling is not the same identity: no
    # identity_alias_exact claim, mirroring the committer refusing a
    # collision for it (test_store).
    assert folded[0]["kind"] == "possible_match"


def test_search_orders_identity_canonical_usage_legacy_and_possible_candidates(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """One query returns every graded layer as labeled candidates in contract order."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="obj-a",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Alpha Team",
                    aliases=["darkside"],
                ),
                ObjectInput(id="obj-b", kind=ObjectKind.ENTITY, canonical_name="darkside"),
                ObjectInput(id="obj-c", kind=ObjectKind.ENTITY, canonical_name="Gamma Team"),
                ObjectInput(id="obj-d", kind=ObjectKind.ENTITY, canonical_name="Delta Team"),
                ObjectInput(id="obj-e", kind=ObjectKind.ENTITY, canonical_name="darkside club"),
            ],
            observations=[_observation("observation-1")],
            assertions=[
                AssertionInput(
                    id="usage-1",
                    subject_id="obj-c",
                    predicate="name_usage",
                    literal="darkside",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
                    qualifiers={"community": "lpl", "language": "zh"},
                    event_time_start=datetime(2026, 8, 1, tzinfo=UTC),
                )
            ],
        ),
        "layered-seed",
    )
    # obj-d carries no identity claim: only a legacy object_aliases row.
    path = tmp_path / "world.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO object_aliases (object_id, normalized_alias) VALUES ('obj-d', 'darkside')"
        )
        connection.execute("UPDATE objects_fts SET aliases = 'darkside' WHERE id = 'obj-d'")

    candidates = WorldRecall(store).search("darkside").candidates

    assert [entry["kind"] for entry in candidates] == [
        "identity_alias_exact",
        "canonical_exact",
        "name_usage",
        "legacy_name",
        "possible_match",
    ]
    assert [entry["id"] for entry in candidates] == ["obj-a", "obj-b", "obj-c", "obj-d", "obj-e"]
    # the name_usage hit carries its assertion context; the legacy hit is
    # explicitly denied identity authority
    assert candidates[2]["name_usage"]["assertion_id"] == "usage-1"
    assert candidates[3]["identity_authority"] is False
    # exact identity is a candidate, never a write or merge authorization
    assert all("identity_authority" not in entry for entry in candidates[:3])


def test_canonical_exact_returns_multiple_objects(tmp_path: pytest.TempPathFactory) -> None:
    """A canonical name is not unique: every object sharing it returns canonical_exact."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="acme-1", kind=ObjectKind.ENTITY, canonical_name="Acme"),
                ObjectInput(id="acme-2", kind=ObjectKind.ENTITY, canonical_name="Acme"),
            ]
        ),
        "shared-name",
    )

    candidates = WorldRecall(store).search("acme").candidates

    assert [entry["kind"] for entry in candidates] == ["canonical_exact", "canonical_exact"]
    assert [entry["id"] for entry in candidates] == ["acme-1", "acme-2"]


def test_legacy_lookup_uses_legacy_forms_but_has_no_identity_authority(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Legacy rows answer through alias_lookup_forms yet never claim identity authority."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="modern", kind=ObjectKind.ENTITY, canonical_name="Dale Hall"),
                ObjectInput(id="legacy-t1", kind=ObjectKind.ENTITY, canonical_name="T1 Stars"),
            ]
        ),
        "seed",
    )
    # legacy rows as pre-folding databases stored them: a space-preserving
    # form and a whitespace-collapsed form (normalize_alias).
    path = tmp_path / "world.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO object_aliases (object_id, normalized_alias) VALUES ('modern', '老粉称呼')"
        )
        connection.execute(
            "INSERT INTO object_aliases (object_id, normalized_alias) VALUES ('legacy-t1', 't1stars男团出道')"
        )
        connection.execute("UPDATE objects_fts SET aliases = '老粉称呼' WHERE id = 'modern'")
        connection.execute("UPDATE objects_fts SET aliases = 't1stars男团出道' WHERE id = 'legacy-t1'")
    recall = WorldRecall(store)

    fan_name = recall.search("老粉称呼").candidates
    glued = recall.search("T1 Stars男团出道").candidates
    spaced = recall.search("T1 Stars 男团出道").candidates

    assert fan_name == [
        {
            "id": "modern",
            "kind": "legacy_name",
            "match_term": "老粉称呼",
            "domain_hints": [],
            "identity_authority": False,
            "match_surface": "老粉称呼",
        }
    ]
    # the dual-form legacy lookup reaches the collapsed row from both the
    # glued and the space-preserving spelling, still as legacy_name
    assert glued[0]["kind"] == "legacy_name"
    assert glued[0]["identity_authority"] is False
    assert spaced[0]["kind"] == "legacy_name"
    assert spaced[0]["identity_authority"] is False
    # no object in this store holds any identity alias, so no layer may ever
    # claim identity authority for these hits
    assert all(entry["kind"] != "identity_alias_exact" for entry in [*fan_name, *glued, *spaced])


def test_name_usage_returns_community_language_time_and_evidence_context(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A name_usage hit carries its assertion id, qualifiers, time, and evidence counts."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="faker",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Faker Lee",
                    domain_hints=["lol"],
                )
            ],
            observations=[
                _observation("observation-1"),
                _observation("observation-2"),
            ],
            assertions=[
                AssertionInput(
                    id="usage-1",
                    subject_id="faker",
                    predicate="name_usage",
                    literal="皮蛋",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[
                        EvidenceInput(observation_id="observation-1", role="supports"),
                        EvidenceInput(observation_id="observation-2", role="context"),
                    ],
                    qualifiers={"community": "lpl", "language": "zh"},
                    event_time_start=datetime(2026, 8, 1, tzinfo=UTC),
                )
            ],
        ),
        "usage-seed",
    )

    bundle = WorldRecall(store).search("皮蛋")

    assert [row["id"] for row in bundle.anchor_objects] == ["faker"]
    assert bundle.candidates == [
        {
            "id": "faker",
            "kind": "name_usage",
            "match_term": "皮蛋",
            "domain_hints": ["lol"],
            "match_surface": "皮蛋",
            "name_usage": {
                "assertion_id": "usage-1",
                "literal": "皮蛋",
                "qualifiers": {"community": "lpl", "language": "zh"},
                "time": "2026-08-01T00:00:00+00:00",
                "evidence_counts": {"supports": 1, "context": 1, "contradicts": 0},
            },
        }
    ]


def test_identity_query_with_internal_space_does_not_collapse_words_together(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The identity and canonical layers keep internal spaces; glued queries degrade."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="t1",
                    kind=ObjectKind.ENTITY,
                    canonical_name="T1 Stars",
                    aliases=["T1 Stars 男团出道"],
                    domain_hints=["lol"],
                ),
                ObjectInput(
                    id="t1c",
                    kind=ObjectKind.ENTITY,
                    canonical_name="T1 Stars 男团出道",
                    domain_hints=["lol"],
                ),
            ]
        ),
        "seed",
    )
    recall = WorldRecall(store)

    spaced = recall.search("T1 Stars 男团出道").candidates
    glued = recall.search("T1 Stars男团出道").candidates

    # the identity layer answers the space-preserving spelling exactly
    assert [entry["kind"] for entry in spaced] == ["identity_alias_exact", "canonical_exact"]
    # a whitespace-collapsed spelling is never an exact identity or canonical
    # claim: it degrades to possible_match (the legacy layer would only
    # answer if a legacy row existed)
    assert [entry["kind"] for entry in glued] == ["possible_match", "possible_match"]
    assert all(entry["kind"] != "identity_alias_exact" for entry in glued)


def test_search_possible_match_threshold_is_the_single_recall_line() -> None:
    """The read-side possible_match line is the one recall threshold (0.20).

    G5b-2 retired the world-agent ghost-candidate search along with its
    graph.py ``_GHOST_CANDIDATE_SIMILARITY_THRESHOLD`` pin target; the
    recall_search.py constant is now the single source of truth, and the
    runtime graph must not define a competing similarity threshold.
    """
    from leave_information_bubble.world.recall import _MATCH_SIMILARITY_THRESHOLD

    assert _MATCH_SIMILARITY_THRESHOLD == 0.20
    graph_source = (
        Path(__file__).parents[2]
        / "src"
        / "leave_information_bubble"
        / "world_agent"
        / "graph.py"
    ).read_text(encoding="utf-8")
    assert "SIMILARITY_THRESHOLD" not in graph_source


def test_search_limits_fts_and_open_inquiries_in_sql(tmp_path: pytest.TempPathFactory) -> None:
    """Catch search that loads every FTS hit or open inquiry before slicing its bundle."""
    store = _TracingWorldStore(str(tmp_path / "world.sqlite3"))
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("subject")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion(f"assertion-{index}", "subject", literal=f"needle fact {index}")
                for index in range(8)
            ],
            inquiries=[
                InquiryInput(
                    id=f"inquiry-{index}",
                    subject_id="subject",
                    prompt=f"needle question {index}",
                    rationale="Gap",
                )
                for index in range(8)
            ],
        ),
        "many-search-hits",
    )

    bundle = WorldRecall(store).search("needle", limit=2)

    assert len(bundle.assertions) == len(bundle.inquiries) == 2
    assert bundle.truncated is True
    assert any("assertions_fts" in statement and "LIMIT 3" in statement for statement in store.statements)
    assert any(
        "FROM inquiries WHERE status = 'open'" in statement and "LIMIT 3" in statement
        for statement in store.statements
    )


def test_search_clamps_hostile_limit_and_query_terms_but_keeps_results(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch oversized native arguments bypassing recall's runtime ceilings."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("needle")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion(f"assertion-{index}", "needle", literal=f"needle fact {index}")
                for index in range(40)
            ],
        ),
        "hostile-search",
    )
    query = "needle " + " ".join(f"term{index}" for index in range(1_200))

    bundle = WorldRecall(store).search(query, limit=10_000)

    assert [row["id"] for row in bundle.anchor_objects] == ["needle"]
    assert len(bundle.assertions) == 30
    assert len(bundle.evidence_refs) == 30


def test_search_excludes_superseded_assertions_but_keeps_current_ones(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch recall that still surfaces assertions retired by a superseding one (B2-8).

    A later assertion that names an older one in ``supersedes_id`` retires it:
    the old claim must leave search results entirely instead of silently
    inflating the cognitive graph (audit A3/A5-F1). Assertions nobody
    supersedes keep returning unchanged.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("s16")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("assertion-old", "s16", literal="needle old title"),
                _assertion("assertion-other", "s16", literal="needle unrelated current claim"),
            ],
        ),
        "old-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("assertion-new", "s16", literal="needle new title").model_copy(
                    update={"supersedes_id": "assertion-old"}
                )
            ],
        ),
        "supersede-claim",
    )

    bundle = WorldRecall(store).search("needle")

    assert {row["id"] for row in bundle.assertions} == {"assertion-new", "assertion-other"}
    assert "assertion-old" not in {row["id"] for row in bundle.assertions}


def test_search_filters_world_time_before_its_sql_limit(tmp_path: pytest.TempPathFactory) -> None:
    """Catch as-of search that consumes its FTS limit on future assertions."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("subject")],
            observations=[_observation("observation-1")],
            assertions=[
                *[
                    _assertion(
                        f"a-future-{index}",
                        "subject",
                        literal=f"needle future {index}",
                        event_time_start=NOW + timedelta(days=1),
                    )
                    for index in range(3)
                ],
                *[
                    _assertion(
                        f"z-past-{index}",
                        "subject",
                        literal=f"needle past {index}",
                        event_time_start=NOW - timedelta(days=1),
                    )
                    for index in range(2)
                ],
            ],
        ),
        "world-time-limit",
    )

    bundle = WorldRecall(store).search("needle", limit=2, as_of=NOW, clock=RecallClock.WORLD)

    assert [row["id"] for row in bundle.assertions] == ["z-past-0", "z-past-1"]


def test_terms_splits_cjk_runs_into_bigrams() -> None:
    """Catch query terms that keep a long CJK run whole (audit B-1)."""
    from leave_information_bubble.world.recall import _terms

    terms = _terms("持续观察中文英雄联盟社区")
    assert terms[:3] == ["持续", "续观", "观察"]
    assert "英雄" in terms
    assert "联盟" in terms
    assert "社区" in terms
    # the whole run never survives as one unusable FTS5 phrase
    assert "持续观察中文英雄联盟社区" not in terms
    # a two-character run keeps its single gram
    assert _terms("转会") == ["转会"]
    # mixed latin/CJK tokens split into their parts: the latin side stays
    # whole (mirroring the unicode61 index token) and the CJK run is
    # gram-expanded without polluting the latin terms with digit pairs
    mixed = _terms("2026EWC英雄联盟总决赛DK夺冠")
    assert "2026ewc" in mixed
    assert "dk" in mixed
    assert "英雄" in mixed
    assert "联盟" in mixed
    assert "26" not in mixed
    assert "wc" not in mixed


def test_search_observer_text_hits_chinese_world(tmp_path: pytest.TempPathFactory) -> None:
    """Catch mission_relevant returning zero for representative Chinese observer text.

    The unicode61 FTS index stores each CJK run as one token, so the whole-run
    query terms of the old ``_terms`` never matched (audit B-1). The mission
    text must now recall objects and assertions through its CJK bigrams.
    """
    mission = "持续观察中文英雄联盟社区，关注赛事、转会与社区语境。"
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="ewc-final",
                    kind=ObjectKind.EVENT,
                    canonical_name="英雄联盟总决赛DK夺冠",
                ),
                ObjectInput(
                    id="lpl-transfer",
                    kind=ObjectKind.EVENT,
                    canonical_name="英雄联盟转会窗口",
                ),
            ],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("league-announce", "ewc-final", literal="联盟官方公告"),
                _assertion("community-post", "lpl-transfer", literal="社区运营公告"),
            ],
        ),
        "chinese-world",
    )
    recall = WorldRecall(store)

    bundle = recall.search(mission, limit=8)

    # the mission's CJK bigrams hit as FTS5 prefixes: "英雄" matches the
    # "英雄联盟..." runs, "联盟" the "联盟官方公告" token, "社区" the
    # "社区运营公告" token — the old whole-run terms matched none of these
    assert {row["id"] for row in bundle.anchor_objects} == {"ewc-final", "lpl-transfer"}
    assert {row["id"] for row in bundle.assertions} == {"league-announce", "community-post"}




# --- Slice 2a: structured filters, count mode, offset, candidate payload ---

def test_search_filters_objects_by_kind(tmp_path: pytest.TempPathFactory) -> None:
    """kind restricts the object bucket to one ObjectKind."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("orbit-project"),
                _event_object("orbit-launch", event_time_start=NOW),
            ],
        ),
        "kind-filter",
    )

    bundle = WorldRecall(store).search("orbit", kind=ObjectKind.EVENT.value)

    assert [row["id"] for row in bundle.anchor_objects] == ["orbit-launch"]


def test_search_filters_objects_by_event_time_window(tmp_path: pytest.TempPathFactory) -> None:
    """time_from/time_to keep only objects whose event span overlaps the window."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="orbit-launch",
                    kind=ObjectKind.EVENT,
                    canonical_name="Orbit Launch",
                    event_time_start=NOW - timedelta(days=10),
                    event_time_end=NOW - timedelta(days=8),
                ),
                ObjectInput(
                    id="orbit-review",
                    kind=ObjectKind.EVENT,
                    canonical_name="Orbit Review",
                    event_time_start=NOW + timedelta(days=3),
                    event_time_end=NOW + timedelta(days=5),
                ),
            ],
        ),
        "time-window",
    )

    early = WorldRecall(store).search(
        "orbit",
        time_from=NOW - timedelta(days=9),
        time_to=NOW - timedelta(days=7),
    )
    assert [row["id"] for row in early.anchor_objects] == ["orbit-launch"]

    late = WorldRecall(store).search(
        "orbit",
        time_from=NOW + timedelta(days=4),
        time_to=NOW + timedelta(days=6),
    )
    assert [row["id"] for row in late.anchor_objects] == ["orbit-review"]


def test_search_filters_assertions_by_predicate_and_time_window(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """predicate restricts the assertion bucket; the time window cuts spans."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("orbital-project")],
            observations=[_observation("observation-1")],
            assertions=[
                AssertionInput(
                    id="orbit-status-old",
                    subject_id="orbital-project",
                    predicate="status",
                    literal="orbit nominal",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
                    event_time_start=NOW - timedelta(days=10),
                    event_time_end=NOW - timedelta(days=8),
                ),
                AssertionInput(
                    id="orbit-status-new",
                    subject_id="orbital-project",
                    predicate="status",
                    literal="nominal orbit",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
                    event_time_start=NOW - timedelta(days=2),
                    event_time_end=NOW,
                ),
                _assertion("orbit-related", "orbital-project", literal="orbit payload"),
            ],
        ),
        "predicate-window",
    )

    by_predicate = WorldRecall(store).search("orbit", predicate="status")
    assert [row["id"] for row in by_predicate.assertions] == [
        "orbit-status-new",
        "orbit-status-old",
    ]

    recent = WorldRecall(store).search(
        "orbit",
        time_from=NOW - timedelta(days=5),
        time_to=NOW - timedelta(days=4),
    )
    # both status spans sit outside [NOW-5d, NOW-4d]; the unbounded assertion survives
    assert [row["id"] for row in recent.assertions] == ["orbit-related"]


def test_search_filters_events_by_has_participants(tmp_path: pytest.TempPathFactory) -> None:
    """has_participants keeps events with (or without) a current participant edge."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _event_object("orbit-launch", event_time_start=NOW),
                _event_object("orbit-review", event_time_start=NOW),
                _object("astro-7"),
            ],
            observations=[_observation("observation-1")],
            assertions=[
                _participant_edge("launch-edge", "orbit-launch", "astro-7", role="payload"),
            ],
        ),
        "participants",
    )

    with_edges = WorldRecall(store).search("orbit", has_participants=True)
    assert [row["id"] for row in with_edges.anchor_objects] == ["orbit-launch"]

    without_edges = WorldRecall(store).search("orbit", has_participants=False)
    assert [row["id"] for row in without_edges.anchor_objects] == ["orbit-review"]


def test_search_filters_objects_by_assertion_count_range(tmp_path: pytest.TempPathFactory) -> None:
    """assertion_count_min/max count current subject-anchored assertions."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("orbit-project"), _object("orbit-ground")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("count-1", "orbit-project", literal="orbit payload"),
                _assertion("count-2", "orbit-project", literal="orbit telemetry"),
                _assertion("count-3", "orbit-ground", literal="orbit antenna"),
            ],
        ),
        "count-filter",
    )

    busy = WorldRecall(store).search("orbit", assertion_count_min=2)
    assert [row["id"] for row in busy.anchor_objects] == ["orbit-project"]

    light = WorldRecall(store).search("orbit", assertion_count_max=1)
    assert [row["id"] for row in light.anchor_objects] == ["orbit-ground"]


def test_search_offset_paginates_through_anchors(tmp_path: pytest.TempPathFactory) -> None:
    """offset pages through a bounded result set; truncated reflects the page."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _event_object("orbit-a", event_time_start=NOW),
                _event_object("orbit-b", event_time_start=NOW),
                _event_object("orbit-c", event_time_start=NOW),
            ],
        ),
        "paging",
    )
    recall = WorldRecall(store)

    first = recall.search("orbit", limit=2)
    assert [row["id"] for row in first.anchor_objects] == ["orbit-a", "orbit-b"]
    assert first.truncated is True

    second = recall.search("orbit", limit=2, offset=2)
    assert [row["id"] for row in second.anchor_objects] == ["orbit-c"]
    assert second.truncated is False

    beyond = recall.search("orbit", limit=2, offset=4)
    assert beyond.anchor_objects == []


def test_search_count_reports_per_bucket_totals(tmp_path: pytest.TempPathFactory) -> None:
    """search_count returns totals for the same query+filters, without rows."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("orbit-project"), _event_object("orbit-launch")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("count-1", "orbit-project", literal="orbit payload"),
                _assertion("count-2", "orbit-project", literal="orbit telemetry"),
            ],
        ),
        "count-mode",
    )
    recall = WorldRecall(store)

    totals = recall.search_count("orbit")
    assert totals == {"objects": 2, "assertions": 2, "inquiries": 0}

    events = recall.search_count("orbit", kind=ObjectKind.EVENT.value)
    assert events["objects"] == 1

    by_predicate = recall.search_count("orbit", predicate="related_to")
    assert by_predicate["assertions"] == 2


def test_search_count_empty_query_returns_zero_totals(tmp_path: pytest.TempPathFactory) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(CognitiveDelta(objects=[_object("orbital-project")]), "empty-count")

    assert WorldRecall(store).search_count("") == {"objects": 0, "assertions": 0, "inquiries": 0}


def test_search_candidates_carry_match_surface_for_exact_layers(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """R7: the candidate carries the raw name-surface text where the hit landed."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("team"),
                _object("team-alpha-bravo-charlie-delta-echo"),
                ObjectInput(
                    id="team-a",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Team A",
                    aliases=["lng"],
                ),
            ],
        ),
        "surface-seed",
    )
    recall = WorldRecall(store)

    canonical = {entry["id"]: entry for entry in recall.search("team").candidates}
    assert canonical["team"]["kind"] == "canonical_exact"
    assert canonical["team"]["match_surface"] == "Team"
    assert "match_surface_truncated" not in canonical["team"]
    # a pure full-text hit reports no name surface (there is no name hit)
    assert canonical["team-alpha-bravo-charlie-delta-echo"]["kind"] == "text_match"
    assert "match_surface" not in canonical["team-alpha-bravo-charlie-delta-echo"]

    aliased = {entry["id"]: entry for entry in recall.search("lng").candidates}
    assert aliased["team-a"]["kind"] == "identity_alias_exact"
    assert aliased["team-a"]["match_surface"] == "lng"


def test_search_candidates_carry_name_usage_surface(tmp_path: pytest.TempPathFactory) -> None:
    """name_usage candidates surface the matched assertion literal."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("hope")],
            observations=[_observation("observation-1")],
            assertions=[
                AssertionInput(
                    id="usage-1",
                    subject_id="hope",
                    predicate="name_usage",
                    literal="courage",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
                ),
            ],
        ),
        "usage-surface",
    )

    candidates = WorldRecall(store).search("courage").candidates

    assert [entry["kind"] for entry in candidates] == ["name_usage"]
    assert candidates[0]["match_surface"] == "courage"


def test_search_candidates_truncate_long_match_surface(tmp_path: pytest.TempPathFactory) -> None:
    """R7: surfaces are bounded to 80 characters with an explicit truncation flag."""
    long_alias = "l" * 86
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="team-a",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Team A",
                    aliases=[long_alias],
                ),
            ],
        ),
        "long-surface",
    )

    candidates = WorldRecall(store).search(long_alias).candidates

    assert candidates[0]["kind"] == "identity_alias_exact"
    surface = candidates[0]["match_surface"]
    assert surface == long_alias[:79] + "…"
    assert len(surface) == 80
    assert candidates[0]["match_surface_truncated"] is True


def test_search_candidates_carry_object_time_range(tmp_path: pytest.TempPathFactory) -> None:
    """R7: candidates expose the object's event span for time-aware judgment."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="orbit-launch",
                    kind=ObjectKind.EVENT,
                    canonical_name="Orbit Launch",
                    event_time_start=NOW - timedelta(days=1),
                    event_time_end=NOW,
                ),
            ],
        ),
        "time-range",
    )

    candidates = WorldRecall(store).search("orbit").candidates

    assert candidates[0]["time_range"] == {
        "start": (NOW - timedelta(days=1)).astimezone(UTC).isoformat(),
        "end": NOW.astimezone(UTC).isoformat(),
    }
