"""Offline acceptance matrix for the durable cognition graph contract (spec §11.1).

Every bullet of §11.1 maps to one test below, asserted structurally through
the public world APIs (``WorldStore``, ``ProposalCommitter``, ``WorldRecall``,
``submit_cognition_schema``, ``WorldTools.schemas``) and the public
``ModelRequestEnvelope`` from ``world_agent.model_calls``. No node-count
quotas, no private helpers from other test modules, no network, no model
calls — isolated tmp databases only.

§11.1 map:
1. organization/place -> entity, method/rule -> concept routing, schema enum
   narrowed to the three core kinds.
2. Canonical names coexist; identity-alias conflicts return candidates,
   never an auto-merge.
3. Alias add/remove/demote keep full audit (identity_aliases history,
   world_audit alias_operations, proposal_attempts, commit_receipts).
4. Community name written as a name_usage assertion with evidence/qualifiers
   and a traceable concept lift.
5. Two entities + one event compile explicit participant edges; titles and
   results never substitute participants.
6. Same participants, same day, different type/granularity coexist instead
   of being merged.
7. Fuzzy names, domain overlap and ghost hash ids are never auto-remapped.
8. Field-level conflicts omit only the failing alias operation and keep the
   update's other safe fields.
9. map -> ego -> detail reads are bounded and explainable.
10. Broad/Deep/Proposal share one identity contract sentence.
11. Request envelope fingerprints satisfy the cache invariants.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from leave_information_bubble.world import (
    AssertionInput,
    AssertionProposal,
    CognitionDeltaProposal,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    GraphRef,
    InquiryInput,
    NewObjectProposal,
    ObjectInput,
    ObjectKind,
    ObjectUpdateProposal,
    ObservationDepth,
    ObservationInput,
    ProposalCommitter,
    ProposalValidationError,
    ReviewOutcome,
    WorldRecall,
    WorldStore,
    WorldTools,
    submit_cognition_schema,
)
from leave_information_bubble.world.committer import durable_id
from leave_information_bubble.world.graph_contract_text import IDENTITY_MODEL_SENTENCE
from leave_information_bubble.world.proposal import EventParticipantProposal, ReviewIssueCode
from leave_information_bubble.world_agent.model_calls import ModelRequestEnvelope

MATCH_TIME = datetime(2026, 8, 10, 9, 0, tzinfo=UTC)


def _store(tmp_path) -> WorldStore:
    return WorldStore(tmp_path / "world.sqlite3")


def _connect(path: str) -> sqlite3.Connection:
    """Open a row-access connection to the isolated database for assertions."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _observation(identifier: str, depth: ObservationDepth = ObservationDepth.CONTENT) -> ObservationInput:
    return ObservationInput(
        id=identifier,
        source_uri=f"https://example.test/{identifier}",
        source_kind="web",
        title=identifier,
        depth=depth,
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )


def _object_rows(connection: sqlite3.Connection) -> list[tuple[str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2]) if row[2] else "")
        for row in connection.execute(
            "SELECT id, kind, COALESCE(type_key, '') FROM objects ORDER BY id"
        ).fetchall()
    ]


def _identity_alias_rows(connection: sqlite3.Connection) -> list[tuple[str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2]))
        for row in connection.execute(
            "SELECT object_id, normalized_alias, status FROM identity_aliases ORDER BY object_id"
        ).fetchall()
    ]


# ── §11.1-1: legacy kinds route to core kinds; schema enum stays core ──


def test_acceptance_legacy_kinds_route_to_core_kinds(tmp_path) -> None:
    """organization/place fold to entity, method/rule to concept, source is unsupported."""
    store = _store(tmp_path)
    proposal = CognitionDeltaProposal(
        new_objects=[
            NewObjectProposal(local_ref="org", kind=ObjectKind.ORGANIZATION, canonical_name="Org A"),
            NewObjectProposal(local_ref="place", kind=ObjectKind.PLACE, canonical_name="Place B"),
            NewObjectProposal(local_ref="method", kind=ObjectKind.METHOD, canonical_name="Method C"),
            NewObjectProposal(local_ref="rule", kind=ObjectKind.RULE, canonical_name="Rule D"),
            NewObjectProposal(
                local_ref="person", kind=ObjectKind.ENTITY, canonical_name="Person E", type_key="player"
            ),
            NewObjectProposal(local_ref="src", kind=ObjectKind.SOURCE, canonical_name="Source F"),
        ]
    )

    receipt = ProposalCommitter(store).commit(proposal, "legacy-routing")

    assert receipt.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    unsupported = [
        issue
        for issue in receipt.review_issues
        if issue.code is ReviewIssueCode.UNSUPPORTED_OBJECT_KIND
    ]
    assert len(unsupported) == 1
    assert unsupported[0].item_kind == "new_object"
    assert unsupported[0].item_index == 5
    assert "src" not in receipt.object_ids_by_local_ref
    with _connect(store.path) as connection:
        expected_kinds = {
            "org": ("entity", "organization"),
            "place": ("entity", "place"),
            "method": ("concept", "method"),
            "rule": ("concept", "rule"),
            "person": ("entity", "player"),
        }
        assert {
            row[0]: (row[1], row[2]) for row in _object_rows(connection)
        } == {
            durable_id("object", "legacy-routing", local_ref): kind_type
            for local_ref, kind_type in expected_kinds.items()
        }
    # the provider-facing schema narrows the kind enum to the three core kinds
    kind_enum = submit_cognition_schema()["function"]["parameters"]["$defs"]["ObjectKind"]["enum"]
    assert kind_enum == ["entity", "event", "concept"]


# ── §11.1-2: canonical 同名可共存，identity alias 冲突返回候选而不自动合并 ──


def test_acceptance_canonical_coexists_and_alias_conflict_returns_candidate(tmp_path) -> None:
    store = _store(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="obj-bin", kind=ObjectKind.ENTITY, canonical_name="Bin", aliases=["48"]
                )
            ]
        ),
        "seed-w1",
    )

    # A second object with the same canonical name is created, not merged.
    same_name = ProposalCommitter(store).commit(
        CognitionDeltaProposal.model_validate(
            {"new_objects": [{"local_ref": "bin2", "kind": "entity", "canonical_name": "Bin"}]}
        ),
        "wake-2",
    )
    assert same_name.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    same_name_issue = same_name.review_issues[0]
    assert same_name_issue.code is ReviewIssueCode.AMBIGUOUS_NAME_CANDIDATES
    assert same_name_issue.severity == "warning"
    assert same_name_issue.match_basis == ["canonical_name_exact"]
    assert same_name_issue.candidate_ids == ["obj-bin"]
    assert same_name.object_ids_by_local_ref == {
        "bin2": durable_id("object", "wake-2", "bin2")
    }
    assert same_name.resolved_object_refs == []
    assert same_name.remapped == []

    # Claiming the owner's identity alias omits the item with the owner as
    # repair candidate — the reference is never rewritten to the owner.
    claim = ProposalCommitter(store).commit(
        CognitionDeltaProposal.model_validate(
            {
                "new_objects": [
                    {"local_ref": "bin3", "kind": "entity", "canonical_name": "Bin", "aliases": ["48"]},
                    {"local_ref": "safe", "kind": "entity", "canonical_name": "BLG"},
                ]
            }
        ),
        "wake-3",
    )
    assert claim.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    conflict = [
        issue for issue in claim.review_issues if issue.code is ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT
    ]
    assert len(conflict) == 1
    assert conflict[0].severity == "warning"
    assert conflict[0].candidate_ids == ["obj-bin"]
    assert conflict[0].match_basis == ["identity_alias_active_owner"]
    assert conflict[0].actual_value == {"aliases": ["48"], "existing_object_ids": ["obj-bin"]}
    assert "use_existing_object_id" in conflict[0].suggested_actions
    assert "bin3" not in claim.object_ids_by_local_ref
    assert claim.resolved_object_refs == []
    assert claim.remapped == []
    with _connect(store.path) as connection:
        assert _identity_alias_rows(connection) == [("obj-bin", "48", "active")]
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM identity_aliases WHERE normalized_alias = '48' AND status = 'active'"
            ).fetchone()[0]
            == 1
        )


# ── §11.1-3: alias 添加、删除和降级均保留审计 ──


def test_acceptance_alias_lifecycle_is_audited(tmp_path) -> None:
    store = _store(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("obs-1")],
            objects=[ObjectInput(id="obj-x", kind=ObjectKind.ENTITY, canonical_name="X", aliases=["a1"])],
        ),
        "seed-w1",
    )

    def update(commit_id: str, expected_version: int, **alias_ops) -> ReviewOutcome:
        return ProposalCommitter(store).commit(
            CognitionDeltaProposal(
                object_updates=[
                    ObjectUpdateProposal(
                        target=GraphRef(memory_id="obj-x"),
                        expected_version=expected_version,
                        **alias_ops,
                    )
                ]
            ),
            commit_id,
        ).review_outcome

    assert update("wake-add", 1, add_aliases=["a2"]) is ReviewOutcome.ACCEPT
    assert update("wake-remove", 2, remove_aliases=["a1"]) is ReviewOutcome.ACCEPT
    demote = ProposalCommitter(store).commit(
        CognitionDeltaProposal(
            object_updates=[
                ObjectUpdateProposal(
                    target=GraphRef(memory_id="obj-x"),
                    expected_version=3,
                    demote_aliases=["a2"],
                )
            ],
            assertions=[
                AssertionProposal(
                    subject=GraphRef(memory_id="obj-x"),
                    predicate="name_usage",
                    literal="a2",
                    epistemic_role=EpistemicRole.COMMUNITY_VIEW,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                )
            ],
        ),
        "wake-demote",
    )
    assert demote.review_outcome is ReviewOutcome.ACCEPT

    with _connect(store.path) as connection:
        rows = connection.execute(
            "SELECT normalized_alias, status, added_commit_id, removed_commit_id"
            " FROM identity_aliases WHERE object_id = 'obj-x' ORDER BY normalized_alias"
        ).fetchall()
        deltas = {
            str(row["commit_id"]): CognitiveDelta.model_validate_json(str(row["delta_json"]))
            for row in connection.execute("SELECT commit_id, delta_json FROM world_audit").fetchall()
        }
        assert connection.execute("SELECT COUNT(*) FROM proposal_attempts").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM commit_receipts").fetchone()[0] == 4
        version = connection.execute("SELECT version FROM objects WHERE id = 'obj-x'").fetchone()[0]
    assert [tuple(row) for row in rows] == [
        ("a1", "removed", "seed-w1", "wake-remove"),
        ("a2", "removed", "wake-add", "wake-demote"),
    ]
    assert [(op.action.value, op.raw_alias) for op in deltas["wake-add"].alias_operations] == [
        ("add", "a2")
    ]
    assert [(op.action.value, op.raw_alias) for op in deltas["wake-remove"].alias_operations] == [
        ("remove", "a1")
    ]
    assert [(op.action.value, op.raw_alias) for op in deltas["wake-demote"].alias_operations] == [
        ("demote", "a2")
    ]
    assert version == 4


# ── §11.1-4: 社区称呼写成带 evidence/qualifiers 的 assertion，可追溯地提升 concept ──


def test_acceptance_community_name_usage_with_evidence_qualifiers_and_concept(tmp_path) -> None:
    store = _store(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("obs-1")],
            objects=[ObjectInput(id="obj-peyz", kind=ObjectKind.ENTITY, canonical_name="Peyz")],
        ),
        "seed-w1",
    )
    wake = CognitionDeltaProposal(
        new_objects=[
            NewObjectProposal(local_ref="concept-pi", kind=ObjectKind.CONCEPT, canonical_name="皮蛋称呼")
        ],
        assertions=[
            AssertionProposal(
                subject=GraphRef(memory_id="obj-peyz"),
                predicate="name_usage",
                literal="皮蛋",
                qualifiers={"community": "lol-zh", "language": "zh-CN"},
                epistemic_role=EpistemicRole.COMMUNITY_VIEW,
                confidence=0.8,
                evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
            ),
            AssertionProposal(
                subject=GraphRef(local_ref="concept-pi"),
                predicate="refers_to",
                object=GraphRef(memory_id="obj-peyz"),
                epistemic_role=EpistemicRole.SEMANTIC_EXPLANATION,
                confidence=0.7,
                evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
            ),
        ],
    )

    receipt = ProposalCommitter(store).commit(wake, "wake-usage")
    assert receipt.review_outcome is ReviewOutcome.ACCEPT
    usage_id, refers_id = receipt.commit.assertion_ids
    assert usage_id == durable_id("assertion", "wake-usage", "0")
    assert refers_id == durable_id("assertion", "wake-usage", "1")
    concept_id = durable_id("object", "wake-usage", "concept-pi")
    with _connect(store.path) as connection:
        usage_row = connection.execute(
            "SELECT literal_json, qualifiers_json, epistemic_role FROM assertions WHERE id = ?",
            (usage_id,),
        ).fetchone()
        refers_row = connection.execute(
            "SELECT object_id FROM assertions WHERE id = ?", (refers_id,)
        ).fetchone()
        link = connection.execute(
            "SELECT observation_id, role FROM assertion_evidence WHERE assertion_id = ?",
            (usage_id,),
        ).fetchone()
        concept_kind = connection.execute(
            "SELECT kind FROM objects WHERE id = ?", (concept_id,)
        ).fetchone()
    assert json.loads(str(usage_row["literal_json"])) == "皮蛋"
    usage_qualifiers = (
        json.loads(str(usage_row["qualifiers_json"])) if usage_row["qualifiers_json"] else None
    )
    assert usage_qualifiers == {
        "community": "lol-zh",
        "language": "zh-CN",
    }
    assert usage_row["epistemic_role"] == "community_view"
    assert str(refers_row["object_id"]) == "obj-peyz"
    assert tuple(link) == ("obs-1", "supports")
    assert concept_kind[0] == "concept"

    # recall surfaces the usage through the name_usage layer with its context
    bundle = WorldRecall(store).search("皮蛋")
    entry = next(candidate for candidate in bundle.candidates if candidate["id"] == "obj-peyz")
    assert entry["kind"] == "name_usage"
    assert entry["name_usage"] == {
        "assertion_id": usage_id,
        "literal": "皮蛋",
        "qualifiers": {"community": "lol-zh", "language": "zh-CN"},
        "time": None,
        "evidence_counts": {"supports": 1, "context": 0, "contradicts": 0},
    }


# ── §11.1-5: 显式参与边产生，标题与结果不能替代 participants ──


def test_acceptance_participant_edges_compile_title_and_result_do_not(tmp_path) -> None:
    store = _store(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("obs-1")],
            objects=[
                ObjectInput(id="team-t1", kind=ObjectKind.ENTITY, canonical_name="Team T1"),
                ObjectInput(id="team-blg", kind=ObjectKind.ENTITY, canonical_name="Team BLG"),
            ],
        ),
        "seed-w1",
    )
    wake = CognitionDeltaProposal(
        new_objects=[
            NewObjectProposal(
                local_ref="match-1",
                kind=ObjectKind.EVENT,
                canonical_name="T1 vs BLG",
                type_key="match",
                event_time_start=MATCH_TIME,
                event_time_precision="exact",
                participants=[
                    EventParticipantProposal(
                        object=GraphRef(memory_id="team-t1"),
                        role="home",
                        epistemic_role=EpistemicRole.FACT,
                        confidence=0.9,
                        evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                    ),
                    EventParticipantProposal(
                        object=GraphRef(memory_id="team-blg"),
                        role="away",
                        epistemic_role=EpistemicRole.FACT,
                        confidence=0.9,
                    ),
                ],
            ),
            NewObjectProposal(
                local_ref="finals",
                kind=ObjectKind.EVENT,
                canonical_name="T1 vs BLG 决赛",
                event_time_start=MATCH_TIME,
                event_time_precision="exact",
            ),
        ],
        assertions=[
            AssertionProposal(
                subject=GraphRef(local_ref="match-1"),
                predicate="result",
                literal="2:1",
                epistemic_role=EpistemicRole.FACT,
                confidence=0.9,
            )
        ],
    )

    receipt = ProposalCommitter(store).commit(wake, "wake-match")
    assert receipt.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    incomplete = [
        issue
        for issue in receipt.review_issues
        if issue.code is ReviewIssueCode.EVENT_PARTICIPANT_INCOMPLETE
    ]
    assert len(incomplete) == 1
    assert incomplete[0].item_index == 1
    match_id = receipt.object_ids_by_local_ref["match-1"]
    edge_home = durable_id("assertion", "wake-match", "match-1:participant:0")
    edge_away = durable_id("assertion", "wake-match", "match-1:participant:1")
    with _connect(store.path) as connection:
        rows = _object_rows(connection)
        edge_rows = connection.execute(
            "SELECT id, subject_id, predicate, object_id, epistemic_role, confidence, qualifiers_json"
            " FROM assertions WHERE predicate = 'has_participant' ORDER BY id"
        ).fetchall()
        result_rows = connection.execute(
            "SELECT predicate, literal_json, object_id FROM assertions WHERE predicate = 'result'"
        ).fetchall()
        links = connection.execute("SELECT assertion_id, role FROM assertion_evidence").fetchall()
        entity_count = connection.execute(
            "SELECT COUNT(*) FROM objects WHERE kind = 'entity'"
        ).fetchone()[0]
    assert len(rows) == 4  # two teams + two events; the title minted no identities
    assert entity_count == 2
    assert [str(row[0]) for row in edge_rows] == [edge_home, edge_away]
    assert all(str(row[1]) == match_id for row in edge_rows)
    assert {str(row[3]) for row in edge_rows} == {"team-t1", "team-blg"}
    assert {str(row[4]) for row in edge_rows} == {"fact"}
    assert {float(row[5]) for row in edge_rows} == {0.9}
    assert json.loads(str(edge_rows[0][6])) == {"role": "home"}
    assert json.loads(str(edge_rows[1][6])) == {"role": "away"}
    assert len(result_rows) == 1
    assert json.loads(str(result_rows[0][1])) == "2:1"
    assert result_rows[0][2] is None  # the result is a literal, never an edge
    assert [tuple(row) for row in links] == [(edge_home, "supports")]

    bundle = WorldRecall(store).expand([match_id], depth=1, limit=10)
    assert {edge["role"] for edge in bundle.event_edges} == {"home", "away"}
    home_edge = next(edge for edge in bundle.event_edges if edge["role"] == "home")
    assert home_edge["evidence_counts"] == {"supports": 1, "context": 0, "contradicts": 0}
    assert home_edge["epistemic_role"] == "fact"


# ── §11.1-6: 同参与方同日不同 type/granularity 不被错误合并 ──


def _participant(memory_id: str, role: str) -> EventParticipantProposal:
    return EventParticipantProposal(
        object=GraphRef(memory_id=memory_id),
        role=role,
        epistemic_role=EpistemicRole.FACT,
        confidence=0.9,
    )


def test_acceptance_same_day_different_type_and_granularity_coexist(tmp_path) -> None:
    store = _store(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("obs-1")],
            objects=[
                ObjectInput(id="team-a", kind=ObjectKind.ENTITY, canonical_name="Team A"),
                ObjectInput(id="team-b", kind=ObjectKind.ENTITY, canonical_name="Team B"),
            ]
        ),
        "seed-teams",
    )

    def game(local_ref: str, name: str, type_key: str, granularity: str) -> CognitionDeltaProposal:
        return CognitionDeltaProposal(
            new_objects=[
                NewObjectProposal(
                    local_ref=local_ref,
                    kind=ObjectKind.EVENT,
                    canonical_name=name,
                    type_key=type_key,
                    event_time_start=MATCH_TIME,
                    event_time_precision="exact",
                    participants=[_participant("team-a", "home"), _participant("team-b", "away")],
                )
            ],
            assertions=[
                AssertionProposal(
                    subject=GraphRef(local_ref=local_ref),
                    predicate="has_scope",
                    literal="single event",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    qualifiers={"granularity": granularity},
                    evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                )
            ],
        )

    series = ProposalCommitter(store).commit(
        game("series", "A vs B 系列赛", "series", "series"), "wake-series"
    )
    assert series.review_outcome is ReviewOutcome.ACCEPT
    game_receipt = ProposalCommitter(store).commit(
        game("game", "A vs B 第3局", "match", "match"), "wake-game"
    )
    assert game_receipt.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    issues = [
        issue
        for issue in game_receipt.review_issues
        if issue.code is ReviewIssueCode.DUPLICATE_EVENT_CANDIDATE
    ]
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].match_basis == ["participant_set_overlap"]
    assert issues[0].candidate_ids == [series.object_ids_by_local_ref["series"]]
    assert issues[0].actual_value["weakness"] == [
        "type_key_mismatch",
        "granularity_missing_or_different",
    ]
    assert game_receipt.resolved_object_refs == []
    assert game_receipt.remapped == []
    series_id = series.object_ids_by_local_ref["series"]
    game_id = game_receipt.object_ids_by_local_ref["game"]
    assert game_id != series_id
    with _connect(store.path) as connection:
        rows = connection.execute(
            "SELECT id, type_key FROM objects WHERE kind = 'event' ORDER BY id"
        ).fetchall()
        edges = connection.execute(
            "SELECT COUNT(*) FROM assertions WHERE predicate = 'has_participant'"
        ).fetchone()[0]
    assert {str(row[0]): str(row[1]) for row in rows} == {series_id: "series", game_id: "match"}
    assert edges == 4


# ── §11.1-7: fuzzy name、domain overlap 和错误 hash ID 不会自动重映射 ──


def test_acceptance_fuzzy_domain_ghost_id_never_remap(tmp_path) -> None:
    store = _store(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="obj-bin", kind=ObjectKind.ENTITY, canonical_name="Bin", domain_hints=["lol"]
                ),
                ObjectInput(
                    id="obj-skt", kind=ObjectKind.ENTITY, canonical_name="SKT", domain_hints=["lol"]
                ),
            ]
        ),
        "seed-w1",
    )

    # fuzzy name: a resembling name is a duplicate candidate, still created
    fuzzy = ProposalCommitter(store).commit(
        CognitionDeltaProposal.model_validate(
            {"new_objects": [{"local_ref": "bing", "kind": "entity", "canonical_name": "Bing"}]}
        ),
        "wake-fuzzy",
    )
    assert fuzzy.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    fuzzy_issue = fuzzy.review_issues[0]
    assert fuzzy_issue.code is ReviewIssueCode.DUPLICATE_OBJECT_CANDIDATE
    assert fuzzy_issue.match_basis == ["fuzzy_name_match"]
    assert fuzzy_issue.candidate_ids == ["obj-bin"]
    assert fuzzy.object_ids_by_local_ref == {"bing": durable_id("object", "wake-fuzzy", "bing")}
    assert fuzzy.resolved_object_refs == []
    assert fuzzy.remapped == []

    # domain overlap: both a same-name and a domain-hint warning, still created
    overlap = ProposalCommitter(store).commit(
        CognitionDeltaProposal.model_validate(
            {
                "new_objects": [
                    {
                        "local_ref": "skt2",
                        "kind": "entity",
                        "canonical_name": "SKT",
                        "domain_hints": ["lol"],
                    }
                ]
            }
        ),
        "wake-domain",
    )
    assert overlap.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    codes = [issue.code for issue in overlap.review_issues]
    assert codes == [
        ReviewIssueCode.AMBIGUOUS_NAME_CANDIDATES,
        ReviewIssueCode.DUPLICATE_OBJECT_CANDIDATE,
    ]
    assert overlap.review_issues[0].match_basis == ["canonical_name_exact"]
    assert overlap.review_issues[1].match_basis == ["domain_hints_overlap"]
    assert overlap.review_issues[1].candidate_ids == ["obj-skt"]
    assert overlap.resolved_object_refs == []
    assert overlap.remapped == []

    # ghost hash id: fail-closed invalid_reference, never a fuzzy rewrite
    ghost = CognitionDeltaProposal(
        assertions=[
            AssertionProposal(
                subject=GraphRef(memory_id="obj-ghost"),
                predicate="related_to",
                object=GraphRef(memory_id="obj-bin"),
                epistemic_role=EpistemicRole.FACT,
                confidence=0.9,
            )
        ]
    )
    with pytest.raises(ProposalValidationError) as caught:
        ProposalCommitter(store).commit(ghost, "wake-ghost")
    error = caught.value
    assert error.review_issues[0].code is ReviewIssueCode.INVALID_REFERENCE
    assert error.review_issues[0].severity == "error"
    assert error.review_issues[0].match_basis == ["missing_memory_id"]
    assert error.review_issues[0].durable_id == "obj-ghost"
    assert "use_real_store_id" in error.review_issues[0].suggested_actions
    assert error.candidate_object_ids == []


# ── §11.1-8: 字段级冲突不丢弃同一 update 的其他安全修改 ──


def test_acceptance_field_level_conflict_keeps_safe_update_fields(tmp_path) -> None:
    store = _store(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-x", kind=ObjectKind.ENTITY, canonical_name="X", aliases=["a1"])]
        ),
        "seed-w1",
    )
    mixed = CognitionDeltaProposal(
        object_updates=[
            ObjectUpdateProposal(
                target=GraphRef(memory_id="obj-x"),
                expected_version=1,
                remove_aliases=["nope"],  # not active -> the only failing field
                add_aliases=["fresh"],
                provisional=True,
            )
        ]
    )

    receipt = ProposalCommitter(store).commit(mixed, "wake-mixed")
    assert receipt.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    invalid = [
        issue
        for issue in receipt.review_issues
        if issue.code is ReviewIssueCode.ALIAS_OPERATION_INVALID
    ]
    assert len(invalid) == 1
    assert invalid[0].item_kind == "object_update"
    assert invalid[0].match_basis == ["alias_not_active"]
    assert invalid[0].actual_value == {
        "object_id": "obj-x",
        "raw_alias": "nope",
        "normalized_alias": "nope",
        "action": "remove",
    }
    with _connect(store.path) as connection:
        rows = _identity_alias_rows(connection)
        provisional, version = connection.execute(
            "SELECT provisional, version FROM objects WHERE id = 'obj-x'"
        ).fetchone()
        audited = CognitiveDelta.model_validate_json(
            str(
                connection.execute(
                    "SELECT delta_json FROM world_audit WHERE commit_id = 'wake-mixed'"
                ).fetchone()[0]
            )
        )
    # the safe field (add) survived the failing field (remove) within one update
    assert rows == [
        ("obj-x", "a1", "active"),
        ("obj-x", "fresh", "active"),
    ]
    assert provisional == 1
    assert version == 2
    assert [(op.action.value, op.raw_alias) for op in audited.alias_operations] == [("add", "fresh")]

    # a demote without a name_usage assertion in the same proposal is omitted
    bare_demote = ProposalCommitter(store).commit(
        CognitionDeltaProposal(
            object_updates=[
                ObjectUpdateProposal(
                    target=GraphRef(memory_id="obj-x"),
                    expected_version=2,
                    demote_aliases=["a1"],
                )
            ]
        ),
        "wake-demote-bare",
    )
    assert bare_demote.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    demote_issue = bare_demote.review_issues[0]
    assert demote_issue.code is ReviewIssueCode.ALIAS_OPERATION_INVALID
    assert demote_issue.match_basis == ["demote_requires_name_usage"]
    assert demote_issue.actual_value["action"] == "demote"
    with _connect(store.path) as connection:
        rows = _identity_alias_rows(connection)
    assert rows == [
        ("obj-x", "a1", "active"),
        ("obj-x", "fresh", "active"),
    ]


# ── §11.1-9: map -> ego -> detail 返回有界、可解释的读取结果 ──


def test_acceptance_map_ego_detail_bounded_and_explainable(tmp_path) -> None:
    store = _store(tmp_path)
    seeded_at = datetime.now(UTC)
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("obs-1")],
            objects=[
                ObjectInput(id="team-a", kind=ObjectKind.ENTITY, canonical_name="Team A"),
                ObjectInput(id="team-b", kind=ObjectKind.ENTITY, canonical_name="Team B"),
                ObjectInput(
                    id="match-1",
                    kind=ObjectKind.EVENT,
                    canonical_name="A vs B",
                    type_key="match",
                    event_time_start=seeded_at - timedelta(hours=1),
                    event_time_precision="exact",
                ),
            ],
            assertions=[
                AssertionInput(
                    id="part-a",
                    subject_id="match-1",
                    predicate="has_participant",
                    object_id="team-a",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    qualifiers={"role": "home"},
                    evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                ),
                AssertionInput(
                    id="part-b",
                    subject_id="match-1",
                    predicate="has_participant",
                    object_id="team-b",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    qualifiers={"role": "away"},
                ),
            ],
            inquiries=[InquiryInput(id="inq-1", subject_id="team-a", prompt="A 状态如何", rationale="r")],
        ),
        "seed-w1",
    )
    # as_of is captured AFTER the seed so the seed's own audit row falls
    # inside the bounded window (committed_at <= as_of)
    now = datetime.now(UTC)
    recall = WorldRecall(store)

    # map: exact counters, bounded, every surface explained
    overview = recall.overview(as_of=now, limit=10, active_days=30, cold_days=60)
    assert overview.counts.model_dump() == {
        "objects": 3,
        "assertions": 2,
        "open_inquiries": 1,
        "dormant_inquiries": 0,
    }
    assert overview.truncated is False
    match_front = next(candidate for candidate in overview.active_fronts if candidate.id == "match-1")
    assert match_front.kind == "object"
    assert match_front.name == "A vs B"
    assert "recent_cognition_change" in match_front.surfaced_because
    assert match_front.facts
    gap = next(candidate for candidate in overview.coverage_gaps if candidate.id == "inq-1")
    assert gap.kind == "inquiry"
    assert gap.name == "A 状态如何"
    assert gap.surfaced_because == ["underexplored"]
    assert gap.inquiry_coverage is not None

    # ego: the entity's participated events are sorted and capped explicitly
    ego = recall.expand(["team-a"], depth=1, limit=10, event_limit=10)
    assert ego.omitted_counts == {}
    assert ego.sort_basis == "event_time_start DESC, id ASC"
    assert len(ego.participated_events) == 1
    participated = ego.participated_events[0]
    assert participated["id"] == "match-1"
    assert participated["role"] == "home"
    assert participated["assertion_id"] == "part-a"
    assert participated["evidence_counts"] == {"supports": 1, "context": 0, "contradicts": 0}

    # detail: one assertion's evidence carries the exact observation link
    detail = recall.evidence("part-a")
    assert any(
        ref["observation_id"] == "obs-1" and ref["role"] == "supports"
        for ref in detail.evidence_refs
    )


# ── §11.1-10: Broad/Deep/Proposal 使用同一身份契约和示例 ──


def test_acceptance_broad_deep_proposal_share_identity_contract(tmp_path) -> None:
    """The single identity-contract sentence is embedded in the shared schema text.

    The agent-facing layouts (Broad/Deep/Proposal prompt stages) are pinned in
    tests/world_agent/test_prompt.py; here the world-side shared text sources
    are asserted verbatim: the terminal submit_cognition schema description
    and the memory_search tool description both carry the exact sentence.
    """
    store = _store(tmp_path)
    submit = submit_cognition_schema()
    assert IDENTITY_MODEL_SENTENCE in submit["function"]["description"]
    search_schema = next(
        tool
        for tool in WorldTools(store=store, adapters={}).schemas(include_memory_overview=False)
        if tool["function"]["name"] == "memory_search"
    )
    assert IDENTITY_MODEL_SENTENCE in search_schema["function"]["description"]


# ── §11.1-11: request envelope 指纹满足缓存不变式 ──


def test_acceptance_envelope_fingerprint_stable_and_phase_aware(tmp_path) -> None:
    """The request envelope fingerprint is deterministic and order-sensitive.

    The wake-level layout and cache-ledger invariants (Exploration/Proposal
    message prefixes, model_calls rows) are pinned in tests/world_agent/
    test_model_calls.py and test_context_layout.py; here the world-side
    fingerprint properties the ledger relies on are asserted directly.
    """
    base = ModelRequestEnvelope(
        model="gpt-4o-mini",
        thinking=False,
        reasoning_effort=None,
        tool_schemas=[{"name": "memory_search"}, {"name": "submit_cognition"}],
        tool_choice="auto",
        response_format=None,
        max_tokens=4096,
    )
    assert base.fingerprint() == base.fingerprint()

    # tool order is part of identity: schemas are never sorted
    reordered = ModelRequestEnvelope(
        model="gpt-4o-mini",
        thinking=False,
        reasoning_effort=None,
        tool_schemas=[{"name": "submit_cognition"}, {"name": "memory_search"}],
        tool_choice="auto",
        response_format=None,
        max_tokens=4096,
    )
    assert reordered.fingerprint() != base.fingerprint()

    # any other field change breaks the fingerprint
    retargeted = ModelRequestEnvelope(
        model="gpt-4o-mini",
        thinking=False,
        reasoning_effort=None,
        tool_schemas=[{"name": "memory_search"}, {"name": "submit_cognition"}],
        tool_choice={"type": "function", "function": {"name": "submit_cognition"}},
        response_format=None,
        max_tokens=4096,
    )
    bigger = ModelRequestEnvelope(
        model="gpt-4o-mini",
        thinking=False,
        reasoning_effort=None,
        tool_schemas=[{"name": "memory_search"}, {"name": "submit_cognition"}],
        tool_choice="auto",
        response_format=None,
        max_tokens=8192,
    )
    assert retargeted.fingerprint() != base.fingerprint()
    assert bigger.fingerprint() != base.fingerprint()

    # the stable wake projection ignores ONLY tool_choice (the allowed
    # Exploration -> Proposal phase delta)
    assert "tool_choice" not in base.stable_wake_projection()
    assert base.stable_wake_projection() == retargeted.stable_wake_projection()
    assert base.stable_wake_projection() != bigger.stable_wake_projection()
