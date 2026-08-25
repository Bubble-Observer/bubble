# Split from tests/world/test_world_recall.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    InquiryInput,
    InquiryResolution,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    ObservationLinkInput,
    WorldRecall,
    WorldStore,
)
from tests.world._recall_helpers import (
    _assertion,
    _object,
    _observation,
)


def test_inquiries_include_coverage_and_order_shallow_first(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch inquiry recall that hides subject coverage or orders deepest-first."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="obj-deep", kind=ObjectKind.ENTITY, canonical_name="IG"),
                ObjectInput(id="obj-shallow", kind=ObjectKind.ENTITY, canonical_name="Bin"),
                ObjectInput(id="obj-none", kind=ObjectKind.ENTITY, canonical_name="KeSPA"),
            ],
            observations=[
                ObservationInput(
                    id="obs-deep",
                    source_uri="https://example.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.DISCUSSION,
                    observed_at=datetime(2026, 8, 4, tzinfo=UTC),
                )
            ],
            inquiries=[
                InquiryInput(id="inq-1", subject_id="obj-deep", prompt="deep q", rationale="r"),
                InquiryInput(id="inq-2", subject_id="obj-shallow", prompt="shallow q", rationale="r"),
                InquiryInput(id="inq-3", subject_id="obj-none", prompt="none q", rationale="r"),
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object",
                    target_id="obj-deep",
                    observation_id="obs-deep",
                    role="candidate",
                )
            ],
        ),
        "seed-1",
    )
    recall = WorldRecall(store)
    bundle = recall.inquiries()
    rows = {row["id"]: row for row in bundle.inquiries}
    assert rows["inq-1"]["coverage"] == "discussion"
    assert rows["inq-2"]["coverage"] == "none"
    assert rows["inq-3"]["coverage"] == "none"
    ids = [row["id"] for row in bundle.inquiries]
    assert ids == ["inq-2", "inq-3", "inq-1"]  # shallow (none) first, then discussion


def test_inquiries_by_id_returns_single_open_inquiry(tmp_path: pytest.TempPathFactory) -> None:
    """Catch inquiry recall that ignores the inquiry_id lookup and returns the whole frontier."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="X")],
            observations=[_observation("observation-1")],
            inquiries=[
                InquiryInput(id="inq-1", subject_id="obj-1", prompt="p1", rationale="r"),
                InquiryInput(id="inq-2", subject_id="obj-1", prompt="p2", rationale="r"),
            ],
        ),
        "seed-1",
    )
    store.memory_commit(
        CognitiveDelta(
            assertions=[
                _assertion(
                    "answer", "obj-1", literal="Because evidence answers it", answers_inquiry_id="inq-1"
                )
            ],
            resolve_inquiries=[InquiryResolution(id="inq-1", expected_version=1)],
        ),
        "resolve-1",
    )
    recall = WorldRecall(store)
    bundle = recall.inquiries(inquiry_id="inq-2")
    assert [row["id"] for row in bundle.inquiries] == ["inq-2"]
    assert bundle.inquiries[0]["prompt"] == "p2"
    # a resolved inquiry is still findable by id for post-commit feedback lookups
    resolved = recall.inquiries(inquiry_id="inq-1")
    assert [row["id"] for row in resolved.inquiries] == ["inq-1"]
    assert resolved.inquiries[0]["status"] == "resolved"
    # listing paths keep the open-only filter
    assert [row["id"] for row in recall.inquiries().inquiries] == ["inq-2"]


def test_overview_is_explainable_read_only_and_uses_inquiry_specific_coverage(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """P3: four small frontiers never turn subject depth into inquiry coverage."""
    store = WorldStore(tmp_path / "world.sqlite3")
    moment = datetime.now(UTC).replace(microsecond=0)
    old = moment - timedelta(days=100)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object(identifier) for identifier in ("root-a", "root-b", "bridge", "deep")],
            observations=[
                ObservationInput(
                    id="old-source",
                    source_uri="https://example.test/old",
                    source_kind="web",
                    depth=ObservationDepth.CONTENT,
                    observed_at=old,
                ),
                ObservationInput(
                    id="deep-media",
                    source_uri="https://example.test/deep",
                    source_kind="web",
                    depth=ObservationDepth.MEDIA,
                    observed_at=old,
                ),
            ],
            assertions=[
                _assertion("edge-a", "bridge", object_id="root-a").model_copy(
                    update={"evidence": [EvidenceInput(observation_id="old-source", role="supports")]}
                ),
                _assertion("edge-b", "bridge", object_id="root-b").model_copy(
                    update={"evidence": [EvidenceInput(observation_id="old-source", role="supports")]}
                ),
                _assertion("edge-deep", "deep", object_id="root-a").model_copy(
                    update={"evidence": [EvidenceInput(observation_id="old-source", role="supports")]}
                ),
            ],
            inquiries=[
                InquiryInput(id="deep-inquiry", subject_id="deep", prompt="What remains?", rationale="gap"),
                InquiryInput(
                    id="dormant-inquiry",
                    subject_id="deep",
                    prompt="What changed?",
                    rationale="resume after evidence",
                    attempt_count=2,
                    last_attempted_at=moment - timedelta(days=10),
                ),
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object", target_id="bridge", observation_id="old-source", role="context"
                ),
                ObservationLinkInput(
                    target_kind="object", target_id="deep", observation_id="deep-media", role="context"
                ),
            ],
        ),
        "old-graph",
    )
    # Make the graph itself cold; the following commit is the only active event.
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE world_audit SET committed_at = ? WHERE commit_id = 'old-graph'", (old.isoformat(),)
        )
        connection.execute("UPDATE inquiries SET status = 'dormant' WHERE id = 'dormant-inquiry'")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="fresh-a",
                    source_uri="https://example.test/a",
                    source_kind="web",
                    depth=ObservationDepth.CONTENT,
                    observed_at=moment,
                ),
                ObservationInput(
                    id="fresh-b",
                    source_uri="https://example.test/b",
                    source_kind="web",
                    depth=ObservationDepth.DISCUSSION,
                    observed_at=moment,
                ),
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object", target_id="root-a", observation_id="fresh-a", role="context"
                ),
                ObservationLinkInput(
                    target_kind="object", target_id="root-b", observation_id="fresh-b", role="context"
                ),
            ],
        ),
        "recent-observations",
    )

    overview = WorldRecall(store).overview(as_of=moment + timedelta(minutes=1))
    repeated = WorldRecall(store).overview(as_of=moment + timedelta(minutes=1))

    assert overview.model_dump(mode="json") == repeated.model_dump(mode="json")
    assert {item.id for item in overview.active_fronts} >= {"root-a", "root-b"}
    assert [item.id for item in overview.reactivated_fronts] == ["dormant-inquiry"]
    assert [item.id for item in overview.cold_bridges] == ["bridge"]
    assert any(fact.at is not None for fact in overview.cold_bridges[0].facts)
    deep = next(item for item in overview.coverage_gaps if item.id == "deep-inquiry")
    assert deep.inquiry_coverage is not None
    assert deep.inquiry_coverage.coverage == "none"
    assert deep.inquiry_coverage.direct_observation_count == 0
    assert deep.inquiry_coverage.subject_context_depth == "media"
    with store.read_connection() as connection:
        status = connection.execute(
            "SELECT status FROM inquiries WHERE id = 'dormant-inquiry'"
        ).fetchone()[0]
        assert status == "dormant"
        tables = {
            "object": "objects",
            "observation": "observations",
            "assertion": "assertions",
            "inquiry": "inquiries",
        }
        allowed = {
            "recent_observation", "recent_cognition_change", "related_active_neighbor",
            "repeated_inquiry_point", "underexplored", "dormant_reactivated",
        }
        candidates = [
            *overview.active_fronts,
            *overview.reactivated_fronts,
            *overview.cold_bridges,
            *overview.coverage_gaps,
        ]
        for candidate in candidates:
            assert set(candidate.surfaced_because) <= allowed
            for fact in candidate.facts:
                row = connection.execute(
                    f"SELECT 1 FROM {tables[fact.kind]} WHERE id = ?", (fact.id,)
                ).fetchone()
                assert row


def test_overview_coverage_matrix_is_inquiry_specific(tmp_path: pytest.TempPathFactory) -> None:
    """P3 coverage uses direct inquiry evidence, answers, and attempts only."""

    def coverage_for(
        name: str,
        *,
        direct: ObservationDepth | None = None,
        subject: ObservationDepth | None = None,
        attempts: int = 0,
        answer: bool = False,
    ):
        store = WorldStore(tmp_path / name / "world.sqlite3")
        moment = datetime(2026, 8, 10, tzinfo=UTC)
        observations, links = [], []
        if direct is not None:
            observations.append(ObservationInput(
                id="direct", source_uri="https://example.test/direct", source_kind="web",
                depth=direct, observed_at=moment,
            ))
            links.append(ObservationLinkInput(
                target_kind="inquiry", target_id="question", observation_id="direct", role="context",
            ))
        if subject is not None:
            observations.append(ObservationInput(
                id="subject", source_uri="https://example.test/subject", source_kind="web",
                depth=subject, observed_at=moment,
            ))
            links.append(ObservationLinkInput(
                target_kind="object", target_id="subject", observation_id="subject", role="context",
            ))
        assertions = []
        if answer:
            assertions.append(_assertion("answer", "subject", literal="answer").model_copy(
                update={"answers_inquiry_id": "question", "evidence": []}
            ))
        store.memory_commit(CognitiveDelta(
            objects=[_object("subject")], observations=observations, assertions=assertions,
            inquiries=[InquiryInput(id="question", subject_id="subject", prompt="q", rationale="r")],
            observation_links=links,
        ), f"coverage-{name}")
        if attempts:
            with store.write_connection() as connection:
                connection.execute(
                    "UPDATE inquiries SET attempt_count = ?, last_attempted_at = ? WHERE id = 'question'",
                    (attempts, moment.isoformat()),
                )
        candidate = WorldRecall(store).overview(as_of=moment + timedelta(minutes=1)).coverage_gaps[0]
        assert candidate.inquiry_coverage is not None
        return candidate.inquiry_coverage

    media_only = coverage_for("media-only", subject=ObservationDepth.MEDIA)
    assert media_only.coverage == "none"
    assert media_only.direct_observation_count == 0
    assert media_only.direct_max_depth is None
    assert media_only.subject_context_depth == "media"
    attempted = coverage_for("attempted", attempts=1)
    assert attempted.coverage == "attempted"
    assert attempted.attempt_count == 1
    assert attempted.last_attempted_at is not None
    discussion = coverage_for("discussion", direct=ObservationDepth.DISCUSSION)
    assert discussion.coverage == "discussion"
    assert discussion.direct_observation_count == 1
    assert discussion.direct_max_depth == "discussion"
    answered = coverage_for("answered", answer=True)
    assert answered.coverage == "answered"
    assert answered.answering_assertion_count == 1


def test_overview_reason_codes_order_and_category_limits(tmp_path: pytest.TempPathFactory) -> None:
    """All six explainable reasons surface and each nonempty frontier is capped."""
    store = WorldStore(tmp_path / "world.sqlite3")
    moment = datetime.now(UTC).replace(microsecond=0)
    old = moment - timedelta(days=100)
    identifiers = (
        "root-a", "root-b", "change", "bridge-a", "bridge-b", "dormant-a", "dormant-b", "repeat", "gap",
    )
    support = ObservationInput(
        id="support", source_uri="https://example.test/support", source_kind="web",
        depth=ObservationDepth.CONTENT, observed_at=old,
    )
    evidence = [EvidenceInput(observation_id="support", role="supports")]
    def edge(identifier: str, left: str, right: str) -> AssertionInput:
        return _assertion(identifier, left, object_id=right).model_copy(update={"evidence": evidence})
    store.memory_commit(CognitiveDelta(
        objects=[_object(identifier) for identifier in identifiers], observations=[support],
        assertions=[
            edge("bridge-a-a", "bridge-a", "root-a"), edge("bridge-a-b", "bridge-a", "root-b"),
            edge("bridge-b-a", "bridge-b", "root-a"), edge("bridge-b-b", "bridge-b", "root-b"),
            edge("dormant-a-root", "dormant-a", "root-a"), edge("dormant-b-root", "dormant-b", "root-b"),
        ],
        inquiries=[
            InquiryInput(id="repeat-q", subject_id="repeat", prompt="repeat", rationale="r"),
            InquiryInput(id="gap-q", subject_id="gap", prompt="gap", rationale="r"),
            InquiryInput(
                id="dormant-q-a", subject_id="dormant-a", prompt="dormant", rationale="r",
                last_attempted_at=moment - timedelta(days=10),
            ),
            InquiryInput(
                id="dormant-q-b", subject_id="dormant-b", prompt="dormant", rationale="r",
                last_attempted_at=moment - timedelta(days=10),
            ),
        ],
    ), "overview-old")
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE world_audit SET committed_at = ? WHERE commit_id = 'overview-old'", (old.isoformat(),)
        )
        connection.execute(
            "UPDATE inquiries SET status = 'dormant' WHERE id IN ('dormant-q-a', 'dormant-q-b')"
        )
        connection.execute(
            "UPDATE inquiries SET attempt_count = 2, last_attempted_at = ? WHERE id = 'repeat-q'",
            (old.isoformat(),),
        )
    store.memory_commit(CognitiveDelta(
        observations=[
            ObservationInput(
                id="fresh-a", source_uri="https://example.test/a", source_kind="web",
                depth=ObservationDepth.CONTENT, observed_at=moment,
            ),
            ObservationInput(
                id="fresh-b", source_uri="https://example.test/b", source_kind="web",
                depth=ObservationDepth.CONTENT, observed_at=moment,
            ),
        ],
        assertions=[
            _assertion("fresh-change", "change", literal="new").model_copy(
                update={"evidence": evidence}
            )
        ],
        observation_links=[
            ObservationLinkInput(
                target_kind="object", target_id="root-a", observation_id="fresh-a", role="context"
            ),
            ObservationLinkInput(
                target_kind="object", target_id="root-b", observation_id="fresh-b", role="context"
            ),
        ],
    ), "overview-recent")
    recall = WorldRecall(store)
    baseline = recall.overview(as_of=moment + timedelta(minutes=1))
    repeated = recall.overview(as_of=moment + timedelta(minutes=1))
    assert baseline.model_dump(mode="json") == repeated.model_dump(mode="json")
    active_ids = [item.id for item in baseline.active_fronts]
    assert active_ids.index("root-a") < active_ids.index("root-b")
    assert all(len(group) >= 2 for group in (
        baseline.active_fronts, baseline.reactivated_fronts, baseline.cold_bridges, baseline.coverage_gaps,
    ))
    reasons = {
        reason
        for group in (
            baseline.active_fronts, baseline.reactivated_fronts, baseline.cold_bridges, baseline.coverage_gaps
        )
        for item in group for reason in item.surfaced_because
    }
    assert reasons == {
        "recent_observation", "recent_cognition_change", "related_active_neighbor",
        "repeated_inquiry_point", "underexplored", "dormant_reactivated",
    }
    one = recall.overview(as_of=moment + timedelta(minutes=1), limit=1)
    assert all(len(group) == 1 for group in (
        one.active_fronts, one.reactivated_fronts, one.cold_bridges, one.coverage_gaps,
    ))
    assert sum(len(group) for group in (
        one.active_fronts, one.reactivated_fronts, one.cold_bridges, one.coverage_gaps,
    )) == 4
    assert all(len(group) <= 3 for group in (
        baseline.active_fronts, baseline.reactivated_fronts, baseline.cold_bridges, baseline.coverage_gaps,
    ))


def test_overview_active_fronts_order_by_event_time_not_observed_at(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Active fronts rank on event time; re-read observed_at is only a fallback (spec §7)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    moment = datetime.now(UTC).replace(microsecond=0)
    old = moment - timedelta(days=100)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("alpha"), _object("zeta")],
            observations=[
                ObservationInput(
                    id="obs-alpha", source_uri="https://example.test/a", source_kind="web",
                    depth=ObservationDepth.CONTENT, observed_at=old,
                ),
                ObservationInput(
                    id="obs-zeta", source_uri="https://example.test/z", source_kind="web",
                    depth=ObservationDepth.CONTENT, observed_at=old,
                ),
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object", target_id="alpha", observation_id="obs-alpha", role="context"
                ),
                ObservationLinkInput(
                    target_kind="object", target_id="zeta", observation_id="obs-zeta", role="context"
                ),
            ],
        ),
        "overview-old",
    )
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE world_audit SET committed_at = ? WHERE commit_id = 'overview-old'", (old.isoformat(),)
        )
    # A re-read of both old sources (same ids, fresh payload observed_at) plus
    # fresh assertions carrying distinct event times: only the event times may
    # order the two fronts — the re-read must not lift either one above the
    # other via observed_at.
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="obs-alpha", source_uri="https://example.test/a", source_kind="web",
                    depth=ObservationDepth.CONTENT, observed_at=moment,
                ),
                ObservationInput(
                    id="obs-zeta", source_uri="https://example.test/z", source_kind="web",
                    depth=ObservationDepth.CONTENT, observed_at=moment,
                ),
            ],
            assertions=[
                _assertion(
                    "zeta-new", "zeta", literal="recent event",
                    event_time_start=moment - timedelta(days=2),
                ).model_copy(update={"evidence": []}),
                _assertion(
                    "alpha-new", "alpha", literal="older event",
                    event_time_start=moment - timedelta(days=5),
                ).model_copy(update={"evidence": []}),
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object", target_id="alpha", observation_id="obs-alpha", role="context"
                ),
                ObservationLinkInput(
                    target_kind="object", target_id="zeta", observation_id="obs-zeta", role="context"
                ),
            ],
        ),
        "overview-recent",
    )
    overview = WorldRecall(store).overview(as_of=moment + timedelta(minutes=1))
    active_ids = [item.id for item in overview.active_fronts]
    assert {"alpha", "zeta"} <= set(active_ids)
    # Event time ranks zeta (now-2d) above alpha (now-5d); an observed_at
    # ordering or an id tiebreak would put alpha first.
    assert active_ids.index("zeta") < active_ids.index("alpha")
    zeta = next(item for item in overview.active_fronts if item.id == "zeta")
    alpha = next(item for item in overview.active_fronts if item.id == "alpha")
    assert max(f.at for f in zeta.facts if f.at is not None) == moment - timedelta(days=2)
    assert max(f.at for f in alpha.facts if f.at is not None) == moment - timedelta(days=5)
    assert "recent_cognition_change" in zeta.surfaced_because


def test_overview_old_event_reread_is_not_active(tmp_path: pytest.TempPathFactory) -> None:
    """Re-reading an old event must not surface it as an active front (spec §7)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    moment = datetime.now(UTC).replace(microsecond=0)
    old = moment - timedelta(days=100)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("gamma"), _object("delta")],
            observations=[
                ObservationInput(
                    id="obs-gamma", source_uri="https://example.test/gamma", source_kind="web",
                    depth=ObservationDepth.CONTENT, observed_at=old,
                ),
                ObservationInput(
                    id="obs-delta", source_uri="https://example.test/delta", source_kind="web",
                    depth=ObservationDepth.CONTENT, observed_at=old,
                ),
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object", target_id="gamma", observation_id="obs-gamma", role="context"
                ),
                ObservationLinkInput(
                    target_kind="object", target_id="delta", observation_id="obs-delta", role="context"
                ),
            ],
        ),
        "old-graph",
    )
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE world_audit SET committed_at = ? WHERE commit_id = 'old-graph'", (old.isoformat(),)
        )
    # gamma is re-read today (same id, fresh payload observed_at) but its
    # stored observed_at stays at first observation, so it must NOT become an
    # active front; delta is first observed today and legitimately is.
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="obs-gamma", source_uri="https://example.test/gamma", source_kind="web",
                    depth=ObservationDepth.CONTENT, observed_at=moment,
                ),
                ObservationInput(
                    id="obs-delta-new", source_uri="https://example.test/delta-2", source_kind="web",
                    depth=ObservationDepth.CONTENT, observed_at=moment,
                ),
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object", target_id="gamma", observation_id="obs-gamma", role="context"
                ),
                ObservationLinkInput(
                    target_kind="object", target_id="delta", observation_id="obs-delta-new", role="context"
                ),
            ],
        ),
        "re-read-today",
    )
    overview = WorldRecall(store).overview(as_of=moment + timedelta(minutes=1))
    active_ids = [item.id for item in overview.active_fronts]
    assert "gamma" not in active_ids
    assert "delta" in active_ids
    delta = next(item for item in overview.active_fronts if item.id == "delta")
    assert "recent_observation" in delta.surfaced_because


def test_overview_ignores_superseded_edges_and_answers(tmp_path: pytest.TempPathFactory) -> None:
    """Retired edges cannot reactivate, bridge, or count as current answers."""
    store = WorldStore(tmp_path / "world.sqlite3")
    moment = datetime.now(UTC).replace(microsecond=0)
    old = moment - timedelta(days=100)
    evidence = [EvidenceInput(observation_id="support", role="supports")]
    def edge(identifier: str, left: str, right: str) -> AssertionInput:
        return _assertion(identifier, left, object_id=right).model_copy(update={"evidence": evidence})
    store.memory_commit(CognitiveDelta(
        objects=[_object(item) for item in ("a", "b", "c", "bridge", "dormant", "other")],
        observations=[
            ObservationInput(
                id="support", source_uri="https://example.test/support", source_kind="web",
                depth=ObservationDepth.CONTENT, observed_at=old,
            )
        ],
        assertions=[
            edge("old-a", "bridge", "a"), edge("old-b", "bridge", "b"), edge("old-dormant", "dormant", "a"),
            _assertion("old-answer", "other", literal="old answer").model_copy(
                update={"answers_inquiry_id": "answered-q", "evidence": []}
            ),
        ],
        inquiries=[
            InquiryInput(
                id="dormant-q", subject_id="dormant", prompt="dormant", rationale="r",
                last_attempted_at=moment - timedelta(days=10),
            ),
            InquiryInput(id="answered-q", subject_id="other", prompt="answer", rationale="r"),
        ],
    ), "superseded-old")
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE world_audit SET committed_at = ? WHERE commit_id = 'superseded-old'", (old.isoformat(),)
        )
        connection.execute("UPDATE inquiries SET status = 'dormant' WHERE id = 'dormant-q'")
    store.memory_commit(CognitiveDelta(
        assertions=[
            edge("new-a", "other", "a").model_copy(
                update={"evidence": evidence, "supersedes_id": "old-a"}
            ),
            edge("new-b", "other", "b").model_copy(
                update={"evidence": evidence, "supersedes_id": "old-b"}
            ),
            edge("new-dormant", "other", "c").model_copy(
                update={"evidence": evidence, "supersedes_id": "old-dormant"}
            ),
            _assertion("new-answer", "other", literal="replacement").model_copy(
                update={"supersedes_id": "old-answer", "evidence": []}
            ),
        ],
        observations=[
            ObservationInput(
                id="fresh-a", source_uri="https://example.test/a", source_kind="web",
                depth=ObservationDepth.CONTENT, observed_at=moment,
            ),
            ObservationInput(
                id="fresh-b", source_uri="https://example.test/b", source_kind="web",
                depth=ObservationDepth.CONTENT, observed_at=moment,
            ),
        ],
        observation_links=[
            ObservationLinkInput(
                target_kind="object", target_id="a", observation_id="fresh-a", role="context"
            ),
            ObservationLinkInput(
                target_kind="object", target_id="b", observation_id="fresh-b", role="context"
            ),
        ],
    ), "superseded-new")
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM current_assertions WHERE id = 'old-dormant'"
        ).fetchone()
        assert row is None
    overview = WorldRecall(store).overview(as_of=moment + timedelta(minutes=1))
    assert "dormant-q" not in [item.id for item in overview.reactivated_fronts]
    assert "bridge" not in [item.id for item in overview.cold_bridges]
    answered = next(item for item in overview.coverage_gaps if item.id == "answered-q")
    assert answered.inquiry_coverage is not None
    assert answered.inquiry_coverage.answering_assertion_count == 0
    assert answered.inquiry_coverage.coverage == "none"


def test_resolved_inquiry_is_absent_from_active_recall_but_present_in_audit(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch active recall that resurrects resolved questions or deletes their audit history."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("subject")],
            observations=[_observation("observation-1")],
            inquiries=[InquiryInput(id="inquiry-1", subject_id="subject", prompt="Why?", rationale="Gap")],
        ),
        "open-inquiry",
    )
    store.memory_commit(
        CognitiveDelta(
            assertions=[
                _assertion(
                    "answer",
                    "subject",
                    literal="Because evidence answers it",
                    answers_inquiry_id="inquiry-1",
                )
            ],
            resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=1)],
        ),
        "resolve-inquiry",
    )

    assert WorldRecall(store).inquiries().inquiries == []
    with store.read_connection() as connection:
        audit_count = connection.execute("SELECT COUNT(*) FROM world_audit").fetchone()[0]
    assert audit_count == 2


def test_recall_returns_candidate_sources_for_objects_and_inquiries(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch fresh recall that loses the source leads established by Broad exploration."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("ig-lng")],
            observations=[_observation("observation-1")],
            inquiries=[
                InquiryInput(
                    id="inquiry-score",
                    subject_id="ig-lng",
                    prompt="最终比分是什么？",
                    rationale="比赛事实核心。",
                )
            ],
            observation_links=[
                ObservationLinkInput(
                    target_kind="object",
                    target_id="ig-lng",
                    observation_id="observation-1",
                    role="candidate",
                ),
                ObservationLinkInput(
                    target_kind="inquiry",
                    target_id="inquiry-score",
                    observation_id="observation-1",
                    role="context",
                ),
            ],
        ),
        "candidate-memory",
    )

    expanded = WorldRecall(store).expand(["ig-lng"])
    searched = WorldRecall(store).search("比分")

    assert expanded.candidate_observation_refs == [
        {
            "target_kind": "object",
            "target_id": "ig-lng",
            "observation_id": "observation-1",
            "role": "candidate",
            "source_uri": "https://example.test/observation-1",
            "title": "",
            "depth": "content",
            "source_published_at": None,
            "content_ref": "",
        }
    ]
    assert searched.candidate_observation_refs[0]["target_kind"] == "inquiry"
    assert searched.candidate_observation_refs[0]["target_id"] == "inquiry-score"


def test_inquiries_is_pure_read_and_orders_existing_statuses(tmp_path):
    from datetime import timedelta
    store = WorldStore(tmp_path / "world.sqlite3")
    old = datetime.now(UTC) - timedelta(days=10)
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="X")],
            inquiries=[
                InquiryInput(id="inq-open", subject_id="obj-1", prompt="open q", rationale="r"),
                InquiryInput(id="inq-stale", subject_id="obj-1", prompt="stale q", rationale="r",
                             created_at=old, last_attempted_at=old),
            ],
        ),
        "seed-1",
    )
    recall = WorldRecall(store)
    bundle = recall.inquiries()
    rows = {row["id"]: row for row in bundle.inquiries}
    assert rows["inq-stale"]["status"] == "open"
    assert rows["inq-stale"]["dormant"] is False
    assert [r["id"] for r in bundle.inquiries] == ["inq-open", "inq-stale"]
    with store.read_connection() as connection:
        status, version, attempts = connection.execute(
            "SELECT status, version, attempt_count FROM inquiries WHERE id = 'inq-stale'"
        ).fetchone()
    assert (status, version, attempts) == ("open", 1, 0)


def test_changes_since_returns_new_items(tmp_path: pytest.TempPathFactory) -> None:
    """Catch changes recall that misses items after the since boundary or leaks items before it."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="Old")]),
        "seed-1",
    )
    # 1us epsilon: Windows clock ticks can be coarse, and a stamp landing on
    # the same tick as `old` would be excluded by the strict `>` comparison.
    old = datetime.now(UTC) - timedelta(microseconds=1)  # capture before the second commit
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-2", kind=ObjectKind.ENTITY, canonical_name="New")],
            assertions=[AssertionInput(id="a-1", subject_id="obj-2", predicate="p", literal="x",
                                       epistemic_role=EpistemicRole.FACT, confidence=0.9)],
            inquiries=[InquiryInput(id="inq-1", subject_id="obj-2", prompt="q", rationale="r")],
        ),
        "c-1",
    )
    recall = WorldRecall(store)
    result = recall.changes(since=old)
    assert "obj-2" in result["new_objects"]
    assert "a-1" in result["new_assertions"]
    assert "inq-1" in result["new_inquiries"]
    assert "obj-1" not in result["new_objects"]


def test_changes_defaults_to_last_attempt(tmp_path: pytest.TempPathFactory) -> None:
    """Catch changes recall that ignores the last proposal attempt as its default since."""
    store = WorldStore(tmp_path / "world.sqlite3")
    # a commit BEFORE the attempt time is excluded by the default since
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="obj-old", kind=ObjectKind.ENTITY, canonical_name="Old")]),
        "seed-0",
    )
    # insert a proposal_attempts row with attempted_at = now, post-dating seed-0
    # so the exclusion is meaningful: since defaults to the attempt time
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO proposal_attempts (commit_id, thread_id, attempted_at, outcome,"
            " new_objects, assertions, inquiries, omitted_assertions, omitted_inquiries,"
            " omitted_resolutions, resolved_inquiries, error, delta_json)"
            " VALUES ('t-1', 'th', ?, 'committed', 0, 0, 0, 0, 0, 0, 0, NULL, '{}')",
            (datetime.now(UTC).isoformat(),),
        )
    recall = WorldRecall(store)
    result = recall.changes()  # since defaults to the attempt time
    assert result["since"] is not None
    assert "obj-old" not in result["new_objects"]


def test_changes_skips_non_dict_and_idless_deltas(tmp_path: pytest.TempPathFactory) -> None:
    """Catch changes recall crashing on non-dict deltas or rendering id-less items as 'None'."""
    store = WorldStore(tmp_path / "world.sqlite3")
    # 1us epsilon: same-tick stamps must still count as after `since`.
    old = datetime.now(UTC) - timedelta(microseconds=1)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="Keeper")]),
        "seed-1",
    )
    with store.write_connection() as connection:
        for commit_id, delta in (
            ("t-1", '[1,2]'),  # non-dict delta_json must be skipped, not crash
            ("t-2", '{"objects": [{"canonical_name": "no id"}]}'),  # id-less item must not render "None"
        ):
            stamp = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT INTO commit_receipts(commit_id, committed_at, receipt_json)"
                " VALUES (?, ?, '{}')",
                (commit_id, stamp),
            )
            connection.execute(
                "INSERT INTO world_audit(commit_id, committed_at, delta_json)"
                " VALUES (?, ?, ?)",
                (commit_id, stamp, delta),
            )
    recall = WorldRecall(store)
    result = recall.changes(since=old)
    assert "obj-1" in result["new_objects"]
    assert "None" not in result["new_objects"]


def test_changes_returns_newest_first(tmp_path: pytest.TempPathFactory) -> None:
    """Catch changes recall returning the OLDEST changes (audit B-5: ASC order).

    "How the world changed" reads newest-first; the id lists must lead with
    the most recent commit even though several commits share the boundary.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    old = datetime.now(UTC) - timedelta(minutes=5)
    with store.write_connection() as connection:
        # older commit stamps 4 minutes before the boundary, newer 3 minutes
        for commit_id, age in (
            ("older", timedelta(minutes=1)),
            ("newer", timedelta(minutes=2)),
        ):
            stamp = (old + age).isoformat()
            connection.execute(
                "INSERT INTO commit_receipts(commit_id, committed_at, receipt_json)"
                " VALUES (?, ?, '{}')",
                (commit_id, stamp),
            )
            connection.execute(
                "INSERT INTO world_audit(commit_id, committed_at, delta_json)"
                " VALUES (?, ?, ?)",
                (commit_id, stamp, json.dumps({"objects": [{"id": f"obj-{commit_id}"}]})),
            )
    result = WorldRecall(store).changes(since=old)
    assert result["new_objects"] == ["obj-newer", "obj-older"]


