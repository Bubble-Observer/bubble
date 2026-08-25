# Split from tests/world/test_world_recall.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

from datetime import timedelta

import pytest

from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    ObservationDepth,
    RecallClock,
    WorldRecall,
    WorldStore,
)
from tests.world._recall_helpers import (
    NOW,
    _assertion,
    _object,
    _observation,
)


def test_world_as_of_and_agent_knew_as_of_differ_for_late_observation(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch recall that treats an event's time as when the agent learned it."""
    store = WorldStore(tmp_path / "world.sqlite3")
    receipt = store.memory_commit(
        CognitiveDelta(
            objects=[_object("launch")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion(
                    "assertion-1", "launch", literal="The launch happened", event_time_start=NOW
                )
            ],
        ),
        "late-observation",
    )
    recall = WorldRecall(store)

    world = recall.search("launch", as_of=NOW, clock=RecallClock.WORLD)
    knowledge = recall.search(
        "launch", as_of=receipt.committed_at - timedelta(microseconds=1), clock=RecallClock.KNOWLEDGE
    )

    assert [row["id"] for row in world.assertions] == ["assertion-1"]
    assert knowledge.assertions == []


def test_as_of_recall_excludes_evidence_linked_or_published_after_cutoff(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch later evidence leaking into an old assertion's historical bundle."""
    store = WorldStore(tmp_path / "world.sqlite3")
    first = store.memory_commit(
        CognitiveDelta(
            objects=[_object("subject")],
            observations=[
                _observation("observation-1").model_copy(
                    update={"source_published_at": NOW - timedelta(days=1)}
                )
            ],
            assertions=[_assertion("assertion-1", "subject", literal="historical answer")],
        ),
        "original-evidence",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[
                _observation("observation-2").model_copy(
                    update={"source_published_at": NOW + timedelta(days=1)}
                )
            ],
            assertions=[
                _assertion("assertion-2", "subject", literal="historical answer").model_copy(
                    update={
                        "evidence": [
                            EvidenceInput(observation_id="observation-2", role="supports")
                        ]
                    }
                )
            ],
        ),
        "later-evidence",
    )
    recall = WorldRecall(store)

    knowledge = recall.search(
        "historical",
        as_of=first.committed_at,
        clock=RecallClock.KNOWLEDGE,
    )
    world = recall.search("historical", as_of=NOW, clock=RecallClock.WORLD)

    assert [row["id"] for row in knowledge.assertions] == ["assertion-1"]
    assert [row["observation_id"] for row in knowledge.evidence_refs] == ["observation-1"]
    assert [row["observation_id"] for row in world.evidence_refs] == ["observation-1"]


def test_recall_basis_summary_distinguishes_global_detail_truncation_and_material_fields(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Recall keeps each remembered judgment's material basis visible and bounded."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("needle")],
            observations=[
                _observation("observation-0").model_copy(
                    update={
                        "depth": ObservationDepth.SEEN,
                        "source_kind": "search",
                        "metadata": {"material_reliability": "unknown"},
                    }
                ),
                _observation("observation-1").model_copy(
                    update={
                        "depth": ObservationDepth.CONTENT,
                        "source_kind": "web",
                        "metadata": {"material_reliability": "source_direct"},
                    }
                ),
                _observation("observation-2").model_copy(
                    update={
                        "depth": ObservationDepth.DISCUSSION,
                        "source_kind": "forum",
                        "metadata": {
                            "material_reliability": "best_effort",
                            "limitations": ["sampled_thread"],
                        },
                    }
                ),
                _observation("observation-3").model_copy(
                    update={
                        "depth": ObservationDepth.MEDIA,
                        "source_kind": "video",
                        "metadata": {"material_reliability": "automatic"},
                    }
                ),
            ],
            assertions=[
                _assertion("assertion-a", "needle", literal="needle first").model_copy(
                    update={
                        "evidence": [
                            EvidenceInput(observation_id="observation-0", role="context"),
                            EvidenceInput(observation_id="observation-1", role="supports"),
                            EvidenceInput(observation_id="observation-2", role="contradicts"),
                        ]
                    }
                ),
                _assertion("assertion-b", "needle", literal="needle second").model_copy(
                    update={"evidence": [EvidenceInput(observation_id="observation-3", role="supports")]}
                ),
            ],
        ),
        "recall-evidence-cap",
    )

    bundle = WorldRecall(store).search("needle", limit=2)

    assert [row["observation_id"] for row in bundle.evidence_refs] == [
        "observation-0",
        "observation-1",
    ]
    assert set(bundle.evidence_refs[0]) == {
        "assertion_id",
        "observation_id",
        "role",
        "linked_at",
        "source_uri",
        "title",
        "content_ref",
        "depth",
        "source_kind",
        "source_published_at",
        "observed_at",
        "material_reliability",
        "limitations",
    }
    first, second = bundle.assertions
    assert first["basis_summary"] == {
        "total_links": 3,
        "by_role": {"supports": 1, "context": 1, "contradicts": 1},
        "by_depth": {"seen": 1, "content": 1, "discussion": 1, "media": 0},
        "by_source_kind": {"search": 1, "web": 1, "forum": 1},
        "source_kinds_truncated": False,
        "source_kind_values_truncated": False,
        "time_range": {
            "published_at": {"earliest": None, "latest": None},
            "observed_at": {"earliest": NOW.isoformat(), "latest": NOW.isoformat()},
        },
        "has_low_reliability": True,
        "has_limitations": True,
        "representative_refs": [
            {
                "observation_id": "observation-0",
                "role": "context",
                "depth": "seen",
                "source_kind": "search",
                "source_kind_truncated": False,
            },
            {
                "observation_id": "observation-1",
                "role": "supports",
                "depth": "content",
                "source_kind": "web",
                "source_kind_truncated": False,
            },
        ],
        "refs_truncated": True,
    }
    assert second["basis_summary"]["total_links"] == 1
    assert second["basis_summary"]["representative_refs"] == [
        {
            "observation_id": "observation-3",
            "role": "supports",
            "depth": "media",
            "source_kind": "video",
            "source_kind_truncated": False,
        }
    ]
    assert second["basis_summary"]["refs_truncated"] is False


def test_recall_assertion_with_no_basis_is_not_confused_with_a_truncated_summary(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A valid zero-link remembered judgment has explicit zero provenance, not missing data."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("needle")],
            assertions=[
                AssertionInput(
                    id="assertion-without-basis",
                    subject_id="needle",
                    predicate="related_to",
                    literal="remembered without a material link",
                    epistemic_role=EpistemicRole.UNCERTAINTY,
                    confidence=0.2,
                )
            ],
        ),
        "zero-basis",
    )

    card = WorldRecall(store).search("remembered").assertions[0]

    assert card["basis_summary"] == {
        "total_links": 0,
        "by_role": {"supports": 0, "context": 0, "contradicts": 0},
        "by_depth": {"seen": 0, "content": 0, "discussion": 0, "media": 0},
        "by_source_kind": {},
        "source_kinds_truncated": False,
        "source_kind_values_truncated": False,
        "time_range": {
            "published_at": {"earliest": None, "latest": None},
            "observed_at": {"earliest": None, "latest": None},
        },
        "has_low_reliability": False,
        "has_limitations": False,
        "representative_refs": [],
        "refs_truncated": False,
    }


def test_recall_bounds_source_kind_buckets_and_representative_values(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Source-kind summaries retain the leading eight buckets and a counted remainder."""
    top_long = "top-" + "x" * 100
    shared_prefix = "other-" + "y" * 100
    kinds = [
        top_long,
        top_long,
        *[f"kind-{index}" for index in range(7)],
        shared_prefix + "-a",
        shared_prefix + "-b",
    ]
    observations = [
        _observation(f"observation-{index:02}").model_copy(update={"source_kind": kind})
        for index, kind in enumerate(kinds)
    ]
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("needle")],
            observations=observations,
            assertions=[
                _assertion("kind-claim", "needle", literal="needle kind summary").model_copy(
                    update={
                        "evidence": [
                            EvidenceInput(observation_id=observation.id, role="supports")
                            for observation in observations
                        ]
                    }
                )
            ],
        ),
        "kind-summary",
    )

    summary = WorldRecall(store).search("kind summary").assertions[0]["basis_summary"]

    assert summary["source_kinds_truncated"] is True
    assert summary["source_kind_values_truncated"] is True
    assert len(summary["by_source_kind"]) == 9
    assert summary["by_source_kind"]["other"] == 2
    assert all(len(kind) <= 80 for kind in summary["by_source_kind"])
    assert summary["representative_refs"][0]["source_kind"] == top_long[:80]
    assert summary["representative_refs"][0]["source_kind_truncated"] is True


def test_recall_malformed_metadata_and_limitations_are_safe_and_bounded(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Detail provenance survives malformed legacy JSON and limits oversized limitation lists."""
    raw_limitations = [f"{index}-" + "x" * 200 for index in range(10)]
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("needle")],
            observations=[
                _observation("limited").model_copy(update={"metadata": {"limitations": raw_limitations}}),
                _observation("malformed"),
            ],
            assertions=[
                _assertion("limited-claim", "needle", literal="needle limited").model_copy(
                    update={
                        "evidence": [
                            EvidenceInput(observation_id="limited", role="supports"),
                            EvidenceInput(observation_id="malformed", role="context"),
                        ]
                    }
                )
            ],
        ),
        "limited-metadata",
    )
    with store.write_connection() as connection:
        connection.execute(
            "UPDATE observations SET metadata_json = ? WHERE id = ?", ("not-json", "malformed")
        )

    bundle = WorldRecall(store).search("limited")
    limitations = bundle.evidence_refs[0]["limitations"]

    assert limitations[-1] == "limitations_truncated"
    assert len(limitations) <= 8
    assert max(map(len, limitations)) <= 160
    assert sum(map(len, limitations)) <= 640
    malformed = next(ref for ref in bundle.evidence_refs if ref["observation_id"] == "malformed")
    assert malformed["material_reliability"] == "unknown"
    assert malformed["limitations"] == []


def test_as_of_recall_does_not_leak_later_source_flip_flops(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Correction signals only reflect superseders visible to the requested recall clock."""
    source = "https://example.test/flip-source"
    observation = _observation("flip-observation").model_copy(update={"source_uri": source})
    store = WorldStore(tmp_path / "world.sqlite3")
    first = store.memory_commit(
        CognitiveDelta(
            objects=[_object("needle")],
            observations=[observation],
            assertions=[
                _assertion("card-claim", "needle", literal="needle card").model_copy(
                    update={"evidence": [EvidenceInput(observation_id="flip-observation", role="supports")]}
                ),
                _assertion("old-claim", "needle", literal="old claim").model_copy(
                    update={"evidence": [EvidenceInput(observation_id="flip-observation", role="supports")]}
                ),
            ],
        ),
        "before-flip",
    )
    store.memory_commit(
        CognitiveDelta(
            assertions=[
                _assertion("replacement", "needle", literal="replacement claim").model_copy(
                    update={
                        "supersedes_id": "old-claim",
                        "evidence": [EvidenceInput(observation_id="flip-observation", role="supports")],
                    }
                )
            ]
        ),
        "after-flip",
    )

    recall = WorldRecall(store)
    knowledge = {
        row["id"]: row
        for row in recall.search("needle", as_of=first.committed_at, clock=RecallClock.KNOWLEDGE).assertions
    }
    world = {row["id"]: row for row in recall.search("needle", as_of=NOW, clock=RecallClock.WORLD).assertions}
    current = {row["id"]: row for row in recall.search("needle").assertions}

    assert "source_flip_flops" not in knowledge["card-claim"]
    assert "source_flip_flops" not in world["card-claim"]
    assert current["card-claim"]["source_flip_flops"] == {source: 1}


def test_knowledge_as_of_flip_flops_exclude_late_deduplicated_evidence_links(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A later evidence append cannot retroactively make an old claim a source flip-flop."""
    source_a = "https://example.test/source-a"
    source_b = "https://example.test/source-b"
    store = WorldStore(tmp_path / "world.sqlite3")
    first = store.memory_commit(
        CognitiveDelta(
            objects=[_object("needle")],
            observations=[
                _observation("card-source").model_copy(update={"source_uri": source_a}),
                _observation("old-source").model_copy(update={"source_uri": source_b}),
            ],
            assertions=[
                _assertion("card-claim", "needle", literal="needle card").model_copy(
                    update={"evidence": [EvidenceInput(observation_id="card-source", role="supports")]}
                ),
                _assertion("old-claim", "needle", literal="old claim").model_copy(
                    update={"evidence": [EvidenceInput(observation_id="old-source", role="supports")]}
                ),
            ],
        ),
        "dedup-before",
    )
    second = store.memory_commit(
        CognitiveDelta(
            assertions=[
                _assertion("replacement", "needle", literal="replacement").model_copy(
                    update={
                        "supersedes_id": "old-claim",
                        "evidence": [EvidenceInput(observation_id="old-source", role="supports")],
                    }
                )
            ]
        ),
        "dedup-supersede",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("late-source").model_copy(update={"source_uri": source_a})]
        ),
        "dedup-late-observation",
    )
    with store.write_connection() as connection:
        connection.execute(
            "INSERT INTO assertion_evidence(assertion_id, observation_id, role, linked_at) "
            "VALUES (?, ?, ?, ?)",
            (
                "old-claim",
                "late-source",
                "supports",
                (second.committed_at + timedelta(seconds=1)).isoformat(),
            ),
        )

    before_append = {
        row["id"]: row
        for row in WorldRecall(store)
        .search("needle", as_of=second.committed_at, clock=RecallClock.KNOWLEDGE)
        .assertions
    }
    current = {row["id"]: row for row in WorldRecall(store).search("needle").assertions}

    assert "source_flip_flops" not in before_append["card-claim"]
    assert current["card-claim"]["source_flip_flops"] == {source_a: 1}


def test_knowledge_as_of_retires_only_after_replacement_commit(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch as-of knowledge recall retiring an assertion before its replacement committed (B2-8).

    The supersede relationship is a knowledge-commit fact: at the moment the
    replacement was committed the old claim was still current, and an as-of
    snapshot of that moment must still show it. After the replacement commit
    the old claim retires.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    original = store.memory_commit(
        CognitiveDelta(
            objects=[_object("s16")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("assertion-old", "s16", literal="needle old title")],
        ),
        "old-claims",
    )
    replacement = store.memory_commit(
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
    recall = WorldRecall(store)
    assert replacement.committed_at > original.committed_at  # the two snapshot moments are distinct

    before = recall.search("needle", as_of=original.committed_at, clock=RecallClock.KNOWLEDGE)
    after = recall.search("needle", as_of=replacement.committed_at, clock=RecallClock.KNOWLEDGE)

    assert [row["id"] for row in before.assertions] == ["assertion-old"]
    assert [row["id"] for row in after.assertions] == ["assertion-new"]


def test_world_as_of_keeps_superseded_assertion_inside_its_event_window(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch world-clock snapshots losing event-time-valid assertions to knowledge retirement (B2-8).

    World-clock as-of queries reconstruct what the world was like at an event
    time; a later knowledge-layer replacement does not change that. The old
    title assertion stays visible for a moment inside its event window, while
    the current recall surface (no as_of) shows only the replacement.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("s16")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion(
                    "assertion-old",
                    "s16",
                    literal="needle old title",
                    event_time_start=NOW - timedelta(days=1),
                )
            ],
        ),
        "old-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion(
                    "assertion-new",
                    "s16",
                    literal="needle new title",
                    event_time_start=NOW + timedelta(days=1),
                ).model_copy(update={"supersedes_id": "assertion-old"})
            ],
        ),
        "supersede-claim",
    )
    recall = WorldRecall(store)

    world = recall.search("needle", as_of=NOW, clock=RecallClock.WORLD)
    current = recall.search("needle")

    assert [row["id"] for row in world.assertions] == ["assertion-old"]
    assert [row["id"] for row in current.assertions] == ["assertion-new"]


