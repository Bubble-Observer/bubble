# Split from tests/world/test_world_recall.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime

import pytest

from leave_information_bubble.world import (
    CognitiveDelta,
    EvidenceInput,
    ObservationDepth,
    ObservationInput,
    WorldRecall,
    WorldStore,
)
from tests.world._recall_helpers import (
    NOW,
    _assertion,
    _object,
    _observation,
    _TracingWorldStore,
)


def test_assertion_card_carries_supersedes_signal(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Cards actively superseding another assertion carry the correction signal.

    The superseding card's ``supersedes`` field names the retired assertion
    (``supersedes_id``) and the stamp on the superseding row
    (``superseded_at``). Cards without a ``supersedes_id`` carry no
    ``supersedes`` field at all — the passive "who superseded this" signal
    can never fire because recall filters retired assertions out. Pre-v7
    superseders (no stamp) render ``superseded_at`` as null, a normal state.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("s16")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("assertion-old", "s16", literal="needle old title"),
                _assertion("assertion-other", "s16", literal="needle unrelated claim"),
            ],
        ),
        "old-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                _assertion("assertion-new", "s16", literal="needle new title").model_copy(
                    update={
                        "supersedes_id": "assertion-old",
                        # the committer stamps the superseding row at commit time
                        "superseded_at": datetime(2026, 8, 3, 13, 0, tzinfo=UTC),
                    }
                )
            ],
        ),
        "supersede-claim",
    )

    bundle = WorldRecall(store).search("needle")
    cards = {row["id"]: row for row in bundle.assertions}

    assert cards["assertion-new"]["supersedes"] == {
        "supersedes_id": "assertion-old",
        "superseded_at": "2026-08-03T13:00:00+00:00",
    }
    assert "supersedes" not in cards["assertion-other"]

    # a superseder without a stamp (pre-v7 data) shows null, not absent
    with sqlite3.connect(tmp_path / "world.sqlite3") as connection:
        connection.execute("UPDATE assertions SET superseded_at = NULL WHERE id = 'assertion-new'")
    refreshed = {row["id"]: row for row in WorldRecall(store).search("needle").assertions}
    assert refreshed["assertion-new"]["supersedes"] == {
        "supersedes_id": "assertion-old",
        "superseded_at": None,
    }


def test_assertion_card_carries_source_flip_flop_counts(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Cards report how often each evidence source flip-flopped (superseded claims).

    A source's flip-flop count is the number of superseded assertions citing
    it: ``source_flip_flops`` maps only sources with count > 0, so a source
    with zero corrections stays absent. The card can carry both correction
    signals at once, and clean assertions carry neither field.
    """
    source_a = "https://example.test/source-a"
    source_b = "https://example.test/source-b"
    source_c = "https://example.test/source-c"
    source_d = "https://example.test/source-d"

    def observation(identifier: str, uri: str) -> ObservationInput:
        return ObservationInput(
            id=identifier,
            source_uri=uri,
            source_kind="web",
            depth=ObservationDepth.CONTENT,
            observed_at=NOW,
        )

    def claim(observation_id: str) -> EvidenceInput:
        return EvidenceInput(observation_id=observation_id, role="supports")

    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("s16")],
            observations=[
                observation("obs-a", source_a),
                observation("obs-b", source_b),
                observation("obs-c", source_c),
                observation("obs-d", source_d),
            ],
            assertions=[
                _assertion("flip-1", "s16", literal="flip one").model_copy(
                    update={"evidence": [claim("obs-a")]}
                ),
                _assertion("flip-2", "s16", literal="flip two").model_copy(
                    update={"evidence": [claim("obs-a")]}
                ),
                _assertion("old-claim", "s16", literal="needle old claim").model_copy(
                    update={"evidence": [claim("obs-c")]}
                ),
                _assertion("clean-claim", "s16", literal="needle clean claim").model_copy(
                    update={"evidence": [claim("obs-d")]}
                ),
            ],
        ),
        "seed-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            assertions=[
                _assertion("superseder-1", "s16", literal="replacement one").model_copy(
                    update={"supersedes_id": "flip-1", "evidence": [claim("obs-a")]}
                ),
                _assertion("superseder-2", "s16", literal="replacement two").model_copy(
                    update={"supersedes_id": "flip-2", "evidence": [claim("obs-a")]}
                ),
                _assertion("card-claim", "s16", literal="needle claim").model_copy(
                    update={
                        "supersedes_id": "old-claim",
                        "evidence": [claim("obs-a"), claim("obs-b")],
                    }
                ),
            ],
        ),
        "supersede-claims",
    )

    bundle = WorldRecall(store).search("needle")
    cards = {row["id"]: row for row in bundle.assertions}
    assert set(cards) == {"card-claim", "clean-claim"}

    # source A has 2 superseded assertions (flip-1, flip-2); B has none
    assert cards["card-claim"]["source_flip_flops"] == {source_a: 2}
    assert source_b not in cards["card-claim"]["source_flip_flops"]
    # both correction signals ride the same card
    assert cards["card-claim"]["supersedes"]["supersedes_id"] == "old-claim"

    # clean assertion: neither signal
    assert "source_flip_flops" not in cards["clean-claim"]
    assert "supersedes" not in cards["clean-claim"]


def test_source_flip_flops_bound_card_uris_and_preserve_long_uri_identity(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Flip-flop maps use SQL-limited source sets and hash-suffixed long URI labels."""
    long_prefix = "https://example.test/a" + "x" * 300
    source_uris = [
        long_prefix + "-one",
        long_prefix + "-two",
        *[f"https://example.test/simple-{i}" for i in range(9)],
    ]
    store = _TracingWorldStore(str(tmp_path / "world.sqlite3"))
    observations = [
        _observation(f"source-{index}").model_copy(update={"source_uri": uri})
        for index, uri in enumerate(source_uris)
    ]
    card_evidence = [
        EvidenceInput(observation_id=observation.id, role="supports") for observation in observations
    ]
    old_assertions = [
        _assertion(f"old-{index}", "needle", literal=f"retired {index}").model_copy(
            update={"evidence": [EvidenceInput(observation_id=observation.id, role="supports")]}
        )
        for index, observation in enumerate(observations)
    ]
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("needle")],
            observations=observations,
            assertions=[
                _assertion("card-claim", "needle", literal="needle card").model_copy(
                    update={"evidence": card_evidence}
                ),
                *old_assertions,
            ],
        ),
        "flip-uri-seed",
    )
    store.memory_commit(
        CognitiveDelta(
            assertions=[
                _assertion(f"replacement-{index}", "needle", literal=f"replacement {index}").model_copy(
                    update={
                        "supersedes_id": f"old-{index}",
                        "evidence": [EvidenceInput(observation_id=observation.id, role="supports")],
                    }
                )
                for index, observation in enumerate(observations)
            ]
        ),
        "flip-uri-replacements",
    )

    card = WorldRecall(store).search("needle card").assertions[0]
    flips = card["source_flip_flops"]

    assert card["source_flip_flops_truncated"] is True
    assert len(flips) == 8
    assert all(len(uri) <= 240 for uri in flips)
    long_labels = [uri for uri in flips if "~" in uri]
    assert len(long_labels) == 2
    assert len(set(long_labels)) == 2
    assert all(re.fullmatch(r".*~[0-9a-f]{16}", uri) for uri in long_labels)
    assert any(
        "ROW_NUMBER() OVER (PARTITION BY assertion_id ORDER BY source_uri)" in query
        for query in store.statements
    )


def test_assertion_qualifiers_round_trip_through_recall(tmp_path: pytest.TempPathFactory) -> None:
    """recall/detail must return stored qualifiers as a parsed dict."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("object-1")],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("assertion-1", "object-1", literal="remembered").model_copy(
                    update={"qualifiers": {"role": "home", "community": "cn"}}
                )
            ],
        ),
        "qualifiers-commit",
    )
    recall = WorldRecall(store)

    card = recall.search("remembered").assertions[0]
    assert card["qualifiers"] == {"role": "home", "community": "cn"}

    detail = recall.evidence("assertion-1").assertions[0]
    assert detail["qualifiers"] == {"role": "home", "community": "cn"}


def test_legacy_assertion_without_qualifiers_recalls_empty_dict(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Pre-v10 rows (qualifiers_json IS NULL) must recall as an empty map."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("object-1")],
            observations=[_observation("observation-1")],
            assertions=[_assertion("assertion-1", "object-1", literal="legacy note")],
        ),
        "legacy-commit",
    )
    with sqlite3.connect(store.path) as connection:
        # simulate a pre-v10 row regardless of what the modern write path stores
        connection.execute("UPDATE assertions SET qualifiers_json = NULL WHERE id = 'assertion-1'")

    card = WorldRecall(store).search("legacy note").assertions[0]
    assert card["qualifiers"] == {}
