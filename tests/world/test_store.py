from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from leave_information_bubble.world import (
    AssertionInput,
    CognitionDeltaProposal,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    GraphRef,
    InquiryInput,
    InquiryResolution,
    JsonValue,
    NewInquiryProposal,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    ProposalCommitter,
    WorldStore,
)
from leave_information_bubble.world import contracts as world_contracts
from leave_information_bubble.world.contracts import AliasAction, AliasOperation
from leave_information_bubble.world.proposal import ReviewIssueCode, ReviewOutcome
from leave_information_bubble.world.store import (
    CommitReplayConflict,
    _assertion_signature,
    normalize_alias,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

# sha256 of the v1 six-field payload for the _assertion() fixture, computed
# from the pre-v10 algorithm: ["object-1","related_to","confirmed","fact",null,null]
FROZEN_V9_SIGNATURE = "45892a3aa9ea0ee84f1f3644a74b370d06dc641ce6d4df2c77708990b7fd2640"


def _observation(
    identifier: str,
    *,
    depth: ObservationDepth = ObservationDepth.CONTENT,
    metadata: dict[str, JsonValue] | None = None,
    observed_at: datetime = NOW,
) -> ObservationInput:
    return ObservationInput(
        id=identifier,
        source_uri=f"https://example.test/{identifier}",
        source_kind="web",
        depth=depth,
        observed_at=observed_at,
        metadata={} if metadata is None else metadata,
    )


def _object(
    identifier: str = "object-1",
    *,
    expected_version: int | None = None,
    event_time_start: datetime | None = None,
    aliases: list[str] | None = None,
) -> ObjectInput:
    return ObjectInput(
        id=identifier,
        kind=ObjectKind.ENTITY,
        canonical_name=identifier.title(),
        expected_version=expected_version,
        event_time_start=event_time_start,
        aliases=[] if aliases is None else aliases,
    )


def _assertion(
    identifier: str = "assertion-1",
    *,
    evidence_id: str = "observation-1",
    literal: str = "confirmed",
    supersedes_id: str | None = None,
    qualifiers: dict[str, str] | None = None,
) -> AssertionInput:
    return AssertionInput(
        id=identifier,
        subject_id="object-1",
        predicate="related_to",
        literal=literal,
        epistemic_role=EpistemicRole.FACT,
        confidence=0.8,
        evidence=[EvidenceInput(observation_id=evidence_id, role="supports")],
        supersedes_id=supersedes_id,
        qualifiers=qualifiers,
    )


def _count(path: str, table: str) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _seed_legacy_alias_row(path: str, object_id: str, canonical_name: str, normalized: str) -> None:
    """Insert an object and a legacy space-preserving alias row directly.

    Legacy databases predate whitespace folding and stored aliases with
    internal whitespace preserved (e.g. ``'t1 stars 男团出道'``); writing
    these rows by hand simulates that era because the store now persists
    only the collapsed form.
    """
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO objects(id, kind, canonical_name, domain_hints_json,"
            " provisional, event_time_start, event_time_end, version)"
            " VALUES (?, 'event', ?, '[\"lol\"]', 0, NULL, NULL, 1)",
            (object_id, canonical_name),
        )
        connection.execute(
            "INSERT INTO object_aliases(normalized_alias, object_id) VALUES (?, ?)",
            (normalized, object_id),
        )


def test_observation_accepts_nested_json_metadata() -> None:
    """Catch contracts that cannot represent recursive JSON observation metadata."""
    observation = _observation("observation-json", metadata={"nested": [True, {"count": 2}]})

    assert observation.metadata == {"nested": [True, {"count": 2}]}


def test_open_without_schema_initialization_requires_existing_database(tmp_path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        WorldStore(missing, initialize_schema=False)
    assert not missing.exists()

    initialized = WorldStore(tmp_path / "world.sqlite3")
    opened = WorldStore(initialized.path, initialize_schema=False)
    with opened.read_connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 18


def test_replaying_commit_id_returns_same_receipt_without_new_rows(tmp_path: pytest.TempPathFactory) -> None:
    """Catch commits that write new data when their idempotency key is replayed."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    delta = CognitiveDelta(objects=[_object()], observations=[_observation("observation-1")])

    first = store.memory_commit(delta, "commit-1")
    replay = store.memory_commit(delta, "commit-1")

    assert replay.commit_id == first.commit_id == "commit-1"
    assert replay.object_ids == ["object-1"]
    assert replay.observation_ids == ["observation-1"]
    assert replay.replayed is True
    assert _count(path, "objects") == 1
    assert _count(path, "observations") == 1
    assert _count(path, "world_audit") == 1


def test_replaying_commit_id_rejects_a_semantically_different_delta(tmp_path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    original = CognitiveDelta(objects=[_object()])
    store.memory_commit(original, "same-id")

    with pytest.raises(ValueError, match="different cognitive delta"):
        store.memory_commit(
            CognitiveDelta(
                objects=[_object().model_copy(update={"canonical_name": "Different"})]
            ),
            "same-id",
        )

    with store.read_connection() as connection:
        stored = CognitiveDelta.model_validate_json(
            connection.execute(
                "SELECT delta_json FROM world_audit WHERE commit_id = 'same-id'"
            ).fetchone()[0]
        )
    assert stored == original


def test_replay_delta_comparison_normalizes_json_and_utc_datetimes(tmp_path) -> None:
    store = WorldStore(tmp_path / "world.sqlite3")
    delta = CognitiveDelta(
        inquiries=[
            InquiryInput(
                id="inq-normalized",
                subject_id="obj-normalized",
                prompt="Why?",
                rationale="r",
                created_at=datetime(2026, 8, 13, 8, tzinfo=UTC),
            )
        ],
        objects=[
            ObjectInput(
                id="obj-normalized",
                kind=ObjectKind.ENTITY,
                canonical_name="Normalized",
            )
        ],
    )
    store.memory_commit(delta, "normalized-id")
    with store.write_connection() as connection:
        payload = json.loads(
            connection.execute(
                "SELECT delta_json FROM world_audit WHERE commit_id = 'normalized-id'"
            ).fetchone()[0]
        )
        payload["inquiries"][0]["created_at"] = "2026-08-13T16:00:00+08:00"
        connection.execute(
            "UPDATE world_audit SET delta_json = ? WHERE commit_id = 'normalized-id'",
            (json.dumps(payload, indent=2, sort_keys=True),),
        )

    replay = store.memory_commit(delta, "normalized-id")

    assert replay.replayed is True


def test_commit_registers_a_new_domain_predicate_with_its_assertion(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the seed predicate vocabulary becoming a closed domain-specific whitelist."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    assertion = _assertion().model_copy(update={"predicate": "won_match_against"})

    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            assertions=[assertion],
        ),
        "open-predicate-registry",
    )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT name FROM predicates WHERE name = 'won_match_against'"
        ).fetchone() == ("won_match_against",)
        assert connection.execute("SELECT predicate FROM assertions WHERE id = 'assertion-1'").fetchone() == (
            "won_match_against",
        )


def test_candidate_observation_links_commit_with_objects_and_inquiries(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch Broad source leads that are persisted but disconnected from cognition."""
    assert hasattr(world_contracts, "ObservationLinkInput"), (
        "CognitiveDelta must represent candidate Observation graph links"
    )
    link_type = world_contracts.ObservationLinkInput
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)

    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("seen-1", depth=ObservationDepth.SEEN)],
            objects=[_object("event-1")],
            inquiries=[
                InquiryInput(
                    id="inquiry-1",
                    subject_id="event-1",
                    prompt="最终比分是什么？",
                    rationale="完善比赛的事实核心。",
                )
            ],
            observation_links=[
                link_type(
                    target_kind="object",
                    target_id="event-1",
                    observation_id="seen-1",
                    role="candidate",
                ),
                link_type(
                    target_kind="inquiry",
                    target_id="inquiry-1",
                    observation_id="seen-1",
                    role="candidate",
                ),
            ],
        ),
        "candidate-links",
    )

    assert _count(path, "object_observations") == 1
    assert _count(path, "inquiry_observations") == 1


def test_link_targeting_missing_inquiry_is_skipped_not_fatal(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch candidate links referencing an inquiry that will never exist."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)

    receipt = store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-1")],
            observation_links=[
                world_contracts.ObservationLinkInput(
                    target_kind="inquiry",
                    target_id="inquiry-ghost",
                    observation_id="observation-1",
                    role="candidate",
                )
            ],
        ),
        "ghost-target",
    )

    assert receipt.inquiry_ids == []
    assert _count(path, "inquiry_observations") == 0  # link dropped, not fatal
    assert _count(path, "commit_receipts") == 1  # the commit itself succeeded


def test_link_to_dedup_swallowed_inquiry_is_skipped_not_fatal(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the verification crash: an inquiry whose insert the open-inquiry
    dedup index silently swallows must not leave its link firing the FK."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object("event-1")],
            observations=[_observation("observation-1")],
            inquiries=[
                InquiryInput(
                    id="inquiry-existing",
                    subject_id="event-1",
                    prompt="IG 的 has_result 关系未获确认",
                    rationale="earlier recovery",
                )
            ],
        ),
        "seed",
    )
    # the delta declares a *different* id for the same open (subject_id,
    # prompt): the insert is silently swallowed by idx_open_inquiry_dedup
    receipt = store.memory_commit(
        CognitiveDelta(
            inquiries=[
                InquiryInput(
                    id="inquiry-sibling",
                    subject_id="event-1",
                    prompt="IG 的 has_result 关系未获确认",
                    rationale="later recovery",
                )
            ],
            observation_links=[
                world_contracts.ObservationLinkInput(
                    target_kind="inquiry",
                    target_id="inquiry-sibling",
                    observation_id="observation-1",
                    role="candidate",
                )
            ],
        ),
        "swallowed-target",
    )
    assert receipt.inquiry_ids == ["inquiry-sibling"]  # declared, even though not stored
    assert _count(path, "inquiries") == 1  # the sibling insert was swallowed
    assert _count(path, "inquiry_observations") == 0  # its link was dropped, not fatal


def test_missing_candidate_observation_rolls_back_the_whole_delta(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch candidate links that bypass Observation identity validation."""
    assert hasattr(world_contracts, "ObservationLinkInput")
    link_type = world_contracts.ObservationLinkInput
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)

    with pytest.raises(ValueError, match="observation"):
        store.memory_commit(
            CognitiveDelta(
                objects=[_object("event-1")],
                inquiries=[
                    InquiryInput(
                        id="inquiry-1",
                        subject_id="event-1",
                        prompt="最终比分是什么？",
                        rationale="完善比赛的事实核心。",
                    )
                ],
                observation_links=[
                    link_type(
                        target_kind="inquiry",
                        target_id="inquiry-1",
                        observation_id="missing-observation",
                        role="candidate",
                    )
                ],
            ),
            "missing-candidate",
        )

    for table in (
        "objects",
        "inquiries",
        "object_observations",
        "inquiry_observations",
        "commit_receipts",
        "world_audit",
    ):
        assert _count(path, table) == 0


def test_invalid_evidence_rolls_back_objects_assertions_inquiries_and_audit(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch partial writes when an assertion references an unknown observation."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    delta = CognitiveDelta(
        objects=[_object()],
        assertions=[_assertion(evidence_id="missing-observation")],
        inquiries=[InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap")],
    )

    with pytest.raises(ValueError, match="observation"):
        store.memory_commit(delta, "commit-invalid-evidence")

    assert _count(path, "objects") == 0
    assert _count(path, "assertions") == 0
    assert _count(path, "inquiries") == 0
    assert _count(path, "world_audit") == 0


def test_seen_observation_can_retain_agent_selected_support_role(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Store treats evidence role and observation depth as independent dimensions."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    assertion = _assertion().model_copy(
        update={
            "evidence": [
                EvidenceInput(observation_id="observation-1", role="supports"),
                EvidenceInput(observation_id="observation-2", role="supports"),
            ]
        }
    )
    delta = CognitiveDelta(
        objects=[_object()],
        observations=[
            _observation("observation-1", depth=ObservationDepth.SEEN),
            _observation("observation-2", depth=ObservationDepth.SEEN),
        ],
        assertions=[assertion],
    )

    receipt = store.memory_commit(delta, "commit-seen")

    assert receipt.assertion_ids == ["assertion-1"]
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT observation_id, role FROM assertion_evidence ORDER BY observation_id"
        ).fetchall()
    assert rows == [("observation-1", "supports"), ("observation-2", "supports")]


def test_hydrate_upgrade_lets_a_seen_row_support_an_assertion(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the unified-id upgrade path rejecting a hydrate of a discovery row.

    Discovery writes a SEEN row; hydrating the same source writes the SAME
    id at a deeper depth (observation_id, task T13). That deeper proposal is
    the legitimate upgrade path — the delta's depth is adapter-derived from
    a real hydration, never model-fabricated — so the monotone guard must
    accept it (raising the row's depth for evidence) instead of treating it
    as an identity conflict. Observation depth remains independently visible
    from the Agent-selected evidence role.
    """
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1", depth=ObservationDepth.SEEN)],
        ),
        "commit-seen-observation",
    )

    receipt = store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-1", depth=ObservationDepth.CONTENT)],
            assertions=[_assertion()],
        ),
        "commit-hydrated-observation",
    )

    assert len(receipt.assertion_ids) == 1
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT depth FROM observations WHERE id = 'observation-1'"
        ).fetchone()
        role = connection.execute(
            "SELECT role FROM assertion_evidence WHERE observation_id = 'observation-1'"
        ).fetchone()
    assert row["depth"] == "content"
    assert role["role"] == "supports"
    assert _count(path, "world_audit") == 2


def test_existing_assertion_id_with_different_signature_rejects_whole_delta(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a conflicting assertion identifier after another delta row is written."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            assertions=[_assertion(literal="first")],
        ),
        "commit-first-assertion",
    )

    with pytest.raises(ValueError, match="assertion id"):
        store.memory_commit(
            CognitiveDelta(
                observations=[_observation("observation-2")],
                assertions=[_assertion(literal="second", evidence_id="observation-2")],
            ),
            "commit-conflicting-assertion",
        )

    assert _count(path, "observations") == 1
    assert _count(path, "assertions") == 1
    assert _count(path, "world_audit") == 1


def test_missing_superseded_assertion_rejects_whole_delta(tmp_path: pytest.TempPathFactory) -> None:
    """Catch an assertion whose superseded predecessor cannot be resolved before writes."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object()], observations=[_observation("observation-1")]),
        "commit-object",
    )

    with pytest.raises(ValueError, match="supersedes"):
        store.memory_commit(
            CognitiveDelta(
                observations=[_observation("observation-2")],
                assertions=[_assertion(evidence_id="observation-2", supersedes_id="missing-assertion")],
            ),
            "commit-missing-superseded",
        )

    assert _count(path, "observations") == 1
    assert _count(path, "assertions") == 0
    assert _count(path, "world_audit") == 1


def test_store_persists_superseded_at_from_assertion_input(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the assertion INSERT dropping the correction-history timestamp.

    ``superseded_at`` travels on the superseding ``AssertionInput`` (filled by
    the committer) and must survive the pure INSERT unchanged; a NULL input
    value persists as NULL so current assertions keep an empty stamp.
    """
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            assertions=[_assertion("assertion-old")],
        ),
        "old-claims",
    )
    moment = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[
                AssertionInput(
                    id="assertion-new",
                    subject_id="object-1",
                    predicate="related_to",
                    literal="updated",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="observation-2", role="supports")],
                    supersedes_id="assertion-old",
                    superseded_at=moment,
                )
            ],
        ),
        "supersede-claim",
    )

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT superseded_at FROM assertions WHERE id = ?", ("assertion-new",)
        ).fetchone()
    assert datetime.fromisoformat(row["superseded_at"]) == moment


def test_existing_object_alias_mutation_requires_expected_version(tmp_path: pytest.TempPathFactory) -> None:
    """Catch an unversioned object reuse that silently adds an alias."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(CognitiveDelta(objects=[_object(aliases=["alphaalias"])]), "commit-object")

    with pytest.raises(ValueError, match="expected version"):
        store.memory_commit(
            CognitiveDelta(objects=[_object(aliases=["betaalias"])]),
            "commit-unversioned-alias",
        )

    with sqlite3.connect(path) as connection:
        aliases = connection.execute(
            "SELECT normalized_alias FROM identity_aliases"
        ).fetchall()
    assert ("betaalias",) not in aliases


def test_versioned_object_update_rebuilds_fts_with_retained_aliases(tmp_path: pytest.TempPathFactory) -> None:
    """Catch FTS rebuilds that discard aliases already persisted for an object."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(CognitiveDelta(objects=[_object(aliases=["alphaalias"])]), "commit-object")

    store.memory_commit(
        CognitiveDelta(objects=[_object(expected_version=1, aliases=["betaalias"])]),
        "commit-versioned-alias",
    )

    with sqlite3.connect(path) as connection:
        alpha = connection.execute(
            "SELECT id FROM objects_fts WHERE objects_fts MATCH 'alphaalias'"
        ).fetchall()
        beta = connection.execute("SELECT id FROM objects_fts WHERE objects_fts MATCH 'betaalias'").fetchall()
    assert alpha == [("object-1",)]
    assert beta == [("object-1",)]


def test_exact_alias_collision_rejects_whole_delta_with_existing_candidate(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a duplicate Object silently losing its alias while its transaction succeeds."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object("existing", aliases=["IG"])]),
        "seed-existing",
    )

    with pytest.raises(ValueError) as raised:
        store.memory_commit(
            CognitiveDelta(
                objects=[_object("duplicate", aliases=[" ig "])],
                observations=[_observation("must-roll-back")],
            ),
            "duplicate-alias",
        )

    assert raised.value.normalized_alias == "ig"
    assert raised.value.existing_object_id == "existing"
    assert _count(path, "objects") == 1
    assert _count(path, "observations") == 0


def test_same_delta_alias_collision_rejects_every_proposed_object(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch two new Objects in one atomic delta claiming the same normalized identity."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)

    with pytest.raises(ValueError) as raised:
        store.memory_commit(
            CognitiveDelta(
                objects=[
                    _object("first", aliases=["shared name"]),
                    _object("second", aliases=[" Shared Name "]),
                ]
            ),
            "same-delta-collision",
        )

    # the identity normalizer preserves the internal single space
    assert raised.value.normalized_alias == "shared name"
    assert raised.value.existing_object_id == "first"
    assert _count(path, "objects") == 0


def test_normalize_alias_collapses_all_whitespace_and_casefolds() -> None:
    """Whitespace variants of an alias must fold to one identity (T7)."""
    assert normalize_alias("2026 EWC 英雄联盟项目") == "2026ewc英雄联盟项目"
    assert normalize_alias("2026 EWC英雄联盟项目") == "2026ewc英雄联盟项目"
    assert normalize_alias("  IG  ") == "ig"
    assert normalize_alias("Ig NaMe") == "igname"
    assert normalize_alias("") == ""
    assert normalize_alias("   \t\n ") == ""


def test_whitespace_variant_alias_is_a_distinct_identity(tmp_path: pytest.TempPathFactory) -> None:
    """Task 2.1: the identity normalizer preserves internal single spaces, so
    a whitespace variant of an alias is a DISTINCT identity — it never
    collides at the store level (legacy-style folding is read-only history
    and never blocks; the compiler surfaces name overlap as warnings)."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object("ewc-event", aliases=["2026 EWC 英雄联盟项目"])]),
        "seed-ewc",
    )

    store.memory_commit(
        CognitiveDelta(objects=[_object("ewc-twin", aliases=["2026 EWC英雄联盟项目"])]),
        "whitespace-twin",
    )

    assert _count(path, "objects") == 2
    with sqlite3.connect(path) as connection:
        forms = {
            row[0]
            for row in connection.execute(
                "SELECT normalized_alias FROM identity_aliases WHERE status = 'active'"
            ).fetchall()
        }
    assert forms == {"2026 ewc 英雄联盟项目", "2026 ewc英雄联盟项目"}


def test_pure_whitespace_alias_is_skipped_not_crashing(tmp_path: pytest.TempPathFactory) -> None:
    """All-whitespace aliases must be skipped defensively, never persisted."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object(aliases=["   ", "\t\n "])]),
        "commit-blank-alias",
    )

    assert _count(path, "objects") == 1
    with sqlite3.connect(path) as connection:
        aliases = connection.execute(
            "SELECT normalized_alias FROM identity_aliases"
        ).fetchall()
    assert all(row[0] for row in aliases)  # no empty identity rows
    assert ("",) not in aliases

    # a second object with an all-whitespace alias must not collide with the first
    store.memory_commit(
        CognitiveDelta(objects=[_object("object-2", aliases=[" 　 "])]),
        "commit-blank-alias-2",
    )
    assert _count(path, "objects") == 2


def test_legacy_space_form_alias_never_blocks_identity_claim(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A legacy ``object_aliases`` row is read-only history: a submission
    matching it claims its own identity alias instead of colliding (the
    compiler reports the legacy hit as an ambiguous_name_candidates warning)."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    _seed_legacy_alias_row(path, "legacy-t1", "T1 Stars 男团出道", "t1 stars 男团出道")

    store.memory_commit(
        CognitiveDelta(objects=[_object("twin", aliases=["T1 Stars 男团出道"])]),
        "legacy-hit",
    )

    assert _count(path, "objects") == 2
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT object_id FROM identity_aliases"
            " WHERE normalized_alias = 't1 stars 男团出道' AND status = 'active'"
        ).fetchone()
    assert row[0] == "twin"


def test_identity_owner_and_legacy_row_owned_differently_never_block(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The identity form ('t1 stars男团出道', no space) and the legacy
    space-preserving form are DIFFERENT forms: a claim of the space form
    conflicts with neither — legacy never blocks, and the identity index has
    no owner for it."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object("folded-owner", aliases=["T1 Stars男团出道"])]),
        "seed-folded",
    )
    _seed_legacy_alias_row(path, "legacy-owner", "T1 Stars 男团出道", "t1 stars 男团出道")

    store.memory_commit(
        CognitiveDelta(objects=[_object("twin", aliases=["T1 Stars 男团出道"])]),
        "dual-form-collision",
    )

    assert _count(path, "objects") == 3
    with sqlite3.connect(path) as connection:
        rows = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT object_id, normalized_alias FROM identity_aliases WHERE status = 'active'"
            ).fetchall()
        }
    assert ("folded-owner", "t1 stars男团出道") in rows
    assert ("twin", "t1 stars 男团出道") in rows


def test_duplicate_assertion_signature_reuses_assertion_and_adds_new_evidence(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch duplicate semantic assertions instead of accumulating their evidence."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            assertions=[_assertion("assertion-1")],
        ),
        "commit-1",
    )

    receipt = store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-2")],
            assertions=[_assertion("assertion-2", evidence_id="observation-2")],
        ),
        "commit-2",
    )

    assert receipt.assertion_ids == ["assertion-1"]
    assert _count(path, "assertions") == 1
    assert _count(path, "assertion_evidence") == 2


def test_same_signature_new_evidence_links_to_existing_assertion_and_warns(tmp_path) -> None:
    """Signature dedup keeps the existing assertion id, links new evidence, and warns.

    Task 2.4 contract point 5: when a proposal assertion's v1/v2 signature
    hits a stored assertion, the existing assertion id is retained, the valid
    new evidence IS linked to that id, and the commit surfaces a
    ``possible_cognition_conflict`` warning whose match basis names the
    signature collision — the dedup is announced, never silent.
    """
    path = tmp_path / "world.sqlite3"
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            assertions=[_assertion("assertion-1")],
        ),
        "commit-1",
    )
    store.memory_commit(
        CognitiveDelta(observations=[_observation("observation-2")]),
        "commit-observation-2",
    )
    delta = CognitionDeltaProposal.model_validate(
        {
            "assertions": [
                {
                    "subject": {"memory_id": "object-1"},
                    "predicate": "related_to",
                    "literal": "confirmed",
                    "epistemic_role": "fact",
                    "confidence": 0.8,
                    "evidence": [{"observation_id": "observation-2", "role": "supports"}],
                }
            ]
        }
    )

    receipt = ProposalCommitter(store).commit(delta, "commit-2")

    # the existing assertion id is retained end to end
    assert receipt.commit.assertion_ids == ["assertion-1"]
    assert _count(path, "assertions") == 1
    # the valid new evidence is linked to the EXISTING assertion, not dropped
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT assertion_id, observation_id FROM assertion_evidence ORDER BY observation_id"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("assertion-1", "observation-1"),
        ("assertion-1", "observation-2"),
    ]
    # and the dedup is announced with a warning naming the collision basis
    assert receipt.review_outcome is ReviewOutcome.COMMIT_WITH_WARNINGS
    dedup_issues = [
        issue
        for issue in receipt.review_issues
        if issue.match_basis and "assertion_signature_collision" in issue.match_basis
    ]
    assert len(dedup_issues) == 1
    issue = dedup_issues[0]
    assert issue.code is ReviewIssueCode.POSSIBLE_COGNITION_CONFLICT
    assert issue.severity == "warning"
    assert issue.item_kind == "assertion"
    assert issue.item_index == 0
    assert issue.durable_id == "assertion-1"
    assert issue.candidate_ids == ["assertion-1"]
    assert issue.actual_value["existing_assertion_id"] == "assertion-1"
    assert issue.actual_value["linked_evidence_ids"] == ["observation-2"]
    assert issue.actual_value["unreplaced"] == {}


def test_same_signature_confidence_or_supersedes_difference_is_not_silent(tmp_path) -> None:
    """Metadata the dedup does not replace is named in the warning's actual_value.

    When a signature-colliding proposal carries a different confidence, event
    time precision or supersedes declaration, the stored row keeps its old
    values, the proposed values are recorded only in the audit and the
    warning's ``actual_value["unreplaced"]`` — never silently claimed as saved.
    """
    path = tmp_path / "world.sqlite3"
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1"), _observation("observation-2")],
            assertions=[
                _assertion("assertion-0", literal="other"),
                _assertion("assertion-1"),
            ],
        ),
        "commit-1",
    )
    delta = CognitionDeltaProposal.model_validate(
        {
            "assertions": [
                {
                    "subject": {"memory_id": "object-1"},
                    "predicate": "related_to",
                    "literal": "confirmed",
                    "epistemic_role": "fact",
                    "confidence": 0.95,
                    "event_time_precision": "period",
                    "supersedes_id": "assertion-0",
                    "supersede_reason": "more recent report",
                    "evidence": [{"observation_id": "observation-2", "role": "supports"}],
                }
            ]
        }
    )

    receipt = ProposalCommitter(store).commit(delta, "commit-2")

    dedup_issues = [
        issue
        for issue in receipt.review_issues
        if issue.match_basis and "assertion_signature_collision" in issue.match_basis
    ]
    assert len(dedup_issues) == 1
    issue = dedup_issues[0]
    assert issue.actual_value["unreplaced"] == {
        "confidence": {"stored": 0.8, "proposed": 0.95},
        "event_time_precision": {"stored": "unknown", "proposed": "period"},
        "supersedes_id": {"stored": None, "proposed": "assertion-0"},
        "supersede_reason": {"stored": None, "proposed": "more recent report"},
    }
    assert _count(path, "assertions") == 2
    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT confidence, event_time_precision, supersedes_id, supersede_reason,"
            " answers_inquiry_id FROM assertions WHERE id = 'assertion-1'"
        ).fetchone()
        audited = connection.execute(
            "SELECT delta_json FROM world_audit WHERE commit_id = 'commit-2'"
        ).fetchone()[0]
    # the stored row kept the OLD metadata — nothing was replaced
    assert tuple(stored) == (0.8, "unknown", None, None, None)
    # the proposed metadata is visible in the audit, not silently dropped
    proposed = CognitiveDelta.model_validate_json(str(audited)).assertions[0]
    assert proposed.confidence == 0.95
    assert proposed.event_time_precision == "period"
    assert proposed.supersedes_id == "assertion-0"


def test_resolving_inquiry_and_writing_answer_is_one_transaction(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a resolved inquiry when its answering assertion is not committed too."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            inquiries=[InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap")],
        ),
        "commit-inquiry",
    )

    receipt = store.memory_commit(
        CognitiveDelta(
            assertions=[_assertion().model_copy(update={"answers_inquiry_id": "inquiry-1"})],
            resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=1)],
        ),
        "commit-answer",
    )

    assert receipt.assertion_ids == ["assertion-1"]
    assert receipt.resolved_inquiry_ids == ["inquiry-1"]
    with sqlite3.connect(path) as connection:
        state = connection.execute("SELECT status, version FROM inquiries WHERE id = 'inquiry-1'").fetchone()
    assert state == ("resolved", 2)


def test_resolution_sets_resolved_at(tmp_path: pytest.TempPathFactory) -> None:
    """Pin that resolving an inquiry records when it was resolved."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="X")],
            inquiries=[InquiryInput(id="inq-1", subject_id="obj-1", prompt="p", rationale="r")],
        ),
        "seed-1",
    )
    store.memory_commit(
        CognitiveDelta(
            assertions=[
                AssertionInput(
                    id="a-1",
                    subject_id="obj-1",
                    predicate="has_result",
                    literal="2:0",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    answers_inquiry_id="inq-1",
                )
            ],
            resolve_inquiries=[InquiryResolution(id="inq-1", expected_version=1)],
        ),
        "c-1",
    )
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT status, resolved_at FROM inquiries WHERE id='inq-1'"
        ).fetchone()
    assert row["status"] == "resolved"
    assert row["resolved_at"] is not None


def test_resolution_allows_dormant_inquiry_with_substantive_answer(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Dormant is deprioritized unresolved work, so a valid answer may close it."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="X")],
            inquiries=[InquiryInput(id="inq-1", subject_id="obj-1", prompt="p", rationale="r")],
        ),
        "seed-1",
    )
    with store.write_connection() as connection:
        connection.execute("UPDATE inquiries SET status = 'dormant' WHERE id = 'inq-1'")

    receipt = store.memory_commit(
        CognitiveDelta(
            assertions=[
                AssertionInput(
                    id="a-1",
                    subject_id="obj-1",
                    predicate="has_result",
                    literal="2:0",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    answers_inquiry_id="inq-1",
                )
            ],
            resolve_inquiries=[InquiryResolution(id="inq-1", expected_version=1)],
        ),
        "c-1",
    )

    assert receipt.resolved_inquiry_ids == ["inq-1"]
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT status, version FROM inquiries WHERE id = 'inq-1'"
        ).fetchone()
    assert (row["status"], row["version"]) == ("resolved", 2)


def test_inquiry_resolution_rejects_without_real_answer_link(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Audit F2b (the inquiry-7587bb87 shape): same-subject substantive
    assertions that do not declare answers_inquiry_id can never close an
    inquiry — the real answer link is required, not mere subject presence.
    An assertion that claims a DIFFERENT inquiry is equally insufficient."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object(), _object("object-2")],
            observations=[_observation("observation-1")],
            inquiries=[
                InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap"),
                InquiryInput(id="inquiry-2", subject_id="object-2", prompt="What?", rationale="Gap"),
            ],
        ),
        "commit-inquiries",
    )

    with pytest.raises(ValueError, match="answering assertion"):
        store.memory_commit(
            CognitiveDelta(
                assertions=[_assertion().model_copy(update={"answers_inquiry_id": "inquiry-2"})],
                resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=1)],
            ),
            "commit-mismatched-answer",
        )

    assert _count(path, "assertions") == 0
    with sqlite3.connect(path) as connection:
        state = connection.execute("SELECT status FROM inquiries WHERE id = 'inquiry-1'").fetchone()
    assert state == ("open",)  # the inquiry stays open: no partial resolution


def test_two_inquiries_on_one_subject_both_resolve(tmp_path: pytest.TempPathFactory) -> None:
    """d2 regression (audit F1): resolving TWO inquiries anchored on one object
    in a single delta commits. Each resolution is scoped to its own answering
    assertion (answers_inquiry_id == that inquiry), never to every
    same-subject answer — the d2 pair (中单辅助轮换频率变化→7587,
    HLE中单综合表现争议→b881 on object-4d9cfeef) must be structurally valid."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            inquiries=[
                InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap"),
                InquiryInput(id="inquiry-2", subject_id="object-1", prompt="What?", rationale="Gap"),
            ],
        ),
        "commit-inquiries",
    )

    receipt = store.memory_commit(
        CognitiveDelta(
            assertions=[
                _assertion().model_copy(update={"answers_inquiry_id": "inquiry-1"}),
                _assertion("assertion-2", literal="later-confirmed").model_copy(
                    update={"answers_inquiry_id": "inquiry-2"}
                ),
            ],
            resolve_inquiries=[
                InquiryResolution(id="inquiry-1", expected_version=1),
                InquiryResolution(id="inquiry-2", expected_version=1),
            ],
        ),
        "commit-both-answers",
    )

    assert sorted(receipt.resolved_inquiry_ids) == ["inquiry-1", "inquiry-2"]
    assert sorted(receipt.assertion_ids) == ["assertion-1", "assertion-2"]
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT id, answers_inquiry_id FROM assertions ORDER BY id").fetchall()
    assert rows == [("assertion-1", "inquiry-1"), ("assertion-2", "inquiry-2")]
    states = {
        str(row[0]): str(row[1])
        for row in connection.execute("SELECT id, status FROM inquiries ORDER BY id").fetchall()
    }
    assert states == {"inquiry-1": "resolved", "inquiry-2": "resolved"}


def test_inquiry_resolution_accepts_matching_answer_declaration(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Pin that a resolution whose answering assertion claims the same inquiry commits."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            inquiries=[InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap")],
        ),
        "commit-inquiry",
    )

    receipt = store.memory_commit(
        CognitiveDelta(
            assertions=[_assertion().model_copy(update={"answers_inquiry_id": "inquiry-1"})],
            resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=1)],
        ),
        "commit-answer",
    )

    assert receipt.resolved_inquiry_ids == ["inquiry-1"]
    with sqlite3.connect(path) as connection:
        state = connection.execute("SELECT status FROM inquiries WHERE id = 'inquiry-1'").fetchone()
    assert state == ("resolved",)


def test_assertion_answers_inquiry_id_must_reference_existing_inquiry(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch answer edges that claim an inquiry that does not exist."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object()], observations=[_observation("observation-1")]),
        "commit-object",
    )

    with pytest.raises(ValueError, match="nonexistent inquiry"):
        store.memory_commit(
            CognitiveDelta(
                assertions=[_assertion().model_copy(update={"answers_inquiry_id": "ghost-inquiry"})],
            ),
            "commit-ghost-answer",
        )

    assert _count(path, "assertions") == 0


def test_empty_inquiry_resolution_rolls_back_the_whole_delta(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch inquiry resolution without any declarative answering assertion."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            inquiries=[InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap")],
        ),
        "commit-inquiry",
    )

    with pytest.raises(ValueError, match="answering assertion"):
        store.memory_commit(
            CognitiveDelta(
                observations=[_observation("observation-1")],
                resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=1)],
            ),
            "empty-resolution",
        )

    assert _count(path, "observations") == 0
    assert _count(path, "world_audit") == 1
    with sqlite3.connect(path) as connection:
        state = connection.execute("SELECT status, version FROM inquiries").fetchone()
    assert state == ("open", 1)


def test_inquiry_resolution_rejects_answer_for_an_unrelated_subject(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch an unrelated declarative assertion being used to close an inquiry."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object(), _object("object-2")],
            observations=[_observation("observation-1")],
            inquiries=[InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap")],
        ),
        "commit-inquiry",
    )

    with pytest.raises(ValueError, match="answering assertion"):
        store.memory_commit(
            CognitiveDelta(
                assertions=[_assertion().model_copy(update={"subject_id": "object-2"})],
                resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=1)],
            ),
            "unrelated-resolution",
        )

    assert _count(path, "assertions") == 0
    assert _count(path, "world_audit") == 1


def test_validate_delta_aggregates_all_offending_ids(tmp_path: pytest.TempPathFactory) -> None:
    """Catch one-round repair blocking: every bad id must appear in one message."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            inquiries=[InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap")],
        ),
        "commit-base",
    )
    delta = CognitiveDelta(
        objects=[_object("object-2", expected_version=0)],
        observations=[_observation("observation-2")],
        assertions=[
            AssertionInput(
                id="assertion-bad",
                subject_id="object-1",
                predicate="related_to",
                literal="bad",
                epistemic_role=EpistemicRole.FACT,
                confidence=0.8,
                evidence=[
                    EvidenceInput(observation_id="ghost-evidence-a", role="supports"),
                    EvidenceInput(observation_id="ghost-evidence-b", role="supports"),
                ],
                supersedes_id="ghost-assertion",
            )
        ],
        observation_links=[
            world_contracts.ObservationLinkInput(
                target_kind="object",
                target_id="object-1",
                observation_id="ghost-link-observation",
                role="candidate",
            )
        ],
        resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=0)],
    )

    with pytest.raises(ValueError) as raised:
        store.memory_commit(delta, "commit-aggregated")

    message = str(raised.value)
    for identifier in (
        "ghost-evidence-a",
        "ghost-evidence-b",
        "ghost-link-observation",
        "ghost-assertion",
        "object-2",
        "inquiry-1",
    ):
        assert identifier in message, f"missing {identifier} in {message!r}"
    assert _count(path, "assertions") == 0  # still one atomic rollback, one raise


def test_validate_delta_error_message_truncated_to_sane_length(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch an aggregated error message exploding past the ledger's 2000-char cap."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object()], observations=[_observation("observation-1")]),
        "commit-base",
    )
    evidence = [
        EvidenceInput(observation_id=f"ghost-evidence-{index:04d}", role="supports")
        for index in range(120)
    ]

    with pytest.raises(ValueError) as raised:
        store.memory_commit(
            CognitiveDelta(assertions=[_assertion().model_copy(update={"evidence": evidence})]),
            "commit-truncated",
        )

    assert len(str(raised.value)) <= 2000
    assert "ghost-evidence-0000" in str(raised.value)


def test_stale_object_or_inquiry_version_rejects_whole_delta(tmp_path: pytest.TempPathFactory) -> None:
    """Catch optimistic-concurrency conflicts that permit partial writes."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            inquiries=[InquiryInput(id="inquiry-1", subject_id="object-1", prompt="Why?", rationale="Gap")],
        ),
        "commit-base",
    )

    with pytest.raises(ValueError, match="stale"):
        store.memory_commit(
            CognitiveDelta(
                objects=[_object(expected_version=0)],
                observations=[_observation("observation-1")],
                resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=0)],
            ),
            "commit-stale",
        )

    assert _count(path, "observations") == 0
    assert _count(path, "world_audit") == 1


def test_four_world_times_remain_distinct(tmp_path: pytest.TempPathFactory) -> None:
    """Catch time-field collapse across event, publication, observation, and commit time."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    event_start = NOW - timedelta(days=4)
    source_published = NOW - timedelta(days=3)
    observed = NOW - timedelta(days=2)
    assertion_event = NOW - timedelta(days=1)
    receipt = store.memory_commit(
        CognitiveDelta(
            objects=[_object(event_time_start=event_start)],
            observations=[
                ObservationInput(
                    id="observation-1",
                    source_uri="https://example.test/observation-1",
                    source_kind="web",
                    depth=ObservationDepth.CONTENT,
                    source_published_at=source_published,
                    observed_at=observed,
                )
            ],
            assertions=[_assertion().model_copy(update={"event_time_start": assertion_event})],
        ),
        "commit-times",
    )

    with sqlite3.connect(path) as connection:
        object_time = connection.execute("SELECT event_time_start FROM objects").fetchone()[0]
        publication_time, observation_time = connection.execute(
            "SELECT source_published_at, observed_at FROM observations"
        ).fetchone()
        assertion_time = connection.execute("SELECT event_time_start FROM assertions").fetchone()[0]
    assert object_time == event_start.isoformat()
    assert publication_time == source_published.isoformat()
    assert observation_time == observed.isoformat()
    assert assertion_time == assertion_event.isoformat()
    assert receipt.committed_at > observed


def test_demote_stale_inquiries(tmp_path):
    from datetime import timedelta
    store = WorldStore(tmp_path / "world.sqlite3")
    old = datetime.now(UTC) - timedelta(days=10)
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="X")],
            inquiries=[
                InquiryInput(id="inq-old", subject_id="obj-1", prompt="old q", rationale="r",
                             created_at=old, last_attempted_at=old),
                InquiryInput(id="inq-fresh", subject_id="obj-1", prompt="fresh q", rationale="r",
                             created_at=datetime.now(UTC)),
                InquiryInput(id="inq-stateful", subject_id="obj-1", prompt="watch q", rationale="r",
                             kind="stateful", created_at=old, last_attempted_at=old),
            ],
        ),
        "seed-1",
    )
    count = store.demote_stale_inquiries(datetime.now(UTC), max_age_days=7)
    assert count == 1  # only inq-old
    with store.read_connection() as connection:
        statuses = dict(connection.execute("SELECT id, status FROM inquiries").fetchall())
    assert statuses["inq-old"] == "dormant"
    assert statuses["inq-fresh"] == "open"
    assert statuses["inq-stateful"] == "open"


def test_write_and_read_observation_body(tmp_path: pytest.TempPathFactory) -> None:
    """Catch full-text body writes that cannot round-trip through observation_bodies."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )
    store.write_observation_body("hupu:obs-1", "article", '{"paras": ["a", "b"]}', 12)
    body = store.read_observation_body("hupu:obs-1")
    assert body is not None
    assert body["content_type"] == "article"
    assert "a" in body["body_json"]
    # overwrite is idempotent
    store.write_observation_body("hupu:obs-1", "article", '{"paras": ["c"]}', 8)
    assert "c" in store.read_observation_body("hupu:obs-1")["body_json"]
    # unknown id → None
    assert store.read_observation_body("nope") is None


def test_upgrade_observation_depth_promotes_seen_to_content(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch full-body reads leaving the observation at SEEN depth (spec D1 unity)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(observations=[_observation("observation-1", depth=ObservationDepth.SEEN)]),
        "seed-1",
    )

    store.upgrade_observation_depth("observation-1")

    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT depth FROM observations WHERE id = 'observation-1'"
        ).fetchone()
    assert row == ("content",)


def test_upgrade_observation_depth_is_monotone_and_idempotent(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Pin that upgrades only ever raise depth and unknown ids are silent no-ops."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                _observation("seen-1", depth=ObservationDepth.SEEN),
                _observation("content-1", depth=ObservationDepth.CONTENT),
                _observation("discussion-1", depth=ObservationDepth.DISCUSSION),
            ]
        ),
        "seed-1",
    )

    store.upgrade_observation_depth("seen-1")
    store.upgrade_observation_depth("seen-1")  # re-upgrade is a no-op
    store.upgrade_observation_depth("content-1")  # already CONTENT: untouched
    store.upgrade_observation_depth("discussion-1")  # never downgrades
    store.upgrade_observation_depth("missing-1")  # missing row: silent no-op

    with sqlite3.connect(store.path) as connection:
        depths = dict(connection.execute("SELECT id, depth FROM observations").fetchall())
    assert depths == {"seen-1": "content", "content-1": "content", "discussion-1": "discussion"}


def test_recommit_at_seen_after_upgrade_keeps_content_and_supports_evidence(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a re-scanned observation (SEEN re-commit) rejecting the batch or dragging depth back to SEEN."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1", depth=ObservationDepth.SEEN)],
        ),
        "seed-1",
    )
    store.upgrade_observation_depth("observation-1")

    receipt = store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-1", depth=ObservationDepth.SEEN)],
            assertions=[_assertion()],
        ),
        "rescan-1",
    )

    # INSERT OR IGNORE cannot overwrite the depth, and the monotone identity
    # guard treats the shallower re-commit as the same observation: no raise,
    # stored CONTENT kept, and the assertion's supports evidence is judged
    # against the stored CONTENT depth, not the proposed SEEN one.
    assert receipt.assertion_ids == ["assertion-1"]
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT depth FROM observations WHERE id = 'observation-1'"
        ).fetchone()
    assert row == ("content",)
    assert _count(path, "world_audit") == 2


def test_recommit_does_not_refresh_observed_at(tmp_path: pytest.TempPathFactory) -> None:
    """Re-reading a source keeps the first observation time (spec §7)."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    first = datetime(2026, 8, 1, 12, tzinfo=UTC)
    later = datetime(2026, 8, 15, 12, tzinfo=UTC)
    store.memory_commit(
        CognitiveDelta(observations=[_observation("observation-1", observed_at=first)]),
        "seed-1",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[
                _observation(
                    "observation-1",
                    observed_at=later,
                    metadata={"refreshed": True},
                )
            ]
        ),
        "rescan-1",
    )
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT observed_at, metadata_json FROM observations WHERE id = 'observation-1'"
        ).fetchone()
    # The stored observed_at is pinned to the first observation; the audit
    # still records the rescan, and non-time refresh behavior is untouched.
    assert row[0] == first.isoformat()
    assert json.loads(row[1])["refreshed"] is True
    assert _count(path, "world_audit") == 2


def test_depth_upgrade_recommit_keeps_first_observed_at(tmp_path: pytest.TempPathFactory) -> None:
    """The seen→content hydrate re-write never refreshes observed_at either."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    first = datetime(2026, 8, 1, 12, tzinfo=UTC)
    later = datetime(2026, 8, 15, 12, tzinfo=UTC)
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-1", depth=ObservationDepth.SEEN, observed_at=first)]
        ),
        "seed-1",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[_observation("observation-1", depth=ObservationDepth.CONTENT, observed_at=later)]
        ),
        "hydrate-1",
    )
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT observed_at, depth FROM observations WHERE id = 'observation-1'"
        ).fetchone()
    assert row == (first.isoformat(), "content")


def test_recommit_at_different_uri_after_upgrade_still_raises(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch an upgraded observation row being hijacked by a different source under the same id."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(observations=[_observation("observation-1", depth=ObservationDepth.SEEN)]),
        "seed-1",
    )
    store.upgrade_observation_depth("observation-1")

    with pytest.raises(ValueError, match="different observation"):
        store.memory_commit(
            CognitiveDelta(
                observations=[
                    ObservationInput(
                        id="observation-1",
                        source_uri="https://example.test/other",
                        source_kind="web",
                        depth=ObservationDepth.SEEN,
                        observed_at=NOW,
                    )
                ]
            ),
            "hijack-1",
        )

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT depth FROM observations WHERE id = 'observation-1'"
        ).fetchone()
    assert row == ("content",)


def test_committer_touches_attempted_at_on_deepens(tmp_path):
    from datetime import timedelta
    store = WorldStore(tmp_path / "world.sqlite3")
    old = datetime.now(UTC) - timedelta(days=3)
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="Bin")],
            inquiries=[InquiryInput(id="inq-a", subject_id="obj-1", prompt="48bin 起源", rationale="r",
                                    created_at=old, last_attempted_at=old)],
        ),
        "seed-1",
    )
    committer = ProposalCommitter(store, inquiry_similarity_threshold=0.2)
    proposal = CognitionDeltaProposal(
        new_inquiries=[NewInquiryProposal(local_ref="nq1", subject=GraphRef(memory_id="obj-1"),
                                          prompt="48bin 的准确起源", rationale="r",
                                          deepens_inquiry_id="inq-a")]
    )
    committer.commit(proposal, "c-1")
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT attempt_count, last_attempted_at FROM inquiries WHERE id='inq-a'"
        ).fetchone()
    assert row["attempt_count"] == 1
    assert row["last_attempted_at"] > old.isoformat()


def test_assertion_event_time_precision_and_supersede_reason_round_trip(tmp_path) -> None:
    """v9 fields persist on the assertion row without entering its signature."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion().model_copy(
                    update={
                        "event_time_precision": "interval",
                        "supersede_reason": "replaced by a later finding",
                    }
                )
            ],
        ),
        "precision-commit",
    )
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT event_time_precision, supersede_reason FROM assertions WHERE id = 'assertion-1'"
        ).fetchone()
        assert row == ("interval", "replaced by a later finding")


def test_object_event_time_precision_round_trip_insert_and_update(tmp_path) -> None:
    """Object precision persists on INSERT, is mutable via expected_version, and
    a mutation without expected_version is rejected."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object().model_copy(update={"event_time_precision": "period"})]),
        "object-precision-1",
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT event_time_precision, version FROM objects WHERE id = 'object-1'"
        ).fetchone() == ("period", 1)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object(expected_version=1).model_copy(update={"event_time_precision": "exact"})]
        ),
        "object-precision-2",
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT event_time_precision, version FROM objects WHERE id = 'object-1'"
        ).fetchone() == ("exact", 2)
    with pytest.raises(ValueError, match="object mutation requires expected version"):
        store.memory_commit(
            CognitiveDelta(objects=[_object().model_copy(update={"event_time_precision": "interval"})]),
            "object-precision-3",
        )


def test_type_key_round_trips_and_participates_in_replay_comparison(tmp_path) -> None:
    """type_key persists on INSERT and UPDATE, gates the no-change comparison,
    and a replayed commit id with a different type_key is a replay conflict."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    with_type_key = _object().model_copy(update={"type_key": "organization"})
    store.memory_commit(CognitiveDelta(objects=[with_type_key]), "type-key-1")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT kind, type_key, version FROM objects WHERE id = 'object-1'"
        ).fetchone() == ("entity", "organization", 1)
    # the identical object (same type_key) commits again without a version
    store.memory_commit(CognitiveDelta(objects=[with_type_key]), "type-key-2")
    # a same-id mutation without expected_version is rejected when type_key differs
    with pytest.raises(ValueError, match="object mutation requires expected version"):
        store.memory_commit(
            CognitiveDelta(objects=[with_type_key.model_copy(update={"type_key": "place"})]),
            "type-key-3",
        )
    # a versioned update rewrites type_key
    store.memory_commit(
        CognitiveDelta(
            objects=[with_type_key.model_copy(update={"type_key": "place", "expected_version": 1})]
        ),
        "type-key-4",
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT type_key, version FROM objects WHERE id = 'object-1'"
        ).fetchone() == ("place", 2)
    # replay: the identical delta is idempotent, a different type_key conflicts
    replay = store.memory_commit(CognitiveDelta(objects=[with_type_key]), "type-key-1")
    assert replay.replayed is True
    with pytest.raises(ValueError, match="different cognitive delta"):
        store.memory_commit(
            CognitiveDelta(objects=[with_type_key.model_copy(update={"type_key": "rule"})]),
            "type-key-1",
        )


def test_inquiry_close_reopen_lattice(tmp_path) -> None:
    """Closed inquiries store a durable reason; the lattice is
    open→dormant→closed→open(reopen) with a version bump per transition."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            inquiries=[
                InquiryInput(
                    id="inq-lattice", subject_id="object-1", prompt="Why?", rationale="r"
                )
            ],
        ),
        "inq-lattice-seed",
    )
    store.close_inquiry("inq-lattice", "no longer relevant")
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status, version, closed_reason FROM inquiries WHERE id = 'inq-lattice'"
        ).fetchone()
        assert row == ("closed", 2, "no longer relevant")
    with pytest.raises(ValueError, match="not open or dormant"):
        store.close_inquiry("inq-lattice", "again")
    with pytest.raises(ValueError, match="must be non-empty"):
        store.close_inquiry("inq-lattice", "   ")
    store.reopen_inquiry("inq-lattice")
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status, version, closed_reason FROM inquiries WHERE id = 'inq-lattice'"
        ).fetchone()
        assert row == ("open", 3, None)
    with pytest.raises(ValueError, match="not closed"):
        store.reopen_inquiry("inq-lattice")
    with pytest.raises(ValueError, match="does not exist"):
        store.close_inquiry("inq-missing", "why")
    with pytest.raises(ValueError, match="does not exist"):
        store.reopen_inquiry("inq-missing")


def test_old_format_commit_replays_without_precision_fields(tmp_path) -> None:
    """v9 fields default on old-format stored deltas, so idempotent replay
    still matches; a genuinely different precision is a replay conflict."""
    store = WorldStore(tmp_path / "world.sqlite3")
    delta = CognitiveDelta(
        objects=[_object()],
        observations=[_observation("observation-1")],
        assertions=[_assertion()],
    )
    store.memory_commit(delta, "old-format")
    with store.write_connection() as connection:
        payload = json.loads(
            connection.execute(
                "SELECT delta_json FROM world_audit WHERE commit_id = 'old-format'"
            ).fetchone()[0]
        )
        # modern commits carry the v9 fields; strip them to simulate an
        # old-format commit written before the migration
        assert "event_time_precision" in payload["objects"][0]
        del payload["objects"][0]["event_time_precision"]
        for assertion in payload["assertions"]:
            del assertion["event_time_precision"]
            del assertion["supersede_reason"]
        connection.execute(
            "UPDATE world_audit SET delta_json = ? WHERE commit_id = 'old-format'",
            (json.dumps(payload, indent=2, sort_keys=True),),
        )

    replay = store.memory_commit(delta, "old-format")

    assert replay.replayed is True
    with pytest.raises(ValueError, match="different cognitive delta"):
        store.memory_commit(
            CognitiveDelta(
                objects=[_object()],
                observations=[_observation("observation-1")],
                assertions=[
                    _assertion().model_copy(update={"event_time_precision": "interval"})
                ],
            ),
            "old-format",
        )


def test_empty_qualifiers_keep_frozen_v9_signature() -> None:
    """None or empty qualifiers must hash byte-identically to the pre-v10 signature.

    The v10 signature contract: an absent qualifier map must not change any
    stored assertion identity, so legacy rows keep matching idempotent replay
    and never split into duplicates. The constant is the sha256 of the v1
    six-field payload, computed before the v2 path existed.
    """
    assert _assertion_signature(_assertion(qualifiers=None)) == FROZEN_V9_SIGNATURE
    assert _assertion_signature(_assertion(qualifiers={})) == FROZEN_V9_SIGNATURE


def test_nonempty_qualifiers_create_role_sensitive_v2_signature() -> None:
    """The same relation with different roles must split into distinct identities."""
    home = _assertion("assertion-home", qualifiers={"role": "home"})
    away = _assertion("assertion-away", qualifiers={"role": "away"})
    assert _assertion_signature(home) != _assertion_signature(away)


def test_qualifier_signature_ignores_dict_order_and_is_deterministic() -> None:
    """Lexicographic key order must make the v2 signature order-free."""
    first = _assertion(qualifiers={"role": "home", "community": "cn"})
    second = _assertion(qualifiers={"community": "cn", "role": "home"})
    assert _assertion_signature(first) == _assertion_signature(second)
    assert _assertion_signature(first) == _assertion_signature(first)


def test_identity_signature_ignores_confidence_evidence_and_supersedes() -> None:
    """Provenance and correction metadata must never split an identity.

    confidence, evidence links and supersedes_id are visibility/correction
    inputs, not identity — the dedup visibility layer owns them. Two
    assertions differing only in these fields must share one signature so the
    second commit attaches to the stored row instead of minting a twin.
    """
    base = _assertion()
    assert _assertion_signature(base) == _assertion_signature(
        base.model_copy(update={"confidence": 0.95})
    )
    assert _assertion_signature(base) == _assertion_signature(
        base.model_copy(
            update={
                "evidence": [
                    EvidenceInput(observation_id="observation-9", role="context")
                ]
            }
        )
    )
    assert _assertion_signature(base) == _assertion_signature(
        base.model_copy(update={"supersedes_id": "assertion-old"})
    )


def test_assertion_qualifiers_persist_in_qualifiers_json_column(tmp_path) -> None:
    """Non-empty qualifiers must persist in the v10 column; empty stays NULL."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            assertions=[
                _assertion("assertion-1", qualifiers={"role": "home", "community": "cn"}),
                _assertion("assertion-2", literal="plain", qualifiers={}),
            ],
        ),
        "qualifiers-commit",
    )
    with sqlite3.connect(path) as connection:
        rows = dict(
            connection.execute("SELECT id, qualifiers_json FROM assertions ORDER BY id").fetchall()
        )
    assert json.loads(rows["assertion-1"]) == {"role": "home", "community": "cn"}
    assert rows["assertion-2"] is None  # v1-equivalent rows stay NULL like legacy rows


# ── Task 1.5: identity alias current index and audited corrections ──────────


def test_canonical_name_is_not_inserted_into_identity_aliases(tmp_path) -> None:
    """Only ObjectInput.aliases initialize the identity index; canonical never does."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object("person-one", aliases=["Alex"])]),
        "commit-identity-init",
    )

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT raw_alias, normalized_alias, status, added_commit_id, removed_commit_id"
            " FROM identity_aliases"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("Alex", "alex", "active", "commit-identity-init", None)
    ]


def test_two_objects_may_share_canonical_name(tmp_path) -> None:
    """canonical_name is a display name, not a world-scope identity alias."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="one",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Shared Name",
                    aliases=["Alpha"],
                ),
                ObjectInput(
                    id="two",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Shared Name",
                    aliases=["Beta"],
                ),
            ]
        ),
        "commit-shared-canonical",
    )

    assert _count(path, "objects") == 2
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT object_id, normalized_alias FROM identity_aliases ORDER BY normalized_alias"
        ).fetchall()
    assert [tuple(row) for row in rows] == [("one", "alpha"), ("two", "beta")]


def test_initial_identity_alias_is_normalized_without_removing_internal_spaces(
    tmp_path,
) -> None:
    """The shared normalizer keeps internal whitespace; only runs collapse."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object("ewc-event", aliases=["2026 EWC  英雄联盟项目"])]),
        "commit-init-identity",
    )

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT raw_alias, normalized_alias, status, added_commit_id, removed_commit_id"
            " FROM identity_aliases"
        ).fetchone()
    assert tuple(row) == (
        "2026 EWC  英雄联盟项目",
        "2026 ewc 英雄联盟项目",
        "active",
        "commit-init-identity",
        None,
    )


def test_add_remove_demote_are_present_in_delta_audit_and_same_transaction(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Alias operations travel in the delta audit and apply in the same commit."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object(aliases=["Alpha", "Bravo", "Charlie"])]),
        "seed-alias-lifecycle",
    )
    operations = [
        AliasOperation(
            object_id="object-1",
            raw_alias="Delta",
            normalized_alias="delta",
            action=AliasAction.ADD,
        ),
        AliasOperation(
            object_id="object-1",
            raw_alias="Alpha",
            normalized_alias="alpha",
            action=AliasAction.REMOVE,
        ),
        AliasOperation(
            object_id="object-1",
            raw_alias="Bravo",
            normalized_alias="bravo",
            action=AliasAction.DEMOTE,
        ),
    ]

    receipt = store.memory_commit(
        CognitiveDelta(alias_operations=operations), "commit-alias-lifecycle"
    )

    assert not receipt.replayed
    assert _count(path, "commit_receipts") == 2
    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT delta_json FROM world_audit WHERE commit_id = 'commit-alias-lifecycle'"
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT raw_alias, status, added_commit_id, removed_commit_id"
            " FROM identity_aliases WHERE object_id = 'object-1' ORDER BY normalized_alias"
        ).fetchall()
    assert CognitiveDelta.model_validate_json(str(stored)).alias_operations == operations
    assert [tuple(row) for row in rows] == [
        ("Alpha", "removed", "seed-alias-lifecycle", "commit-alias-lifecycle"),
        ("Bravo", "removed", "seed-alias-lifecycle", "commit-alias-lifecycle"),
        ("Charlie", "active", "seed-alias-lifecycle", None),
        ("Delta", "active", "commit-alias-lifecycle", None),
    ]


def test_alias_operation_rollback_leaves_alias_state_and_world_audit_unchanged(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A failed commit must not leave identity_aliases or audit traces behind."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object(aliases=["Alpha"])]),
        "seed-rollback",
    )
    with sqlite3.connect(path) as connection:
        before = connection.execute(
            "SELECT raw_alias, status, added_commit_id, removed_commit_id FROM identity_aliases"
        ).fetchall()
        before_audit = connection.execute("SELECT COUNT(*) FROM world_audit").fetchone()[0]

    with pytest.raises(ValueError, match="evidence observation does not exist"):
        store.memory_commit(
            CognitiveDelta(
                alias_operations=[
                    AliasOperation(
                        object_id="object-1",
                        raw_alias="Alpha",
                        normalized_alias="alpha",
                        action=AliasAction.REMOVE,
                    )
                ],
                assertions=[_assertion(evidence_id="missing-observation")],
            ),
            "commit-rollback",
        )

    with sqlite3.connect(path) as connection:
        after = connection.execute(
            "SELECT raw_alias, status, added_commit_id, removed_commit_id FROM identity_aliases"
        ).fetchall()
        after_audit = connection.execute("SELECT COUNT(*) FROM world_audit").fetchone()[0]
    assert after == before
    assert after_audit == before_audit


def test_readd_after_remove_restores_active_owner_and_replay_is_idempotent(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Add-on-removed-row is a deterministic UPSERT; replay never re-transitions."""
    path = str(tmp_path / "world.sqlite3")
    store = WorldStore(path)
    store.memory_commit(
        CognitiveDelta(objects=[_object(aliases=["Alpha"])]),
        "seed-readd",
    )
    store.memory_commit(
        CognitiveDelta(
            alias_operations=[
                AliasOperation(
                    object_id="object-1",
                    raw_alias="Alpha",
                    normalized_alias="alpha",
                    action=AliasAction.REMOVE,
                )
            ]
        ),
        "remove-readd",
    )
    store.memory_commit(
        CognitiveDelta(
            alias_operations=[
                AliasOperation(
                    object_id="object-1",
                    raw_alias="Alpha",
                    normalized_alias="alpha",
                    action=AliasAction.ADD,
                )
            ]
        ),
        "readd-readd",
    )
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status, added_commit_id, removed_commit_id FROM identity_aliases"
            " WHERE object_id = 'object-1' AND normalized_alias = 'alpha'"
        ).fetchone()
    assert tuple(row) == ("active", "readd-readd", None)

    replayed = store.memory_commit(
        CognitiveDelta(
            alias_operations=[
                AliasOperation(
                    object_id="object-1",
                    raw_alias="Alpha",
                    normalized_alias="alpha",
                    action=AliasAction.ADD,
                )
            ]
        ),
        "readd-readd",
    )
    assert replayed.replayed
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status, added_commit_id, removed_commit_id FROM identity_aliases"
            " WHERE object_id = 'object-1' AND normalized_alias = 'alpha'"
        ).fetchone()
    assert tuple(row) == ("active", "readd-readd", None)

    with pytest.raises(CommitReplayConflict):
        store.memory_commit(
            CognitiveDelta(
                alias_operations=[
                    AliasOperation(
                        object_id="object-1",
                        raw_alias="Alpha",
                        normalized_alias="alpha",
                        action=AliasAction.REMOVE,
                    )
                ]
            ),
            "readd-readd",
        )


# ── Wave B: same-delta inquiry create+answer+resolve and deepen (v18) ─────

def test_same_delta_create_answer_and_resolve_inquiry(tmp_path) -> None:
    """Wave B B4-4 store shape: one delta may create the inquiry, answer it
    with an answering assertion, and resolve it — the resolution version gate
    must treat a delta-created inquiry as the version-1 open row the INSERT
    will write, not as a missing row (additive validation branch)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    receipt = store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            observations=[_observation("observation-1")],
            inquiries=[
                InquiryInput(
                    id="inquiry-1",
                    subject_id="object-1",
                    prompt="Why?",
                    rationale="Gap",
                )
            ],
            assertions=[
                _assertion().model_copy(
                    update={"answers_inquiry_id": "inquiry-1"},
                )
            ],
            resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=1)],
        ),
        "commit-complete",
    )
    assert receipt.inquiry_ids == ["inquiry-1"]
    assert receipt.resolved_inquiry_ids == ["inquiry-1"]
    with sqlite3.connect(str(tmp_path / "world.sqlite3")) as connection:
        row = connection.execute(
            "SELECT status, version FROM inquiries WHERE id = 'inquiry-1'"
        ).fetchone()
    assert row == ("resolved", 2)


def test_same_delta_resolution_wrong_expected_version_fails_closed(tmp_path) -> None:
    """The delta-created branch still enforces the version gate: the staged
    expected_version must equal the version the INSERT writes (1), else the
    whole delta is rejected — never a silent version guess."""
    store = WorldStore(tmp_path / "world.sqlite3")
    with pytest.raises(ValueError, match="stale inquiry version"):
        store.memory_commit(
            CognitiveDelta(
                objects=[_object()],
                inquiries=[
                    InquiryInput(
                        id="inquiry-1",
                        subject_id="object-1",
                        prompt="Why?",
                        rationale="Gap",
                    )
                ],
                assertions=[
                    _assertion().model_copy(
                        update={"answers_inquiry_id": "inquiry-1"},
                    )
                ],
                resolve_inquiries=[InquiryResolution(id="inquiry-1", expected_version=2)],
            ),
            "commit-bad-version",
        )
    with sqlite3.connect(str(tmp_path / "world.sqlite3")) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM inquiries WHERE id = 'inquiry-1'"
        ).fetchone()[0] == 0


def test_deepens_id_persists_and_validates_against_delta_and_formal(tmp_path) -> None:
    """Wave B deepen: a delta inquiry may declare a parent via deepens_id
    (delta-created or formal); a missing parent or a self-reference fails the
    whole delta (mirrors the answers_inquiry_id existence gate)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    receipt = store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            inquiries=[
                InquiryInput(
                    id="inquiry-1",
                    subject_id="object-1",
                    prompt="Why?",
                    rationale="Gap",
                ),
                InquiryInput(
                    id="inquiry-2",
                    subject_id="object-1",
                    prompt="Deeper?",
                    rationale="Follow-up",
                    deepens_id="inquiry-1",
                ),
            ],
        ),
        "commit-deepen",
    )
    assert receipt.inquiry_ids == ["inquiry-1", "inquiry-2"]
    with sqlite3.connect(str(tmp_path / "world.sqlite3")) as connection:
        parent = connection.execute(
            "SELECT deepens_id FROM inquiries WHERE id = 'inquiry-2'"
        ).fetchone()
    assert parent == ("inquiry-1",)

    # a missing parent fails the delta
    with pytest.raises(ValueError, match="deepens_id references a nonexistent inquiry"):
        store.memory_commit(
            CognitiveDelta(
                objects=[_object()],
                inquiries=[
                    InquiryInput(
                        id="inquiry-3",
                        subject_id="object-1",
                        prompt="Orphan?",
                        rationale="Gap",
                        deepens_id="no-such-inquiry",
                    )
                ],
            ),
            "commit-orphan",
        )
    # a self-reference fails the delta
    with pytest.raises(ValueError, match="deepens_id references a nonexistent inquiry"):
        store.memory_commit(
            CognitiveDelta(
                objects=[_object()],
                inquiries=[
                    InquiryInput(
                        id="inquiry-4",
                        subject_id="object-1",
                        prompt="Loop?",
                        rationale="Gap",
                        deepens_id="inquiry-4",
                    )
                ],
            ),
            "commit-self",
        )
    # a formal target (already committed) is a legal parent
    store.memory_commit(
        CognitiveDelta(
            objects=[_object()],
            inquiries=[
                InquiryInput(
                    id="inquiry-5",
                    subject_id="object-1",
                    prompt="Chained?",
                    rationale="Gap",
                    deepens_id="inquiry-1",
                )
            ],
        ),
        "commit-chain",
    )
