# Split from tests/world/test_tools.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

from datetime import UTC, datetime

import jsonschema
import pytest

from leave_information_bubble.runtime.inquiry_lease import InquiryLeaseStore
from leave_information_bubble.world import (
    AssertionInput,
    CognitiveDelta,
    EpistemicRole,
    EvidenceInput,
    InquiryInput,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)
from leave_information_bubble.world.tools import WorldTools
from tests.world._recall_helpers import (
    _event_object,
    _object,
    _observation,
    _participant_edge,
)
from tests.world._tools_helpers import (
    CAP,
    NOW,
    SOURCE,
    _anchor,
    _tools,
)
from tests.world.test_tools_digest import (
    _digest_material_context,
    _digest_seed,
)
from tests.world.test_tools_open_depth import (
    _related_adapter,
)

READ_CAP = 15


def test_memory_tool_schemas_encode_paging_and_exclusivity_contracts() -> None:
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in tools.schemas()}
    checker = jsonschema.FormatChecker()

    search = jsonschema.Draft202012Validator(schemas["memory_search"], format_checker=checker)
    assert search.is_valid({"query": "event"})
    assert search.is_valid({"cursor": "opaque"})
    assert not search.is_valid({})
    assert not search.is_valid({"query": "event", "cursor": "opaque"})
    assert schemas["memory_search"]["properties"]["time_from"]["format"] == "date-time"

    read = jsonschema.Draft202012Validator(schemas["memory_read"])
    assert read.is_valid({})
    assert not read.is_valid({"object_id": "o1", "assertion_id": "a1"})

    expand = jsonschema.Draft202012Validator(schemas["memory_expand"])
    assert expand.is_valid({"object_ids": ["o1"], "depth": 2})
    assert expand.is_valid({"before": "opaque"})
    assert not expand.is_valid({})
    assert not expand.is_valid({"object_ids": ["o1"], "before": "opaque"})
    assert not expand.is_valid({"before": "opaque", "limit": 3})


async def test_memory_tools_recall_multiple_connected_objects_without_domain_filter(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch facade recall that filters an otherwise matching connected object by domain hints."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="food", kind=ObjectKind.ENTITY, canonical_name="Apple", domain_hints=["food"]),
                ObjectInput(
                    id="market", kind=ObjectKind.ENTITY, canonical_name="Market", domain_hints=["finance"]
                ),
            ]
        ),
        "seed",
    )

    search = await tools.execute("memory_search", {"query": "apple market"}, "search-call")
    expand = await tools.execute(
        "memory_expand",
        {"object_ids": ["food", "market", "missing-object"]},
        "expand-call",
    )

    assert [item["id"] for item in search["memory"]["anchor_objects"]] == ["food", "market"]
    assert [item["id"] for item in expand["memory"]["anchor_objects"]] == ["food", "market"]
    assert expand["scope"] == {
        "object_ids": ["food", "market", "missing-object"],
        "requested_object_count": 3,
        "object_ids_truncated": False,
        "depth": 1,
        "event_limit": 5,
        "before": None,
        "include_history": False,
    }
    assert expand["found_ids"] == ["food", "market"]
    assert expand["missing_ids"] == ["missing-object"]
    assert expand["limit"] == 30


async def test_memory_search_envelope_carries_bucket_candidates(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the read-side disambiguation envelope: buckets, hints, no confidence floats."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="team-a",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Team A",
                    aliases=["lng"],
                    domain_hints=["lol"],
                ),
                ObjectInput(
                    id="lpl-transfer",
                    kind=ObjectKind.ENTITY,
                    canonical_name="英雄联盟转会市场",
                    domain_hints=["lol"],
                ),
            ]
        ),
        "seed",
    )

    result = await tools.execute("memory_search", {"query": "lng"}, "search-call")
    expand = await tools.execute("memory_expand", {"object_ids": ["team-a"]}, "expand-call")
    recent = await tools.execute("memory_recent", {}, "recent-call")

    assert result["memory"]["candidates"] == [
        {
            "id": "team-a",
            "kind": "identity_alias_exact",
            "match_term": "lng",
            "domain_hints": ["lol"],
            "match_surface": "lng",
        }
    ]
    assert result["returned"]["candidates"] == 1
    assert set(result["memory"]["candidates"][0]) == {
        "id",
        "kind",
        "match_term",
        "domain_hints",
        "match_surface",
    }
    assert all("confidence" not in entry for entry in result["memory"]["candidates"])
    # candidates is additive: read envelopes without object matches carry no key
    assert "candidates" not in expand["memory"]
    assert "candidates" not in recent["memory"]


async def test_memory_expand_envelope_carries_ego_view_fields(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The ego view rides memory_expand: neighbors + predicates, status, timeline.

    The ego sections are additive: other memory reads keep their pre-ego
    payload shape (no empty ego keys), while memory_expand always presents
    the three sections so the model can distinguish "no connections" from
    "legacy payload".
    """
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="bin", kind=ObjectKind.ENTITY, canonical_name="Bin"),
                ObjectInput(id="blg", kind=ObjectKind.ENTITY, canonical_name="BLG"),
            ],
            observations=[
                ObservationInput(
                    id="obs-1",
                    source_uri=SOURCE,
                    source_kind="replay",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                )
            ],
            assertions=[
                AssertionInput(
                    id="edge-1",
                    subject_id="bin",
                    predicate="member_of",
                    object_id="blg",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                    evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                    event_time_start=datetime(2026, 8, 3, tzinfo=UTC),
                ),
                AssertionInput(
                    id="claim-1",
                    subject_id="bin",
                    predicate="status",
                    literal="temporary leave",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.7,
                    evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                ),
            ],
        ),
        "ego-envelope",
    )

    expand = await tools.execute(
        "memory_expand", {"object_ids": ["bin"], "depth": 1, "limit": 30}, "ego-call"
    )
    recent = await tools.execute("memory_recent", {}, "recent-call")
    search = await tools.execute("memory_search", {"query": "bin"}, "search-call")

    memory = expand["memory"]
    assert memory["ego_neighbors"] == [
        {
            "id": "blg",
            "canonical_name": "BLG",
            "predicates": ["member_of"],
            "assertion_ids": ["edge-1"],
        }
    ]
    assert memory["status_assertions"] == [
        {
            "id": "claim-1",
            "predicate": "status",
            "literal": "temporary leave",
            "event_time_start": None,
        }
    ]
    assert [item["id"] for item in memory["event_timeline"]] == ["edge-1"]
    assert memory["event_timeline"][0]["object"] == {"id": "blg", "canonical_name": "BLG"}
    assert expand["returned"]["ego_neighbors"] == 1
    assert expand["returned"]["status_assertions"] == 1
    assert expand["returned"]["event_timeline"] == 1
    # additive: other reads carry no ego keys at all
    assert "ego_neighbors" not in recent["memory"]
    assert "status_assertions" not in recent["memory"]
    assert "event_timeline" not in search["memory"]


async def test_memory_expand_clamps_hostile_native_arguments_and_complete_bundle(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch native arguments bypassing runtime depth, root, result, or evidence caps."""
    tools, store = _tools(tmp_path)
    node_ids = ["root", *[f"node-{index}" for index in range(10)]]
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id=item, kind=ObjectKind.ENTITY, canonical_name=item) for item in node_ids],
            observations=[
                ObservationInput(
                    id="chain-evidence",
                    source_uri=SOURCE,
                    source_kind="replay",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                )
            ],
            assertions=[
                AssertionInput(
                    id=f"edge-{index}",
                    subject_id=node_ids[index],
                    predicate="related_to",
                    object_id=node_ids[index + 1],
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="chain-evidence", role="supports")],
                )
                for index in range(10)
            ],
        ),
        "chain",
    )

    result = await tools.execute(
        "memory_expand",
        {
            "object_ids": ["root", *[f"missing-{index}" for index in range(1_000)]],
            "depth": 10_000,
            "limit": 10_000,
        },
        "hostile-expand",
    )
    memory = result["memory"]

    assert [row["id"] for row in memory["assertions"]] == [f"edge-{index}" for index in range(CAP)]
    assert len(memory["anchor_objects"]) <= CAP
    assert len(memory["evidence_refs"]) == CAP
    assert max(map(len, memory["paths"])) == CAP + 1


async def test_memory_expand_does_not_call_unapplied_root_missing(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Roots outside the applied result cap are marked truncated, not nonexistent."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id=f"obj-{index}", kind=ObjectKind.ENTITY, canonical_name=f"Obj{index}")
                for index in range(3)
            ]
        ),
        "three-roots",
    )

    result = await tools.execute(
        "memory_expand",
        {"object_ids": ["obj-0", "obj-1", "obj-2"], "limit": 2},
        "bounded-roots",
    )

    assert result["scope"] == {
        "object_ids": ["obj-0", "obj-1"],
        "requested_object_count": 3,
        "object_ids_truncated": True,
        "depth": 1,
        "event_limit": 5,
        "before": None,
        "include_history": False,
    }
    assert result["found_ids"] == ["obj-0", "obj-1"]
    assert result["missing_ids"] == []
    assert result["truncated"] is True


async def test_ego_distinguishes_missing_id_from_existing_isolated_object(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A nonexistent requested id is missing; an existing isolated object is found.

    The ego envelope resolves each requested root truthfully: an isolated
    but existing object lands in found_ids with empty ego sections and no
    truncation, while a requested id that exists nowhere lands in missing_ids
    — the two must never be conflated.
    """
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="solo", kind=ObjectKind.ENTITY, canonical_name="Solo")]),
        "solo",
    )

    result = await tools.execute("memory_expand", {"object_ids": ["solo", "ghost-id"]}, "ego-solo")

    assert result["found_ids"] == ["solo"]
    assert result["missing_ids"] == ["ghost-id"]
    assert result["truncated"] is False
    assert result["memory"]["ego_neighbors"] == []
    assert result["memory"]["event_edges"] == []
    assert result["memory"]["participated_events"] == []
    assert result["memory"]["omitted_counts"] == {}
    assert result["memory"]["sort_basis"] == ""
    assert result["returned"]["participated_events"] == 0


async def test_memory_expand_invalid_cursor_falls_back_to_first_page(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A dead or foreign before cursor degrades to the first page, flagged (P14), never an error.

    The event cursor is now a thread-bound token like memory_search's (design
    §4.5 unified pagination): a garbage token with object_ids present falls
    back to the first page of the current call and reports
    ``scope.cursor_status = "invalid_cursor"`` instead of failing the read.
    """
    tools, store = _tools(tmp_path)
    _seed_participated_events(store)

    garbage = await tools.execute(
        "memory_expand",
        {"object_ids": ["team-a"], "event_limit": 1, "before": "not-a-cursor"},
        "bad-cursor",
    )

    assert garbage["ok"] is True
    assert garbage["scope"]["cursor_status"] == "invalid_cursor"
    assert [item["id"] for item in garbage["memory"]["participated_events"]] == ["match-1"]
    # the degradation serves the first page, and the page still carries its cursor
    assert "next_cursor" in garbage


async def test_schemas_and_dispatch_cover_every_model_tool(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a provider schema that advertises a tool the facade cannot dispatch with that schema."""
    store = WorldStore(tmp_path / "world.sqlite3")
    leases = InquiryLeaseStore(tmp_path / "runtime.sqlite3")
    tools = WorldTools(store=store, adapters={"replay": _related_adapter()}, leases=leases)
    # log_inquiry_point appends to the default log path; keep it out of the repo
    monkeypatch.setattr(
        "leave_information_bubble.runtime.curiosity_log.CURIOSITY_LOG_DEFAULT",
        tmp_path / "curiosity-log.jsonl",
    )
    functions = {item["function"]["name"]: item["function"] for item in tools.schemas()}
    schemas = {name: function["parameters"] for name, function in functions.items()}
    expected = {
        "memory_recent",
        "memory_search",
        "memory_read",
        "memory_compare",
        "memory_expand",
        "memory_evidence",
        "memory_inquiries",
        "memory_changes",
        "memory_overview",
        "claim_inquiry",
        "release_inquiry",
        "discover_sources",
        "search_sources",
        "open_source",
        "sample_discussion",
        "inspect_media",
        "follow_related",
        "digest_observation",
        "propose_inquiry",
        "log_inquiry_point",
        "graph_patch",
        "graph_inspect",
        "graph_diff",
    }

    assert set(schemas) == expected
    assert all(
        schema["type"] == "object" and schema["additionalProperties"] is False for schema in schemas.values()
    )
    assert set(schemas["memory_recent"]["properties"]) == {"limit"}
    assert set(schemas["memory_inquiries"]["properties"]) == {"limit", "object_id", "inquiry_id"}
    assert set(schemas["memory_changes"]["properties"]) == {"since", "limit"}
    assert set(schemas["memory_overview"]["properties"]) == {"as_of", "limit"}
    assert set(schemas["propose_inquiry"]["properties"]) == {
        "subject",
        "prompt",
        "rationale",
        "kind",
        "deepens_inquiry_id",
        "answers_inquiry_id",
    }
    assert schemas["propose_inquiry"]["required"] == ["subject", "prompt", "rationale"]
    assert schemas["propose_inquiry"]["properties"]["kind"]["enum"] == [
        "factual",
        "semantic",
        "stateful",
    ]
    # the read tools advertise the widened window so the agent can request 6-15
    assert schemas["memory_recent"]["properties"]["limit"]["maximum"] == READ_CAP
    assert schemas["memory_search"]["properties"]["limit"]["maximum"] == READ_CAP
    assert schemas["memory_changes"]["properties"]["limit"]["maximum"] == READ_CAP
    assert schemas["memory_inquiries"]["properties"]["limit"]["maximum"] == 8
    assert schemas["memory_expand"]["properties"]["object_ids"]["maxItems"] == CAP
    assert schemas["memory_expand"]["properties"]["depth"]["maximum"] == CAP
    assert schemas["memory_expand"]["properties"]["limit"]["maximum"] == 30
    assert schemas["memory_expand"]["properties"]["event_limit"]["maximum"] == 30
    assert "before" in schemas["memory_expand"]["properties"]
    assert schemas["memory_expand"]["properties"]["include_history"]["type"] == "boolean"
    # object_ids are required for a first page, but paging may send before alone
    assert schemas["memory_expand"]["required"] == []
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="lease-center", kind=ObjectKind.ENTITY, canonical_name="Lease")],
            inquiries=[
                InquiryInput(id=identifier, subject_id="lease-center", prompt=identifier, rationale="r")
                for identifier in ("q", "release")
            ],
        ),
        "schema-lease-inquiries",
    )
    _digest_seed(store)
    digest_material_id_value, _ = _digest_material_context()
    anchor_id = await _anchor(tools, "schema-seed")
    calls = {
        "memory_recent": {},
        "memory_search": {"query": "missing"},
        "memory_expand": {"object_ids": ["missing"]},
        "memory_evidence": {"assertion_id": "missing"},
        "memory_inquiries": {},
        "memory_changes": {},
        "memory_overview": {},
        "discover_sources": {"adapter": "replay", "query": "source"},
        "search_sources": {"adapter": "replay", "query": "source"},
        "open_source": {"observation_id": anchor_id},
        "sample_discussion": {"observation_id": anchor_id},
        "inspect_media": {"observation_id": anchor_id},
        "follow_related": {"observation_id": anchor_id},
        "claim_inquiry": {"inquiry_id": "q", "owner_id": "agent"},
        "digest_observation": {
            "material_id": digest_material_id_value,
            "points": ["p"],
            "assertion_candidates": [],
            "evidence_refs": [],
        },
        "propose_inquiry": {
            "subject": {"local_ref": "obj-x"},
            "prompt": "q",
            "rationale": "r",
        },
        "log_inquiry_point": {"topic": "t", "source_ref": "s", "reason": "r"},
    }
    for name, arguments in calls.items():
        assert set(schemas[name]["required"]) <= set(arguments)
        assert (await tools.execute(name, arguments, f"schema-{name}"))["ok"] is True
    lease = await tools.execute("claim_inquiry", {"inquiry_id": "release", "owner_id": "agent"}, "lease")
    released = await tools.execute(
        "release_inquiry", {"lease_token": lease["lease"]["lease_token"]}, "release"
    )

    assert released == {"ok": True, "released": True, "status": "released"}


async def test_memory_overview_rejects_invalid_as_of_and_limit_without_side_effects(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """P3 overview follows the facade's typed invalid-arguments protocol."""
    tools = WorldTools(store=WorldStore(tmp_path / "world.sqlite3"), adapters={})
    for arguments in ({"as_of": "not-a-time"}, {"as_of": "2026-08-10T00:00:00"}, {"limit": 0}):
        assert await tools.execute("memory_overview", arguments, "invalid-overview") == {
            "ok": False,
            "limitations": ["invalid_arguments"],
        }


async def test_expand_returns_up_to_thirty(tmp_path: pytest.TempPathFactory) -> None:
    """Catch the expand card cap hiding graph neighborhoods wider than five assertions."""
    store = WorldStore(tmp_path / "world.sqlite3")
    # seed one center object + 12 assertions linking 12 neighbors (object-valued)
    objects = [ObjectInput(id=f"obj-{i}", kind=ObjectKind.ENTITY, canonical_name=f"N{i}") for i in range(13)]
    assertions = [
        AssertionInput(
            id=f"a-{i}",
            subject_id="obj-0",
            predicate="related_to",
            object_id=f"obj-{i + 1}",
            epistemic_role=EpistemicRole.FACT,
            confidence=0.9,
        )
        for i in range(12)
    ]
    store.memory_commit(CognitiveDelta(objects=objects, assertions=assertions), "seed-1")
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("memory_expand", {"object_ids": ["obj-0"], "depth": 1, "limit": 30}, "c1")

    assert result["ok"] is True
    assert len(result["memory"]["assertions"]) == 12  # > 5, uncapped by the old _CARD_CAP
    assert len(result["memory"]["paths"]) == 12
    assert result["memory"]["paths_text"] == [" → ".join(path) for path in result["memory"]["paths"]]
    assert result["memory"]["paths_text"][0] == "obj-0 → obj-1"


async def test_search_and_recent_read_up_to_fifteen(tmp_path: pytest.TempPathFactory) -> None:
    """Catch agent-initiated recall still hiding behind the five-card injection cap."""
    store = WorldStore(tmp_path / "world.sqlite3")
    # seed 20 assertions across 2 objects so memory_recent can return >5
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id=f"obj-{i}", kind=ObjectKind.ENTITY, canonical_name=f"Obj{i}") for i in range(2)
            ],
            assertions=[
                AssertionInput(
                    id=f"a-{i}",
                    subject_id=f"obj-{i % 2}",
                    predicate="has_result",
                    literal=str(i),
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                )
                for i in range(20)
            ],
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("memory_recent", {"limit": 15}, "c1")

    assert result["ok"] is True
    assert len(result["memory"]["assertions"]) == 15  # > 5


async def test_bootstrap_injection_still_capped_at_five(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch bootstrap injection leaking past the five-card budget into the prompt."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="X")],
            assertions=[
                AssertionInput(
                    id=f"a-{i}",
                    subject_id="obj-1",
                    predicate="p",
                    literal=str(i),
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                )
                for i in range(10)
            ],
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={})
    # bootstrap calls with limit 6 must still return 5 (injection budget)
    result = await tools.execute("memory_recent", {"limit": 6}, "bootstrap-recent")
    assert len(result["memory"]["assertions"]) == 5


async def test_memory_changes_tool(tmp_path: pytest.TempPathFactory) -> None:
    """Catch the memory_changes tool hiding world deltas behind its envelope."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="X")]),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("memory_changes", {}, "c1")

    assert result["ok"] is True
    assert "obj-1" in result["memory"]["new_objects"]


async def test_memory_recent_summary_first_person_with_ids(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch memory returns lacking the first-person translation (spec B4)."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id=f"obj-{i}", kind=ObjectKind.ENTITY, canonical_name=f"Obj{i}") for i in range(2)
            ],
            assertions=[
                AssertionInput(
                    id=f"a-{i}",
                    subject_id=f"obj-{i}",
                    predicate="has_result",
                    literal=str(i),
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.9,
                )
                for i in range(2)
            ],
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("memory_recent", {}, "c1")

    assert result["ok"] is True
    assert "You remember 2 revisable judgments" in result["summary"]
    assert "confirmed" not in result["summary"].casefold()
    # ids are repeated in the summary text so the model can cite them directly
    assert "a-0" in result["summary"]
    assert "a-1" in result["summary"]
    # the machine-readable payload stays parseable for existing consumers
    assert {item["id"] for item in result["memory"]["assertions"]} == {"a-0", "a-1"}


async def test_memory_inquiries_summary_first_person_with_ids(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the inquiries tool omitting the first-person blind-spot translation."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="ShallowCenter")],
            inquiries=[InquiryInput(id="inq-1", subject_id="obj-1", prompt="shallow q", rationale="r")],
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("memory_inquiries", {}, "c1")

    assert result["ok"] is True
    assert "Open remembered questions include: inq-1 (obj-1): shallow q" in result["summary"]
    assert "backlog" not in result["summary"].casefold()
    # the payload keeps the inquiry id parseable for citation
    assert result["memory"]["inquiries"][0]["id"] == "inq-1"


async def test_memory_evidence_distinguishes_missing_from_zero_evidence(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """An existing zero-link judgment is not reported as a missing assertion id."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="Center")],
            assertions=[
                AssertionInput(
                    id="a-1",
                    subject_id="obj-1",
                    predicate="has_state",
                    literal="uncited but durable",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.5,
                )
            ],
        ),
        "seed-zero-evidence",
    )
    tools = WorldTools(store=store, adapters={})

    existing = await tools.execute("memory_evidence", {"assertion_id": "a-1"}, "existing")
    missing = await tools.execute("memory_evidence", {"assertion_id": "a-missing"}, "missing")

    assert existing["scope"] == {"assertion_id": "a-1"}
    assert existing["found_ids"] == ["a-1"]
    assert existing["missing_ids"] == []
    assert existing["returned"]["assertions"] == 1
    assert existing["returned"]["evidence_refs"] == 0
    assert existing["truncated"] is False
    assert missing["scope"] == {"assertion_id": "a-missing"}
    assert missing["found_ids"] == []
    assert missing["missing_ids"] == ["a-missing"]
    assert missing["returned"]["assertions"] == 0
    assert missing["returned"]["evidence_refs"] == 0
    assert missing["truncated"] is False


async def test_memory_recent_reports_exact_cap_truncation(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The memory facade proves truncation with a cap+1 read instead of guessing at equality."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="Center")],
            assertions=[
                AssertionInput(
                    id=f"a-{index}",
                    subject_id="obj-1",
                    predicate="has_state",
                    literal=str(index),
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.5,
                )
                for index in range(3)
            ],
        ),
        "seed-three",
    )
    tools = WorldTools(store=store, adapters={})

    capped = await tools.execute("memory_recent", {"limit": 2}, "capped")
    complete = await tools.execute("memory_recent", {"limit": 3}, "complete")

    assert capped["scope"] == {"order": "newest_first"}
    assert capped["limit"] == 2
    assert capped["returned"]["assertions"] == 2
    assert capped["truncated"] is True
    assert complete["limit"] == 3
    assert complete["returned"]["assertions"] == 3
    assert complete["truncated"] is False


async def test_memory_changes_summary_first_person(tmp_path: pytest.TempPathFactory) -> None:
    """Catch memory_changes omitting the first-person change summary."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="X")]),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("memory_changes", {}, "c1")

    assert result["ok"] is True
    assert "Your memory changed since" in result["summary"]
    assert "1 new object" in result["summary"]  # singular form, not "1 new objects"
    assert "obj-1" in result["memory"]["new_objects"]
    # an explicit since renders as a readable date, never a raw ISO timestamp
    dated = await tools.execute("memory_changes", {"since": "2026-08-01T00:00:00+00:00"}, "c2")
    assert "since 2026-08-01 00:00 UTC" in dated["summary"]
    assert "T00:00:00+00:00" not in dated["summary"]
    # plural counts render with the plural noun
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id=f"obj-{i}", kind=ObjectKind.ENTITY, canonical_name=f"Y{i}")
                for i in range(2, 4)
            ]
        ),
        "seed-2",
    )
    plural = await tools.execute("memory_changes", {}, "c3")
    assert "3 new objects" in plural["summary"]
    assert "0 new revisable judgments" in plural["summary"]


async def test_memory_inquiries_summary_marks_truncated_ids(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch inquiry ids beyond the summary cap disappearing without a marker."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="Center")],
            inquiries=[
                InquiryInput(id=f"inq-{i}", subject_id="obj-1", prompt=f"q{i}", rationale="r")
                for i in range(5)
            ],
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("memory_inquiries", {"limit": 5}, "c1")

    assert result["ok"] is True
    summary = result["summary"]
    assert "inq-0" in summary and "inq-2" in summary
    assert "inq-3" not in summary  # truncated from the summary text
    assert "and 2 more (see memory)" in summary  # honesty marker for the hidden ids
    # every id remains parseable in the payload
    assert {item["id"] for item in result["memory"]["inquiries"]} == {f"inq-{i}" for i in range(5)}


async def test_memory_inquiries_injection_widened_to_eight(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the injection clamp reverting below the widened inquiry window.

    The bootstrap-shallow injection asks for 8 open inquiries; the tool must
    return all of them and clamp anything above the dedicated _INQUIRY_CAP
    rather than the general 5-card cap.
    """
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="obj-1", kind=ObjectKind.ENTITY, canonical_name="Center")],
            inquiries=[
                InquiryInput(id=f"inq-{i}", subject_id="obj-1", prompt=f"q{i}", rationale="r")
                for i in range(9)
            ],
        ),
        "seed-1",
    )
    tools = WorldTools(store=store, adapters={})

    widened = await tools.execute("memory_inquiries", {"limit": 8}, "c1")

    assert widened["ok"] is True
    assert {item["id"] for item in widened["memory"]["inquiries"]} == {f"inq-{i}" for i in range(8)}
    assert widened["limit"] == 8
    assert widened["truncated"] is True
    # a request above the dedicated inquiry cap still clamps at it
    clamped = await tools.execute("memory_inquiries", {"limit": 20}, "c2")
    assert {item["id"] for item in clamped["memory"]["inquiries"]} == {f"inq-{i}" for i in range(8)}
    assert clamped["limit"] == 8
    assert clamped["truncated"] is True


async def test_memory_changes_rejects_bad_since(tmp_path: pytest.TempPathFactory) -> None:
    """Catch an unparsable since string escaping the facade as a typed limitation."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("memory_changes", {"since": "not-a-date"}, "c1")

    assert result["ok"] is False
    assert result["limitations"] == ["invalid_arguments"]


def test_submit_schema_and_memory_tools_use_same_identity_terms() -> None:
    """submit_cognition and memory tools quote the contract's identity sentence.

    The identity model (canonical = display name, not unique; legacy = history;
    only an active identity alias is world-unique; name usage is an explicit
    assertion that may carry evidence) is defined once in
    graph_contract_text and referenced by the terminal schema, the memory tool
    descriptions, and the contract text itself — the same full sentence on
    every surface, never a parallel re-wording.
    """
    from leave_information_bubble.world import submit_cognition_schema
    from leave_information_bubble.world.graph_contract_text import (
        IDENTITY_MODEL_SENTENCE,
        render_contract_text,
    )

    submit = submit_cognition_schema()["function"]["description"]
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in tools.schemas()}

    assert IDENTITY_MODEL_SENTENCE in submit
    assert IDENTITY_MODEL_SENTENCE in descriptions["memory_search"]
    assert IDENTITY_MODEL_SENTENCE in render_contract_text()


def test_shared_contract_does_not_turn_optional_evidence_into_a_write_gate() -> None:
    from leave_information_bubble.world.graph_contract_text import render_contract_text

    lowered = " ".join(render_contract_text().split()).casefold()
    assert "may carry evidence links" in lowered
    assert "may link supporting evidence when available" in lowered
    assert "submit with evidence" not in lowered


def test_contract_text_teaches_six_kinds_and_event_time_anchors() -> None:
    """The shared contract names all six kinds and the event build boundary.

    Kind is the durable display/query axis; type_key refines it with
    domain-neutral words; events are time anchors built for multi-entity
    facts or single-entity state changes, never for momentary statements.
    """
    from leave_information_bubble.world.graph_contract_text import render_contract_text

    lowered = " ".join(render_contract_text().split()).casefold()
    for kind in ("person", "organization", "place", "event", "concept", "entity"):
        assert kind in lowered
    assert "entity is the fallback, not the default" in lowered
    assert "connects multiple entities" in lowered
    assert "marks a change of state" in lowered
    assert "momentary statements" in lowered
    assert "event_time_start" in lowered
    assert "team, player, match, tournament, season, version" in lowered


def test_event_reification_rule_is_shared_by_contract_prompt_and_write_tool() -> None:
    """The active model surfaces use one exact event/direct-edge decision rule.

    This catches drift back to a definition-only event description or to a
    domain-specific workaround. The write schema repeats the same rule because
    provider tool-selection and argument generation may attend to different
    description layers.
    """
    from leave_information_bubble.world.graph_contract_text import (
        EVENT_REIFICATION_RULE,
        render_contract_text,
    )
    from leave_information_bubble.world_agent.prompt import GRAPH_SHELL_MECHANICS

    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    graph_patch = next(
        item["function"] for item in tools.schemas() if item["function"]["name"] == "graph_patch"
    )

    assert EVENT_REIFICATION_RULE in render_contract_text()
    assert EVENT_REIFICATION_RULE in GRAPH_SHELL_MECHANICS
    assert EVENT_REIFICATION_RULE in graph_patch["description"]
    assert EVENT_REIFICATION_RULE in graph_patch["parameters"]["description"]

    lowered = EVENT_REIFICATION_RULE.casefold()
    assert "does not depend on importance" in lowered
    assert "never compress a preserved occurrence" in lowered
    assert "containing episode or period does not replace" in lowered
    assert "qualifiers only refine one assertion edge" in lowered
    assert "host cannot infer a missing event" in lowered


def test_graph_patch_repeats_the_domain_persistence_gate_at_the_write_boundary() -> None:
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    graph_patch = next(
        item["function"] for item in tools.schemas() if item["function"]["name"] == "graph_patch"
    )
    lowered = " ".join(graph_patch["description"].split()).casefold()

    assert "eligible under the domain lens persistence scope" in lowered
    assert "being discovered or associated is not by itself eligible" in lowered


# --- Slice 2a: memory_search mode/cursor/filters at the tool facade ---


async def test_memory_search_count_mode_reports_counts(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """mode=count returns per-bucket totals and page size, not rows."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="orbit-project", kind=ObjectKind.ENTITY, canonical_name="Orbit Project"),
                ObjectInput(id="orbit-launch", kind=ObjectKind.EVENT, canonical_name="Orbit Launch"),
            ],
        ),
        "count-seed",
    )

    result = await tools.execute("memory_search", {"query": "orbit", "mode": "count"}, "count-call")

    assert result["ok"] is True
    assert result["memory"] == {
        "counts": {"objects": 2, "assertions": 0, "inquiries": 0},
        "page_size": 12,
    }
    assert result["scope"]["mode"] == "count"
    assert "anchor_objects" not in result["memory"]
    assert result["truncated"] is False


async def test_memory_search_scope_carries_structured_filters(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The read scope projects every structured filter for traceability."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="team-a", kind=ObjectKind.ENTITY, canonical_name="Team A")]),
        "scope-seed",
    )

    result = await tools.execute(
        "memory_search",
        {
            "query": "team",
            "kind": "entity",
            "predicate": "status",
            "has_participants": True,
            "assertion_count_min": 1,
            "assertion_count_max": 5,
            "time_from": "2026-08-01T00:00:00Z",
            "time_to": "2026-08-05T00:00:00Z",
        },
        "scope-call",
    )

    scope = result["scope"]
    assert scope["kind"] == "entity"
    assert scope["predicate"] == "status"
    assert scope["has_participants"] is True
    assert scope["assertion_count_min"] == 1
    assert scope["assertion_count_max"] == 5
    assert scope["time_from"] == "2026-08-01T00:00:00+00:00"
    assert scope["time_to"] == "2026-08-05T00:00:00+00:00"
    assert result["ok"] is True


async def test_memory_search_cursor_paginates(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Page 1 returns a cursor; the cursor call resumes the same query's next page."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="orbit-a", kind=ObjectKind.EVENT, canonical_name="Orbit A"),
                ObjectInput(id="orbit-b", kind=ObjectKind.EVENT, canonical_name="Orbit B"),
                ObjectInput(id="orbit-c", kind=ObjectKind.EVENT, canonical_name="Orbit C"),
            ],
        ),
        "paging-seed",
    )

    first = await tools.execute("memory_search", {"query": "orbit", "limit": 2}, "page-1")
    assert [row["id"] for row in first["memory"]["anchor_objects"]] == ["orbit-a", "orbit-b"]
    assert first["truncated"] is True
    cursor = first["next_cursor"]
    assert isinstance(cursor, str) and cursor

    second = await tools.execute("memory_search", {"cursor": cursor}, "page-2")
    assert [row["id"] for row in second["memory"]["anchor_objects"]] == ["orbit-c"]
    assert second["truncated"] is False
    assert "next_cursor" not in second
    assert second["scope"]["offset"] == 2
    assert second["scope"]["query"] == "orbit"


async def test_memory_search_invalid_cursor_falls_back_to_first_page(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """P14: a garbage or cross-wake cursor degrades to page one, flagged, never errors."""
    tools, store = _tools(tmp_path, thread_id="wake-a")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="orbit-a", kind=ObjectKind.EVENT, canonical_name="Orbit A"),
                ObjectInput(id="orbit-b", kind=ObjectKind.EVENT, canonical_name="Orbit B"),
            ],
        ),
        "cursor-seed",
    )
    first = await tools.execute("memory_search", {"query": "orbit", "limit": 1}, "page-1")
    token = first["next_cursor"]

    garbage = await tools.execute(
        "memory_search", {"cursor": "not-a-cursor", "query": "orbit", "limit": 1}, "page-2"
    )
    assert garbage["ok"] is True
    assert [row["id"] for row in garbage["memory"]["anchor_objects"]] == ["orbit-a"]
    assert garbage["scope"]["cursor_status"] == "invalid_cursor"
    assert garbage["scope"]["offset"] == 0

    # a dead cursor without a query has nothing to fall back to: fail closed
    orphaned = await tools.execute("memory_search", {"cursor": "not-a-cursor"}, "page-2")
    assert orphaned["ok"] is False
    assert orphaned["limitations"] == ["invalid_cursor"]

    # a cursor minted under a different wake must not resume that wake's page
    foreign, _ = _tools(tmp_path, thread_id="wake-b")
    cross = await foreign.execute("memory_search", {"cursor": token, "query": "orbit", "limit": 1}, "page-2")
    assert cross["scope"]["cursor_status"] == "invalid_cursor"
    assert [row["id"] for row in cross["memory"]["anchor_objects"]] == ["orbit-a"]


async def test_memory_search_rejects_malformed_filters(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Malformed structured arguments fail closed with a typed limitation."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="team-a", kind=ObjectKind.ENTITY, canonical_name="Team A")]),
        "bad-filter",
    )

    bad_time = await tools.execute("memory_search", {"query": "team", "time_from": "not-a-date"}, "bad-call")
    assert bad_time["ok"] is False
    assert bad_time["limitations"] == ["invalid_arguments"]

    bad_kind = await tools.execute("memory_search", {"query": "team", "kind": "galaxy"}, "bad-kind")
    assert bad_kind["limitations"] == ["invalid_arguments"]

    bad_mode = await tools.execute("memory_search", {"query": "team", "mode": "browse"}, "bad-mode")
    assert bad_mode["limitations"] == ["invalid_arguments"]

    contradictory = await tools.execute(
        "memory_search", {"query": "team", "mode": "count", "cursor": "x"}, "bad-combo"
    )
    assert contradictory["limitations"] == ["invalid_arguments"]

    bad_range = await tools.execute(
        "memory_search",
        {"query": "team", "assertion_count_min": 3, "assertion_count_max": 1},
        "bad-range",
    )
    assert bad_range["limitations"] == ["invalid_arguments"]


async def test_memory_read_object_identity_envelope(tmp_path: pytest.TempPathFactory) -> None:
    """memory_read returns the identity portrait under the standard envelope."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="t1",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Team One",
                    aliases=["T1"],
                )
            ],
            observations=[
                ObservationInput(
                    id="observation-1",
                    source_uri=SOURCE,
                    source_kind="web",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                )
            ],
            assertions=[
                AssertionInput(
                    id="a-1",
                    subject_id="t1",
                    predicate="related_to",
                    literal="attribute",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
                )
            ],
        ),
        "seed",
    )

    result = await tools.execute("memory_read", {"object_id": "t1", "views": ["self_attributes"]}, "read")

    assert result["ok"] is True
    assert result["scope"] == {
        "object_id": "t1",
        "assertion_id": None,
        "defaulted": False,
        "views": ["self_attributes"],
        "include_working": False,
    }
    identity = result["memory"]["identity"]
    assert identity["canonical_name"] == "Team One"
    assert identity["active_aliases"][0]["raw_alias"] == "T1"
    assert identity["view_counts"]["self_attributes"] == 1
    assert result["memory"]["views"]["self_attributes"][0]["id"] == "a-1"
    assert result["returned"]["anchor_objects"] == 1
    assert result["found_ids"] == ["t1"]
    assert result["truncated"] is False


async def test_memory_read_default_object_scope_marks_defaulted(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Omitting ids reads the latest committed object and says so in scope."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")]),
        "seed",
    )

    result = await tools.execute("memory_read", {}, "read-default")

    assert result["ok"] is True
    assert result["scope"]["defaulted"] is True
    assert result["memory"]["identity"]["object_id"] == "t1"


async def test_memory_read_include_working_without_staging_reads_formal(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """include_working with an empty working graph reads the formal portrait unchanged."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")]),
        "seed",
    )

    result = await tools.execute("memory_read", {"object_id": "t1", "include_working": True}, "read-working")

    assert result["ok"] is True
    assert result["scope"]["include_working"] is True
    assert result["memory"]["identity"]["canonical_name"] == "Team One"
    assert result["memory"]["identity"]["active_aliases"] == []
    assert result["found_ids"] == ["t1"]


async def test_memory_read_conflicting_ids_rejected(tmp_path: pytest.TempPathFactory) -> None:
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")]),
        "seed",
    )

    result = await tools.execute("memory_read", {"object_id": "t1", "assertion_id": "a-1"}, "read-conflict")

    assert result["ok"] is False
    assert result["limitations"] == ["invalid_arguments"]


async def test_memory_read_unknown_view_rejected(tmp_path: pytest.TempPathFactory) -> None:
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")]),
        "seed",
    )

    result = await tools.execute("memory_read", {"object_id": "t1", "views": ["sideways"]}, "read-bad-view")

    assert result["limitations"] == ["invalid_arguments"]


async def test_memory_read_unknown_object_reports_missing_id(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A requested id that does not exist lands in missing_ids, not a crash."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")]),
        "seed",
    )

    result = await tools.execute("memory_read", {"object_id": "nope"}, "read-missing")

    assert result["ok"] is True
    assert result["memory"]["anchor_objects"] == []
    assert result["missing_ids"] == ["nope"]


async def test_memory_read_assertion_evidence_envelope(tmp_path: pytest.TempPathFactory) -> None:
    """assertion_id renders the evidence view with its chain in scope."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")],
            observations=[
                ObservationInput(
                    id="observation-1",
                    source_uri=SOURCE,
                    source_kind="web",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                )
            ],
            assertions=[
                AssertionInput(
                    id="a-1",
                    subject_id="t1",
                    predicate="related_to",
                    literal="v1",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
                )
            ],
        ),
        "seed",
    )

    result = await tools.execute("memory_read", {"assertion_id": "a-1"}, "read-evidence")

    assert result["ok"] is True
    assert result["scope"] == {
        "object_id": None,
        "assertion_id": "a-1",
        "defaulted": False,
        "views": [],
        "include_working": False,
    }
    assert [card["id"] for card in result["memory"]["assertions"]] == ["a-1"]
    assert result["memory"]["assertion_chain"] == {"supersedes": [], "superseded_by": []}
    assert result["found_ids"] == ["a-1"]


async def test_memory_compare_envelope_carries_payload_and_scope(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """memory_compare resolves both ids as found and reports the side-by-side."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One"),
                ObjectInput(id="t2", kind=ObjectKind.ENTITY, canonical_name="Team 1"),
            ]
        ),
        "seed",
    )

    result = await tools.execute("memory_compare", {"left_id": "t1", "right_id": "t2"}, "compare")

    assert result["ok"] is True
    assert result["scope"] == {"left_id": "t1", "right_id": "t2"}
    compare = result["memory"]["compare"]
    assert compare["mode"] == "object"
    assert compare["left"] == {"id": "t1", "status": "ok", "canonical_name": "Team One"}
    fields = {entry["field"]: entry for entry in compare["fields"]}
    assert fields["canonical_name"]["equal"] is False
    assert fields["kind"]["equal"] is True
    assert result["found_ids"] == ["t1", "t2"]
    assert result["missing_ids"] == []
    assert result["truncated"] is False


async def test_memory_compare_missing_side_reports_missing_ids(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """An unresolvable side lands in missing_ids, no compare payload, no crash."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")]),
        "seed",
    )

    result = await tools.execute("memory_compare", {"left_id": "t1", "right_id": "nope"}, "compare-missing")

    assert result["ok"] is True
    assert result["memory"]["compare"] is None
    assert result["found_ids"] == []
    assert result["missing_ids"] == ["t1", "nope"]


async def test_memory_compare_mixed_kinds_rejected(tmp_path: pytest.TempPathFactory) -> None:
    """An object cannot be compared with an assertion: fail closed, no guesses."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="t1", kind=ObjectKind.ENTITY, canonical_name="Team One")],
            observations=[
                ObservationInput(
                    id="observation-1",
                    source_uri=SOURCE,
                    source_kind="web",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                )
            ],
            assertions=[
                AssertionInput(
                    id="a-1",
                    subject_id="t1",
                    predicate="related_to",
                    literal="x",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
                )
            ],
        ),
        "seed",
    )

    result = await tools.execute("memory_compare", {"left_id": "t1", "right_id": "a-1"}, "compare-mixed")

    assert result["ok"] is False
    assert result["limitations"] == ["invalid_arguments"]


def test_memory_compare_schema_requires_both_ids(tmp_path: pytest.TempPathFactory) -> None:
    """The compare schema demands exactly the two ids and nothing else."""
    tools, _store = _tools(tmp_path)
    parameters = {item["function"]["name"]: item["function"]["parameters"] for item in tools.schemas()}[
        "memory_compare"
    ]

    assert set(parameters["properties"]) == {"left_id", "right_id"}
    assert parameters["required"] == ["left_id", "right_id"]
    assert parameters["additionalProperties"] is False


def _seed_participated_events(store: WorldStore) -> None:
    """Seed team-a participating in three matches (Aug 3 / Aug 2 / Aug 3)."""
    store.memory_commit(
        CognitiveDelta(
            objects=[
                _object("team-a"),
                _event_object("match-1", event_time_start=datetime(2026, 8, 3, tzinfo=UTC)),
                _event_object("match-2", event_time_start=datetime(2026, 8, 2, tzinfo=UTC)),
                _event_object("match-3", event_time_start=datetime(2026, 8, 3, tzinfo=UTC)),
            ],
            observations=[_observation("observation-1")],
            assertions=[
                _participant_edge("m1p", "match-1", "team-a", role="team"),
                _participant_edge("m2p", "match-2", "team-a", role="team"),
                _participant_edge("m3p", "match-3", "team-a", role="team"),
            ],
        ),
        "entity-ego",
    )


async def test_memory_expand_dead_cursor_only_fails_closed(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A cursor-only call whose token is dead fails closed: there is nothing to fall back to."""
    tools, store = _tools(tmp_path)
    _seed_participated_events(store)

    orphaned = await tools.execute("memory_expand", {"before": "not-a-cursor"}, "orphan")

    assert orphaned["ok"] is False
    assert orphaned["limitations"] == ["invalid_cursor"]


async def test_memory_expand_foreign_thread_cursor_degraded(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A cursor minted in another wake pages nothing: it degrades to a first page."""
    tools, store = _tools(tmp_path)
    _seed_participated_events(store)
    foreign = WorldTools(store=store, adapters={}, thread_id="wake-other")
    minted = await foreign.execute(
        "memory_expand", {"object_ids": ["team-a"], "event_limit": 1}, "foreign-call"
    )

    cross = await tools.execute(
        "memory_expand",
        {"object_ids": ["team-a"], "event_limit": 1, "before": minted["next_cursor"]},
        "cross-call",
    )

    assert cross["ok"] is True
    assert cross["scope"]["cursor_status"] == "invalid_cursor"
    assert [item["id"] for item in cross["memory"]["participated_events"]] == ["match-1"]


async def test_memory_expand_pages_participated_events_via_thread_bound_cursor(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """before accepts the envelope-level next_cursor; the payload keeps the raw position marker."""
    tools, store = _tools(tmp_path)
    _seed_participated_events(store)

    first = await tools.execute("memory_expand", {"object_ids": ["team-a"], "event_limit": 2}, "page-1")

    # Aug 3 first (DESC); the two Aug 3 events tie-break by id ASC
    assert [item["id"] for item in first["memory"]["participated_events"]] == ["match-1", "match-3"]
    # the payload keeps the recall layer's raw position token...
    assert first["memory"]["event_next_cursor"] == ('{"t": "2026-08-03T00:00:00+00:00", "id": "match-3"}')
    # ...while the envelope cursor is the thread-bound token, not the raw marker
    assert first["next_cursor"] != first["memory"]["event_next_cursor"]

    second = await tools.execute(
        "memory_expand",
        {"object_ids": ["team-a"], "event_limit": 2, "before": first["next_cursor"]},
        "page-2",
    )

    assert [item["id"] for item in second["memory"]["participated_events"]] == ["match-2"]
    assert "next_cursor" not in second
    assert "cursor_status" not in second["scope"]


async def test_memory_expand_include_history_scope_and_payload(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """include_history surfaces retired claims; the schema and scope report it."""
    tools, store = _tools(tmp_path)
    store.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="s16", kind=ObjectKind.ENTITY, canonical_name="S16")],
            observations=[
                ObservationInput(
                    id="obs-1",
                    source_uri=SOURCE,
                    source_kind="replay",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                ),
                ObservationInput(
                    id="obs-2",
                    source_uri=SOURCE,
                    source_kind="replay",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                ),
            ],
            assertions=[
                AssertionInput(
                    id="assertion-old",
                    subject_id="s16",
                    predicate="related_to",
                    literal="old title",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    evidence=[EvidenceInput(observation_id="obs-1", role="supports")],
                ),
            ],
        ),
        "old-claims",
    )
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="obs-3",
                    source_uri=SOURCE,
                    source_kind="replay",
                    depth=ObservationDepth.CONTENT,
                    observed_at=NOW,
                ),
            ],
            assertions=[
                AssertionInput(
                    id="assertion-new",
                    subject_id="s16",
                    predicate="related_to",
                    literal="new title",
                    epistemic_role=EpistemicRole.FACT,
                    confidence=0.8,
                    supersedes_id="assertion-old",
                    evidence=[EvidenceInput(observation_id="obs-3", role="supports")],
                ),
            ],
        ),
        "supersede-claim",
    )

    historical = await tools.execute(
        "memory_expand", {"object_ids": ["s16"], "include_history": True}, "history-call"
    )

    assert historical["ok"] is True
    assert historical["scope"]["include_history"] is True
    # per-root id order: "assertion-new" sorts before "assertion-old" (n < o)
    assert [row["id"] for row in historical["memory"]["assertions"]] == [
        "assertion-new",
        "assertion-old",
    ]

    current = await tools.execute("memory_expand", {"object_ids": ["s16"]}, "current-call")

    assert current["scope"]["include_history"] is False
    assert [row["id"] for row in current["memory"]["assertions"]] == ["assertion-new"]
