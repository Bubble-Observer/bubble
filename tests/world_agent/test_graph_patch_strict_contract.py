"""C-EXT-01 strict Graph Patch schema and pre-dispatch side-effect proofs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from leave_information_bubble.gateway.client import NativeToolCall
from leave_information_bubble.world import (
    CognitiveDelta,
    ObjectInput,
    ObjectKind,
    WorldStore,
    WorldTools,
)
from leave_information_bubble.world_agent.graph import build_world_agent_graph
from tests.world_agent._graph_helpers import _response, _ScriptedModel

_VARIANTS = (
    {
        "name": "object.create",
        "valid": {
            "op_id": "object-create",
            "kind": "object",
            "action": "create",
            "payload": {
                "canonical_name": "New Object",
                "kind": "concept",
                "provisional": True,
            },
        },
        "missing": ("payload", "canonical_name"),
        "wrong_type": ("payload", "canonical_name", 7),
        "wrong_value": ("payload", "kind", "robot"),
        "neighbor": (None, "target_ref", "formal-object"),
    },
    {
        "name": "object.update",
        "valid": {
            "op_id": "object-update",
            "kind": "object",
            "action": "update",
            "target_ref": "formal-object",
            "expected_version": 1,
            "payload": {"canonical_name": "Revised Object", "kind": "concept"},
        },
        "missing": (None, "target_ref"),
        "wrong_type": (None, "expected_version", "one"),
        "wrong_value": ("payload", "kind", "robot"),
        "neighbor": ("payload", "answers_ref", "inquiry-1"),
    },
    {
        "name": "object.drop",
        "valid": {
            "op_id": "object-drop",
            "kind": "object",
            "action": "drop",
            "target_ref": "wake:s1",
        },
        "missing": (None, "target_ref"),
        "wrong_type": (None, "target_ref", 7),
        "wrong_value": (None, "action", "explode"),
        "neighbor": (None, "payload", {}),
    },
    {
        "name": "assertion.create",
        "valid": {
            "op_id": "assertion-create",
            "kind": "assertion",
            "action": "create",
            "payload": {
                "subject_ref": "chain-wake:s1",
                "predicate": "related_to",
                "object_ref": "object-1",
                "epistemic_role": "fact",
            },
        },
        "missing": ("payload", "predicate"),
        "wrong_type": ("payload", "confidence", "high"),
        "wrong_value": ("payload", "epistemic_role", "certain"),
        "neighbor": ("payload", "supersedes_ref", "assertion-old"),
    },
    {
        "name": "assertion.update",
        "valid": {
            "op_id": "assertion-update",
            "kind": "assertion",
            "action": "update",
            "target_ref": "wake:a1",
            "expected_version": 1,
            "payload": {"confidence": 0.75, "answers_ref": "inquiry-1"},
        },
        "missing": (None, "payload"),
        "wrong_type": ("payload", "confidence", "high"),
        "wrong_value": ("payload", "epistemic_role", "certain"),
        "neighbor": ("payload", "subject_ref", "other-subject"),
    },
    {
        "name": "assertion.drop",
        "valid": {
            "op_id": "assertion-drop",
            "kind": "assertion",
            "action": "drop",
            "target_ref": "wake:a1",
        },
        "missing": (None, "target_ref"),
        "wrong_type": (None, "target_ref", 7),
        "wrong_value": (None, "action", "explode"),
        "neighbor": (None, "payload", {"confidence": 0.4}),
    },
    {
        "name": "assertion.supersede",
        "valid": {
            "op_id": "assertion-supersede",
            "kind": "assertion",
            "action": "supersede",
            "payload": {
                "subject_ref": "chain-wake:s1",
                "predicate": "score",
                "literal": "3-1",
                "epistemic_role": "fact",
                "supersedes_ref": "assertion-old",
            },
        },
        "missing": ("payload", "supersedes_ref"),
        "wrong_type": ("payload", "evidence", "observation-1"),
        "wrong_value": ("payload", "epistemic_role", "certain"),
        "neighbor": (None, "target_ref", "assertion-old"),
    },
    {
        "name": "inquiry.create",
        "valid": {
            "op_id": "inquiry-create",
            "kind": "inquiry",
            "action": "create",
            "payload": {
                "subject_ref": "chain-wake:s1",
                "prompt": "Why?",
                "rationale": "Open question",
                "kind": "factual",
                "deepens_ref": "inquiry-root",
            },
        },
        "missing": ("payload", "prompt"),
        "wrong_type": ("payload", "prompt", 7),
        "wrong_value": ("payload", "kind", "causal"),
        "neighbor": ("payload", "answers_ref", "assertion-1"),
    },
    {
        "name": "inquiry.resolve",
        "valid": {
            "op_id": "inquiry-resolve",
            "kind": "inquiry",
            "action": "resolve",
            "target_ref": "inquiry-1",
            "payload": {"expected_version": 1, "answers_ref": "assertion-1"},
        },
        "missing": ("payload", "answers_ref"),
        "wrong_type": ("payload", "expected_version", "one"),
        "wrong_value": ("payload", "expected_version", 0),
        "neighbor": ("payload", "deepens_ref", "inquiry-root"),
    },
    {
        "name": "inquiry.drop",
        "valid": {
            "op_id": "inquiry-drop",
            "kind": "inquiry",
            "action": "drop",
            "target_ref": "wake:i1",
        },
        "missing": (None, "target_ref"),
        "wrong_type": (None, "target_ref", 7),
        "wrong_value": (None, "action", "deepen"),
        "neighbor": (None, "payload", {"deepens_ref": "inquiry-root"}),
    },
)


def _schema() -> dict:
    tools = WorldTools(store=WorldStore(":memory:"), adapters={})
    return next(
        item["function"]["parameters"]
        for item in tools.schemas()
        if item["function"]["name"] == "graph_patch"
    )


def _remove(item: dict, path: tuple[str | None, str]) -> dict:
    changed = deepcopy(item)
    parent, field = path
    (changed if parent is None else changed[parent]).pop(field)
    return changed


def _replace(item: dict, change: tuple[str | None, str, object]) -> dict:
    changed = deepcopy(item)
    parent, field, value = change
    (changed if parent is None else changed[parent])[field] = value
    return changed


@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda item: item["name"])
def test_graph_patch_schema_accepts_each_canonical_variant(variant) -> None:
    validator = jsonschema.Draft202012Validator(_schema())
    assert list(validator.iter_errors({"items": [variant["valid"]]})) == []


def test_graph_patch_assertion_schema_exposes_event_or_direct_edge_decision() -> None:
    """Argument-generation fields carry the same semantic topology boundary."""
    schema = _schema()
    variants = schema["properties"]["items"]["items"]["oneOf"]
    assertion_create = next(
        variant
        for variant in variants
        if variant["properties"]["kind"].get("const") == "assertion"
        and variant["properties"]["action"].get("const") == "create"
    )
    payload = assertion_create["properties"]["payload"]

    assert "bounded occurrence" in payload["description"]
    assert (
        "instead of connecting the participants directly"
        in payload["properties"]["object_ref"]["description"]
    )
    qualifier_description = payload["properties"]["qualifiers"]["description"]
    assert "date, result, or participant list" in qualifier_description


@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda item: item["name"])
@pytest.mark.parametrize(
    "case",
    ("missing", "unknown", "wrong_type", "wrong_value", "neighbor"),
)
def test_graph_patch_schema_rejects_every_variant_near_miss(variant, case) -> None:
    item = deepcopy(variant["valid"])
    if case == "missing":
        item = _remove(item, variant[case])
    elif case == "unknown":
        parent = "payload" if "payload" in item else None
        item = _replace(item, (parent, "unknown_semantic_field", "must reject"))
    else:
        item = _replace(item, variant[case])
    validator = jsonschema.Draft202012Validator(_schema())
    assert list(validator.iter_errors({"items": [item]})), f"{variant['name']} accepted {case}: {item}"


@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda item: item["name"])
async def test_each_variant_rejection_has_zero_dispatch_and_sqlite_side_effects(
    tmp_path: Path, monkeypatch, variant
) -> None:
    invalid = deepcopy(variant["valid"])
    parent = "payload" if "payload" in invalid else None
    invalid = _replace(invalid, (parent, "unknown_semantic_field", "must reject"))
    store = WorldStore(tmp_path / f"{variant['name'].replace('.', '-')}.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="strict-thread",
        wake_id="strict-wake",
    )
    implementation_calls: list[str] = []
    original_execute = WorldTools.execute

    async def recording_execute(self, name, arguments, call_id):
        implementation_calls.append(name)
        return await original_execute(self, name, arguments, call_id)

    monkeypatch.setattr(WorldTools, "execute", recording_execute)
    model = _ScriptedModel(
        [
            _response(
                "invalid near-miss",
                NativeToolCall(
                    id="invalid-patch",
                    name="graph_patch",
                    arguments={"items": [invalid]},
                ),
            )
        ]
    )
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="strict-thread",
        wake_id="strict-wake",
        max_turns=1,
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Reject invalid patch."}]})

    assert result["terminal_status"] == "staged_unpublished"
    assert implementation_calls == []
    with store.read_connection() as connection:
        for table in (
            "staged_objects",
            "staged_assertions",
            "staged_inquiries",
            "staged_patch_receipts",
            "objects",
            "assertions",
            "inquiries",
            "commit_receipts",
            "finalize_receipts",
            "world_audit",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def _xor_both_item(op_id: str) -> dict:
    return {
        "op_id": op_id,
        "kind": "assertion",
        "action": "create",
        "payload": {
            "subject_ref": "chain-wake:s1",
            "predicate": "related_to",
            "object_ref": "object-1",
            "literal": "both-present",
            "epistemic_role": "fact",
        },
    }


def _xor_literal_item(op_id: str) -> dict:
    return {
        "op_id": op_id,
        "kind": "assertion",
        "action": "create",
        "payload": {
            "subject_ref": "chain-wake:s1",
            "predicate": "related_to",
            "literal": "literal-only",
            "epistemic_role": "fact",
        },
    }


def test_graph_patch_aggregates_every_invalid_item_in_one_rejection() -> None:
    """C-EXT-01: one call reports ALL invalid items, not just the first."""
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_arguments_violation,
    )

    violation = graph_patch_arguments_violation(
        {
            "items": [
                _xor_both_item("a1"),
                _xor_both_item("a2"),
                {"op_id": "x3", "kind": "bogus", "action": "create"},
            ]
        },
        schema=_schema(),
    )
    assert violation is not None
    assert violation.code == "invalid_tool_arguments"
    assert "3 invalid item(s)" in violation.message
    # every failing item is addressed so the model can fix the whole batch
    assert "items[0]" in violation.message
    assert "items[1]" in violation.message
    assert "items[2]" in violation.message
    assert "Unsupported Graph Patch variant 'bogus'.'create'" in violation.message
    # identical action hints are deduplicated
    assert violation.action_hint.count("exactly one of object_ref") == 1


def test_graph_patch_assertion_xor_feedback_is_specialized() -> None:
    """C-EXT-01: assertion create/supersede oneOf failures name the exact fix."""
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_item_violation,
    )

    both = graph_patch_item_violation(_xor_both_item("a1"), index=2)
    assert both is not None
    assert "provides both object_ref and literal; exactly one is required" in both.message
    assert "exactly one of object_ref" in both.action_hint

    neither = graph_patch_item_violation(
        {
            "op_id": "a2",
            "kind": "assertion",
            "action": "create",
            "payload": {"subject_ref": "chain-wake:s1", "predicate": "related_to"},
        },
        index=0,
    )
    assert neither is not None
    assert "provides neither object_ref nor literal; exactly one is required" in neither.message

    supersede = graph_patch_item_violation(
        {
            "op_id": "a3",
            "kind": "assertion",
            "action": "supersede",
            "payload": {
                "subject_ref": "chain-wake:s1",
                "predicate": "related_to",
                "supersedes_ref": "assertion-1",
                "object_ref": "object-1",
                "literal": "both",
            },
        },
        index=1,
    )
    assert supersede is not None
    assert "exactly one is required" in supersede.message

    # a non-XOR variant keeps the generic hint (no oneOf in its schema)
    object_bad = graph_patch_item_violation(
        {"op_id": "o1", "kind": "object", "action": "create"},
        index=0,
    )
    assert object_bad is not None
    assert "exactly one of object_ref" not in object_bad.action_hint


def test_inquiry_missing_fields_are_reported_together_without_assertion_xor_text() -> None:
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_item_violations,
    )

    violations = graph_patch_item_violations(
        {
            "op_id": "i1",
            "kind": "inquiry",
            "action": "create",
            "payload": {},
        },
        index=0,
    )

    fields = {violation.field for violation in violations}
    assert {
        "items[0].payload.subject_ref",
        "items[0].payload.prompt",
        "items[0].payload.rationale",
    } <= fields
    assert all("object_ref and literal" not in violation.message for violation in violations)


def test_unsupported_variant_still_reports_common_independent_fields() -> None:
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_item_violations,
    )

    violations = graph_patch_item_violations(
        {
            "op_id": 3,
            "kind": "object",
            "action": "bogus",
            "bogus": 1,
        },
        index=0,
    )

    assert {violation.field for violation in violations} == {
        "items[0].op_id",
        "items[0].action",
        "items[0].bogus",
    }


def test_supported_variant_splits_multiple_additional_properties() -> None:
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_item_violations,
    )

    violations = graph_patch_item_violations(
        {
            "op_id": "o1",
            "kind": "object",
            "action": "drop",
            "target_ref": "object-1",
            "bad0": True,
            "bad1": True,
        },
        index=0,
    )

    assert {violation.field for violation in violations} == {
        "items[0].bad0",
        "items[0].bad1",
    }


def test_graph_patch_splits_multiple_top_level_additional_properties() -> None:
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_arguments_violation,
    )

    violation = graph_patch_arguments_violation(
        {
            "items": [
                {
                    "op_id": "o1",
                    "kind": "object",
                    "action": "drop",
                    "target_ref": "object-1",
                }
            ],
            "bad0": True,
            "bad1": True,
        },
        schema=_schema(),
    )

    assert violation is not None
    assert violation.total_violations == 2
    assert {detail["field"] for detail in violation.violations} == {
        "arguments.bad0",
        "arguments.bad1",
    }
    assert violation.violations_truncated is False


def test_graph_patch_aggregate_guarantees_every_bad_item_then_marks_truncation() -> None:
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_arguments_violation,
    )

    items = [
        {
            "op_id": f"i{index}",
            "kind": "inquiry",
            "action": "create",
            "payload": {},
        }
        for index in range(20)
    ]
    violation = graph_patch_arguments_violation({"items": items}, schema=_schema())

    assert violation is not None
    assert len(violation.violations) == 32
    assert violation.total_violations > 32
    assert violation.violations_truncated is True
    fields = {detail["field"] for detail in violation.violations}
    assert all(any(field.startswith(f"items[{index}].") for field in fields) for index in range(20))


def test_graph_patch_reports_both_bad_event_times_and_reversed_interval() -> None:
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_item_violations,
    )

    malformed = graph_patch_item_violations(
        {
            "op_id": "event-bad-times",
            "kind": "object",
            "action": "create",
            "payload": {
                "canonical_name": "Bad Event",
                "kind": "event",
                "event_time_start": "bad-start",
                "event_time_end": "bad-end",
            },
        },
        index=0,
    )
    assert {violation.field for violation in malformed} >= {
        "items[0].payload.event_time_start",
        "items[0].payload.event_time_end",
    }

    reversed_interval = graph_patch_item_violations(
        {
            "op_id": "event-reversed",
            "kind": "object",
            "action": "create",
            "payload": {
                "canonical_name": "Reversed Event",
                "kind": "event",
                "event_time_start": "2026-08-24T12:00:00Z",
                "event_time_end": "2026-08-24T11:00:00Z",
            },
        },
        index=0,
    )
    assert any("must not precede" in violation.message for violation in reversed_interval)


async def test_aggregated_rejection_breaks_the_failure_chain(tmp_path: Path, monkeypatch) -> None:
    """C-EXT-01: the model sees every failing item in one rejection, fixes the
    whole batch, and the run no longer exhausts the tool-error streak
    (round-3 regression: three bad items previously cost three streak slots)."""
    subject = {
        "op_id": "obj-1",
        "kind": "object",
        "action": "create",
        "payload": {"canonical_name": "Subject One", "kind": "concept"},
    }

    def distinct_literal_item(op_id: str, literal: str) -> dict:
        # distinct literals so T2-2 exact-duplicate dedup does not reject them
        return {
            "op_id": op_id,
            "kind": "assertion",
            "action": "create",
            "payload": {
                "subject_ref": "chain-wake:s1",
                "predicate": "related_to",
                "literal": literal,
                "epistemic_role": "fact",
            },
        }

    fully_fixed = [
        subject,
        distinct_literal_item("a1", "literal-only-1"),
        distinct_literal_item("a2", "literal-only-2"),
        distinct_literal_item("a3", "literal-only-3"),
    ]

    store = WorldStore(tmp_path / "failure-chain.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="chain-thread",
        wake_id="chain-wake",
    )
    model = _ScriptedModel(
        [
            _response(
                "one valid object and three invalid assertion items",
                NativeToolCall(
                    id="bad-1",
                    name="graph_patch",
                    arguments={
                        "items": [
                            subject,
                            _xor_both_item("a1"),
                            _xor_both_item("a2"),
                            _xor_both_item("a3"),
                        ]
                    },
                ),
            ),
            _response(
                "fix every reported item",
                NativeToolCall(
                    id="good-2",
                    name="graph_patch",
                    arguments={"items": fully_fixed},
                ),
            ),
            _response("staged; stop here"),
        ]
    )
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="chain-thread",
        wake_id="chain-wake",
        max_turns=3,
    )

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Stage three assertions."}]})

    # the second model call sees EVERY failing item, not just items[0]
    second_call_messages = model.requests[1]
    tool_results = [message["content"] for message in second_call_messages if message.get("role") == "tool"]
    assert tool_results, "expected tool feedback before the second model call"
    assert "3 invalid item(s)" in tool_results[0]
    assert "items[3]" in tool_results[0]
    # the batch was fully staged; the run ends on the normal turn boundary,
    # not on the tool-error-loop limit that the old first-only reporting
    # would have exhausted
    assert result["patch_success_count"] == 4
    assert "Tool error loop limit reached" not in result["terminal_summary"]


# --- P1 audit follow-ups: semantic-rejection envelope repair ----------------
#
# The staging layer is the second validation ring: an item can pass the
# schema pre-validation yet be refused on identity, version, dependency or
# duplicate grounds. These tests pin the envelope's message/hints to the
# actual repair path (audit P1-1/P1-2/P1-4/P1-5).


def test_event_time_contract_rejects_bad_and_naive_instants_at_patch_time() -> None:
    """P1-4: the advertised '(UTC ISO)' event_time contract fails at patch
    time with the exact field named — not at finalize compile."""
    from leave_information_bubble.world.graph_patch_contract import (
        graph_patch_item_violation,
    )

    base = {
        "op_id": "et-1",
        "kind": "object",
        "action": "create",
        "payload": {"canonical_name": "Dated", "kind": "event"},
    }
    bad = graph_patch_item_violation(
        {
            **base,
            "payload": {**base["payload"], "event_time_start": "not-an-iso-datetime"},
        },
        index=0,
    )
    assert bad is not None
    assert bad.field == "items[0].payload.event_time_start"
    assert "not an ISO-8601 instant" in bad.message
    assert "2026-08-22T10:30:00Z" in bad.action_hint

    naive = graph_patch_item_violation(
        {**base, "payload": {**base["payload"], "event_time_end": "2026-08-22T10:30:00"}},
        index=1,
    )
    assert naive is not None
    assert naive.field == "items[1].payload.event_time_end"
    assert "tz-aware" in naive.message

    ok = graph_patch_item_violation(
        {
            **base,
            "payload": {**base["payload"], "event_time_start": "2026-08-22T10:30:00Z"},
        },
        index=2,
    )
    assert ok is None


async def test_version_conflict_envelope_names_current_version_and_remedy(
    tmp_path: Path,
) -> None:
    """P1-1: a version-conflicted update envelope carries the current version
    and points at expected_version — the generic 'keep a stable new op_id'
    hint used to send the model in the wrong direction."""
    store = WorldStore(tmp_path / "version-conflict.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="formal-obj",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Formal Object",
                )
            ]
        ),
        "seed-version",
    )
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="vc-thread",
        wake_id="vc-wake",
    )
    model = _ScriptedModel(
        [
            _response(
                "update with a stale expected_version",
                NativeToolCall(
                    id="stale",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "u1",
                                "kind": "object",
                                "action": "update",
                                "target_ref": "formal-obj",
                                "expected_version": 999,
                                "payload": {"canonical_name": "Revised", "kind": "concept"},
                            }
                        ]
                    },
                ),
            ),
            _response("stop"),
        ]
    )
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="vc-thread",
        wake_id="vc-wake",
        max_turns=2,
    )
    await graph.ainvoke({"messages": [{"role": "user", "content": "Update it."}]})

    second_call = model.requests[1]
    tool_results = [message["content"] for message in second_call if message.get("role") == "tool"]
    assert tool_results, "expected tool feedback before the second model call"
    assert "current version 1" in tool_results[0]
    assert "expected_version" in tool_results[0]
    assert "rejected its arguments" not in tool_results[0]


async def test_identity_candidate_exists_envelope_is_not_a_rejection(
    tmp_path: Path,
) -> None:
    """P1-1: needs_identity_resolution is a decision, not a refusal — the
    envelope must not frame it as 'was rejected', which used to make the
    model abandon the create instead of confirming distinct."""
    store = WorldStore(tmp_path / "identity.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="alpha-esports",
                    kind=ObjectKind.ORGANIZATION,
                    canonical_name="Alpha Esports",
                )
            ]
        ),
        "seed-identity",
    )
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="id-thread",
        wake_id="id-wake",
    )
    model = _ScriptedModel(
        [
            _response(
                "create a canonical-exact candidate",
                NativeToolCall(
                    id="dup",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "d1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "ALPHA ESPORTS",
                                    "kind": "organization",
                                },
                            }
                        ]
                    },
                ),
            ),
            _response("stop"),
        ]
    )
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="id-thread",
        wake_id="id-wake",
        max_turns=2,
    )
    await graph.ainvoke({"messages": [{"role": "user", "content": "Stage the org."}]})

    second_call = model.requests[1]
    tool_results = [message["content"] for message in second_call if message.get("role") == "tool"]
    assert tool_results, "expected tool feedback before the second model call"
    assert "was rejected" not in tool_results[0]
    assert "identity resolution" in tool_results[0]
    assert "confirm_distinct" in tool_results[0]


async def test_semantic_rejection_aggregates_every_problem_item(tmp_path: Path) -> None:
    """P1-2: an all-rejected batch names every problematic item — the staging
    layer gets the same aggregation the schema pre-validation has, so the
    model is not ground down one item per streak."""
    store = WorldStore(tmp_path / "semantic.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="formal-obj",
                    kind=ObjectKind.CONCEPT,
                    canonical_name="Formal Object",
                ),
                ObjectInput(
                    id="alpha-esports",
                    kind=ObjectKind.ORGANIZATION,
                    canonical_name="Alpha Esports",
                ),
            ]
        ),
        "seed-semantic",
    )
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="sem-thread",
        wake_id="sem-wake",
    )
    model = _ScriptedModel(
        [
            _response(
                "three semantically invalid items",
                NativeToolCall(
                    id="bad",
                    name="graph_patch",
                    arguments={
                        "items": [
                            {
                                "op_id": "v1",
                                "kind": "object",
                                "action": "update",
                                "target_ref": "formal-obj",
                                "expected_version": 999,
                                "payload": {"canonical_name": "Revised", "kind": "concept"},
                            },
                            {
                                "op_id": "i1",
                                "kind": "object",
                                "action": "create",
                                "payload": {
                                    "canonical_name": "ALPHA ESPORTS",
                                    "kind": "organization",
                                },
                            },
                            {
                                "op_id": "a1",
                                "kind": "assertion",
                                "action": "create",
                                "payload": {
                                    "subject_ref": "ghost-object",
                                    "predicate": "related_to",
                                    "literal": "x",
                                    "epistemic_role": "fact",
                                },
                            },
                        ]
                    },
                ),
            ),
            _response("stop"),
        ]
    )
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="sem-thread",
        wake_id="sem-wake",
        max_turns=2,
    )
    await graph.ainvoke({"messages": [{"role": "user", "content": "Stage the batch."}]})

    second_call = model.requests[1]
    tool_results = [message["content"] for message in second_call if message.get("role") == "tool"]
    assert tool_results, "expected tool feedback before the second model call"
    assert "3 problematic item(s)" in tool_results[0]
    assert "items[0] version_conflict (current version 1)" in tool_results[0]
    assert "items[1] identity_candidate_exists" in tool_results[0]
    assert "items[2] dependency_unavailable" in tool_results[0]
    # the actionable envelope anchors on the first item's real repair path
    assert "expected_version" in tool_results[0]


async def test_memory_changes_naive_since_rejected(tmp_path: Path) -> None:
    """P1-3: a naive since instant is rejected instead of being silently
    shifted by the server's local zone."""
    store = WorldStore(tmp_path / "changes.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="ch-thread",
        wake_id="ch-wake",
    )
    naive = await tools.execute("memory_changes", {"since": "2026-08-22T10:30:00"}, "call-naive")
    assert naive.get("ok") is False
    assert "invalid_arguments" in naive["limitations"]

    aware = await tools.execute("memory_changes", {"since": "2026-08-22T10:30:00Z"}, "call-aware")
    assert aware.get("ok") is True


async def test_dead_cursor_wrap_hint_is_directional(tmp_path: Path) -> None:
    """P1-5: a dead cursor gets a state-level hint (drop the cursor) instead
    of the schema-correction hint, which would send the model in circles on
    an already-valid call."""
    store = WorldStore(tmp_path / "cursor.sqlite3")
    tools = WorldTools(
        store=store,
        adapters={},
        thread_id="cur-thread",
        wake_id="cur-wake",
    )
    model = _ScriptedModel(
        [
            _response(
                "page with a dead cursor",
                NativeToolCall(
                    id="cur",
                    name="memory_search",
                    arguments={"cursor": "dead-token"},
                ),
            ),
            _response("stop"),
        ]
    )
    graph = build_world_agent_graph(
        model=model,
        tools=tools,
        store=store,
        thread_id="cur-thread",
        wake_id="cur-wake",
        max_turns=2,
    )
    await graph.ainvoke({"messages": [{"role": "user", "content": "Page results."}]})

    second_call = model.requests[1]
    tool_results = [message["content"] for message in second_call if message.get("role") == "tool"]
    assert tool_results, "expected tool feedback before the second model call"
    assert "invalid_cursor" in tool_results[0]
    assert "drop it and re-issue" in tool_results[0]
