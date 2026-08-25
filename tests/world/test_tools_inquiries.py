# Split from tests/world/test_tools.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

from datetime import timedelta

import pytest

from leave_information_bubble.runtime.inquiry_lease import InquiryLeaseStore
from leave_information_bubble.world import (
    CognitiveDelta,
    InquiryInput,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)
from leave_information_bubble.world import (
    tools as world_tools,
)
from leave_information_bubble.world.store import observation_id
from leave_information_bubble.world.tools import WorldTools
from tests.world._tools_helpers import (
    NOW,
    SOURCE,
    _tools,
)


def _inquiry_seed(store: WorldStore) -> None:
    store.memory_commit(
        CognitiveDelta(
            objects=[
                ObjectInput(
                    id="obj-anchor-1",
                    kind=ObjectKind.ENTITY,
                    canonical_name="Anchor",
                )
            ],
            inquiries=[
                InquiryInput(
                    id="inq-open",
                    subject_id="obj-anchor-1",
                    prompt="What remains unclear?",
                    rationale="Lease feedback fixture",
                )
            ],
        ),
        "inquiry-seed-1",
    )


async def test_inquiry_lease_reports_occupied_and_invalid_release_token(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Lease conflicts and invalid tokens are failures with distinct factual causes."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _inquiry_seed(store)
    leases = InquiryLeaseStore(tmp_path / "runtime.sqlite3")
    assert leases.claim("inq-open", "first-owner", timedelta(minutes=15)) is not None
    tools = WorldTools(store=store, adapters={}, leases=leases)

    occupied = await tools.execute(
        "claim_inquiry",
        {"inquiry_id": "inq-open", "owner_id": "second-owner"},
        "occupied",
    )
    invalid_release = await tools.execute(
        "release_inquiry",
        {"lease_token": "not-a-real-token"},
        "bad-release",
    )

    assert occupied == {
        "ok": False,
        "limitations": ["inquiry_lease_occupied"],
        "error": {
            "code": "inquiry_lease_occupied",
            "field": "inquiry_id",
            "same_request_can_change": True,
            "change_condition": "current_lease_expires_or_is_released",
        },
    }
    assert invalid_release == {
        "ok": False,
        "limitations": ["invalid_lease_token"],
        "error": {
            "code": "invalid_lease_token",
            "field": "lease_token",
            "same_request_can_change": False,
            "change_condition": "a_valid_active_lease_token_is_supplied",
        },
    }


@pytest.mark.parametrize(
    ("arguments", "code", "field", "change_condition"),
    [
        (
            {"inquiry_id": "missing", "owner_id": "agent"},
            "unknown_or_closed_inquiry_id",
            "inquiry_id",
            "an_open_or_dormant_inquiry_with_this_id_exists",
        ),
        (
            {"inquiry_id": "inq-open", "owner_id": ""},
            "invalid_lease_field",
            "owner_id",
            "request_field_is_corrected",
        ),
        (
            {"inquiry_id": "inq-open", "owner_id": "agent", "ttl_seconds": 0},
            "invalid_lease_field",
            "ttl_seconds",
            "request_field_is_corrected",
        ),
    ],
)
async def test_claim_inquiry_reports_exact_field_failure(
    tmp_path: pytest.TempPathFactory,
    arguments: dict[str, object],
    code: str,
    field: str,
    change_condition: str,
) -> None:
    """Claim failures distinguish durable-id state from malformed request fields."""
    store = WorldStore(tmp_path / f"{field}.sqlite3")
    _inquiry_seed(store)
    tools = WorldTools(
        store=store,
        adapters={},
        leases=InquiryLeaseStore(tmp_path / f"{field}-runtime.sqlite3"),
    )

    result = await tools.execute("claim_inquiry", arguments, f"claim-{field}")

    assert result["ok"] is False
    assert result["limitations"] == [code]
    assert result["error"]["code"] == code
    assert result["error"]["field"] == field
    assert result["error"]["change_condition"] == change_condition


async def test_propose_inquiry_validates_and_returns_payload(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a propose_inquiry call that drops fields or invents a local_ref."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _inquiry_seed(store)
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute(
        "propose_inquiry",
        {
            "subject": {"memory_id": "obj-anchor-1"},
            "prompt": "为什么这个信号是此时出现的？",
            "rationale": "同一对象的信号已多次出现，但触发条件仍未弄清",
            "kind": "stateful",
        },
        "c1",
    )

    assert result["ok"] is True
    assert result["inquiry"] == {
        "subject": {"memory_id": "obj-anchor-1"},
        "prompt": "为什么这个信号是此时出现的？",
        "rationale": "同一对象的信号已多次出现，但触发条件仍未弄清",
        "kind": "stateful",
    }
    assert "local_ref" not in result["inquiry"]


async def test_propose_inquiry_rejects_unknown_memory_id(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a propose_inquiry anchored to an object id that never entered the store."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _inquiry_seed(store)
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute(
        "propose_inquiry",
        {"subject": {"memory_id": "no-such-object"}, "prompt": "p", "rationale": "r"},
        "c2",
    )

    assert result == {
        "ok": False,
        "limitations": ["unknown_memory_id"],
        "error": {
            "code": "unknown_memory_id",
            "field": "subject.memory_id",
            "same_request_can_change": True,
            "change_condition": "the_subject_object_exists_in_durable_memory",
        },
    }


async def test_propose_inquiry_rejects_missing_prompt(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a propose_inquiry accepted without the question text."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _inquiry_seed(store)
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute(
        "propose_inquiry",
        {"subject": {"memory_id": "obj-anchor-1"}, "rationale": "r"},
        "c3",
    )

    assert result == {
        "ok": False,
        "limitations": ["invalid_inquiry_field"],
        "error": {
            "code": "invalid_inquiry_field",
            "field": "prompt",
            "same_request_can_change": False,
            "change_condition": "request_field_is_corrected",
        },
    }


async def test_propose_inquiry_accepts_local_ref_subject_without_store_check(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch the tool requiring a store object for proposal-local subjects."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute(
        "propose_inquiry",
        {"subject": {"local_ref": "obj-x"}, "prompt": "p", "rationale": "r"},
        "c4",
    )

    assert result["ok"] is True
    assert result["inquiry"]["subject"] == {"local_ref": "obj-x"}
    assert result["status"] == "pending_terminal_subject_resolution"
    assert result["writes_store"] is False


async def test_propose_inquiry_rejects_invalid_kind(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a propose_inquiry with a kind outside the factual/semantic/stateful enum."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _inquiry_seed(store)
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute(
        "propose_inquiry",
        {
            "subject": {"memory_id": "obj-anchor-1"},
            "prompt": "p",
            "rationale": "r",
            "kind": "wishful",
        },
        "c5",
    )

    assert result == {
        "ok": False,
        "limitations": ["invalid_inquiry_field"],
        "error": {
            "code": "invalid_inquiry_field",
            "field": "kind",
            "same_request_can_change": False,
            "change_condition": "request_field_is_corrected",
        },
    }


async def test_propose_inquiry_rejects_blank_rationale(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a propose_inquiry accepted with a whitespace-only rationale."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _inquiry_seed(store)
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute(
        "propose_inquiry",
        {"subject": {"memory_id": "obj-anchor-1"}, "prompt": "p", "rationale": "   "},
        "c6",
    )

    assert result == {
        "ok": False,
        "limitations": ["invalid_inquiry_field"],
        "error": {
            "code": "invalid_inquiry_field",
            "field": "rationale",
            "same_request_can_change": False,
            "change_condition": "request_field_is_corrected",
        },
    }


async def test_propose_inquiry_rejects_malformed_subject(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a propose_inquiry whose subject is not a valid exactly-one-identity GraphRef."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute(
        "propose_inquiry",
        {"subject": {"prompt": "not a ref"}, "prompt": "p", "rationale": "r"},
        "c7",
    )

    assert result["ok"] is False
    assert result["limitations"] == ["invalid_inquiry_field"]
    assert result["error"]["code"] == "invalid_inquiry_field"
    assert result["error"]["field"].startswith("subject")


async def test_log_inquiry_point_registered_and_logs(tmp_path: pytest.TempPathFactory) -> None:
    """Catch the inquiry-point tool missing from the advertised schema set."""
    tools, _ = _tools(tmp_path, thread_id="t-1")
    schemas = {item["function"]["name"] for item in tools.schemas()}
    assert "log_inquiry_point" in schemas


async def test_log_inquiry_point_writes_curiosity(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch an inquiry-point call that confirms without appending a log row."""
    tools, _ = _tools(tmp_path, thread_id="t-1")
    log = tmp_path / "curiosity-log.jsonl"
    monkeypatch.setattr(
        "leave_information_bubble.runtime.curiosity_log.CURIOSITY_LOG_DEFAULT",
        log,
    )
    result = await tools.execute(
        "log_inquiry_point",
        {"topic": "48梗是什么意思", "source_ref": "obs-1", "reason": "社区反复使用"},
        "call-1",
    )
    assert result.get("ok") is True
    assert result["status"] == "runtime_trace_recorded"
    assert result["writes_world"] is False
    assert log.exists()
    assert "48梗是什么意思" in log.read_text(encoding="utf-8")


async def test_log_inquiry_point_missing_fields_limited(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch an inquiry-point call accepted without source_ref or reason."""
    tools, _ = _tools(tmp_path, thread_id="t-1")
    result = await tools.execute("log_inquiry_point", {"topic": "48梗是什么意思"}, "call-1")
    assert result == {
        "ok": False,
        "limitations": ["missing_required_inquiry_point_field"],
        "error": {
            "code": "missing_required_inquiry_point_field",
            "field": "source_ref,reason",
            "same_request_can_change": False,
            "change_condition": "request_field_is_corrected",
        },
    }


async def test_log_inquiry_point_blank_topic_limited(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch an inquiry-point call accepted with a whitespace-only topic."""
    tools, _ = _tools(tmp_path, thread_id="t-1")
    result = await tools.execute(
        "log_inquiry_point",
        {"topic": "   ", "source_ref": "obs-1", "reason": "r"},
        "call-1",
    )
    assert result == {
        "ok": False,
        "limitations": ["missing_required_inquiry_point_field"],
        "error": {
            "code": "missing_required_inquiry_point_field",
            "field": "topic",
            "same_request_can_change": False,
            "change_condition": "request_field_is_corrected",
        },
    }


def _merge_input(
    depth: ObservationDepth,
    excerpt: str,
    *,
    identifier: str = observation_id("replay", SOURCE),
    title: str = "",
) -> ObservationInput:
    """Build one deterministic ObservationInput for merge-depth tests."""
    return ObservationInput(
        id=identifier,
        source_uri=SOURCE,
        source_kind="replay",
        title=title,
        excerpt=excerpt,
        depth=depth,
        observed_at=NOW,
        metadata={},
    )


def test_merge_group_depth_takes_deepest_observation_not_primary() -> None:
    """Catch B2-13: a CONTENT body must not collapse to SEEN because its excerpt is shorter.

    The b8 regression scenario: a video title excerpt (74 chars) outlengthens
    the description excerpt, so primary is the SEEN metadata row; the merged
    depth must still follow the deepest observation in the group (CONTENT).
    """
    metadata = _merge_input(
        ObservationDepth.SEEN,
        "T" * 74,  # the long title makes this the primary
        title="T" * 74,
    )
    body = _merge_input(ObservationDepth.CONTENT, "Shorter description than the title.")
    merged = world_tools._merge_group([metadata, body])
    assert merged.depth == ObservationDepth.CONTENT
    assert merged.excerpt == metadata.excerpt  # primary (longest excerpt) unchanged
    assert merged.title == metadata.title
    assert merged.id == observation_id("replay", SOURCE)


def test_merge_group_depth_takes_deepest_when_primary_order_flipped() -> None:
    """Catch B2-13 ordering: CONTENT wins even when it appears before the longer SEEN row."""
    body = _merge_input(ObservationDepth.CONTENT, "Shorter description.")
    metadata = _merge_input(
        ObservationDepth.SEEN,
        "T" * 74,
        title="T" * 74,
    )
    merged = world_tools._merge_group([body, metadata])
    assert merged.depth == ObservationDepth.CONTENT
    assert merged.excerpt == metadata.excerpt


def test_merge_group_all_seen_stays_seen() -> None:
    """Acceptance: a group with no deeper read must not upgrade past SEEN."""
    first = _merge_input(ObservationDepth.SEEN, "short")
    second = _merge_input(ObservationDepth.SEEN, "T" * 74, title="T" * 74)
    merged = world_tools._merge_group([first, second])
    assert merged.depth == ObservationDepth.SEEN
    assert merged.excerpt == second.excerpt


def test_merge_observations_takes_group_max_without_downgrading() -> None:
    """Acceptance: mixed SEEN+CONTENT+DISCUSSION folds to the deepest (max semantics)."""
    main = [
        _merge_input(ObservationDepth.SEEN, "T" * 74, title="T" * 74),
        _merge_input(ObservationDepth.DISCUSSION, "mid-depth row"),
        _merge_input(ObservationDepth.CONTENT, "body row"),
    ]
    merged = world_tools._merge_observations(main)
    assert len(merged) == 1
    assert merged[0].depth == ObservationDepth.DISCUSSION
    # a distinct-id row in the same batch is untouched by the fold
    assert merged[0].id == observation_id("replay", SOURCE)


def test_merge_group_single_item_keeps_its_depth() -> None:
    """Edge: a one-element group merges to itself (no phantom upgrade)."""
    seen = _merge_input(ObservationDepth.SEEN, "lone seen row")
    assert world_tools._merge_group([seen]).depth == ObservationDepth.SEEN
    content = _merge_input(ObservationDepth.CONTENT, "lone content row")
    assert world_tools._merge_group([content]).depth == ObservationDepth.CONTENT


def test_merge_group_empty_excerpts_still_take_deepest_depth() -> None:
    """Edge: depth must not depend on excerpt length, including empty excerpts."""
    seen = _merge_input(ObservationDepth.SEEN, "")
    content = _merge_input(ObservationDepth.CONTENT, "")
    merged = world_tools._merge_group([seen, content])
    assert merged.depth == ObservationDepth.CONTENT
    assert merged.excerpt == ""  # primary selection unaffected by the depth fix


def test_merge_group_same_depth_merge_keeps_that_depth() -> None:
    """Edge: merging same-depth rows must not shift the depth."""
    first = _merge_input(ObservationDepth.CONTENT, "a short body")
    second = _merge_input(ObservationDepth.CONTENT, "T" * 74, title="T" * 74)
    assert world_tools._merge_group([first, second]).depth == ObservationDepth.CONTENT
