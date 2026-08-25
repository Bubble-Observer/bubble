"""Bounded graph vocabulary: normalization, alias operations, stable feedback codes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from leave_information_bubble.world.contracts import (
    AliasOperation,
    AssertionInput,
    CognitiveDelta,
    ObjectInput,
    ObjectKind,
)
from leave_information_bubble.world.graph_contract import (
    AliasAction,
    normalize_identity_alias,
    normalize_qualifiers,
    normalize_type_key,
)
from leave_information_bubble.world.proposal import (
    AssertionProposal,
    EpistemicRole,
    GraphRef,
    NewObjectProposal,
    ReviewIssue,
    ReviewIssueCode,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Foo  Bar ", "foo bar"),
        ("ＦＯＯ", "ｆｏｏ"),
        ("é", "é"),
    ],
)
def test_normalize_identity_alias_preserves_word_boundaries(raw: str, expected: str) -> None:
    assert normalize_identity_alias(raw) == expected


def test_normalize_identity_alias_rejects_non_string_or_empty() -> None:
    with pytest.raises(ValueError):
        normalize_identity_alias("   ")
    with pytest.raises(ValueError):
        normalize_identity_alias(42)


def test_type_key_and_qualifiers_are_bounded() -> None:
    assert normalize_type_key("  Acquisition.Deal ") == "acquisition.deal"
    assert normalize_qualifiers({"community": "  forum-a  "}) == {"community": "forum-a"}
    with pytest.raises(ValueError):
        normalize_type_key("contains spaces")
    with pytest.raises(ValueError):
        normalize_qualifiers({"unknown": "x"})
    with pytest.raises(ValueError):
        normalize_qualifiers({"scope": {"nested": True}})


def test_alias_operation_rejects_host_or_agent_supplied_commit_id() -> None:
    with pytest.raises(ValidationError):
        AliasOperation.model_validate(
            {
                "object_id": "object-1",
                "raw_alias": "Acme",
                "normalized_alias": "acme",
                "action": "add",
                "commit_id": "forged",
            }
        )


def test_alias_operation_normalized_alias_must_match_shared_normalizer() -> None:
    operation = AliasOperation(
        object_id="object-1",
        raw_alias="  Acme  Corp ",
        normalized_alias="acme corp",
        action=AliasAction.ADD,
    )
    assert CognitiveDelta(alias_operations=[operation]).alias_operations == [operation]
    with pytest.raises(ValidationError):
        AliasOperation(
            object_id="object-1",
            raw_alias="Acme",
            normalized_alias="agent-chosen-form",
            action=AliasAction.ADD,
        )


@pytest.mark.parametrize(
    ("value", "member"),
    [
        ("identity_alias_claim_conflict", ReviewIssueCode.IDENTITY_ALIAS_CLAIM_CONFLICT),
        ("ambiguous_name_candidates", ReviewIssueCode.AMBIGUOUS_NAME_CANDIDATES),
        ("duplicate_object_candidate", ReviewIssueCode.DUPLICATE_OBJECT_CANDIDATE),
        ("duplicate_event_candidate", ReviewIssueCode.DUPLICATE_EVENT_CANDIDATE),
        ("invalid_reference", ReviewIssueCode.INVALID_REFERENCE),
        ("possible_cognition_conflict", ReviewIssueCode.POSSIBLE_COGNITION_CONFLICT),
        ("unsupported_object_kind", ReviewIssueCode.UNSUPPORTED_OBJECT_KIND),
        ("alias_operation_invalid", ReviewIssueCode.ALIAS_OPERATION_INVALID),
        ("event_participant_incomplete", ReviewIssueCode.EVENT_PARTICIPANT_INCOMPLETE),
    ],
)
def test_graph_review_issue_codes_are_stable(value: str, member: ReviewIssueCode) -> None:
    """The nine graph-contract feedback codes stay durable string members."""
    assert ReviewIssueCode(value) is member
    assert member.value == value
    assert len(ReviewIssueCode) == 27
    assert ReviewIssueCode("stale_version") is ReviewIssueCode.STALE_VERSION


def test_review_issue_candidate_contract_is_typed() -> None:
    """match_basis and omitted_dependencies serialize; unknown fields stay forbidden."""
    issue = ReviewIssue(
        issue_id="issue-1",
        code=ReviewIssueCode.AMBIGUOUS_NAME_CANDIDATES,
        severity="warning",
        failed_rule="graph-vocabulary-basis",
        actual_value={"candidates": ["alex-1", "alex-2"]},
        item_kind="object",
        message="several stored objects match the proposed name",
        match_basis=["exact_name_match"],
        omitted_dependencies=[{"object_id": "alex-2", "reason": "ambiguous_candidate"}],
    )
    dumped = issue.model_dump(mode="json")
    assert dumped["match_basis"] == ["exact_name_match"]
    assert dumped["omitted_dependencies"] == [
        {"object_id": "alex-2", "reason": "ambiguous_candidate"}
    ]
    assert ReviewIssue.model_validate(dumped) == issue
    with pytest.raises(ValidationError):
        ReviewIssue.model_validate({**dumped, "commit_id": "forged"})


def test_proposal_and_store_contracts_route_through_shared_normalizers() -> None:
    """type_key and qualifiers run the same bounded normalizer on both sides."""
    proposed_object = NewObjectProposal(
        local_ref="acme",
        kind=ObjectKind.ENTITY,
        canonical_name="Acme",
        type_key=" Acquisition.Deal ",
    )
    assert proposed_object.type_key == "acquisition.deal"
    assert (
        ObjectInput(
            id="object-1",
            kind=ObjectKind.ENTITY,
            canonical_name="Acme",
            type_key=" Acquisition.Deal ",
        ).type_key
        == "acquisition.deal"
    )
    proposed_assertion = AssertionProposal(
        subject=GraphRef(local_ref="acme"),
        predicate="has_ticker",
        literal="ACME",
        epistemic_role=EpistemicRole.FACT,
        confidence=0.9,
        qualifiers={"scope": "  trading "},
    )
    assert proposed_assertion.qualifiers == {"scope": "trading"}
    assert (
        AssertionInput(
            id="assertion-1",
            subject_id="object-1",
            predicate="has_ticker",
            literal="ACME",
            epistemic_role=EpistemicRole.FACT,
            confidence=0.9,
            qualifiers={"scope": "  trading "},
        ).qualifiers
        == {"scope": "trading"}
    )
    with pytest.raises(ValidationError):
        AssertionProposal(
            subject=GraphRef(local_ref="acme"),
            predicate="has_ticker",
            literal="ACME",
            epistemic_role=EpistemicRole.FACT,
            confidence=0.9,
            qualifiers={"unknown": "x"},
        )
