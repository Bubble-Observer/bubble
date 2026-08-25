"""Agent-facing graph delta proposal contracts with no storage effects."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .contracts import (
    CommitReceipt,
    EpistemicRole,
    EventTimePrecision,
    EvidenceInput,
    JsonValue,
    ObjectKind,
    WorldModel,
)
from .graph_contract import (
    QUALIFIER_KEYS,
    TYPE_KEY_PATTERN,
    normalize_qualifiers,
    normalize_type_key,
)
from .graph_contract_text import IDENTITY_MODEL_SENTENCE


class ProposalModel(WorldModel):
    """Frozen proposal model that rejects fields outside the public schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ReviewOutcome(StrEnum):
    """The durable review result for one proposal attempt."""

    ACCEPT = "ACCEPT"
    COMMIT_WITH_WARNINGS = "COMMIT_WITH_WARNINGS"
    REJECT_AND_REPAIR = "REJECT_AND_REPAIR"


class ReviewIssueCode(StrEnum):
    """Stable, machine-actionable review rule identifiers."""

    SEEN_ONLY_SUPPORT = "seen_only_support"
    COMMENT_ONLY_FACT = "comment_only_fact"
    MISSING_EVIDENCE_ID = "missing_evidence_id"
    DUPLICATE_INQUIRY = "duplicate_inquiry"
    STALE_RESOLUTION = "stale_resolution"
    MISSING_RESOLUTION = "missing_resolution"
    STALE_OBJECT_UPDATE = "stale_object_update"
    MISSING_OBSERVATION_LINK = "missing_observation_link"
    UNKNOWN_MEMORY_ID = "unknown_memory_id"
    UNKNOWN_LOCAL_REFERENCE = "unknown_local_reference"
    ALIAS_COLLISION = "alias_collision"
    STALE_VERSION = "stale_version"
    UNKNOWN_ISSUE_ID = "unknown_issue_id"
    EMPTY_AMENDMENT = "empty_amendment"
    APPROVED_ITEM_RESUBMITTED = "approved_item_resubmitted"
    LOW_RELIABILITY_FACT = "low_reliability_fact"
    MIXED_RELIABILITY_PRIMARY = "mixed_reliability_primary"
    NO_VALID_EVIDENCE = "no_valid_evidence"
    # Durable cognition graph contract (Task 1.1): stable feedback codes for
    # the later convergence slices; never reorder or revalue historical codes.
    IDENTITY_ALIAS_CLAIM_CONFLICT = "identity_alias_claim_conflict"
    AMBIGUOUS_NAME_CANDIDATES = "ambiguous_name_candidates"
    DUPLICATE_OBJECT_CANDIDATE = "duplicate_object_candidate"
    DUPLICATE_EVENT_CANDIDATE = "duplicate_event_candidate"
    INVALID_REFERENCE = "invalid_reference"
    POSSIBLE_COGNITION_CONFLICT = "possible_cognition_conflict"
    UNSUPPORTED_OBJECT_KIND = "unsupported_object_kind"
    ALIAS_OPERATION_INVALID = "alias_operation_invalid"
    EVENT_PARTICIPANT_INCOMPLETE = "event_participant_incomplete"


class ReviewIssue(ProposalModel):
    """One precise, durable, repairable proposal-review finding."""

    issue_id: str
    code: ReviewIssueCode
    severity: Literal["warning", "error"]
    failed_rule: str
    actual_value: JsonValue
    item_kind: str
    item_index: int | None = Field(default=None, ge=0)
    durable_id: str | None = None
    message: str
    suggested_actions: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    match_basis: list[str] = Field(default_factory=list)
    omitted_dependencies: list[dict[str, JsonValue]] = Field(default_factory=list)
    dropped_evidence_ids: list[str] = Field(default_factory=list)


class AttemptContext(ProposalModel):
    """Identity of one append-only review attempt within a logical run."""

    run_commit_id: str
    attempt_id: str
    attempt_no: int = Field(ge=1)
    parent_attempt_id: str | None = None
    addresses_issue_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_attempt_context(self) -> AttemptContext:
        _require_text(self.run_commit_id, "run commit id")
        _require_text(self.attempt_id, "attempt id")
        if self.parent_attempt_id is not None:
            _require_text(self.parent_attempt_id, "parent attempt id")
        _require_unique(self.addresses_issue_ids, "addressed issue id")
        return self


class GraphRef(ProposalModel):
    """Reference either a proposal-local declaration or durable world memory."""

    local_ref: str | None = None
    memory_id: str | None = None

    @model_validator(mode="after")
    def _validate_identity(self) -> GraphRef:
        if (self.local_ref is None) == (self.memory_id is None):
            raise ValueError("graph reference requires exactly one identity")
        identity = self.local_ref if self.local_ref is not None else self.memory_id
        _require_text(identity, "graph reference identity")
        return self


class EventParticipantProposal(ProposalModel):
    """One explicit participant declaration on a new event object.

    The compiler expands only this explicit declaration into a
    ``has_participant`` assertion; it never infers participants from titles,
    literals or other assertions. ``role`` folds into the compiled assertion's
    ``qualifiers["role"]`` when non-empty; a role conflicting with
    ``qualifiers.role`` is rejected at the proposal boundary instead of being
    silently resolved.
    """

    object: GraphRef
    role: str | None = Field(
        default=None,
        description=(
            "Participant role in this event, at most 120 characters; folds into "
            "the compiled assertion's qualifiers['role'] when non-empty."
        ),
    )
    qualifiers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Bounded qualifiers (role, language, community, scope, granularity) "
            "narrowing how this participant relation should be interpreted; "
            "normalized by the shared graph vocabulary."
        ),
    )
    epistemic_role: EpistemicRole
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceInput] = Field(default_factory=list)

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not trimmed:
            # an empty role is the same as no role at all
            return None
        if len(trimmed) > 120:
            raise ValueError("participant role must not exceed 120 characters")
        return trimmed

    @field_validator("qualifiers", mode="before")
    @classmethod
    def _normalize_qualifiers(cls, value: object) -> object:
        return normalize_qualifiers(value)

    @model_validator(mode="after")
    def _validate_participant(self) -> EventParticipantProposal:
        if (
            self.role is not None
            and "role" in self.qualifiers
            and self.role != self.qualifiers["role"]
        ):
            raise ValueError("participant role conflicts with qualifiers.role")
        return self


class NewObjectProposal(ProposalModel):
    """A new reusable Object identified only within this proposal."""

    local_ref: str
    kind: ObjectKind
    type_key: str | None = Field(
        default=None,
        description=(
            "A type_key is a normalized coarse subtype — person, organization, "
            "match, acquisition, rule — never an identity key."
        ),
    )
    canonical_name: str
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Initial identity aliases, normalized before comparison; only an "
            "active identity alias is world-unique."
        ),
    )
    domain_hints: list[str] = Field(default_factory=list)
    provisional: bool = False
    event_time_start: datetime | None = None
    event_time_end: datetime | None = None
    event_time_precision: EventTimePrecision = "unknown"
    participants: list[EventParticipantProposal] = Field(
        default_factory=list,
        description=(
            "Event-only: the explicit participant declarations compiled into "
            "has_participant assertions. Every participant's role, qualifiers, "
            "epistemic role, confidence and evidence is Agent-submitted "
            "metadata; the host never infers participants or their metadata "
            "from titles, literals or other assertions. A non-provisional "
            "event without participants commits with an incomplete warning."
        ),
    )

    @field_validator("type_key", mode="before")
    @classmethod
    def _normalize_type_key(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return normalize_type_key(value)

    @model_validator(mode="after")
    def _validate_object(self) -> NewObjectProposal:
        _require_text(self.local_ref, "object local reference")
        _require_text(self.canonical_name, "object canonical name")
        _validate_interval(self.event_time_start, self.event_time_end)
        if self.participants and self.kind is not ObjectKind.EVENT:
            raise ValueError("only event objects may declare participants")
        return self


class ObjectUpdateProposal(ProposalModel):
    """A narrow versioned refinement of an existing Object.

    Identity alias corrections (Task 1.5): ``add_aliases`` grows the current
    identity alias index, ``remove_aliases`` retires an alias from active
    identity, and ``demote_aliases`` marks an alias as no longer identity-
    relevant (same persisted state as remove; the action difference lives in
    the delta audit). The same normalized alias may not appear in more than
    one action of one update.
    """

    target: GraphRef
    add_aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Aliases to grow this object's active identity index with; "
            "normalized by the shared alias vocabulary and world-unique "
            "while active."
        ),
    )
    remove_aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Aliases to retire from active identity; the same normalized "
            "alias may appear in at most one action of one update."
        ),
    )
    demote_aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Aliases to mark as no longer identity-relevant; recorded as "
            "name usage. Same persisted state as remove; the action "
            "difference lives in the delta audit."
        ),
    )
    provisional: bool | None = None
    expected_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _require_memory_target(self) -> ObjectUpdateProposal:
        if self.target.memory_id is None:
            raise ValueError("object updates require a durable memory reference")
        return self


_ASSERTION_TARGET_SCHEMA = {
    "oneOf": [
        {
            "required": ["object"],
            "properties": {
                "object": {"not": {"type": "null"}},
                "literal": {"type": "null"},
            },
        },
        {
            "required": ["literal"],
            "properties": {
                "object": {"type": "null"},
                "literal": {"not": {"type": "null"}},
            },
        },
    ]
}


class AssertionProposal(ProposalModel):
    """A revisable judgment proposed for durable world cognition."""

    # Keep this to simple JSON Schema composition supported by OpenAI-compatible
    # tool providers.  The Pydantic validator below remains the runtime source
    # of truth; this makes the same mutual exclusion visible to the model.
    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=_ASSERTION_TARGET_SCHEMA)

    subject: GraphRef
    predicate: str = Field(
        description=(
            "A compact relationship or property label, not the complete claim body. "
            "The target belongs in exactly one of object or literal."
        )
    )
    object: GraphRef | None = Field(
        default=None,
        description=(
            "Object-valued target when the assertion relates its subject to another "
            "existing or newly declared Object. Use exactly one of object or literal, "
            "never both."
        ),
    )
    literal: JsonValue | None = Field(
        default=None,
        description=(
            "Literal-valued target for claimed text, number, boolean, or structured value. "
            "Use exactly one of object or literal, never both."
        ),
    )
    epistemic_role: EpistemicRole = Field(
        description="How the Agent characterizes this revisable cognition, not a host truth label."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The Agent's confidence in this judgment, preserved without host rewriting.",
    )
    evidence: list[EvidenceInput] = Field(
        default_factory=list,
        description=(
            "Optional links to real observations. Their supports/context/contradicts roles "
            "record how the Agent used material, not a host-certified evidence grade. "
            "A judgment may honestly have no valid links."
        ),
    )
    qualifiers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Bounded qualifiers (role, language, community, scope, granularity) "
            "narrowing how this assertion should be interpreted; normalized by the "
            "shared graph vocabulary."
        ),
    )
    event_time_start: datetime | None = None
    event_time_end: datetime | None = None
    event_time_precision: EventTimePrecision = "unknown"
    supersedes_id: str | None = None
    supersede_reason: str | None = None
    answers_inquiry_id: str | None = None

    @field_validator("qualifiers", mode="before")
    @classmethod
    def _normalize_qualifiers(cls, value: object) -> object:
        return normalize_qualifiers(value)

    @model_validator(mode="after")
    def _validate_assertion(self) -> AssertionProposal:
        _require_text(self.predicate, "assertion predicate")
        if (self.object is None) == (self.literal is None):
            raise ValueError("assertion requires exactly one of object or literal")
        _validate_interval(self.event_time_start, self.event_time_end)
        if self.supersedes_id is not None:
            _require_text(self.supersedes_id, "superseded assertion id")
        if self.supersede_reason is not None:
            _require_text(self.supersede_reason, "supersede reason")
        if self.answers_inquiry_id is not None:
            _require_text(self.answers_inquiry_id, "answers inquiry id")
        return self


class NewInquiryProposal(ProposalModel):
    """A useful unresolved question anchored to a world Object."""

    local_ref: str
    subject: GraphRef
    prompt: str
    rationale: str
    kind: Literal["factual", "semantic", "stateful"] = "factual"
    deepens_inquiry_id: str | None = None
    answers_inquiry_id: str | None = None

    @model_validator(mode="after")
    def _validate_inquiry(self) -> NewInquiryProposal:
        _require_text(self.local_ref, "inquiry local reference")
        _require_text(self.prompt, "inquiry prompt")
        _require_text(self.rationale, "inquiry rationale")
        if self.deepens_inquiry_id is not None:
            _require_text(self.deepens_inquiry_id, "deepened inquiry id")
        if self.answers_inquiry_id is not None:
            _require_text(self.answers_inquiry_id, "answers inquiry id")
        return self


class InquiryResolutionProposal(ProposalModel):
    """A version-checked request to retire an active Inquiry."""

    memory_id: str
    expected_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_resolution(self) -> InquiryResolutionProposal:
        _require_text(self.memory_id, "inquiry memory id")
        return self


class ObservationLinkProposal(ProposalModel):
    """Candidate or context material linked without host truth certification."""

    target_kind: Literal["object", "inquiry"]
    target: GraphRef
    observation_id: str
    role: Literal["candidate", "context"]

    @model_validator(mode="after")
    def _validate_link(self) -> ObservationLinkProposal:
        _require_text(self.observation_id, "observation id")
        return self


_WAKE_IMPRESSION_CATEGORIES = ("attention", "surprise", "unresolved_pull", "released")


class WakeImpression(ProposalModel):
    """A bounded subjective trace with no fact, citation, or evidence authority."""

    attention: list[str] = Field(
        default_factory=list,
        max_length=2,
        description=(
            "Subjective attention only. Any identifier mentioned here is non-authoritative "
            "and must be recovered through memory tools before use as fact or evidence."
        ),
    )
    surprise: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="Subjective surprise only; never a fact, citation, or evidence reference.",
    )
    unresolved_pull: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="Subjective pull only; never a durable inquiry or evidence reference.",
    )
    released: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="Subjectively released thread only; never durable world cognition.",
    )

    @field_validator(*_WAKE_IMPRESSION_CATEGORIES, mode="before")
    @classmethod
    def _normalize_items(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [item.strip() if isinstance(item, str) else item for item in value]

    @field_validator(*_WAKE_IMPRESSION_CATEGORIES)
    @classmethod
    def _validate_items(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("wake impression items must be non-empty")
        if any(len(item) > 160 for item in value):
            raise ValueError("wake impression items must not exceed 160 characters")
        return value

    @model_validator(mode="after")
    def _validate_total(self) -> WakeImpression:
        items = self.items()
        if len(items) > 4:
            raise ValueError("wake impression must not exceed 4 total items")
        if sum(len(text) for _, text in items) > 600:
            raise ValueError("wake impression text must not exceed 600 characters")
        return self

    def items(self) -> list[tuple[str, str]]:
        """Return categorized items in stable display order."""
        return [
            (category, item)
            for category in _WAKE_IMPRESSION_CATEGORIES
            for item in getattr(self, category)
        ]

    @property
    def is_empty(self) -> bool:
        """Report whether this optional envelope section contains no item."""
        return not self.items()


class CognitionDeltaProposal(ProposalModel):
    """Only new or changed graph cognition produced after Agent exploration."""

    schema_version: Literal["1"] = "1"
    new_objects: list[NewObjectProposal] = Field(default_factory=list)
    object_updates: list[ObjectUpdateProposal] = Field(default_factory=list)
    assertions: list[AssertionProposal] = Field(default_factory=list)
    new_inquiries: list[NewInquiryProposal] = Field(default_factory=list)
    resolve_inquiries: list[InquiryResolutionProposal] = Field(default_factory=list)
    observation_links: list[ObservationLinkProposal] = Field(default_factory=list)
    addresses_issue_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_local_declarations(self) -> CognitionDeltaProposal:
        _require_unique([item.local_ref for item in self.new_objects], "object local reference")
        _require_unique([item.local_ref for item in self.new_inquiries], "inquiry local reference")
        _require_unique(self.addresses_issue_ids, "addressed issue id")
        return self


class CognitionProposalEnvelope(CognitionDeltaProposal):
    """Terminal proposal envelope separating cognition from subjective continuity."""

    wake_impression: WakeImpression | None = Field(
        default=None,
        description=(
            "Optional subjective continuity for a later Deep wake. It is stored in runtime "
            "state and is never a fact, assertion, inquiry, or evidence reference. Any "
            "identifier in its text has no fact or evidence authority and must be recovered "
            "through memory tools before use."
        ),
    )

    def cognition(self) -> CognitionDeltaProposal:
        """Return only the world-authoritative cognition portion of the envelope."""
        return CognitionDeltaProposal.model_validate(
            self.model_dump(mode="json", exclude={"wake_impression"})
        )


class ProposalCommitReceipt(ProposalModel):
    """Durable receipt plus the proposal-local identity mapping."""

    commit: CommitReceipt
    review_outcome: ReviewOutcome = ReviewOutcome.ACCEPT
    review_issues: list[ReviewIssue] = Field(default_factory=list)
    attempt: AttemptContext | None = None
    object_ids_by_local_ref: dict[str, str] = Field(default_factory=dict)
    inquiry_ids_by_local_ref: dict[str, str] = Field(default_factory=dict)
    omitted_assertion_indexes: list[int] = Field(default_factory=list)
    evidence_missing_assertion_indexes: list[int] = Field(default_factory=list)
    omitted_resolution_ids: list[str] = Field(default_factory=list)
    omitted_inquiry_indexes: list[int] = Field(default_factory=list)
    deepened_inquiry_ids: list[str] = Field(default_factory=list)
    answerable_inquiries: list[str] = Field(default_factory=list)
    dropped_evidence: dict[int, list[str]] = Field(
        default_factory=dict,
        description=(
            "Proposal assertion index to the evidence observation ids dropped "
            "because no such observation exists (hallucinated ids sanitized at "
            "the proposal boundary instead of rejecting the delta)."
        ),
    )
    remapped: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Historical receipts may carry ghost memory ids that were "
            "auto-remapped to a stored object by name identity match at the "
            "validation boundary; each record is "
            "{from: <ghost id>, to: <stored object id>, matched_name: <name>}. "
            "This field is read-only history: the similarity remap was "
            "removed in Task 2.2 (fail-closed invalid_reference) and new "
            "commits always write an empty list."
        ),
    )
    resolved_object_refs: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Historical receipts may carry deterministic event rewrites — "
            "{from: <local_ref>, to: <existing event id>, basis: <basis>} "
            "records for the (participant set, date) auto-dedup removed in "
            "Task 2.3 (spec §4.1, basis \"participant_set+date\") or the "
            "name-driven auto merge removed in Task 2.1 (basis "
            "\"exact_alias+domain_overlap\"). This field is read-only history: "
            "events only ever produce duplicate_event_candidate review issues "
            "now, and new commits always write an empty list."
        ),
    )


_CORE_OBJECT_KINDS = ("entity", "event", "concept")


def _narrow_object_kinds(parameters: dict[str, object]) -> None:
    """Narrow the schema's ObjectKind enum to the three core kinds, in place.

    The runtime model keeps every legacy member (old commits, replay and
    console parsing depend on them); only the provider-facing schema restricts
    what the Agent may propose. ``model_json_schema`` returns a fresh dict per
    call, so mutating it here never leaks into the model contracts.
    """
    definitions = parameters.get("$defs")
    if not isinstance(definitions, dict):
        return
    kind_def = definitions.get("ObjectKind")
    if isinstance(kind_def, dict):
        kind_def["enum"] = list(_CORE_OBJECT_KINDS)


def _property_schema(
    definitions: dict[str, object], model_name: str, property_name: str
) -> dict[str, object] | None:
    """Return one property schema dict from a model def, or None when absent."""
    model_def = definitions.get(model_name)
    if not isinstance(model_def, dict):
        return None
    properties = model_def.get("properties")
    if not isinstance(properties, dict):
        return None
    prop = properties.get(property_name)
    return prop if isinstance(prop, dict) else None


def _expose_contract_bounds(parameters: dict[str, object]) -> None:
    """Expose the compile-time normalizer bounds on the provider schema, in place.

    The model-visible schema shows the same limits the compiler validates:
    the ``type_key`` pattern, the five bounded qualifier keys with the value
    length cap, and the participant role cap. Runtime enforcement stays in
    ``graph_contract``; this only tells the model the bounds up front.
    ``model_json_schema`` returns a fresh dict per call, so mutating it here
    never leaks into the model contracts.
    """
    definitions = parameters.get("$defs")
    if not isinstance(definitions, dict):
        return
    type_key = _property_schema(definitions, "NewObjectProposal", "type_key")
    if type_key is not None:
        type_key["pattern"] = TYPE_KEY_PATTERN
    role = _property_schema(definitions, "EventParticipantProposal", "role")
    if role is not None:
        role["maxLength"] = 120
    qualifier_keys = sorted(QUALIFIER_KEYS)
    for model_name in ("AssertionProposal", "EventParticipantProposal"):
        qualifiers = _property_schema(definitions, model_name, "qualifiers")
        if qualifiers is None:
            continue
        qualifiers["propertyNames"] = {"enum": qualifier_keys}
        qualifiers["maxProperties"] = len(qualifier_keys)
        qualifiers["additionalProperties"] = {"type": "string", "maxLength": 120}


def submit_cognition_schema() -> dict[str, object]:
    """Return the provider-native terminal proposal function schema."""
    parameters = CognitionProposalEnvelope.model_json_schema()
    _narrow_object_kinds(parameters)
    _expose_contract_bounds(parameters)
    return {
        "type": "function",
        "function": {
            "name": "submit_cognition",
            "description": (
                "Make this wake's one final, terminal proposal containing only "
                "new or changed revisable durable cognition. Calling this ends the "
                "wake; it does not mean the domain is exhausted, every inquiry "
                "is resolved, or any coverage quota was met. When nothing worthwhile "
                "changed, submit an empty cognition delta rather "
                "than inventing cognition. For each Assertion, keep predicate as a compact "
                "relationship or property label rather than the complete claim, and put "
                "the target in exactly one of object or literal. Titles, comments, automatic "
                "transcripts, "
                "seen-only cards, and mixed-depth material may inform a judgment when "
                "its role, confidence, evidence use, and source- or community-aware "
                "subject/predicate/literal wording are honest; no minimum evidence count "
                "is required. Unknown observation ids are dropped "
                "from links, while an otherwise valid no-link judgment may still commit "
                "with a warning. The proposal is validated and committed "
                "by the host; this function does not write the database directly. "
                + IDENTITY_MODEL_SENTENCE
            ),
            "parameters": parameters,
        },
    }


def _require_text(value: str | None, label: str) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _validate_interval(start: datetime | None, end: datetime | None) -> None:
    if start is not None and end is not None and start > end:
        raise ValueError("time interval start must be before or equal to end")
