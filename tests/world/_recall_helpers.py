# Split from tests/world/test_world_recall.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime

from leave_information_bubble.world import (
    AssertionInput,
    EpistemicRole,
    EvidenceInput,
    ObjectInput,
    ObjectKind,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)


class _TracingWorldStore(WorldStore):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.statements: list[str] = []

    @contextmanager
    def read_connection(self):  # type: ignore[no-untyped-def]
        with super().read_connection() as connection:
            connection.set_trace_callback(self.statements.append)
            yield connection


def _object(
    identifier: str, *, aliases: list[str] | None = None, domain_hints: list[str] | None = None
) -> ObjectInput:
    return ObjectInput(
        id=identifier,
        kind=ObjectKind.ENTITY,
        canonical_name=identifier.replace("-", " ").title(),
        aliases=aliases or [],
        domain_hints=domain_hints or [],
    )


def _observation(identifier: str) -> ObservationInput:
    return ObservationInput(
        id=identifier,
        source_uri=f"https://example.test/{identifier}",
        source_kind="web",
        depth=ObservationDepth.CONTENT,
        observed_at=NOW,
    )


def _assertion(
    identifier: str,
    subject_id: str,
    *,
    literal: str | None = None,
    object_id: str | None = None,
    event_time_start: datetime | None = None,
    answers_inquiry_id: str | None = None,
) -> AssertionInput:
    return AssertionInput(
        id=identifier,
        subject_id=subject_id,
        predicate="related_to",
        literal=literal,
        object_id=object_id,
        epistemic_role=EpistemicRole.FACT,
        confidence=0.8,
        evidence=[EvidenceInput(observation_id="observation-1", role="supports")],
        event_time_start=event_time_start,
        answers_inquiry_id=answers_inquiry_id,
    )


def _event_object(identifier: str, *, event_time_start: datetime | None = None) -> ObjectInput:
    return ObjectInput(
        id=identifier,
        kind=ObjectKind.EVENT,
        canonical_name=identifier.replace("-", " ").title(),
        event_time_start=event_time_start,
    )


def _participant_edge(
    identifier: str,
    event_id: str,
    participant_id: str,
    *,
    role: str,
    qualifiers: dict[str, str] | None = None,
    evidence: list[tuple[str, str]] | None = None,
) -> AssertionInput:
    """One compiled has_participant edge: event subject -> participant object.

    Mirrors the committer's participant-edge compilation (role folded into
    qualifiers["role"], epistemic role/confidence/evidence pass through).
    """
    return AssertionInput(
        id=identifier,
        subject_id=event_id,
        predicate="has_participant",
        object_id=participant_id,
        epistemic_role=EpistemicRole.FACT,
        confidence=0.8,
        qualifiers={**(qualifiers or {}), "role": role},
        evidence=[
            EvidenceInput(observation_id=observation_id, role=evidence_role)
            for observation_id, evidence_role in (evidence or [("observation-1", "supports")])
        ],
    )


