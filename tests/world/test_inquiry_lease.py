from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from leave_information_bubble.runtime.inquiry_lease import InquiryLeaseStore
from leave_information_bubble.world import CognitiveDelta, InquiryInput, ObjectInput, ObjectKind, WorldStore


def test_only_one_unexpired_lease_can_exist_per_inquiry(tmp_path: pytest.TempPathFactory) -> None:
    """Catch concurrent inquiry claims that overlap before the first lease expires."""
    leases = InquiryLeaseStore(tmp_path / "runtime.sqlite3")

    first = leases.claim("inquiry-1", "agent-a")
    second = leases.claim("inquiry-1", "agent-b")

    assert first is not None
    assert first.owner_id == "agent-a"
    assert second is None
    assert leases.release(first.lease_token) is True
    assert leases.release(first.lease_token) is False


def test_expired_lease_is_reclaimable_and_runtime_deletion_does_not_change_world(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch runtime recovery that either retains expired locks or mutates durable cognition."""
    world_path = tmp_path / "world.sqlite3"
    runtime_path = tmp_path / "runtime.sqlite3"
    world = WorldStore(world_path)
    world.memory_commit(
        CognitiveDelta(
            objects=[ObjectInput(id="subject", kind=ObjectKind.ENTITY, canonical_name="Subject")],
            inquiries=[InquiryInput(id="inquiry-1", subject_id="subject", prompt="Why?", rationale="Gap")],
        ),
        "durable-inquiry",
    )
    leases = InquiryLeaseStore(runtime_path)
    first = leases.claim("inquiry-1", "agent-a", ttl=timedelta(seconds=1))
    assert first is not None
    connection = sqlite3.connect(runtime_path)
    try:
        connection.execute(
            "UPDATE inquiry_leases SET expires_at = ? WHERE inquiry_id = ?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), "inquiry-1"),
        )
        connection.commit()
    finally:
        connection.close()
    second = leases.claim("inquiry-1", "agent-b")

    assert second is not None
    assert second.owner_id == "agent-b"
    runtime_path.unlink()
    with world.read_connection() as connection:
        inquiry = connection.execute("SELECT status FROM inquiries WHERE id = 'inquiry-1'").fetchone()
    assert inquiry[0] == "open"
