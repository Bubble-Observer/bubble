# ruff: noqa: D101, D102

from __future__ import annotations

import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class InquiryLease(BaseModel):
    model_config = ConfigDict(frozen=True)

    inquiry_id: str
    owner_id: str
    lease_token: str
    acquired_at: datetime
    expires_at: datetime


class InquiryLeaseStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        connection.execute(
            "CREATE TABLE IF NOT EXISTS inquiry_leases ("
            "inquiry_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, "
            "lease_token TEXT NOT NULL UNIQUE, acquired_at TEXT NOT NULL, expires_at TEXT NOT NULL)"
        )
        connection.close()

    def claim(
        self, inquiry_id: str, owner_id: str, ttl: timedelta = timedelta(minutes=15)
    ) -> InquiryLease | None:
        if not inquiry_id.strip() or not owner_id.strip() or ttl <= timedelta():
            raise ValueError("inquiry, owner, and positive ttl are required")
        now = datetime.now(UTC)
        expires_at = now + ttl
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT expires_at FROM inquiry_leases WHERE inquiry_id = ?", (inquiry_id,)
            ).fetchone()
            if row is not None and datetime.fromisoformat(row["expires_at"]) > now:
                connection.rollback()
                return None
            if row is not None:
                connection.execute("DELETE FROM inquiry_leases WHERE inquiry_id = ?", (inquiry_id,))
            lease = InquiryLease(
                inquiry_id=inquiry_id,
                owner_id=owner_id,
                lease_token=secrets.token_urlsafe(24),
                acquired_at=now,
                expires_at=expires_at,
            )
            connection.execute(
                "INSERT INTO inquiry_leases VALUES (?, ?, ?, ?, ?)",
                (
                    lease.inquiry_id,
                    lease.owner_id,
                    lease.lease_token,
                    lease.acquired_at.isoformat(),
                    lease.expires_at.isoformat(),
                ),
            )
            connection.commit()
            return lease
        finally:
            connection.close()

    def release(self, lease_token: str) -> bool:
        connection = self._connect()
        deleted = connection.execute(
            "DELETE FROM inquiry_leases WHERE lease_token = ?", (lease_token,)
        ).rowcount
        connection.commit()
        connection.close()
        return deleted == 1

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection
