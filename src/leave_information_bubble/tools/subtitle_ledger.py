"""Persistent subtitle hash ledger — cross-video hash counting for subtitle reliability.

B站 serves a shared fallback pool of "someone else's subtitles" to videos that
have no subtitles of their own; the same AI-subtitle hash then recurs across
many unrelated videos.  This ledger persists every observed
``(domain, hash, bvid)`` and answers the blacklist question the reliability
task needs: "has this hash been seen for at least ``BLACKLIST_THRESHOLD``
distinct bvids within this domain?" — a hash that qualifies is treated as
fallback-pool garbage.

Runtime artifact (``data/runtime/subtitle-hash-ledger.json``)::

    {"lol_cn": {"<hash10>": {"bvids": ["BV1xxx", ...], "count": 2}}}

Design rules:
    * ``count`` counts distinct bvids — re-fetching the same hash for the same
      video never increments (``count`` is derived from ``bvids`` and written
      on every persist; readers decide from the ``bvids`` length).
    * The hash value is supplied by the caller (whitespace-stripped md5
      prefix); this module never computes it.
    * Writes are atomic (temp file + ``os.replace``); a failed write keeps the
      previous file on disk and only logs a warning.
    * Missing or corrupt files degrade to an empty ledger (never blacklisted)
      without raising.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger(__name__)

#: Default runtime artifact location, relative to the worktree root.
DEFAULT_PATH = Path("data/runtime/subtitle-hash-ledger.json")

#: Blacklist threshold — a hash observed for this many distinct bvids within a
#: domain is treated as fallback-pool garbage (plan binding constraint:
#: threshold constants must be named).
BLACKLIST_THRESHOLD = 2

#: In-memory ledger state: domain -> hash -> set of distinct bvids.
LedgerState = dict[str, dict[str, set[str]]]


class SubtitleLedger:
    """Persistent, domain-scoped subtitle hash counter with blacklist queries.

    One JSON file holds, per domain, per subtitle hash, the set of distinct
    bvids the hash was observed for.  A hash observed for at least
    ``BLACKLIST_THRESHOLD`` distinct bvids in a domain is blacklisted
    (fallback-pool garbage).

    In-process concurrent callers (threads or asyncio tasks) are serialized by
    an exclusive lock around each read-modify-write cycle, so repeated
    ``record`` calls within one process always observe each other's updates.
    """

    def __init__(self, path: Path) -> None:
        """Create a ledger backed by ``path``.

        Args:
            path: Path to the ledger JSON file.  Its parent directory must
                exist; the file itself may be missing and is created on the
                first successful record.

        """
        self._path = path
        self._lock = threading.Lock()

    def record(self, domain: str, hash: str, bvid: str) -> None:  # noqa: A002 - API contract names the param "hash"
        """Record one subtitle-hash observation; idempotent per (domain, hash, bvid).

        Re-recording an already-known (domain, hash, bvid) triple is a no-op:
        the ledger counts distinct bvids, never duplicate fetches.

        Args:
            domain: Domain name (e.g. "lol_cn").
            hash: Subtitle content hash (whitespace-stripped md5 prefix,
                computed by the caller; the ledger never derives it).
            bvid: Bilibili video id the hash was observed on.

        """
        with self._lock:
            entries = self._load_entries()
            entries.setdefault(domain, {}).setdefault(hash, set()).add(bvid)
            self._write_entries(entries)

    def is_blacklisted(self, domain: str, hash: str) -> bool:  # noqa: A002 - API contract names the param "hash"
        """Return True when ``hash`` hit ``BLACKLIST_THRESHOLD`` distinct bvids in ``domain``.

        Args:
            domain: Domain name (e.g. "lol_cn").
            hash: Subtitle content hash prefix (see ``record``).

        Returns:
            True if the hash is blacklisted in this domain; False when the
            ledger is missing, corrupt, or the hash has fewer than 2 distinct
            bvids.

        """
        with self._lock:
            entries = self._load_entries()
        return len(entries.get(domain, {}).get(hash, ())) >= BLACKLIST_THRESHOLD

    def _load_entries(self) -> LedgerState:
        """Load the ledger from disk; degrade to an empty state when unreadable.

        Missing file, unparseable JSON, or structurally invalid content all
        yield ``{}`` (never blacklisted, never raised).  A malformed file is
        left on disk untouched; the next successful write replaces it.
        """
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("subtitle ledger %s unreadable (%s); treating as empty", self._path, exc)
            return {}
        if not self._is_valid(payload):
            logger.warning("subtitle ledger %s has invalid structure; treating as empty", self._path)
            return {}
        return {
            domain: {
                hash_key: set(entry["bvids"])
                for hash_key, entry in hashes.items()
            }
            for domain, hashes in payload.items()
        }

    @staticmethod
    def _is_valid(payload: object) -> bool:
        """Return True when ``payload`` matches the expected nested dict shape.

        Domains and hashes must be strings; each hash entry must be a dict
        whose ``bvids`` is a list of strings.  A stale or missing ``count`` is
        tolerated — it is advisory only and recomputed on every write.
        """
        if not isinstance(payload, dict):
            return False
        for domain, hashes in payload.items():
            if not isinstance(domain, str) or not isinstance(hashes, dict):
                return False
            for hash_key, entry in hashes.items():
                if not isinstance(hash_key, str) or not isinstance(entry, dict):
                    return False
                bvids = entry.get("bvids")
                if not isinstance(bvids, list) or not all(isinstance(b, str) for b in bvids):
                    return False
        return True

    def _write_entries(self, entries: LedgerState) -> None:
        """Persist the ledger atomically; on failure keep the old file and warn.

        Writes to a sibling temp file, fsyncs, then ``os.replace`` onto the
        target.  Any ``OSError`` (disk full, permissions, target is a
        directory, ...) is logged and swallowed — the caller keeps running
        with the previous file still in place.
        """
        payload = {
            domain: {
                hash_key: {"bvids": sorted(bvids), "count": len(bvids)}
                for hash_key, bvids in hashes.items()
            }
            for domain, hashes in entries.items()
        }
        tmp = self._path.with_name(f".{self._path.name}-{uuid.uuid4().hex}.tmp")
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.warning(
                "subtitle ledger write to %s failed (%s); keeping previous file",
                self._path,
                exc,
            )
        finally:
            if tmp.exists():
                with suppress(OSError):  # pragma: no cover - best-effort cleanup
                    tmp.unlink()
