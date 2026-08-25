"""Append-only curiosity-log for exploration inquiry points (runtime artifact).

One JSON line per inquiry point; the log is shared across threads, each
row carries its thread_id. The write side appends rows; the read side
(load_curiosity / aggregate_curiosity) filters and aggregates them for
the A2 interest-clue consumer. Write failures are logged and swallowed
so exploration is never blocked by log problems.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CURIOSITY_LOG_DEFAULT = Path("data/runtime/curiosity-log.jsonl")


def append_curiosity(
    path: Path,
    thread_id: str,
    round_no: int,
    source_type: str,
    source_ref: str,
    topic: str,
    reason: str,
) -> None:
    """Append one inquiry point as a JSON line, creating parents as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "thread_id": thread_id,
            "round": round_no,
            "source_type": source_type,
            "source_ref": source_ref,
            "topic": topic,
            "reason": reason,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    except OSError as error:
        logger.warning("curiosity-log append failed: %s", error)


TOPIC_KEY_LIMIT = 80
CURIOSITY_MIN_COUNT = 2
CURIOSITY_ROUNDS_BACK = 3


def _topic_key(topic: str) -> str:
    """Normalize a topic into a comparison key: whitespace stripped, capped."""
    return re.sub(r"\s+", "", topic)[:TOPIC_KEY_LIMIT]


def load_curiosity(path: Path, thread_id: str, min_round: int | None = None) -> list[dict[str, object]]:
    """Parse JSON lines for one thread; corrupt rows are skipped."""
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if row.get("thread_id") != thread_id:
                    continue
                if min_round is not None and int(row.get("round", 0) or 0) < min_round:
                    continue
                rows.append(row)
    except OSError as error:
        logger.warning("curiosity-log read failed: %s", error)
    return rows


def aggregate_curiosity(
    rows: list[dict[str, object]],
    min_count: int = CURIOSITY_MIN_COUNT,
    max_rounds_back: int = CURIOSITY_ROUNDS_BACK,
    current_round: int | None = None,
) -> list[dict[str, object]]:
    """Aggregate rows by topic key into interest-clue candidates.

    Keeps topics seen at least min_count times, fresh enough to still be
    live exploration fuel: when current_round is given, last_round must be
    within the last max_rounds_back rounds counting back from current_round.
    """
    buckets: dict[str, dict[str, object]] = {}
    for row in rows:
        topic = str(row.get("topic", "") or "")
        key = _topic_key(topic)
        if not key:
            continue
        bucket = buckets.setdefault(key, {"topic": topic, "count": 0, "last_round": 0, "source_refs": []})
        bucket["count"] = int(bucket["count"]) + 1
        bucket["last_round"] = max(int(bucket["last_round"]), int(row.get("round", 0) or 0))
        ref = str(row.get("source_ref", "") or "")
        if ref and ref not in bucket["source_refs"]:
            bucket["source_refs"].append(ref)
    result = []
    for bucket in buckets.values():
        if int(bucket["count"]) < min_count:
            continue
        # Freshness window is the max_rounds_back rounds ending at current_round;
        # e.g. back=3 at round 5 keeps last_round in [3, 5], so round 2 is stale.
        if current_round is not None and int(bucket["last_round"]) < current_round - max_rounds_back + 1:
            continue
        result.append(bucket)
    result.sort(key=lambda b: (-int(b["count"]), str(b["last_round"])))
    return result
