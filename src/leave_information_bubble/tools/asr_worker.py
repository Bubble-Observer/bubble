"""One-request faster-whisper worker with a line-delimited JSON protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .transcription import (
    ASR_CAPACITY_TIMEOUT_LIMITATION,
    ASR_CUDA_UNAVAILABLE_LIMITATION,
    ASR_FAILED_LIMITATION,
    ASRCapacityTimeoutError,
    ASRCudaUnavailableError,
    FasterWhisperTranscriber,
    TranscriptionResult,
)


def _write(payload: dict[str, object]) -> None:
    """Write and flush one protocol message."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _progress(event: dict[str, object]) -> None:
    """Translate internal progress into the worker protocol."""
    _write({"type": "progress", "stage": str(event.get("stage", ""))})


def _run(payload: dict[str, Any]) -> TranscriptionResult:
    """Execute one validated request synchronously inside this worker."""
    transcriber = FasterWhisperTranscriber(
        str(payload["model_size"]),
        device=str(payload["device"]),
        compute_type=str(payload["compute_type"]),
        local_files_only=bool(payload["local_files_only"]),
        language=str(payload["language"]),
        hotwords=str(payload["hotwords"]),
        lock_path=Path(str(payload["lock_path"])),
        lock_timeout_seconds=float(payload["lock_timeout_seconds"]),
        progress=_progress,
    )
    try:
        segments, warnings = transcriber._transcribe_path(Path(str(payload["media_path"])))  # noqa: SLF001
        return TranscriptionResult(segments=segments, warnings=warnings)
    except ASRCapacityTimeoutError:
        return TranscriptionResult(limitation=ASR_CAPACITY_TIMEOUT_LIMITATION)
    except ASRCudaUnavailableError:
        return TranscriptionResult(limitation=ASR_CUDA_UNAVAILABLE_LIMITATION)
    except Exception as error:  # noqa: BLE001 - worker must always return typed output
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        sys.stderr.flush()
        return TranscriptionResult(limitation=ASR_FAILED_LIMITATION)


def main() -> int:
    """Read exactly one request and emit exactly one final result."""
    # The parent side speaks an explicit UTF-8 JSONL protocol.  Reconfigure
    # Windows pipes so localized model progress cannot leak a legacy code page
    # into stdout and make an otherwise valid result undecodable.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    line = sys.stdin.readline()
    if not line:
        sys.stderr.write("missing ASR request\n")
        return 2
    try:
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        result = _run(payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        sys.stderr.write(f"invalid ASR request: {error}\n")
        return 2
    _write({"type": "result", "result": result.as_dict()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
