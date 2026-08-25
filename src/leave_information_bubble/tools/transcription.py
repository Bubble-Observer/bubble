"""Bounded local audio transcription with machine-wide GPU serialization."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Protocol, cast

logger = logging.getLogger(__name__)

ASR_AUDIO_EMPTY_LIMITATION = "local_asr_audio_unavailable"
ASR_AUDIO_TOO_LARGE_LIMITATION = "local_asr_audio_exceeds_byte_limit"
ASR_CAPACITY_TIMEOUT_LIMITATION = "local_asr_capacity_wait_timeout"
ASR_CUDA_UNAVAILABLE_LIMITATION = "local_asr_cuda_backend_unavailable"
ASR_DURATION_EXCEEDED_LIMITATION = "local_asr_video_duration_exceeds_limit"
ASR_FAILED_LIMITATION = "local_asr_failed"
ASR_GPU_FALLBACK_LIMITATION = "local_asr_gpu_unavailable_used_cpu"
ASR_NO_SPEECH_LIMITATION = "local_asr_returned_no_speech"
ASR_TIMEOUT_LIMITATION = "local_asr_timeout"

_CUDA_RUNTIME_ERROR_MARKERS = (
    "cublas",
    "cuda",
    "cudnn",
    "cufft",
    "curand",
    "driver version is insufficient",
    "nvrtc",
)


@dataclass(frozen=True)
class TranscriptSegment:
    """One immutable timestamped ASR segment."""

    start: float
    end: float
    content: str
    language: str
    confidence: float
    acquisition_method: str

    def as_dict(self) -> dict[str, Any]:
        """Return the adapter-compatible segment representation."""
        return {
            "start": self.start,
            "end": self.end,
            "content": self.content,
            "language": self.language,
            "confidence": self.confidence,
            "acquisition_method": self.acquisition_method,
            "reliability": "automatic_speech_recognition",
            "attempts": 1,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    """Typed local-ASR outcome safe to surface through a channel adapter."""

    segments: tuple[TranscriptSegment, ...] = ()
    limitation: str | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return the worker-safe JSON representation."""
        return {
            "segments": [segment.as_dict() for segment in self.segments],
            "limitation": self.limitation,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: object) -> TranscriptionResult:
        """Validate and restore one worker result without trusting stdout."""
        if not isinstance(payload, dict):
            raise ValueError("ASR worker result must be an object")
        raw_segments = payload.get("segments", [])
        raw_warnings = payload.get("warnings", [])
        if not isinstance(raw_segments, list) or not isinstance(raw_warnings, list):
            raise ValueError("ASR worker result has invalid collection fields")
        segments: list[TranscriptSegment] = []
        for raw in raw_segments:
            if not isinstance(raw, dict):
                raise ValueError("ASR worker segment must be an object")
            segments.append(
                TranscriptSegment(
                    start=float(raw["start"]),
                    end=float(raw["end"]),
                    content=str(raw["content"]),
                    language=str(raw["language"]),
                    confidence=float(raw["confidence"]),
                    acquisition_method=str(raw["acquisition_method"]),
                )
            )
        limitation = payload.get("limitation")
        if limitation is not None and not isinstance(limitation, str):
            raise ValueError("ASR worker limitation must be a string or null")
        return cls(
            segments=tuple(segments),
            limitation=limitation,
            warnings=tuple(str(item) for item in raw_warnings),
        )


class AudioTranscriber(Protocol):
    """Asynchronous contract consumed by media channel adapters."""

    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        """Transcribe one bounded audio payload without retaining it."""
        ...


class _WhisperModel(Protocol):
    """Narrow runtime contract implemented by faster-whisper and test doubles."""

    def transcribe(self, audio: str, **kwargs: object) -> tuple[Iterable[object], object]:
        """Return an iterable of segments and detected-language metadata."""
        ...


class ASRCapacityTimeoutError(TimeoutError):
    """Raised when another process owns the machine-wide ASR capacity lock."""


class ASRCudaUnavailableError(RuntimeError):
    """Raised when an explicitly requested CUDA backend cannot run."""


class _InterProcessFileLock:
    """Small standard-library advisory lock released automatically on process exit."""

    def __init__(self, path: Path, *, timeout_seconds: float, poll_seconds: float = 0.1) -> None:
        self._path = path
        self._timeout_seconds = max(0.0, timeout_seconds)
        self._poll_seconds = max(0.01, poll_seconds)
        self._handle: BinaryIO | None = None

    def __enter__(self) -> _InterProcessFileLock:
        """Acquire one byte of the lock file or raise after the bounded wait."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+b")
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                self._lock_once(handle)
                self._handle = handle
                return self
            except OSError as error:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise ASRCapacityTimeoutError(str(self._path)) from error
                time.sleep(self._poll_seconds)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the advisory lock and close its file handle."""
        del exc_type, exc_value, traceback
        if self._handle is None:
            return
        try:
            self._unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None

    @staticmethod
    def _lock_once(handle: BinaryIO) -> None:
        """Attempt one non-blocking platform lock acquisition."""
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        """Release one platform lock acquisition."""
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class FasterWhisperTranscriber:
    """Run faster-whisper once per request under a machine-wide capacity lock."""

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        local_files_only: bool = True,
        language: str = "zh",
        hotwords: str = "",
        lock_path: Path = Path("data/runtime/asr-gpu.lock"),
        lock_timeout_seconds: float = 300.0,
        timeout_seconds: float = 900.0,
        max_audio_bytes: int = 50_000_000,
        model_factory: Callable[[str, str, str], _WhisperModel] | None = None,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._local_files_only = local_files_only
        self._language = language
        self._hotwords = hotwords
        self._lock_path = lock_path
        self._lock_timeout_seconds = max(0.0, lock_timeout_seconds)
        self._timeout_seconds = max(0.01, timeout_seconds)
        self._max_audio_bytes = max(1_000_000, max_audio_bytes)
        self._model_factory = model_factory
        self._progress = progress
        self._local_gate = asyncio.Lock()
        self._worker_command = (
            sys.executable,
            "-m",
            "leave_information_bubble.tools.asr_worker",
        )

    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        """Transcribe temporary audio while bounding memory and GPU concurrency."""
        if not audio:
            return TranscriptionResult(limitation=ASR_AUDIO_EMPTY_LIMITATION)
        if len(audio) > self._max_audio_bytes:
            return TranscriptionResult(limitation=ASR_AUDIO_TOO_LARGE_LIMITATION)

        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".m4s", delete=False) as temporary:
                temporary.write(audio)
                path = Path(temporary.name)
            async with self._local_gate:
                self._emit_progress("waiting_for_asr_capacity")
                if self._model_factory is not None:
                    # Injected factories are deterministic test doubles. Real
                    # faster-whisper inference always runs in the killable worker.
                    segments, warnings = await asyncio.wait_for(
                        asyncio.to_thread(self._transcribe_path, path),
                        timeout=self._timeout_seconds,
                    )
                    result = TranscriptionResult(segments=segments, warnings=warnings)
                else:
                    result = await asyncio.wait_for(
                        self._transcribe_subprocess(path),
                        timeout=self._timeout_seconds,
                    )
        except ASRCapacityTimeoutError:
            return TranscriptionResult(limitation=ASR_CAPACITY_TIMEOUT_LIMITATION)
        except ASRCudaUnavailableError:
            logger.exception("Explicit CUDA ASR backend is unavailable")
            return TranscriptionResult(limitation=ASR_CUDA_UNAVAILABLE_LIMITATION)
        except TimeoutError:
            self._emit_progress("asr_timed_out")
            return TranscriptionResult(limitation=ASR_TIMEOUT_LIMITATION)
        except Exception:
            logger.exception("Local ASR failed")
            return TranscriptionResult(limitation=ASR_FAILED_LIMITATION)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

        if result.limitation is not None:
            return result
        if not result.segments:
            return TranscriptionResult(
                limitation=ASR_NO_SPEECH_LIMITATION,
                warnings=result.warnings,
            )
        return result

    async def _transcribe_subprocess(self, media_path: Path) -> TranscriptionResult:
        """Run production inference in a child that cancellation can terminate."""
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source_root, existing_pythonpath) if part
        )
        process = await asyncio.create_subprocess_exec(
            *self._worker_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        request = {
            "media_path": str(media_path),
            "model_size": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
            "local_files_only": self._local_files_only,
            "language": self._language,
            "hotwords": self._hotwords,
            "lock_path": str(self._lock_path),
            "lock_timeout_seconds": self._lock_timeout_seconds,
        }
        process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode())
        await process.stdin.drain()
        process.stdin.close()
        stderr_task = asyncio.create_task(self._forward_worker_stderr(process.stderr))
        result: TranscriptionResult | None = None
        try:
            while line := await process.stdout.readline():
                # Windows child processes may inherit a non-UTF-8 console
                # encoding even though this protocol is JSONL.  Decoding with
                # replacement keeps progress telemetry from aborting the
                # acquisition; the typed final result still determines success.
                event = json.loads(line.decode("utf-8", errors="replace"))
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "progress":
                    stage = str(event.get("stage", ""))
                    if stage:
                        self._emit_progress(stage, worker_pid=process.pid)
                elif event.get("type") == "result":
                    result = TranscriptionResult.from_dict(event.get("result"))
            return_code = await process.wait()
            if return_code != 0 or result is None:
                raise RuntimeError(f"ASR worker exited without a result (code={return_code})")
            return result
        except BaseException:
            await self._terminate_worker(process)
            raise
        finally:
            await stderr_task

    async def _forward_worker_stderr(self, stream: asyncio.StreamReader) -> None:
        """Forward bounded worker diagnostics instead of hiding child failures."""
        while line := await stream.readline():
            logger.warning("ASR worker: %s", line.decode(errors="replace").rstrip())

    @staticmethod
    async def _terminate_worker(process: asyncio.subprocess.Process) -> None:
        """Terminate one ASR worker, escalating to kill after a short grace."""
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            process.kill()
            await process.wait()

    def _emit_progress(self, stage: str, **details: object) -> None:
        """Publish one best-effort operational event without affecting ASR."""
        if self._progress is None:
            return
        try:
            self._progress({"component": "asr", "stage": stage, **details})
        except Exception:  # noqa: BLE001 - observability must never break acquisition
            logger.debug("ASR progress sink failed", exc_info=True)

    def _transcribe_path(
        self,
        media_path: Path,
    ) -> tuple[tuple[TranscriptSegment, ...], tuple[str, ...]]:
        """Load, run, and release one model while holding the global capacity lock."""
        with _InterProcessFileLock(
            self._lock_path,
            timeout_seconds=self._lock_timeout_seconds,
        ):
            self._emit_progress("asr_capacity_acquired")
            device, compute_type = self._resolve_backend()
            try:
                return self._transcribe_with_backend(
                    media_path,
                    device=device,
                    compute_type=compute_type,
                )
            except RuntimeError as error:
                requested_device = self._device.strip().casefold() or "auto"
                cuda_failure = device == "cuda" and self._is_cuda_runtime_error(error)
                if requested_device != "auto" or not cuda_failure:
                    if requested_device == "cuda" and cuda_failure:
                        raise ASRCudaUnavailableError(str(error)) from error
                    raise
                logger.warning(
                    "CUDA ASR failed; retrying once on CPU with int8: %s: %s",
                    type(error).__name__,
                    error,
                )
                segments, _ = self._transcribe_with_backend(
                    media_path,
                    device="cpu",
                    compute_type="int8",
                )
                return segments, (ASR_GPU_FALLBACK_LIMITATION,)

    @staticmethod
    def _is_cuda_runtime_error(error: RuntimeError) -> bool:
        """Recognize backend availability failures without retrying bad media."""
        message = str(error).casefold()
        return any(marker in message for marker in _CUDA_RUNTIME_ERROR_MARKERS)

    def _transcribe_with_backend(
        self,
        media_path: Path,
        *,
        device: str,
        compute_type: str,
    ) -> tuple[tuple[TranscriptSegment, ...], tuple[str, ...]]:
        """Run exactly one backend attempt and release its model eagerly."""
        model: _WhisperModel | None = None
        raw_segments: Iterable[object] | None = None
        info: object | None = None
        try:
            self._emit_progress("loading_asr_model", device=device)
            model = self._create_model(device, compute_type)
            self._emit_progress("transcribing", device=device)
            raw_segments, info = model.transcribe(
                str(media_path),
                language=self._language,
                vad_filter=True,
                word_timestamps=False,
                condition_on_previous_text=True,
                beam_size=1,
                best_of=1,
                hotwords=self._hotwords or None,
            )
            language = str(getattr(info, "language", "") or self._language)
            acquisition_method = f"faster_whisper:{self._model_size}:{device}"
            result: list[TranscriptSegment] = []
            for raw in raw_segments:
                content = str(getattr(raw, "text", "") or "").strip()
                if not content:
                    continue
                average_log_probability = float(getattr(raw, "avg_logprob", -10.0) or -10.0)
                confidence = max(0.0, min(1.0, 1.0 + average_log_probability / 5.0))
                result.append(
                    TranscriptSegment(
                        start=float(getattr(raw, "start", 0.0) or 0.0),
                        end=float(getattr(raw, "end", 0.0) or 0.0),
                        content=content,
                        language=language,
                        confidence=confidence,
                        acquisition_method=acquisition_method,
                    )
                )
            return tuple(result), ()
        finally:
            if raw_segments is not None:
                del raw_segments
            if info is not None:
                del info
            if model is not None:
                del model
            gc.collect()

    def _create_model(self, device: str, compute_type: str) -> _WhisperModel:
        """Construct one model through the injected or optional runtime factory."""
        if self._model_factory is not None:
            return self._model_factory(self._model_size, device, compute_type)
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError("faster-whisper media dependency is not installed") from error
        return cast(
            _WhisperModel,
            WhisperModel(
                self._model_size,
                device=device,
                compute_type=compute_type,
                local_files_only=self._local_files_only,
            ),
        )

    def _resolve_backend(self) -> tuple[str, str]:
        """Resolve portable auto settings to a conservative CPU or CUDA backend."""
        device = self._device.strip().casefold() or "auto"
        if device == "auto":
            try:
                import ctranslate2  # type: ignore[import-untyped]

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except (ImportError, RuntimeError):
                device = "cpu"
        compute_type = self._compute_type.strip().casefold() or "auto"
        if compute_type == "auto":
            compute_type = "int8_float16" if device == "cuda" else "int8"
        return device, compute_type


__all__ = [
    "ASR_AUDIO_EMPTY_LIMITATION",
    "ASR_AUDIO_TOO_LARGE_LIMITATION",
    "ASR_CAPACITY_TIMEOUT_LIMITATION",
    "ASR_CUDA_UNAVAILABLE_LIMITATION",
    "ASR_DURATION_EXCEEDED_LIMITATION",
    "ASR_FAILED_LIMITATION",
    "ASR_GPU_FALLBACK_LIMITATION",
    "ASR_NO_SPEECH_LIMITATION",
    "ASR_TIMEOUT_LIMITATION",
    "AudioTranscriber",
    "FasterWhisperTranscriber",
    "TranscriptSegment",
    "TranscriptionResult",
]
