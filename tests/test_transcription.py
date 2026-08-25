"""Offline tests for bounded local ASR resource control."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leave_information_bubble.tools.transcription import (
    ASR_AUDIO_TOO_LARGE_LIMITATION,
    ASR_CAPACITY_TIMEOUT_LIMITATION,
    ASR_CUDA_UNAVAILABLE_LIMITATION,
    ASR_FAILED_LIMITATION,
    ASR_GPU_FALLBACK_LIMITATION,
    ASR_TIMEOUT_LIMITATION,
    FasterWhisperTranscriber,
    _InterProcessFileLock,
)


class _FakeModel:
    def __init__(
        self,
        *,
        on_run: Any | None = None,
        captured_paths: list[Path] | None = None,
    ) -> None:
        self._on_run = on_run
        self._captured_paths = captured_paths

    def transcribe(self, path: str, **kwargs: Any) -> tuple[list[Any], Any]:
        del kwargs
        if self._captured_paths is not None:
            self._captured_paths.append(Path(path))
        if self._on_run is not None:
            self._on_run()
        return (
            [SimpleNamespace(start=1.0, end=2.5, text=" spoken text ", avg_logprob=-0.5)],
            SimpleNamespace(language="zh"),
        )


@pytest.mark.asyncio
async def test_transcriber_deletes_temporary_audio_and_releases_model(tmp_path: Path) -> None:
    paths: list[Path] = []
    model_refs: list[weakref.ReferenceType[_FakeModel]] = []

    def factory(model_size: str, device: str, compute_type: str) -> _FakeModel:
        assert (model_size, device, compute_type) == ("small", "cpu", "int8")
        model = _FakeModel(captured_paths=paths)
        model_refs.append(weakref.ref(model))
        return model

    transcriber = FasterWhisperTranscriber(
        device="cpu",
        compute_type="int8",
        lock_path=tmp_path / "asr.lock",
        model_factory=factory,
    )

    result = await transcriber.transcribe(b"encoded-audio")

    assert result.limitation is None
    assert result.segments[0].content == "spoken text"
    assert result.segments[0].acquisition_method == "faster_whisper:small:cpu"
    assert paths and not paths[0].exists()
    assert model_refs[0]() is None


@pytest.mark.asyncio
async def test_transcriber_rejects_oversized_audio_before_model_load(tmp_path: Path) -> None:
    loaded = False

    def factory(model_size: str, device: str, compute_type: str) -> _FakeModel:
        del model_size, device, compute_type
        nonlocal loaded
        loaded = True
        return _FakeModel()

    transcriber = FasterWhisperTranscriber(
        max_audio_bytes=1_000_000,
        lock_path=tmp_path / "asr.lock",
        model_factory=factory,
    )

    result = await transcriber.transcribe(b"x" * 1_000_001)

    assert result.limitation == ASR_AUDIO_TOO_LARGE_LIMITATION
    assert loaded is False


@pytest.mark.asyncio
async def test_two_transcriber_instances_share_one_capacity_lock(tmp_path: Path) -> None:
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def on_run() -> None:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.08)
        with state_lock:
            active -= 1

    def factory(model_size: str, device: str, compute_type: str) -> _FakeModel:
        del model_size, device, compute_type
        return _FakeModel(on_run=on_run)

    lock_path = tmp_path / "shared-asr.lock"
    first = FasterWhisperTranscriber(lock_path=lock_path, model_factory=factory)
    second = FasterWhisperTranscriber(lock_path=lock_path, model_factory=factory)

    results = await asyncio.gather(
        first.transcribe(b"first"),
        second.transcribe(b"second"),
    )

    assert all(result.segments for result in results)
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_capacity_wait_timeout_is_typed_and_does_not_load_model(tmp_path: Path) -> None:
    loaded = False
    lock_path = tmp_path / "busy-asr.lock"

    def factory(model_size: str, device: str, compute_type: str) -> _FakeModel:
        del model_size, device, compute_type
        nonlocal loaded
        loaded = True
        return _FakeModel()

    transcriber = FasterWhisperTranscriber(
        lock_path=lock_path,
        lock_timeout_seconds=0.03,
        model_factory=factory,
    )

    with _InterProcessFileLock(lock_path, timeout_seconds=0.1):
        result = await transcriber.transcribe(b"encoded-audio")

    assert result.limitation == ASR_CAPACITY_TIMEOUT_LIMITATION
    assert loaded is False


@pytest.mark.asyncio
async def test_auto_cuda_runtime_failure_retries_once_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts: list[tuple[str, str]] = []

    class _FailingCudaModel(_FakeModel):
        def transcribe(self, path: str, **kwargs: Any) -> tuple[list[Any], Any]:
            del path, kwargs
            raise RuntimeError("Library cublas64_12.dll is not found")

    def factory(model_size: str, device: str, compute_type: str) -> _FakeModel:
        del model_size
        attempts.append((device, compute_type))
        return _FailingCudaModel() if device == "cuda" else _FakeModel()

    transcriber = FasterWhisperTranscriber(
        device="auto",
        compute_type="auto",
        lock_path=tmp_path / "asr.lock",
        model_factory=factory,
    )
    monkeypatch.setattr(transcriber, "_resolve_backend", lambda: ("cuda", "int8_float16"))

    with caplog.at_level("WARNING"):
        result = await transcriber.transcribe(b"encoded-audio")

    assert attempts == [("cuda", "int8_float16"), ("cpu", "int8")]
    assert result.limitation is None
    assert result.warnings == (ASR_GPU_FALLBACK_LIMITATION,)
    assert result.segments[0].acquisition_method == "faster_whisper:small:cpu"
    assert "retrying once on CPU" in caplog.text
    assert "cublas64_12.dll" in caplog.text


@pytest.mark.asyncio
async def test_explicit_cuda_failure_is_typed_without_cpu_fallback(tmp_path: Path) -> None:
    attempts: list[tuple[str, str]] = []

    class _FailingCudaModel(_FakeModel):
        def transcribe(self, path: str, **kwargs: Any) -> tuple[list[Any], Any]:
            del path, kwargs
            raise RuntimeError("CUDA backend unavailable")

    def factory(model_size: str, device: str, compute_type: str) -> _FakeModel:
        del model_size
        attempts.append((device, compute_type))
        return _FailingCudaModel()

    transcriber = FasterWhisperTranscriber(
        device="cuda",
        compute_type="float16",
        lock_path=tmp_path / "asr.lock",
        model_factory=factory,
    )

    result = await transcriber.transcribe(b"encoded-audio")

    assert attempts == [("cuda", "float16")]
    assert result.limitation == ASR_CUDA_UNAVAILABLE_LIMITATION
    assert result.segments == ()


@pytest.mark.asyncio
async def test_auto_cuda_media_failure_does_not_retry_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    class _BadMediaModel(_FakeModel):
        def transcribe(self, path: str, **kwargs: Any) -> tuple[list[Any], Any]:
            del path, kwargs
            raise RuntimeError("Invalid audio stream")

    def factory(model_size: str, device: str, compute_type: str) -> _FakeModel:
        del model_size, compute_type
        attempts.append(device)
        return _BadMediaModel()

    transcriber = FasterWhisperTranscriber(
        device="auto",
        lock_path=tmp_path / "asr.lock",
        model_factory=factory,
    )
    monkeypatch.setattr(transcriber, "_resolve_backend", lambda: ("cuda", "int8_float16"))

    result = await transcriber.transcribe(b"not-valid-audio")

    assert attempts == ["cuda"]
    assert result.limitation == ASR_FAILED_LIMITATION
    assert result.warnings == ()


@pytest.mark.asyncio
async def test_production_worker_timeout_kills_process_and_releases_capacity_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "worker.lock"
    script = (
        "import json,msvcrt,sys,time,pathlib;"
        "p=json.loads(sys.stdin.readline());"
        "h=pathlib.Path(p['lock_path']).open('a+b');"
        "h.write(b'x');h.flush();h.seek(0);"
        "msvcrt.locking(h.fileno(),msvcrt.LK_NBLCK,1);"
        "print(json.dumps({'type':'progress','stage':'transcribing'}),flush=True);"
        "time.sleep(30)"
    )
    events: list[dict[str, object]] = []
    terminated: list[int | None] = []
    transcriber = FasterWhisperTranscriber(
        device="cpu",
        lock_path=lock_path,
        timeout_seconds=0.3,
        progress=events.append,
    )
    transcriber._worker_command = (sys.executable, "-c", script)
    original = transcriber._terminate_worker

    async def tracked_terminate(process: asyncio.subprocess.Process) -> None:
        await original(process)
        terminated.append(process.returncode)

    monkeypatch.setattr(transcriber, "_terminate_worker", tracked_terminate)

    result = await transcriber.transcribe(b"encoded-audio")

    assert result.limitation == ASR_TIMEOUT_LIMITATION
    assert terminated and terminated[0] is not None
    assert any(event.get("stage") == "transcribing" for event in events)
    with _InterProcessFileLock(lock_path, timeout_seconds=0.2):
        pass


@pytest.mark.asyncio
async def test_worker_protocol_tolerates_non_utf8_progress_bytes(
    tmp_path: Path,
) -> None:
    """Malformed localized progress must not discard a valid typed result."""
    script = (
        "import json,sys;"
        "json.loads(sys.stdin.buffer.readline());"
        "sys.stdout.buffer.write(b'{\\\"type\\\":\\\"progress\\\",\\\"stage\\\":\\\"bad-\\xff\\\"}\\n');"
        "sys.stdout.buffer.write((json.dumps({'type':'result','result':"
        "{'segments':[{'content':'ok','start':0.0,'end':1.0,'language':'zh',"
        "'confidence':0.9,'acquisition_method':'local_asr'}],'warnings':[],"
        "'limitation':None}})+'\\n').encode());"
        "sys.stdout.buffer.flush()"
    )
    transcriber = FasterWhisperTranscriber(
        device="cpu",
        lock_path=tmp_path / "worker.lock",
    )
    transcriber._worker_command = (sys.executable, "-c", script)

    result = await transcriber.transcribe(b"encoded-audio")

    assert result.limitation is None
    assert [segment.content for segment in result.segments] == ["ok"]
