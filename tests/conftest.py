"""Shared fixtures for the active test suite."""

from __future__ import annotations

import gc
import os
import shutil
import stat
import uuid
from pathlib import Path

import pytest

_TEST_WORKSPACE = Path(".test_artifacts")


def _force_rmtree(root: Path) -> None:
    """Remove a tree even when it holds Windows read-only files (git repos).

    Plain ``shutil.rmtree`` stops at the first read-only attribute; this
    clears the write bit on every entry first, then removes the tree.
    """
    if not root.exists():
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                os.chmod(path, stat.S_IWRITE)
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def tmp_path() -> Path:
    """Provide a writable isolated path under the managed Windows workspace.

    Removed at session end, not per test: per-test deletion races Windows
    file handles still held by short-lived sqlite connections
    (WinError 32), which would leak the directory anyway. A hard-killed
    session leaves trees behind; the next session's start sweeps them
    (``pytest_sessionstart`` below).
    """
    root = _TEST_WORKSPACE / f"pytest-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root.resolve()


def pytest_sessionstart(session: object) -> None:
    """Sweep leftovers from a previously hard-killed session (fresh process,
    so no open handles)."""
    _force_rmtree(_TEST_WORKSPACE)


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Remove the managed test workspace once every test object is gone.

    gc.collect() forces refcount-collectible sqlite connections to release
    their Windows handles before the sweep.
    """
    gc.collect()
    _force_rmtree(_TEST_WORKSPACE)
