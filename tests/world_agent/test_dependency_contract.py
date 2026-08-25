"""I-1: jsonschema is a declared runtime dependency, not a transitive accident.

The Graph Shell schema validator (``jsonschema.Draft202012Validator`` in
``world_agent/graph.py``) is production code reached by both the legacy and
Graph Shell CLI paths. Until this fix it was undeclared in ``pyproject.toml``
and installed only as a transitive requirement of the optional ``mcp``
package — a clean install without ``mcp`` would crash at import time. These
tests pin the clean-install contract: the dependency is declared with a floor
that guarantees the used API, and the CLI import chain works against the
installed environment.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

import jsonschema

_PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def _declared_dependencies() -> dict[str, str]:
    """Name → raw version spec for every entry in ``[project].dependencies``."""
    with _PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    declared: dict[str, str] = {}
    for raw in project["dependencies"]:
        name = re.match(r"[A-Za-z0-9_.-]+", raw)
        assert name is not None, f"unparseable dependency entry: {raw!r}"
        declared[name.group(0)] = raw
    return declared


def test_pyproject_declares_jsonschema_runtime_dependency() -> None:
    """I-1: the wheel's runtime contract names jsonschema explicitly.

    A package that imports ``jsonschema`` at module load (``graph.py``) must
    list it under ``[project].dependencies`` — not rely on the optional ``mcp``
    extra's transitive install. The declared floor must be >= 4.0, the first
    release whose ``Draft202012Validator`` matches the API the Graph Shell
    dispatch uses.
    """
    spec = _declared_dependencies().get("jsonschema")
    assert spec is not None, (
        "jsonschema is imported by leave_information_bubble.world_agent.graph "
        "but missing from [project].dependencies; clean installs without the "
        "optional mcp extra would crash at CLI import"
    )
    lower_bound = re.search(r">=(\d+)(?:\.(\d+))?", spec)
    assert lower_bound is not None, f"jsonschema spec must pin a floor: {spec!r}"
    major = int(lower_bound.group(1))
    assert major >= 4, (
        f"jsonschema floor {spec!r} allows drafts before 2020-12; "
        "Draft202012Validator requires jsonschema>=4"
    )


def test_installed_jsonschema_satisfies_declared_floor() -> None:
    """I-1: the running environment meets the declared dependency floor."""
    spec = _declared_dependencies()["jsonschema"]
    floor = re.search(r">=([\d.]+)", spec).group(1)  # type: ignore[union-attr]
    installed = importlib.metadata.version("jsonschema")
    assert tuple(int(part) for part in installed.split(".")) >= tuple(
        int(part) for part in floor.split(".")
    ), f"installed jsonschema {installed} below declared floor {spec!r}"


def test_cli_import_chain_reaches_jsonschema_validator() -> None:
    """I-1: legacy and Graph Shell CLI composition roots import with jsonschema.

    ``world_agent.cli`` builds both the legacy and the Graph Shell graphs, and
    ``graph.py`` imports jsonschema at module scope — so a working CLI import
    proves a clean install resolves the declared dependency end to end.
    """
    from leave_information_bubble.world_agent import cli  # noqa: F401

    validator = jsonschema.Draft202012Validator({"type": "object"})
    assert validator.is_valid({"a": 1})
