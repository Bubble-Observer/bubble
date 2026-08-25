"""Offline gate for the Wave E golden scenarios (Slice 7 behavioral goldens).

Loads ``scripts/graph_shell_golden_scenarios.py`` by file path (it is a
standalone script, not a package) and runs all seven scenarios against a
fresh tmp dir with scripted deterministic models. Asserts that every
scenario passes its formal-graph invariants, and that the E2 aggregate
metrics the reporting pipeline depends on are exactly what the traces
produce — no fabricated numbers, no duplicate escapes, and a recorded
interpretation boundary (E3: these proofs are tooling/state-machine proofs,
not claims about real LLM behavior).
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import sys
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).parents[2] / "scripts" / "graph_shell_golden_scenarios.py"
_MODULE_NAME = "graph_shell_golden_scenarios"


def _load_golden_module() -> Any:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # the script's @dataclass decorator resolves KW_ONLY against sys.modules
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[_MODULE_NAME]
    return module


def test_seven_golden_scenarios_pass_with_exact_expected_metrics(tmp_path: Path) -> None:
    golden = _load_golden_module()
    results = asyncio.run(golden.run_all_scenarios(tmp_path / "golden"))
    gc.collect()  # release any transient sqlite handles before tmp_path cleanup

    assert [result.key for result in results] == ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    failed = [result for result in results if not result.passed]
    assert failed == [], [f"{result.key}: {failure}" for result in failed for failure in result.failures]

    metrics = golden._aggregate_metrics(results)  # noqa: SLF001

    # referents: 16 reuses across the seven traces, 5 new objects
    assert metrics["referents"]["reuse_count"] == 16
    assert metrics["referents"]["reuse_rate"] == round(16 / 21, 3)

    # duplicates never escape staging into the formal graph
    assert metrics["duplicate_escape_count"] == 0

    # 8 of 12 formal assertions carry object_ref edges
    assert metrics["object_ref_assertions"]["count"] == 8
    assert metrics["object_ref_assertions"]["rate"] == round(8 / 12, 3)

    # G1 + G6 supersede chains; no literal used as an entity hint
    assert metrics["supersede"]["formed_count"] == 2
    assert metrics["literal_entity_hint_count"] == 0

    # inspect/diff usage matches the traces exactly
    assert metrics["inspect_use_count"] == 3  # G2, G4, G6
    assert metrics["diff_use_count"] == 1  # G6
    assert metrics["inspect_diff_then_modify_count"] == 2  # G4 blocker fix, G6 drop

    # G4 zero-connection blocker surfaced by inspect and fixed by re-patch
    assert metrics["blockers"]["reported"] == 1
    assert metrics["blockers"]["fixed"] == 1
    assert metrics["blockers"]["fix_rate"] == 1.0

    # G5 wake2 + G7 wake1/wake2 staged_unpublished; G7 same-wake resume;
    # G5 + G7 abandons (G7 also exercises the owner-mismatch refusal)
    assert metrics["staged_unpublished_count"] == 3
    assert metrics["resume_count"] == 1
    assert metrics["abandon_count"] == 3

    # 10 wakes produce exactly 8 {wake}:finalize receipts (no legacy roots)
    assert metrics["formal_commits_per_wake"] == {"commits": 8, "wakes": 10}
    assert metrics["turns"]["total"] == 26
    assert metrics["unhandled_exception_count"] == 0
    assert metrics["tokens"] == "fixture (scripted model; not measured)"
    assert metrics["cost_usd"] == "fixture (scripted model; 0.0 by construction)"

    # E3: the artifact pipeline records the interpretation boundary verbatim
    blob = golden._metrics_json(results)  # noqa: SLF001
    assert blob["metrics"] == metrics
    assert "do NOT prove a real LLM will take these actions" in blob["interpretation_boundary"]
    markdown = golden._markdown(blob)  # noqa: SLF001
    assert "| G1 |" in markdown
    assert "| G7 |" in markdown
    assert blob["interpretation_boundary"] in markdown
