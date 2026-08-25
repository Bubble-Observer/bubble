"""Compatibility wrapper for the installed world-agent CLI."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from leave_information_bubble.world_agent.cli import main


if __name__ == "__main__":
    main()
