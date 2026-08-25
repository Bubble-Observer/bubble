# Contributing

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ; macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q --import-mode=importlib --no-cov
python examples/offline_demo.py
```

## What the project values

- **Small, targeted changes.** Prefer a narrow fix over a new framework or
  workflow layer.
- **Deterministic host, subjective agent.** Everything the host decides is
  deterministic and fail-closed; everything requiring judgment is the model's
  call, recorded with rationale.
- **Evidence-anchored cognition.** Assertions link back to observations;
  corrections are new assertions (supersede), never silent rewrites.
- **Honest evaluation.** Scripted scenarios prove the protocol and state
  machine; only documented real-wake runs count as behavioral evidence. Never
  substitute one for the other.
- **Offline tests.** The suite runs without network or model calls and covers
  failure paths.

## Test conventions

- New behavior goes with a test that exercises the real composition root
  (`world_agent/cli.py`) with scripted-model fixtures on an isolated database
  pair — never against configured real databases.
- Failure paths are first-class: rejection verdicts, conflict replay, stale
  bases, and crash recovery are tested, not just happy paths.

## Code style

Run before pushing:

```bash
ruff check src tests scripts examples
```

`mypy` is advisory; the retained world and world-agent roots carry a
pre-existing typing baseline that is not a merge gate.

## What is out of scope

Multi-agent coordination, object merge/retract operations, production
concurrency, and new cognition features in the deterministic output finalizer
are deliberate non-goals. Discuss a proposal before building it.
