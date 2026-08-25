# Limitations

Honest boundaries of this personal project. If a capability is not listed
here as working, assume it does not exist.

## Scope

- **Single agent, single domain.** The validation domain is the Chinese
  League of Legends community (`lol_cn`). Domain configuration
  (`world/domain_config.py`) is a lens, not a boundary — but generalization
  to other domains is unproven.
- **Single SQLite database, single writer.** No multi-agent, no automatic
  semantic merging, no object-level merge/retract, no production-grade
  concurrency. The single-writer lease has **no TTL and no auto-preemption**:
  a crashed wake holds its lease until explicitly abandoned
  (`--graph-shell-abandon <wake-id>`). This is a deliberate trade-off for a
  personal demo.
- **Not a memory framework.** The artifact is the Graph Shell protocol for one
  agent on one domain, not a reusable long-term-memory library.

## Behavior

- **The model decides.** The host never finalizes on the agent's behalf and
  never merges identities silently. When a real model does not finalize (as in
  the 2026-08-22 D1/D2 runs, both `staged_unpublished`), the working graph
  simply stays staged for an explicit resume. Prompt design still matters:
  a minimal prompt produced no natural stop signal.
- **Scripted evaluation ≠ real behavior.** G1–G7 and A–J prove the protocol
  and state machine. They do not prove that a real model will duplicate-reuse,
  build relationships, or supersede. Real-wake evidence is reported
  separately and is still a small sample (documented runs: 2026-08-18
  two-wake canary, 2026-08-22 two-wake experiment, earlier phase runs).
- **Known behavioral gaps (from real wakes, not speculation):**
  - relationship objects not materialized (proper nouns stay inside literal
    relationships);
  - duplicate inquiries should be deepened rather than re-created;
  - cross-wake name/alias reuse not yet demonstrated;
  - inconsistent event-time attachment across some broad wakes.

## Sources

- Live adapters depend on third-party sites (Bilibili, NGA, Hupu, public web)
  and credentials; site structure and availability change. Capability
  baselines are frozen per adapter, but live access is not guaranteed.
- Acquisition is bounded and honest: adapters report their real surface role
  (`PLATFORM_SEARCH` vs `BOUNDED_BOARD`) and limitations; e.g. NGA community
  streams are a bounded board, not a full search index.
- ASR transcription (faster-whisper) runs in a killable child process with
  CUDA/CPU fallback, but is an optional extra (`media`), not part of the core
  demo.

## Omissions

- No multi-model comparisons, no production concurrency story, no federation,
  no web UI for the cognition loop (the console and edition frontend are
  internal, not part of this repo).
- The live cost/deadline guards exist (`--live-hard-cap-usd`,
  `--live-deadline-seconds`) and are tested, but they are guards, not
  accounting products.
- mypy strict passes with a known baseline of errors in the retained core;
  typing cleanup is ongoing and non-blocking.
