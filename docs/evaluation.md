# Evaluation

Two layers, kept strictly separate:

1. **Offline contract evaluation** — deterministic, no model, no network.
   Proves the protocol, the state machine, and the formal-graph results.
2. **Real wake evidence** — documented runs with a real model and real
   sources. Behavioral evidence that an actual model can operate the
   protocol.

Scripted scenarios are **not** evidence that a real model behaves that way;
real wakes are not evidence about protocol edge cases. Both are reported
below.

---

## 1. Offline contract evaluation

All commands run from a fresh checkout with no API key and no network.

### Layers

| Layer | What it proves | Where |
|---|---|---|
| A–J contract acceptance | 11 scenarios mapping spec §11.1: kind routing, canonical-name coexistence, alias audit, community-name usage, participant edges, same-day granularity coexistence, no fuzzy auto-remap, field-level conflict safety, bounded ego reads, shared identity contract, envelope fingerprints | `tests/world/test_contract_acceptance.py` |
| G1–G7 golden scenarios | formal-graph behavior acceptance driven through the **real CLI** with scripted models: G1 duplicate reuse & correction (supersede), G2 same-name explicit decision, G3 relationships as edges, G4 zero-connection blocker, G5 cross-wake continuity & staging isolation, G6 proactive correction + inspect/diff before revise, G7 recovery & manual recovery without manual SQL | `scripts/graph_shell_golden_scenarios.py`, pytest mirror `tests/world_agent/test_graph_shell_golden_scenarios.py` |
| Strict `graph_patch` contract | op_id idempotency, replay/conflict, batch rejection, malformed-call side-effect-freedom, wake-closed guard | `tests/world_agent/test_graph_patch_strict_contract.py` |
| Protocol fingerprint | the 19-tool Graph Shell surface and schema stability | `tests/world_agent/test_graph_shell_protocol.py` |
| Defect-proof | mutation-based proofs that retired runtime modules are unreachable (internal scripted proofs; the public export keeps only the surviving runtime) | `tests/world_agent/test_dependency_contract.py` |
| Recovery | CP1/CP2/CP6 crash semantics, resume, single-writer lease, abandon | `tests/world_agent/test_graph_shell_recovery.py`, `tests/world_agent/test_graph_shell_writer_lease.py` |
| Migration | additive schema migration from a blank or previous schema | `tests/world/test_schema_migration.py` |
| Guards | live pre-call hard cap (G1) and deadline semantics | `tests/world_agent/test_live_cost_guard.py`, `tests/world_agent/test_live_deadline.py` |
| Full suite | unit + integration + scenarios, all offline | `python -m pytest -q` |

### Current offline result

```text
source SHA 75822543  (tag pre-public-export-20260822)
1570 passed, 1 skipped, 0 failures     # the skip is a platform-specific symlink case
golden scenarios: 7/7 PASS             # scripts/graph_shell_golden_scenarios.py
```

---

## 2. Real wake evidence

Every run used an **isolated copy** of a fixed snapshot world database, a
separate runtime database, a single writer, and a documented cost/deadline
boundary. `deepseek-v4-flash` via the DeepSeek API.

### 2.1 Durable-cognition live canary — 2026-08-18 (PASS)

Two wakes, serial single-writer, on an isolated v11 world DB.
Report: internal `docs/reports/2026-08-18-durable-cognition-live-canary.md`.

| | Value |
|---|---|
| Wakes | 2 (broad → deep, 2nd with a given object id) |
| Model calls | 31, all `deepseek-v4-flash` / `provider_effective` / thinking off |
| Cost / wall time | $0.035193 / 267.19 s (budget $1.00, 30 min) |
| Wake 1 (broad) | 26 objects / 35 assertions / 4 inquiries / 25 evidence links; `COMMIT_WITH_WARNINGS`, omissions = 0 |
| Wake 2 (deep) | 7 of 10 new assertions reused wake-1 persistent entity ids (entity reuse + event-participant recall); all four memory navigation tools used; cross-wake inquiry touches honestly omitted |
| Integrity | 0 orphans, 0 wrong merges, 0 alias miswrites; cache accounting real (digest-family 0 hits recorded truthfully) |
| Follow-ups | 4 observations → follow-up list (relationship objects not materialized; duplicate inquiries should deepen; digest cache layout; stale-resolution touch accounting) |

### 2.2 Curiosity-prompt experiment D1/D2 — 2026-08-22

Two deep wakes on the same continuation graph (the r2 formal graph: 43
objects / 103 assertions / 3 open inquiries), thinking on vs. off, no
perspective injection, max 48 turns, soft $0.30 / hard $0.45 / 1800 s.
Report: internal `docs/canary/2026-08-22-curiosity-deep-report.md`.

| | D1 (thinking on) | D2 (thinking off) |
|---|---|---|
| Turns | 42 | 48 (max) |
| Cost / wall time | $0.107952 / 797.77 s | $0.068200 / 455.72 s |
| Stop reason | live pre-call hard cap projection | turn boundary |
| Terminal | `staged_unpublished` | `staged_unpublished` |
| Staged | 15 objects / 35 assertions / 5 inquiries | 4 isolated objects / 0 assertions / 0 inquiries |
| Cross-wake continuity | all 3 legacy inquiries taken up, 1 resolved; 12+ references to existing objects | none |
| Event-time filling | objects 6/15 | objects 3/4 |

Total spend $0.176152 (authorized $1.00). Neither wake finalized: under the
minimal prompt the deep posture never produced a natural stop signal, so the
working graphs remain in staging awaiting an explicit resume. This is a real
finding about prompt design — not a protocol failure.

### 2.3 Earlier documented runs (summary)

| Run | Wakes / commits | Result |
|---|---|---|
| Phase 4 isolated four-wake chain (2026-08) | broad→broad→deep→deep, 4 cognition commits | 38 assertions / 11 inquiries, 0 illegal omissions |
| Broad-search experience canary (2026-08-15) | fresh broad → same-world broad → same-world deep, 3 commits | 36 assertions / 7 inquiries, cost $0.191991, behavior acceptance MIXED/OPEN (documented honestly: inconsistent attachment, missing event times in 21 assertions) |

### 2.4 What real wakes have NOT yet shown

- Cross-wake name/alias reuse (same entity under a new name) — not covered.
- A real-model natural finalize under the minimal deep prompt — D1/D2 ended
  `staged_unpublished` (see above).
- Relationship-object materialization (proper nouns inside literal
  relationships) — an open follow-up from the 8-18 canary.
