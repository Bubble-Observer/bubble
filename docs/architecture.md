# Architecture

> This document describes the **current** code. Paths and symbol names are
> stable anchors; line numbers are omitted on purpose. Everything here can be
> verified in `src/` and `tests/`.

## 0. Component inventory

| Component | Module | Responsibility |
|---|---|---|
| Composition root | `world_agent/cli.py` | CLI parsing, wiring, run entry `run(args, ...)` |
| Runtime graph | `world_agent/graph.py` | 5-node LangGraph (`bootstrap → agent ⇄ tools → completion_reminder / stage_unpublished`) |
| Prompt | `world_agent/prompt.py` | single-surface `graph_shell_prompt(mode, object_id, focus, ...)` |
| World store | `world/store.py`, `world/schema.py` | SQLite durable cognition; schema + additive migrations |
| Staging | `world/staging.py` | idempotent working graph (`apply_patch`, op_id ledger) |
| Preflight | `world/preflight.py` | read-only working-graph inspection, readiness blockers, diff |
| Finalize | `world/finalize.py` | **only** formal publication: deterministic compile + one transaction |
| Recall | `world/recall*.py` | memory read side: search/read/compare/expand/overview/changes/evidence/inquiries |
| Tool facade | `world/tools.py` | `WorldTools`: 23 schemas; Graph Shell exposes 19 (see §4.5) |
| Adapters | `channels/` | source acquisition: bilibili / nga / hupu / public-web / replay |
| Model client | `gateway/client.py`, `gateway/deepseek_client.py` | provider-neutral `ModelClient`, DeepSeek implementation |
| Guards | `world_agent/live_cost_guard.py`, `live_deadline.py` | live-run pre-call hard cap (G1) and deadline |
| Ledger | `world_agent/model_calls.py` | per-call model accounting |
| Report | `world_agent/run_report.py` | end-of-wake run report |
| Runtime | `runtime/inquiry_lease.py`, `runtime/curiosity_log.py` | inquiry leases, append-only exploration log |
| Security | `security/content_boundary.py`, `security/url_policy.py` | external content never becomes instructions; URL safety |
| Config | `config.py` | `Settings` from `.env` (pydantic-settings) |
| Personal console | `console/app.py`, `profiles.py`, `runs.py`, `inspection.py` | local-only Starlette UI: profile registry, single-writer runs, read-only memory browsing, pending-wake manual finalize (see §11) |

## 1. Run entry

### 1.1 CLI / composition root

`bubble-world` and `scripts/run_world_agent.py` both call
`leave_information_bubble.world_agent.cli:run` — there is exactly one
composition root. A minimal invocation:

```bash
bubble-world --thread-id demo-1 \
  --world-db world.sqlite3 --runtime-db runtime.sqlite3 \
  --mode broad --domain lol_cn
```

- `--world-db` and `--runtime-db` must be different files.
- `--mode broad|deep` selects the wake posture; `--object-id <id>` is valid
  only in deep mode (a starting object).
- Fresh wake vs. resume: `--resume --wake-id <id>` restores the wake from its
  LangGraph checkpoint; identity fields (thread, wake, execution mode, world
  identity, domain, object seed) are re-validated against the request and fail
  closed on mismatch (`graph.py:_ensure_protocol`).
- Provider/model injection: the model client is built from `Settings`
  (`deepseek_api_key`, `deepseek_base_url`, `deepseek_model`); `--thinking` /
  `--reasoning-effort` forward reasoning mode to the provider.
- Offline: `--replay-fixture` + `--scripted-model-fixture` run the same graph
  with deterministic fixtures (this is what the offline demo and G1–G7 use).

### 1.2 Graph nodes and state

```text
START → bootstrap → agent ⇄ tools
                  tools → completion_reminder → agent | stage_unpublished → END
```

| Node | Responsibility |
|---|---|
| `bootstrap` | protocol guard, single-surface system prompt, world anchors (object/assertion/inquiry counts, local time) |
| `agent` | one model call (`invoke_tools`), turn/cost/deadline budget checks; stops at legal terminal states |
| `tools` | schema-first validation (single schema source, F-03), Graph Shell allow-list, batch rejection normalization |
| `completion_reminder` | retry signal after the tool loop terminates (back-off after 3 error streaks) |
| `stage_unpublished` | legal end on turn/cost/deadline/model-error boundary; working graph stays in staging |

Durable state lives in the world SQLite (cognition) and the runtime DB
(checkpoint + staging + ledger). `GraphShellState` (a `TypedDict`) carries
runtime fields: `messages`, `turn_count`, `total_cost_usd`, `terminal_status`,
`finalize_status`, `finalize_receipt`, `resume_allowed`, …

### 1.3 Terminal states

- `published` — the model called `finalize_graph` and the commit landed
  (`finalize_status = committed`, `commit_id == "{wake_id}:finalize"`).
- `staged_unpublished` — a legal boundary (turns / cost / deadline / model
  error) ended the wake with an un-finalized working graph. Nothing leaked to
  the formal graph; the wake can be resumed (`--resume --wake-id`) or
  abandoned (`--graph-shell-abandon <wake-id>`).
- Host behavior rule: the host never calls `finalize_graph` on the model's
  behalf, not even on budget or deadline boundaries.

## 2. Source acquisition and Observation

```text
discover/search/open/follow → source material → Observation
→ tool-time durable persistence → assertion evidence reference
```

- Channel adapters (`channels/base.py:ChannelAdapter`) answer
  `discover(ScanRequest)` → `ScanResult` and `hydrate(HydrationRequest)` →
  `ObservationBatch`, with `limitations` as the platform's honest answer, not
  an error. One adapter failing never kills the run.
- `WorldTools` projects adapter results into bounded, model-readable factual
  feedback (`outcome / scope / completeness / error`), never into retry or
  pivot instructions.
- Acquisition and recall faithfully return depth, source kind, time,
  reliability/limitations, and evidence role; tool-time persistence
  (`world/materials.py` observation bodies, `store.memory_commit` for
  observations only) is separate from formal cognition finalize.
- An **Observation is a traceable evidence anchor** (source_uri, source_kind,
  depth, observed_at, excerpt, content_ref), not a generic text chunk.
  Assertions reference observations as evidence; observations survive across
  wakes and are reused as evidence by later assertions.
- Only acquisition actually performed is claimed: capability baselines are
  frozen per adapter (`channels/` + `docs`), and public-web shell pages are not
  persisted as full material (qualification happens upstream in acquisition).

## 3. Memory and world model

Five core durable structures (schema: `world/schema.py`):

| Structure | Table(s) | Meaning |
|---|---|---|
| Object | `objects`, `object_aliases` | entity / event / concept; canonical name + normalized aliases; `kind`; optional `event_time_start/end` |
| Assertion | `assertions`, `predicates`, `assertion_evidence` | subject–predicate–object-or-literal with `epistemic_role`, `confidence`, evidence links, optional `supersedes_id` |
| Inquiry | `inquiries` | open question on a subject; status lifecycle new → deepen → answer → resolve; `attempt_count` audited |
| Observation | `observations`, `object_observations`, `inquiry_observations` | evidence anchor; links to objects and inquiries |
| Evidence / history | `assertion_evidence`, `supersedes_id`, `world_audit`, `commit_receipts` | every commit is appended to the audit; supersede records replacement without deletion |

Contract types live in `world/contracts.py` (`ObjectInput`, `AssertionInput`,
`InquiryInput`, `InquiryResolution`, `EvidenceInput`, `ObservationInput`,
`ObservationLinkInput`, `CognitiveDelta`, `CommitReceipt`; `ObjectKind`;
`ObservationDepth`; `EpistemicRole`).

Semantics:

- **literal assertion vs. object reference**: an assertion either carries a
  literal value (`literal_json`) or points to another object (`object_id`); a
  relationship between objects is a first-class assertion with evidence, never
  a text hint.
- **epistemic role** must be distinguished — it is the foundation of review
  criteria: `fact` / `community_view` / `semantic_explanation` /
  `agent_synthesis` / `uncertainty` / `meta_knowledge`.
- **supersede**: correcting an assertion creates a new assertion whose
  `supersedes_id` points at the old one. History is never deleted; the
  "current" view is a projection over supersede chains.
- **inquiry lifecycle**: `new` → `deepen` (more context) → `answer` /
  `resolve`; answer assertions reference the inquiry (`answers_inquiry_id`),
  resolutions are counted once per valid reference (audited, replay-safe).
- **formal view**: after finalize, the formal graph is what future wakes read
  through the memory tools (`world/recall*.py`).

## 4. Recall / memory read side

Each public read tool answers one question. All return bounded, typed JSON
with explicit scope (counts, truncation, found/missing for exact IDs) — never
instructions.

| Tool | Answers | Input | Returns | Differs from |
|---|---|---|---|---|
| `memory_recent` | What did I learn recently? | limit | recent assertions/objects, paged | time-ordered, not query-ranked |
| `memory_search` | What do I know matching this? | query + optional filters (kind, time, predicate, counts), page/count mode with cursors | ranked candidates with match reasons | ranks across the formal graph; cursor-safe paging |
| `memory_read` | Full picture of one object/assertion | exact id(s) | complete profile incl. assertions, evidence, supersede chain | exact-ID depth, not search |
| `memory_compare` | How do two candidates differ? | two ids | side-by-side fields/deltas; the agent decides, host never adjudicates | presentation only |
| `memory_expand` | What is connected to this? | object id | neighbors along relationships | graph traversal from one node |
| `memory_evidence` | What evidence backs this? | assertion/object id | linked observations with roles | evidence projection |
| `memory_inquiries` | What questions are open? | status filter | inquiries incl. attempt history | open-questions view |
| `memory_changes` | What changed since when? | `since` | committed changes in a window | formal-graph change feed |
| `memory_overview` | Shape of the formal graph? | `as_of`, limit | coverage, underexplored, repeated-inquiry points | aggregate stats, not items |

## 5. Graph Shell

The Graph Shell is the model's edit loop over the persistent graph. Four tools
(three from `WorldTools` + `finalize_graph` injected by the graph):

- **`graph_patch`** — apply a bounded batch (≤ 20) of items to the *working*
  graph: `new_object` / `assertion` / `new_inquiry` / `inquiry_resolution` /
  `observation_link` / `object_update`. Every item carries its own `op_id`
  idempotency key. Staged ids are host-issued (`{wake_id}:s{n}` / `a{n}` /
  `i{n}` / `r{n}`). Identity pre-check runs on create; a candidate hit returns
  `needs_identity_resolution` unless the item carries an explicit
  `confirm_distinct` decision (tracked in `identity_basis_json`). Rejected
  items carry structured feedback and action hints; malformed batches are
  rejected wholesale (normalized stable error loop, no side effects).
- **`graph_inspect`** — current working-graph state read from durable staging
  (survives restarts): item counts by kind, blockers, warnings, readiness
  (`ready` / `not_ready`), optional single-item history
  (`world/preflight.py:inspect_working_graph`, `staged_item_history`).
- **`graph_diff`** — this wake's staged changes relative to the formal graph
  (`world/preflight.py:diff_working_graph`); every entry carries
  formal-before / staged-after and is marked unpublished.
- **`finalize_graph`** — zero arguments; the **only** formal publication
  entry. Contract text inside the schema: the host never calls it
  automatically. Repeat calls return the durable receipt.

Patches are visible to the model on the next turn (the tools node reads
staging through the same store). After finalize, `graph_patch` on the same
wake fails closed (`wake_closed`).

## 6. Working graph / staging

- Staging is **durable and world-scoped**: rows in the world DB
  (`staged_objects`, `staged_assertions`, `staged_inquiries`,
  `staged_patch_receipts`), visible to the owning wake, invisible to other
  wakes until finalize.
- Patch ledger: `staged_patch_receipts` records `op_id → payload_hash →
  result_json`. Replay with identical payload returns the original result;
  replay with a different payload is an explicit `op_id_reused` conflict.
- Malformed tool calls and rejected patches never mutate staging
  (schema-first validation before dispatch).
- Status lifecycle: `active` → `finalized` (on finalize) or `abandoned`
  (`--graph-shell-abandon`); `staged_unpublished` wakes can be resumed.
- A `wake_mutation_lock` (writer lease, `world/writer_lease.py`) serializes
  graph_patch and finalize; closed-wake guard rejects new work after a durable
  finalize receipt.

## 7. Preflight and deterministic defenses

| The agent decides | The host guarantees deterministically |
|---|---|
| whether two names are the same referent | ID/schema/reference validity; identity pre-check on create; `confirm_distinct` tracked |
| create vs. reuse | working/formal isolation |
| attribute vs. relationship | evidence rows must exist |
| whether to supersede | readiness checks + Store hard gates at finalize |
| when to finalize | transaction, idempotency, receipt, audit |
| how to express cognition | no silent merging, no false success |

Common blockers reported by `graph_inspect` (with structured feedback):
identity candidates, unknown references, zero-connection objects (compile
gate), missing evidence, stale base, duplicate/signature conflicts, wake
closed, writer lease lost. Full error-code lists stay internal; the model sees
bounded, actionable feedback.

## 8. Finalize and formal commit

```text
working graph → inspect/preflight → deterministic compile → Store hard gates
→ one SQLite transaction → formal graph + audit + receipts + staging finalized
```

`world/finalize.py:finalize_graph(store, wake_id)`:

1. Takes `wake_mutation_lock`.
2. `compile_final_delta` — **deterministic** compile: staged objects /
   assertions / inquiries → formal delta (`_resolve_object_ref`,
   `_resolve_inquiry_ref`, `_resolve_assertion_ref`, `_supersede_write_order`,
   `_merge_evidence`). Compile failures (`FinalizeCompileError`) keep staging
   intact and are reported through `graph_inspect` / feedback.
3. `_make_finalizer` — one transaction: staging rows finalized
   (active → finalized), `finalize_receipts` inserted
   (`INSERT OR IGNORE`; a concurrent deterministic compile converges on the
   winner's receipt), `commit_receipts` inserted, `world_audit` row appended,
   `commit_id == "{wake_id}:finalize"`.
4. Repeat finalize returns the durable receipt with `replayed: true` — no
   double commit.

There is no second publication path: legacy `submit_cognition` is retired;
`finalize_graph` is the only route.

## 9. Recovery and the single writer

Authoritative sources, in order of trust during recovery:

| Artifact | Authority |
|---|---|
| LangGraph checkpoint | which node/state the wake was in |
| Durable staging + patch ledger | what the working graph contains and every op's exact result |
| Finalize receipt | whether the wake already committed (replay, no double commit) |
| World audit + commit receipts | append-only history of every formal commit |

- CP1 / CP2 / CP6 (crash-point semantics): a crash before finalize leaves
  staging intact and resumable; a crash during/after finalize is covered by
  the idempotent receipt — the system never publishes twice and never loses a
  published commit.
- The writer lease (`world/writer_lease.py`) guarantees **at most one active
  writer per world**. No TTL, no auto-preemption: a crashed wake holds the
  lease until explicitly abandoned (`--graph-shell-abandon <wake-id>` after
  `--graph-shell-status` shows the owner). Deliberate single-user trade-off;
  not production-grade concurrency.
- `staged_unpublished` + explicit `--resume` / `--graph-shell-abandon` are the
  normal crash/restart workflow; no manual SQL is required.

## 10. Evaluation

Two strictly separated layers — see `docs/evaluation.md` for the full picture:

1. **Offline contract evaluation** (no model, no network):
   - A–J contract acceptance scenarios (`tests/world/test_contract_acceptance.py`);
   - G1–G7 golden scenarios driving the real CLI with scripted models
     (`scripts/graph_shell_golden_scenarios.py`, pytest mirror in
     `tests/world_agent/test_graph_shell_golden_scenarios.py`);
   - strict `graph_patch` contract, defect-proof, recovery, golden metrics;
   - full offline suite (unit + integration + scenarios).
   Scripted scenarios prove the protocol and the state machine. They do not
   prove what a real model will do.
2. **Real wake evidence** (documented, paid, isolated): latest runs and exact
   costs are in `docs/evaluation.md`, separated from scripted metrics.

## 11. Personal console

`bubble-console` is a local-only management UI that reuses the cognition
chain without duplicating it — it calls the same `world_agent.cli:run`
composition root in-process via asyncio.

| Module | Responsibility |
|---|---|
| `console/app.py` | Starlette API + static files; defaults to `--host 127.0.0.1 --port 8765`, non-loopback hosts are rejected |
| `console/profiles.py` | `AgentProfile` registry (`data/run-configs/agents/*.json`): stable IDs, DomainFocus, Broad/Deep defaults, pairwise-distinct world/runtime paths, editable `operator_instructions`; never stores keys or cookies |
| `console/runs.py` | `RunManager` — single writer (at most one active wake), in-process asyncio call into `world_agent.cli.run`, ring buffer of run events |
| `console/inspection.py` | strictly read-only per-profile cognition browsing (objects/assertions/inquiries) with explicit run↔commit linkage; missing DBs are never created, old schemas degrade safely |
| `console/local_settings.py` | `.env` read/write for model keys (local-only) |

API surface: `/api/system` (`key_configured: bool` only), `/api/local-settings`,
`/api/profiles` (create/read/update + quick-create + clone + prompt-preview, no delete), read-only
`/api/profiles/{id}/memory/*` endpoints, `/api/runs` (start/query), and
`/api/profiles/{id}/pending-wakes` + `{wake_id}/finalize` — the manual
finalize endpoint reuses the same deterministic finalize entry as the CLI
management flag (idempotent, no model), so the console can never diverge
from it. The parameter contract between console and CLI is pinned in
`docs/console-run-parameters.md`.

The console is single-user and loopback-only by design: it is a personal
observatory, not a web service. Frontend pages are plain HTML/CSS/JS with no
framework. `web/` (the public edition viewer fed by `web/data`) is a separate
read-only front end and is unrelated to the console.
