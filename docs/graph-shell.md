# Graph Shell

> The working-graph edit protocol: how the agent observes, edits, reviews, and
> publishes structured cognition. Code anchors: `world/staging.py`,
> `world/preflight.py`, `world/finalize.py`, `world/tools.py`,
> `world_agent/graph.py`.

## 1. Two graphs

- **Working graph** — this wake's editable staging area. Durable in the world
  DB; visible to the owning wake; invisible to every other wake until
  finalize. Status: `active` → `finalized` | `abandoned`.
- **Formal graph** — published cognition, read by memory tools. Future wakes
  build on it. Only `finalize_graph` moves rows from working to formal.

```text
graph_patch ──► working graph ──► graph_inspect / graph_diff ──► finalize_graph
                                                                     │
                                                     formal graph ◄──┘
```

## 2. Tools

| Tool | What it does | Contract notes |
|---|---|---|
| `graph_patch` | Applies a batch (≤ 20 items) of `new_object` / `assertion` / `new_inquiry` / `inquiry_resolution` / `observation_link` / `object_update` to the working graph | each item needs `op_id` (idempotency key); staged ids are host-issued; identity pre-check on create; malformed batch → wholesale rejection, no side effects |
| `graph_inspect` | Reports working-graph state: counts, blockers, warnings, `readiness` (`ready`/`not_ready`); `item_id` → one item's state + patch history | read from durable staging, authoritative after resume |
| `graph_diff` | Lists this wake's staged changes vs. the formal graph (formal-before / staged-after) | every entry marked `unpublished`; committed changes live in `memory_changes` |
| `finalize_graph` | Publishes the working graph: deterministic compile → Store hard gates → one transaction | zero arguments; the **only** publication entry; repeat calls return the durable receipt |

Patch items are visible on the next turn. After a durable finalize receipt,
`graph_patch` on the same wake fails closed (`wake_closed`).

## 3. Idempotency

`staged_patch_receipts` maps `op_id → payload_hash → result_json`:

- same `op_id` + identical payload → original result returned (`replayed: true`);
- same `op_id` + different payload → explicit `op_id_reused` conflict;
- malformed or unknown tool calls → typed limitation, no state change.

This makes retries and crash-resume safe without the model having to
remember what it did.

## 4. Identity

- On `new_object`, the host runs an identity pre-check against existing
  objects (name/alias similarity). A candidate hit returns
  `needs_identity_resolution` unless the item carries an explicit
  `confirm_distinct` decision, which is persisted in `identity_basis_json` —
  the model's reasoning for treating two names as distinct is traceable.
- The host never silently merges. Reusing an existing object (referencing its
  id) is how an agent expresses "this is the same referent".
- A canonical name is not a unique identity: aliases normalize onto one object
  id, and the agent is expected to copy host-returned ids verbatim into later
  calls.

## 5. Preflight and readiness

`graph_inspect` reports blockers that are **hard gates** at finalize time:

- identity candidates unresolved;
- unknown reference (staged item points at a non-existent id);
- zero-connection object (an object with no edges — compile gate);
- missing evidence (assertion without evidence rows);
- stale base (compiled delta no longer matches the store);
- duplicate/signature conflicts;
- wake closed / writer lease lost.

Warnings are observable but do not block (e.g. thin evidence), so the host
never deletes legitimate subjective judgment on the model's behalf.

## 6. Finalize

`finalize_graph` is the single formal publication path
(`world/finalize.py:finalize_graph`):

1. acquire `wake_mutation_lock` (single writer);
2. `compile_final_delta` — deterministic compilation of staged rows into the
   formal delta, resolving references and supersede write order; compile
   failure (`FinalizeCompileError`) keeps staging intact and surfaces through
   `graph_inspect`;
3. one SQLite transaction: staging rows finalized, `finalize_receipts`
   (`INSERT OR IGNORE`), `commit_receipts`, `world_audit` row appended,
   `commit_id == "{wake_id}:finalize"`;
4. repeated finalize returns the durable receipt (`replayed: true`), never a
   double commit.

Rules:

- One wake → at most one formal cognition commit.
- The model decides when to finalize. The host never calls it — not on turn
  boundaries, not on cost or deadline boundaries.
- There is no second publication path (`submit_cognition` is retired).

## 7. Recovery

| Artifact | Role in recovery |
|---|---|
| LangGraph checkpoint | which node/state the wake was in |
| Staging + patch ledger | exact working-graph contents and op results |
| Finalize receipt | whether the wake already committed |
| World audit + commit receipts | append-only formal history |

- Crash before finalize → staging intact, wake resumable
  (`--resume --wake-id <id>`).
- Crash during/after finalize → idempotent receipt covers it; no double
  publish, no lost commit.
- `--graph-shell-status` shows the lease owner, staging, receipt, and the
  deterministic recovery action; `--graph-shell-abandon <wake-id>` releases
  the lease and marks staging abandoned in one world transaction (refuses
  owner mismatch; runs no model).
- Single-writer lease: at most one active writer per world; no TTL, no
  auto-preemption — a personal-project trade-off. A wake that ended
  `staged_unpublished` legitimately holds its working graph until resume or
  explicit abandon.
