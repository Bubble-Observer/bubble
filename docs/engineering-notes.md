# Engineering Notes

A short record of key architecture decisions, kept intentionally brief. Full
reasoning and review history live in the internal repository.

## 1. Working graph → formal graph (two-layer state)

**Decision:** all edits land in a durable working graph (staging); only an
explicit `finalize_graph` compiles them into the formal graph in one
transaction.

**Why:** a single layer forces a choice between "every model output is
permanent" (garbage accumulates) and "the host writes on the model's behalf"
(the model cannot review its own work). Staging makes review a first-class
step: `graph_inspect` reports readiness blockers, `graph_diff` shows the
change set, and the model decides when to publish.

## 2. Deterministic host, subjective agent

**Decision:** everything the host does (identity pre-checks, references,
evidence, supersede chains, readiness, compile, transaction, receipts) is
deterministic and fail-closed. Everything that requires judgment (is this the
same referent? is this a fact or a community view?) is the model's call,
recorded with rationale (`identity_basis_json`, `EpistemicRole`).

**Why:** mixing the two makes neither auditable. Splitting them means review
can distinguish "the host got the protocol wrong" from "the model made a
judgment we disagree with" — and corrections are always a new assertion
(supersede), never a silent rewrite.

## 3. One publication path, one commit per wake

**Decision:** `finalize_graph` is the only path from working to formal; one
wake yields at most one `commit_id == "{wake_id}:finalize"`; repeat calls
return the durable receipt.

**Why:** multiple publication paths (the earlier `submit_cognition` design)
create unreachable legacy branches that rot. A single idempotent path makes
crash-recovery simple: the receipt is the authority on whether the commit
happened, so resume can never double-publish.

## 4. Evidence-anchored cognition

**Decision:** assertions and objects carry explicit links to observations
(sources) as evidence; an assertion without evidence rows does not finalize.

**Why:** without an evidence anchor, a "correction" is indistinguishable from
a hallucination. The traceability constraint is what makes cross-wake
continuity (reuse, supersede, resolve) meaningful — every step can be audited
back to a source.

## 5. Scripted evaluation and real wakes are separate evidence

**Decision:** offline scenarios (A–J, G1–G7) prove protocol and state-machine
correctness; real-wake runs (documented, isolated, paid) are the behavioral
evidence. Never substitute one for the other.

**Why:** a scripted model will always do what the fixture says — that proves
the machine, not the agent. The honest question "will a real model reuse,
supersede, and resolve?" is answered only by real runs, and the answer so far
is: yes (2026-08-18 canary), with known gaps (2026-08-22 no-natural-stop
finding).
