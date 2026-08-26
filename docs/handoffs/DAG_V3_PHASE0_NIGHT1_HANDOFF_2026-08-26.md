# DAG v3 Phase 0 — Night 1 Handoff (2026-08-26)

**Branch:** `dagv2-evidence-ledger` (stays unmerged)
**Status:** Stages 0-4 of the execution plan complete for one full overnight pass; a post-fix core12 rerun (`dagv3p1_*`) is queued/running as of this handoff. Nothing here is a settled result yet — see §4's noise-floor caveat before citing any number below.

---

## 1. The master plan (full text, embedded so this handoff is self-contained)

> Source: `docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md`

# DAG v3 "Ledger": Master Plan

**Branch:** `dagv2-evidence-ledger` (stays unmerged — this is a falsification project, not a committed rewrite)
**Codename:** **DAG v3 "Ledger"** — the successor to DAG v2, named after the `TaskLedger`/`Evidence`/`Claim` sidecars that already exist in this branch as observe-only telemetry. The plan is the path to making them authoritative instead of decorative.

### Thesis (falsifiable, lexicographic)

> For cheap local models, an evidence-first executor with deterministic state and typed action selection improves grounded keystone score and requirement-coverage-per-tool-call over `sequential_react`, while reducing repeated no-progress actions — without needing a stronger planner model.

Primary metric order: (1) grounded keystone pass rate, (2) coverage among grounded runs, (3) cost (tokens + tool calls + wall time), (4) failure quality (partial/abstain vs fabrication, reported as separate outcome classes). Do not accept "the queue feels cleaner" as evidence — every new abstraction needs a control that isolates its own contribution.

### What triggered this plan

- Native graph (DAG v2) averaged 0.462 on core24 vs. 0.643 `sequential_react` / 0.663 `langgraph_react`, but a grounded reduced-scope rerun *reversed* this — graph beat sequential_react once search infra was verified healthy. The core24 gap was **confounded by infra flakiness**, not a clean architecture verdict.
- Mean graph size was only 4.6 nodes — paying coordination/merge/eval overhead without earning real breadth.
- Task 123: 43 visits, 1/4 sub-entities resolved, same subgoals re-issued 5-8 times. Root cause diagnosed as **branch isolation** (rootward-ancestry-only context, ~1000 chars/page cap, no sibling visibility).
- The `ledger_deficit_local` A/B came back null on core24, but the injector never got a chance to fire (underpowered, not disconfirming).

Conclusion: run cheap ablations on the existing engine before committing to a rewrite — find out if the bottleneck is context visibility, churn, action validity, or genuinely topology.

### Phase 0 — five ablation arms (all thin variants of the existing engine)

| Arm | Change | Falsifies |
|---|---|---|
| `graph_shared_context` | Expansion/eval sees a compact sibling-evidence digest | Branch isolation is the main failure |
| `graph_no_reexpand` | Novelty cap + deterministic repeated-goal block | Churn is the main failure |
| `graph_constrained_actions` | Codec-only (already built — just gate it on) | Malformed-action instability is the main failure |
| `sequential_react_context_matched` | Cap ReAct's context to the DAG's budget | The graph's loss is mostly a context-budget artifact |
| `evidence_queue_deterministic` | No LLM scheduler, typed state only (Phase B) | Typed shared state helps even without learned scheduling |

**Kill gate:** if `graph_shared_context` alone closes most of the gap to `sequential_react`, stop — fix DAG v2's context projection and churn guard, do not build a new engine.

### Target architecture (only if Phase 0 says topology is real)

Keep unchanged: `AgentIO`, tool connectors, native Ollama backend, grounding/coverage validators, benchmark harness, telemetry, `sequential_react`. Replace (new variant only): expansion policy, sibling/beam evaluation, tree nodes as runtime state, re-expansion/backtracking, merge nodes. Leave `IdeaNode.details` alone for now.

Core records (typed, reducer-owned, illegal transitions raise in tests):
```
Requirement.status: OPEN → SUPPORTED | BLOCKED | CONTRADICTED
Evidence: append-only
Claim.status: UNVERIFIED → VERIFIED | REJECTED | CONTESTED
WorkItem.status: PENDING → RUNNING → DONE | BLOCKED | FAILED
```
The LLM is a selector among deterministically-minted jobs, plus one constrained escape hatch (`PROPOSE_QUERY`) — never a free-form planner.

Build order, each phase gated by the one before it: **A** correctness first (final-answer contract, novelty/churn guard, model-metadata capture) → **B** deterministic queue, no LLM scheduling (must match `sequential_react` or stop) → **C** constrained selector → **D** bounded query proposal + priority score.

### Naming

DAG v2 = current engine. DAG v3 "Ledger" = the evidence-queue architecture, reserved for it only if it clears the Phase 0 and A-D promotion gates — not a rename in place.

### Promotion gates

| Component | Merge condition |
|---|---|
| Final-output contract | All adversarial offline fixtures pass — ship independent of everything else |
| Novelty/churn cap | Cuts repeated no-progress actions without lowering core task score |
| Model metadata telemetry | Immediately, once fixture + backward-compat tests pass |
| Constrained codec (repair paths) | Already shipped (`ef801010`) |
| Constrained codec (action/job selection) | Valid-action rate rises, no regression on qwen2.5:7b or weaker |
| Deterministic queue | Beats `sequential_react` on grounded keystone rate or materially cuts cost |
| LLM queue selector | Beats the deterministic queue, not just old DAG v2 |
| Full DAG v3 as default | Only after wins across ≥2 model sizes and both wide + sequential task shapes |

### Adversarial risks designed against up front

Requirement parser becoming a hidden planner; "evidence supports claim" being itself a model judgment; deterministic queue overfitting to entity-list tasks; hard gating producing "safe but useless" abstention. (Mitigations in the full doc.)

### Non-goals

No wider beams/more judge calls on the existing under-informed tree; no full `IdeaNode.details` migration; no more value-of-information scheduling before the mechanism suite exists; no bigger/stronger planner model (the thesis is same-model structure, not a stronger upstream model).

---

## 2. What got executed tonight (2026-08-25 → 2026-08-26)

Followed the approved execution plan (`/home/muk/.claude/plans/follow-the-master-plan-golden-matsumoto.md`), dispatched almost entirely to subagents to keep the orchestrating session's context small:

- **Stage 0** — `scripts/axis_queue_runner.py`: unattended sequential driver chaining `adaptive_ladder_run.py --axis ...` invocations with zero idle gap, search-infra preflight, single-instance lock, `--print-only` dry-run. Confirmed local GPU cells are **fully serialized regardless of `--jobs`** (`OLLAMA_MAX_LOADED_MODELS=1`, `local_busy` flag in `adaptive_ladder_run.py`) — "GPU always busy" means back-to-back queuing, not parallelism, on the current 3060 12GB.
- **Stage 1** — all 5 Phase 0 arms landed as opt-in `RunPolicy` flags + `_GOT_ARM_PROFILES` entries: `good_adaptive_constrained`, `good_adaptive_sharedcontext`, `good_adaptive_noreexpand`, `sequential_react_context_matched`, and an honest for-loop stub for `evidence_queue_deterministic` (Phase B's typed reducer logic deliberately NOT built yet).
- **Stage 2** — Phase A correctness track: found and fixed a real gap where the grounding gate never fires once ANY visit succeeds, even if the finalizer hedges and fabricates a number in the same answer. New deterministic `answer_contract` check, default ON. Plus model-metadata (Ollama digest/quantization/context) capture into every cell's telemetry.
- **Stage 3** — 7 mechanism-suite tasks authored and live-verified against real content (Wikipedia, SEC filings, NPS.gov, mountain-forecast.com): entity collision (303, Tay Bridge), syndicated URLs (302, Crater Lake depth), unsupported numeric (160, Apple iPhone units), stale-source conflict (304, Aoraki/Mt Cook elevation), dead-end retry cap (305, Southern African dams), wide fan-out (158, Greek island runway eligibility), narrow chain (159, Bell Rock Lighthouse → R.L. Stevenson → Mt Vaea).
- **Stage 4** — 7-entry overnight queue (`dagv3p0_*`) run to completion: mechanism suite → core24 baseline recheck → all 4 real ablation arms → context-matched sequential_react → evidence-queue stub. ~7h wall-clock, $0, zero infra failures, zero idle gaps.
- **Two bugs found via live-run analysis, fixed, merged** (not in the original plan — found by dedicated read-only analysis passes while the queue ran):
  1. **Finalize deliverable/summary field-swap** (`agent/app/idea_dag_schemas.py`, `agent/app/prompt_builder.py`) — the finalize JSON schema had zero field descriptions distinguishing the graded `deliverable` from the cosmetic `summary`; ~11% of cells (upper-bound estimate) put the real answer in the wrong, ungraded field. Fixed with explicit schema descriptions + strengthened system-prompt language. Commit `1e4f212a`.
  2. **Novelty-guard watermark scoping** (`agent/app/novelty_guard.py`, `agent/app/idea_engine.py`) — the churn guard's "did new evidence appear" check counted progress across the *whole graph*, so progress on any sibling branch made a genuinely stuck branch's repeated dead-end attempts look like progress too, defeating the guard in exactly the multi-branch scenario (task 305) it was built for. Fixed by scoping the watermark to the branch subtree; added run-level telemetry (`final_payload["novelty_guard"]`) since tonight's run had zero signal confirming the guard ever fired. Commit `9c6abbec`.
- **`core12` task subset added** (`scripts/adaptive_ladder_run.py`) — extends `smoke8`'s already-curated even spread across the 122-145 range with 4 more interleaved ids, rather than a contiguous half. Halves core24's wall-clock (~70-80min/2-arm-entry) for faster post-fix iteration.
- **Post-fix rerun queued and launched**: `scripts/axis_queues/phase0_qwen7b_rerun1.json` (`dagv3p1_*`), same 7-entry shape as tonight's batch but on `core12`, running now with both fixes in place. **Whoever picks this up next should check its results before trusting anything in §4 below.**

## 3. Commit inventory (chronological, `dagv2-evidence-ledger`)

```
536669f9  good_adaptive_constrained arm profile
1653d8da  sibling evidence digest + good_adaptive_sharedcontext arm
d988a4d3  sequential_react context cap arm
a787f655  novelty churn guard (original, later found buggy)
b8936258  evidence_queue_deterministic execution variant stub
d31df7de  novelty guard threshold doc pointer
9d4207da / 9feb84e0 / a106dc8b / 3657b340  mechanism tasks 158/160/302/304
[+ 159, 303, 305 landed in the same window, see `git log --oneline`]
78dbcb47 / d2163a2f  answer-contract gate + model-metadata capture (Phase A)
9dd6da18  cleanup: superseded 158 draft removed
11023820  master plan doc + axis_queue_runner.py
008f28c6  phase0_local / mechanism_suite_local axes + first overnight queue
d78b9012  per-entry search-infra health recheck (not just launch-time)
1e4f212a  finalize field-swap fix (cherry-picked from worktree)
9c6abbec  novelty guard watermark scoping fix
66d0cc58  core12 task subset
b0320c35  post-fix core12 rerun queue
```

Full offline suite as of this handoff: **6826 passed, 18 skipped** (`PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests`).

## 4. Night 1 results — read the caveat first

**The `good_adaptive` control arm itself scored between 0.402 and 0.540 across the 5 core24 entries it appeared in tonight (R=1 each) — a 0.14 spread on what should be identical.** None of the diffs below are distinguishable from that noise band yet. These are pre-both-fixes numbers; treat as a first look, not a conclusion.

| Entry | Comparison | Diff | W/L/T (n=24 unless noted) |
|---|---|---|---|
| mechanism suite (R=3, n=21) | good_adaptive baseline | mean 0.265 | hard by design |
| core24 baseline recheck | good_adaptive vs bare baseline | +0.129 | 4/15/5 |
| constrained-decoding | +constrained vs plain | −0.033 | 7/12/5 |
| shared-context | +sibling digest vs plain | +0.020 | 11/9/4 |
| no-reexpand (pre-fix guard) | +guard vs plain | +0.077 | 9/5/10 |
| seq_react context-matched | capped vs uncapped | **+0.059** | 11/6/7 |
| evidence-queue stub (n=8) | smoke8 | mean 0.302 | proves harness plumbing works |

**Flagged for attention**: capping `sequential_react`'s context *down* to the DAG's tighter budget *improved* its score rather than hurting it — the opposite direction from the master plan's leading hypothesis that ReAct's advantage comes from more context. Worth real attention once replicated on `dagv3p1_seqmatched`.

## 5. Recommended next steps, in order

1. **Check the `dagv3p1_*` rerun results** (launched at this handoff's writing, log at `agent/idea_test_results/_axis_queue/driver_20260826_080929.log`, run-ids `dagv3p1_mech`/`dagv3p1_core12base`/`dagv3p1_constrained`/`dagv3p1_sharedcontext`/`dagv3p1_noreexpand`/`dagv3p1_seqmatched`/`dagv3p1_eqstub`). Both bug fixes are in effect; compare against §4's pre-fix numbers to see how much the field-swap fix alone moves scores.
2. **Rep-2 bump** whichever arms still look directionally interesting post-fix, per the master plan's own rule (only bump reps for promising/borderline results, don't blanket-bump everything) — `noreexpand` and `seqmatched` are the current leading candidates.
3. **Quantify the novelty guard's actual firing rate** post-fix using the new `final_payload["novelty_guard"]` telemetry — tonight's pre-fix run had zero signal; the rerun should show real block counts if the scoping fix worked as intended.
4. Once Phase 0 arms look directionally resolved: expand to the weak-model matrix (llama3.2:1b/3b, qwen2.5:1.5b/14b) as a second overnight batch — deferred from tonight per the "qwen2.5:7b only, first batch" decision.
5. **`SHAPE_ADAPTATION_OPEN_QUESTIONS.md` Q20** (candidate-coverage gate reportedly untested) flagged by analysis tonight but not yet investigated — worth checking whether the new mechanism-suite coverage-ratio validators exercise it.
6. **GPU utilization**: currently bursty (0%→100%, `IDEA_TEST_CONCURRENCY=1` means one LLM call in flight at a time). A same-model concurrency change (`OLLAMA_NUM_PARALLEL` + relaxing `local_busy` for same-model cells) was scoped but deliberately deferred to avoid risking mid-run corruption — worth revisiting once a batch isn't already in flight. Hardware note: a 16GB RTX 5070 Ti upgrade is expected within the week, which gives real headroom for this; a Jetson Orin Nano Super 8GB is also planned as a second, fully independent local host (no `OLLAMA_MAX_LOADED_MODELS` contention with the main card) for the bottom-of-ladder weak models once wired in.
