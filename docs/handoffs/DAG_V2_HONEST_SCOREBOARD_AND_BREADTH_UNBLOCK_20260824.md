# DAG v2: honest scoreboard, breadth unblock, and a closed-environment suite

**Date:** 2026-08-24 · **Model:** qwen2.5:7b (local ollama) + a gpt-5-mini cross-check
**Spend:** $0 local; ~870 Serper searches (quota 50,000); < $0.20 paid API
**Tests:** 6182 → 6365 passing, 18 skipped

---

## 1. What this cycle was for

A four-way baseline (96 cells, 24 tasks) had put the native DAG engine last:

| arm | score | total nodes | visits | searches |
|---|---|---|---|---|
| langgraph_react | 0.800 | — | 3.4 | 21.3 |
| sequential_react | 0.528 | — | 3.8 | 21.0 |
| graph (native DAG) | 0.335 | **4.6** | 1.3 | 8.7 |
| sequential (DAG, chain-forced) | 0.136 | 4.2 | 0.6 | 6.3 |

The headline read "the DAG loses". The mechanism forensics said something narrower and more
actionable: **the DAG's mean graph was 4.6 nodes against a `max_branching` of 5.** The cap was
not shaping the fan-out, it *was* the graph. A 7-candidate question was being answered by a
structure with fewer nodes than the question had parts, and since a candidate typically costs
two nodes (search, then visit), two or three of seven were all that was ever reachable.

So the parallelism thesis had not been falsified. It had never been *tested* — the mechanism it
depends on could not run.

---

## 2. Defects found and fixed

### 2.1 The scoreboard was dishonest (Track A1)

`SimpleMergePolicy.merge()` recursed to the root and stamped a **keyword-overlap**
`goal_achieved` onto it at merge-*creation* time (`merge.py:738`), before any merge node's LLM
call. The real coverage-aware merge later computed `goal_achieved=False` +
`merge_should_skip=True` on the merge node, the engine marked it SKIPPED — and
`build_final_payload` read the **root first** (`idea_finalize.py:1115`), falling back to a
*disjunction* over merge nodes that could raise False to True but never lower True to False.

Cells reported `success=True` at 2-of-7 coverage. The `merge.py:759-770` docstring had warned
about exactly this and was never closed.

**Fixed:** the cheap pre-check now writes `goal_achieved_provisional`; `resolve_goal_achieved()`
prefers the **root-most** merge node's verdict and treats `merge_should_skip` as a veto;
`MergeLeafAction`'s failure branch now propagates its negative verdict upward, symmetric with
the success branch.

> **Not fixed, flagged:** `success` is still `deliverable and (goal_achieved or not
> has_critical_failures)` (`idea_finalize.py:1138`), so a cell with no *critical* failures
> reports `success=True` regardless of coverage. Observed live: `success=True`,
> `goal_achieved=False`, score 0.18. `goal_achieved` is now honest; `success` is a weaker claim
> than its name suggests, and the forensics CSV has a `success` column.

### 2.2 Nothing was traceable (Track A2–A4)

Three independent losses, compounding:

* all six execution variants **unconditionally `unlink()`** the trace on success — a trace
  existed only if the run *crashed*;
* `telemetry_raw` was **popped wholesale** from the result JSON below verbosity 3;
* `record_timing` **received `started_at` and discarded it**, so no artifact anywhere carried a
  call start time and concurrency could only be inferred from suspiciously-equal durations.

Plus a latent one: no trace path carried the repeat index or config fingerprint, while
`TraceRecorder` opens in **append** mode — so simply un-deleting traces would have interleaved
concurrent cells into one corrupt file. (`graph` and `sequential` were worse: their trace path
carried no variant component at all, so those two arms shared a path.)

**Fixed:** `IDEA_TEST_KEEP_TRACES` guards all six unlinks; `build_trace_path()` gives every cell
a unique name matching its result JSON; `record_timing` keeps a session-relative
`[t_start, t_end]`; `IdeaNode` gains `started_at`/`ended_at` on the same timeline;
`slim_telemetry_raw()` retains timings/decisions/events and drops only the bulk;
`IDEA_TEST_CAPTURE_LLM_IO` decouples raw capture from report verbosity; the langgraph arm's
post-hoc `connector_io` events are marked `synthesized: true` so ordering analysis refuses them.

**Verified live** on a real cell: 25/25 timings carry intervals, **3 overlapping `llm_call`
pairs detected**, node intervals stamped, trace retained with a matching filename.

### 2.3 The graph was capped below the size of the question (Track B1)

`max_branching = 5` is a **global** budget shared by every task shape. Raising it
unconditionally risks chain and narrow tasks. Made demand-driven instead: when the mandate
enumerates N candidates (the same parser the coverage gate uses, which fails open below two
names), the **root** may widen to N, bounded by `breadth_branching_max`. Prompt and slice are
computed from the same value, so the model is never asked for children that get truncated.

### 2.4 The root could never re-expand (Track B2) — the keystone defect

`step()` gated expansion on `if not node.children`. Once the root had children it could never be
expanded again. Yet **every** remediation path ends by setting `root.status = ACTIVE`:
the coverage extension (whose own comment says *"re-activate the root so the extended budget
re-expands and re-checks the missing candidates"* — which the code could not do), the grounding
replan, and both budget-exhaustion branches. A re-activated root with children falls to
`_handle_intermediate_node`, which only picks among nodes that **already exist**.

The engine's entire "notice we are incomplete → go get more" machinery re-entered a structure it
was architecturally forbidden from widening.

**Fixed:** an explicit, single-use widen request, bounded by `root_reexpansion_max`, carrying
the existing children into the prompt as exclusions (without which the model re-emits its first
answer).

### 2.5 The coverage gate measured visits but every lever created searches (Track B3)

`candidate_coverage` counts **only successful visits**. But its remediation reached the
visit-injecting hooks solely through `_grounding_replan`, and each of those early-returns unless
the mandate carries a `must_visit` phrase or nav targets — which an ordinary "for each of the
following, find X" mandate does not.

**Fixed:** `inject_coverage_visits()` deterministically mints a VISIT per missing candidate,
reusing an already-completed search for that candidate where one exists. Deterministic on
purpose — the same reasoning as the gate itself, which ignores what the model says about its own
progress.

### 2.6 The planner was starved of evidence (Track B4)

The expansion prompt truncated each ancestor's page content at a **hard-coded 1000 chars** with
no config knob, from ≤5 ancestors, along the **root-ward path only** — a node planning its next
hop can never see what a sibling found. `sequential_react` sees **6000 chars of every page** in
its linear history.

**Fixed (knob only):** lifted to `expansion_ancestor_content_chars`, default preserving 1000
exactly. This exists to be raised and measured. The evidence says the DAG is *starved*, not
overloaded (1.3 visits vs 3.8), so no trimming was done anywhere in this push.

### 2.7 The search key failed silently (found incidentally, high value)

`SERPER_KEY` was read with a bare `os.environ.get()` and **never stripped**. `services/keys.env`
is CRLF-terminated, so the trailing `\r` produced a **403 Unauthorized**, which
`ConnectorSearch` turns into *"no results"* rather than an error. Runs completed, scored badly,
recorded `infra.failed = False`, and read as **model** failures.

Reproduced live: a 42-character read of a 40-character key, `search.count = 0` on every cell.
The LLM key path already stripped — which is why the model worked and only search was dead, the
most confusing possible presentation. This is almost certainly the real cause of the
"2026-08-23 Serper outage"; the quota is 50,000 searches and was never the problem.

**Fixed:** `_clean_secret()` in `services/shared/connector_config.py`, 18 tests.

---

## 3. Results

### 3.1 The mechanism chain (n=1, task 152, directional)

Each fix's effect is visible in the counters, not just the score:

| arm | score | root kids | nodes | visit nodes | **visits** | searches |
|---|---|---|---|---|---|---|
| control (all off) | 0.143 | 6 | 7 | 4 | 4 | 9 |
| B1+B2 | 0.071 | 10 | 11 | 2 | **2** | 55 |
| B1+B2+B3 | 0.286 | 16 | 17 | 10 | **10** | 45 |

B1+B2 removes the structural cap and the freed budget flows into **searching** — visits halve,
score drops. B3 redirects the same capacity: visits 2→10 with *fewer* searches. Adding budget to
a system whose remediation points at the wrong action makes it worse, which is exactly what the
earlier n=24 coverage A/B measured as "null".

### 3.2 The A/B (24 cells, paired, alternating condition order per rep)

Data: `docs/handoffs/data/bstack_ab_20260824.csv`

| shape | n | Δ (bfix − control) | dz | p (raw) | p (Holm) | W/T/L |
|---|---|---|---|---|---|---|
| **wide breadth** | 6 | **+0.159 ± 0.090** | +1.85 | **0.031** | 0.125 | **6/0/0** |
| chain | 3 | −0.022 ± 0.191 | −0.29 | 1.000 | 1.000 | 1/0/2 |
| narrow breadth | 3 | −0.042 ± 1.400 | −0.07 | 1.000 | 1.000 | 1/1/1 |
| pooled | 12 | +0.063 ± 0.171 | +0.24 | 0.442 | 1.000 | 8/1/3 |

Mechanism, wide breadth:

| metric | control | bfix | Δ |
|---|---|---|---|
| visits | 2.2 | 9.0 | **+6.8 ± 1.4** |
| searches | 5.0 | 79.5 | +74.5 ± 30.9 |
| nodes | 4.8 | 24.5 | +19.7 ± 6.1 |
| llm calls | 6.3 | 20.7 | +14.3 ± 4.3 |
| prompt tokens | 23,300 | 64,768 | +41,468 ± 21,971 |

Sanity: control had **5/12** cells with zero successful searches or zero visits and **1 fully
ungrounded**; bfix had **0/12** of either.

**How to read this.** Wide breadth is the pre-specified target (B1 and B3 are breadth
mechanisms; chain is the regression control). The raw p=0.031 with a clean **6/0/0** sweep and
dz=1.85 is a strong signal, but **n=6, and Holm across the four shape tests puts it at 0.125 —
it does not clear 0.05 after correction.** Treat this as a well-supported direction that needs a
larger breadth-only run to call, not as a settled result. The chain control shows no regression,
which is the thing B1 most risked.

**Cost.** ~2.8× prompt tokens on breadth. Per this repo's standing framing, that premium is the
strategy rather than a defect — but 79.5 searches per cell is high enough to suggest B1+B2 are
still pouring budget into searches and B3 is adding visits *on top of* that rather than
displacing it. Worth attacking next.

### 3.3 Cross-model check (gpt-5-mini) — PARTIAL, and it surfaced a real risk

The control cell on task 152 scored **0.036 with 1 visit and a 4-node graph** — the same
starvation shape as qwen2.5:7b, from a much stronger model. Expected, since `max_branching` is a
config ceiling rather than a model behaviour: no amount of capability gets past a cap that stops
the graph being built. This makes a qwen-specific artefact unlikely (cf. task 154's arithmetic
bug, which *was* model-specific).

**The matching bfix cell never completed**, and the trace says why:

```
{"name": "llm_call", "duration": 180.088, "t_start": 230.160, "t_end": 410.248,
 "success": false, "payload": {"model": "gpt-5-mini"}}
```

A 180s LLM call that failed — the `final_timeout_seconds: 180` cap. **Widening the graph raises
the final-synthesis prompt enough that a slow reasoning model times out.** Invisible on qwen
(fast, local); fatal on gpt-5-mini. This is a genuine operational risk of the B-fix stack and
must be addressed before any paid breadth run: either raise the final timeout for widened runs,
or bound what the merge is handed.

Worth noting how this was found. **Before this cycle it would have been undiagnosable** — the
trace was deleted on success, trace filenames collided across cells, and `record_timing` threw
away the start time. The `t_start`/`t_end` pair added in A3 is what turned "the cell just never
appeared" into "a 180.088s LLM call failed at offset 230". That is the whole argument for Track A
in one artifact.

**Status: the paid cross-check is incomplete** (1 of 4 cells). Do not cite a paid A/B.

---

## 4. Closed-environment suite (Track D)

### 4.1 All four arms now share one sandbox surface

Previously only the native engine could manipulate a sandbox filesystem (`SandboxToolPack`), and
the codebench matrix drove only the *compiled* variant — so a closed-environment task was
measurable on **one arm**, which cannot support any DAG-vs-linear claim.

`agent/app/sandbox_tool_surface.py` is now the single definition of the surface (eight file
verbs), consumed by `sequential_react` and `langgraph_react` through the existing shared
dispatcher rather than a third copy of the translation. `codebench_run_task.py` takes `--arm`
(`compiled | graph | sequential | sequential_react | langgraph_react`), all driven against the
same workdir and prompt — only `compiled` additionally gets `plan.json`, which is the difference
being measured.

A parity test pins the shared surface against `SandboxToolPack.ACTION_CLASSES`: if they drift,
the arms are being compared on **capabilities** rather than reasoning. `run_python` /
`run_pytest` / `search_web` are reachable through the dispatcher and are refused on **every**
arm.

### 4.2 c53 — bin rebalancing

`agent/app/idea_code_tests/test_c53_bin_rebalance.py`. Four container files, twelve items; move
items so every container totals exactly 27. The deliverable is the **final filesystem state**.

* **Constructed ground truth** — nothing to recall, nothing to fabricate; the grader re-adds
  every total from the files.
* **Instance selected against a property:** descending first-fit greedy **fails**, so a model
  that reaches for the obvious heuristic lands in a dead end. Solvable nonetheless, confirmed by
  a **second, differently-formulated solver**; exactly 14 assignments work.
* **No code execution** — the sandbox surface is file verbs only, so the agent must do the
  arithmetic itself and write intermediate results down.
* **Staged**: one canonical check per stage (parse → weights unaltered → items conserved →
  **keystone: totals hit target** → totals.txt agrees → moves.txt reconciles).
* **Adversarial cases pinned as caught**: rewriting a weight, dropping an item, a totals file
  that disagrees, an unexplained move.
* Fan-out-then-merge shape by construction: per-container totals are independent; the rebalance
  is the merge.

### 4.3 c53 live calibration — RAN, and the verdict is "too hard as pitched"

Two live runs through the real codebench harness (Docker, `--internal` network, canonical tests
re-injected, $0):

| model | score | keystone | what it actually did |
|---|---|---|---|
| qwen2.5:7b | **0.500** | ✗ | wrote `survey.txt`, then wrote a `balance_containers.py` it never closed the loop on. **Containers left byte-identical to the start** — no item was ever moved. |
| qwen2.5:14b | **0.167** | ✗ | wrote `alpha.txt` = `gear 8 / tool 6 / part 13` — a container totalling exactly 27, built from **items that do not exist**. Fabrication, caught by `test_items_are_conserved`. |

Two things this establishes and one it doesn't.

**Establishes — the anti-fabrication design works.** A naive validator asking only "does each
container total 27?" would have **credited 14b's answer as correct**. That is exactly why
conservation is a separate stage rather than folded into the keystone.

**Establishes — the graduated scoring discriminates failure modes.** 0.5 vs 0.167 separates
"didn't attempt the reasoning" from "attempted and fabricated", which a pass/fail task could not.

**Does NOT establish that the task is usable as a benchmark.** Neither model earned the keystone,
so the task currently has **no headroom** and cannot measure improvement. Solvability is proven
offline (reference solver + canonical tests accept it), but no *live* solve exists at any model
tier. Given this repo's standing problem that the active suite already floors weak models, c53
should NOT be added to a measurement suite until either:

  * a strong model is shown to solve it live (confirming it is achievable in-harness), **or**
  * an easier sibling task lands (fewer items / a looser band) so there is a rung to climb.

**Calibration status: self-solve PASSED, live-difficulty RAN but FAILED the "not unsolvable"
half.** Treat c53 as a promising instrument, not a calibrated one.

---

## 5. Flags shipped (all default OFF unless noted)

| flag | default | what it does |
|---|---|---|
| `breadth_aware_branching_enabled` | off | root widens to the enumerated candidate count |
| `breadth_branching_max` | 8 | ceiling on the widened root |
| `root_reexpansion_enabled` | off | remediation may re-expand a node with children |
| `root_reexpansion_max` | 2 | bound on re-expansions per node |
| `coverage_visit_injection_enabled` | off | coverage remediation mints visits |
| `coverage_visit_injection_max` | 8 | per-pass ceiling |
| `expansion_ancestor_content_chars` | 1000 | planner's per-page view (unchanged default) |
| `IDEA_TEST_KEEP_TRACES` | off | retain per-cell JSONL traces |
| `IDEA_TEST_CAPTURE_LLM_IO` | off | raw prompt/completion into the trace only |
| — | **on** | A1 goal_achieved precedence (correctness) |
| — | **on** | A3 call/node intervals, A4 slimmed telemetry retention |
| — | **on** | search-key whitespace stripping |

---

## 6. What is NOT done

1. **A5 four-way re-baseline on the honest scoreboard.** The 0.800 / 0.528 / 0.335 / 0.136 table
   remains *provisional, measured on a dishonest gate*. It also had **fixed arm order** (graph
   first, langgraph last) and the order-control experiment was never run — any re-baseline must
   alternate arm order per rep.
2. **The B5 gate.** Wide breadth improves, but does the DAG now beat `seq_react` on breadth? Not
   measured — this A/B was DAG-vs-DAG.
3. **Track C (frontier restructure).** Not started; it was gated on B5.
4. **c53 headroom** — live-calibrated (§4.3) but unsolved at both tiers tested; needs a live
   strong-model solve or an easier sibling before it can measure anything.
5. **The widened-graph final-synthesis timeout** (§3.3). A blocker for any paid breadth run.
6. **B4's sibling-summary channel.** Only the config knob landed; a node still cannot see what a
   sibling found. The 1000→6000 A/B is unrun.
7. **`success` semantics** (see §2.1).

## 7. Next steps, in order

1. Re-baseline four-way on the honest scoreboard, alternating arm order. Everything else is
   measured against this.
2. Breadth-only A/B at larger n to settle the +0.159 (Holm-corrected 0.125 today).
3. Attack the 79.5-searches-per-cell inefficiency — B3 currently adds visits without displacing
   the search spend B1+B2 unlocked.
4. Fix the widened-graph timeout, then give c53 headroom (strong-model solve or easier
   sibling) and author 2-3 more closed-environment tasks, then run the four-way on them. That substrate has no site flakiness, bounded context, and non-fabricable
   ground truth.
5. Only then decide on Track C.

## 8. Artifacts

* `docs/handoffs/data/bstack_ab_20260824.csv` — 24 cells, per-cell scores and mechanism counters
* `agent/idea_test_results/bstack_{off,on}_rep{1,2,3}_*.json` — raw cells (+ retained `.jsonl`
  traces, the first run in this project's history to have them)
* `docs/handoffs/data/dagbase_20260824_mechanism.csv` — the 96-cell baseline this cycle started from
