# Handoff: DAG v2 benchmark policy + improvement steps (2026-08-15)

**You are picking up an unfinished, uncommitted preflight.** The DAG v2 relaunch was about to be
run. An adversarial review stopped it, because four bugs would have produced a benchmark that
looked clean and was wrong — one of them severe enough to invert the headline comparison. Those are
fixed. What is NOT settled is **benchmark policy**: what the relaunch measures, how it reports,
what counts as a fair arm, and which of DAG v2's weaknesses are worth fixing before spending money.

Your job is to answer the open questions below with evidence, and to turn the policy sketch into
something executable. **You will need to gather more information; almost nothing here should be
taken on faith — the last session's most important finding was that a widely-repeated documented
claim was simply false.**

Read `docs/handoffs/DAG_V2_PREFLIGHT_2026-08-15.md` first (the findings), then this (the policy
questions). `docs/handoffs/HANDOFF.md` is the standing project handoff.

---

## Part 1 — Background you need

### 1.1 What this project is

webRAG (product name **Euglena**) is a Graph-of-Thoughts research agent. The central thesis, which
is the yardstick for every architecture call: **boost cheap/weak models via structure + memory on
long-running agentic tasks where hallucination compounds.** A change that makes an expensive model
slightly better is off-thesis. A change that makes `gpt-4.1-nano` behave like a mid-tier model is
on-thesis.

Engine generations (established 2026-08-14, see `README.md#versioning`):

| name | when | what |
|---|---|---|
| **DAG v1** | 2026-02–03 | the original native Graph-of-Thoughts rewrite |
| **Compiled v1** | 2026-06 | expensive model authors a plan offline, cheap model executes it |
| **DAG v2** | 2026-07–present | native interleaved plan→act→observe→decide; **the current emphasis** |
| **v3** | planned | continuable/chatbot-like interaction, tool expansion, codebench |

The relaunch you are scoping is **DAG v2's closing argument**: a bigger, harder benchmark than DAG
v1's, including repeated DAG v1 tasks to show direct improvement, dropping the old sequential-mode
arm, and adding an off-the-shelf public agent system as an external baseline.

### 1.2 The machinery, concretely

**Arms** (`agent/app/idea_test_runner.py:429-550`, `_GOT_ARM_PROFILES`, selected via
`IDEA_TEST_ARM`) — these are *config profiles*, not code paths:
- `baseline` — adaptive fully OFF (the bare cheap model)
- `good_adaptive` — re-expansion + step-confidence judge + confidence-triggered re-grounding +
  corrective context + tool-failure recovery. **The proven winner.**
- `max_burn` — `good_adaptive` + deeper re-expansion + wider hop/beam + finalize reconcile chain
- `full` — **dead**, measured net-negative, dropped. `BENCHMARK_SUITE_50.md`'s rung table still
  shows it and is stale. The real ladder is `baseline → good_adaptive → max_burn`.
- ablations: `reexpand_only`, `confidence_only`, `kvote_only`, `backtrack_only`

**Execution variants** (a separate axis — `_parse_execution_variants`, dispatched in
`agent/app/testing/runner.py:91+`): `graph` (the engine, what the ladder arms run),
`sequential_react` (this repo's own ReAct loop — the premium *reference bar*, not a rung),
`naive_discretion` (the honest floor), `graph_compiled` (Compiled v1), `graph_compiled_code`
(codebench), baselines (`parametric`/`naive_rag`/`minimal`), and **`langgraph_react`** (new, added
this session — the external baseline).
- Bare `sequential` is the DEAD 0-visit legacy path. Never use it. Documented in
  `scripts/LADDER_PREREGISTRATION.md`.

**Task suite**: `agent/app/idea_tests/test_*.py`, 163 files, ~10 authoring eras
(`docs/BENCHMARK_SUITE_HISTORY.md`). The active set is **59 tasks**, defined in
`scripts/adaptive_ladder_run.py` as `TASK_SETS["suite59"]` and pinned against
`agent/tests/validator_lint_test.py::ACTIVE_SUITE_IDS` by a test so they cannot diverge. 71 tasks
are marked invalid — do not run them without re-auditing.

**Scoring** — verified this session, and **previously mis-documented**:
`agent/app/testing/validation.py:190-224`. `overall_score` = unweighted mean over the function
validators **plus the LLM judge as one additional term** (divisor N+1 when a judge exists).
`overall_passed = overall_score >= 0.75`, a bare literal at `validation.py:224`.
**`scripts/level_ladder.py`, `gate_report.py`, `recovery_curve.py` contain no 0.75 at all** — they
report a continuous mean. Saying "the pass bar is 0.75" next to a ladder table is misleading.
All three read `validation.overall_score` via `scripts/bench_common.py:76`, which **returns None
and silently drops the row** if that key is missing.

**Grounding gate**: nearly every valid task requires `observability["visit"]["count"] > 0` before
crediting its keystone, so a 0-visit hallucination scores 0. This is the suite's integrity
backbone; `scripts/validator_lint.py` is a CI gate enforcing it over the active 59.

**Cost**: `agent/app/model_costs.py`. Static `MODEL_PRICING` + a 24h-TTL OpenRouter cache
(`.model_pricing_cache.json`) that is refetched **only** when `LLM_PROVIDER == "openrouter"`.

**Driver**: `scripts/adaptive_ladder_run.py` — resume-safe, PID-locked, per-cell isolated
subprocess at `IDEA_TEST_CONCURRENCY=1`, embedded per-cell Chroma, global `--budget` + per-run
`IDEA_TEST_USD_CEILING`. Read `run_cell`/`cell_env`/`base_env` before running anything live.

### 1.3 Hard-won operational facts (do not rediscover these)

- Use `.venv/bin/python3`, and `PYTHONPATH=.:services:agent`. Bare `python3` lacks `openai` and
  every test errors at collection.
- Chroma must be up on `:8001`. Keys live in `services/keys.env` (CRLF — strip `\r`).
- `IDEA_TEST_CONCURRENCY=1` is mandatory.
- Point `IDEA_TEST_VALIDATION_MODEL` at the cell's own model; a mismatched validation model's
  preflight can 402 and abort the whole run.
- A cell can exit **rc=0 with no result JSON**. Always capture per-cell logs.
- Results are **flat files**, `{run_id}_{test_id}_{model}_{variant}_r{n}.json`, not directories.
- `telemetry_raw` is **stripped** from persisted results below `IDEA_TEST_REPORT_VERBOSITY=3`.
- Subagents cannot reliably drive backgrounded live runs here — **the coordinator must own live
  calibration centrally.** Run live cells yourself.

### 1.4 What the last session changed (all uncommitted — 37 files)

Fixes, each with a test: cross-arm grounding evidence now flows through
`observability["evidence"]`, projected from `telemetry.documents_seen` by
`runner.run_complete_test` (helpers in `agent/app/idea_test_utils.py`); `rescore_results.py`
refuses to re-score evidence-dependent tasks instead of fabricating a regression; the new
`langgraph_react` arm got forced-synthesis, `astream` token retention, tool-retry parity, infra
quarantine, and `connector_io` emission so `llm.calls` isn't 0; benchmark-axis model prices pinned
statically. New: `agent/app/langgraph_solver.py`, `agent/app/testing/execution_langgraph.py`, five
test files, two handoff docs. Offline suite: **4701 passed, 18 skipped.**

**A retracted claim you will see repeated in older docs:** "task 024's 0-visit hallucination scores
0.786 and PASSES the 0.75 bar" is **false** — 0.786 is the 7-validator mean, but 024 has a judge so
the stored score is 5.5/8 = 0.6875, which fails. Corrected in `BENCHMARK_SUITE_50.md` (F35) and
two code comments. The decision to drop 024 still stands on other grounds. **Treat this as the
cautionary tale for this whole project: a number was right, the conclusion attached to it was
wrong, and it propagated into five files and a CI test's docstring for three weeks.**

### 1.5 The live evidence you are inheriting

13 cells, **$0.045**, `openai/gpt-4.1-nano`, run ids `itsmoke_140608_*`, `cmp_141108_*`,
`cmp_141408_*` in `agent/idea_test_results/` (flat JSONs; logs in `_itsmoke_logs/`).

| task | shape | graph:base | graph:adapt | langgraph | seq_react |
|---|---|---|---|---|---|
| 122 | 4-entity fan-out | — | 1.000 @ $0.0123 | **1.000 @ $0.0009** (n=2) | — |
| 134 | 3-hop chain | 0.283 | **0.767 @ $0.0049** | 0.575 (n=2) | — |
| 140 | disambiguation | 0.800 @ $0.0042 | 0.800 @ $0.0059 | **0.800 @ $0.0006** (n=2) | — |
| 022 | doc extraction | 0.750 (1 visit) | — | — | **0.944 (12 visits)** |

**n=1–2, one model, no significance. Directional only.** Do not quote these as results.

---

## Part 2 — Open questions on benchmark policy

These are the decisions the relaunch is blocked on. Each needs evidence, not an opinion.

### Q1. What is the unit of fairness across arms? (the deepest one)
Arms currently differ in internal step budgets (`max_steps` 50–90 for graph, 25 for
sequential/langgraph), tool budgets, and token spend. Equalizing one un-equalizes the others.
- Should arms be equalized on **wall-clock**, **tool calls (search+visit)**, **$ spend**, or
  **nothing** (each arm runs its own natural configuration)?
- The current implicit answer is "nothing", which is defensible but must be *stated*, because at
  equal score LangGraph costs 7–13× less and a score-only table hides that.
- Consider reporting an **iso-cost** comparison: give every arm the same $ and see who scores
  highest. That is arguably the honest framing of the project's own thesis.

### Q2. Is DAG v2's token premium buying anything? (the most important improvement question)
Observed: DAG v2 spends 7–13× the tokens while making **fewer** tool calls (1 visit on 022 vs
`seq_react`'s 12; 1 visit on 140; 2 on 134). The thesis is "burn cheap tokens for better
reasoning" — but the one task where a linear agent clearly won (022) was won by *looking more*.
- Where do the tokens actually go? Instrument the split: expansion prompts vs evaluation vs
  finalize vs leaf actions. `telemetry.timings`/`llm_usage` and the `decisions` trace should let
  you attribute this without new plumbing.
- Is a budget binding too early? Check `grounding_max_replans`, `max_steps`, `got_beam_max`,
  `max_branching`, and the visit budget against runs that scored below 1.0.
- **Cheap decisive experiment:** one arm identical to `good_adaptive` but with a raised visit
  budget, on ~5 chain tasks, R=3. If score tracks visits, the lever is evidence volume and a lot
  of planning machinery is dead weight.

### Q3. Which shapes does structure actually win, and does the suite over-represent the others?
The single chain task (134) is the only place adaptive pulled away (+0.48 over baseline). Fan-out
(122) and disambiguation (140) were ties, with LangGraph 7–13× cheaper.
- Recompute the active-59's shape balance (`BENCHMARK_SUITE_50.md` has a stale-ish table) and ask
  whether the suite is weighted toward shapes where the mechanism does nothing.
- If chains are where DAG v2 wins, is the suite's chain population big enough to carry the claim?
- **Policy question:** should the relaunch report shape-stratified results as the *primary* table,
  with the pooled number secondary? Pooling averages a real win into a wash.

### Q4. What is the LangGraph arm actually for, and is it configured fairly?
It exists to answer "is your custom engine better than what anyone can `pip install`?" That is a
sharp question and the honest answer may be unflattering on some shapes.
- The subset `TASK_SETS["langgraph10"]` = `122,125,128,129,130,134,135,138,140,144` — **8 serial
  shapes, 2 fan-out, and it was picked before we knew chains are where DAG v2 wins.** It contains
  134/135 (chains) but the balance was not chosen deliberately. Re-pick it on purpose.
- Fairness knobs not yet examined: `IDEA_TEST_LANGGRAPH_MAX_STEPS` (25) vs the graph arms' 50–90;
  `search_k=6`/`page_chars=6000` (matched to `sequential_react`, not to the graph arm); the system
  prompt (deliberately mirrors the native one, but the tool *contract* is LangGraph's own
  function-calling schema — is that a confound or the point?).
- Is one framework enough? smolagents/CrewAI would cost little to add now that the `Solver` seam
  exists (`agent/app/solver.py`).

### Q5. Reps vs tasks — how much statistical power do we actually need?
The preregistration asserts power comes from task count, not reps, and prior analysis warned that
rep-level stats are pseudoreplicated (task-level paired p=0.016 was real; rep-level p=0.001 was
not). Proposal on the table: **R=3 instead of R=5** at 80 tasks, saving ~40%.
- Verify with the existing corpus: bootstrap historical per-task scores at R=3 vs R=5 and measure
  how often the ladder ordering flips. That is a free, data-backed answer.
- What effect size does the relaunch need to detect? Nobody has stated one. State it before
  running, per the preregistration's own write-before-you-see-results discipline.

### Q6. Model axis and the allocation matrix
The user's sketch (recorded, not finalized): ~80 tasks total; cheap models (up to Gemini Flash
tier) get the full 80; **pro-tier gets ~30 tasks and only sets the naive baseline** — it is not
supposed to run the adaptive ladder; local models get the full 80; LangGraph gets 10.
- Which 30 for pro-tier, and chosen how (stratified by shape? the hardest 30?)
- Does "naive baseline only" mean `parametric` alone, or `parametric` + `sequential_react`?
- How does this interact with items that carry their own task lists — the `diverse_ground` A/B
  (055–060) and the remediated DAG v1 repeats (012/021/022/023)? They must sit inside whichever
  subset each tier actually runs, or they will not be comparable.
- Local models are free but slow and error-prone. Is the full 80 realistic in wall-clock?

### Q7. What gets published, and what would falsify the thesis?
- Minimum honest table: score, cost, **visits**, per shape, per arm, with CIs and n.
- State in advance what result would mean DAG v2 lost. If LangGraph ties at 1/10th the cost on
  every shape but chains, is that a win, a wash, or a refutation? Decide **before** the data.
- `infra_failed` is consumed **only** by `adaptive_ab_analyze.py`; `level_ladder`, `gate_report`
  and `recovery_curve` include infra-poisoned cells at face value. Should the headline pipeline
  quarantine them? (Recommend yes; currently it does not.)

### Q8. Suite integrity items still open
- **012/021/023 have not been exercised live since the evidence refactor.** Only 022 has. 012
  requires 10 evidenced links and is the most likely to false-fail an honest agent — the page's
  `links` field is capped (`max_links_per_visit`, as low as 5 for some variants; `links_full` is
  uncapped and is what the validator now prefers). **Verify live before trusting it.**
- `test_125`'s citation slug `wiki/huajiang_canyon_bridge` may be stale vs the live article title;
  if so it hard-misses for every arm equally but depresses that task's ceiling.
- `BENCHMARK_SUITE_50.md` lists task **015** as both substance-invalid and pool-valid. Unresolved.
- Several tasks' citation validators score `cited/3` where the natural solution path only reaches
  2 slugs (e.g. 138) — a structural ceiling below 1.0. Is that intended?
- `observability["search"]["count"]` is a **document** count, not a search-call count (it
  increments per returned result). Any validator or claim reading it as "number of searches" is
  wrong — check `test_023`'s `validate_searches` and any analysis using it.
- `observability["llm"]["calls"]` counts `connector_io` events, and `ConnectorLLM` emits **two per
  logical call** — so it is ≈2× the real call count for native arms. The new LangGraph arm was
  deliberately matched to that same doubled scale for comparability. Anyone quoting "LLM calls" as
  a literal count is wrong; consider fixing the metric or renaming it.

---

## Part 3 — Improvement steps for DAG v2 (candidates, ranked by expected value)

1. **Evidence-volume lever (Q2).** Highest expected value, cheapest test. If DAG v2 under-visits,
   raising the visit/replan budget may capture most of the linear agents' advantage while keeping
   the chain-shape win.
2. **Chain depth.** `auto_parallel_siblings: true` (default) executes siblings in one step, keeping
   graphs depth-1 by construction, so a later step cannot condition on an earlier sibling's result
   — the thing a ReAct loop gets free. Re-expansion partially substitutes, which is consistent with
   134 being the only clear win. Worth testing `auto_parallel_siblings: false` on chain tasks
   despite the wall-clock cost. This is a deliberate design tradeoff; changing it is a real
   architecture decision, not a tweak.
3. **Kill dead weight.** A long list of mechanisms is recorded as inert or disproven
   (step-confidence judge AUC 0.571 — worse than a free statistic; confidence early-exit that
   cannot fire; backtrack needing ≥5-node chains the corpus never produces; candidate-coverage
   gate; narrative exemplars; plan-library retrieval). If the token premium is planning overhead,
   deleting inert planning is a direct cost win with no quality cost. Verify each before removing.
4. **Do not chase JSON reliability.** Measured 99.2% valid across 1,864 recorded decisions
   (81 telemetry files). This was my prior hypothesis for DAG v2's weakness and the data refuted
   it. Re-verify if you like — `agent/idea_test_results/*_json_telemetry.jsonl`, enabled by
   `IDEA_TEST_JSON_TELEMETRY=1` — but do not spend engineering effort on it without new evidence.

---

## Part 4 — Working agreements

- **Verify before asserting.** Two of my four session hypotheses were wrong (JSON fragility; the
  0.786 pass claim). Both were plausible and both would have wasted real money.
- **Live spend needs explicit authorization and a bound.** `IDEA_TEST_USD_CEILING` per cell plus a
  global `--budget`. The 13-cell smoke cost $0.045; there is no excuse for an unbounded run.
- **Cheap models are the point, not a compromise** — `gpt-4.1-nano` is both the thesis's subject
  and the best fault-tolerance test, since it errors more.
- **Nothing is committed.** 37 files. Commit style for this repo: single lowercase line, no
  punctuation, no body, no trailer. Multiple sessions edit this repo concurrently — check
  `git log`/`git status` fresh before assuming anything about branch state.
- Prefer correcting a document over adding a new one; this project's biggest failure mode so far
  has been stale docs propagating a wrong number.
