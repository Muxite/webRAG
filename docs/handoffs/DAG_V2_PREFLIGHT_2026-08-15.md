# DAG v2 relaunch — adversarial preflight (2026-08-15)

Adversarial review of the 2026-08-15 relaunch-scoping work (items 1–4), plus a $0.045 live
integration smoke on `openai/gpt-4.1-nano`. Four benchmark-invalidating bugs found and fixed, one
documented claim retracted, and the first head-to-head DAG v2 vs LangGraph numbers.

**Status: relaunch is NOT yet cleared to run.** See "Before the relaunch" at the bottom.

**Continuing this work?** `BENCHMARK_POLICY_HANDOFF_2026-08-15.md` (same directory) carries the
background, the open benchmark-policy questions (Q1–Q8) and the ranked improvement candidates.
This document is the evidence; that one is the agenda.

## 1. Bugs found and fixed

### 1a. Cross-arm evidence: would have inverted the headline result (CRITICAL)

The DAG v1 repeat tasks (012/021/022/023) were rewritten to prove grounding by matching a claimed
fact against the page it came from, sourcing that page from `result["graph"]`. Only the `graph`
and `naive_discretion` variants emit a populated graph — `sequential_react`, `graph_compiled`,
`langgraph_react` and every baseline return `_empty_graph()` (`testing/execution.py:32`). Every
grounded check therefore scored a structural 0 on those arms.

Measured on a real run (`itsmoke_140608_sr022`, task 022, `sequential_react`, gpt-4.1-nano):

| | score |
|---|---|
| graph-sourced evidence (the bug) | **0.417** — FAIL |
| telemetry-sourced evidence (fixed) | **0.944** — PASS |

The same run's `graph` arm scored 0.750. So the bug would have produced "the graph engine beats
ReAct 0.75 vs 0.42 on the DAG v1 tasks" when the truth is the reverse — ReAct scored higher
(0.944) because it visited 12 pages to the graph arm's 1. A headline-inverting artifact.

**Fix:** evidence is projected from `telemetry.documents_seen` — recorded identically by every arm
— into `observability["evidence"]` by `runner.run_complete_test`, injected into a *copy* so the
persisted payload is unchanged. Shared helpers live in `idea_test_utils.py`
(`visited_evidence`/`evidence_text`/`visited_domains`/`visited_link_urls`). Regression-guarded by
`agent/tests/dag_v1_repeat_cross_arm_parity_test.py`, which asserts an identical run scores the
same whether its evidence arrives graph-shaped or telemetry-shaped.

### 1b. Post-hoc re-scoring silently fabricates a regression

`idea_test_runner` strips `telemetry_raw` from the persisted result at the default report
verbosity, so the evidence is gone by the time `scripts/rescore_results.py` runs. Re-scoring task
022's real result offline reproduces exactly the 0.944 → 0.417 collapse above, as a *fictitious*
regression. `rescore_results.py` now detects evidence-dependent tasks and **refuses loudly**
instead (verified: it skips with an explanation). Run with `IDEA_TEST_REPORT_VERBOSITY=3` if
post-hoc re-scoring must stay possible.

### 1c. LangGraph arm fairness and accounting (would have understated the external baseline)

Four defects, each of which made the off-the-shelf arm look worse or cheaper than reality:

- **No forced final synthesis.** `execution_sequential` gives its agent a last-turn synthesis from
  gathered evidence when steps run out; LangGraph raised `GraphRecursionError` and returned
  nothing — a hard 0 on runs that had done the work. Now mirrors the native path.
- **Token accounting lost on crash.** `ainvoke` discards state when it raises, so a failed run
  reported **$0** for money genuinely spent. Now streams via `astream` and keeps partial messages.
- **No tool-retry parity.** The native arms retry transient search/visit failures (F16); the
  LangGraph tools did not, so one flaky search became a permanent error for this arm only. Now
  shares `_call_tool_with_retry` and the same `connector_retry_*` policy.
- **`llm.calls` read 0.** `summarize_observability` derives LLM stats from `connector_io` telemetry
  events, which only `ConnectorLLM` emits — so an arm driving its own calls reported zero LLM work
  with non-zero tokens. Now emits matching in/out events (verified live: 6 calls, 5,217 tokens).

Also: fails fast on a missing API key rather than letting `ChatOpenAI` fall back to a stray
`OPENAI_API_KEY` and 401 mid-run; and F17 infra-failure quarantine is recorded for provider-side
failures but deliberately NOT for a step-budget stop (that is a genuine agent outcome).

### 1d. Cost silently degrades to "unpriced"

`MODEL_PRICING` held only three entries; every other slug was priceable only via a 24h-TTL
OpenRouter cache that is refetched **only** when `LLM_PROVIDER == "openrouter"`. On an expired
cache the failure is quiet and inconsistent: `recovery_curve.py` drops the row, `level_ladder.py`
mislabels the model **"local"** (i.e. free), and `idea_test_runner`'s run total counts it as $0 —
so the premium reference bar could appear to cost nothing. Every benchmark-axis model is now
pinned in the static table (values verified against the live cache), confirmed working with the
cache disabled.

## 2. A documented claim was wrong — retracted

`overall_score` is the unweighted mean over the function validators **plus the LLM judge as one
more term** (`testing/validation.py:190-224`), and `overall_passed` is `>= 0.75`.

The widely-propagated claim that task 024's 0-visit hallucination "scores 0.786 and PASSES the
0.75 bar" conflates the 7-validator mean with the 8-term score actually stored. 024 *does* define
a judge, so the stored score is `5.5/8 = 0.6875` — it **fails** unless the judge itself awards
≥ 0.50, and that judge is explicitly told "Visit actions executed: 0". Corrected in
`BENCHMARK_SUITE_50.md` (F35), `scripts/adaptive_ladder_run.py`, `validator_lint_test.py`.
**Dropping 024 still stands on its own merits** — it is genuinely un-gated. Only the "provably
passes" claim is retracted.

The equivalent figures for 012/021/022/023 *are* valid, because all four return `None` from
`get_llm_validation_function()`, so their divisor really is the function count.

## 3. Live integration smoke — 13 cells, $0.045, `openai/gpt-4.1-nano`

Deliberately the weakest/cheapest model, to exercise fault tolerance. All 13 cells exited rc=0,
none tripped the $0.25/cell ceiling, `infra_failed` false throughout, every cell priced
(`cost.estimated == False`).

| task | shape | arm | n | score | visits | tokens | usd | sec |
|---|---|---|---|---|---|---|---|---|
| 122 | 4-entity fan-out | graph:adapt | 1 | 1.000 | 4 | 115,264 | 0.01228 | 42.7 |
| 122 | | **langgraph** | 2 | **1.000** | 4 | 7,752 | **0.00091** | 7.5 |
| 134 | 3-hop chain | graph:adapt | 1 | **0.767** | 2 | 45,471 | 0.00488 | 56.3 |
| 134 | | graph:base | 1 | 0.283 | 3 | 50,528 | 0.00530 | 37.7 |
| 134 | | langgraph | 2 | 0.575 | 2.5 | 15,328 | 0.00161 | 10.1 |
| 140 | disambiguation | graph:adapt | 1 | 0.800 | 1 | 56,342 | 0.00593 | 31.5 |
| 140 | | graph:base | 1 | 0.800 | 1 | 40,779 | 0.00424 | 14.1 |
| 140 | | **langgraph** | 2 | 0.800 | 1 | 5,215 | **0.00058** | 5.9 |
| 022 | doc extraction | graph:base | 1 | 0.750 | 1 | 14,529 | 0.00173 | 16.9 |
| 022 | | **seq_react** | 1 | **0.944** | 12 | 36,680 | 0.00405 | 25.1 |

**These are n=1–2, one model, no significance testing. Directional only — not results.**

## 4. Where DAG v2 is weaker than LangGraph, and what to do

### The one place structure clearly wins
On the 3-hop chain (134), `good_adaptive` scored 0.767 vs LangGraph's 0.575 and its own baseline's
0.283. The adaptive re-expansion lever is doing real work — **+0.48 over baseline on the same
model** — recovering from a bad first hop. This is the shape to build the DAG v2 story on.

### Weakness 1 — the token burn does not buy evidence (the big one)
DAG v2 consistently spends 7–13× the tokens while making **fewer tool calls** than a linear agent:
1 visit on 022 (vs seq_react's 12), 1 visit on 140, 2 on 134. The premium is going into expansion
and planning prompts, not into gathering more pages. The project's thesis is "burn cheap tokens for
better reasoning" — on this evidence the tokens are being burned on graph bookkeeping, and the
measurable win (022) went to the arm that simply *looked more*.

*What to do:* treat visits-per-run as a first-class metric next to score and cost, and check
whether the grounding gate's replan budget (`grounding_max_replans`) is binding before the agent
has enough evidence. The cheapest experiment is an arm identical to `good_adaptive` but with a
raised visit budget — if score tracks visits, the lever to pull is evidence volume, not more
planning tokens.

### Weakness 2 — flat, wide graphs can't chain
`auto_parallel_siblings: true` (default) executes siblings in one step, keeping graphs depth-1 by
construction, so a later step cannot condition on an earlier sibling's result. That is exactly the
capability a ReAct loop has for free, and it is consistent with 134 being the only task where
adaptive pulled away (re-expansion partially substitutes for real chaining). Fan-out shapes, where
this design should shine, are precisely where LangGraph tied at 1/13th the cost.

*What to do:* this is a design tradeoff, not a bug — but the relaunch should report chain-shaped
and fan-out-shaped tasks **separately**. Pooling them averages a genuine win into a wash.

### Weakness 3 — no cost-normalized comparison
At equal score, LangGraph costs 7–13× less. A score-only table makes DAG v2 look neutral-to-good;
a $/solved table would make it look bad. Both are true. The honest framing (and the project's own
stated framing) is quality-at-a-cost-tier, so the relaunch must publish score, cost, **and**
visits together, per shape.

### Not a weakness: JSON fragility
I expected prompted-JSON decisions to be DAG v2's structural disadvantage against LangGraph's
native tool-calling. **The recorded data refutes it** — across 1,864 JSON decisions in 81
telemetry files, 1,849 were valid (99.2%): `qwen2.5:1.5b` 1.9% failures, `gpt-4.1-nano` 0.5%,
`claude-sonnet-5` 2.6%, several local models 0.0%. Do not spend effort here.

## 5. Making the relaunch cheap without losing the signal

Empirical per-cell costs (this smoke + 490 historical priced results) support a much cheaper
design than the ~$60 the 50-task plan assumed:

- **Reps: R=3, not R=5.** The preregistration already states power comes from task count, not
  reps. At 80 tasks, dropping R=5→3 cuts the cheap-model ladder ~40% for a negligible power loss.
- **Drop the `graph:base` arm on tasks where it is uninformative.** On 140 baseline and adaptive
  tied exactly (0.800/0.800) while adaptive cost 40% more. The ladder's value is concentrated on
  the shapes where the mechanism fires (chain/re-expansion).
- **Pro-tier really is naive-baseline-only.** Per the allocation sketch: ~30 tasks, `parametric`
  and/or `sequential_react` only, R=3. Never run the pro tier through the adaptive ladder — at
  $2/$12 per 1M it dominates the bill and answers a question nobody asked.
- **LangGraph arm: 10 tasks × R=3 ≈ $0.03.** Effectively free; the only reason to keep it at 10 is
  authoring/analysis effort, not money.
- **Estimated total** at 80 tasks, R=3, cheap+local full / pro 30-task naive-only:
  roughly **$20–30**, versus ~$60 for the older 50-task R=5 shape.

**Task-shape balance matters more than task count.** The 10-task LangGraph subset is currently
8 serial shapes + 2 fan-out (122/125) and **zero** of the chain shape where DAG v2 actually won.
Task 134/135-style chains must be in that subset or the comparison will understate DAG v2.

## Before the relaunch

1. **Re-run the 4 remediated tasks' probes against real runs.** The 022 smoke passed, but 012, 021,
   023 have not been exercised live since the evidence refactor. 012 in particular requires 10
   evidenced links and is the most likely to false-fail an honest agent.
2. **Decide the visits-vs-tokens question** (Weakness 1) — it changes what the relaunch is
   measuring. Cheap to test: one raised-visit-budget arm on ~5 chain tasks.
3. **Fix the langgraph10 subset's shape balance** — add 134/135, consider dropping one fan-out.
4. **Check `test_125`'s citation slug** (`wiki/huajiang_canyon_bridge`) against the live article
   title; if stale it hard-misses for every arm.
5. **Set `IDEA_TEST_REPORT_VERBOSITY=3`** for the relaunch if results must stay re-scorable.
6. Note that `infra_failed` is consumed **only** by `adaptive_ab_analyze.py` — `level_ladder`,
   `gate_report` and `recovery_curve` include infra-poisoned cells at face value.
