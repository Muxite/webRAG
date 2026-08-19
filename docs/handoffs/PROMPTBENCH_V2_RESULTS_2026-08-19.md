# Promptbench v2 results, 2026-08-19

Pre-registration: `PROMPTBENCH_V2_PREREGISTRATION_2026-08-19.md`. Design:
`docs/superpowers/specs/2026-08-19-promptbench-expansion-design.md`. v1:
`PROMPTBENCH_RESULTS_2026-08-19.md`.

**8,965 graded cells, 5 models x 1,793, zero transport errors, zero duplicate cell
keys.** Local tier 5,379 cells (~2.5 h GPU, $0); API tier 3,586 cells, **$0.2487**
measured as an OpenRouter key-usage delta against a $3 ceiling.

Every number here is an effect **across this roster**. Models are fixed effects by
design; nothing below estimates an effect across models in general.

---

## 1. Headline

**Reason-before-answer beats the engine's convention, and the convention's real
cost is not tokens but silent degeneracy.**

Pooled `A2 - A1` on `verify` is **+0.142, CI [+0.053, +0.232], permutation
p = 0.0119, positive on 5 of 5 models**. The interval excludes zero, so §7's first
branch fires.

The larger finding is one the primary estimator structurally cannot report. On
`goal_achieved` — the family mirroring `MergeLeafAction`'s boolean — the engine's
shipped ordering is **degenerate on 5 of 5 models**: every model answers ACHIEVED
100% of the time, scoring ~0.500 on a balanced set while judging nothing. Reason-first
breaks the rubber stamp on three of them.

## 2. H1 (primary): answer position

`A2 - A1`, pooled per-item, cluster-bootstrapped over task modules, equal weight per
model. `*` marks an interval excluding zero.

| family | pairs | clus | mdl | delta | CI95 | halfw | perm p | dir |
|---|---|---|---|---|---|---|---|---|
| `verify` | 190 | 19 | 5 | **+0.142** | [+0.053, +0.232] | 0.089 | **0.0119** | 5/0 * |
| `followup` | 112 | 28 | 2 | **+0.196** | [+0.080, +0.312] | 0.116 | **0.0064** | 2/0 * |
| `keystone_claim` | 120 | 15 | 4 | +0.000 | [-0.042, +0.033] | 0.038 | 1.0000 | 0/0 |
| `select` | 76 | 19 | 4 | +0.000 | [-0.105, +0.105] | 0.105 | 1.0000 | 1/1 |
| `goal_achieved` | 0 | 0 | 0 | NOT MEASURED | — | — | — | — |

Other arms on `verify`: `A0` +0.053 [-0.021, +0.126]; `A3` +0.016 [-0.095, +0.116];
`A4` **+0.084 [+0.016, +0.153]** *, p = 0.0424; `SHIPPED` -0.039 [-0.112, +0.039].

Two things the table is not allowed to be read as saying:

- **The effect does not replicate everywhere.** `keystone_claim` returns exactly
  +0.000 at a half-width of 0.038 — a tight null, not a wide one. `qwen2.5:7b`,
  `gpt-4.1-nano` and `gemini-2.5-flash-lite` all sit at or near 1.000 on that family
  in every arm, so there is no headroom for an ordering effect to appear in. This is
  a ceiling, not a refutation.
- **§7's divergence clause is partially triggered.** `verify` and `keystone_claim` do
  not disagree in *direction* (0.000 is not negative), so the clause does not fire on
  its literal terms. But two families move and two do not, so the next cycle should
  still be scoped per call site rather than as one global reordering.

`select` carries the run's one significant *negative*: `A4` (<=40 words, then answer)
is **-0.105, CI [-0.197, -0.026], 0/4 models**. Word-bounded reasoning hurts
selection. `A4` helps on `verify` (+0.084 *) and hurts on `select`, which is direct
evidence that these recommendations do not generalise across decision types.

## 3. The unmeasurable family, which is the result

`goal_achieved` has **no measurable contrast on any of the 5 models**, because the
`A1` baseline is disqualified on all of them. Marginal accuracies:

| model | A1 (shipped order) | A2 (reason first) |
|---|---|---|
| `qwen2.5:0.5b` | 0.500 — **ACHIEVED on 100%** | 0.482 |
| `llama3.2:3b` | 0.500 — **ACHIEVED on 100%**, 14.3% parse fail | **0.607** |
| `qwen2.5:7b` | 0.482 — **ACHIEVED on 100%**, 44.6% parse fail | **0.929** |
| `openai/gpt-4.1-nano` | 0.500 — **ACHIEVED on 100%**, 37.5% parse fail | **0.911** |
| `google/gemini-2.5-flash-lite` | 0.393 — **ACHIEVED on 100%**, 60.7% parse fail | 0.393 (still degenerate) |

A paired delta against a constant is meaningless, which is exactly why the
pre-registered rule excludes the cell — so this **cannot** be reported as a
pre-registered contrast, and is not. It is a marginal observation, and it is the
strongest thing in the run.

It also supplies a mechanism for an already-recorded number.
`CONFIDENCE_JUDGE_MISCALIBRATION.md` measures this step at **AUC 0.288**
(anti-predictive). The step is not judging badly; it is not judging. Under the
shipped ordering the model emits the boolean before it has written anything to
condition on, and "achieved" is the prior.

Caveat: adding three models did **not** restore measurability, contrary to what was
expected when the run was planned. The degeneracy is a property of the prompt shape,
not of model capability — `gpt-4.1-nano` and `qwen2.5:7b` rubber-stamp it exactly as
`qwen2.5:0.5b` does.

## 4. H2: the token saving

Gate: `A0 - A1` lower bound **above -0.02** licenses dropping the unread `reason`
field. On `verify` the interval is **[-0.021, +0.126]**.

**It fails by 0.001, so the field is not dropped this cycle.** Recording that
honestly rather than rounding it away: the point estimate is +0.053 and every family
is directionally non-negative (`keystone_claim` +0.027 [-0.013, +0.067] would pass on
its own), but the pre-registered threshold was written before the numbers and is not
being renegotiated after seeing them. Per §7's "otherwise" clause, the cost of the
reason is recorded instead.

Mean completion tokens, `verify`:

| model | A0 | A1 | A2 | A3 | A4 | SHIPPED | A1/A0 |
|---|---|---|---|---|---|---|---|
| `qwen2.5:0.5b` | 4.2 | 170.6 | 95.7 | 312.2 | 100.3 | 111.6 | **40.5x** |
| `llama3.2:3b` | 4.2 | 47.6 | 33.9 | 174.6 | 42.5 | 267.3 | 11.2x |
| `qwen2.5:7b` | 4.7 | 49.6 | 54.7 | 272.1 | 33.9 | 114.7 | 10.6x |
| `openai/gpt-4.1-nano` | 4.2 | 42.4 | 64.4 | 296.6 | 53.4 | 124.5 | 10.0x |
| `google/gemini-2.5-flash-lite` | 3.5 | 32.9 | 37.9 | 303.1 | 40.6 | 123.4 | 9.4x |

`A1` costs 9.4-40.5x the completion tokens of `A0` for an effect whose interval
straddles zero. `A3` (unbounded step-by-step) remains the worst trade in the suite:
37-72x `A0`'s tokens for +0.016 [-0.095, +0.116] on `verify`.

## 5. H3: calibration

Gate: `C_expected` beats `C_A1` on Brier **and** clears the LLM-free structural
baseline of AUC 0.655.

`C_expected` improves Brier on 3 of 5 models (0.5b 0.785 -> 0.571, 3b 0.428 -> 0.345,
nano 0.221 -> 0.174) and worsens it on 2 (7b 0.244 -> 0.344, gemini 0.158 -> 0.206).
Its AUCs are 0.475, 0.448, 0.646, 0.627, 0.621 — **none clears 0.655**.

**P2 joins `CONFIDENCE_JUDGE_MISCALIBRATION.md`'s explicitly-not-worth-pursuing
list.** That is a real result, obtained cheaply, and it is reported as one.

Across the entire 25-cell calibration table, exactly **one** arm clears the free
baseline: `qwen2.5:7b` / `C_A2` at AUC 0.662. Every other cell is flagged. A
structural baseline that costs no LLM call beats 24 of 25 LLM-produced confidence
signals on this task.

**The anti-calibration finding fires.** §7: a negative slope on `C_A1` is worth its
own cycle regardless of the other gates. `C_A1` slope is negative on **3 of 5**
models — `qwen2.5:0.5b` **-4.952**, `llama3.2:3b` -0.079, `gemini-2.5-flash-lite`
-1.622. On those models, higher stated confidence predicts *lower* accuracy under the
engine's literal `{confidence, reason}` ordering.

## 6. Gate outcome (§7 lookup)

| gate | result | action |
|---|---|---|
| `A2 - A1` pooled | +0.142, CI [+0.053, +0.232], p = 0.0119 | **CI excludes 0, positive** -> implement reason-first at the tabulated call sites, **flagged, default OFF**. Next cycle: end-to-end transfer A/B on mixed-shape tasks. |
| `A0 - A1` pooled | +0.053, CI [-0.021, +0.126] | Lower bound below -0.02 by 0.001 -> **do not drop the `reason` field**; record its cost (9.4-40.5x). |
| calibration | `C_expected` AUC 0.448-0.646, never clears 0.655 | **P2 to the not-worth-pursuing list.** |
| `C_A1` slope | negative on 3/5 | **Own cycle**, per §7. |
| per-family divergence | 2 families move, 2 are null; no direction reversal | Scope per call site, not one global reordering. |

**Hard non-goal, honoured: no shipped prompt was changed by this cycle.**

H4 (size interaction) stays dropped. The roster was amended to 2 models before any
row was collected and recovered to 5 opportunistically; at 5 models with an
unbalanced tier split it remains descriptive-only, and no ordering by size is
claimed. `qwen2.5:1.5b` is now **permanently deferred** — the GPU was withdrawn after
this run.

## 7. Defects found

- **`analyze.py` crashed on its own data.** `print_primary` formatted `mean_delta`
  with `:+8.3f` while `pooled_report` correctly returns `None` for a contrast with
  zero surviving pairs. It died *after* the summary table printed, so the run looked
  half-finished. Six contrasts hit it, five of them the whole `goal_achieved` family.
  The pre-check tested the raw rows *before* exclusions, so it could not see the
  case. Latent at 5 models in v1; reachable only once the amended roster shrank.
  Fixed, with a regression test that also pins the converse (a guard rendering every
  delta as a dash would equally never crash).
- **`report.apply_exclusions` had no rule for a cell with zero usable rows.**
  `n_clusters` counts error rows, so a wholly transport-errored cell could clear the
  5-cluster rule and reach the printer with `None` metrics. It could not fire on
  local Ollama; it is squarely on the API path. Fixed.
- **The runner is serial, and that was a 17x wall-clock tax on the API tier.** One
  process measured ~11 rows/min against OpenRouter. Partitioning the matrix by family
  across parallel processes — family-disjoint, so the `model|family|variant|item_id|rep`
  cell key cannot collide — reached ~185 rows/min with zero errors. No code change.

## 8. Reproduce

Raw rows stay gitignored (`.gitignore:33`), as in v1; the derived
`promptbench_v2_analysis.json` is the committed record.

```
# local tier (one model per invocation; append-and-resume, so re-running is a no-op)
PYTHONPATH=.:services:agent ./.venv/bin/python -m agent.app.promptbench.runner \
  --models llama3.2:3b \
  --families verify select calibration keystone_claim goal_achieved followup \
  --variants A0 A1 A2 A3 A4 SHIPPED F_json G_nostatement G_noevidence \
             C_A1 C_A2 C_verbal C_expected \
  --out agent/idea_test_results/promptbench_runs_v2.jsonl

# API tier, partitioned by family across parallel processes (see scratchpad launch_api.sh)
#   --base-url https://openrouter.ai/api/v1 --api-key "$OPENROUTER_API_KEY"

# pooled estimate over all 9 files
PYTHONPATH=.:services:agent ./.venv/bin/python -m agent.app.promptbench.analyze \
  --runs agent/idea_test_results/promptbench_runs_v2.jsonl \
         agent/idea_test_results/promptbench_api_runs_v2.jsonl \
         agent/idea_test_results/promptbench_api_v2_nano_{b,c,d}.jsonl \
         agent/idea_test_results/promptbench_api_v2_gemini.jsonl \
         agent/idea_test_results/promptbench_api_v2_gem_{b,c,d}.jsonl \
  --json-out agent/idea_test_results/promptbench_v2_analysis.json
```

`--runs` now takes several paths and pools them into one estimate. It hard-fails on a
duplicate cell key across files: `pooled_delta_records` is last-write-wins, so a
duplicate would silently pick one of two recorded answers for the same prompt.

## 9. Next

1. **Reason-first behind a flag, default OFF**, at `idea_policies/actions.py:2202`
   (verify) and `:2101` (merge `goal_achieved`). The merge site has the stronger case:
   its shipped ordering is degenerate on every model tested.
2. **End-to-end transfer A/B on mixed-shape tasks.** Everything above is
   single-call accuracy on frozen items; none of it demonstrates transfer to the
   agent loop, and it must not be quoted as if it does.
3. **The `C_A1` anti-calibration cycle**, per §7.
4. Deferred: `qwen2.5:1.5b` (GPU withdrawn), reps > 1 (T = 0 nondeterminism still
   unquantified), and `goal_achieved` measurability — which now looks like it needs a
   non-degenerate baseline arm rather than more models.
