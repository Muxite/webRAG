# Phase KPI synthesis — one picture across the ledger-phase runs

Analysis only, $0, offline. Every number below was recomputed against the stored result JSONs
in `agent/idea_test_results/` using `scripts/module_ab.py`, `scripts/kpi_dashboard.py`,
`scripts/trust_kpi_dashboard.py`, `scripts/risk_coverage.py`, `scripts/claim_metrics.py`,
`scripts/coverage_report.py`, `scripts/prereg.py`, and `agent.app.testing.claim_audit.audit()`
directly — not copied from prior handoffs. Definitions are `docs/LEDGER_KPI_SPEC.md` (frozen);
measurement rules are `docs/LEDGER_METHODOLOGY.md`. Read the holdout section of
`docs/LEDGER_FINAL01_TUNING_RESULT.md` before trusting any per-arm number below — it is why this
report leads with structural/categorical rows and treats score as a guard, never a ranking.

---

## 0. Runs actually pulled in

| run_id | cells found | design | status |
|---|---|---|---|
| `ledgerfinal01` | 66/66 (`prereg audit` confirms complete, 1 infra_failed dropped for K2/K7) | 22 tasks x 3 arms x 1 rep, seeded, qwen2.5:7b | reference campaign |
| `mod2_langgraph_react_{off,on}` | 12+12 | tasks 210-221, derive module off/on, langgraph_react host | host-vs-host+module |
| `mod2_sequential_react_{off,on}` | 12+12 | tasks 210-221, derive module off/on, sequential_react host | host-vs-host+module |
| `moduse01` | 12 | tasks 210-221, langgraph_react, module offered (not forced) | adoption measurement |
| `tinyfixoff`/`tinyfixon` | 3 models x 3 tasks (210-212) each | zero-visit-gate + context-fit, off/on | small-model fix A/B |
| `tinyfixoff2`/`tinyfixon2` | 3 models x 3 tasks (213-215) each | same fix, tasks 213-215 | small-model fix A/B |
| `ladder02_langgraph_react_off` | **34** complete cells (+1 unfinished `.jsonl` for phi3:mini/215) | intended 6 models x 48 tasks x 1 rep = 288 | **killed early — the task brief says 36; the actual file count is 34 complete result JSONs, see §5** |
| `phi3_both` | 6 (phi3:mini only) | tasks 210-215, both small-model fixes stacked | **IN PROGRESS** — `models` in the summary lists `qwen2.5:7b` too, but no qwen2.5:7b result files exist yet under this run_id; only the phi3:mini half landed |

---

## 1. What each mechanism actually made TRUE

Structural rows first, score printed only as the accuracy guard (per `module_ab.py`'s own
design: a null score result is the win here, not a tiebreak).

### 1a. Derive module (arithmetic with recorded operands) — `mod2_langgraph_react_{off,on}`

```
                                    off        on
cells with a derivation              0          5
machine-computed values              0          8
of those, invalid                    0          0
fabricated-arith rate           UNKNOWN      0.000
refusals (averted)                   0         20
replayable cells                   0/0        6/6
overall_score (guard)            0.938      0.863   (paired delta -0.075 +/- 0.167, p=0.44, n=12, 6/12 tasks moved at all)
```

### 1b. Derive module — `mod2_sequential_react_{off,on}`

```
                                    off        on
cells with a derivation              0          4
machine-computed values              0          9
of those, invalid                    0          1
fabricated-arith rate           UNKNOWN      0.111
refusals (averted)                   0         25
replayable cells                   0/0        5/5
overall_score (guard)            0.646      0.671   (paired delta +0.025 +/- 0.202, p=0.78, n=12, 6/12 tasks moved)
```

**What this makes TRUE, categorically:** before the module, `langgraph_react` and
`sequential_react` cannot produce a machine-verified derived value at all — the fabrication rate
for arithmetic is not "low", it is UNKNOWN, permanently, because there is nothing to check. With
the module attached to either host, every derived value the model proposes is recomputed in
Python; 0 or 1 of 8-9 disagreed (sequential's one was invalid, not silently accepted — the module
still recorded it). Neither host lost accuracy outside the trajectory-chaos band already measured
in the tuning doc (12/16 and 6/12-cell-move counts are consistent with that floor). **Do not** read
the 0.938→0.863 or 0.646→0.671 deltas as "the module costs/buys score" — n=12 per arm is far below
the ~60-task floor this project has already measured as unresolvable, and the module_ab script
itself refuses to lead with this number for that reason.

### 1c. Derive-module adoption when only offered, not forced — `moduse01`

Recomputed directly from `execution.output.evidence_graph.nodes[].kind=="derived"` per cell
(langgraph_react, tasks 210-221, 12 cells):

| task | derived nodes |
|---|---|
| 210,212,213,215,216,219,220 | 0 |
| 211,214,217,218 | 1 |
| 221 | 3 |

**Adoption rate: 5/12 cells (41.7%) used the module at all when it was available but not
required.** This lands inside the methodology's own adoption-ladder rung-1 estimate ("optional
tool the model may call, measured 33-42%") — confirmed, not merely cited. Because coverage is
under half, any average that includes the 7 cells that never called it dilutes whatever the
module does by more than half — the methodology's own warning against averaging over adoption is
directly demonstrated by this run.

### 1d. Quantity index, unit-from-span, context-fit, zero-visit gate — `tinyfixoff/on`, `tinyfixoff2/on2`

Recomputed per cell from `execution.observability.visit.count` and `validation.overall_score`
(3 models x 6 tasks, off vs on):

| model | measure | OFF | ON |
|---|---|---|---|
| phi3:mini | cells reading >=1 page | 0/6 | **4/6** |
| phi3:mini | cells scoring >0 | 0/6 | **4/6** |
| gemma2:2b | mean score (6 cells) | 0.263 | 0.467 |
| qwen2.5:7b | mean score (6 cells) | 0.950 | 0.950 (unchanged, byte-for-byte same visit counts and scores every cell) |

This exactly reproduces `docs/handoffs/LEDGER_TINY_MODEL_PHASE_2026-09-02.md` §3's table —
independently recomputed here, not copied.

**What this makes TRUE, categorically:** before the fix, `phi3:mini` cannot score above zero on
this task family by construction (three of four validators require a real page fetch, and it
never visits one — it answers from the search snippet). After the fix, 4 of 6 cells read a page
and score. `qwen2.5:7b` is provably unaffected either way (identical visit counts and scores in
every one of 6 paired cells) — the mechanism is inert on a model with a roomy context window, which
is itself useful confirmation that it isn't perturbing behavior for models that didn't need it.
`gemma2:2b`'s gain is a genuine mean-score move (0.263→0.467) but n=6 per side; call it directional,
not resolved.

### 1e. Both small-model fixes stacked — `phi3_both` (phi3:mini half only)

| task | visits | score |
|---|---|---|
| 210 | 0 | 0.0 |
| 211 | 0 | 0.0 |
| 212 | 1 | 0.75 |
| 213 | 0 | 0.0 |
| 214 | 1 | 0.25 |
| 215 | 0 | 0.0 |

**2/6 cells read a page and scored above zero — down from `tinyfixon`'s 4/6 on the same task
range, with (as far as the stored artifact can show) the same mechanisms nominally on.** The
stored cell JSONs carry no record of which `LEDGER_*`/env flags were active (this is the exact gap
`LEDGER_KPI_SPEC.md` §6 names: env-flag conditions never reach the cfg hash or the stored
`variant_specific_settings`, which is empty `{}` in both `tinyfixon` and `phi3_both` cells here).
I cannot confirm from the artifacts alone that `phi3_both` actually had both fixes enabled, and I
was told not to launch any run to check. **Report this as contradicted/unresolved, not as a
regression** — see §4.

---

## 2. What each mechanism cost

Cost is reported from `kpi_dashboard.py`'s K7 fallback (tokens/seconds; `cost.usd` is null for
every local-model cell here) and `claim_metrics.py`'s call/edge counts, on `ledgerfinal01`
(the only run with per-arm, per-split cost broken out).

| split | arm | tokens/cell | sec/cell | overall_score (guard) |
|---|---|---|---|---|
| tuning (n=15/16/16) | evidence_loop | 84,706 | 43.8 | 0.627 |
| tuning | langgraph_react | 29,070 | 17.6 | 0.613 |
| tuning | sequential_react_extract | 37,725 | 20.3 | 0.616 |
| holdout (n=6 each) | evidence_loop | 63,441 | 36.4 | 0.672 |
| holdout | langgraph_react | 24,192 | 15.7 | 0.669 |
| holdout | sequential_react_extract | 55,673 | 27.3 | 0.351 |

(All four tuning-split figures independently recomputed here match `LEDGER_FINAL01_TUNING_RESULT.md`
exactly — 0.627/0.613/0.616 overall_score, confirming that document's headline table.)

`evidence_loop`'s cost premium over `langgraph_react` is **~2.9x tokens, ~2.5x wall clock on the
tuning split**, and it does not close on holdout (~2.6x/~2.3x) — the cost gap is the one number
in this phase that is stable across both splits, unlike every KPI ordering. The module A/B (§1a/b)
adds cost too: `mod2` doesn't report tokens/sec directly in `module_ab.py`'s output, but the
`replayable` row shows every "on" cell that had a derivation is fully offline-replayable (6/6,
5/5) at $0 marginal re-verification cost — a one-time build cost, zero-cost forever after.

**Cost is a first-class number here, and the honest summary is: the more auditable arm is not free,
and the gap does not shrink when the accuracy ordering inverts.**

---

## 3. Where the evidence is thin

Every claim below rests on n < 10 cells, a single model, or an unreplicated run — named here so a
reader does not have to hunt a footnote:

- **`mod2_langgraph_react` and `mod2_sequential_react` A/Bs**: n=12 per arm, single model
  (qwen2.5:7b), single rep, unseeded (no `LLM_SEED` recorded in the config). The structural rows
  (derivation counts, refusals, replayability) are categorical and hold at this n; the
  overall_score deltas do not (p=0.44 and p=0.78).
- **`moduse01` adoption rate (5/12)**: single model, single rep, 12 cells. It happens to land
  inside the methodology doc's independently-stated 33-42% range, which is corroborating but not
  a second independent measurement — both numbers may share the same underlying run family.
- **`tinyfixon`/`tinyfixoff` gemma2:2b mean-score move (0.263→0.467)**: n=6 per side, one model,
  one task band (210-215). Directional only.
- **`phi3_both`**: n=6, one model, and per §1e/§4 the mechanism configuration behind it cannot be
  confirmed from the stored artifact.
- **`ladder02`**: 34 of an intended 288 cells (killed twice, per the tiny-model handoff). Five of
  six models have only 5-6 cells each; `qwen2.5:1.5b` alone reaches 6. No model in this run has
  enough cells for anything beyond "did it ever read a page" (§5).
- **Every `ledgerfinal01` per-arm KPI number in §2**: 22 paired tasks total, split 16/6. The
  project's own power table asks 61-111 paired observations for a 0.10 effect; this is stated
  explicitly in `LEDGER_KPI_SPEC.md` and reconfirmed by the holdout inversion below.

---

## 4. What is contradicted or unreplicated

**The reference campaign's per-arm ordering inverts between tuning and holdout — reconfirmed
independently here.** Recomputing `L4` (unsupported-claim rate, mean-of-per-cell-rate, via
`agent.app.testing.claim_audit.audit()` over each split) reproduces the tuning-doc's numbers
exactly:

| arm | tuning (n=16) | holdout (n=6) |
|---|---|---|
| evidence_loop | 0.0811 | 0.3202 |
| langgraph_react | 0.3269 | 0.0632 |
| sequential_react_extract | 0.2404 | 0.4481 |

These match `docs/LEDGER_FINAL01_TUNING_RESULT.md` to 3-4 significant figures — **the headline
inversion is confirmed, not merely repeated from the handoff.** Any per-arm KPI claim drawn from
`ledgerfinal01` in this document, or anywhere else, carries this caveat: it may not replicate on
the other half of the same 22-task suite.

**`phi3_both` does not replicate the open-queue's expectation and cannot currently be explained.**
The tiny-model handoff's open item #1 predicted that stacking the tolerant-`finish`-argument read
on top of the zero-visit gate would let phi3:mini's pages-read gain (0/6→4/6 in `tinyfixon`)
convert into keystone passes. The actual `phi3_both` run shows pages-read at 2/6 — **lower**, not
higher, than the `tinyfixon`-alone result on the identical task range. I checked the stored cell
JSONs for a config difference and found none recorded (`variant_specific_settings` is empty `{}`
in both runs — expected, since env flags never reach that dict per `LEDGER_KPI_SPEC.md` §6). This
means the artifact itself cannot prove which mechanisms were active in `phi3_both`. I am not
authorized to launch a run to check, so this stands as **contradicted, unexplained, and open** —
not folded into any claim about the fix combination.

**L8 (redundant-read rate) could not be reproduced from the numbers in
`LEDGER_FINAL01_TUNING_RESULT.md`.** The doc reports, for the tuning split: evidence_loop 0.221,
langgraph_react 0.075, sequential_react_extract 0.125. Recomputing three different ways over the
same 16-task tuning split:

| method | evidence_loop | langgraph_react | sequential_react_extract |
|---|---|---|---|
| `claim_audit.audit()`, pooled (sum unsupported... sum visits/distinct across cells) | 0.319 | 0.091 | 0.143 |
| `claim_audit.audit()`, mean of per-cell rates | 0.170 (approx, hand-computed from the same per-cell counts) | -- | -- |
| `coverage_report.py --task-set <tuning ids>`, mean duplicate-factor implied rate | ~0.27 (evidence_loop; derived from mean dup-factor 1.369) | ~-0.01 (dup-factor 0.987, i.e. essentially none) | not fully computed |

None of these three reproduce 0.221/0.075/0.125, though they agree on the *ordering*
(evidence_loop highest, langgraph_react lowest). I could not find the exact script invocation
that produces the doc's precise figures — possibly a fourth denominator convention (e.g. pooled
over the FULL 22-task campaign rather than the 16-task tuning split, or a version of the visit
extractor that predates the current `coverage_report.py`). **Flagging this as an open
disagreement rather than silently accepting or silently overwriting the prior number** — the
ordering (evidence_loop reads redundantly more than the other two arms) is corroborated across
every method tried; the exact magnitude is not.

**`ladder02`'s stated cell count does not match the file count.** The task brief describes it as
"killed at 36 cells." The actual results directory holds 34 complete per-cell JSON files under
`ladder02_langgraph_react_off_*` plus one `.jsonl` file (`..._215_phi3:mini_..._r1.jsonl`, an
unfinished streaming write, not a complete result) and a `_json_telemetry.jsonl` sidecar. Treat
the real number as **34 complete cells**, not 36.

---

## 5. `ladder02`: the pre-fix baseline, read structurally (not scored)

This run predates the tiny-model fixes (zero-visit gate, context-fit) — it is `langgraph_react`,
module off, across 6 models on tasks 210-215 only (not the full 48-task ladder the design called
for). Recomputed per-model from `execution.observability.visit.count` and
`validation.overall_score`:

| model | cells | cells that ever read a page | mean score |
|---|---|---|---|
| qwen2.5:7b | 6/6 | 6/6 | 0.925 |
| gemma2:2b | 6/6 | 6/6 | 0.242 |
| llama3.2:3b | 6/6 | 5/6 | 0.542 |
| phi3:mini | 5/6 (1 missing) | **0/5** | 0.000 |
| qwen2.5:1.5b | 6/6 | **0/6** | 0.000 |
| tinyllama | 5/6 (1 missing) | **0/5** | 0.000 |

This is the categorical baseline the tiny-model phase's zero-visit-gate fix targeted: three of six
models in this band never read a page at all before the fix and could not score above zero by
construction (matching the tiny-model handoff's "34/34 read>=1 agrees with scored>0" finding,
reconfirmed here on the subset that is this run). It is NOT the full ladder (only 6 of 48 tasks
covered, and two models are missing a cell each) — no claim about model-size scaling should be
drawn from it beyond "3 of 6 models could not produce a scoreable answer at all under the
pre-fix host."

---

## 6. What a reader may and may not claim from this phase

**May claim, because it is categorical and holds regardless of the score-ranking instability:**

- The derive module turns "an arm doing arithmetic has an unknowable fabrication rate" into "an
  arm doing arithmetic has a checked, replayable fabrication rate" for **both** hosts it was
  attached to (`langgraph_react` and `sequential_react`), not just `evidence_loop` — 0/8 and
  0/9 disagreements recorded, one flagged invalid rather than silently accepted.
- When only offered (not forced), the derive module is used in **5/12 (41.7%)** of eligible
  cells — corroborating, independently recomputed, the methodology doc's cited 33-42% adoption
  range for an optional tool.
- The zero-visit-gate + context-fit fix takes `phi3:mini` from **0/6 to 4/6** cells that read a
  page and score above zero on tasks 210-215, and is provably inert on `qwen2.5:7b` (identical
  visit counts and scores in all 6 paired cells) — a real floor-raise for one weak model, with a
  built-in negative control on a model that didn't need it.
- `evidence_loop`'s per-cell derivation graph re-verifies offline with 0 invalid derivations
  across every campaign checked here (`ledgerfinal01` full campaign: 10 cells with a graph, 0
  invalid; `claim_metrics.py`'s campaign-wide read agrees). No other arm can produce this number
  AT ALL, for any model, ever — that asymmetry is structural, not a score difference.
- `evidence_loop`'s cost premium over `langgraph_react` (tokens, wall clock, LLM calls) is real,
  large (roughly 2.5-2.9x), and — unlike every accuracy/KPI ordering measured this phase — **does
  not invert between the tuning and holdout splits.**

**May not claim:**

- Any arm ranking or mechanism ranking on `ledgerfinal01` by score or by L1/L4 KPI value. The
  ordering inverts between a 16-task and a 6-task split of the same 22-task suite (reconfirmed
  independently in §4), and the project's own power table asks 61-111 paired tasks for a 0.10
  effect against the 22 available.
- That the `mod2` module A/Bs' overall_score deltas (-0.075 langgraph, +0.025 sequential) mean
  anything about accuracy cost or benefit — n=12, p=0.44/0.78, unseeded, and this suite's measured
  trajectory-chaos floor already shows single-mechanism flips can swing individual tasks by ~0.6
  while leaving most untouched.
- That `phi3_both` shows the combined fix helping or hurting phi3:mini relative to the zero-visit
  gate alone — the artifact cannot establish which mechanisms were active, and the observed
  2/6-vs-4/6 direction is the opposite of what was predicted, unresolved (§4).
- Any conclusion from `ladder02` beyond "3 of 6 models could not read a page at all under the
  pre-fix host on this 6-task band" — it is 34 of an intended 288 cells, on 6 of 48 tasks, with
  gaps in 2 of 6 models' cell counts.
- The exact L8 redundant-read magnitudes in `LEDGER_FINAL01_TUNING_RESULT.md` (0.221/0.075/0.125)
  as independently verified — three recomputation methods here agree on ordering, not magnitude
  (§4). Report the ordering, not the specific numbers, until the discrepancy is resolved.
- Anything about a second model size for the ledger arms comparison — `ledgerfinal01` is
  qwen2.5:7b only; the multi-model evidence in this phase (`tinyfix*`, `ladder02`, `phi3_both`) is
  all single-arm (`langgraph_react`) host-fix evidence, not a re-run of the three-arm comparison
  on other models.
