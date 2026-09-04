# Tiny-model FinishGate compliance/cost probe (2026-09-04)

Workstream 2 §3 feasibility probe. $0, local `badmodel-ollama` only, `SEARCH_PROVIDER=corpus`
(deterministic replay, `fallback=none` confirmed in every log — no live search/API calls of any
kind). Branch `dagv2-evidence-ledger`, merge `b1438e18` (contains `FinishGate` + the SOURCE-node
quote-capture fix). `LLM_SEED=12345` for byte-reproducibility. `LEDGER_CORPUS_DIR=agent/idea_test_results/corpus/numeric22`.

**36/36 cells completed, zero "Setup failed" in any of the 6 campaign logs.**

## Matrix

- Models: `qwen2.5:0.5b`, `qwen2.5:1.5b`, `llama3.2:3b` (all served by `badmodel-ollama` on
  `127.0.0.1:11435`)
- Tasks: dev numeric 210, 211, 212, 214, 215, 220 (holdout 213/217/221 never run/cited)
- Arms: `gate_off` (baseline settings, `final_require_derivation_for_numeric` absent/false),
  `gate_on` (same settings file + `final_require_derivation_for_numeric: true`, applied via
  `IDEA_DAG_SETTINGS_PATH` pointing at a full copy of `idea_dag_settings.json` with that one key
  flipped — no code edited)
- Both arms: `sequential_react` host with `LEDGER_HOST_MODULES=derive` (derive module bound in
  both arms; only the *gate* differs)
- Run IDs: `gateprobe01_{q05,q15,l3b}_{off,on}`, each a `scripts/run_campaign.sh` launch, run
  sequentially (singleton lock respected — no other campaign was alive; `pgrep -af
  idea_test_runner` was clean before launch)
- Cell/report artifacts: `agent/idea_test_results/gateprobe01_*_r1.json` (36 cells) +
  matching `*_report_v3.json` (verbosity 3) + `_campaigns/gateprobe01_*.{env,log}`

## 1. Headline compliance/cost table (per model, averaged over 6 dev tasks)

| model | arm | avg score | avg LLM calls | avg tokens | avg visits | derived nodes (sum/6) | abstain-banner cells |
|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b | off | 0.048 | 14.2 | 20,614 | 0.7 | 0 | 0 |
| qwen2.5:0.5b | on  | 0.048 | 14.7 | 21,481 | 0.7 | 0 | 1 |
| qwen2.5:1.5b | off | 0.255 | 18.8 | 70,252 | 4.5 | 1 | 0 |
| qwen2.5:1.5b | on  | 0.297 | 22.0 | 85,196 | 5.5 | 1 | 2 |
| llama3.2:3b  | off | 0.497 | 23.3 | 102,411 | 12.2 | 1 | 0 |
| llama3.2:3b  | on  | 0.497 | 26.5 | 117,582 | 14.2 | 1 | 6 |

Token deltas ON vs OFF are driven almost entirely by a small number of cells that hit a real
gate retry cycle — 13/18 paired cells are **byte-identical token/call counts** between arms. See
§2 for the per-cell breakdown.

## 2. Gate telemetry — per-cell, ON arm only (18 cells)

| model | task | gate_evaluations | gate_refusals | gate_final_state | gate_retries_used | score Δ (on−off) | token Δ% |
|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 210 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| qwen2.5:0.5b | 211 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| qwen2.5:0.5b | 212 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| qwen2.5:0.5b | 214 | 2 | 2 | forced_by_budget | 1 | 0.00 | +17.3 |
| qwen2.5:0.5b | 215 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| qwen2.5:0.5b | 220 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| qwen2.5:1.5b | 210 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| qwen2.5:1.5b | 211 | 2 | 2 | forced_by_budget | 1 | **+0.25** | **+820.6** |
| qwen2.5:1.5b | 212 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| qwen2.5:1.5b | 214 | 1 | 1 | forced_by_budget | 0 | 0.00 | 0.0 |
| qwen2.5:1.5b | 215 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| qwen2.5:1.5b | 220 | 1 | 0 | passed | 0 | 0.00 | 0.0 |
| llama3.2:3b | 210 | 1 | 1 | forced_by_budget | 0 | 0.00 | 0.0 |
| llama3.2:3b | 211 | 1 | 1 | forced_by_budget | 0 | 0.00 | 0.0 |
| llama3.2:3b | 212 | 2 | 2 | forced_by_budget | 1 | 0.00 | +7.6 |
| llama3.2:3b | 214 | 1 | 1 | forced_by_budget | 0 | 0.00 | 0.0 |
| llama3.2:3b | 215 | 2 | 2 | forced_by_budget | 1 | 0.00 | **+542.7** |
| llama3.2:3b | 220 | 1 | 1 | forced_by_budget | 0 | 0.00 | 0.0 |

`gate_final_state` distribution across the 18 ON cells: **9 `passed`, 9 `forced_by_budget`, 0
`abstained`** (no cell ever completed the full 2-retry cycle to a clean "abstained" state — see
§3). `gate_retries_used` distribution: 14×0, 4×1, 0×2.

## 3. Does the mandate produce a usable `derive()` call, loop, or garbage?

**No cell in the entire 36-cell matrix shows an increase in successful `derive()` usage between
arms.** Summed derived-node count per model, OFF vs ON, is byte-identical: qwen2.5:0.5b 0/0,
qwen2.5:1.5b 1/1, llama3.2:3b 1/1 (out of 6 tasks each). The one derive node each of qwen2.5:1.5b
and llama3.2:3b produced appears on the **same task (212) in both arms** — i.e. the gate did not
cause a NEW derive call anywhere; the only task where any of these three models ever call
`derive()` at all is one they'd already call it on with the gate off.

Ledger-side rejections (`evidence_graph.counts.rejected*` — a malformed/mismatched derive
argument) are **0 in all 36 cells**: whenever a model does call `derive`, the call itself is
well-formed. The compliance failure is entirely upstream of the tool call — the models simply do
not reach for `derive` when computing an answer; they do the arithmetic in prose and hand the
gate a text answer with no ledger node behind it.

Concretely (task 210, llama3.2:3b, `gate_on`, `final_state=forced_by_budget`):

```
[ABSTAINED -- one or more numbers in this answer could not be verified against the
evidence ledger.] Based on the provided evidence: ...
419.7 metres - 381 meters = 38.7 metres.
Report:
(a) The computed absolute difference is 38.7 metres.
```

The model computed `419.7 - 381 = 38.7` in prose (never called `derive`), the gate correctly
flagged `38.7` as unbacked, refused the finish, then — with the step budget already exhausted —
force-terminated and **prepended a banner to the SAME unedited answer text**. No loop, no
garbage: this is the modal failure mode (9/18 ON cells), and it is exactly the shape
`docs/analysis/GATE_PRECISION_PRECHECK_2026-09-04.md` motivated the gate against, but the gate
as scoped cannot compel the corrective behavior — it can only decorate the output after the
fact once the run is out of steps.

## 4. Step/token cost delta ON vs OFF — the metric that killed `require_finish_tool`

13/18 paired cells: **exactly 0% token/step delta.** The model was already at (or one step from)
`IDEA_TEST_SEQUENTIAL_MAX_STEPS` (25, default) when it called `finish()`, so a refused finish
had no budget left for even one corrective retry — `gate_retries_used=0`, `forced_by_budget`
fires immediately, and the run's cost is unchanged. This is a **measurement-instrumentation
implication, not a mechanism win**: the gate "worked for free" only because there was nothing
left to spend.

The 4 cells where a retry *did* fire (`gate_retries_used=1`) show the real cost signature:

| model/task | token Δ% | score Δ |
|---|---|---|
| qwen2.5:0.5b/214 | +17.3% | 0.00 |
| qwen2.5:1.5b/211 | **+820.6%** (7→26 calls) | **+0.25** |
| llama3.2:3b/212 | +7.6% | 0.00 |
| llama3.2:3b/215 | **+542.7%** (8→26 calls) | 0.00 |

Two of four retry cells burn **5.4×–9.2× the LLM-call budget** for zero or marginal score
change — the same shape that sank `require_finish_tool` (see MEMORY: "step-budget cost").
qwen2.5:1.5b/211 is the ONLY cell in the whole 36-cell matrix where the gate changed the score at
all, and it cost 8.2× the tokens to do it.

## 5. Abstention-banner rate and score movement (secondary, n=36, no claims)

Raw abstention-banner counts (deliverable prefixed with `[ABSTAINED ...]`): qwen2.5:0.5b 1/6,
qwen2.5:1.5b 2/6, llama3.2:3b 6/6 — llama3.2:3b's higher step usage (avg 23.3 calls off-arm,
near the 25-step cap) means almost every one of its finish calls lands with zero retry budget
left, so the gate degrades to "always banner, never correct."

Score movement: 17/18 paired cells **unchanged** (banner is cosmetic — it prepends text but does
not strip or replace the model's committed number, so the deterministic grep validator scores
the SAME underlying number either way). 1/18 improved (+0.25, qwen2.5:1.5b/211, at 8.2× cost).
0/18 got worse. **The banner does not corrupt a correct answer**, but it also does not fix a
wrong one — it is presently pure disclosure, not correction.

## 6. Quote-capture fix live check (critical finding requested)

Only **4/36 cells** produced any SOURCE node in the evidence graph at all — all four are task
212 (llama3.2:3b off/on, qwen2.5:1.5b off/on), 2 SOURCE nodes each. **Every one of those 8 SOURCE
nodes has `quote_verified: true`** — fraction = **8/8 = 100%**, matching the "~100% where nodes
exist" expectation. qwen2.5:0.5b never produced a single SOURCE or DERIVED node across all 12 of
its cells (both arms), so it contributes 0 data points to this check, not a null/false one.

**Verdict: the quote-capture fix is live and confirmed working** wherever the ledger actually
gets a node — but coverage is the real constraint, not verification quality. 32/36 cells (89%)
never engage the ledger's derive path at all, so `quote_verified` never has anything to be
true/false/null ABOUT in those cells. This is the same "starved upstream, not broken downstream"
pattern flagged in `docs/handoffs/LEDGER_COMPONENT_PHASE_2026-09-03.md` §4 — the constraint is
acquisition (getting tiny models to call `derive`), not the evidence layer.

## 7. Clause-4 pass rate (manual — `scripts/ledger_risk_coverage.py` does not support run-id
filtering)

`ledger_risk_coverage.py:310/317` hardcodes `_CELL_FILE_RE = re.compile(r"^ladder03_.+_r\d+\.json$")`
and globs only `ladder03_*.json` — it silently discovers 0 files against a `gateprobe01_*`
results dir with no error. Not run against this probe's cells (no code edit made to add a
filter flag); clause-4 (quote_verified on every SOURCE node backing a derivation operand) is
summarized manually above: 8/8 = 100% where evaluable, 0 evaluable nodes in the other 32 cells.

## 8. Bugs / gaps found (no code changed)

1. **The finish gate cannot compel `derive()` usage** — it only ever refuses/decorates a
   `finish()` call that already happened; it has no mechanism that nudges the model toward the
   tool earlier in the run. Given tiny models spend most of their step budget before ever
   reaching finish (llama3.2:3b avg 23.3/25 calls off-arm), the gate structurally has almost no
   room left to iterate once it fires. This is the single biggest lever for a follow-up: either
   reserve step budget for the corrective retry (e.g. cap the free-form step count so N steps
   are always held back for gate retries) or move the nudge earlier (system-prompt emphasis /
   mid-run reminder once N numeric tokens have appeared without a `derive` call).
2. **`scripts/ledger_risk_coverage.py` is a hard `ladder03_*`-only tool** — any future probe
   with a different run-id prefix gets silently zero-filed rather than an error. Worth a
   `--campaign-prefix` flag (not added here; out of scope, no code edited).
3. Confirms (does not newly find) the `FinalConfig.require_derivation_for_numeric` docstring's
   own caveat: "a no-op wherever the run has no bound ledger to check against" is slightly
   incomplete in one respect — the gate is ALSO effectively a no-op whenever the ledger is bound
   but empty (model never called `derive`, `has_ledger_context=False` in
   `execution_sequential.py:257`), which is the dominant case for the tiniest model (5/6 `passed`
   trivially for qwen2.5:0.5b). This is by design (refinement 4, documented in the code), but it
   means "gate ON" measures almost nothing for a model that never touches the ledger — the KPI
   this probe was built to move (compliance) is gated by a DIFFERENT, unaddressed problem: get
   the model to call `derive` in the first place.

## Files

- Analysis inputs: `/home/muk/projects/webRAG/agent/idea_test_results/gateprobe01_*_r1.json`
  (36 cells) and matching `*_report_v3.json`
- Campaign envs/logs: `/home/muk/projects/webRAG/agent/idea_test_results/_campaigns/gateprobe01_*.{env,log,pid}`
- Gate-on settings file used (not committed, scratch): copy of `agent/app/idea_dag_settings.json`
  with `final_require_derivation_for_numeric: true`, passed via `IDEA_DAG_SETTINGS_PATH`
- Relevant source read (not edited): `agent/app/testing/execution_sequential.py` (`FinishGate`,
  `_finish_gate_predicate`, telemetry emission ~L690-737), `agent/app/idea_policies/config.py`
  (`FinalConfig.require_derivation_for_numeric` ~L425-434), `agent/tests/execution_sequential_finish_gate_test.py`
