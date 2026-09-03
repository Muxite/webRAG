# bughunt01 — handoff, 2026-09-03

Branch `dagv2-evidence-ledger`, launched 2026-09-02T22:48:16Z at commit `c5eb2106`, run_id
`bughunt01`. No prior handoff mentions this campaign.

## 1. What ran

3 models (`qwen2.5:7b`, `gemma2:2b`, `phi3:mini`) x 2 arms (`langgraph_react`, `evidence_loop`) x
12 tasks (210-221, the numeric22 chain/derivation family) = **72 cells**, seeded
(`LLM_SEED=12345`), frozen corpus (`SEARCH_PROVIDER=corpus`, `LEDGER_CORPUS_DIR=.../numeric22`,
`LEDGER_MAX_LIVE_FALLBACKS=0`), `IDEA_TEST_CAPTURE_LLM_IO=1`, `IDEA_TEST_KEEP_TRACES=1`, $0. Every
cell carries a `run_config` block, so the env this campaign actually ran under is
self-verifiable rather than asserted.

It answers `docs/handoffs/LEDGER_TINY_MODEL_PHASE_2026-09-02.md` §7 item 1, verbatim: *"the
zero-visit gate got [phi3] to grounded evidence; the tolerant finish read landed after those
runs, so the combination is unmeasured. Run `phi3_both` and see whether pages-read converts to
keystone."* All numbers below were computed independently from the 72 stored cell JSONs plus
`bughunt01_summary.json` — none were copied from a prior write-up of this run.

## 2. The answer: NO — pages-read does not convert to keystone for phi3

| model | arm | pages read >=1 | scored >0 | keystone passed |
|---|---|---|---|---|
| phi3:mini | evidence_loop | 5/12 | 6/12 | **0/12** |
| phi3:mini | langgraph_react | 7/12 | 7/12 | **1/12** |
| gemma2:2b | evidence_loop | 11/12 | 11/12 | 2/12 |
| gemma2:2b | langgraph_react | 12/12 | 12/12 | 5/12 |
| qwen2.5:7b | evidence_loop | 12/12 | 12/12 | 8/12 |
| qwen2.5:7b | langgraph_react | 12/12 | 12/12 | 11/12 |

("scored >0" = `validation.overall_score > 0`; "keystone passed" = the `keystone_<task_id>` entry
in `validation.grep_validations`.)

phi3 now reads pages and clears the `overall_score > 0` floor almost every time it visits one
(5/5 evidence_loop, 7/7 langgraph_react cells that read a page also scored above zero) — the
zero-pages-read bug from §3 of the tiny-model handoff is fixed. But the keystone check — whether
the specific number the task asks for is present and grounded — passes on only 1 of the 24 phi3
cells across both arms. Getting phi3 to the page is necessary and, on this evidence, nowhere near
sufficient: it reads the page, scores partial credit for doing so, and still does not land the
number the task is graded on. The open question from §7 item 1 is answered in the negative.

Reading further does not fix it either: qwen2.5:7b (12/12 pages everywhere) still only converts
8/12 and 11/12 of that into keystone passes — so even at 100% page-read coverage there's a real
gap between "read the source" and "grounded the graded number," it is just far smaller for the
stronger model. This campaign cannot say why phi3's gap is so much larger — that needs
per-cell failure-mode reading, not aggregate counts.

## 3. infra_failed: 10/72 cells, mostly one model/arm pair, mostly full visit failure

| model | arm | infra_failed |
|---|---|---|
| phi3:mini | evidence_loop | 9/12 |
| gemma2:2b | evidence_loop | 1/12 (task 218) |
| all other model x arm pairs | 0/12 |

Of the 9 phi3/evidence_loop `infra_failed` cells, **7 carry a `visit`/`http_request` failure rate
of exactly 1.0** (tasks 210, 212, 213, 214, 216, 218, 220); the other 2 do not (task 217: 0.6,
task 219: 0.5455) — every fetch phi3 attempted in those two cells did not fail, only most of
them. Report "9 of them ... with visit failure rates of 1.0" as approximately but not exactly
right: 7/9, not 9/9, are literally 1.0.

**These 10 cells are flagged for investigation as a probable misclassification.** The current
`infra_failed` gate does not yet distinguish "the network/site was actually down" from "the model
chose a URL that predictably 404s or gets refused" — a model-chosen bad URL currently counts as
infrastructure failure. Until that gate is fixed, **any rate computed by including or excluding
these 10 cells is provisional**, including the pages-read and keystone rates in §2 above (which
include them as-is, unmodified — they are not excluded, and excluding them would also be a
choice, not a neutral default).

## 4. Derive-module adoption is a model property, not a constant — and disagrees with the
   number I was given to check

Measuring "adoption" as *the evidence_graph in this evidence_loop cell contains at least one
`derived` node* (i.e. the model actually got a `derive` call to produce output, not merely typed
`{"action": "derive", ...}` and had it refused):

| model | derive-adoption (evidence_loop only) |
|---|---|
| qwen2.5:7b | 9/12 (75%) |
| gemma2:2b | 3/12 (25%) |
| phi3:mini | 0/12 (0%) |

**I independently computed 9/12 and 3/12 here; the number I was asked to verify was 8/12 and
4/12.** Both are off by one cell in opposite directions from my count (qwen: 212, 213, 214, 215,
216, 217, 218, 220, 221 = 9 cells with >=1 derived node; gemma: 211, 213, 217 = 3 cells). phi3's
0/12 matches exactly. I did not find a second, equally reasonable node-based definition that
reproduces 8/4 exactly — a looser definition ("attempted a derive call at all, counting refusals
that produced zero derived nodes") gives 11/12 and 7/12 instead, which is further off, not
closer. I flag this as an open discrepancy rather than silently adopting either count.

Whichever exact count is right, the shape is unambiguous and is the actual finding:
`docs/LEDGER_METHODOLOGY.md` §1.2 cites a rung-1 adoption estimate of **"measured 33-42%"**,
sourced from a single model (`moduse01`, `langgraph_react` host, 5/12 and 4/12 — see
`docs/analysis/PHASE_KPI_SYNTHESIS.md` §1c). Across three models under seeded, identical
conditions, this campaign's adoption rates are **0%, 25-33%, and 75-92%** — the 33-42% band is
contradicted at both ends, not just refined. Report derive-module adoption as a per-model
property from here on, never as a single constant carried over from a one-model measurement.

## 5. Zero invalid derivations

Across all 72 cells (36 `evidence_loop` cells x however many `derived` nodes each produced, plus
the 36 `langgraph_react` cells which have no evidence_graph at all), zero nodes have
`derivation_valid: False`. This confirms the number I was given for this item; no disagreement.

## 6. What may and may not be claimed

**May claim:**
- Getting phi3:mini to read a page (the tiny-model-phase fix) does not convert to grounded
  keystone answers; the gap it closed and the gap that remains are different gaps.
- The 33-42% single-model adoption band in the methodology doc does not generalize across models;
  treat adoption as per-model.
- Zero invalid (fabricated-arithmetic) derivations across all 72 cells, all models, both arms.
- The `infra_failed` flag currently conflates true infrastructure failure with model-chosen bad
  URLs; 10 cells here are affected and any rate touching them is provisional pending that fix.

**May not claim:**
- Any mean-score ranking between `langgraph_react` and `evidence_loop`, or between any two
  models. This is 12 tasks per cell group; `docs/LEDGER_PLAN_2026-09-01.md` §2 and
  `docs/LEDGER_FINAL01_TUNING_RESULT.md` have already shown per-arm KPI orderings on this same
  numeric-suite family INVERT between task subsets at n well above 12. Nothing here is powered to
  say one arm is better than the other.
- A single derive-module adoption percentage. See §4 — it varies by roughly 75 percentage points
  across three models measured identically.
- That the derive-adoption counts in §4 are settled to the cell. I disagree by one cell each with
  the number I was asked to check and did not find a definition that closes the gap; treat both
  as approximate until someone re-derives it from the raw scratchpad text (not captured in this
  campaign — `IDEA_TEST_CAPTURE_LLM_IO` records char counts, not the raw completions, on this
  transport; see the open tiny-model-phase queue item 5).
