# Ledger handoff — 2026-09-01

**Branch:** `dagv2-evidence-ledger` · **Session start:** `0be7e38f` · **Session end:** `88a57429`
**Suite:** 8420 passed, 18 skipped, 0 failed (session start baseline: 8315)
**Spend:** $0.11 Serper (corpus build, prior session) + $0 this session. All inference local
(qwen2.5:7b via Ollama), all search served from a frozen corpus.
**Operative plan:** `docs/LEDGER_PLAN_2026-09-01.md`. It supersedes
`docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md` §1/§3/§4; that document's §7 risks and §9 non-goals
still stand.

---

## 1. What shipped

| # | SHA | What | Live? |
|---|---|---|---|
| 1 | `8981c13d` | Total, lossless quantity parsing (`parse_quantity`) | **yes**, default |
| 2 | `361af0a1` | Arm comparison scoped to execution variant + identical-arm guard | **yes** |
| 3 | `87fdcf24` | Repeat-refusal guard + derive operation aliases | **yes**, default on |
| 4 | `85ce9912` | Preregistration abort conditions become real gates | **yes** |
| 5 | `8ed673b6` | Yield-conditioned search-behaviour metric | report-only, no gate |
| 6 | `88a57429` | Per-search provenance (corpus / live / empty), persisted | **yes**, additive |

Plus one preregistered campaign, `ledgernum22r3`: 22 tasks × 3 arms × 3 reps = **198/198 cells,
100% completion, 2% infra-failure rate, $0**.

## 2. The headline result, stated plainly

**The master plan's founding thesis is not supported.** `evidence_loop` does not beat the linear
control on grounded score for a cheap model:

| pair | Δ score | p | p_holm | Δ tokens | Δ LLM calls |
|---|---|---|---|---|---|
| evidence_loop − langgraph_react | **−0.100** | 0.017 | 0.051 | +47,259 | +27.9 |
| evidence_loop − sequential_react_extract | +0.001 | 0.985 | 0.985 | +21,558 | +8.4 |
| langgraph_react − sequential_react_extract | +0.093 | 0.046 | 0.091 | −26,118 | −19.6 |

It ties `sequential_react_extract` exactly and loses to `langgraph_react` at ~3× wall clock
(52.6s vs 16.9s per cell). Nothing clears Holm, so nothing is a settled ranking — but the sign is
against the thesis and the cost gap is not marginal.

**What survives, and is stronger:** `evidence_loop` is the only arm whose confidence is
calibrated. Risk-coverage at threshold 0.5:

| arm | ALL | ANSWER_OR_PARTIAL | ANSWER_ONLY |
|---|---|---|---|
| evidence_loop (native verdict) | 0.439 | 0.453 | **0.600** (n=5) |
| langgraph_react (derived verdict) | 0.591 | 0.629 | **0.273** (n=11) |
| sequential_react_extract (derived verdict) | 0.576 | 0.585 | **0.231** (n=13) |

`evidence_loop` is monotone. Both derived-verdict arms are **inverted at their most confident
tier** — and that inversion, previously n=4–5 and flagged as too small to call, reproduced and
strengthened at n=11 and n=13. ~~**Calibration cannot be retrofitted onto an arm that kept no
ledger while it ran.** That is the defensible differentiator.~~

> **RETRACTED 2026-09-03.** Both claims in this section are superseded and must not be cited.
>
> *"Calibration cannot be retrofitted"* was refuted the same day it was written, in
> `LEDGER_KPI_PHASE_HANDOFF_2026-09-01.md` §3: a post-hoc, arm-blind audit reading only
> `output.pages` and the final text produces a MONOTONE channel for arms that kept no ledger at
> all. The code agrees — `agent/app/testing/claim_audit.py` and `confidence_channel.py` do exactly
> that, and the inversion above turned out to be an artifact of the old literal-match support
> rule, not a property of the arms.
>
> *"`evidence_loop` is monotone"* did not survive either. `docs/LEDGER_FINAL01_TUNING_RESULT.md`
> finds it NOT monotone on the tuning split (0.634 -> 0.623) and monotone on the sealed holdout,
> while the other two arms flip the other way. Monotonicity is a split artifact at this n, not a
> settled property of any arm.
>
> The numbers in the table are left exactly as measured; only their interpretation is withdrawn.

## 3. Layer 4 (the derivation graph) — working

Over 66 `evidence_loop` cells: 65 carry a graph artifact, 58 admit ≥1 verified SOURCE node, 26
emit ≥1 DERIVED node. **248 source nodes, 57 derived nodes, ZERO invalid derivations,
fabricated-arithmetic rate 0.** Verdicts are a real distribution: 36 ANSWER / 26 PARTIAL / 4
ABSTAIN.

Per cluster (mean score, evidence_loop / langgraph_react / sequential_react_extract):
arithmetic 0.629 / 0.788 / 0.629 · unit-refusal 0.431 / 0.361 / 0.472 · missing-operand
0.382 / 0.583 / 0.468 · fabrication-bait **0.447 / 0.272 / 0.307**.

The bait cluster is the one place `evidence_loop` leads clearly — which is the shape its
refuse-rather-than-fabricate machinery exists for.

## 4. The bug that mattered most

`parse_quantity` replaced a parser that produced **confidently wrong numbers marked valid**:

```
sum("121 crore", "162753003")  ->  162,753,124   derivation_valid: True
true value                     ->  1,372,753,003            (wrong by 8.4x)
```

Scale words were treated as opaque unit suffixes, so ×10⁷ was silently discarded; and
`_check_common_unit` treats a missing unit on one side as compatible, so the mismatched operands
combined without complaint. This was strictly worse than the bug the layer was built to prevent —
here the graph *vouched* for the wrong figure.

The replacement is **total**: every character is classified as currency, number, scale, unit or
restatement, and any residue refuses the parse. Five rules make it safe; the dangerous one is that
**single-letter abbreviations scale only behind a currency** — `$5m` is five million, `1,158 m` is
a tower, and 245 stored values end in a bare letter and are overwhelmingly metres.

**Regression gate, run over all 1,577 distinct stored value strings:** 14 magnitudes changed, *all*
scale-word corrections; **0 changed for any other reason**; 197 newly parsed (dual-unit
parentheticals, currency prefixes); 2 newly refused, both previously silent lower-bound leaks
(`"1 trillion to 2.6 trillion"` had been returning **1.0**).

## 5. May and may not be claimed

**May:** the derivation layer admits re-verifiable evidence and recomputes values with zero invalid
derivations; fabricated-arithmetic rate is 0 on this suite; ~~`evidence_loop`'s native verdict is
monotone against ground truth while both derived verdicts are inverted at ANSWER~~ (**RETRACTED
2026-09-03 — see §2**: the inversion was an artifact of the literal-match support rule, and
`evidence_loop`'s own monotonicity does not hold on the FINAL01 tuning split); the numeric suite
is leak-free (parametric floor 0.015 overall, exactly 0.000 on 18 of 22 tasks, against 0.48–0.61
with tools); `evidence_loop` costs ~3× wall clock and ~+47k tokens per cell versus
`langgraph_react`.

**May not:** any arm ranking. n=22 paired tasks — the pairing unit is the task, so reps do not
raise n — against this repo's own power table asking 61–111 paired observations. The preregistration
fixed this before data existed and the data does not change it. Also may not: treat `ledgernum22`
(reps=1) and `ledgernum22r3` (reps=3) as a controlled comparison — the parser, repeat-guard and
alias changes landed between them, so they are different systems.

## 6. Retraction ledger

1. **"reps=5 → n=110 per arm, crossing the power threshold."** RETRACTED, mine. The pairing unit in
   a paired comparison is the **task**; 22 tasks at any rep count is 22 paired observations. Reps
   cut within-task noise, they do not move n. The repo already knew this
   (`DAG_V2_PREFLIGHT_2026-08-15.md:164`: "power comes from task count, not reps").
2. **"Zero live fallbacks" on `ledgernum22`, stated as measured.** RETRACTED, mine. It was an
   *inference* — `LEDGER_MAX_LIVE_FALLBACKS=0` means a fallback would raise and no cell errored —
   not a measurement. `live_fallbacks` lived only on the connector instance and was never persisted.
   Fixed in `88a57429`; the gate reports UNKNOWN for every cell predating it.
3. **"Bad searching costs 2.2× more queries" (strong signal).** WEAKENED on the larger sample.
   evidence_loop high-vs-low scoring cells were 2.60 vs 5.75 queries at n=22; at n=66 it is 2.86 vs
   3.47. Direction still holds in all three arms, magnitude much smaller. The first estimate was
   partly small-sample noise. This is exactly why the metric was kept report-only.

Earlier retractions (the "233 traces / capture never ran" claim, the trace-JSONL misdiagnosis, the
two-file registration error, Lane C's blast-radius reasoning) are recorded in
`docs/handoffs/LEDGER_PROGRAM_HANDOFF_2026-08-31.md` and are not repeated here.

## 7. Open work, in priority order

1. **Visit dedup.** `evidence_loop` re-reads 71 of 264 visits (27%); `sequential_react_extract` 101
   of 296 (34%); `langgraph_react` only 11%. The loop dedups searches and has **no visit dedup at
   all**. Every re-visit re-injects a full page into the prompt — a direct attack on the +47k-token
   gap, and now evidence-backed rather than speculative.
2. **Test the claim-poverty hypothesis** behind the inverted ANSWER tier: the derived rule marks
   ANSWER when *every* checkable claim is grounded, which may select for terse, claim-poor answers.
   Testable offline against stored cells. Do not change the rule before testing it.
3. **Decide the live-fallback gate policy** — sum across the run, or max per cell? Provenance now
   persists; absent must stay UNKNOWN, never 0.
4. **Claim-level metrics on the numeric suite** (`scripts/claim_metrics.py`) per arm — the KPI the
   pivot named.
5. **A second model size.** Every number in this handoff is qwen2.5:7b.
6. **`sequential_react` evidence persistence** — only `langgraph_react` was fixed (`2cdc9066`).
7. **Reconcile `PRE_BARRAGE_AUDIT.md`**, which still declares `IDEA_TEST_CONCURRENCY=1` "mandatory,
   correctness-load-bearing" (self-dated 2026-07-06) although the per-slot connector pool shipped
   2026-08-09 and `ADAPTIVE_ENGINE.md:143` calls concurrency "a choice, not a hard limit."

**Explicitly deferred:** the Phase B deterministic queue (no evidence scheduling is the
bottleneck); promoting tasks 210–231 into `ACTIVE_SUITE_IDS` (moves the benchmark denominator);
subsystems 1 and 2, including the `requires_data` backfill bug whose writer sets `source_node_id`
from ancestors of the node's *parent* while the check reads the parent's *children*, so its output
can never satisfy the check that reads it.

## 8. How to reproduce anything here

```
# the campaign
SEARCH_PROVIDER=corpus LEDGER_CORPUS_DIR=agent/idea_test_results/corpus/numeric22 \
LEDGER_MAX_LIVE_FALLBACKS=0 LLM_PROVIDER=ollama MODEL_API_URL=http://127.0.0.1:11435/v1 \
OPENAI_API_KEY=dummy IDEA_TEST_MODELS=qwen2.5:7b IDEA_TEST_VALIDATION_MODEL=qwen2.5:7b \
IDEA_TEST_IDS=$(seq -s, 210 231) IDEA_TEST_RUNS=3 IDEA_TEST_CONCURRENCY=1 \
IDEA_TEST_EXECUTION_VARIANTS=evidence_loop,sequential_react_extract,langgraph_react \
IDEA_TEST_RUN_ID=ledgernum22r3 PYTHONPATH=.:services:agent \
  ./.venv/bin/python -m agent.app.idea_test_runner

# the analyses
./.venv/bin/python scripts/prereg.py audit --run-id ledgernum22r3
./.venv/bin/python scripts/risk_coverage.py --run-ids ledgernum22r3 --threshold 0.5
./.venv/bin/python scripts/query_quality.py --run-ids ledgernum22r3 --by-score
./.venv/bin/python scripts/compare_arms.py ledgernum22r3@evidence_loop:evidence_loop \
    ledgernum22r3@langgraph_react:langgraph_react
```

**Two traps for the next session.** `IDEA_TEST_VALIDATION_MODEL` defaults to `gpt-5-mini`, which a
local Ollama does not serve — the run aborts at pre-flight unless you set it. And the runner must be
invoked as `python -m agent.app.idea_test_runner`, not by path, or `agent` fails to resolve as a
package.
