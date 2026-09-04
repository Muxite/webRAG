# Pre-registration: Ledger risk-coverage discrimination analysis (2026-09-04)

Written **before** `scripts/ledger_risk_coverage.py` exists or runs. This is a $0, offline,
analysis-only prereg — no live cells are authorized by this document. Genre/discipline follows
`DAG_V2_FIXES_REVALIDATION_PREREG_2026-08-21.md`. Plan of record:
`~/.claude/plans/plan-the-next-phase-soft-graham.md` (Workstream 3), incorporating the
adversarial-review blockers (pooled-signal/arm confound; operand_supported circularity;
quote_verified null flood).

## The question

> Among cells that actually produced an evidence ledger (derive-ON arms), does the composite
> certify signal separate correct from incorrect final answers — i.e., if a consumer accepted
> only certified answers, how much does error rate drop, and at what coverage?

This is a **discrimination** claim about the audit signal, explicitly NOT an arm-vs-arm or
model-vs-model score claim (`arm_ranking_claim_permitted: false` stands, frozen by
`agent/tests/ledger_kpi_spec_frozen_test.py`). KPI: `L1 risk_coverage_calibration` in
`scripts/ledger_kpi_spec.json`.

## Certify signal (fixed before analysis)

A cell is **certified** iff ALL of:
1. ≥1 DERIVED node exists in its stored `evidence_graph`;
2. every DERIVED node has `derivation_valid is True` (arithmetic axis);
3. every DERIVED node has `operand_supported is True` (new W1 axis, recomputed over stored
   graphs by one code version via `reverify_graph` — never by mid-sweep engine edits);
4. `quote_verified` is True on every SOURCE node backing a derivation operand —
   **null is fail-closed** (null ⇒ not certified). Known: null rate is ~100% on ladder03
   weak-model nodes, so per-stratum null rates MUST be published next to every curve;
5. no unbacked final-answer number: every number in the deliverable matches (≤0.5% relative
   tolerance) some evidence-graph node value.

Sensitivity variants (reported, not headline): the signal minus clause 4 (quote axis), and
minus clause 3 (operand axis) — to show which axis does the discriminating work.

## Cell inclusion (fixed)

- **Campaigns**: `ladder03_*` only (homogeneous prompts, both hosts, 7 local models × {off,
  derive}). Historical campaigns (`dagv3p*`, `night_a1`, `wk2_el`, `capability_spectrum_v2`, …)
  are EXCLUDED (heterogeneous prompts/hosts). The queued paid-API ladder cells
  (`api_*`, per `agent/idea_test_results/prereg/api_order.json`) join as a separate stratum
  when they land — same rules.
- **Split**: dev tasks only for all headline numbers. Holdout tasks **213/217/221** are
  analyzed ONLY in the operand_supported confirmatory section (below) and clearly labeled.
- Cells with `infra_failed` severity are excluded and counted.

## Reporting shape (fixed)

1. **Headline**: risk-vs-coverage curve WITHIN derive-ON cells, pooled across models, dev
   split. Fixed operating points: coverage 50% / 80% / 100% (plus the signal's natural
   operating point, reported alongside, never substituted). Full curve always published.
2. **Context only**: the pooled all-arms curve, WITH the arm composition of certified cells
   disclosed (expected ~100% derive-ON; the pooled curve is mechanically confounded with arm
   membership because OFF cells store no evidence_graph — this is why it cannot be the
   headline).
3. **Baselines**: (a) accept-everything; (b) answer-present (deliverable contains any number
   and no abstention banner). Same strata.
4. **Gate-precision audit**: explicit inversion lists — certified cells scoring ≤0.2 and
   rejected cells scoring ≥0.9, by filename (the roster-gate test).
5. **Strata**: per-model and per-host tables wherever n≥6 cells; smaller strata shown as raw
   counts without rates.
6. **Correctness**: deterministic validator score; "wrong" = score < 0.5 (secondary table at
   < 0.9). No LLM grading exists in the numeric suite (confirmed 2026-09-04).

## operand_supported circularity control (fixed)

The operand_supported rule was motivated by inspecting dev-split failures (tasks 215/219/220,
gemma 210). Its discrimination power is therefore reported in TWO clearly separated sections:
- **Development evidence**: dev-split cells (includes the motivating cells; circular by
  construction; labeled as such).
- **Confirmatory**: holdout 213/217/221 only. If holdout n is too small to conclude, the
  report says exactly that; dev-split numbers are never promoted to confirmation.

## What would count as each outcome

- **Signal works**: within derive-ON cells, certified error rate is meaningfully below
  uncertified at ≥50% coverage, and the inversion lists are short/explainable.
- **Signal inverts** (roster-gate mode): certified cells are NOT enriched for correctness, or
  high-score cells are systematically rejected — reported plainly as the phase result.
- **Signal starved**: coverage ≈ 0 because of the quote_verified null flood — reported as
  "the certify chain is blocked at quote capture," which redirects the next phase to quote
  capture, not to more certification logic.

No p-values are promised at ladder03 n; this is a descriptive discrimination analysis with
pre-fixed operating points, not a hypothesis test.
