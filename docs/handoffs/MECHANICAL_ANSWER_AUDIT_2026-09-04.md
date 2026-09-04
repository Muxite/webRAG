# Mechanical answer-audit phase — results and handoff (2026-09-04)

Prereg: `MINT01_PREREG_AMENDMENT_2026-09-04.md` (+ 4 machine specs, prereg/mint01_*.json).
Corpus: mint01 = 144 seeded $0 corpus-replay cells (4 local models x 2 hosts x 9 dev tasks x 2
reps), all four run_ids 36/36 complete, 0 infra_failed, 0 live fallbacks. Code: 9239b1ce +
cbf068bb; analysis `ledger_risk_coverage.py --prefix mint01`. Base wrong-rate (score<0.5):
70.8% pooled; 79.2% on the zero-derive stratum (n=120); 29.2% on nonzero-derive (n=24).

## 1. The 5-clause certify chain works live now — quote starvation is over

First post-quote-fix corpus: quote_null_rate 0.0% on all 77 checked SOURCE nodes (was ~100%).
Full 5-clause signal: 4/144 certified at **0% risk vs 70.8% base**; minus_quote is now
IDENTICAL to the full signal (the quote clause no longer subtracts coverage). Zero
certified-but-wrong cells. All 4 certified cells are qwen2.5:7b (11.1% coverage in-model at 0%
risk vs its 25% base). Coverage is bounded by derivation itself: 120/144 cells still mint zero
DERIVED nodes.

## 2. answer_supported on the zero-derive stratum: HONEST NEGATIVE per the prereg rule

backed_or_derived: 14/120 accepted (11.7% coverage) at 57.1% risk vs 79.2% base = 0.72x. The
prereg rule needed <=0.5x at >=15% coverage — both prongs fail. Directionally right, not
enough. Verdict: answer_audit stays informational telemetry; do NOT promote answer_supported
to a consumer-facing accept signal for zero-derive cells. Root cause is upstream of provenance:
on these tasks a wrong answer routinely restates real page numbers (wrong entity, wrong
operand), which provenance alone cannot see.

## 3. Where derivation exists, answer_supported nearly triples certify coverage at the same 0% risk

Nonzero-derive stratum (n=24): 5-clause certifies 4/24 (16.7%); answer_supported
(backed_or_derived) accepts **11/24 (45.8%) at 0% risk** vs 29.2% base. Zero
supported-but-wrong cells in the stratum. Off-prereg secondary finding — worth a confirmatory
prereg'd measurement before any consumer relies on it.

## 4. backed_only is an ANTI-signal on computation tasks (7/7 wrong)

Cells whose answer numbers are all verbatim-backed with no mechanical derivation possible:
100% wrong (7/7). A fully-verbatim answer to a computation question means the model copied
instead of computing. Cheap deterministic wrongness flag; candidate for the next phase.

## 5. Machinery notes

- A 2-cell smoke caught 5 real defects a 9,277-test green suite missed (unit-spelling
  arithmetic refusal m/metres; ambiguity 72x-inflated by duplicate page registrations;
  execution_langgraph dropping answer_audit; "GRES-2" extracted as -2.0; parenthetical
  ft-conversions needing a unit-ANCHORED fallback gated on `unit_bearing`). All fixed+tested
  (cbf068bb) and re-smoked before the campaign.
- Post-launch analysis-code fix, disclosed: arm detection used string equality on
  LEDGER_HOST_MODULES and classified all mint01 cells "off"; fixed to token membership
  (test added). No clause math touched; old-corpus outputs unchanged.
- operand_supported: still 24/24 True live (unfalsified outside offline fixtures).
  op_appropriateness: 42 derived nodes checked, 0 flags — no wrong-operation case recurred in
  this corpus; the checks remain validated only by adversarial unit fixtures.
- gate-precision inversion lists: 0 certified-but-wrong; 12 rejected-but-right (conservative
  failure mode, acceptable).

## Next phase candidates, ranked

1. **backed_only-as-wrongness-flag**: prereg + measure the anti-signal (cheap, deterministic,
   already computed per cell).
2. **Confirmatory run for finding 3** (answer_supported on derive-bearing cells) on a fresh
   prereg'd corpus, including the paid-API stratum where derive rates are higher.
3. **Attack the zero-derive wall itself**: the bottleneck is now clearly "tiny models never
   derive" AND "provenance can't catch wrong-entity copying". Candidate: mechanical
   task-shape-driven derivation (host derives the difference/ratio the mandate asks for from
   index quantities and compares to the answer) — extends audit_answer from explaining the
   model's number to checking it.
4. Deferred unchanged: gpu0831 parity rerun; second-hostname visit policy.

Traps: never cite quote-clause coverage across e63dc112; mint01smoke*/mint01smoke2* cells are
quarantined in smoke_archive (pre-fix code paths) and must never join an analysis corpus;
the zero-derive decision rule verdict is final for THIS corpus — do not re-sweep thresholds
post hoc to find a passing variant.
