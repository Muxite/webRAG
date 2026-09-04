# Pre-registration AMENDMENT: mechanical answer-audit minting (mint01)

Amends LEDGER_RISK_COVERAGE_PREREG_2026-09-04.md. Written before any mint01 cell runs.
$0 campaign: SEARCH_PROVIDER=corpus (numeric22), local ollama (badmodel-ollama, gpu-lock),
seeded (LLM_SEED set), verbosity 3 (mandate + traces retained).

## What changed since the original prereg
1. Quote capture fixed at e63dc112 — all mint01 SOURCE nodes carry verifiable quotes; the
   5-clause chain gets its first fair coverage measurement (never compare quote-clause
   coverage to pre-fix corpora).
2. NEW mechanical answer-audit minting (host-side, model-invisible; LEDGER_HOST_MODULES token
   "answer_audit"): finish-time backing/derivation of answer numbers against the quantity
   index; minted nodes tagged minted_by="answer_audit" and EXCLUDED from the 5-clause chain
   (clause semantics stable across the minting boundary).
3. NEW signal under test: answer_supported (graded: backed/derived/unbacked, unit-consistency,
   trivial-number exclusion, ambiguity<=1) — defined for ALL cells including zero-derive.

## Design
- run_ids: mint01_l3b, mint01_q05, mint01_q15, mint01_q7b (one per model; prereg.audit is model-blind)
- models: llama3.2:3b, qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:7b (same tier as gateprobe01 + one
  mid model for contrast)
- hosts/arms: sequential_react + langgraph_react, single arm per host:
  LEDGER_HOST_MODULES=derive,answer_audit (answer_audit is model-invisible, so an audit-off
  arm would be execution-identical under a fixed seed — measured, not duplicated)
- tasks: dev split only (210,211,212,214,215,216,218,219,220); holdout 213/217/221 excluded
  from all headline numbers, reserved for a single confirmatory pass
- reps: 2, seeded
- denominator: 9 tasks x 2 hosts x 2 reps = 36 cells per run_id, 144 total; a missing cell is a failure
- primary endpoint: risk (wrong-rate among accepted, wrong = score<0.5) vs coverage of
  (a) the unchanged 5-clause certify (first post-quote-fix measurement) and
  (b) answer_supported on the ZERO-DERIVE stratum (the 89% population the signal exists for)
- secondary: graded sweeps (backed-only vs backed+derived; trivial in/out; ambiguity),
  op_appropriateness informational counts, certified-but-wrong inversion lists by filename
- abort conditions: infra_failed rate > 0.20; budget_usd 0 (corpus replay;
  LEDGER_MAX_LIVE_FALLBACKS default cap applies); driver.lock singleton
- analysis: scripts/ledger_risk_coverage.py --prefix mint01, code frozen and committed before launch; specs in agent/idea_test_results/prereg/mint01_*.json

## Decision rules (fixed now)
- answer_supported "useful": on the zero-derive stratum, risk at its natural operating point
  <= 0.5x the stratum base wrong-rate with coverage >= 0.15, and zero-or-explained
  supported-but-wrong inversions. Else: honest negative, mechanism ships default-OFF-equivalent
  (informational telemetry only).
- 5-clause certify coverage expected to rise from 0% (quote starvation) — if it stays 0%,
  trace the next starved clause before any redesign.
