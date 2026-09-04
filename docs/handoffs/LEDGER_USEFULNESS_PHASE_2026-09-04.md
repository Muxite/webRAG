# Ledger usefulness phase — results and handoff (2026-09-04)

Plan of record: `~/.claude/plans/plan-the-next-phase-soft-graham.md`. Prereg:
`LEDGER_RISK_COVERAGE_PREREG_2026-09-04.md` (committed before the analysis existed). All
analysis $0; live spend this phase: **$0.81** (paid API ladder, cap was $4.50) + one $0 local
probe. Merge: `b1438e18` on `dagv2-evidence-ledger`.

## What this phase claimed and what it found

The phase replaced two dead questions (cross-model derive adoption; module mean-score delta —
both established unresolvable) with one: **does the ledger's audit signal tell a consumer which
answers to trust?** (risk-coverage discrimination, the KPI spec's own L1).

### 1. Headline (prereg outcome 3): the certify signal is starved at quote capture

Full pre-registered certify signal on the completed ladder03 corpus (dev derive-ON, n=123
local / n=159 with the paid stratum): **0% coverage**. Sole blocking clause: `quote_verified`
null on ~100% of stored SOURCE nodes (null is fail-closed per prereg).

### 2. But the machinery under it discriminates

Minus-quote sensitivity variant (pre-registered): 20/159 cells certified at **10% risk vs a
54.7% base wrong-rate** (~5.5x error reduction, 12.6% coverage). Zero certified-but-wrong
cells on the whole prereg corpus. The derive + arithmetic-valid + operand-supported +
no-unbacked-numbers chain carries real signal where it exists.

### 3. Quote capture had a one-file mechanical root cause — fixed

`LedgerToolkit._locate` never passed `quote=` to `add_source` → `QUOTE_FAIL_EMPTY` always
(`QUOTE_CAPTURE_GAP_2026-09-04.md`). Fixed in `e63dc112`: line-expanded verbatim page-text
slice, guaranteed to verify, both hosts inherit. **Live-confirmed** on the gateprobe01 cells:
8/8 SOURCE nodes `quote_verified: true`. Stored pre-fix corpora stay null forever — never
compare quote-clause coverage across the fix boundary.

### 4. The structural finish gate does NOT fix the tiny-model adoption cliff — negative result

`GATE_PROBE_2026-09-04.md` (36 cells, seeded, $0): the gate (`final_require_derivation_for_numeric`,
default OFF, shipped `6a1b75b8` with all four precheck refinements) **never induced a single
new derive call** on qwen2.5:0.5b/1.5b or llama3.2:3b. 9/18 ON cells ended `forced_by_budget`
(step budget exhausts before the corrective cycle), 0 reached a productive retry→derive→pass
path, and the 4 cells where retries fired reproduced the `require_finish_tool` cost blowup
(+543%/+820% tokens for ≤+0.25 score). The abstention banner is cosmetic (never changes the
committed number). **Verdict: keep default OFF; do not iterate on finish-time coercion for
sub-3B models.** 89% of probe cells minted zero ledger rows at all — the bottleneck is
upstream at row minting, not at finish.

### 5. First certified-but-wrong cells, immediately after the quote fix (n=2, off-prereg)

gateprobe01 task 212: qwen2.5:1.5b certified a **-30.55 km** "height difference" (wrong
operand order); llama3.2:3b certified a *ratio* (2.66, dimensionless) where a difference was
asked. Arithmetic valid, operands supported, quotes verified — all green. The chain's residual
blind spot is **operation/operand appropriateness for the task**, exactly as predicted when
`operand_supported` was designed (it never fired False on any corpus this phase — it needs
adversarial cases; its discrimination power remains unmeasured).

### 6. Honest revisions of this phase's own earlier numbers

- Gate-precheck lift revised **down** on the real sequential corpus: 1.0–1.4x (holdout exactly
  1.00x). The early "2.67x" was an n=2 artifact — retired, never cite it
  (`GATE_PRECISION_PRECHECK_2026-09-04.md` addendum).
- `operand_supported`'s dev-split numbers are development evidence (circular); holdout n=11
  all-True says nothing yet.

## Shipped (merge b1438e18)

quotient/ratio `unit_note` annotation; `extract_unit` scale-word fix ("25 million" ≠ unit
"million"); `operand_supported` axis + `reverify_graph` recomputation; FinishGate (default
OFF) + telemetry; `scripts/ledger_risk_coverage.py` (now takes `--prefix`, default ladder03 =
the prereg corpus; fails loudly on zero files); quote capture at source minting;
`_resolve_llm_api_key` quote-stripping. Suite: 9093+ passed; the 14 known environmental
failures (4 bench_shim, 10 corpus/data) predate the phase.

## Next phase: highest-value moves, in order

1. **Mechanical row minting.** 89% of tiny-model cells mint zero rows; the host already builds
   a quantity index per visited page. Auto-mint SOURCE nodes from the index (zero model
   compliance required) and re-measure certify coverage. This attacks the real bottleneck the
   gate probe exposed and follows the same philosophy as the quote fix (deterministic > asking
   the model).
2. **Operation-appropriateness checks** for the n=2 blind spot: deterministic candidates —
   sign/plausibility for magnitude questions, task-shape vs operation match; or the
   {full/partial/conflict/abstain} taxonomy from Claim-Selective Certification
   (arxiv 2605.21949). Author adversarial wrong-operand fixtures so `operand_supported` and
   any new check have something to fail on.
3. **Re-run the risk-coverage analysis on the first post-fix corpus** (needs a fresh campaign;
   all stored corpora predate quote capture). Prereg an amendment first.
4. Deferred debts unchanged: seeded gpu0831 parity rerun; evidence_loop gate graduation
   (probe evidence now argues against ever graduating finish-coercion for tiny models);
   second-hostname visit policy.

Trap list for the next session: never cite quote-clause coverage across the e63dc112 boundary;
gateprobe cells are off-prereg (probe tasks/conditions); `ledger_risk_coverage.py --prefix`
defaults to the prereg corpus deliberately; holdout 213/217/221 stays uncited.
