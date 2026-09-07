# State of evidence — 2026-09-07

**What this is.** One page that says what webRAG can currently claim, what it cannot, what was
retracted, and what comes next. It exists because the front door (README) and thirteen
handoffs disagreed about all four. Every number below has a source document; when a later
document contradicts an earlier one, the later one wins and the earlier one is named.

**Provenance.** Branch `dagv2-evidence-ledger` at `b3409571`, merged fast-forward into `master`
the same day this document was written. Latest campaign: mint02
(`docs/handoffs/MINT02_RESULTS_2026-09-05.md`). Operative product doc: `docs/LEDGER.md`.
Roadmap decision: `docs/handoffs/ROADMAP_2026-09-07.md`.

---

## 1. What the project is now

Four architecture generations, one live line:

| Generation | Dates | Status |
|---|---|---|
| DAG v1 | 2026-02 → 03 | closed |
| Compiled v1 | 2026-06 | closed, proven (README "Benchmark Results"); its result is a *strong planner + cheap executor* claim and must not be borrowed as support for DAG v2 |
| DAG v2 | 2026-07 → 08 | **closed by the pivot**; its barrage relaunch was never run and is retired |
| **Euglena Ledger** | 2026-08-31 → | **the live line**: an auditable, replayable evidence compiler; a component other agents call |

The Ledger's claim, stated as an engineering target: *at parity accuracy with a baseline doing
identical evidence-gathering, only this system knows when it does not know, and every step that
produced the answer can be replayed at $0.* Mean score is a guard, never the target.

---

## 2. Claims that are currently defensible

Each row: the claim, the strongest evidence, and the caveat that travels with it.

| Claim | Evidence | Caveat |
|---|---|---|
| Auditability became measurable where it was UNKNOWN before | `LEDGER_MODULE_EXPERIMENT.md` 2×2: fabricated-arithmetic rate and replay fidelity went from UNKNOWN (host alone) to measured (host + module) on two hosts (LangGraph tool-calling API and string-dispatch ReAct), qwen2.5:7b, seeded, $0 | Categorical, not a score claim. The module caught 1 real fabrication in 9 ReAct derivations |
| The certify chain rejects wrong answers at a useful rate once quotes are captured | mint02 (192 cells, 4 local models, 2 hosts, 12 tasks): 10.4% coverage at 6.7% risk vs 63.5% base wrong-rate; mint01 (144 cells): 4/144 certified at 0% risk vs 70.8% base | Coverage is low because 120/144 cells mint zero derivations. Pre-`e63dc112` corpora have ~100% quote nulls and cannot be compared across the fix |
| `answer_supported` is a clean signal where derivations exist | 18/18 accepted cells right across mint01+mint02 (0% risk); coverage 45.8% (dev, mint01) then 18.9% (prereg, mint02) | Honest negative against its own prereg bar (needed 30% coverage). Informational, not consumer-facing. Do not promote for zero-derive cells (0.72× risk, both prongs failed) |
| `backed_only` on computation tasks flags copying | mint01 7/7 wrong; mint02 12/15 (80% precision) | Missed the 90% bar; all 3 misses scored exactly 0.5 (operands gathered, never derived). Stays informational |
| Zero invalid derivations where the arithmetic is checked | 195/197 derived nodes corpus-wide valid; both invalid are quotient ops in `sequential_react` | `derivation_valid` checks arithmetic, **not operand choice**: `381 m − 25 million = 356 m` passed once (scale-word bug, fixed `b1438e18`), and the mint02 certified-wrong cell divided km by minutes with a decoy-backed operand |
| Frozen-corpus replay + preregistration + seeded A/B is a working methodology | Every campaign since 2026-09-01 is byte-reproducible at $0 (15/15 after `adeeffa5`); `scripts/prereg.py` gates denominators from design, not filesystem | `LLM_SEED` did not reach `langgraph_react` before `adeeffa5`; any null on that arm before 2026-09-03 is a weakened null (see §4) |
| Graph loses badly on aggregation-shaped tasks | `AGGREGATION_SHAPE_FINDING_2026-08-30.md`: −0.461 (t=−7.73, n=23), blind shape classification of 59 tasks, with a code-level mechanism (root-ward-only context, lossy merge, no extraction step) | Fan-out *width* is parity-tied (−0.002/−0.007); width and shape are different variables. The proposed fix (deterministic coverage × typed records, `EXTERNAL_REVIEW_2026-08-30.md`) has never been run |
| Tiny models fail upstream of the ledger | phi3:mini 7/12 visited but keystone 1/12; adoption of `derive` is a cliff: 0/9 on every model below 7b, 9/9 at qwen2.5:7b | Finish-time coercion does not fix this: 0 new derives, 9/18 `forced_by_budget`, +543–820% tokens (`LEDGER_USEFULNESS_PHASE_2026-09-04.md`) |
| Compiled v1 works | 1,026 live runs, gpt-5-mini compiled 0.896 @ $0.017 = premium ceiling @ 1/10 cost; CI-disjoint on hardest tier | Closed-out; a different hypothesis from everything after it |

---

## 3. Claims that must NOT be made

- **Any mean-score win for the Ledger.** `ledgernum22r3` (198 cells): evidence_loop −0.100 vs
  langgraph_react at ~3× wall clock, +47k tokens; nothing clears Holm. The master plan's
  "evidence-first executor improves grounded score" thesis is unsupported (`LEDGER_PLAN_2026-09-01.md` §1).
- **"evidence_loop is monotone."** Holds on one split, inverts on another
  (`LEDGER_FINAL01_TUNING_RESULT.md`). Split artifact at this n.
- **Any arm ranking from ≤22 paired tasks.** The pairing unit is the task; reps do not raise n;
  the power table asks 61–111 paired observations for a 0.10 effect. Arm orderings inverted
  between tuning and holdout on unsupported-claim rate (0.081→0.320 vs 0.327→0.063).
- **Any single-mechanism A/B at n≈16.** Seeded, one flag flipped: 12/16 tasks byte-identical,
  2 swing ~0.6. The noise floor eats the effect (`project_mechanism_ab_noise_floor`).
- **"Calibration cannot be retrofitted."** Retracted same day; an arm-blind post-hoc audit gives
  every arm a monotone channel (`LEDGER_KPI_PHASE_HANDOFF_2026-09-01.md` §3).
- **"Breadth parity established" as a standing fact.** n=6, per-task reliability ~0.11, unseeded
  arm. The *retirement* of "graph collapses on fan-out" (−0.266) is unaffected; the positive
  parity claim is downgraded to "not contradicted".
- **The "2.67×" gate-precheck lift.** n=2 artifact; the real sequential corpus gives 1.0–1.4×.
- **"79.5 searches/cell" or "4.00 vs 21.85 searches/cell".** `search.count` counted result
  documents at `search_k=6`; every pre-`0fa6e733` figure is 6× too high (~13 and ~0.67 vs ~3.64).
- **"Parallelism compensates for a weak model."** Never testable on the suite; not supported
  where tested. Retired.
- **LLM-judge-driven gating of any kind.** Merge AUC 0.288 (below chance); roster gate blocked
  46/48 including two 1.00 cells.

---

## 4. Measurement debts still open

| Debt | Why it matters | Cost |
|---|---|---|
| Seeded rerun of `gpu0831`/`gpu0831b` | The pivot's parity premise was measured on an unseeded `langgraph_react`; three handoffs call it highest priority, none did it | $0, GPU hours |
| L5 replay fidelity on a post-provenance corpus | UNKNOWN campaign-wide on `ledgernum22r3` (provenance landed after the run) | $0 |
| Second model size for the core numbers | Everything in `LEDGER_PLAN` is qwen2.5:7b; the ladder gives categorical endpoints only | $0, GPU hours |
| `stall_recovery_gate`, `require_finish_tool` nulls | Both on the unseeded arm; shelved on a weakened null | $0 |
| Tasks 210–231 promotion into `ACTIVE_SUITE_IDS` | Changes the benchmark denominator; a campaign decision, deliberately deferred | decision only |

---

## 5. Consolidated retraction ledger

Dated, with the document that recorded each. Read this before citing anything older.

| Date | Retracted claim | Where |
|---|---|---|
| 2026-08-15 | A thesis-supporting result published at n=6, retracted at n=10 (one task's keystone gate) | `CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md` |
| 2026-08-15 | "Benchmark passes" claim propagated across five files and a CI docstring for three weeks | `DAG_V2_PREFLIGHT_2026-08-15.md` §2 |
| 2026-08-16 | "19 cells confabulated their answer" | `CYCLE_CLOSEOUT_2026-08-16.md` |
| 2026-08-23 | Two same-day addenda measured with zero working searches (Serper key outage) | `GRAPH_VS_SEQREACT_GAP_INVESTIGATION_2026-08-22.md` |
| 2026-08-28 | Citation-format mitigation; Pearson(difficulty, delta)=+0.23 (Simpson artifact); the offline estimate | `DAG_V3_S1_BREADTH_COLLAPSE_AND_GRADING_ASYMMETRY_2026-08-28.md` |
| 2026-08-31 | "Task 047 is a broken task" (it is a capability floor); the shape-split hypothesis; seven absence-based claims in one night | `EVIDENCE_STACK_NIGHT_2026-08-31.md` |
| 2026-08-31 | "233 traces, none contain prompt_text"; "trace JSONL is the wrong file"; "TEST_PRIORITY_ORDER duplicated"; Lane C blast-radius reasoning | `LEDGER_PROGRAM_HANDOFF_2026-08-31.md` §Retraction ledger |
| 2026-09-01 | "reps=5 → n=110 crosses the power threshold"; "zero live fallbacks, stated as measured" | `LEDGER_HANDOFF_2026-09-01.md` §6 |
| 2026-09-01 | "Calibration cannot be retrofitted" | `LEDGER_KPI_PHASE_HANDOFF_2026-09-01.md` §3 |
| 2026-09-03 | All unseeded `langgraph_react` nulls downgraded; `search.count` figures ÷6 | `analysis/SEEDING_AUDIT_2026-09-03.md`, `WEAK_MODEL_LADDER_PHASE_2026-09-03.md` |
| 2026-09-04 | The "2.67×" gate-precheck lift | `LEDGER_USEFULNESS_PHASE_2026-09-04.md` |
| 2026-09-05 | mint01's 7/7 `backed_only` and 45.8% `answer_supported` did not replicate at prereg | `MINT02_RESULTS_2026-09-05.md` |

Recurring cause, recorded five times: **green tests, never executed.** A passing suite says
nothing about whether production code ran; verify every layer on a real cell.

---

## 6. The structural finding that sets the next phase

The Ledger's verifier is sound but starved. 120/144 mint01 cells and 89% of tiny-model cells
mint zero derived nodes, so the certify chain has nothing to certify. Every "missing mechanism"
in the component phase traced upstream to acquisition or row minting, never to the evidence
layer. Auditability that depends on the model choosing to be audited is not a guarantee.

Therefore the next phase mints rows mechanically from the host's per-page quantity index and
filters operands by provenance (one operand per named entity), which also closes all four
"wrong-entity vindication" cells and the certified-wrong laundering path observed in mint02.

---

## 7. Roadmap (decided 2026-09-07)

Full text in `docs/handoffs/ROADMAP_2026-09-07.md`. In order:

1. **Consolidate** (this document; README rewrite; merge to master). Done.
2. **Break the zero-derive wall**: mechanical row minting + operand-provenance filter +
   "computed result must be present" for clause 5. Prereg'd, seeded, $0.
3. **Make the component real**: fix `ledger_api.py`'s two API defects (supplied sources
   truncated to 6,000 chars; `clean_operation` reshapes text before offsets), freeze the
   `LedgerResult` contract, package, ship one consumer outside this harness.
4. **Decide on the aggregation-shape 2×2** (deterministic coverage × typed records) only after
   step 2 reports whether coverage moved (target ≥25% at ≤5% risk).

**Retired, do not re-open:** DAG v2 barrage relaunch and `max_burn` ladder; Phase B typed
action queue; finish-time coercion for sub-3B models; contradiction relaxation
(`_is_wildcard`); unit or currency conversion; `Interval` wiring (1 real case corpus-wide);
v3 continuable chat and codebench (parked, not on any live line); underpowered n≈16 mechanism
reruns.
