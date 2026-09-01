# Ledger KPI specification — FROZEN 2026-09-01

Status: **frozen**. These definitions were written before any build work in the phase they
measure, so that a later result cannot be produced by moving a definition. Changing any formula
below requires a dated amendment section at the bottom of this file naming what changed and why;
a silent edit invalidates every number reported against it.

Operative plan: `docs/LEDGER_PLAN_2026-09-01.md` (§3 metric order — calibration > verifiability >
cost > mean score). Phase plan: level the field across arms, then build to win on these metrics.

## Reading rules that apply to every KPI

1. **The accuracy anchor is always printed.** No KPI is ever reported alone. Every table carries
   `validation.overall_score` for the same cells, so a KPI gain bought with real accuracy is
   visible in the same glance. This is `trust_kpi_dashboard.py`'s existing rule, generalised.
2. **UNKNOWN is a value, absent is never zero.** A cell that cannot produce a KPI reports UNKNOWN
   with a reason string. A KPI is never imputed as 0, and a group's denominator never silently
   shrinks to the cells that happened to work — the trap `prereg.py` was built to close, where
   `langgraph_react` lost 6-7 of 48 cells and had its mean computed over the survivors.
3. **Denominators come from the preregistration, not the filesystem.** `prereg.expected_cells()`
   supplies the count. A missing cell is a failure, not an absence.
4. **Arms are measured by arm-blind code.** Every KPI below is computed from fields that all three
   arms emit, by `agent/app/testing/evidence_audit.py`, which never reads `execution_variant`,
   `tooling_profile`, or any arm-exclusive field. Where a KPI genuinely cannot be arm-symmetric,
   it is marked ARM-LIMITED and the limitation is printed with the number.
5. **Holdout is reported separately.** See the split at the end of this file.
6. **One condition, one `run_id`.** Verified in code: `idea_test_runner.py:1706` computes the
   filename's `cfg` hash from `variant_specific_settings` only. An arm knob read directly from
   `os.environ` (every `IDEA_TEST_*` / `LEDGER_*` flag this phase introduces) never reaches that
   dict, so two A/B conditions launched under one `run_id` produce identical filenames and the
   second silently overwrites the first. Every condition therefore gets its own `run_id`. This is
   an experiment-design rule, not a preference.
7. **The accuracy anchor is genuinely independent.** All 22 numeric-suite modules return `None`
   from `get_llm_validation_function()`, and `validation.llm_validation` is null in 198 of 198
   stored cells — `overall_score` is produced entirely by deterministic grep validators, with no
   model in the loop. It cannot be moved by any change to a prompt, a verdict rule, or a KPI.

---

## L1 — Risk-coverage (calibration)

**Question.** When this arm says it is confident, is it right more often?

**Formula.** For confidence tiers ordered ABSTAIN < PARTIAL < ANSWER, the mean
`validation.overall_score` within each tier, plus the selective risk-coverage curve produced by
`scripts/risk_coverage.py`. The arm is **monotone** if mean score is non-decreasing across the
tier order. Monotonicity is the headline; the curve is the detail.

**Source.** Two series, always reported side by side, neither replacing the other:

- `native` — `output.confidence`, the arm's own channel (added this phase).
- `audit` — the arm-blind derived tier, computed by the offline auditor under the L4 support
  rule (which admits `recomputable`, so it does not inherit the structural defect below).

**Structural defect in the pre-phase derived rule, recorded so the baseline is readable.** The
old `arm_verdict.derive_verdict` rule marks ANSWER only when *every* checkable claim matches a
page literally. Because the suite is leak-proofed, a derived keystone never appears on a page, so
that rule cannot credit a correct arithmetic answer. Measured: 28 of 29 ANSWER cells come from
the non-arithmetic tasks 222-231, and `langgraph_react`'s ANSWER tier scores 0.362 against its
PARTIAL tier's 0.712. The baseline inversion below is therefore partly an artifact of the rule,
not purely a property of the arms — a replacement rule must fix the support definition (L4), not
merely soften a threshold.

**Denominator.** All non-`infra_failed` cells in the group. A cell with no `output.confidence`
is UNKNOWN, never folded into a tier.

**Direction.** Monotone increasing is the win. A tier inversion at the most confident tier is the
specific failure this KPI exists to catch — both derived-verdict arms currently show it
(0.591 -> 0.273 and 0.576 -> 0.231 on `ledgernum22r3`).

**Baseline (`ledgernum22r3`, threshold 0.5).** evidence_loop native ledger verdict
0.439 / 0.453 / 0.600 (monotone); langgraph_react derived 0.591 / 0.629 / 0.273 (inverted);
sequential_react_extract derived 0.576 / 0.585 / 0.231 (inverted).

## L2 — Claim-level precision and recall per pipeline edge

**Question.** Where in retrieved -> extracted -> verified -> stated does the arm lose claims?

**Formula.** The four-stage decomposition already implemented in `scripts/claim_metrics.py`
(`pipeline_edges`, `aggregate_pipeline_edges`), following RAGChecker's decomposition but **not**
its LLM judge — every edge is a mechanically-verified field:

- `retrieved` — a page the arm fetched (`output.pages`)
- `extracted` — a typed record read off one of those pages
- `verified` — an extracted record whose VALUE was mechanically located on the page it cites
- `stated` — a verified value the final answer actually asserts

**Source.** `output.pages`, `output.extractions[].value_verified`, `extract_final_text`.

**ARM-LIMITED.** The `retrieved -> extracted` edge requires typed extraction records, which only
`evidence_loop` and `sequential_react_extract` emit. For `langgraph_react` that single edge is
UNKNOWN; the `retrieved` and `stated` ends are arm-symmetric and are reported for all three.

**Direction.** Higher retention at every edge. A large drop at one edge localises the defect
instead of leaving a score gap to be guessed at.

## L3 — Fabricated-arithmetic rate

**Question.** How often does the arm assert a computed number that no valid derivation produced?

**Formula.** Of the evidence graph's DERIVED nodes, the fraction with
`derivation_valid is False` — the model's proposed value disagreed with the Python recomputation.
A derived node's stored value is ALWAYS the recomputation; a disagreeing proposal is recorded and
marks the node invalid rather than being silently overwritten.

**Source.** `output.evidence_graph.nodes[] where kind == "derived"`.

**ARM-LIMITED.** Only `evidence_loop` builds a derivation graph. For the other arms this is
UNKNOWN — and that is itself the finding: an arm that performs arithmetic without recording the
operands cannot have this rate computed at all, by anyone, ever.

**Direction.** Lower. **Baseline: 0.0** across 57 derived nodes on `ledgernum22r3` (0 invalid).

**Do not confuse with `kpi_dashboard.k4_fabrication`**, which is a different, cruder cell-level
signal (`visit.count == 0` AND `success` AND non-empty deliverable). Both are reported; they are
never blended.

## L4 — Unsupported-claim rate

**Question.** What share of the arm's checkable claims have no evidence span behind them?

**Formula.** `1 - (supported claims / checkable claims)`. A claim is a standalone numeric token
of >= 2 digits or a double-quoted span (`arm_verdict._claims`). A claim is **supported** if it is
either:

- `on_page` — it appears in the text of a page the arm stored, located by the offset-aware,
  magnitude-safe `evidence_graph.verify_value` rather than a naive substring test; or
- `recomputable` — it is reproduced, within the suite's own relative tolerance, by applying one
  whitelisted operation to operands that are themselves `on_page`.

Anything else is `unsupported`. A pair that would combine but fails the dimension check is
recorded separately as `refused_unit_mismatch` — a refusal is a correct outcome on this suite,
never an unsupported claim.

**Why `recomputable` is not optional.** Measured on `ledgernum22r3`: under a pure on-page rule,
28 of 29 ANSWER-tier cells come from the non-arithmetic tasks 222-231, and `langgraph_react`'s
ANSWER tier scores 0.362 against its own PARTIAL tier's 0.712. The suite's anti-leak design
guarantees a *derived* keystone never appears verbatim on any page, so a literal-match rule can
never credit a correct arithmetic answer and instead selects for arms that produced no derived
number. A metric with that property measures our own leak-proofing, not the arm.

**Whitelisted operations (frozen):** `+`, `-`, `|a-b|`, `*`, `/`, `a/(a+b)` — the six the suite's
tasks use. **Tolerance (frozen):** relative `2e-2`, the suite's own `VALUE_TOL`. **No unit or
currency conversion, ever** (`LEDGER_PLAN` §7): operands must be dimensionless or share a unit
under `evidence_graph.extract_unit`.

**Mandatory false-positive floor.** A recomputability search over a candidate set can match by
coincidence. Every report of L2/L3/L4 must also print the cross-cell control: run each cell's
claims against a *different, task-mismatched* cell's pages and report the resulting spurious
`recomputable` rate. If that floor exceeds 0.05, tighten the tolerance or shrink the operation
set before the number is reported as a finding.

**Source.** `extract_final_text` and `output.pages[].text` — present on all three arms, so this
KPI is fully arm-symmetric.

**Denominator.** Checkable claims. A cell whose final text contains no checkable claim reports
UNKNOWN, not 0 — an answer with nothing checkable in it has not earned a perfect score.

**Direction.** Lower. **Baseline (`trust_kpi_dashboard`, extraction-based):** evidence_loop 47.5%
over 387 checked claims; sequential_react_extract 39.5% over 446; langgraph_react uncomputable.
The evidence-first arm is currently WORSE on its own signature metric. Moving this is the
phase's primary target.

## L5 — Replay fidelity

**Question.** What fraction of a recorded run can be re-verified at $0 from what was stored?

**Formula.** Two independent components, reported separately, never averaged together:

- **quote replay** — of the stored extractions, the fraction whose quote re-verifies against the
  stored page (`execution_evidence_loop.reverify_cell`). Misses on a page marked `truncated` are
  counted `unverifiable`, never `absent`.
- **derivation replay** — of the stored derived nodes, the fraction that recompute identically
  and whose source pages have not drifted (`evidence_graph.reverify_graph`, `page_drift` count).

A third component, **corpus-service fidelity**, applies only to runs recorded after
`88a57429`: the fraction of a cell's searches served by the frozen corpus rather than a live
fallback (`timings[name == "search"].payload.search_provenance`).

**Source.** `scripts/reverify.py` per cell; `telemetry_raw.timings` for provenance.

**Status on `ledgernum22r3`: UNKNOWN, campaign-wide.** Provenance landed at 05:47 and the
campaign ran at 05:08, so all 198 cells carry `search_provenance == None`. This component is
reported UNKNOWN for the frozen data and becomes computable only on re-runs. Absent is not zero.

**Direction.** Higher. A run that cannot be replayed is a run whose conclusions rest on trust.

## L6 — Cost and wall clock

**Question.** What did the quality cost?

**Formula.** Per cell: `observability.llm.total_tokens`, `observability.llm.calls`,
`execution.duration_seconds`, and `observability.cost.usd` where priced. For local models
`cost.usd` is null, so tokens and wall clock are the currency — `kpi_dashboard.py` already falls
back this way.

**Direction.** Lower, and **first-class**, not a tiebreak. The operative plan's §3 is explicit: a
3x wall-clock premium for a -0.100 score delta is not a rounding error.

**Baseline (`ledgernum22r3`, per cell).** evidence_loop 52.6s; sequential_react_extract 39.4s;
langgraph_react 16.9s. evidence_loop carries +47,259 tokens and +27.9 LLM calls over
langgraph_react.

## L7 — Quote-fabrication rate (NEW this phase)

**Question.** How often does the arm attach an invented sentence to a real number?

**Formula.** Of SOURCE evidence nodes with `verified is True` (the value WAS located on the
page), the fraction with `quote_verified is False` and `quote_fail_reason == "absent"`.

**Source.** `output.evidence_graph.nodes[]`, and arm-symmetrically via
`execution_evidence_loop.verify_against_stored_page` over `output.pages` for arms that emit
quotes without a graph.

**Denominator.** Value-verified source nodes only. Nodes whose value was never located are
excluded — they belong to L4, not here. `quote_fail_reason` of `no_page` or `empty` is
UNKNOWN, not a fabrication.

**Validity note, checked before freezing.** This rate is not a truncation artifact.
`store_page` records `chars` as the length of the WHOLE fetched text and
`verify_against_stored_page` downgrades a miss on a truncated page to `no_page` (unverifiable)
rather than `absent`. On `ledgernum22r3` the model's visible window and the stored window are the
same 6000 characters, so an `absent` quote was absent from what the model actually read.

**Direction.** Lower. **Baseline: 38.7%** — 96 of 248 value-verified source nodes on
`ledgernum22r3`.

## L8 — Redundant-read waste (NEW this phase)

**Question.** How much of what the arm read, it had already read?

**Formula.** Two views, both reported:

- **visit view (PRIMARY, arm-symmetric)** — `1 - (distinct normalized URLs / total visit
  events)`, over `telemetry_raw.timings` entries with `name == "visit"`, normalised through
  `coverage_report.normalize_url`. This is the only view valid across arms.
- **character view (evidence_loop / sequential_react_extract ONLY)** — the share of stored page
  characters belonging to a page whose `content_hash` was already seen in the same cell. It
  converts the defect into the currency of the token gap, but see the trap below.

**Trap, verified in code.** The character view MUST NOT be computed for `langgraph_react`.
`execution_langgraph.pages_from_telemetry` skips a URL already in its `seen` set, so that arm's
`output.pages` is deduplicated *at write time*. Computing redundancy from `output.pages` reports
0% for `langgraph_react` — a persistence artifact, not behaviour. Its true repeat rate is 14.0%.
Any redundancy figure sourced from `output.pages` for that arm is invalid.

**Source.** `telemetry_raw.timings` (primary) and `output.pages` (character view, two arms only).

**Direction.** Lower. **Baseline (visit view, `ledgernum22r3`):** evidence_loop 29.2%
(264 visits / 187 distinct), langgraph_react 14.0% (193 / 166), sequential_react_extract 36.8%
(296 / 187).

---

## The accuracy anchor

`validation.overall_score` (and `pass_rate`) are **reporting metrics, not targets**, and are
printed beside every KPI above. Baseline on `ledgernum22r3`: evidence_loop 0.538,
langgraph_react 0.612, sequential_react_extract 0.535.

No arm ranking may be claimed from 22 paired tasks. The pairing unit is the TASK, so reps do not
raise n; the repo's power table asks 61-111 paired observations for a 0.10 effect.

---

## Holdout split — FROZEN, deterministic, outcome-blind

The 22 numeric-suite tasks fall into six mechanism clusters, readable from `test_metadata.category`
before any result exists:

| cluster | tasks |
|---|---|
| sum / difference over same-unit operands | 210, 211, 212, 213 |
| ratio / quotient | 214, 215, 216, 217 |
| computed-ratio argmax | 218, 219, 220, 221 |
| unit-mismatch refusal | 222, 223, 224 |
| missing-operand abstention | 225, 226, 227 |
| plausible-but-unsupported numeric | 228, 229, 230, 231 |

**Rule: the highest task id in each cluster is held out.**

- **HOLDOUT (n=6, sealed during tuning):** 213, 217, 221, 224, 227, 231
- **TUNING (n=16):** 210, 211, 212, 214, 215, 216, 218, 219, 220, 222, 223, 225, 226, 228, 229, 230

The rule reads only cluster metadata, never a score, so the split cannot be steered by looking at
results — and it puts one task from every mechanism on the far side of the wall, which is what
generalisation means on this suite. Balance is therefore an *observation*, not a selection
criterion: holdout mean cross-arm spread 0.288 and mean score 0.548, against tuning 0.313 and
0.560.

Holdout cells are still executed (the grid stays 22 x arms x reps), but no KPI on holdout may be
inspected while any rule or threshold is being tuned. Reporting scripts take
`--split dev|holdout|all`, default `dev`, and printing `holdout` emits a banner naming this
file's hash.

**What the holdout is for, stated honestly.** 22 tasks is already under-powered for arm ranking,
and splitting costs power the phase does not have. The holdout's job is to **detect a
KPI/accuracy divergence** — a KPI gain on the tuning split that does not appear on the holdout is
reported as not replicated. It is not a second experiment and confers no ranking authority.

## Amendments

### 2026-09-01 — v1.0.0 to v1.1.0: operands must appear in the answer

**Changed.** `support.operands_must_appear_in_answer: true` added. A claim is `recomputable` only
when the operands that reproduce it are themselves quantities the answer states, not merely
numbers present somewhere on a visited page. Also recorded:
`support.measured_cross_cell_false_positive_floor: 0.0188`.

**Why.** The spec's own mandatory control caught the original definition before it reported a
number. Measured on the tuning split (140 cells, 2,601 cross-cell trials): with operands drawn
from every number on every page, the coincidence floor was **0.2526** — five times the 0.05
ceiling, and *higher* than the genuine recomputable rate of 0.145. A pair search over 24 operands
and 6 operations fires roughly 3,300 candidates at a 2% tolerance window, so it reproduces almost
any target by luck. The class was measuring its own arithmetic luck.

**Alternatives measured and rejected.** Requiring unit-bearing operands alone: floor 0.128, still
over ceiling. Tightening the tolerance to 1e-3: floor 0.0008 but genuine detection collapses to
0.068, rejecting the legitimate rounding the suite's own `VALUE_TOL` of 2e-2 exists to allow.
Requiring stated operands *and* unit-bearing: floor 0.0038 with genuine 0.263 — a better ratio,
but it discards about a quarter of genuine detections for a floor already well inside the
ceiling.

**Chosen.** Stated operands only. Measured with the shipped implementation: floor **0.0188**,
inside the 0.05 ceiling. The rationale is also substantive rather than merely statistical — a real
derivation is reported together with its inputs.

The standalone comparison that selected this option reported 0.0138 for the same rule. The
shipped figure is higher because the implementation gives `on_page` priority before testing
recomputability, so the two are measuring slightly different denominators. **0.0188 is the
reportable number**, because it is the one the code that produces the KPI actually yields; the
comparison's figures are retained above only for the relative ordering that drove the choice.

**Effect on prior numbers.** None published. The original definition never produced a reported
figure; the control ran before any support number was released.
