# Cycle closeout: promptbench, executed sam-style (2026-08-18/19)

Two things ran at once, on deliberately disjoint resources: the **webRAG** work
(build and run `promptbench`) and a **sam tier-3 demo** (drive the build units
through `../sam`'s contract → orchestration → escalation → ledger machinery, no
rental). A third deliverable fell out of the second: written feedback on sam.

**Spend: $0.0685 webRAG API (of $5) + $0.34 sam burst (of $3). Suite 4808 → 4920.**

---

## 1. webRAG — what was learned

Full detail in `PROMPTBENCH_RESULTS_2026-08-19.md`; pre-registration in
`PROMPTBENCH_PREREGISTRATION_2026-08-18.md`, committed before any row was read.

**The engine's universal answer-before-justification convention is not the best
shape at any measured point, and it is the most expensive way to be no better.**
Asking for the answer alone is indistinguishable from answer-then-justification
(4 of 5 models favour answer-only, sign p = 0.375) at **16.3x fewer completion
tokens**. Reason-then-answer beats answer-then-justify on **5 of 5** models with
a usable baseline — a perfectly consistent direction across a 14x parameter
range and two providers — but the aggregate sign test lands at p = 0.062, so it
is a direction, not an established result. The engine's *literal shipped verify
prompt* is significantly worse than even the plain answer-first shape on
`qwen2.5:7b` (−0.211, p = 0.008).

Eight per-call-site recommendations are tabulated in the results doc. **No
shipped prompt was changed**, by design: a micro-eval win does not transfer to
task score for free.

**Three measurement artifacts were caught before they became findings**, and
that is the durable methodological outcome:

1. The grader tokenised on `\w+`, so multi-word answers could never match —
   inflating "parse failure" to 79–95% on every prose arm of one family while
   leaving JSON arms untouched. Caught by an *asymmetry* (0% parse failure on
   the two-short-option family, same model, same arm), not by inspection.
2. The SHIPPED arm was graded against the wrong vocabulary — models emitted
   well-formed `{"verdict": "FALSE"}` and scored 92% parse failure.
3. Six cells were **degenerate** — constant answers scoring exactly 0.500 on a
   balanced set, including one model's *A1 baseline*, which invalidated every
   delta measured against it. Only detectable because the family is balanced.

The pre-registration predicted failure mode #1 in the abstract ("a shape-blind
grader would fabricate the very effect the benchmark is testing") and it
happened anyway. Writing the warning down is what made it recognisable in the
data an hour later.

**A plan-invalidating discovery:** `link_select` and `extract_value` are **not
buildable from this repo**. `agent/idea_test_results/` (3.3 GB) stores only
aggregates — `observability.visit` is `{"count": 1, "chars": 26142}`. No page
text, no URLs, no link sets; the telemetry sidecars hold a ~200-char completion
prefix with no prompt. The families were rebuilt from `agent/app/idea_tests/`
task modules instead, which turned out to be a **better** source: committed
rather than gitignored, exact, hand-authored, and small enough to freeze into a
92 KB fixture. This also removes the "the corpus is 3.3 GB and gitignored"
obstacle from the deferred general component bench.

## 2. sam — did it save effort?

**Yes, on this shape of work, and the model was never the bottleneck.**

| unit | outcome | tier | tools | wall | USD |
|---|---|---|---|---|---|
| `pb-availability-001` | `solved` | Senior (Gemini 3.7 Flash) | 7 | 24 s | $0.0304 |
| `pb-grade-transport-002` | `solved` | Senior | 14 | 71 s | $0.0800 |
| `pb-report-003` | `solved_after_repair` | Senior | 19 | 298 s | $0.1255 |

Three for three, first tier tried, no escalation, ~7 minutes of model time for
~600 lines that passed their acceptance tests unchanged after transplant.

**But:** writing the acceptance test and seeding the workspace took me
substantially longer than the chain took to satisfy them, and **cross-repo
dispatch does not exist** (`graph.py:541` hardcodes
`source_repo = contract.task_dir / "public"`; `contract.repo` is parsed and never
read), so every unit needed a hand-seeded tree and a hand-transplanted diff.
sam's leverage here is bounded by how fast a human can author a
machine-checkable contract, not by tier or model.

**The sharpest lesson, and it generalises beyond sam:** `constraints:` and
`invariants:` are rendered into the prompt and enforced by nothing. Every
constraint not *also* encoded in the acceptance test was quietly skipped —
`Item` came back mutable where the contract said frozen; `transport.py` shipped
no HTTP client where the contract said it might. **Only the test binds.** That
is tier-1 doctrine working exactly as written, and it is easy to forget while
writing prose that looks authoritative.

Two real sam bugs were found and fixed (see §3).

## 3. sam feedback delivered

`../sam/docs/DOGFOOD_FEEDBACK_webRAG_2026-08-18.md`. Two bugs fixed in place:

- **`diff()` could not see created files.** Bare `git diff` reports tracked
  modifications only, so any task whose solution *creates* a file produced an
  empty diff and empty changed-file list — silently making `max_files_changed` /
  `max_diff_lines` budgets vacuous, making `allowed_paths` / `must_not_change`
  unable to detect a violation, and making `fingerprint_diff("")` constant so
  `is_duplicate_attempt` flagged every new-file repair as a duplicate. Live on
  sam's own `remote-gpu-status-001`. Fixed with `git add -A -N`. The fix is
  self-demonstrating: applying it flipped a run from `solved` to
  `api_escalation` because the scope check finally *ran*.
- **A solved run left no recoverable work** — workspace destroyed, diff nowhere.
  Fixed with 6 lines in `node_finish`.

Plus: the disposable workspace needs a `.gitignore` (running the acceptance test
creates `__pycache__`, which then violates `must_not_change: tests/`); contract
YAML errors surface as raw PyYAML tracebacks; the cost preview under-estimated
by 12x; the classifier never chose the Intern tier. **Biggest token saving
available:** prompt:completion ran 51–55:1 with only a 32% cache hit rate —
74,000 prompt tokens to write a 102-line file — so stabilising the prefix and
marking cache breakpoints is worth roughly 3x on the dominant cost term.

## 4. Next cycle

**Stage 1 — act on this cycle's result.** Two changes, both flagged and default
OFF, both A/B'd end-to-end on the mixed-shape tasks:
(a) drop the `reason` field at the short-answer sites whose reason is not
consumed (16.3x token saving, no measured accuracy cost);
(b) move `reasoning` before `verdict` in `VerifyLeafAction._DEFAULT_SYSTEM_PROMPT`
— the highest-value single edit, given SHIPPED is −0.211 (p = 0.008) on the
strongest local model. **The transfer test is the point**; treat a null result
as informative.

**Stage 1b — the confidence-calibration family.** The anti-calibration in
`CONFIDENCE_JUDGE_MISCALIBRATION.md` motivated this cycle and **this bench
cannot speak to it**: it measures discrete accuracy, the inversion concerns a
continuous score against run outcomes. A calibration family (predicted
confidence vs realised correctness, Brier / ECE) is a small, cheap addition to
`promptbench` and would close the loop the cycle opened.

**Stage 2 — ship the chrome-filter link fix.** Unchanged and still the highest
value-per-line on the shelf: 0% → 51.7% containment at zero token cost, 74.1% at
k=35 for +276 tokens against an ~88,000-token budget. `actions.py:1172` /
`expansion.py:791`, flag it, default OFF, guard fan-out.

**Stage 3 — the general component bench.** `promptbench`'s `availability.py`,
item envelope and materialised label pass carry over directly, and the corpus
problem is now *solved differently*: build from task modules, not from run
results. Reduces the estimate materially. Two functions are deleted rather than
ported (`measure_dataflow_slots.classify_outcome`,
`replay_chain_failures.print_unintended_behaviors`), and striking the unsupported
"0% false-positive rate" claim from `dataflow.py`'s docstring goes with them.

**Stage 4 — measure the mixed shape** (054/085/055/061/146/147/149/122). Still
the largest hole in the project's evidence and the only place the DAG has a
structural advantage to earn back.

**Stage 5 — the grounding gate** (`grounding.py:105-107` checks that *a* visit
happened, not the right one; 13 of 19 audited wins confabulated). Its own cycle,
and repairing it will make scores fall — plan that communication first.

**On sam:** cross-repo dispatch (honour `contract.repo`) is worth more than any
model-tier or orchestration change, on this cycle's evidence. It would delete
the seed-and-transplant step that dominated the sam-side cost and widen the set
of tasks sam can be pointed at beyond greenfield modules.

**Also carried, cheap:** amend Q12 to state its arm (+0.051, concentrated in
`baseline`); the free defects §5.2 / §5.4 / §5.5 / §5.7; commit
`scripts/rescore_results.py`.
