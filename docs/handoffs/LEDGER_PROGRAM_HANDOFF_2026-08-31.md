# Handoff — Euglena Ledger program (lanes A-H), 2026-08-31

**Provenance.** HEAD at write time is the tip of `dagv2-evidence-ledger` after 14 commits off
baseline `a3c53421` (`git log --oneline a3c53421..HEAD`). Model context: qwen2.5:7b local Ollama,
$0 inference throughout except a ~$0.11 Serper corpus build. Suite: **8304 passed, 18 skipped,
0 failed** (baseline `a3c53421`: 7840 passed, 18 skipped). Plans this executes:
`~/.claude/plans/5-3-in-sequence-validated-alpaca.md` (the program) and
`~/.claude/plans/the-program-is-written-witty-moth.md` (the execution log — read that for full
per-lane derivations; this document is the publishable synthesis). Product doc: `docs/LEDGER.md`.

## What shipped

| Lane | Piece | Where | Live? |
|---|---|---|---|
| A | Derivation layer wired into `evidence_loop`: typed `derive` action, `DerivationError` subclasses, verdict gate | `agent/app/testing/execution_evidence_loop.py`, `evidence_graph.py` | **yes** — 21/22 numeric-suite cells carry a graph artifact |
| A (fix) | Unit-spelling bug that made the graph admit 0 nodes in production despite green tests | `evidence_graph._candidates` | **yes** — same cell went 0 nodes -> 2 SOURCE + 1 DERIVED |
| B | 22-task numeric suite (210-231): sum/diff, ratio, argmin/argmax, unit-refusal, missing-operand, fabrication-bait | `agent/app/idea_tests/test_21*..23*_*.py` | **yes**, registered in `TEST_PRIORITY_ORDER`, corpus built live |
| C | Closed the verify-leaf `optional_url` leak on the 6 remaining tasks (056, 066, 128, 129, 131, 133) | `agent/app/idea_tests/test_{056,066,128,129,131,133}_*.py` | **yes** |
| C | `[LEAK]` lint severity, un-reintroducible | `scripts/validator_lint.py`, `agent/tests/validator_lint_test.py` | **yes** |
| D | Call identity (`call_id`, not file order), `stage`/`node_id`, sampling params, `seed`, retry `attempts`, trace reader | `connector_base.py`, `connector_llm.py`, `llm_backends.py`, `trace_recorder.py`, `scripts/trace_read.py` | **yes** — confirmed with a retained trace JSONL actually holding `prompt_text`/`completion_text` |
| E | Counterfactual replay + offline reverify CLI | `scripts/replay_call.py`, `scripts/reverify.py` | **yes**, provider-limited (see Retraction 2 area / honest limits below) |
| F | Arm-symmetric derived verdict, risk-coverage, claim metrics | `scripts/risk_coverage.py`, `scripts/claim_metrics.py` | **yes**, run over the archive and over the fresh Lane G cells |
| F (fix) | `output["pages"]` fallback for arms that never persisted `result["graph"]` | `agent/app/idea_test_utils.py` | **yes** |
| G | `execution_langgraph.py` evidence-persistence fix (`pages_from_telemetry`) | `agent/app/testing/execution_langgraph.py` | **yes** — `langgraph_react` went from 0/966 archived cells with recoverable evidence to a working risk-coverage curve |
| G | `build_corpus.py` robustness: one failed search no longer discards a whole paid harvest | `scripts/build_corpus.py` | **yes** — found by a live run that actually failed this way |
| G | Fresh 3-arm run `ledgernum22`, preregistered, evidence-persisting | `scripts/prereg.py` (first real gate), corpus replay | **yes** — 66/66 cells, 100% |
| H | This document + `docs/LEDGER.md` updates | `docs/**` | doc only |

Subsystems 1 (typed action queue) and 2 (DAG evidence-dependency analysis) remain **explicitly
out of scope**. Their starting points, unchanged from the program document:

- `agent/app/testing/execution_evidence_queue.py` — its own docstring calls it "a HARNESS STUB…
  there is no scheduler at all — the 'queue' is a for-loop over a list", built so one function
  body can be swapped; `idea_engine.py:2148-2360` has the real `asyncio.Semaphore` + `gather`
  batch, coupled to `IdeaDag` and `max_total_nodes`.
- `IdeaDag` has no edge objects (`parent_id`/`children` id lists only); readiness ignores
  predecessors entirely (`idea_node_state.is_action_ready`); dependency inference is two narrow
  rules in `idea_sequencing.detect_state_dependencies`.
- **Known open bug, not fixed here:** the always-in-path `requires_data` backfill writer sets
  `source_node_id` from ancestors of the node's *parent*, while the condition it feeds checks
  membership in the parent's *children* — so that writer's output can never satisfy the check
  that reads it.
- **Known open scheduler bug, not fixed here:** the AND-join does not fire when a sibling stalls
  at `search`; synthesis proceeds on partial evidence reporting `success: true, warning: None`
  (task 156 rep1, three search branches, merge node `skipped`).

Also open, not fixed here: `sequential_react`'s evidence-persistence gap (fixed for
`langgraph_react`, not this arm — same class of bug, same fix shape, unapplied); whether the
derived verdict is a working selective classifier for non-ledger arms (see Lane G results
below); whether tasks 210-231 should be promoted into `ACTIVE_SUITE_IDS` (deliberately deferred
by user decision — it changes the benchmark denominator).

## The finding that justified the whole live-check detour

Lane A passed every offline test and produced a correct artifact in-process. Run against a real
cell (task 130, qwen2.5:7b, `evidence_loop`, corpus replay, $0) it admitted **zero nodes and
recorded six rejections**. Green tests, zero executions — the exact failure mode this program
exists to catch.

**Root cause.** The source page said `"20,310 feet"`. The extractor reported
`value="20,310 ft"` with `unit="feet"` — right number, right page, unit correctly declared in
its own field, only the spelling inside `value` differing. `_candidates` built the value+unit
spelling only for *bare* values, so for an already-unit-bearing value it tried `"20,310 ft"`
alone and never `"20,310 feet"`. Result: `absent`, no node.

**Fix** (`evidence_graph._candidates`): when a value is unit-bearing and its embedded unit
differs from the declared `unit` field, also try the value's numeric half joined with the
model's own **declared** unit. Not a loosening of the no-fuzzy-matching rule — the spelling is
still built from a field the model itself supplied, still one exact boundary-checked span, still
unit-bearing (so it keeps the 12-55x decoy resistance unit-bearing matches carry over bare
ones). The alternative — falling back to the bare number — would have thrown that resistance
away. Re-run on the same cell: 2 verified SOURCE nodes + 1 DERIVED node, 1 correct rejection (a
metric figure genuinely absent from an imperial-only page), `reverify_graph` 2/2 with zero page
drift. Score moved 0.733 -> 0.867, n=1, not an effect claim — the finding is that the graph went
from admitting nothing to admitting re-verifiable evidence.

## Two of three arms never persisted the evidence they read

Measured against the stored archive (random samples of `*_r1.json` cells):

| arm | stored | sampled | visited | `documents_seen` non-empty | text-bearing | recoverable via `visited_evidence` |
|---|---|---|---|---|---|---|
| `sequential_react_extract` | 148 | 40 | 37/40 | 0/40 | 0 | 12/12 |
| `evidence_loop` | 283 | 40 | 34/40 | 0/40 | 0 | 7/12 |
| `langgraph_react` | 966 | 40 | 36/40 | 1/40 | **0** | **0/12** |
| `sequential_react` | 631 | 40 | 35/40 | 0/40 | **0** | **0/12** |

`slim_telemetry_raw` strips `documents_seen` for every arm by design. The only surviving
evidence text is `output["pages"]` (only `evidence_loop`/`sequential_react_extract`) or
`result["graph"]` action-results (`graph`/`naive_discretion`). `langgraph_react` and
`sequential_react` emitted neither — the archive recorded THAT they visited, never WHAT they
read, so a three-arm risk-coverage curve from history was structurally impossible.

Fixed for `langgraph_react` (`2cdc9066`): `pages_from_telemetry()` freezes every VISITED
document (`source="visit"`, full cleaned page text) into the same `store_page` shape
`evidence_loop` already uses. Search hits are deliberately **not** frozen — a result snippet is
not a page the agent read. `sequential_react` still has the gap; named here as open work, not
fixed.

## A paid-harvest robustness bug

The first `build_corpus.py --live` run for tasks 210-231 died on its first search (`Request
failed after 3 attempts`, status=None) and discarded the whole 22-task harvest already paid for.
The Serper key was verified working immediately before and after. Root cause: `live_harvest`
wraps the `visit` call ("one bad fetch must not abort the run") but not `query_search` —
asymmetric, and one transient failure anywhere in a ~110-search paid build threw away everything.
Fixed: a failed search skips that query and logs rather than aborting, and the attempt still
counts against `max_searches`, so a dead key still burns the ceiling down and stops instead of
retrying forever. This is exactly the defect class the earlier `core_long24` corpus build could
not have found — it happened to have no transient failure.

## The reconciliation this document owns: "breadth parity" vs. the aggregation-shape loss

`docs/LEDGER.md`'s retired-hypotheses section states breadth parity is established
(−0.002 / −0.007, retiring an earlier −0.266 "graph collapses on fan-out" result).
`docs/AGGREGATION_SHAPE_FINDING_2026-08-30.md` reports graph losing to `sequential_react` by
**−0.461 (t=−7.73, n=23)** on aggregation-shaped tasks, with a stated mechanism ("fan-out
multiplies work but not coverage"; branches have no sibling visibility,
`expansion.py:287-300`). Read together without qualification these look contradictory. They are
not — they measure different variables, and the evidence for saying so is on disk in the same
handoff that produced both numbers (`docs/handoffs/DAG_V3_S1_BREADTH_COLLAPSE_AND_GRADING_ASYMMETRY_2026-08-28.md`,
Findings 11-13; per-shape breakdown reproduced in `docs/handoffs/EVIDENCE_STACK_NIGHT_2026-08-31.md:145-163`).

**The −0.002/−0.007 dead tie is `breadth` as a literal fan-out-width variable** — the N=4→32
sweep (nested-prefix rosters, so per-item difficulty cannot drift with N) and a 6-task shape
bucket explicitly labeled `breadth`, run across two search backends.

**The −0.461 (t=−7.73) loss is `aggregation` as a task-shape variable** — a blind
source-only reclassification of all 59 `suite59` tasks into `aggregation`
(fan-out/count/argmax/AND-filter) vs `chain`, independent of how many branches any given task
happens to fan out into. In the finer-grained six-way shape split that also produced the
breadth number, `aggregation` and `breadth` are **two separate labels in the same table**:
`aggregation` shows −0.196 (Serper) / −0.140 (SearXNG), `breadth` shows −0.002 / −0.007. The
aggregation number was underpowered there (n=3-6/shape) and explicitly flagged "do not rebuild
this argument without n large enough per shape" — the blind 59-task reclassification (n=23) is
the number that cleared that bar, not a replacement finding that contradicts the smaller one.

**Resolution:** widening a fan-out (more branches, same task) is parity-tied with sequential
execution — the literal breadth axis. Whether the task's answer requires **combining** evidence
across those branches (a count, an argmax, an AND-filter — the aggregation axis) is a different
property, and on that axis graph loses badly, with a measured, code-level cause: expansion
context is root-ward only (`IdeaDag.path_to_root`), so siblings cannot see each other's queries,
URLs, or findings — the structural precondition for combining coverage across branches is
absent regardless of how many branches there are. Dose-response supports this reading:
Pearson(n_items, delta) = −0.491 corpus-wide (more distinct items the answer needs, larger the
deficit), which is a coverage/retention signature, not a branching-count signature. Both results
stand; `docs/LEDGER.md`'s wording has been corrected to name the axis each result actually
measures rather than using "breadth" for both.

**What would fully resolve it, if this reasoning is wrong:** run the literal `breadth`-labeled
tasks (or the N-sweep family) but change their grading so the correct answer requires combining
counts/values across the fanned-out branches, instead of retrieving any one of several
independently-sufficient facts. If width-controlled aggregation tasks still show the −0.461-scale
loss, shape is confirmed as the driver; if they instead show parity like the current breadth
set, then something about the *specific* 23 aggregation-shaped tasks (authoring era, item count,
etc.) rather than the aggregation property itself would need to be the explanation, and the
current reconciliation would need retracting. That experiment has not been run.

## Lane G RESULTS — `ledgernum22`, 66/66 cells, $0 inference

`prereg.py audit --run-id ledgernum22` → **66/66 cells (100.0%)**, denominator from the design,
not the filesystem. Zero infra failures, zero live fallbacks, `usd=0.0000` on every cell. Corpus:
313 documents built for **110 searches, ~$0.11**.

Per-arm means, reported for completeness, **not as a ranking**:

| arm | n | mean score | pages/cell | secs/cell |
|---|---|---|---|---|
| `evidence_loop` | 22 | 0.624 | 4.18 | 44.5 |
| `langgraph_react` | 22 | 0.602 | 2.23 | 18.1 |
| `sequential_react_extract` | 22 | 0.478 | 4.00 | 30.5 |

**Derivation graph:** 21/22 cells with an artifact, 19/22 admitting a SOURCE node, 8/22 emitting
a DERIVED node, 109 source + 16 derived nodes, **zero invalid derivations**, 78 rejections,
verdicts 13 ANSWER / 8 PARTIAL / 1 ABSTAIN. Fabricated-arithmetic rate **0.0 on n=8** — the first
time this rate has ever been computed at all.

**Risk-coverage (ALL / ANSWER_OR_PARTIAL / ANSWER_ONLY):**

| arm | ALL | ANSWER_OR_PARTIAL | ANSWER_ONLY (n) |
|---|---|---|---|
| `evidence_loop` | 0.545 | 0.571 | **1.000** (n=2) |
| `langgraph_react` | 0.636 | 0.700 | **0.200** (n=5) |
| `sequential_react_extract` | 0.500 | 0.524 | **0.250** (n=4) |

`langgraph_react` had 0.000 coverage at every ANSWER tier across 966 archived cells; it now has
a curve — structurally impossible before this session's persistence fix.

**Do not present this curve as a clean win.** `evidence_loop`'s NATIVE verdict is monotone
(0.545 -> 0.571 -> 1.000). Both arms using the DERIVED verdict score **lower** at ANSWER than at
ALL (0.200 vs 0.636; 0.250 vs 0.500) — the wrong direction. At n=4-5 that is far too small to
call an inversion and must not be reported as one, but it is flagged prominently as the first
thing a powered follow-up must test: it is the same shape as the roster-gate inversion this repo
already got burned by (blocked 46/48 eligible cells including two scoring 1.00, passed only
0.40-scoring cells).

### What MAY and MAY NOT be claimed from `ledgernum22`

**MAY:** the derivation layer admits real, re-verifiable evidence and emits recomputed values
with zero invalid derivations; a fabricated-arithmetic rate is computable and is 0.0 on n=8; all
three arms now persist auditable evidence; a three-arm risk-coverage curve exists;
`evidence_loop`'s native verdict is monotone against ground truth on this set; `evidence_loop`
costs ~2.5x `langgraph_react`'s wall clock (44.5s vs 18.1s per cell).

**MAY NOT:** any arm ranking. n=22 paired tasks at reps=1 against this repo's own power table
asking n=61-111 for a 0.10 effect — the prereg said so before the data existed and the data does
not change it. The mean-score column above is reported for completeness, not as a comparison.

**Claim metrics** (`evidence_loop`, `ledgernum22`): retrieved->extracted 0.902,
extracted->verified **0.743**, verified->stated 0.692, orphaned_extractions 0. For contrast, the
same function measured extracted->verified at 0.34 (129/2553 orphans) on the pre-fix archive —
different task set and corpus, so this is indicative of the unit fix's direction, **not** a
controlled before/after.

## Retraction ledger

1. **"233 traces, not one contains `prompt_text`; the full-capture path has never produced an
   artifact."** RETRACTED. `prompt_text` appears in 99 result JSONs including a run named
   `fullcapture_check_20260823`.
2. **"Prompt text never reaches the trace JSONL; the trace is the wrong file."** RETRACTED —
   this was the coordinator's own misdiagnosis mid-program, not a program-document claim.
   `connector_llm.py`'s `if _full_capture: in_payload["prompt_text"] = ...` is pre-existing, and
   `_record_io` routes through `_record_event` -> `TelemetrySession.record_event` ->
   `TraceRecorder.record`. The capture path always worked; no retained trace had ever been
   produced with `IDEA_TEST_CAPTURE_LLM_IO` turned on. Lane D's real contribution was `call_id`
   pairing (in/out were previously correlated by FILE ORDER while the in-flight semaphore allows
   32 concurrent calls), `stage`/`node_id`, sampling parameters, retry `attempts`, and
   `scripts/trace_read.py`.
3. **"`TEST_PRIORITY_ORDER` is duplicated across two files, both need updating."** RETRACTED.
   `idea_test_runner.py:324` is the live 121-id (now 143-id) list; `testing/config.py:46` is a
   stale 25-id list reachable only through an import that `idea_test_runner.py:1551` shadows with
   its own definition. Registration touches ONE file.
4. **Lane C's blast-radius reasoning** ("only `graph_compiled` exercises the leak") was wrong —
   `optional_url` is written by the live planner too (`expansion.py:1838`, `idea_engine.py:1450`),
   and 77 of 119 verify nodes across 334 stored graph-arm cells carry a URL. Its CONCLUSION (no
   historical number needs re-deriving) nevertheless holds, on better evidence: in every sampled
   cell the verify URL had ALSO been independently visited earlier in the same run, so the
   auto-fetch was redundant rather than free grounding.

## Verification

```
PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests
8304 passed, 18 skipped, 0 failed   (baseline before this program: 7840 passed, 18 skipped)
```

All work except the corpus build (~$0.11 Serper) and local Ollama inference for `ledgernum22`
was offline/$0. No paid inference occurred anywhere in this program
(`IDEA_TEST_USD_CEILING=0.01`, `LEDGER_MAX_LIVE_FALLBACKS=0` for the Lane G run; both held at
zero the whole run).

## Open work carried forward

- **`sequential_react` evidence persistence** — same gap `langgraph_react` had, same fix shape
  (`pages_from_telemetry`-style), not applied to this arm.
- **Derived-verdict ANSWER-tier inversion** — n=4-5, flagged not claimed; the first thing a
  powered follow-up must test (see Lane G results above).
- **Powered re-measurement** of `ledgernum22` — this repo's own power table asks n=61-111 for a
  0.10 effect; n=22 rep=1 settles nothing. Corpus replay makes reps nearly free in money, not in
  GPU wall clock — budget that explicitly before running it.
- **Task 210-231 promotion into `ACTIVE_SUITE_IDS`** — deliberately deferred by user decision; it
  changes the benchmark denominator and is a campaign decision, not a lint side effect.
- **Subsystems 1 and 2** (typed action queue, DAG evidence-dependency analysis) — out of scope
  this program; starting points and the two known bugs (`requires_data` backfill, AND-join
  stall) are recorded above, unchanged from the program document.
- **Aggregation-shape fix** — UNMEASURED: whether restoring sibling visibility across fan-out
  branches (shared-evidence context instead of root-ward-only) closes the −0.461 gap. Queued in
  `docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md` §6 as the `graph_shared_context` ablation, not
  built or run by this program.
