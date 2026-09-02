# Methodology: improving modules, KPIs, and measurement

Every rule here exists because its absence produced a wrong answer in this project, and the
failure is named next to the rule. Nothing below is general advice.

The motivating record, from 2026-09-01/02 alone: **three headline findings dissolved under
checking** (an evidence-retention gap that was answer-length confounding; a "4x lower
unsupported-claim rate" that inverted on its own holdout; "averted fabrications" that were mostly
our own parser failing), and **two metrics turned out to be measuring our own instrumentation**
rather than the system. On this suite, checking does more work than building.

---

# 1. Improving modules

## 1.1 The lifecycle — no step is optional

```
build  ->  attach to >=2 hosts  ->  verify on a REAL cell  ->  measure adoption  ->  make structural
```

**Attach to at least two structurally different hosts.** A module that works only in LangGraph may
merely suit LangGraph's tool API. `derive` binds to a tool-calling API and to a plain string
dispatch loop with no tool API at all; that is what makes "component" a claim rather than a
description.

**Verify on a real cell before believing any offline test.** The `derive` module passed 6 unit
tests, wired correctly, persisted its artifact — and on the first live cell the model **never
called it**. A green suite tells you the module works, never that it runs. This project's dominant
failure mode is "green tests, layer never executed": the tool-call shim sat default-ON for weeks
with zero stored cells that exercised it.

**Measure adoption before measuring effect.** `derive` was used in 5/12 and 4/12 cells. On the
rest the module delivered nothing, and any average over all 12 dilutes the effect by the adoption
rate. Report adoption as a first-class number, not a footnote.

## 1.2 The adoption ladder — prefer benefit that needs no cooperation

| rung | mechanism | coverage | cost |
|---|---|---|---|
| 1 | optional tool the model may call | measured 33-42% | none |
| 2 | prompt pressure / nudge | unmeasured | prompt bloat, trajectory perturbation |
| 3 | policy: refuse `finish` unless the claim came through the module | ~100% of finishes | can block a correct answer |
| 4 | post-hoc audit of stored artifacts | 100%, retroactive | cannot change behaviour, only report |

**Design rule: prefer rungs 3 and 4.** A module whose benefit depends on the model choosing to use
it inherits the model's weakness, which is precisely the weakness we are trying to compensate for.
Rung 4 is why the arm-blind auditor survived scrutiny when comparative KPIs did not — it needs no
cooperation from anything.

## 1.3 Every refusal rule needs an over-refusal test

Refusing is the module's whole value AND its dominant failure mode, and the two are
indistinguishable in a count. The page read `1,470\nm`; the model wrote `"1,470 metres"`; the
module refused a value the model had genuinely read. Those refusals were being reported as averted
fabrications.

For any rule that declines something, a test must assert that a legitimate case in a different
surface form still passes. When reporting refusals, split them:

- **averted** — the value appears on no page the host read
- **over-refusal** — the value is present in another surface form (unit spelling, digit grouping)
- **mis-attributed** — present on a *different* page the host read

The split is the finding. On the extraction gate, 95.9% were genuine and 4.1% mis-attributed —
which is what told us "retry against the other pages" would recover almost nothing.

## 1.4 Standing constraints

- **Compose, never reimplement.** Arithmetic, unit logic and value location live in
  `evidence_graph`. A module is an attachment surface.
- **Default OFF, own env flag, own `run_id`.** The cfg hash covers only
  `variant_specific_settings` (`idea_test_runner.py:1706`), so two env-flag conditions under one
  `run_id` silently overwrite each other.
- **Absent is never zero.** Module off must write no key at all, so a rate over that cell reads
  UNKNOWN. Writing `{}` would let "we never checked" read as "we checked and found none".
- **Ground only in what the model could read.** Register the truncated window shown to the model,
  never the fuller fetched text — otherwise a number recalled from memory that happens to sit past
  the truncation is credited as read.

---

# 2. Improving KPI results

## 2.1 Improvement versus capture

**A change that improves a KPI by construction is not an improvement.** Refusing to record
unsupported extractions improves an unsupported-claim rate mechanically; it is hiding evidence
unless something else also gets better. The extraction value gate fired exactly as designed — 42
of 115 extractions excluded — and is not shipped, because the KPI it targets moved the wrong way
and accuracy was flat.

**The paired criterion, declared before the run:** the KPI improves **and** the independent
accuracy anchor does not fall. Either half alone is not a result.

## 2.2 Raise the floor, not the mean

The durable wins this project has produced are all changes in what is *possible*, not in an
average:

- fabricated-arithmetic rate: UNKNOWN -> measurable
- selective accuracy at max confidence: 0.000 -> a working classifier
- a model LangGraph cannot run at all -> runs and scores
- a run that cannot be replayed -> replays with 0 drift

None needed a small mean difference resolved, which is why they survived the holdout that killed
the mean-based headline. **Prefer a KPI target whose success is categorical.**

## 2.3 Freeze, guard, amend in the open

Definitions are frozen before any build work, hashed by a test, and changed only by an amendment
that records what changed and why. This is not ceremony: the `recomputable` support class was
caught at a **25.3% cross-cell coincidence floor against a 5% ceiling** — higher than its genuine
detection rate — *before* it reported a number.

**Never report a metric whose negative control has not been run.** For any matching or search
metric, run it against deliberately mismatched inputs and publish that floor beside the result.

## 2.4 The holdout is not optional and is opened once

Seal a split by a rule that reads no outcome data (ours: highest task id per mechanism cluster).
Write the tuning analysis down and commit it, then open the holdout once. It inverted a 4x
headline. A KPI gain that does not replicate is reported as not replicating.

---

# 3. Improving measurements

| rule | the failure that produced it |
|---|---|
| **Set `LLM_SEED` on every local A/B.** Verified byte-identical reruns. | An unseeded 16-task A/B showed −0.054 that was pure sampling noise and had to be discarded. |
| **Seeding does not remove trajectory chaos.** | One mechanism flip left 12/16 tasks byte-identical and swung 2 by ~0.6. A mean over that is not a ranking. |
| **Report the movement distribution, never a bare mean:** how many cases moved at all, and their magnitudes. | The −0.0025 headline was an average of mostly zeros and two large cancelling swings. |
| **Distinguish "used the module" from "was offered it".** | LangGraph cells that used `derive` moved −0.072; cells that never called it moved −0.077. The cost was prompt perturbation, not the module. |
| **Absent is never zero; UNKNOWN is a value.** | The live-fallback gate reported UNKNOWN for a whole campaign rather than a fabricated pass — and became real only once provenance persisted. |
| **One `run_id` per condition.** | The cfg hash ignores env flags; conditions would overwrite each other silently. |
| **Denominators come from the preregistration, not the filesystem.** | A dead cell writes no file; one arm silently lost 6-7 of 48 and had its mean computed over survivors. |
| **No arm or mechanism ranking below ~60 paired tasks.** | Per-arm KPI orderings INVERT between a 16-task and a 6-task split of the same suite. |

## 3.1 The instrumentation self-check

Before reporting any metric, ask: **could this number be measuring my own code rather than the
system?** Two of this session's metrics were, and both looked like findings:

- "0% redundant reads" for one arm — its pages are deduplicated at write time.
- "61/176 averted fabrications" — mostly a unit-spelling parse failure, inflated further by a
  per-page rejection scan.

Concrete checks: compute the same quantity from a second, independent field; verify a
suspiciously good or bad number against the raw text by hand; and when two of your own metrics
disagree by an order of magnitude, reconcile them before publishing either. A `stated-claim
precision` of 0.040 sat next to an auditor's 0.224 unsupported rate; the 0.040 was a metric that
could not fire, testing exact string equality between `"6,300 km (3,900 mi)"` and `"1.16"`.

## 3.2 Delegation

Subagents in this project are strong at implementation and uneven at inference. One inverted a
task-range attribution so its docstring contradicted its own evidence; one concluded the suite was
flaky when it was reading another lane's half-written files. **Check every number an agent
reports.** Brief the concurrency hazard explicitly — a failure outside your own file manifest
means re-run, never "pre-existing flake" — and the agent briefed that way diagnosed it correctly.

---

# 4. Alternatives worth taking seriously

The binding constraint is measured: **~60 paired tasks before any comparative claim**, and the
suite has 22. Each alternative below attacks that constraint from a different side. Ranked by
value per unit of work.

## 4.1 Procedurally generated derivation questions over the frozen corpus — RECOMMENDED

Generate the *question* mechanically from values that are demonstrably present in real harvested
pages, rather than authoring tasks by hand.

**Feasibility, measured:** the 313-document frozen corpus holds **2,744 distinct (value, unit)
pairs**, 102 documents carry >=4 distinct numeric values, and there are ~99,000 within-document
value pairs. Ground truth is known by construction, leakage is impossible (the question did not
exist before generation), pages retain real-world messiness, and the cost is $0.

This converts n from a task-authoring budget into a compute budget, and it is the only option that
makes the comparative claims measurable at all. Risk to control: generated questions may be
systematically easier than authored ones — calibrate against the existing 22 before trusting a
number, and keep the authored suite as the harder reference set.

## 4.2 Counterfactual replay at a fixed prefix

`scripts/replay_call.py` already reconstructs one LLM call from a trace and re-issues it, which is
the seam for changing one thing *without* re-running the whole trajectory. This attacks trajectory
chaos directly: fix the prefix, vary the module, compare completions. Requires
`IDEA_TEST_CAPTURE_LLM_IO=1` and `IDEA_TEST_KEEP_TRACES=1`, both default off. The honest limit is
in the script's own docstring and should be read before relying on it.

## 4.3 Sub-problem harness — test the module without the agent

Extract derivation sub-problems from stored runs (operand pairs, units, expected results) and
exercise the module directly, with no loop, no model and no trajectory. n in the hundreds
immediately, at $0. This would have caught the unit-spelling over-refusal in minutes rather than
after two full campaigns.

**Rule of thumb: if a module's behaviour can be tested without the agent, test it there first.**

## 4.4 Fault injection for categorical outcomes

Deliberately serve a page with the needed value removed and measure whether the host fabricates or
refuses. The outcome is categorical, so small n suffices, and it targets the property we actually
care about instead of a mean score. This is the cheapest way to make "trustworthy under pressure"
measurable.

## 4.5 The weak-model ladder

Effects on availability are ~100x larger than any KPI difference that failed to replicate:
`gemma2:2b` goes from an instant HTTP 400 to a scoring run. Three of ten local models have no
tool-calling endpoint. Within-model, host-vs-host+module comparisons across a 0.5B-14B ladder ask
the project's actual thesis — does structure help weak models more? — in a form small n can answer.

## 4.6 Reconsider the target

Two of three measured axes say the arms are equal or unmeasurable; the third is enormous and
robust. The defensible claims are **availability** (runs models the host cannot run at all) and
**auditability** (the only configuration whose arithmetic can be checked by anyone, ever). Neither
needs a comparison to survive — which is exactly why they held when the comparative headline did
not.
