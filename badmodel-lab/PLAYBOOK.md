# Bad-Model Lab — Opus experiment playbook

You (Opus) are the **experimenter**. The bad local model is the **subject**. Your
job: find, for each subject, the cheapest mitigation stack that makes it
*agentic-feasible* on the benchmark — or prove it can't be. You do the part a
script can't: read *why* a model failed and choose the next thing to try.

One iteration = pick a cell → run it → read score + telemetry → diagnose →
decide the next cell. Repeat until every subject has cleared the bar or is proven
infeasible, then hand off to the reporting stage.

## The bar (define "feasible" before you start)

A `(model, mitigation)` is **agentic-feasible** if, at **R=3**, its **keystone
pass-rate ≥ 0.75** on the `micro` tier, within the `$5` ceiling. (0.75 is the
repo's own `overall_passed` line — same bar used in `cross_tier_analyze.py` /
`adaptive_ab_analyze.py`.) Latency is a secondary gate — flag anything over
~90 s/task. Confirm at R=3; never believe R=1 (house rule — weak signals lie).

## The knobs

- **Subjects / anchors:** `roster.yaml`
- **Mitigations (independent variable):** `profiles/*.env`
  - `m0_baseline` — react leaf (JSON path). The control; measures the floor + the
    parse-failure class mix.
  - `m1_thin` — thin leaf, **no JSON at all**. The biggest single lever.
  - `m2_thin_votes` — thin + self-consistency. Only for **extraction-bound** tasks.
  - `m3_react_bigtokens` — react + bigger budget. Only when telemetry says
    `truncated_json`.
  - `m4_thin_lean` — thin, fewer steps / tighter pages. For the weakest models.
- **Tiers (task difficulty):** `tiers.yaml` — `sanity → micro → reachable → hard`.

## The loop

```
1. Look at state:   ./.venv/bin/python badmodel-lab/analyze.py
2. Pick the next cell = a hypothesis. Start every new subject at:
      run_cell.sh <model> m0_baseline sanity 1     # does the pipeline work at all?
      run_cell.sh <model> m0_baseline micro 1      # baseline floor + telemetry
3. Run it. Then re-run analyze.py and READ THE TELEMETRY CLASS MIX.
4. DIAGNOSE from the dominant failure class -> pick the mitigation:
      prose / refusal        -> m1_thin      (stop asking it for JSON)
      fenced_json            -> m1_thin, or add a fence-strip repair to the react path
      truncated_json         -> m3_react_bigtokens
      malformed_json         -> m1_thin (or grammar-constrained decoding, deeper)
      valid_json but score 0 -> not a format problem: wrong entity/aggregation.
                                Check the deliverable; try m2 votes if extraction-bound,
                                or restate entity->value in the plan aggregation.
5. Run the chosen mitigation on `micro` at R=1 to explore; if promising, CONFIRM at R=3.
6. Log the finding (one line: model, profile, tier, score, ks%, why). Move to the
   next hypothesis or the next subject.
7. Escalate a feasible subject up the ladder: micro -> reachable. Expect `hard` to
   floor — that's fine, it's for the anchors + the merge.
```

## Merge step ("simpler tests on the better places")

Once subjects are mapped, run the **anchors** (`qwen2.5:7b`, `openai/gpt-4.1-nano`)
on `micro` + `reachable` to set the ceiling (they should score ~1). Then build the
merged recovery curve across tiers × model tiers with
`scripts/level_ladder.py` + `scripts/recovery_curve.py`, and the LinkedIn figures
per `CHART_SPEC.md`.

## Budget & throughput discipline

- Local subjects are **free**; keep them local. `$5` OpenRouter ceiling
  (`BADMODEL_USD_CEILING`, default 5) covers only anchor runs — those are tiny.
- Deterministic validators are free; **leave `IDEA_TEST_RUBRIC` off** while exploring
  (the LLM judge costs money).
- Keep task subsets small and use R=1→R=3 escalation. Don't run `hard` on bad models
  except to demonstrate the floor once.
- One model resident at a time (12 GB card). Run cells serially.

## Stop rules

- A subject is **done** when you've found its feasible mitigation (logged, R=3) OR
  shown every mitigation floors it on `micro` (that's a real, publishable result:
  "below ~Nb params, agentic recovery fails even with the thin scaffold").
- Don't keep throwing votes/candidates at a task — voting only helps extraction
  bottlenecks and hurts arithmetic (house rule).
