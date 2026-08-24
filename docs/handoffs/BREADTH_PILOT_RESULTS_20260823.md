# Breadth pilot A/B results — 2026-08-23

Cycle 1 (LOAD-BEARING cycle) re-run of the graph-vs-seq_react gap-closing plan, this time on
tasks authored specifically to be BREADTH-shaped (genuinely independent-arms fan-out then
merge) — the shape the graph engine's scheduler was designed for, and a shape core24 barely
contains (see `docs/handoffs/GRAPH_VS_SEQREACT_GAP_INVESTIGATION_2026-08-22.md`).

## Setup

- Local, $0, `badmodel-ollama`, `qwen2.5:7b`, `good_adaptive` arm, `--axis e1_dedup_local`
  (reused purely for its ladder table; `--tasks`/`--variant` overridden).
- Tasks: 152 (7-way fan-out argmax), 153 (5-way fan-out argmin), 154 (2-arm comparison), 155
  (2-arm comparison), 156 (7-item count/filter), 157 (7-item count/filter) — all Schema v2,
  genuinely independent leaves (empty `depends_on`), WebFetch-verified keystones.
- Run-ids: `breadth_pilot_v2_20260823_graph` (`langgraph_react`) and
  `breadth_pilot_v2_20260823_seqreact` (`sequential_react`), 3 reps/task = 18 cells/engine, 36
  total.

## Grounding verification

**Pre-check (mandatory, before any full run)**: one smoke cell (task 152, `langgraph_react`,
3 reps — the axis bakes reps=3 in) was run first.
`grep -c "Setup failed\|Serper health probe failed"` → **0** across all 3 smoke cell logs. All
3 logs show `Serper search API OPERATIONAL`. The rep1 deliverable cited real resolved URLs (e.g.
`https://en.wikipedia.org/wiki/Vinson_Massif`) with the correct keystone answer (Vinson Massif,
1966). Pre-check passed; full run launched.

**Full-run audit (mandatory, after completion)**:
```
grep -c "Setup failed\|Serper health probe failed" \
  agent/idea_test_results/_breadth_pilot_v2_20260823_graph/cell_logs/*.log     → 0 across all 18 logs
grep -c "Setup failed\|Serper health probe failed" \
  agent/idea_test_results/_breadth_pilot_v2_20260823_seqreact/cell_logs/*.log  → 0 across all 18 logs
```
All 36 cell logs (18+18) show `Serper search API OPERATIONAL`. This is a clean, fully-grounded
run — no repeat of the 2026-08-23-early dead-Serper-key failure mode anywhere in the matrix.

## Paired results

`scripts/analyze_breadth_pilot_v2_20260823.py`, paired on `(task_id, rep)`:

- **Paired n = 17** (of 18 matched keys; 0 missing score, 1 infra-failed cell excluded —
  `breadth_pilot_v2_20260823_seqreact` task 154 rep3).
- **SCORE mean delta (graph − seq_react): −0.266** (`langgraph_react` mean 0.462 vs
  `sequential_react` mean 0.728), sd=0.367, t=−2.99, df=16, **p≈0.009 — significant LOSS for
  the graph engine.** W/T/L = 3/2/12 (graph unfavorable, and this reverses the sign of the
  core24-subset re-run's +0.142 graph advantage).
- **PROMPT TOKENS mean delta: −23,470** (graph 42,160 vs seq_react 65,630), t=−2.52,
  p≈0.023 — graph also used far *fewer* tokens, but that's because it stopped early, not
  because it was more efficient (see root cause below).
- **TOTAL TOKENS mean delta: −24,169**, t=−2.62, p≈0.019, same pattern.

### Per-task breakdown (3 reps each)

| Task | graph mean | seq_react mean | delta | W/T/L |
|---|---|---|---|---|
| 152 (7-way argmax, mountains) | 0.262 | 0.821 | −0.560 | 0/0/3 |
| 153 (5-way argmin, canals) | 0.500 | 0.767 | −0.267 | 1/0/2 |
| 154 (2-arm, dam height) | 0.750 | 0.500 | +0.250 | 1/1/0 |
| 155 (2-arm, wingspan) | 0.958 | 0.958 | +0.000 | 1/1/1 |
| 156 (7-item count, dams) | 0.071 | 0.437 | −0.365 | 0/0/3 |
| 157 (7-item count, bridges) | 0.325 | 0.810 | −0.484 | 0/0/3 |

The graph engine only wins/ties on the two 2-arm comparison tasks (154, 155); it loses
decisively on every 5-7-way fan-out task (152, 153, 156, 157).

## Root cause (not a measurement artifact)

Visit counts confirm this is real, not scoring noise:
- `seq_react` consistently visits 5-9 pages per 7-item task (near-complete coverage): e.g. task
  157 visits were 7/6/7 across reps.
- `langgraph_react` mostly stalls far short of the 7 leaves it planned: task 156 visits were
  1/1/4; task 152 visits were 0/0/7 (one rep visited nothing at all).

Two distinct failure modes observed directly in deliverables:
1. **Under-decomposition / premature termination**: task 156 rep1's deliverable is cut off
   mid-plan after a single visit — *"The height of the Vajont Dam is 262 meters... Next, I will
   search for the Katse Dam's Wikipedia page."* — with no further content. The graph engine
   fanned out to 7 leaves but the control loop stopped after 1.
2. **Wrong grounding / fabricated citations**: task 152 rep1 has **visit count = 0** yet the
   deliverable lists all 7 mountains with real-looking Wikipedia URLs (`https://en.wikipedia.
   org/wiki/Mount_Erebus`) and a wrong final answer (Mount Erebus, 1985 — hallucinated; correct
   keystone is Vinson Massif, 1966). The URLs are real domains but were never fetched — this is
   parametric-memory recall dressed as grounded citation, not a leaf-id echo.

This is the *opposite* of the parallelism-compensates-for-a-weak-model hypothesis: on genuinely
independent multi-arm fan-out, qwen2.5:7b's graph engine under-executes the plan it authored
(stops early or skips visits entirely), while `sequential_react`'s simpler one-step-at-a-time
loop reliably grinds through all N leaves.

## GO/NO-GO verdict

**NO-GO.** The graph engine does not hold or grow its advantage on this fair breadth
population — it takes a large, significant LOSS (−0.266, p≈0.009), driven by systematic
under-execution of its own fan-out plans on qwen2.5:7b, not scoring noise or grounding failure.
Do **not** proceed to Cycle 4 (suite-wide reframe of comparison/survivor/count/
conflicting-source task families toward independent-arms-then-merge validators) as scoped. The
parallelism-compensates-for-a-weak-model hypothesis is retired for this model/executor
combination. Redirect effort toward the narrower, already-partially-evidenced
correctness-under-conflicting-sources niche (`race_value_agreement`,
`project_strong_agent_trace_guidance` memory entry) instead — and, if graph-engine breadth
work is revisited later, first fix the control-loop's fan-out completion/early-exit gate (it
appears to treat "planned N leaves" and "executed N leaves" as decoupled), not the task suite.
