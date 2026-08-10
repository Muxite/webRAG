# Agent failure-mode map (2026-08-10)

Point-in-time analysis, not a living doc (repo's spec convention). Produced by a read-only, $0
pass over existing benchmark data — no live LLM calls. Goal: answer "where do webRAG's agents
perform worst, and where do they perform best?" against the project's central thesis (structure +
memory should boost cheap/weak-model performance on long-running agentic tasks — see
`[[project_central_thesis_cheap_model_performance]]`).

Sources: `codebench/results/runs.jsonl` (135 rows) + `submission/*.py` under
`codebench/results/runs/*/`, `badmodel-lab/results/cells.jsonl` + `cells_long.csv`,
`badmodel-lab/MODEL_TIER_LIST.md`, the full `agent/idea_test_results/` corpus (894 files) via
`scripts/level_ladder.py` and `scripts/unified_bench_report.py`.

**Operational hazard found and already resolved**: `badmodel-lab/analyze.py` and
`codebench/analyze_code.py` are *regenerators*, not pure readers — running `analyze.py` silently
rewrote the committed `badmodel-lab/results/cells_long.csv` from 622 rows to 181 (several raw
result JSONs it joins against have since rotated off disk locally). Caught via `git diff --stat`,
reverted with `git checkout -- badmodel-lab/results/cells_long.csv`, verified clean. Flag this for
anyone running `analyze.py` "just to look."

## Worst-performing combinations

**1. codebench: `qwen2.5:14b` × `badmodel` (graph_compiled_code JSON-action scaffold) on hard
coding tasks — mean 0.244 vs aider's 0.623 (n=63 each, Δ=-0.379).** The largest, cleanest gap in
the dataset, and a protocol bug, not a reasoning gap. Compiling every `submission/*.py` with
Python's own parser: **33/82 badmodel files (40%) are outright `SyntaxError`s** vs **0/62 for
aider**. Two dominant, reproducible sub-patterns, both traceable to `write_file(path, content)`
embedding a whole source file as one JSON string field
(`agent/app/idea_policies/extra_actions/sandbox_tools.py`,
`agent/app/testing/execution_compiled_code.py`):
- **Triple-quote miscounting**: `"""docstring"""` collapses to `""docstring""` — e.g.
  `codebench/results/runs/calibrate_c33/c33__badmodel__qwen2.5_14b/submission/shortest_paths.py`,
  `coordinator_batch1/c30__badmodel__qwen2.5_14b/submission/line_diff.py`.
- **Double-escaped newlines**: the whole file lands as one line with literal `\n` characters
  instead of real newlines — e.g.
  `coordinator_batch3/c44__badmodel__qwen2.5_14b/submission/report.py`,
  `coordinator_batch3/c50__badmodel__qwen2.5_14b/submission/mercury_orbit.py`.
- **65% of badmodel rows (41/63) never reach a gradeable test** (`tests_total==0`); **29 of those
  41 hit exactly the leaf's 10-action step-budget cap** (`.codebench_run_summary.json`:
  `"outcome": "step budget exhausted after 10 steps; last successful action: read_file"`) without
  ever calling `finish`.
- Per-task deltas: c30 (-0.93), c35 (-0.91), c36 (-0.88), c42 (-0.82), c22 (-0.81), c40 (-0.72).
- **Actionable**: the write protocol (one-shot full-file JSON string) is a worse interface than
  aider's diff/search-replace loop for this model tier, independent of raw coding competence.

**2. Native-adaptive engine at the hardest task-complexity tier.** Via `scripts/level_ladder.py`
across the whole corpus (n=274 native vs n=219 `graph_compiled`): native = **0.198** vs
`graph_compiled` = **0.725** at the "graph" level (Δ=+0.528, Cohen's d=**1.77**, significant); at
"integration" level 0.227 vs 0.813 (d=**1.95**, significant). Confirmed on one apples-to-apples
case: `bmladapt__qwen2.5-14b__a0_native_baseline__reachable` (n=21, same task set
`MODEL_TIER_LIST.md` reports at 0.97 for `m1_thin`) scores **0.196**. Raw JSONs
(`agent/idea_test_results/bmladapt__qwen2.5-14b__a0_native_baseline__reachable_0{62,70,76}_*_r1.json`)
show **visit-starvation**: `visits: 2`-`4`, coverage/citation checks failing, while the model
still emits a fluent, confident, wrong answer (e.g. wrong-peak prominence figure for task 062 —
same failure class already documented for gemini-2.5-flash-lite on this task in
`MODEL_TIER_LIST.md`, now reproduced independently on a different model+engine). Caveat: the
qwen2.5:14b `m1_thin`=0.97 comparison point's backing JSONs have rotated off disk locally — trusted
from the doc, matched only on task-id-set/n, not independently re-verified this pass.

**3. Native-adaptive engine is a near-total floor for the smallest local models; bolt-on
mitigations don't rescue it.** `llama3.2:1b`/`a0_native_baseline`/reachable = **0.000** (n=4);
`qwen2.5:0.5b` across `a0_native_baseline`/`a5_native_vote_k_tiered`/`a7_native_io_framing`/
`a8_native_io_framing_retry` all sit at **0.00-0.03** (n=4 each). Self-consistency voting and
I/O-framing retries — layered on top of the free-running native loop — don't fix it. The same
model on the compiled `m1_thin` profile jumps to **0.472** (n=6), >15x. `phi3:mini`'s vote profiles
score **0.067**/**0.000** (n=4 each) vs that model's `m1_thin` reachable of **0.973** — the gap is
*control-flow ownership*, not raw model capability.

**4. `google/gemini-3.1-pro-preview` × `graph_compiled` on the Stage0/D1 "hard mixed" battery —
mean 0.223 (n=31).** Read six raw deliverables directly; ruled out the two already-documented
Stage0 harness bugs (author max_tokens, `expansion.py` meta TypeError —
`[[project_capability_tiered_dag_stage0]]`). Outputs are coherent, not truncated/corrupted —
failures are substantive (incomplete fact-gathering, wrong figures, bad source selection e.g.
citing a `chegg.com/homework-help` page). **Lower confidence**: didn't run the hand-authored-plan
counterfactual to rule out an auto-plan-authoring artifact specific to this model on this
already-abandoned (NO-GO) experiment battery.

**5. codebench floor cluster for tiny local models — low confidence.** `gemma2:2b` (0.0, n=4) and
`llama3.2:3b` (0.233, n=5) on hard coding. But some "both agents score 0" task_ids (c43/c46/c47/
c49/c51) also fail intermittently for **aider** across batches (c48: aider 5/5 in batches 2-3, 0/5
in batch 4) — part of this signal is harness/infra flakiness, not a stable capability floor.

## Best-performing combinations

**1. Compiled scaffold + thin (plain-text) leaf lifts small/mid local models to ceiling.**
`llama3.1:8b`/`m1_thin` = **1.00** on reachable (n=6) and hard (n=4); `phi3:mini`/`m1_thin` =
**0.973**/**0.992** (n=6/4) — vs the same models' native-vote profiles floored at 0.000-0.067
above. Small-n caveat (n=4-6).

**2. `gemma2:2b` × `m1_thin` reachable — 0.864, n=26 (largest-n "best" cell), 81% honest-keystone
rate.** A 2B model performing like a mid-tier model from structure alone. Nuance found in the raw
data: one replicate (`bml__gemma2-2b__m1_thin__064_...r1.json`, score 0.96) correctly computes and
selects the right lake's volume/area ratio; a sibling replicate on the **same task**
(`bml__gemma2-2b__m1_thin__reachable_064_...r1.json`, score 0.36) computes all five per-lake ratios
correctly and then **names the wrong lake as the winner**. The scaffold reliably fixes grounding/
fact-collection, but a residual argmax/compare-over-computed-values bug survives even in the best
cells.

**3. `qwen2.5:7b` × `fs2_thin_assemble` on the format-stress tier — 0.92, 100% keystone,
Wilson-lower 0.76 (n=12).** The only `✅✅ CONFIRMED` (not just point-estimate) feasible cell across
`analyze.py`'s entire leaderboard.

**4. `graph_compiled` beats native most decisively exactly where the task is hardest, and gives no
benefit where it's trivial.** At "micro" level (single-fact lookup): `graph_compiled` 0.786 (n=21)
vs native 0.866 (n=82), Δ=-0.080, not significant — structure doesn't help. At "graph"/
"integration" levels the gap flips to d=1.77/1.95 in favor of structure. Monotonic relationship:
structure's payoff scales with task complexity. The cleanest, largest-n confirmation of the central
thesis in the dataset.

## Cross-cutting patterns

- **Structure helps most exactly where multi-step decomposition is unavoidable, and doesn't help
  (or mildly hurts) on single-fact lookups.** Actionable: don't route trivial "micro" tasks through
  the compiled scaffold — the overhead buys nothing there.
- **Two distinct root causes account for nearly all "worst" cells, needing different fixes**: (a)
  web/QA — the native loop stops exploring too early (2-4 visits when the task needs more), a
  confident-wrong failure vote-k/IO-framing retries don't fix (the problem is exploration depth,
  not output formatting); (b) coding — JSON-embedded raw source code is a brittle serialization
  surface (40% SyntaxError rate) independent of coding competence, fixable by moving the write
  protocol toward something closer to aider's diff/search-replace format (0% corruption on the
  identical model/tasks).
- **A residual "correct facts, wrong final compare/select" bug persists even in the best cells**
  (gemma2:2b task 064 above) — matches the already-documented qwen2.5:7b k-th-ordinal gap in
  `MODEL_TIER_LIST.md`, now reproduced at a much smaller model size on a simpler 5-way argmax.
- **Bolt-on mitigations on the native loop (vote-k, I/O-framing retry) don't rescue floor-tier
  models** — only moving control flow entirely out of the model's hands (compiled plan + thin leaf)
  does. "Give the model a pre-authored plan" is qualitatively different from "give the model a
  self-consistency trick," not a point on the same spectrum.

## Confidence caveats

- Most badmodel-lab cells are n=3-6 reps; `MODEL_TIER_LIST.md` itself says to treat single-digit-%
  differences as noise — several numbers above (llama3.1:8b, phi3:mini, gemma2:2b codebench)
  inherit that small-n weakness.
- codebench's qwen2.5:14b comparison is solid at n=63/63, but n=4-5 for gemma2:2b/llama3.2:3b, with
  direct evidence of batch-to-batch infra flakiness affecting even aider on some "floor" task_ids —
  don't read those as clean difficulty floors.
- The gemini-3.1-pro-preview finding is scoped to one already-abandoned (NO-GO) experiment battery;
  the auto-plan vs hand-plan question isn't disambiguated.
- No full shape-by-shape (survivor/chain/conflicting/breadth/argmax/etc., per
  `agent/app/BENCHMARK_SUITE_50.md`) breakdown was done beyond the level-ladder's complexity axis
  and a handful of named single-task illustrations — a natural follow-up, not done here.

## Suggested next-cycle candidate

Worst-finding #1 (codebench JSON-embedded source corruption, 40% syntax-error rate) is the most
concrete and actionable item here: root cause is pinned to two specific, reproducible corruption
patterns in `write_file`'s JSON-string serialization, with file-level examples. A bug-fix cycle
changing the write protocol (or adding a repair/validation pass before grading) is well-scoped for
the dev-cycle methodology.
