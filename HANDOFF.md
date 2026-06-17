# webRAG / Euglena — Session Handoff (2026-06-16)

Start here. This is the project-wide pickup doc; the deep benchmark log is
`services/agent/app/COST_BENCHMARK_HANDOFF.md` (Rounds 1–4).

## TL;DR — current state
Everything below is **committed and on `master` (origin @ `3ab4d82`)** — previously all uncommitted.
Six commits landed this session:
`da4b649` benchmark+DAG layer → `4ec260c` thin-leaf+price-aware vote → `2a622e0` Round-4 doc →
`09ece84` dev agents → `0e6b05b` Phase-1 deletions → `3ab4d82` god-class breakup.

## The thesis (proven)
A cheap model executing an **expensive-model-authored DAG scaffold** recovers premium-reference
accuracy at a fraction of the dollar cost — and beats the same cheap model *building* the graph
itself. Live cross-shape matrix (`run_id xshape_full_20260615_164736`, tasks 050–054, 4 cheap
models + reference, n=3):

- **gpt-4.1-nano B-auto: 0.96 @ $0.0016/task ≈ reference 0.97 @ $0.0655 (≈1/42 the cost).**
- Native cheap-built graph: ~0.33–0.46. Level-ladder (hard graph-level tasks): compiled **0.923**
  vs native graph **0.332** vs sequential 0.755 — compiled is also *cheaper* than the native graph.
- Artifact: `services/agent/idea_test_results/recovery_curve.png` (gitignored).

## Phase 2 — structured work makes weak models stronger (the active research line)
Pushing structure down to the leaf lifts cheap models toward the ceiling:
- **Atomic decomposition** (one fact / one page; chain cross-entity hops with `depends_on`+`{dep_id}`).
  Folding cross-*page* hops into one leaf *regressed* chains — don't.
- **Thin leaf** (`IDEA_TEST_COMPILED_LEAF_MODE=thin`): fixed `search → pick-wiki → visit → extract`
  micro-pipeline; harness owns control flow, LLM only answers micro-questions. Beats the JSON-ReAct
  leaf on weak models at ~half cost.
- **Price-aware anchored voting** (`_votes_for_model` via `model_costs`): cheap→k=5, gpt-5-mini→3,
  premium→1. k independent neutral-prompt extractions → majority prune → repeat-cycle to a 2nd page,
  anchored on the temp-0 read. Result: **nano avg 0.87→0.95** (051 chain 0.71→1.00); flash-lite
  ~0.95 (redundancy helps the *weaker* model more — the thesis).
- Default `LEAF_MODE` is still `react` (thin not yet tested on the premium reference).

## Architecture map (`services/agent/app/`)
- **Engine:** `idea_engine.py` (`IdeaDagEngine`, now 1327 lines after the Phase-3 breakup) + extracted
  modules it delegates to: `idea_chunking.py`, `idea_visit_dedup.py`, `idea_sequencing.py`.
- **Policies/config:** `idea_policies/*` — typed-config views in `config.py` (read `self._cfg.<view>.<field>`,
  never raw `settings.get()`), `actions.py` (2162-line multi-class action registry), schemas in
  `idea_dag_schemas.py`, defaults in `idea_dag_settings.json`.
- **Compiled scaffold:** `testing/compiled_plan.py` (DAG schema v2: leaves+depends_on+`{dep_id}`,
  topo waves), `testing/scaffold_compiler.py` (offline `compile_plan`, disk-cached by mandate hash),
  `testing/execution_compiled.py` (executor: react + thin + price-aware voting).
- **Other arms:** `testing/execution.py` (graph + parametric/naive_rag baselines),
  `execution_sequential.py` (sequential_react). **Runner:** `idea_test_runner.py`, `testing/runner.py`.
- **Tasks:** `idea_tests/test_*.py` — tiered ladder 048–054; **050–054 are the cross-shape set**
  (chain / dependent-chain / breadth-argmin / breadth-argmax / mixed-DAG).
- **Scripts:** `scripts/{compile_plans,cross_shape_experiment,gate_report,recovery_curve,level_ladder,prewarm_fixtures}.py`.
- **Dev agents:** `.claude/agents/*.md` (see below). Gitignored-but-tracked dir; rest of `.claude/` ignored.

## How to run (live = real $; see COST_BENCHMARK_HANDOFF.md for the full recipe)
```
# keys.env is CRLF — strip \r; chroma must be on :8001; PYTHONPATH needs BOTH roots
export PYTHONPATH=services:services/agent
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1   # MANDATORY (shared connectors)
./.venv/bin/python -m agent.app.idea_test_runner   # driver: scripts/cross_shape_experiment.sh
```
Key knobs: `IDEA_TEST_{IDS,MODELS,EXECUTION_VARIANTS,RUNS,FIXTURES(record|replay),RUN_ID}`,
`IDEA_TEST_COMPILED_{PLAN_SOURCE(hand|auto),LEAF_MODE(react|thin),VOTES,CONCURRENCY,AUTHOR_MODEL}`.
Variants: `graph, sequential_react, graph_compiled, parametric, naive_rag`. Author plans offline first:
`scripts/compile_plans.py --tests 050,051,052,053,054 --max-tokens 4096` (052 needs 4096). Analyze by
run-id: `gate_report.py` / `level_ladder.py` / `recovery_curve.py`.

Offline tests (no $): `PYTHONPATH=services:services/agent ./.venv/bin/python -m pytest -q <files>`.

## Dev agents (`.claude/agents/`) — load next session
Five consolidated Claude Code subagents: `task-author` (author+verify+harden a task),
`benchmark` (run live matrices, singleton/$ + analyze), `strategy-tuner` (the Phase-2 A/B loop),
`engine-dev` (engine/policy work), `reviewer` (pre-commit gate + git hygiene, no Claude trailer).
README frames Phase 1 (build the system) vs Phase 2 (improve cheap models). NOTE: custom agents only
register at session START — they were authored this session, so invoke them by name next time
(this session used `general-purpose` with the brief injected).

## Known debt / open items
1. ~~import-context test failures~~ **RESOLVED 2026-06-16 (commit `4717ab8`): suite green —
   0 failed, 377 passed, 17 skipped.** The "import-context / cannot import name IdeaActionType"
   diagnosis was STALE. Actual root causes (17 failures, not 13): `idea_engine_features` ×5 = the
   god-class breakup moved mandate-URL enforcement into `MandateUrlInjectionHook`; restored a thin
   delegating `_enforce_visit_nodes_for_mandate_urls` on the engine (test surface). `got_backtrack`
   ×7 = test self-pollution (stub installer short-circuited when a sibling test imported the real
   `idea_memory` first → stubs now install conditionally, no reverse pollution). `visit_url_extraction`
   ×1 = stale assertion (VISIT uses `io.fetch_url`, not `io.visit`). `connectors_smoke` + `pre_deploy_sanity`
   = environment guards (missing LLM key / `supabase` deploy dep absent from lean venv). `task_definition_preflight_env`
   ×2 = test drift; MIN_CHARS aligned to config source-of-truth `2000` (test had a never-passing `20000`
   — **confirm whether 20000 was the real deploy intent**, else this is settled).
2. **054 (mixed-DAG) nano ≈ 0.75** — REFRAMED 2026-06-16: **plan/config-specific, NOT a genuine
   task floor.** `bhand_recovery_20260616` shows nano on 054 = **0.750 under the HAND plan** vs
   **1.000 under the AUTO (compiler) plan** (both react leaf). The hand plan's 054 leaves are too
   instruction-dense for the weakest models (nano −0.25, gpt-5-mini −0.10 vs auto). Fix = `task-author`
   simplifies the 054 hand-plan leaves, OR just rely on the auto plan (production path, already 1.000).
   Before any "054 push" spend, confirm which config a remaining 0.75 was seen under (likely thin+vote).
3. **Thin on the premium reference — FIXED & RE-VALIDATED 2026-06-16.** Was a catastrophic regression
   (`thinref_20260616`: mean 0.28 vs react 0.94) from thin micro-prompts pinning `max_tokens=24` —
   `gemini-3.1-pro-preview` returned `content=None`/finish_reason=length → all-UNKNOWN cascade (0.25 floor).
   FIX (commit `98818ac`): `_thin_max_tokens_for_model` rides the `_votes_for_model`/`model_costs` buckets
   (cheap 24 unchanged, mid 64, premium 128) + `_thin_micro_query` absorbs a starved RuntimeError as a "" miss.
   RE-VALIDATION (`thinref_fixed_20260616`, $0.63): content=None errors **0** (was 18+ on 054); chain 050 &
   mixed-DAG 054 recovered to react parity (0.88 / 0.94). BUT mean only 0.61 vs react 0.94 — residual gap is
   on breadth fan-out 052/053, and it's **search WRONG-GROUNDING (ambiguous query → wrong page), NOT token
   budget** (128 is sufficient; thin amplifies wrong-grounding because it can't re-search after a bad page).
   Net: content=None bug CLOSED; `thin` safe for chain/mixed on premium; **default stays `react`, thin opt-in.**
   NEXT LEVER (paid, user-greenlight): thin query disambiguation / hand search fixtures for 052/053 breadth.
   **UPDATE 2026-06-17 (UNCOMMITTED, `execution_compiled.py`+test): pursued the breadth lever → it became a SPEED win.**
   Added deterministic `_target_entity(instruction)` (the entity a leaf names; strips the resolved
   "— source: <url>" dep tail) feeding three coupled changes: (a) TITLE-AWARE page-pick `_pick_pages`
   — A/B `thinab_before/after` (n=3 replay) shows it's ~NEUTRAL on breadth (cheap models already
   0.94–1.00 on plain wiki-first; the hoped 0.61→0.94 lift did NOT appear; ref 052≈0.34 is **k=1
   premium-vote coverage dropout, not grounding**); kept because safe + enables (b)/(c). (b) INITIALS
   BUG FIX (the real find): `_target_entity` truncated "F. Scott Fitzgerald"→"F"→grounded to the
   `wiki/F` letter page (masked under the LLM query). (c) QUERY-SKIP — delete the per-leaf
   search-query LLM call, search the entity verbatim. MEASURED (`thinspeed2`, 052 live): **LLM
   calls/ref-run 26→13; reference 86.8s→~40s (~54%), nano 20.2→15.2s, flash-lite 14.1→11.8s; nano &
   flash-lite 052 = 1.00 (faster AND more accurate).** Prod `IDEA_TEST_COMPILED_CONCURRENCY`=6 already
   (script throttles to 3). Offline: 35 compiled / 426 full-suite green. TODO: commit; premium-breadth
   k≥2 to lift ref 052; `.model_pricing_cache.json` is a gitignored artifact — exclude from the commit.
4. ~~Per-cheap-model B-hand lost~~ **RECOVERED 2026-06-16 (`bhand_recovery_20260616`, $0.31, 60 cells,
   pipeline CLEAN).** Hand-vs-auto (052–054 mean): hand WINS pure fan-out 052 (+0.22 flash, +0.13
   flash-lite, +0.10 nano), parity 053, hand DEFICIT on mixed-DAG 054 (nano −0.25, gpt-5-mini −0.10).
   nano hand cost = auto cost = $0.0015/task. Aside: 051 gpt-5-mini=0.25 is wrong-grounding (model
   knowledge, not plan — flash=1.0 on same plan). Summary JSON: `idea_test_results/bhand_recovery_20260616_summary.json`.
5. **God-class breakup — continuing.** 2026-06-16: extracted the pure node-state helpers
   (`is_action_ready`/`get_pending_executable_nodes`/`select_best_global`/`sanitize_action_result`) to
   `idea_node_state.py` (commit `7b9ced9`; engine 1343→1297; +10 unit tests for previously-uncovered code).
   STILL stateful in `idea_engine.py`: `_execute_action` / `_handle_action_result` / the `_handle_*`
   orchestrators (mutate node.status/details; call io/actions/memory/telemetry; only transitively covered by
   `idea_dag_recovery_test::test_dag_action_retry_recovery`). NEXT (TEST-FIRST, careful): `_handle_action_result`
   → new `idea_action_result.py` — largely a pure `(result,node,step,cfg-flags)`→status transition; thread out
   `actions.get().post_execute_provides`, the `_cfg.action.*` retry flags, and keep `_recover_pruned_sibling`
   on the engine (invoke from the caller). BEFORE moving it, add offline tests: VISIT empty-content→retryable;
   retryable-with-attempts→BLOCKED+cooldown; terminal→FAILED+error_details; provides_data contract on success.
   Then extract `_recover_pruned_sibling` (add its sibling-revival test first). Also: dead `_is_leaf_node`
   (defined, never called) can be deleted.
6. ~~Fold legacy `EvaluationWeights` into `EvaluationConfig`~~ **DONE 2026-06-16 (commit `9ecea35`)** —
   class deleted, both eval policies read `self._cfg.evaluation.*`, `weight_for` made None-safe/case-insensitive;
   no runtime behavior change (dead 0.1/0.2 defaults dropped in favor of the JSON-true 0.4/0.5). +2 tests.

## Recommended next order
(1) ✅ green the suite [4717ab8] → (3) ✅ recover B-hand [bhand_recovery_20260616] →
(2) ✅ thin+vote reference test [thinref_20260616] → (NEW) ✅ fix+revalidate thin content=None bug
[98818ac / thinref_fixed_20260616 — premium content=None CLOSED; chain/mixed at parity; breadth grounding gap remains] →
(6) ✅ EvaluationWeights fold [9ecea35] → (5) ✅ engine breakup slice 1 [7b9ced9 — node-state helpers] →
REMAINING (next session): (5b) `_handle_action_result` extraction TEST-FIRST (see debt #5) →
[paid, user-greenlight] thin query disambiguation for 052/053 breadth; 054 hand-plan simplify (low pri).

Memory: `project_compiled_scaffold_thesis`, `project_engine_canonical`, `project_cost_benchmark_state`,
`project_benchmark_run_recipe` (run recipe), `project_typed_config_layer`.
