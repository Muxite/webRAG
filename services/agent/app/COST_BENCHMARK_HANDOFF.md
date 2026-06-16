# Cost-Recovery Benchmark — Handoff

**Goal (the thesis):** prove webRAG's graph turns a cheap model's larger token budget into
premium-model quality at a fraction of the dollar cost — "gpt-4.1-nano/gemini-flash ≈
premium reference at single-digit % cost, with a benchmark to back it." Reference plan:
`/home/muk/.claude/plans/humming-swimming-moler.md`. Operational recipe:
see memory `project_benchmark_run_recipe` (chroma:8001, keys.env CRLF, **concurrency=1**).

---

## ROUND 2 (2026-06-14): harder tests + `verify` capability — DONE, uncommitted

Plan: `/home/muk/.claude/plans/1-services-agent-app-cost-benchmark-hand-splendid-meadow.md`.
Built the harder/differentiating test layer the original pilot said was needed.

**New agent capability — `verify` action.** `IdeaActionType.VERIFY` (base.py) + `VerifyLeafAction`
(actions.py: cross-checks a claim vs. gathered visit evidence, optional 1 authoritative fetch,
emits `{verdict, confidence, supporting_url, contradicting_url, quote}`). Registered in
`LeafActionRegistry`; wired into `idea_dag_settings.json` (`allowed_actions`, `verify_system_prompt`,
expansion ACTIONS block, evaluation reward + `evaluation_weight_verify` in evaluation.py).

**5 new gated/bimodal tests** (`idea_tests/test_040..044`, registered in `TEST_PRIORITY_ORDER`):
040 multi-hop dependent chain · 041 6-source breadth matrix · 042 anti-parametric contradiction
(obscure discovery years; exercises `verify`) · 043 capstone multi-hop breadth + LLM-judged synthesis ·
044 **security CVE root-cause analysis** (CVE-2026-35414 OpenSSH; keystone = vulnerable function
`match_principals_option`). IMPORTANT LESSON: the function name is NOT in NVD/vendor summaries OR even
the detailed write-ups (cve.news) — it is essentially only in the OpenSSH SOURCE. A pure "research the
CVE" version was a constant-0 (test-025 trap: graph stopped at 1 summary page; naive_rag's search 422'd).
**Fixed by seeding two authoritative URLs in the mandate** (the github raw `auth2-pubkeyfile.c` source +
a CVE advisory) so it's reproducible and the agent reads the C to identify the function by its role.
Live (gemini-2.5-flash): graph 0.933 = naive_rag 0.933 > parametric 0.733. Honest discrimination
profile: separates parametric (no visits) from evidence variants via the visit gate, but **does NOT
separate graph from naive_rag** (a bounded 2-source read) and the function name **leaks parametrically**
(15-yr-old OpenSSH function). Value: anti-parametric + code-comprehension + topical diversity, not a
graph-vs-naive_rag discriminator — 040 (dependent chain) remains that. To make it graph-favoring, it
would need breadth (>3 source sections) or a dependent commit→function→helper trace.
Validators use a **keystone 0/1 gate** + secondary checks that **short-circuit to 0 when the keystone
is absent** → bimodal scores, no constant-0.44 trap. Ground truth fetched from live Wikipedia.

**Infra:** `IDEA_TEST_FIXTURES=replay_strict` (miss=fail, no live fallback) in web_fixtures.py +
connector_http.py; fixture hit/miss/miss_rate in `summarize_observability` (utils.py);
`recovery_curve.py --run-id <YYYYMMDD_HHMMSS>` for clean per-run aggregation.

**TWO BUGS FOUND & FIXED in `idea_test_runner.py`:**
1. Line ~827 hardcoded `allowed_actions` WITHOUT `verify` → the runner silently stripped the new
   action so the planner could never pick it. Now includes `verify`.
2. **Intra-graph parallelism breaks wide breadth.** `parallel_action_limit=4` runs 4 visits at once
   on the SHARED http connector; CPU-bound page parsing starves the loop and all visits time out at
   20s (test 041 graph scored 0.000 with 0 visits). New env overrides:
   `IDEA_TEST_PARALLEL_ACTION_LIMIT=1` (serialize) + `IDEA_TEST_{VISIT,FETCH,ACTION}_TIMEOUT`.
   With `=1` + 45s timeouts, 041 graph went **0.000 → 1.000 (6/6 visits)**. **The matrix MUST set
   `IDEA_TEST_PARALLEL_ACTION_LIMIT=1` or breadth tests fail.**

**Live smoke (gemini-2.5-flash, serialized, FIXTURES=record):**
- 040: graph **0.833** ≫ naive_rag 0.083 > parametric 0.000 — STRONGEST discriminator; the multi-hop
  *dependent* chain is the only test naive_rag structurally cannot do. ✓
- 041: graph **1.000** > naive_rag 0.875 > parametric 0.750 (breadth, after the parallelism fix). ✓
- 042 (rewritten, obscure years): graph **1.000** > naive_rag 0.938 > parametric 0.750. NOTE:
  gemini-2.5-flash's parametric *still knew* 1807/1901 (keystone=1.0 at 0 visits) — even "obscure"
  real Wikipedia facts aren't obscure to a capable model. The `visit_count` check is what now
  separates parametric (caps it at 0.75). For a capable model, naive_rag (3 visits) stays
  competitive on breadth/contradiction; only the **dependent multi-hop chain (040)** truly breaks it.

**LIMITATIONS / next-step findings:**
- **`verify` is enabled but the planner does NOT select it** (verify_nodes=0 in every smoke cell,
  incl. 042). The action works in isolation; the planner just visits+synthesizes instead. Added a
  stronger expansion-prompt mandate clause ("MUST create a verify node for fact-check mandates") —
  NOT yet confirmed to change behavior. To actually exercise verify, likely need engine-level
  enforcement (mirror the existing "ENFORCE: injected visit node for mandate URL" in idea_engine.py)
  or a harder mandate. Until then 042's signal comes from visit+synthesis, not verify.
- **Parametric leak is intrinsic for capable cheap models.** Truly anti-parametric tasks need
  answers that are *computed/combined* across sources (not a single stored fact) or genuinely
  un-memorizable values. 040's dependent-chain shape is the most robust pattern; consider more 040-style
  tests for the headline.

**Final scored subset (pass via `IDEA_TEST_IDS`):** `026,036,040,041,042,043,044` (026=floor, drop
025/019; 025's adjacency validator is the constant-0.44 culprit). 044 is the deepest-research test.

### Layered taxonomy (NEW format for all tests from 040 onward)
The original 001-039 suite is kept as the **comparison barrage**. Every test from 040 on carries
`level` + `weight` in `get_test_metadata()` and follows a 4-layer ladder (micro -> integration ->
navigation -> graph). Aggregate by level with **`scripts/level_ladder.py --run-id <id>`** (prints
success / visits / tool-calls / usd / secs / groundedness per level per variant).
- **micro** (short): `045` single-page fact extract + stop-early (efficiency-graded).
- **integration** (long/short): `041` 6-source breadth, `042` contradiction/verify, `044` CVE 2-source read.
- **navigation** (long): `046` link-following traversal (Apollo 11 -> Saturn V -> Boeing); adjacency
  verified OBJECTIVELY from the agent's own visit `links_full` (defeats search/parametric shortcuts).
- **graph** (long): `040` dependent multi-hop chain, `043` capstone, `047` wiki-race shortest chain
  (Pizza -> Roman Empire); every reported hop verified against visited-page links — the proper fix for
  the weak self-reported adjacency that made `025` a constant 0.44.
New helper `idea_test_utils.build_visit_link_graph(result)` returns `{url: set(outgoing links)}` from
visit results; `normalize_url` for matching. Offline-validated: 046 parametric→0 (must actually visit
the hop), 047 claimed-but-not-traversed chain→0.

**Not yet done (real $ — user-triggered): the Part-5 live matrix.** Prewarm (record, reference
model), then full matrix in `replay`, `RUNS=3`, both refs + cheap, tiers `0,10,20,40`,
`IDEA_TEST_PARALLEL_ACTION_LIMIT=1`, then `recovery_curve.py --run-id`. 043 + rewritten 042 still
need a live discrimination confirmation. Note: `verify` is available but the planner only chooses it
when the mandate is fact-check-shaped AND the facts aren't trivially known — watch verify-node usage.

---

## ROUND 3 (2026-06-15): compiled-scaffold DAG generalization — DONE offline, uncommitted

Plan (user): generalize the `graph_compiled` scaffold from a flat parallel leaf set to a full
**DAG**, add an **offline compiler** that authors plans for any mandate, and run a cross-shape
A/B/C experiment. Thesis at full strength: *an expensive-model-compiled DAG + cheap execution
beats a cheap-model-built DAG across task SHAPES* (fan-out, dependent chains, mixed). One
mechanism now subsumes the breadth win (052) and the chains (050/051).

**Built (all offline-tested — 100 unit tests green; LIVE matrix is the next user-$ step):**
- `testing/compiled_plan.py` (NEW, pure/no-I/O) — plan **schema v2**: leaves gain optional
  `depends_on:[ids]` and `{dep_id}` one-level templating. `validate_plan` (cycle / missing-dep /
  dup-id), `topological_waves` (Kahn — independent leaves share a wave → parallel),
  `substitute_deps`, `plan_structure` (answer-free shape for auto-vs-hand diffing). Backward-
  compatible: no deps → single wave = the old v1 fan-out, so **052 is unchanged**.
- `testing/scaffold_compiler.py` (NEW) — `compile_plan(mandate, author_model, agent_io)`,
  **cache-first** by `mandate_hash`; strong meta-prompt → DAG JSON; disk-cache at
  `services/agent/compiled_plans/<hash>.json` (the "paid-offline" artifact, env
  `IDEA_TEST_COMPILED_PLANS_DIR`). `parse_plan` tolerates ```json fences/prose. Default author
  `google/gemini-3.1-pro-preview` (env `IDEA_TEST_COMPILED_AUTHOR_MODEL`).
- `testing/execution_compiled.py` — `_execute_plan` now runs **topological waves** (parallel
  within a wave, dependent leaves after their deps with `{dep_id}` substituted), then aggregates
  over all leaves in plan order. New `_resolve_plan` = **auto-wiring**:
  `IDEA_TEST_COMPILED_PLAN_SOURCE=hand` (default: hand `get_compiled_plan()` else compiler) /
  `auto` (always compiler — measures **B-auto**). The compiler runs on a **separate
  telemetry/AgentIO** so its offline cost never pollutes runtime $; the run result now carries
  `output.plan_source`, `output.plan_structure`, and a top-level `compiler` cost block (cache
  hits cost nothing). `IDEA_TEST_COMPILED_FORCE_RECOMPILE` to bypass cache.
- **New tasks** (registered 048–054 in `TEST_PRIORITY_ORDER`):
  - `053` — 2nd breadth/argmax, different domain, **page-only facts**: deepest of 6 lakes by max
    depth (Baikal **1,642 m** keystone; margin 172 m over Tanganyika). Each depth verified live
    against the lake's Wikipedia infobox 2026-06. Coverage diagnostic forces 6 page reads
    (depths aren't memorizable), hardening grounding vs 052's parametric-leakable birth years.
  - `054` — **mixed DAG**: 2 independent author look-ups (parallel wave) + 1 dependent hop on the
    first (Beloved→Morrison→MA **Cornell** keystone), templating `{author_beloved}`. Exercises
    the executor's full topology; the hand plan is what auto-compiled plans are diffed against.
- `scripts/compile_plans.py` — offline pre-author the cache (`--tests`, `--author-model`,
  `--force`, `--dry-run`, `--show`). `scripts/cross_shape_experiment.sh` — the A/B/C driver
  (A=graph, B-hand & B-auto=graph_compiled, C=sequential_react; reference ceiling; `REPEATS=3`;
  pins `IDEA_TEST_RUN_ID` for clean `--run-id` analysis). Added `IDEA_TEST_RUN_ID` override to
  `idea_test_runner.py`.

**Offline verification:** `compiled_plan_test.py`, `scaffold_compiler_test.py` (mocked author),
`execution_compiled_test.py` (stubbed `_run_leaf` — fan-out / chain templating / mixed DAG / cycle
reject / leaf-failure isolation), `breadth_argmax_validators_test.py` (053),
`mixed_dag_validators_test.py` (054). All 54 idea_tests modules load; variant parser maps
`graph_compiled`. `PYTHONPATH=services:services/agent ./.venv/bin/python -m pytest -q
services/agent/tests/{compiled_plan,scaffold_compiler,execution_compiled,breadth_argmax_validators,mixed_dag_validators}_test.py`.

**LIVE CROSS-SHAPE MATRIX — RUN 2026-06-15 (gated). THESIS PROVEN.**
Run via `bash scripts/cross_shape_experiment.sh` (chroma:8001, keys.env). Gated rollout: Stage A
(author 5 plans) → 052 B-auto gate (1.00) → flash full cross-shape (Gate 2) → re-author+B-auto
n=3 (Gate 2.5) → full matrix (4 cheap models + reference, n=3). run_id `xshape_full_20260615_164736`.

Level-ladder (all models pooled): on the **graph level** (051/052/053/054 — chains/breadth/mixed)
`graph_compiled` **0.923** vs native `graph` **0.332** vs `sequential_react` 0.755, and compiled is
*cheaper* than native graph ($0.0123 < $0.0176). Native graph barely grounds (0.25 vs 0.80). On the
**navigation level** (050, easy 2-hop) compiled 0.990 ≥ seq 0.885 ≥ graph 0.760.

Per cheap model, B-auto (cheap exec of the auto-compiled DAG) over the 5 tasks vs reference ceiling:
| model | B-auto | native graph | sequential | $/task |
|---|---|---|---|---|
| openai/gpt-4.1-nano | 0.96 | 0.46 | 0.79 | $0.0016 |
| google/gemini-2.5-flash | 0.95 | 0.47 | 0.79 | $0.0071 |
| openai/gpt-5-mini | 0.95 | 0.29 | 0.75 | $0.0144 |
| google/gemini-2.5-flash-lite | 0.88 | 0.46 | 0.81 | $0.0017 |
| **ref gemini-3.1-pro (ceiling)** | **0.97** | 0.38 | 0.76 | $0.0655 |
HEADLINE: **gpt-4.1-nano B-auto 0.96 @ $0.0016/task = ~99% of the premium reference (0.97) at 1/42
the cost**; the same cheap model *building* the graph at runtime gets 0.46. Pareto knee = nano-compiled.
Artifact: `services/agent/idea_test_results/recovery_curve.png`.

Auto vs hand: the compiler reproduces hand quality — 052 auto chose a *more granular* 12-leaf
[6,6] DAG (author wave → birth-year wave) vs the hand 6-leaf [6] yet matches it (flash hand 1.00 /
auto 0.78-0.94; ref hand 052/053/054 = 1.00/0.96/1.00). 050/051 have no hand plan so "hand" source
falls back to the compiler.

TWO BUGS FOUND & FIXED during gating (both have regression tests):
1. **Compiler grounding wording.** Auto leaves originally said "search X, open its page" — weaker
   than the hand plan's "read DIRECTLY from the page, do not guess from memory". On 053 (page-only
   depths) the cheap model guessed Baikal at 1,700 m (real 1,642) → keystone miss (B-auto 053 0.29).
   Fix: `scaffold_compiler._META_PROMPT` rule 1 now mandates that grounding clause in every leaf.
   B-auto 053: 0.29 → 0.97.
2. **Validator newline brittleness.** 052/053 keystone proximity used `[^.\n]`, so a report that put
   the answer on the line AFTER "Deepest lake:" header failed to match a *correct* answer. Fix:
   `[^.\n]`→`[^.]` (newline-tolerant, period-bounded) and dropped non-superlative trigger
   "maximum depth" from 053. Re-scoring the same deliverables lifted B-auto 053 0.47 → 0.97.

DRIVER BUG FIXED: Stage C1 (cheap hand) and C2 (cheap auto) shared one run-id, so auto's
`graph_compiled` files overwrote C1's hand files for 052/053/054 (lost cheap B-hand for this run;
recovered via Gate-2 flash + reference). Fixed: C2 now uses `${RUN_ID}_auto` (still prefix-matched
by `--run-id`). The committed run's per-model B-hand can be recovered with a small C1-only rerun.

Watch: compiler over-/under-decomposing (logged via `plan_structure`); compiled-leaf parallelism vs
the shared-http starvation the Round-2 fix addressed (`IDEA_TEST_COMPILED_CONCURRENCY` default 3).
052 needs author `--max-tokens 4096` (12-leaf plan overflows 2048). Fixture caveat: 050-054 are
URL-free so the driver records on the reference pass and replays (replay-or-record) for cheap models
— not byte-identical evidence; tighten only after a prewarm that captures the discovered URLs.

---

## ROUND 4 (2026-06-16): structured-work strategy makes weak models stronger — DONE, uncommitted on branch

Branch `compiled-scaffold-dag` (commits 66b5718 benchmark+DAG layer, 815f284 thin+vote). Goal:
push structure DOWN to the leaf to lift cheap/weak models. Gated A/B experiments (flash-lite, nano,
n=3, on 050-054) drove every choice.

DECOMPOSITION — atomic wins. A "fold same-entity multi-hop into one leaf" rule helped independent
breadth (052 flash 0.78->0.97) but REGRESSED chains by cramming cross-PAGE navigation into one
budget-limited leaf (050 nano 1.00->0.38, 051 gpt-5-mini 0.75->0.17). Reverted to ATOMIC (one fact
/ one page; chain cross-entity hops with depends_on + {dep_id} templating). Restores chains.

THIN LEAF (`IDEA_TEST_COMPILED_LEAF_MODE=thin`, default still `react`). Replaces the per-leaf JSON
ReAct loop with a fixed micro-pipeline: thin search query -> pick the wiki result (heuristic) ->
visit -> extract the value. The harness owns control flow; the LLM only answers micro-questions with
tiny outputs. Beats react on weak models at ~HALF the cost (flash-lite 052 0.86->1.00, $0.0040->$0.0023).

PRICE-AWARE VOTING (`_votes_for_model` reads output $/Mtok via model_costs). A dirt-cheap (weaker)
model spends cheapness on redundancy: k independent NEUTRAL-prompt extractions -> majority-vote
prune -> repeat-cycle to a 2nd page if no consensus; ANCHORED on the temp-0 read (ties break to it)
so clean facts stay stable while uncertain ones get rescued. k: out<=$1/Mtok->5, <=$5->3, premium->1
(nano=5, flash-lite=5, gpt-5-mini=3, gemini-3.1-pro=1). Override `IDEA_TEST_COMPILED_VOTES`.
Unanchored temp-0.5 voting helped chains but hurt clean breadth (nano 052 0.93->0.81); the temp-0
ANCHOR fixed it (nano 052 ->0.99) while keeping the chain win (nano 051 0.71->1.00).

RESULT (atomic + thin + anchored price-aware vote): **nano avg 0.87 -> 0.95** (051 chain 0.71->1.00,
052 0.93->0.99); flash-lite ~0.95 (already strong — redundancy helps the WEAKER model more, exactly
the thesis). Lone residual: 054 nano 0.75 (mixed-DAG task floor, not a strategy artifact).
CITATION FIX: aggregation facts numbered "Fact N" (not "[leaf_id]") + "cite only URLs" — weak models
stopped echoing leaf ids as citations (flash-lite 054 citation 0.00 fixed).

Knobs: `IDEA_TEST_COMPILED_LEAF_MODE=thin|react`, `IDEA_TEST_COMPILED_VOTES=<k>` (else price-aware),
`IDEA_TEST_COMPILED_CONCURRENCY` (lower to ~2 when voting — each leaf fans out k inner calls).
NEXT: try thin+vote on the reference model (untested — keep react default until then); push 054;
recover per-cheap-model B-hand (Round-3 driver collision, now fixed).

---

The ORIGINAL pilot section below is retained for the instrumentation map and gotchas.

---

## 1. What is already built & verified (do not redo)

All changes are in the working tree (uncommitted, per user instruction). Verified by
byte-compile + live runs against OpenRouter.

| Area | File(s) | What |
|---|---|---|
| USD cost | `testing/utils.py`, `testing/execution.py`, `testing/report.py` | `summarize_observability(result, telemetry, model_name)` adds a `cost` block via `model_costs.estimate_cost`; chars/token fallback flagged `estimated`. Surfaced in report. **Live-verified** (real OpenRouter usage). |
| Roster + JSON gate | `testing/config.py` (`BENCHMARK_ROSTER`, aliases), `idea_test_runner.py` (`_preflight_json_capable`) | Models registered as OpenRouter slugs; preflight drops models that can't emit `json_mode`. Use `IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096` so reasoning models aren't false-dropped. **All 8 roster models pass.** |
| Baselines | `testing/execution.py` (`run_baseline_execution`, `_run_parametric`, `_run_naive_rag`), `testing/runner.py`, `idea_test_runner.py` | New variants `parametric` (1 call, no tools) and `naive_rag` (1 search+visit round → 1 synthesis). Same output shape as graph. |
| Fixtures | `agent/app/web_fixtures.py`, `connector_http.py` | `IDEA_TEST_FIXTURES=record|replay|off`, keyed by method+url+params (auth excluded). Hooked into `ConnectorHttp.request`. **replay = replay-or-record** (fills misses live — see blocker #3). |
| Effort tiers | `idea_test_runner.py` (`_parse_effort_tiers`, `_apply_effort_tier`), `testing/execution.py` | `IDEA_TEST_EFFORT_TIERS="0,20,40"` (0=default budget); caps `max_total_nodes` + `max_steps`. Matrix dimension; baselines forced to tier 0. |
| Analysis | `scripts/recovery_curve.py` | Aggregates priced runs → CSV + Pareto + recovery-curve PNG with 2 reference lines + crossing report. Flags `--tests`, `--since`, `--reference-models`. |
| One-off | `scripts/_probe_roster.py`, `scripts/run_pilot.sh` | Roster JSON-gate probe; pilot driver (reference record → cheap replay → curve). |

**Roster (OpenRouter slugs), all pass the JSON gate:**
- Reference: `google/gemini-3.1-pro-preview` (NOT `...3.1-pro` — that 400s). Secondary: `openai/gpt-5`, `google/gemini-2.5-pro`.
- Cheap: `google/gemini-2.5-flash`, `openai/gpt-5-mini`, `openai/gpt-4.1-nano`, `google/gemini-2.5-flash-lite`, `openai/gpt-5-nano`.

---

## 2. Pilot result (clean, 6-test subset, n=6, concurrency=1)

Per-test GRAPH score (tier 0):

| test | ref-3.1pro | gpt5mini | gpt4.1nano | flash | flash-lite | note |
|---|---|---|---|---|---|---|
| 026 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | saturated (no signal) |
| 025 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 | **constant → broken validator** |
| 019 | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 | near-saturated |
| 037 | 0.71 | 0.97 | 0.93 | 0.98 | 0.54 | discriminates |
| 038 | 0.50 | 0.96 | 0.99 | 0.00 | 0.00 | discriminates; gemini graph collapses |
| 036 | 0.93 | 0.86 | 0.77 | 0.41 | 0.85 | discriminates |

Headline *appears* huge (nano naive_rag 0.892 @ $0.0007 vs reference graph 0.764 @ $0.204)
but is **not yet trustworthy** because of the blockers below.

---

## 3. The three blockers to fix (next-session tasks)

### Task A — Harden the test subset (the gate to a credible number)
- **Inspect & fix/drop test 025** (`idea_tests/test_025_*`): every model returns exactly 0.44 → the validator is non-discriminating (likely a fixed partial score). Either fix the scoring or drop it.
- **Drop 026 and 019 from the scored subset** (saturated at ~1.0) — keep 026 only as a sanity floor if desired.
- **Author 3–4 genuinely hard tasks where one-shot `naive_rag` FAILS but the graph should win.** This is the crux: on the current subset `naive_rag ≥ graph` for weak models, so the graph's value is undemonstrated. Need deep multi-hop chains, cross-source reconciliation, and faithfulness/verification (e.g. extend `test_039` style). Prefer **function-based validators** (objective); test module API: `get_test_metadata / get_task_statement / get_required_deliverables / get_success_criteria / get_validation_functions / get_llm_validation_function` (see `idea_tests/test_026_*` for the pattern). Keep any LLM judge on the fixed `gpt-5-mini`.
- Target final subset: ~6 tests spanning LOW→VERY-HIGH context, all discriminating.

### Task B — Make the reference comparison fair
- Today the reference ran in `record` (live) and under-explored (55K ctx) while cheap models replayed (176K). Evidence wasn't identical.
- Fix: **pre-warm complete fixtures** (visit every mandate-named URL + the search results) once, then run **all** models — including the reference — in `replay`. Consider adding a strict-replay mode to `web_fixtures` (miss = fail, not live) so asymmetry is impossible, or a small prewarm script that fetches all subset URLs.

### Task C — Data hygiene in analysis
- `recovery_curve.py --since` matches on filename prefix (`YYYYMMDD_HHMMSS`). Multiple same-day runs pollute aggregates (flash showed n=18 from earlier broken runs). Workaround used: `--since 20260614_0715`. Better: add `--run-id` filter, or archive/move stale `idea_test_results/*.json` before a clean run.

### Then — Task D — Full matrix for the real number
Once A–C are done: both references + cheap models, variants `parametric,naive_rag,graph`,
tiers `0,10,20,40`, `IDEA_TEST_RUNS=3` (variance), `IDEA_TEST_FIXTURES=replay`,
`IDEA_TEST_CONCURRENCY=1`. Then `recovery_curve.py`.

---

## 4. How to run (copy-paste)

```bash
cd /home/muk/projects/webRAG
export OPENROUTER_API_KEY="$(grep -E '^OPENROUTER_API_KEY=' services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/')"
export SEARCH_API_KEY="$(grep -E '^SEARCH_API_KEY=' services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/')"
export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5
# IDEA_TEST_CONCURRENCY=1 is MANDATORY (shared connectors). PYTHONPATH needs BOTH roots.
IDEA_TEST_MODELS="..." IDEA_TEST_IDS="..." IDEA_TEST_EXECUTION_VARIANTS="graph,parametric,naive_rag" \
IDEA_TEST_EFFORT_TIERS="0,20" IDEA_TEST_CONCURRENCY=1 IDEA_TEST_RUNS=1 IDEA_TEST_FIXTURES=replay \
IDEA_TEST_MAX_STEPS=40 IDEA_TEST_REPORT_VERBOSITY=1 MODEL_NAME=openai/gpt-4.1-nano \
PYTHONPATH=services:services/agent ./.venv/bin/python -m agent.app.idea_test_runner

./.venv/bin/python scripts/recovery_curve.py --since 20260614_HHMM --tests "<ids>"
```

`scripts/run_pilot.sh` is a working reference driver (reference-record → cheap-replay → curve).

---

## 5. Known gotchas / optional improvements
- **Shared connectors force serial runs.** A real fix = per-task `ConnectorLLM/Search/Http` instances in `idea_test_runner.run_single_test`; would let the matrix parallelize (big wall-clock/cost win). Until then, concurrency=1.
- `keys.env` is CRLF — always `tr -d '\r'` when extracting.
- ChromaDB must be reachable; `euglena-chroma` publishes host `:8001` (compose says `chroma:8000`).
- `idea_graph_analyzer.py` has a stray `from app.…` import → benchmark runner needs `PYTHONPATH=services:services/agent`.
- Artifacts from this session live in `services/agent/idea_test_results/` (`recovery_curve.{csv,png}`, `web_fixtures/`, per-run JSON).
