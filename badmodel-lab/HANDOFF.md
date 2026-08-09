# Bad-Model Lab — Session Handoff

> ## ⚠️ STATUS UPDATE (2026-07-29, continued overnight) — plan library built, dogfooded, and
> ## HONESTLY does not help yet; a second, more consequential bug found along the way
>
> **What happened after the 2026-07-28 block below.** The user redirected mid-session: instead
> of more mitigation A/B on the reachable-tier wall, build a persistent, semantically-searchable
> **plan library** — pre-authored, parameterized composition-strategy templates, retrieved
> automatically (a pre-expansion short-circuit) and on-demand (an LLM-chosen action) — targeting
> that same reachable-tier wall. Full design + decision log:
> `/home/muk/.claude/plans/my-current-idea-is-wise-papert.md`. Built via 6 sequential engine-dev
> passes (contract-outcome log; schema+adapter; retrieval+6 seed templates; a slot-fill extraction
> layer, a gap found mid-build; the automatic short-circuit; the on-demand action), each
> **independently re-verified** (diffs read, tests re-run myself, not just trusted) before the
> next phase started. Offline suite grew 1955 -> 2134 passed / 18 skipped, zero regressions. One
> real bug caught and fixed mid-build: the sync script's manifest trusted a repo-committed hash
> file without ever confirming the TARGET Chroma actually had the vectors — a fresh checkout
> following the script's own documented usage would have silently synced nothing, forever;
> reproduced, fixed, re-verified.
>
> **First live dogfooding run: the retrieval mechanism works perfectly; the accuracy result is
> negative, and a second, bigger bug was found investigating why.** Synced the library into the
> real benchmark Chroma, ran qwen2.5:7b on the reachable tier. Raw score (0.045) was *worse* than
> the earlier organic baseline (0.068) — investigated rather than accepted at face value, and the
> reason was NOT retrieval quality (it fired correctly on 100% of 42 total reps across two
> attempts, matching sensible templates every time). **The reason: `got_dedup_enabled` defaults
> to `true` — baseline engine behavior, not one of the A1-A5 opt-in levers — and the per-mandate
> memory namespace is keyed on mandate TEXT only, never cleared between R>1 reps of the identical
> task.** Plan-library candidates are perfectly reproducible rep-to-rep (same template, same
> filled entities every time), so rep 2/3's regenerated candidates score 0.86-0.96 similarity
> against rep 1's own already-stored memory (threshold 0.75) and get filtered as duplicates of
> themselves — collapsing every template's 5-7 leaves down to `filter_duplicate_candidates`'s
> all-filtered fallback of exactly 1. Confirmed directly: task 062's very first rep, run before
> any memory existed for that mandate, scored 0.63 on its own — the clean signal was real; every
> later rep, in both attempts, inherited polluted memory off the same persistent Chroma.
>
> **Consequence bigger than the plan library itself:** since this is baseline engine behavior,
> not something scoped to tonight's new mechanism, **every R=3 reachable-tier number from earlier
> tonight (a0=0.068, a2=0.178, the 14B scale test=0.196) may be confounded by the same
> interaction**, to a degree this session did not have time to audit (organic/LLM candidates vary
> more rep-to-rep than a template's deterministic fill, which may be why it was never visibly
> triggered before — but nothing confirms it wasn't happening quietly). Added
> `IDEA_TEST_GOT_DEDUP` (new env override, `idea_test_runner.py`, the exact existing 4-hop
> pattern) so a benchmark can control for this; re-verified the offline suite stays green with it.
>
> **Clean, controlled re-test (both arms, `IDEA_TEST_GOT_DEDUP=0`, fresh):**
> - **a0 baseline (organic): avg 0.140** — roughly 2x the original polluted reading. Zero dedup-
>   collapse lines anywhere in this run.
> - **a4 plan library: avg 0.088.** **Honest conclusion: the plan library performs WORSE than
>   organic planning on this first live test**, not the hoped-for improvement. Task 062 (the
>   highest-branching seed template, 6 candidates) is the clearest regression: all 3 reps show
>   **zero visits** despite 6 search candidates being created — the searches never chained into a
>   page visit at all, vs organic's reliable 2 visits/7 nodes. Working hypothesis, NOT confirmed:
>   the adapter (`idea_policies/plan_library.py`) emits every filled leaf uniformly as
>   `action="search"`, never `action="visit"` with a resolved link or explicit search-then-visit
>   pairing — plausibly weaker than whatever organic planning does to reliably escalate a search
>   into a visit, especially at higher branching factor. Scoped, concrete next step for a future
>   session, not chased further tonight.
>
> **Net assessment:** the engineering is sound and fully verified end-to-end (100% retrieval hit
> rate across 42 reps, correct slot extraction from raw mandates every time, structurally valid
> candidates, both logs working as designed) — but the mechanism AS IMPLEMENTED does not yet help
> accuracy, and likely hurts it, with one concrete lead on why. Reported as a real negative
> result, not smoothed over. The dedup/memory-persistence discovery is arguably the more valuable
> finding of the two: a general benchmark-methodology bug, not specific to the plan library, that
> may be quietly distorting other multi-rep results in this project and deserves the more urgent
> follow-up. Full blow-by-blow: `badmodel-lab/OVERNIGHT_2026-07-28.md`.

> ## ⚠️ STATUS UPDATE (2026-07-28, overnight) — pivot to NATIVE (non-compiled) planning + a real
> ## infra bug found and fixed; read this first, then `OVERNIGHT_2026-07-28.md` for the full log
>
> **The redirect.** Everything below this block (2026-07-23 and earlier) is about the COMPILED
> scaffold (`graph_compiled`): a strong model authors a DAG plan offline, a cheap/weak model executes
> it. That thesis is proven and settled — see `project_compiled_scaffold_thesis` memory. It is
> explicitly **not** the goal of webRAG; the native (non-compiled) adaptive engine
> (`execution_variant=graph`, where the model itself plans/expands/decides each turn — see
> `agent/app/ADAPTIVE_ENGINE.md`) is. Nobody had run *that* against local sub-15B models
> before this session — only cloud models (gpt-5-mini). This session's mandate: make 6-12GB-class
> local models (qwen2.5:7b, llama3.1:8b, qwen2.5:14b — one at a time on a 12GB card) viable to
> **plan**, not just execute a plan, for the adaptive engine. Ran ~6h autonomous/overnight
> (2026-07-28 09:00-~12:40 UTC) with an independent adversarial review at every step per explicit
> user instruction (see `OVERNIGHT_2026-07-28.md` for the full step-by-step log, including
> everything that was checked and how).
>
> **Headline finding: a real infra bug was silently corrupting every native-mode local-model score.**
> Ollama was serving every request at its runtime default `n_ctx_slot=4096` regardless of a model's
> actual trained context (qwen2.5:7b supports 32768) — `docker logs badmodel-ollama` showed active
> `context shift` events silently discarding ~half the context on any prompt over ~4095 tokens. The
> native engine's merge/finalize step routinely builds far bigger prompts than that (confirmed even a
> single-leaf micro task's finalize call was ~5,400 tokens). Fixed via `OLLAMA_CONTEXT_LENGTH=16384`
> in `docker-compose.yml` (server-wide; the OpenAI-compat endpoint does NOT honor a per-request
> `options.num_ctx` override — tested and confirmed). **Verified empirically** (not assumed): an
> 8,052-token test prompt now processes with `truncated=0`. Before/after on the *identical* task set
> (qwen2.5:7b, `a0_native_baseline`, R=3): **micro tier 0/9 (avg 0.00) -> 6/9 (avg 0.83)** — the
> earlier "complete floor" was mostly this bug, not model incapability.
>
> **Micro tier (atomic single-page fact extraction) is a real, replicated win across the whole
> 6-12GB band once the bug is fixed** — avg score qwen2.5:7b 0.83 (n=36, fresh R=12 confirmation run),
> llama3.1:8b 0.89 (n=9), qwen2.5:14b 0.94 (n=9). **Honest caveat: NOT statistically "confirmed"**
> per this project's own strict convention (95% Wilson lower bound on binary pass rate >= 0.75) —
> computed properly, all four results land at Wilson-lower 0.45-0.57. The signal is real and
> replicated (0.83 avg held from n=9 to n=36), just not proven at the project's usual bar; would need
> substantially more reps than tonight's budget allowed. Diagnosed the residual gap between a "1.0"
> and a "0.5" rep: **every 0.5 case already has the correct fact — it just omits the source URL from
> the final answer text.** A narrow, targeted mitigation (`a3_native_expect_contract`, isolating ONLY
> the A4 `expansion_expect_contract_enabled` flag) nudged qwen2.5:7b micro from 0.83 -> 0.94 avg
> (n=9) — promising, same small-n caveat applies.
>
> **Reachable tier (Tier-5 multi-page composition: argmax/subset-sum/negation) is a genuine wall, not
> an infra artifact** — the context fix did NOT move it (0.083 -> 0.068 avg, unchanged within noise),
> consistent with this same model capping at 57% even via the compiled path (this doc, above). Two
> independent "burn more compute" levers were tried and gave near-identical, modest lift: raw scale
> (qwen2.5:7b -> qwen2.5:14b, avg 0.068 -> 0.196, n=21, broad across 6/7 tasks) and inference-time
> decision-making (`a2_native_good_adaptive` = reexpand + confidence-judge + confidence-reexpand, on
> qwen2.5:7b, avg 0.068 -> ~0.178, n=19/21) — the latter confirmed genuinely firing (100+ log
> mentions, graphs reaching 20+ steps vs baseline's ~7) but at **3-10x the wall-clock cost per task**.
> Neither comes close to the 0.75 bar. Also found (and left as-is, a genuine hardware ceiling, not a
> bug): the biggest reachable-tier merges (up to ~330k chars, app-layer-capped at 100k) can still
> exceed even the fixed 16384-token context on the largest local model — `nvidia-smi` showed
> qwen2.5:14b at 16384 ctx already using 10,857/12,288 MiB, no safe headroom to raise it further on
> this card.
>
> **Final ablation round, closes the story cleanly.** On micro tier, `a1_native_reexpand` alone
> (7/9, avg 0.89) and the full `a2` combo (7/9, avg 0.89) land at the SAME lift — the extra
> confidence-judge machinery buys nothing over plain reexpand here — while `a3_native_expect_contract`
> does marginally best (8/9, avg 0.94). Reading: micro tier was already near-saturated at baseline
> (0.83); the residual gap was small and specifically citation-shaped, so several different adaptive
> mechanisms nudge it similarly, with the one actually targeted at citations doing best. **The
> cleanest result of the ablation:** running `a3_native_expect_contract` on the REACHABLE tier as a
> negative control gave **0/21, avg 0.069 — statistically identical to `a0`'s 0.068 baseline.** The
> citation-fix genuinely does nothing for composition tasks. This confirms the two tiers fail for two
> different, non-overlapping reasons (citation-formatting vs. multi-hop reasoning composition), not
> one flag that happens to help everything — the diagnosis is mechanism-specific, which is exactly
> what you want to see before trusting it.
>
> **New files this session** (all uncommitted — nothing pushed, per repo convention of only
> committing when asked): `run_adaptive_cell.sh` (native-`graph`-variant counterpart to the
> compiled-only `run_cell.sh`), `profiles/a0_native_baseline.env` / `a1_native_reexpand.env` /
> `a2_native_good_adaptive.env` / `a3_native_expect_contract.env`, `docker-compose.yml`'s
> `OLLAMA_CONTEXT_LENGTH` addition, and one small production-engine change (adversarially reviewed,
> independently re-verified, full offline suite green at 1955/18 skipped):
> `idea_policies/expansion.py` now emits `json_telemetry` for the native expand step too (previously
> a silent no-op there — a gap an adversarial review caught before any run was trusted). Full
> step-by-step log, every adversarial-review verdict, and exact numbers: `OVERNIGHT_2026-07-28.md`.
>
> **Open next steps:** (1) get real Wilson-confirmation on micro tier (needs far more than R=12 given
> how close binary-pass bounds run at this n — probably 50-100+ reps); (2) the task-076 outlier under
> a2 (0.36 vs ~0.13-0.17 elsewhere) deserves a closer look, not a claim; (3) reachable tier likely
> needs a qualitatively different intervention than anything tried tonight (both extra scale and
> extra inference-time compute gave the same modest, insufficient lift) — possibly decomposition-
> granularity or exemplar-based mitigations, unexplored this session; (4) the same context-length bug
> plausibly affected some of this doc's EARLIER (2026-07-23) compiled-mode numbers too, not just
> tonight's native-mode work — flagged, not re-litigated; (5) reconciling this work with the actual
> (currently stale, 1-commit) `badmodel` git branch is a deliberate, separate git-hygiene task, not
> done tonight to avoid mixing branch surgery with an unattended live run.

> ## ⚠️ STATUS UPDATE (2026-07-23) — the lab is BUILT and RUN; this doc's "next steps" are DONE
>
> The design-agent text below was written from the pre-execution state and is kept as design rationale.
> Since then the lab was fully implemented and executed. **Current source of truth: `README.md` +
> `PLAYBOOK.md` + the `results/`/`gallery/` artifacts**, not the "What's NOT done" list below.
>
> **Built & run:**
> - Harness: `run_cell.sh`, `run_matrix.sh`, `analyze.py`, `make_report.py`; mitigation `profiles/m0..m4`.
> - Micro tier authored & live-verified: `idea_tests/test_m01/m02/m03` (obscure single-page facts).
>   *(Not `test_045` — the lab uses its own m01–m03.)* Reachable = curated IDs 062/064/069/070/072/076/078.
> - Telemetry `task_id` gap: **CLOSED** (`json_telemetry` stamped via `runner.run_complete_test`).
>   `keystone_score` is recovered in `analyze.py` from `grep_validations` (no schema change needed).
> - `make_report.py` is **self-contained** (matplotlib + the spec's validated hexes), not a `plot_style` fork;
>   emits `gallery/` (5 figs, hero + 3×2 grid, light+dark) + `results/cells_long.csv`.
> - Fixes: single-leaf grounding passthrough; matrix resident-check silent-skip; **run_id/tier attribution bug**
>   (run_id now encodes tier; analyze derives tier from task_id).
>
> **Results (R=3, all local, $0):** micro tier — gemma2:2b (thin) & llama3.2:3b (react) & qwen2.5:1.5b clear
> the 0.75 bar; JSON is NOT the wall (97–100% valid_json every model). Reachable tier — a real wall, **nobody
> clears 0.75, ceiling qwen2.5:7b = 57%**; thin-leaf flips to winning for the stronger small models.
>
> **The actual open next step:** the **format-stress tier** (multi-field tool schemas) — the only place the
> "can't make JSON" wall is expected to appear, since micro/reachable schemas were simple enough that every
> model emitted valid JSON.
>
> ## ⚠️ STATUS UPDATE (2026-07-23, later) — methodology audit + forward-plan docs added (no LLM run)
>
> A soundness pass (two sonnet audits + code map) added three artifacts and patched the analyzer. **Read
> these before the next run:**
> - **`METHODOLOGY.md`** — the audit. Key corrections: (1) `valid_json` measures *json.loads success, not
>   schema compliance* — the 97–100% figure is a measurement artifact, so the JSON-wall thesis is untested,
>   not disproven; (2) at n=9 even 9/9 has a 95% Wilson lower bound of 0.70 < 0.75 — **the existence proof is
>   unconfirmable at R=3**, needs n≥12; (3) feasibility was selected on `overall_score`, hiding one feasible
>   cell (qwen2.5:1.5b/micro/m0); (4) the honest grounding gate is `visits>0` (URL-citation regex is a text
>   proxy — 2 zero-visit passes were marked grounded); (5) Fig 3 has no story (fenced/prose/refusal/empty = 0).
> - **`FORMAT_STRESS_TIER.md`** — the design for the open next step, with pre-registered predictions +
>   falsification. Two coupled changes: an additive `structured_json` aggregation (multi-field, number+bool
>   typed) AND a schema-aware classifier (else the metric stays blind). Micro extraction held constant so the
>   new failures are *format*, not *fact*. Ladder: fs0 unaided → fs1 grammar-constrained → fs2 thin-assemble.
> - **`schema_check.py`** — the linchpin classifier from FORMAT_STRESS_TIER §1b, built + unit-tested (13/13,
>   no LLM). Ready to wire into `json_telemetry.record(..., schema_ok=…)` and `analyze.py` at build time.
> - **`analyze.py` patched** (verified read-only): honest `visits>0` gate (`hks%`), 95% Wilson lower bound
>   (`ksLo`), feasibility computed over *all* profiles, `grounding_pass`/`honest_pass` CSV columns. No feasible
>   cell is CONFIRMED — all show `Lo≈0.70`, encoding the n=9 ceiling honestly.
>
> Ordered next steps now live in `FORMAT_STRESS_TIER.md §6` (static build → static classifier review →
> one live run at R≥12, last). Figures still need the Fig 3 recaption + honest-gate/interval marks per
> `METHODOLOGY.md §5/§7`.

*Original design-agent handoff (pre-execution) follows — kept for the rationale, palette validation, and
schema decisions. Treat its "What's NOT done / next steps" as historical.*

## What this is

An addendum to the proven webRAG/Euglena cost-recovery study. **Goal (clarified by the user this session):
demonstrate SOME fully-local working agent** — a model running entirely on local hardware (ollama, 0.5–3B)
that completes an agentic web-research task, reproducibly — and produce a tasteful, LinkedIn-ready
data-viz package around it. It is an **existence proof**, not a cost-Pareto win against the cloud.

Main-project context: `../HANDOFF.md`, `../README.md`, `../linkedin_package_38tests_2026-07-08/` (the proven
gallery + house style). The compiled-scaffold thesis (cheap model executes an expensive-model-authored DAG,
recovers premium accuracy) is already proven; this pushes it to the capability floor.

## Deliverables produced this session (both in `badmodel-lab/`)

1. **`CHART_SPEC.md`** — the rigorous chart spec a generator will fill in later. Contains: 5 figures
   (mitigation-lift ladder, cost/accuracy Pareto, parse-failure stack, merged recovery curve, feasibility
   frontier), each with a **3×2 grid + 1×1 hero** layout; a **validated** accessible palette (every hex run
   through the `dataviz` skill's `validate_palette.js` — transcripts quoted in §1.4); LinkedIn export specs
   (1200²/3840² feed, 1080×1350 4:5 carousel PDF, dual light/dark surfaces); the **required long-format data
   schema** (`cells.csv` one row per model×mitigation×task, + `runs.csv`); and the generator design.
   §0.1 folds in the fully-local reframe (feasibility frontier is the hero, USD axis demoted to a latency axis).
2. **`DEMO_PREP.md`** — a look-don't-run checklist: prerequisites, the curated demo cell matrix, task
   selection, config gotchas, what to inspect in results, pre-registered success criteria, artifacts to
   produce, and open decisions.

## Key findings that shaped the specs (verified against code this session)

- **THE critical gotcha:** `IDEA_TEST_COMPILED_LEAF_MODE` defaults to `auto`
  (`agent/app/testing/execution_compiled.py::_leaf_mode_for_model`, L115–133), and `auto` gives
  cheap/unknown-price models the **`react` JSON leaf** — the exact format a local 3B can't emit. The demo
  **must hard-set `thin`**. Thin-leaf (harness owns control flow, LLM only perceives/extracts) is the unlock.
- **Local inference is already wired** — `llm_backends.py` (L144–156) accepts `LLM_PROVIDER=ollama|local` over
  `MODEL_API_URL` (e.g. `http://localhost:11434/v1`). No harness change needed to go local. Web search/fetch
  stay online (the environment); the DAG plan is compiled once offline by a big model → "compile once offline,
  run local forever."
- **Task tiers:** `micro` exists (`idea_tests/test_045_micro_extract.py`, level `micro`). **`reachable` is NOT
  a named level** — map it onto the simple-keystone tasks the memory doc marks cheap-executor-reliable
  (counts 072/078, argmax 062/064, entity 069/076, subset-sum 070); re-confirm each for 0.5–3B. Avoid the hard
  adaptive archetypes (`smoke8`) for the demo.
- **Parse-failure telemetry already exists:** `testing/json_telemetry.py`, env-gated by
  `IDEA_TEST_JSON_TELEMETRY=1`, emits **seven** classes (`valid_json, fenced_json, malformed_json,
  truncated_json, prose, refusal, empty` — one more than the brief's six) to `<run_id>_json_telemetry.jsonl`.
- **Feasibility bar = 0.75** (used verbatim in `scripts/cross_tier_analyze.py`, `adaptive_ab_analyze.py`,
  `LADDER_PREREGISTRATION.md`). Ceiling band = strong-tier reference mean±CI (`recovery_curve._reference_lines`).
- **House-style change we committed to:** stop using magma-as-categorical (it's a sequential ramp — forbidden
  for identity by the `dataviz` method); keep magma only for the 0..1 score heatmap; give identity/ordinal
  dimensions the validated `dataviz` hues. See `CHART_SPEC.md` §1.
- **For local, USD ≈ 0/run** → the meaningful cost axes are latency (`duration_seconds`, heavy-tailed), tokens,
  and hardware footprint. Schema already carries `latency_s_{mean,p50,p95}`, `tokens_mean`, `model_params_b`.

## Two harness gaps to fix before generating the real figures

1. `json_telemetry.record()` logs `model/phase/class` but **not `task_id` or the mitigation/arm**. Per-model
   parse composition works via the `run_id`→arm join; per-task needs `task_id` added (one line). Recommend
   adding `arm`/`mitigation` explicitly too. (`CHART_SPEC.md` §4.1.)
2. Only `validation.overall_score` lands in result JSON. The schema wants a separate `validation.keystone_score`
   so the discriminating metric is directly plottable; until then `analyze.py` falls back to overall (documented).

## What's NOT done / next steps (rough order)

1. **Decide the `reachable` tier**: reuse the curated simple-keystone task IDs vs. author explicit
   `level: "reachable"` tasks (reuse is faster/lower-risk). See `DEMO_PREP.md` §C/§H.
2. **Audit `agent/compiled_plans/*.json`** for the demo tasks; run the one-time offline compile
   (`scripts/compile_plans.py`, cloud, offline) for any missing plan.
3. **Pull ollama models** and confirm `MODEL_API_URL`/`LLM_PROVIDER` (A2/A3 in `DEMO_PREP.md`).
4. **Run the demo matrix** (the one live step): 3B tier × {m0 baseline, m1 thin-leaf} × {micro, reachable},
   R≥3, `IDEA_TEST_COMPILED_LEAF_MODE=thin`, `IDEA_TEST_JSON_TELEMETRY=1`. Inspect per §E.
5. **Build `badmodel-lab/make_report.py`** per `CHART_SPEC.md` §6: extend `plot_style.py` with the validated
   categorical/ordinal palettes (add `CAT_HUES_*`, `ORDINAL_MODEL_TIER_*`, `ORDINAL_MITIGATION_*`, status +
   a light/dark `mode` switch); author each figure as a `draw_<fig>(ax, df, *, hero, mode)` panel function so
   the 1×1 hero and 3×2 grid share one implementation; export PNG/SVG + a carousel PDF. $0, reads `cells.csv`.
6. **Have `analyze.py` emit `cells.csv` + `runs.csv`** with the exact schema in `CHART_SPEC.md` §4 (roll up the
   JSONL telemetry into the 7 `parse_*` shares).

## Conventions

- **Commit policy (project memory `webrag-commit-policy`):** webRAG commits carry **NO Claude authorship
  trailers** — no `Co-Authored-By`, no `Claude-Session`. Only commit/push when the user asks; branch first if
  on default.
- Relevant memory notes: `webrag-benchmark-test-design` (what discriminates for a cheap model; the
  simple-keystone wall), `webrag-commit-policy`.
- Source-of-truth harness docs: `agent/app/BENCHMARK_NATIVE.md`, `BENCHMARK_SUITE_50.md`,
  `scripts/LADDER_PREREGISTRATION.md`, `scripts/adaptive_ab_analyze.py` / `cross_tier_analyze.py` /
  `bench_common.py` / `recovery_curve.py`.
- Everything this session was design/prep; no code changed, nothing ran, no numbers were fabricated.
