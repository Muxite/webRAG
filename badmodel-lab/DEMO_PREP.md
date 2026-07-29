# Bad-Model Lab — Local Agent Demo: Prep Checklist (items to look at)

*Goal: demonstrate SOME fully-local working agent — a model running entirely on local hardware
(ollama, 0.5–3B) that completes an agentic web-research task, reproducibly. This is a **prep list**:
what to look at and decide **before** any live run. Nothing here is executed. Companion:
`CHART_SPEC.md` (the figures) — this doc is "what to point those figures at." All paths absolute-from-repo
`/home/muk/projects/webRAG`.*

---

## A. Prerequisites to verify (look, don't run)

| # | Item | Where to look | What "ready" means |
|---|---|---|---|
| A1 | Local inference path is wired | `services/agent/app/llm_backends.py` (L144–156) | `LLM_PROVIDER` accepts `ollama`/`local`/`openai_compatible`; OpenAI-compatible Chat Completions covers Ollama/vLLM/llama.cpp. **Already supported.** |
| A2 | Which local models are pulled | `ollama list` (inspect only) | Demo targets present: `llama3.2:3b`, `phi3:mini`, `gemma2:2b`. Diagnosis/contrast: `llama3.2:1b`, `qwen2.5:0.5b`, `tinyllama`. |
| A3 | Endpoint the harness will hit | env: `MODEL_API_URL` | Points at ollama's OpenAI shim, e.g. `http://localhost:11434/v1`; `LLM_PROVIDER=ollama`. |
| A4 | Non-local dependencies (be honest) | env: `SEARCH_API_KEY`, chroma on `:8001` | Web **search + page fetch are the environment**, not the model — they stay online. "Fully local" = the *model/inference* is local. State this plainly in the writeup. |
| A5 | Harness run contract | `HANDOFF.md` "How to run" | `PYTHONPATH=services:services/agent`, `IDEA_TEST_CONCURRENCY=1` (mandatory, shared connectors), chroma up. |

---

## B. The demo cell matrix (small and curated — decide this first)

A "cell" = `(model × mitigation × task)`. Keep the demo matrix tiny; the frontier chart only needs a few
honest cells. Proposed axis to look at and lock:

- **Models (rows).** Demo target = **3B tier** (`llama3.2:3b`, `phi3:mini`, `gemma2:2b`). Contrast row =
  `llama3.2:1b` + `qwen2.5:0.5b` (expected to fail — that's the "why the ladder" panel, not the demo).
- **Mitigations (cols).** Minimum viable contrast = **m0 baseline (react/JSON leaf)** vs **m1 thin-leaf
  (JSON-free)**. That single contrast *is* the story. Add m2 (thin+votes) / m3 (bigger token budget) only if
  m1 alone doesn't clear the bar.
- **Tasks.** Micro + reachable only — see §C. **Do not use `smoke8` (122/125/…/144)** for the demo: those
  are the *hard* adaptive archetypes; they belong in the diagnosis panel, not the existence proof.

---

## C. Task selection — which tasks to look at

| Tier | Status in repo | Task IDs to look at | Note |
|---|---|---|---|
| **micro** | **Exists** | `test_045_micro_extract.py` (level `micro`, difficulty 2/10: visit ONE page, extract ONE value, reward stop-early) | The cleanest existence-proof task. Check for siblings: `grep -l '"level": "micro"' services/agent/app/idea_tests/*.py`. |
| **reachable** | **Not a named level yet** — curate from existing simple-keystone tasks | single-COUNT: `072`, `078` · single-ARGMAX: `062`, `064` · threshold-ENTITY: `069`, `076` · subset-sum-to-one-total: `070` | These are the keystones the memory doc marks *cheap-executor-reliable* (compiled 0.80–1.00). **Caveat: validated on nano, not on 0.5–3B local** — re-confirm each is truly simple before trusting it as "reachable." |
| **hard** | Exists (the 38-task suite + adaptive archetypes) | `smoke8`, `suite50` (`scripts/adaptive_ladder_run.py` `TASK_SETS`) | Diagnosis/ceiling context only, not the demo. |

Also look at: `services/agent/app/BENCHMARK_SUITE_50.md` and `BENCHMARK_SUITE_64.md` for the vetted task-set
definitions before finalizing the reachable list.

**The memory-doc wall to respect when picking tasks:** cheap/local executors reliably produce a *single*
count / argmax / threshold-entity, but **cannot** reliably produce k-th ordinal, exact subset membership, or
closest-to/argmin. Pick keystones from the first group only.

---

## D. Config gotchas to look at (these decide pass/fail before the model does)

1. **`IDEA_TEST_COMPILED_LEAF_MODE=thin` — hard-set it.** Look at `execution_compiled.py::_leaf_mode_for_model`
   (L115–133): default is `auto`, which resolves to **`react` (the JSON path) for cheap/unknown-price models**
   — i.e. local models get the exact format they can't emit. The demo **must override to `thin`**, or it will
   fail for the wrong reason. This is the single most important prep item.
2. **Compiled scaffold path.** `IDEA_TEST_EXECUTION_VARIANTS=graph_compiled` + a compiled plan must exist for
   each demo task. Look at `services/agent/compiled_plans/*.json` (hashed by mandate). If a demo task has no
   plan, it must be authored offline via `scripts/compile_plans.py` — **this is the one cloud/offline step**
   ("compile once offline, run local forever"). Decide *when* to run it; it is prep, not the demo.
3. **Grounding gate.** Keystone credit requires `observability.visit.count > 0` (`BENCHMARK_NATIVE.md`
   "Scoring integrity"): an ungrounded parametric guess scores ~0. Good — it keeps the demo honest — but it
   means a local model that never visits a page scores 0 regardless of the answer. Watch `visit.count` in
   results.
4. **JSON telemetry ON.** `IDEA_TEST_JSON_TELEMETRY=1` → writes `<run_id>_json_telemetry.jsonl`. This powers
   the before/after parse-health panel (the "thin-leaf converted refusal/prose → valid output" story). Cheap
   to enable, high narrative value. (Note: it logs `model/phase/class` but not `task_id`/arm — fine for the
   per-model panel; see `CHART_SPEC.md` §4.2 gap.)
5. **Latency expectations.** Local 3B on CPU can be minutes/run; on a consumer GPU, seconds. Decide the
   hardware and set `DEFAULT_TIMEOUT` accordingly. Latency (`duration_seconds`) is the *real* local cost axis,
   not USD (≈ 0 locally) — see `CHART_SPEC.md` §0.1.
6. **Replication.** R ≥ 3 for the demo (R=5 if time allows) — a single lucky run is not a demonstration
   (the whole project's "confirm at R=3, R=1 wins dissolve" lesson).

---

## E. What to inspect in the results (the "items to look at" after a run)

Per-run JSON lands in `services/agent/idea_test_results/<run_id>_<task>_<model>_<variant>_r*.json`. Fields to
read (via `scripts/gate_report.py` / `bench_common.load_row`, no live calls):

- `validation.overall_score` (and keystone, if emitted) — did it clear **0.75**?
- `execution.observability.visit.count` — **> 0** (did it actually go read a page — the grounding proof)?
- `execution.observability.cost.usd` — ≈ 0 for local (sanity-check the local path really ran local).
- `execution.duration_seconds` — the latency reality.
- `<run_id>_json_telemetry.jsonl` `class` distribution — `valid_json` share at m0 vs m1 (the mitigation working).
- The **trace**: search → visit → extract → answer, for one clean run — this becomes the demo's trace card.

Summarize with: `scripts/gate_report.py --run-id <id>` (per-cell score + USD table) and
`scripts/level_ladder.py` (per-level rollup) — both read-only.

---

## F. Success criteria (pre-register before running — honesty)

- **PRIMARY (existence proof).** ≥ 1 fully-local cell — a 3B model × thin-leaf × (micro or reachable) —
  with **mean keystone ≥ 0.75 at R ≥ 3, grounded (visit.count > 0 on the scoring runs)**. One honest cell
  clears the whole bar for "a local agent works."
- **SECONDARY (the mechanism).** m1 thin-leaf > m0 baseline for the same model, AND the parse-telemetry shows
  `valid_json` (or successful thin extraction) rising from m0 → m1. Report regressions too.
- **CONTEXT (not required to "win").** Where the local cell lands vs the cloud ceiling on accuracy; latency and
  tokens/run as the local cost.
- **Report even if it fails:** if no 3B cell clears micro, that's a real finding (the floor is higher than
  hoped) — the parse-health panel then diagnoses *why* (format vs reasoning vs grounding).

---

## G. Artifacts to prepare to produce (from CHART_SPEC.md)

Order them for the "it works locally" narrative:

1. **Trace card** (new, add to spec if wanted): one clean run — local model, task, search→visit→extract→
   correct answer, "R/R correct." The most convincing single artifact.
2. **Fig 3 — parse-health before/after** (m0 vs m1 per model): "we removed the JSON demand and the failures
   became valid actions."
3. **Fig 1 — mitigation lift**: score vs mitigation rung, feasibility bar, per 3B model.
4. **Fig 5 — feasibility frontier**: the model × mitigation grid with ✓ on cells that clear the bar — "the
   smallest local model + cheapest mitigation that works."
5. **Fig 2 (latency variant) / Fig 4**: supporting cost-in-seconds and merged-recovery context.

---

## H. Open decisions / risks to settle in prep

- **Reachable tier is not authored as a level** — decide: reuse the curated simple-keystone task IDs (§C) as
  "reachable," or author explicit `level: "reachable"` tasks. Reusing is faster and lower-risk for a demo.
- **Compiled plans for demo tasks** may not all exist — audit `compiled_plans/` and decide the one-time
  offline compile.
- **"Fully local" scope** — confirm the framing: local *inference* + live web + a plan compiled once offline.
  If the ask is stricter (local model authors its own plan), that's the native self-planning path, where
  0.5–3B models hit the format wall — much higher risk, likely not the demo.
- **`_price_tier` for local models** — look at `execution_compiled.py::_price_tier` (L~238): unknown local
  models may classify as `cheap` → default `react`. Reinforces item D1 (hard-set `thin`).
- **Which hardware** the demo runs on (sets latency and whether 3B is seconds or minutes).
