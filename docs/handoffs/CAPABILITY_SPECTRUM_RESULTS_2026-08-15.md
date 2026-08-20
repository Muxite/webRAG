# Results: capability-spectrum sweep, DAG v2 vs off-the-shelf LangGraph (2026-08-15)

Companion to `CAPABILITY_SPECTRUM_PREREG_2026-08-15.md` (design, written before any cell ran) and
`BENCHMARK_POLICY_HANDOFF_2026-08-15.md` (the open policy questions this partly answers).

**Spend: $0.42 of $5 authorized** at time of writing. ~400 live cells across 8 models, 3–5 arms,
10 tasks. Local models cost $0.

---

## The one-paragraph version

DAG v2 has one clearly demonstrated edge, and it is **availability, not accuracy**: 4 of the 8
models tested cannot be run by `langgraph.prebuilt.create_react_agent` *at all*, because they
expose no tool-calling endpoint — verified on two independent serving stacks. On **accuracy**,
restricted to models both systems can run, DAG v2's `good_adaptive` scores 0.560 against
LangGraph's 0.497 and this repo's bare `graph:baseline`'s 0.498 — i.e. the third-party ReAct loop
**ties our own unadapted scaffold**, and the adaptive machinery adds a modest amount on top, at
roughly 2.5× the cost. Arm effects beyond that are **model-specific and swamped by task-level
variance**; see the retraction below.

> ### ⚠ Retraction, recorded because it is the most important lesson here
>
> Mid-run, at n = 5–6 per cell, this document's earlier draft reported a clean thesis-supporting
> result: adaptive lift **monotonically decreasing in model capability** (+0.141 / +0.037 / +0.032).
> **That did not survive to n = 10.** The completed easy-wave numbers are +0.094 / **−0.174** /
> +0.208 — no capability ordering at all.
>
> The entire nano reversal is **one task**: on 049, baseline scored 1.000 and adaptive 0.000, a
> −1.000 delta that moves a 5-task mean by −0.200. Every other nano task sits within ±0.10.
> Root cause: both adaptive cells visited *both* required pages (7–8 visits) and missed the
> keystone value, and the keystone gate withholds **all** credit
> (`[x] visited_both: 0.0 Keystone absent -> visit credit withheld`).
>
> **Generalisable lesson:** keystone gates make per-task scores near-**binary**. A run that gathers
> all the right evidence and misses one value scores 0, not 0.6. Means over 5 tasks are therefore
> dominated by whichever tasks happened to cliff, and no amount of *reps* fixes it — only more
> *tasks*. This is independent quantitative support for the pre-registration's "power comes from
> task count, not reps", and it is exactly the error mode this project's own cautionary tale
> (the 0.786 claim) warns about. I reproduced it, at n=6, in under an hour.

---

## Finding 1 — the off-the-shelf arm cannot run half the cheap market

`create_react_agent` binds tools through the OpenAI **function-calling API**. A model without a
tool-calling endpoint is rejected before inference. The native engine asks for JSON *as text* and
parses it itself, so it can attempt any model that can emit characters.

| model | serving stack | LangGraph | native engine |
|---|---|---|---|
| `tinyllama:latest` | ollama | **HTTP 400** `does not support tools` | runs |
| `phi3:mini` | ollama | **HTTP 400** `does not support tools` | runs |
| `gemma2:2b` | ollama | **HTTP 400** (probe only) | runs |
| `meta-llama/llama-3.2-1b-instruct` | **OpenRouter** | **HTTP 404** `No endpoints found that support tool use` | runs |
| `qwen2.5:0.5b`, `llama3.2:1b`, `qwen2.5:1.5b`, `llama3.2:3b`, `qwen2.5:7b`, `llama3.1:8b`, `qwen2.5:14b` | ollama | runs | runs |
| `openai/gpt-4.1-nano`, `google/gemini-2.5-flash-lite` | OpenRouter | runs | runs |

**Why this is more than a packaging quirk:** it reproduces on two independent serving stacks. If
only ollama refused, the finding would be "ollama's template registry is incomplete." OpenRouter
refuses the same class of model for a different reason — no *provider* serving that model exposes
tool use. The tool-calling API surface is simply not universal at the cheap end of the market,
which is exactly the market this project targets.

**Scope discipline — what this does NOT claim.** It does not claim these models are incapable of
agentic work. A hand-written text-parsing ReAct loop could drive `gemma2:2b`. The claim is exactly:
*the off-the-shelf path anyone can `pip install` cannot run them; the native engine can.* Whether
it runs them **usefully** is Finding 3, and there the answer is mostly no.

**A second, distinct failure mode showed up unprompted.** `qwen2.5:0.5b` *can* emit tool calls —
and under LangGraph it made 24 visits and 50 LLM calls before returning
`"Sorry, need more steps to process this request."` Tool-calling capability is necessary, not
sufficient. Any headline must separate **cannot start** from **starts and thrashes**.

*Recorded judgement call:* the 400/404 is **not** tagged `infra_failed`, so it scores as a genuine
0 rather than being quarantined. That is defensible — "this framework cannot run this model" is a
capability outcome, not a transient provider blip — but it materially shapes the headline and
belongs in the writeup, not buried in a flag.

---

## Finding 2 — the suite's difficulty calibration hides the project's central claim

Difficulty distribution, parsed from task docstring headers:

| difficulty | 1–4 | 5–6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| **active-59** | **1** | 0 | 1 | 11 | 26 | 19 |
| whole pool (163) | 19 | 14 | 16 | 21 | 54 | 37 |

**56 of 59 active tasks are difficulty ≥8.** The suite is well-calibrated for the `gpt-4.1-nano`
tier it was built against. For weak models it is entirely out of range, so every arm scores ~0 and
the comparison carries no information. The pool contains 19 in-range tasks; the active set uses one.

**The floor effect itself is real and large.** Same models, same arms, different task difficulty
(final n = 10 per cell):

| tier | **in-range wave** (diff 3–6) | | | **hard wave** (diff 9–10) | | |
|---|---|---|---|---|---|---|
| | baseline | good_adaptive | lift | baseline | good_adaptive | lift |
| API 1B (`$0.027/$0.201`) | 0.327 | 0.421 | +0.094 | 0.090 | 0.045 | −0.045 |
| API nano (`$0.10/$0.40`) | 0.661 | 0.487 | −0.174¹ | 0.605 | 0.617 | +0.012 |
| API flash-lite (`$0.10/$0.40`) | 0.429 | 0.637 | +0.208 | 0.480 | 0.665 | +0.185 |

¹ entirely attributable to task 049 — see the retraction above.

What this **does** establish: the weakest model goes from 0.090 (hard) to 0.327 (in-range) on
baseline alone, and its zero-score rate drops correspondingly. The hard suite genuinely cannot
measure weak models — every arm floors and the comparison carries no information.

What this **does not** establish: any capability-ordered pattern in the adaptive lift. The
between-model differences (+0.094 / −0.174 / +0.208) are larger than any trend and are driven by
individual task cliffs. **At this task count, arm ranking is not resolvable.** More tasks, not
more reps, is the fix.

The in-range wave used tasks **012, 021, 022, 023, 049**. Absolute numbers there are provisional
(see Finding 5), but the cross-arm comparison within a wave is sound: every arm faces identical
validators.

---

## Finding 2b — accuracy head-to-head needs both framings, or it lies

| arm | as deployed across the roster | restricted to models LangGraph *can* run |
|---|---|---|
| `graph:baseline` | 0.386 (n=68) | **0.498** (n=44) |
| `graph:good_adaptive` | **0.428** (n=65) | **0.560** (n=41) |
| `langgraph` | 0.323 (n=40) | **0.497** (n=26) |

Paired on identical (model, task) pairs: `good_adaptive` 0.317 vs `langgraph` 0.284 —
**8 wins / 8 ties / 6 losses**. A coin flip with a slight lean.

Where LangGraph can run, it **ties `graph:baseline` almost exactly** (0.497 vs 0.498): a
third-party ReAct loop matches this repo's bare graph scaffold. The adaptive machinery adds
+0.063 on top. The roster-wide gap (0.428 vs 0.323) is therefore driven almost entirely by the
**structural zeros** from models LangGraph cannot start.

Publishing only the first column overstates a quality win that is really an availability win.
Publishing only the second erases the availability win. **Both columns, or neither.**

On the strongest local model the result is openly unfavourable: `qwen2.5:7b` scored
**LangGraph 0.694 vs graph 0.150–0.158**, with LangGraph opening more pages. That is a real loss
and should be reported as one.

---

## Finding 2c — the barrier is the tool-calling API, not ReAct (wave 3)

`sequential_react` — this repo's own **text-prompted** ReAct loop — runs
`meta-llama/llama-3.2-1b-instruct` fine: 26 real LLM calls, 6–12 search documents per cell. The
same model, the same tasks, under LangGraph's **function-calling** ReAct returns
`404 No endpoints found that support tool use` and never starts.

Same orchestration pattern, opposite outcome. The exclusion in Finding 1 is caused by the
**tool-calling API surface**, not by ReAct as a strategy. This is the honest framing of the
availability claim, and it is stronger for being narrower.

(It searched but never *visited*, so it still scored 0 on the hard wave — participation is not
competence.)

---

## Finding 2d — a linear loop from this same codebase matches the graph engine at ⅓ the input cost

Wave 3 ran `sequential_react` — this repo's own **linear** ReAct loop — over the same 10 tasks and
3 API models. Paired on identical (model, task) pairs, excluding `infra_failed`:

| A | B | n | A mean | B mean | W / T / L |
|---|---|---|---|---|---|
| `seq_react` | `graph:good_adaptive` | 29 | 0.466 | 0.458 | 10 / 4 / **15** |
| `seq_react` | `graph:baseline` | 29 | 0.466 | 0.430 | 12 / 7 / 10 |
| `seq_react` | `langgraph` | 29 | 0.466 | 0.322 | **17** / 6 / 6 |
| `graph:good_adaptive` | `langgraph` | 40 | 0.376 | 0.291 | **19** / 12 / 9 |
| `graph:good_adaptive` | `graph:baseline` | 40 | 0.376 | 0.342 | **16** / 13 / 11 |

**A second unpaired-mean trap, caught.** The unpaired table made `seq_react` look like the best arm
outright (flash-lite 0.712 vs `good_adaptive`'s 0.651). Paired, `good_adaptive` wins 15 pairs to 10
with near-identical means — the apparent gap was **task-coverage mismatch** (seq_react n=14–18 vs
20), not quality. Same failure mode as the retraction above, different surface. Any arm comparison
in this repo should be paired by default; the unpaired mean has now misled twice in one session.

**Where the two genuinely differ is economics** (API models only):

| arm | prompt tok | completion tok | visits | score |
|---|---|---|---|---|
| `seq_react` | **19,362** | **3,609** | 3.1 | 0.482 |
| `graph:good_adaptive` | 58,335 | 2,166 | 3.1 | 0.479 |
| `graph:baseline` | 22,962 | 1,238 | 1.3 | 0.432 |
| `langgraph` | 19,334 | 245 | 2.2 | 0.328 |

Same score, **same evidence volume (3.1 visits), one-third the input tokens** — and `seq_react` is
the only arm that spends most of its budget on *generation* rather than re-sent context. This is
Finding 3 with a concrete comparator attached: the graph engine's 3× context premium is not buying
score, and an arm already in this repo demonstrates that.

**The open question this raises is uncomfortable and belongs in the relaunch:** if a linear ReAct
loop matches the graph engine on score at a third of the cost with equal evidence, what is the
graph structure for on these task shapes? The candidate answers — chain depth, and the shapes the
suite under-represents — are exactly what Finding 2 says the current suite cannot resolve.

---

## Finding 3 — the token premium is context, not reasoning

Across all graph cells (same models, same tasks, both arms):

| arm | prompt tok | completion tok | ratio | real LLM calls | visits | score |
|---|---|---|---|---|---|---|
| `graph:baseline` | 27,425 | 1,379 | 19.9 | 3.8 | 0.9 | 0.321 |
| `graph:good_adaptive` | 55,828 | **1,637** | 34.1 | 15.2 | 2.0 | 0.379 |

**4× the LLM calls and 2× the input tokens buy +19% output tokens.** Reasoning happens in
*completion* tokens; the premium is overwhelmingly **re-sent accumulated context**. The slogan
"burn cheap tokens for better reasoning" describes something the telemetry does not show. What the
engine actually buys is more *decisions over the same evidence* — decision-stage counts go from
8.5/cell to 19.5/cell, with the growth in `action` (2.7→5.7), `selection` (1.0→4.4),
`expansion` (1.0→2.8) and `reexpand` (0→1.8).

At an input:output ratio of ~34:1, **prompt caching is the single largest cost lever available**,
and it requires no quality tradeoff. Nothing in the current path appears to use it.

**Iso-cost (Q1's suggested framing) — score per $0.01 spent:**

| tier | graph:baseline | graph:good_adaptive |
|---|---|---|
| API 1B | 1.45 | 0.72 |
| API nano | 1.36 | 0.69 |
| API flash-lite | 1.26 | 0.57 |

On a pure score-per-dollar basis the bare model wins about 2:1 at every tier. Any published table
that shows score without cost is telling half the story — and the flattering half.

---

## Finding 4 — a selection effect in the driver that would have corrupted the API axis

`cell_env()` sets `IDEA_TEST_PREFLIGHT_JSON=0` **only for local cells**, with an explicit comment
that a weak model failing JSON "is exactly what this barrage exists to observe, not silently drop."
API cells keep the gate **on**. The first launch of this sweep therefore dropped
`meta-llama/llama-3.2-1b-instruct` before inference:

```
[json-gate] JSON probe error: Expecting value: line 1 column 1 (char 0)
[FAILED] Structured-JSON pre-flight — model could not emit parseable json_mode output (dropped)
No valid execution models after pre-flight checks. Aborting.
```

The cell exited **rc=0 with no result JSON** — the documented silent-failure mode — and the driver
logged it as `ok`.

Consequences:
1. The two axes are **not symmetric**: a weak local model is measured, an equally weak cheap-API
   model is excluded before it runs.
2. Every prior "cheap API model" claim in this repo was made over a **JSON-gate-passing
   subpopulation**, never stated as such.
3. The gate is **non-deterministic** for a borderline model — task 122's cell passed it seconds
   before task 134's failed it. So the subpopulation is not even stable between cells.

All results in this document were produced with `IDEA_TEST_PREFLIGHT_JSON=0` exported for both
axes. `base_env()` copies `os.environ` and never overwrites the key, so a parent-shell export is
sufficient — no code change was needed.

**Recommendation:** make the gate's setting an explicit, logged, per-axis decision rather than an
implicit consequence of `provider` being set.

---

## Finding 5 — two Q8 items resolved, one correction to the handoff

**012 is not false-failing on grounding — it is penalizing a reasonable navigation strategy.**
A `gpt-4.1-nano` baseline cell visited 10 pages and returned 10 well-formed links with
descriptions, and scored 0.50. The reason: 9 of the 10 were *sister projects*
(wikidata, wiktionary, wikivoyage, wikibooks…) and only 1 was an `en.wikipedia.org` URL. An agent
that starts at Wikipedia's **Main Page** finds sister-project links most prominently. Whether that
is the intended discriminator is a task-design decision, not a validator bug.

**`observability["evidence"]` is empty in 105/105 persisted results — by design, not a bug.**
I initially read this as the evidence refactor failing live. It is not:
`runner.py:205-210` projects evidence into a **copy** of observability *for validation only*, and
leaves the persisted copy untouched. Validators did receive it. This also confirms why
`rescore_results.py` must refuse evidence-dependent tasks: `telemetry_raw`, the evidence source, is
popped from persisted results below `IDEA_TEST_REPORT_VERBOSITY=3`, so offline re-scoring genuinely
cannot reconstruct grounding.

**`search.count` confirmed to be a document count** — `testing/utils.py:78-87` increments once per
entry in `telemetry.documents_seen` with `source == "search"`, i.e. per returned *result*, not per
search *call*. `visit.count` is a true per-visit count and is the sound evidence metric.

**Correction to the handoff's Q5.** It proposes bootstrapping "the existing corpus at R=3 vs R=5"
and calls this "a free, data-backed answer." Reps are stored in the trailing `_r{n}` suffix, not
`_rep{n}_`. The corpus holds **116 files at r3, 9 at r4, 5 at r5** — enough for R=3, not for R=5.
The R=3 analysis that *is* possible (badmodel-lab `bmladapt` family, 7 model×task cells, 4 arms):

| resample | full arm ordering differs from R=3 | top arm differs |
|---|---|---|
| R=1 | 55.8% | 42.8% |
| R=2 | 36.9% | 37.4% |

**Read with care.** The top two arms in that corpus differ by **0.007** across **7 cells**. When
the true gap is that small, ordering is near-arbitrary at any rep count. This *supports* the
pre-registration's "power comes from task count, not reps" — and therefore supports R=5 → R=3,
provided the saved budget buys tasks. It does not support trusting any headline ordering built on
few tasks with small gaps, at any R.

---

## Finding 6 — an engine bug that silently zeroes weak-model runs (FIXED, with tests)

The local graph cells were reaching **zero visits**, which the grounding gate converts to a
guaranteed 0. Traced to a crash, not to model incompetence:

```
[EXPANSION] Exception during expansion: dictionary update sequence element #0 has length 1; 2 is required
  → agent/app/idea_policies/expansion.py:1335 in _parse_candidates
[EXPANSION] Policy returned 0 candidates → EXPANSION FAILED
[EXPANSION] Created fallback candidate: "Analyze and plan next steps..."      ← action=None
[GROUNDING-GATE] zero opened pages on a grounded-research mandate (stripped 3 citations)
[134] llama3.2:3b [graph]: FAILED (score: 0.00, 18.0s)
```

Root cause: `details = candidate.get("details") or {}` substitutes only on **falsy** values, so a
truthy-but-wrong-shaped `details` (a weak model emitting a list or string where an object belongs)
reached `dict(details)` and raised `ValueError`. `dict(["a"])` produces that exact message. The
exception escaped `_parse_candidates` and destroyed the **whole** expansion step — every sibling
candidate lost, engine falls back to an action-less node, no tool ever called, run scores 0.

**This is the same bug class already guarded 8 lines above for `meta`**, whose comment documents
the identical live failure. Fixed in one place, missed in the adjacent one.

Fix: mirror the `meta` guard (`if not isinstance(details, dict): details = {}`). New regression
test `agent/tests/expansion_malformed_details_test.py` (7 cases: the 5 malformed shapes, sibling
survival, and well-formed passthrough). It fails with 6 errors before the fix and passes after.
**Full offline suite: 4708 passed, 18 skipped** (inherited 4701 + 7 new).

Blast radius: any run whose model emits a malformed `details` — disproportionately weak/local
models, i.e. exactly the population the thesis is about. No infra flag, no driver-visible error;
the cell reports `ok` with a score of 0. **Local results in this document were collected before
the fix and understate the graph arms.**

---

## Method notes / caveats

- **Exploratory, not confirmatory.** n = 8–20 per (model, arm) on the API side, 1–2 on the local
  side. Powered for categorical and large effects only. **No p-values are quoted**, per the
  pre-registration.
- **Step budgets are not equalized** in the wave-1/2 comparison: graph arms get `max_steps` 50–90,
  LangGraph 25. Wave 3 re-runs LangGraph at `IDEA_TEST_LANGGRAPH_MAX_STEPS=60` to test this.
- **2 cells were `infra_failed`** and are excluded from every table. Note that
  `level_ladder`/`gate_report`/`recovery_curve` would *not* have excluded them.
- **Cost is recomputed from raw tokens** in `scripts/capspec_report.py`, because
  `observability.cost.usd` reports **$0** for any slug absent from `model_costs.MODEL_PRICING`
  (`meta-llama/llama-3.2-1b-instruct` is absent). A naive cost-per-point table would have shown the
  weakest model as infinitely cost-efficient.
- **`llm.calls` is ≈2× the logical call count** for every arm; the report divides by 2.
- Local models ran at `OLLAMA_CONTEXT_LENGTH=16384`; the engine's merge/finalize prompts can exceed
  that and be silently truncated, so a weak local score may be a context artifact.

## Reproducing

All three analysis scripts now live in `scripts/` (they were authored in scratch during the run):
`capspec_tool_probe.py` (the $0 tool-calling probe), `capspec_report.py` (the main tables), and
`capspec_chain_report.py` (the chain-shape paired analysis).

```bash
# capability probe ($0) — which models can do OpenAI tool-calling vs plain-text JSON
./.venv/bin/python3 scripts/capspec_tool_probe.py

# one axis, one variant (repeat per variant; run-ids must differ)
export IDEA_TEST_PREFLIGHT_JSON=0
PYTHONPATH=.:services:agent ./.venv/bin/python3 scripts/adaptive_ladder_run.py \
  --axis capspec_api --variant graph --arms baseline,good_adaptive \
  --run-id csapi_g --tasks "134,135,122,140,128" --no-ref --jobs 4 \
  --budget 2.5 --real-budget 3.0 --max-attempts 2

# reports (re-run any time; they re-read every result file on disk)
./.venv/bin/python3 scripts/capspec_report.py          # all shapes, all waves
./.venv/bin/python3 scripts/capspec_chain_report.py    # chain-shape paired analysis
```

---

# Part 2 — Open questions

Grouped by what they block. Questions marked **[decision]** need a human call; the rest are
answerable with work.

## A. Is the benchmark able to test the thesis at all?

1. **[decision]** The active-59 is 56/59 at difficulty ≥8. Was that deliberate calibration for the
   `gpt-4.1-nano` tier, or accumulated drift? It is well-suited to that tier and unusable below it.
2. **[decision]** Should the relaunch add an explicit in-range tier? If so: how many tasks, at what
   difficulty, and drawn from the 19 easy pool tasks or authored fresh with modern grounding gates?
3. If we add one, the claim becomes "structure helps weak models **on tasks within their range**."
   **[decision]** Is that the claim you want to publish, or does it read as a retreat from the
   stronger one?
4. Should **difficulty** become a reported stratification axis alongside shape? Nothing currently
   reports score-by-difficulty, which is how this went unnoticed.
5. 44/59 active tasks carry fan-out; ~10 are chains without it. If chains are where structure wins,
   is the suite mis-weighted ~4:1 against the very claim it exists to support?
6. Keystone gates make scores near-binary, so task-level variance dominates. **[decision]** Is that
   the intended scoring behaviour? A partial-credit path for "visited the right pages, missed the
   value" would cut variance enormously — but it also weakens the anti-hallucination guarantee.
   Which do you want?
7. Given (6), what is the **minimum task count** at which arm ranking is resolvable? Nobody has
   stated a target effect size. It should be stated before the relaunch, per the preregistration's
   own discipline.

## B. What is the LangGraph arm for?

8. **[decision]** Is "cannot run without a tool-calling endpoint" a headline result or a footnote?
   It is the most defensible finding here and also the one most likely to be called a strawman.
9. Wave 3 shows our own **text-prompted** ReAct runs the model LangGraph's function-calling ReAct
   cannot. Should the published comparison therefore be three-way (graph / our-ReAct /
   their-ReAct), so orchestration is separated from API surface?
10. Step budget: graph 50–90 vs LangGraph 25. Wave 3 re-runs LangGraph at 60. **If matching the
    budget closes the gap, does the score comparison become uninteresting** and the availability
    result the only real one?
11. Do smolagents / CrewAI share the same tool-calling requirement? If yes, adding one replicates
    Finding 1 for almost nothing and makes it a property of the ecosystem rather than of LangGraph.
12. On `qwen2.5:7b`, LangGraph beat the graph engine 0.694 vs 0.158. **[decision]** Does that get
    published as prominently as the wins?

## C. Where does DAG v2's cost actually go?

13. Prompt:completion is ~34:1. **Is prompt caching wired anywhere?** At that ratio it is the
    largest available cost lever and costs nothing in quality. If it isn't wired, why not?
14. `good_adaptive` makes 4× the LLM calls of baseline for +19% completion tokens and roughly 2×
    the visits. **[decision]** Is "more decisions over the same evidence" the intended trade?
15. Given (13)–(14), does **"burn cheap tokens for better reasoning"** need restating in README /
    ADAPTIVE_ENGINE.md / public material? The telemetry shows context re-sending, not reasoning.
16. Iso-cost, score per $0.01: baseline beats good_adaptive ~2:1 at every API tier. **[decision]**
    Should iso-cost be the headline table? It is the thesis's own framing and it is unflattering.

## D. Evidence volume

17. LangGraph averaged 4.3 visits where it could run; `good_adaptive` 3.7 and `baseline` 1.7. On
    `qwen2.5:0.5b` LangGraph made **24** visits. Is the native engine under-visiting by policy or
    by budget? Which knob binds first — `grounding_max_replans`, `max_steps`, `got_beam_max`?
18. The one task a linear agent clearly won earlier (022) was won by *looking more*. Should the
    raised-visit-budget arm run **before** any further planning work?
19. `sequential_react` on `llama-3.2-1b` searched 6–12 documents and visited **zero** pages. Why
    does search succeed and visit never fire for weak models? That looks like a second, separate
    bug of the same family as Finding 6.

## E. Strategy

20. **[decision]** Is the goal to beat LangGraph on quality, or to serve the models it cannot run?
    Those are different roadmaps: one is engine tuning, the other is breadth of model support plus
    reliability engineering.
21. **[decision]** If the honest summary is *"ties on accuracy where both run, costs ~2.5× more,
    runs models nothing off-the-shelf can run"* — is that a product? It is a real and unusual niche.
22. **[decision]** Q7 of the policy handoff is still unanswered: **what result would falsify the
    thesis?** State it before the relaunch.

## F. Integrity items

23. Should `IDEA_TEST_PREFLIGHT_JSON` be an explicit, logged, per-axis decision rather than an
    implicit consequence of `provider` being set? (Finding 4)
24. Should the headline pipeline quarantine `infra_failed`? 3 cells here; `level_ladder`,
    `gate_report` and `recovery_curve` would all have counted them at face value.
25. `llm.calls` is ≈2× the logical count and `search.count` is a document count. **[decision]** Fix
    the metrics or rename them? Any external reader will misread both.
26. Task 012 penalises an agent that starts at Wikipedia's Main Page (sister-project links).
    **[decision]** Intended discriminator, or fix the task?
27. Should `_parse_candidates` fail **loudly** — surfacing a driver-visible error or an
    `infra_failed`-style flag — instead of degrading to a silent action-less fallback? Finding 6
    was invisible to every layer of accounting above the engine.

---

# Part 3 — Future experiments, ranked by expected value

**1. Prompt-caching ablation.** Highest EV, zero quality risk. At 34:1 input:output, caching the
stable prompt prefix could cut cost 50–80%. Measure: same tasks, same arms, caching on/off, compare
$/cell and score. If it lands, the whole cost objection to `good_adaptive` weakens at a stroke.

**2. Re-run the local matrix after the Finding 6 fix.** All local numbers here predate it, and the
bug disproportionately hit weak models — the thesis's own population. Free (GPU only). This is the
cheapest way to find out whether the local story changes entirely.

**3. Constrained decoding for action selection.** The irony in the data: LangGraph's schema-enforced
function calling drove `qwen2.5:0.5b` to 24 visits, while the native engine's free-text JSON left
weak models emitting unparseable plans and doing nothing. Ollama and most APIs support
JSON-schema-constrained output. Hypothesis: constraining *just the action decision* captures
LangGraph's reliability while keeping the engine's ability to run any model. **This is the single
most promising engine change suggested by this sweep.**

**4. Difficulty ladder.** One model, tasks spanning difficulty 2→10, both arms. Finds where the
adaptive benefit peaks and gives the relaunch a principled basis for task selection instead of an
inherited one. Directly follows Finding 2.

**5. Raised visit budget.** The standing #1 candidate, now with a concrete comparator: LangGraph
gets 4.3 visits, `good_adaptive` 3.7, `baseline` 1.7. Test whether score tracks visits.

**6. Task-count power curve.** Bootstrap arm ranking stability against task count (10 / 20 / 40 /
80) using the pooled corpus. Answers "how many tasks do we need" with data rather than assertion,
and prices the R=5→R=3 trade properly.

**7. Three-way orchestration comparison** (graph / `sequential_react` / `langgraph_react`) on
models all three can run. Separates "graph vs linear" from "our code vs theirs" — currently
confounded.

**8. `auto_parallel_siblings: false` on chain tasks.** The standing #2 candidate; unchanged by this
sweep but still untested.

**9. Dead-weight ablation.** Remove mechanisms recorded as inert (step-confidence judge at AUC
0.571, unreachable confidence early-exit, backtrack needing ≥5-node chains) and measure the cost
delta. If the premium is planning overhead, this is free savings.

**10. Local context-length sweep.** `OLLAMA_CONTEXT_LENGTH=16384` may be silently truncating
merge/finalize prompts. Re-run `qwen2.5:7b` at 32768 and compare — cheap, and it would invalidate
some local conclusions if it matters.

---

# Addendum (wave 3) — questions and experiments this changes

Finding 2d lands after Part 2 was written. It does not invalidate anything above, but it
re-prioritises:

**New questions, both [decision]:**

28. **If `sequential_react` matches `graph:good_adaptive` on score at ⅓ the input cost with equal
    evidence volume, what is the graph structure buying on these shapes?** The honest candidate
    answers are chain depth and the shapes the suite under-represents (Finding 2) — which is to say
    the current suite cannot resolve it. This is arguably the single most important question the
    relaunch has to answer, and it is *internal*: it does not involve LangGraph at all.
29. Should `sequential_react` be promoted from "premium reference bar" to a **first-class ladder
    arm**? It is currently framed as a ceiling to compare against, but on this evidence it is a
    competitive, cheaper alternative to the graph engine on the cheap-API tier.

**Method change, not optional:** every arm comparison in this repo should be **paired by
(model, task)** by default. The unpaired mean pointed the wrong way twice in one session — once via
a single task cliff (the retraction), once via task-coverage mismatch (Finding 2d). Neither was
subtle in hindsight and neither was visible in the aggregate.

**Experiment re-ranking.** Insert above the previous EV-4:

- **New EV-4 — graph-vs-linear on chain shapes specifically.** `seq_react` vs
  `graph:good_adaptive`, paired, on the ~10 chain-without-fan-out tasks only. If the graph engine's
  advantage is real it should appear there and nowhere else. If it does not appear there either,
  that is a genuine and publishable negative result about the architecture, and it is cheap to get.

The previously listed EV-1 (prompt caching), EV-2 (re-run local after the Finding 6 fix) and EV-3
(constrained decoding) are unchanged and still lead — and EV-1 is now more attractive, since
Finding 2d shows the context premium is not buying score.

---

# Addendum 2 (wave 3, `lg60`, RUN-COMPLETE) — the step budget is not the confound, and more compute buys nothing for anyone

The pre-registration listed unequal step budgets (graph `max_steps` 50–90 vs LangGraph 25) as
confound #1 and explicitly did not resolve it. Wave 3 re-ran the LangGraph arm at
`IDEA_TEST_LANGGRAPH_MAX_STEPS=60`, now complete at 60/60 cells. Paired by (model, task):

| A | B | n | A mean | B mean | W / T / L |
|---|---|---|---|---|---|
| `langgraph@60` | `langgraph@25` | 30 | 0.344 | 0.328 | 8 / 16 / 6 |
| `graph:good_adaptive` | `langgraph@60` | 30 | 0.472 | 0.344 | **15** / 9 / 6 |
| `seq_react` | `graph:good_adaptive` | 30 | 0.492 | 0.472 | 12 / 4 / **14** |
| `seq_react` | `graph:baseline` | 30 | 0.492 | 0.432 | **13** / 7 / 10 |
| `graph:good_adaptive` | `graph:baseline` | 41 | 0.373 | 0.339 | **16** / 14 / 11 |
| `graph:good_adaptive` | `langgraph` | 41 | 0.373 | 0.284 | **20** / 12 / 9 |

**More than doubling the step budget moves LangGraph +0.016 (8W/16T/6L).** The arm was not
step-starved; the confound the pre-registration flagged is not what produces the gap. Q1's "unit of
fairness" stays open in general, but on this axis at these budgets it is not load-bearing. DAG v2's
advantage over the off-the-shelf arm survives its own fairness check (15/9/6 against the
budget-matched version).

### The new result: more compute buys nothing, for the third-party agent either

Run-complete per-arm economics (API models):

| arm | prompt tok | completion tok | visits | score |
|---|---|---|---|---|
| `seq_react` | **19,255** | **3,374** | 3.1 | **0.516** |
| `graph:good_adaptive` | 58,335 | 2,166 | 3.1 | 0.479 |
| `graph:baseline` | 22,962 | 1,238 | 1.3 | 0.432 |
| `langgraph@60` | **94,935** | 269 | 3.2 | 0.344 |
| `langgraph@25` | 19,334 | 245 | 2.2 | 0.328 |

Raising LangGraph's budget made it burn **4.9× the prompt tokens** and open 45% more pages for
**+0.016 score**. That is the same diminishing-return-on-context pattern Finding 3 identified in
DAG v2 — reproduced in a completely independent third-party agent, on the same tasks and models.

**This reframes Finding 3.** The context premium buying almost no score is not a defect specific to
this architecture; it looks like a property of the **task/model regime**. Two independently-written
agents both spend 3–5× more context for a rounding error in score. The implication for the roadmap
is that context-spend is the wrong lever *in general* here, and the arm that wins
(`seq_react`: highest score, lowest prompt tokens, most completion tokens) is the one that spends
its budget on **generation** instead.

### A third unpaired-mean trap, recorded — and its correction

At 29/60 complete, the `langgraph@60` per-arm row read 0.9 visits and 3,805 prompt tokens, which
looked like a behavioural regression. It was completion-order skew: `meta-llama/llama-3.2-1b`
**404s instantly** (0 tokens, 0 visits) so those cells finish first and dominate a partial average.
Run-complete, the true figures are 3.2 visits and 94,935 prompt tokens — the *opposite* direction.

That is three times in one session that an unpaired or partial mean pointed the wrong way (the
task-049 cliff, the seq_react coverage mismatch, and this). Treat
**paired-by-(model, task), run-complete** as the default contract for every arm comparison in this
repo — see question 28's method note.

---

# Addendum 3 — the chain-shape test: the adaptive lift is real, and it is shape-specific

Addendum 1 raised the internal question: if `sequential_react` matches `graph:good_adaptive` at ⅓
the input cost, what is the graph structure buying? The mechanistic answer, if there is one, is
**chains** — the one shape where a later step must condition on an earlier step's result, which is
what re-expansion and re-grounding provide. Fan-out does not need it.

Tested directly: 9 chain-without-fan-out tasks (`135, 136, 137, 138, 139, 046, 047, 065, 093`,
plus existing `134` data), 3 API models, R=1 — deliberately **more tasks at R=1** rather than fewer
at R=2, per this session's own finding that task count is what resolves arm ranking. Graph stage
complete at 54/54 cells.

| comparison | n (paired) | A | B | W / T / L |
|---|---|---|---|---|
| `good_adaptive` vs `baseline` — **chains only** | 33 | **0.356** | 0.233 | **18 / 8 / 7** |
| `good_adaptive` vs `baseline` — all shapes | 41 | 0.373 | 0.339 | 16 / 14 / 11 |

**On chains the adaptive lift is +0.123 (18W/8T/7L). Across all shapes it is +0.034 (16W/14T/11L)
— a wash.** The mechanism is roughly 4× stronger on the shape where it has a reason to work.

This is the first result in this sweep that **predicts** rather than post-hoc explains, and it is
the clearest answer to "is there anywhere we have an edge": **yes — chain shapes, on the internal
ladder (`good_adaptive` over `baseline`).**

### It also sharpens Finding 2's suite critique into a concrete number

The active-59 contains roughly 10 chain-without-fan-out tasks against 44 carrying fan-out. A real
+0.123 effect on ~17% of the suite dilutes to ~+0.034 pooled — which is exactly the wash the
all-shapes table shows. **The suite is not measuring the mechanism it was built to justify.**
Reporting shape-stratified results as the *primary* table (policy handoff Q3) is not a presentation
preference; on this evidence it is the difference between seeing the effect and not.

### Chain-shape economics (graph stage complete)

| arm | n | score | visits | prompt | completion | $/cell |
|---|---|---|---|---|---|---|
| `graph:good_adaptive` | 42 | 0.377 | 3.3 | 87,697 | 2,363 | $0.00857 |
| `graph:baseline` | 44 | 0.220 | 1.1 | 29,099 | 1,091 | $0.00287 |
| `langgraph` | 17 | 0.326 | 2.5 | 24,499 | 216 | $0.00068 |

The adaptive arm costs 3× baseline on chains and earns it here — unlike the all-shapes picture.
Note it still runs 3× the prompt tokens of LangGraph for a +0.05 score edge, so the context-spend
critique from Addendum 2 stands even where the mechanism works.

### RUN-COMPLETE: graph-vs-linear on chains — the graph scaffold does not win its own best case

`cschain_sr` finished (27/27). Paired, run-complete, chains only:

| comparison | n | A | B | W / T / L |
|---|---|---|---|---|
| `good_adaptive` vs `baseline` | 33 | 0.356 | 0.233 | **18 / 8 / 7** |
| `good_adaptive` vs `seq_react` | 28 | 0.400 | 0.360 | 10 / 7 / **11** |
| `baseline` vs `seq_react` | 29 | 0.260 | 0.348 | 6 / 10 / **13** |

| arm | n | score | visits | prompt | completion | $/cell |
|---|---|---|---|---|---|---|
| `seq_react` | 37 | **0.396** | 3.1 | **21,017** | **3,101** | **$0.00232** |
| `graph:good_adaptive` | 42 | 0.377 | 3.3 | 87,697 | 2,363 | $0.00857 |
| `graph:baseline` | 44 | 0.220 | 1.1 | 29,099 | 1,091 | $0.00287 |

Per-model paired (`good_adaptive` vs `seq_react`): flash-lite 0.615 vs 0.528 (4/1/5),
nano 0.440 vs 0.476 (3/2/5), llama-3.2-1b 0.080 vs 0.005 (3/4/1).

**The decomposition is the finding.** Chains were chosen as the graph engine's *best case*. On it:

1. The **adaptive mechanisms** are real and valuable — **+0.123 over bare graph (18W/7L)**.
2. The **graph scaffold itself** looks like a **net negative**: bare graph *loses* to bare linear
   (6W/13L), and the adaptive machinery only recovers to **parity** (10W/11L) — while spending
   **4× the input tokens and 3.7× the cost**.

Read together: re-expansion and re-grounding appear to be recovering ground the graph structure
lost in the first place. That is a materially different conclusion from "the adaptive engine
works", and it is the strongest evidence this sweep produced about the architecture itself.

**It reframes the roadmap question.** Not "how do we tune the graph engine" but **"what happens if
the adaptive mechanisms are ported onto the linear loop?"** On this evidence that combination
should beat both arms at a fraction of the cost. It is now the highest-value experiment available
and is added as **EV-0** below.

**Caveats, stated plainly.** n=28 paired at R=1; chain tasks are mostly difficulty 10 so absolute
scores are low (0.36–0.40) and partly floor-limited; and `sequential_react` is this repo's own
long-standing reference arm, so it may simply carry more tuning history than the graph path. None
of those explain a 6W/13L loss for the bare scaffold, but all three argue for replication before
anything irreversible is decided.

### EV-0 (new highest-value experiment) — adaptive mechanisms on the linear loop

Port re-expansion / confidence re-grounding / corrective context onto `sequential_react` and run it
against `graph:good_adaptive` on the same chain set. Three outcomes, all informative:
- **linear+adaptive wins** → the graph scaffold is dead weight and DAG v3 should be a linear
  adaptive loop. Large, clean, publishable.
- **it ties** → the mechanisms are what matter and the scaffold is neutral; keep whichever is
  cheaper (linear, by 3.7×).
- **it loses** → the graph structure *is* contributing something the paired scores hide, and the
  next question is what.

This supersedes the previous EV-4 (graph-vs-linear on chains), which is now answered.

---

# Addendum 4 — local models on in-range tasks: the capability-crossover hypothesis does NOT hold

Mid-session, partial local in-range data suggested a clean crossover — the native engine winning
below ~3B, LangGraph pulling ahead at 7B. It was flagged at the time as a hypothesis, not a
finding. With more cells it **does not hold**.

Task-matched comparison (tasks 012 + 022 only — the tasks where all three arms have data):

| model | graph (base / adapt) | langgraph | favours |
|---|---|---|---|
| `tinyllama:latest` | 0.25 / 0.08 | **0.00 / 0.00** (cannot run) | graph — categorically |
| `phi3:mini` | 0.25 / 0.17 | **0.00 / 0.00** (cannot run) | graph — categorically |
| `qwen2.5:0.5b` | 0.25 / 0.17 | 0.50 / 0.42 | langgraph |
| `llama3.2:3b` | 0.50 / 0.58 | 0.23 / 0.25 | graph |
| `qwen2.5:7b` | 0.28 / 0.17 | 0.50 / 0.58 | langgraph |

The smallest **tool-capable** model (`qwen2.5:0.5b`) favours LangGraph while the 3B favours the
graph engine. There is no monotone capability ordering — among models both systems can run, the
local result is **mixed, at n=1 per cell**.

**The only consistent local edge remains the categorical one.** `tinyllama` and `phi3:mini` score
0.25/0.08 and 0.25/0.17 under the native engine and a structural 0.00 under LangGraph, because it
cannot start them.

**A fourth coverage-mismatch near-miss, recorded.** The per-model *means* over this data compare
graph arms (which had only tasks 012/022 complete) against LangGraph (which had all four, including
the two where `qwen2.5:7b` scored 0.86 and 1.00). Those means are not comparable and would have
shown LangGraph far ahead. The table above is re-cut task-matched. This is the fourth time in one
session — see the paired-and-run-complete contract in Addendum 2.

**One thing that improved:** these local in-range graph cells ran **after** the Finding 6
`expansion.py` fix, unlike the local hard-wave numbers, which predate it and understate the graph
arms. A clean re-run of the local hard wave is experiment EV-2 and remains outstanding.

---

# Addendum 5 (RUN-COMPLETE) — local weak models in-range: the clearest support for the thesis

`csleasy_g` finished (40/40). Local models on in-range tasks (012/022/023/049), task-matched paired,
infra-quarantined cells excluded:

| comparison | n | A | B | W / T / L |
|---|---|---|---|---|
| `good_adaptive` vs `graph:baseline` | 19 | **0.278** | 0.202 | **8 / 9 / 2** |
| `good_adaptive` vs `langgraph` | 20 | 0.272 | 0.295 | 10 / 3 / 7 |
| `graph:baseline` vs `langgraph` | 19 | 0.202 | 0.311 | 7 / 5 / 7 |

**Only 2 losses in 19 pairs** — a 4:1 win ratio, the best measured anywhere in this sweep. On weak
local models running tasks inside their competence range, the adaptive machinery reliably improves
on the bare graph. `llama3.2:3b` shows it most clearly: 0.28→0.50 (012), 0.29→0.57 (023),
0.00→0.75 (049).

Against LangGraph the local picture stays mixed: `good_adaptive` wins more pairs (10 vs 7) but has
a slightly lower mean, because LangGraph's wins are large (1.00 on 049 for both `llama3.2:3b` and
`qwen2.5:7b`) while the graph arm's wins are small. `tinyllama` and `phi3:mini` remain a structural
0.00 for LangGraph across all four tasks.

**These cells ran after the Finding 6 `expansion.py` fix.** The local *hard*-wave numbers earlier in
this document predate it and understate the graph arms; EV-2 (re-run the local hard wave) remains
the cheapest outstanding experiment.

## The consistent picture, across every run-complete comparison in this document

| claim | evidence | verdict |
|---|---|---|
| Adaptive mechanisms beat bare graph | 8/9/2 local in-range; 18/8/7 chains; 16/14/11 pooled | **holds everywhere tested** |
| The graph scaffold beats a linear loop | 6/10/13 on chains — its own best case | **fails** |
| DAG v2 beats off-the-shelf LangGraph on quality | 19/12/9 pooled, but ties `graph:baseline` where it can run; mixed locally | **weak / mixed** |
| DAG v2 runs models LangGraph cannot | 4 of 8 models, two serving stacks | **holds, categorically** |

**The mechanisms are the asset; the scaffold is the liability.** That is a coherent roadmap and it
is the opposite of what a pooled score table shows — which is why EV-0 (port the adaptive
mechanisms onto the linear loop) is now the highest-value experiment available.

## Measurement-integrity check that came back clean

Infra-quarantine rates are uneven by arm (`seq_react` 4.6%, `good_adaptive` 2.7%, `baseline` 0.9%,
`langgraph` 0%), and all `seq_react` quarantines were `llama-3.2-1b` cells scoring 0 — so excluding
them inflates that arm. Tested on the chain set: including quarantined cells moves `seq_react`
0.396→0.376 and `good_adaptive` 0.377→0.359. The exclusion is **roughly symmetric**, the gap holds
at ~+0.018 either way, and the paired conclusions are unaffected (pairing already drops a pair when
either arm is missing). Recorded because the asymmetry is real and a future run with more
quarantines could break this symmetry.

---

# Addendum 6 — CORRECTION: chains are the graph engine's WORST case, not its best

Addenda 3 and 5 framed chain tasks as "the graph engine's best case — the one shape where a later
step must condition on an earlier one," and then reported that it fails there. **That framing was
wrong, and the conclusion drawn from it was overstated.**

A chain is `A → B → C` with strict sequential dependency — a **path**, i.e. a degenerate graph.
There is no branching to exploit, so multi-candidate expansion, frontier selection and merges add
overhead and failure surface with nothing to gain. A ReAct loop *is* natively that shape
(observe→act→observe→act). **Chains are the LINEAR loop's best case.**

Worse, `BENCHMARK_POLICY_HANDOFF_2026-08-15.md` already said so, in improvement candidate #2:

> `auto_parallel_siblings: true` (default) executes siblings in one step, keeping graphs **depth-1
> by construction**, so a later step cannot condition on an earlier sibling's result — the thing a
> ReAct loop gets free.

Verified still live: `idea_dag_settings.json:153` and `idea_policies/config.py:695` both default it
to `true`, gated at `idea_engine.py:1597`. On a chain the engine is **structurally handicapped**.

## The data, re-cut by dependency shape

`chain` = strict sequential path; `parallel` = independent sub-problems that fan out and join
(a real DAG); `other` = disambiguation / conflicting-source / extraction.

**Bare `graph:baseline` vs `seq_react` — is the scaffold itself worse?**

| shape | n | graph | seq_react | W / T / L |
|---|---|---|---|---|
| **chain** | 29 | 0.260 | 0.348 | **6 / 10 / 13** |
| parallel | 9 | 0.470 | 0.481 | 4 / 3 / 2 |
| other | 15 | 0.497 | 0.489 | 6 / 3 / 6 |

**`good_adaptive` vs `graph:baseline` — where does re-expansion have to work hardest?**

| shape | n | adaptive | baseline | W / T / L |
|---|---|---|---|---|
| **chain** | 33 | 0.356 | 0.233 | **18 / 8 / 7** (+0.123) |
| parallel | 24 | 0.266 | 0.243 | 7 / 12 / 5 (+0.023) |
| other | 24 | 0.444 | 0.417 | 12 / 6 / 6 (+0.027) |

## What this changes

1. **The scaffold's loss to a linear loop is confined to chains** (6W/13L) and disappears on
   parallel (4W/2L) and other (6W/6L). Addendum 3's "the graph scaffold is a liability" generalised
   from its single worst shape. **Retracted in that general form**; the supported claim is *the
   graph scaffold is a liability on chains and roughly neutral elsewhere.*
2. **The +0.123 chain lift is a repair, not a feature.** Re-expansion's value is concentrated
   exactly where the depth-1 handicap bites (chain +0.123 vs parallel +0.023). It is compensating
   for a structure the engine imposes on itself, which is a very different thing from the mechanism
   adding capability. This also explains the original smoke's task-134 result (+0.48 adaptive over
   baseline) — the biggest lift on the most handicapped shape.
3. **The graph's actual best case is barely tested.** Only 9 paired cells on parallel/fan-out
   shapes. The claim the architecture most needs to defend has almost no evidence either way.

## Revised experiment ranking

**New EV-0a — `auto_parallel_siblings: false` on chain tasks.** Directly tests whether the chain
deficit is caused by the depth-1 batching. Cheap, decisive, and it converts "the graph loses on
chains" into either "…because of one flag" or "…for deeper reasons." This was already improvement
candidate #2 in the policy handoff; this sweep supplies the evidence that makes it urgent.

**New EV-0b — graph vs linear on PARALLEL shapes, properly powered.** The graph's real best case:
independent sub-problems a linear loop must serialise while carrying all of them in context. With
n=9 we have nothing. This is the experiment that should have been run instead of the chain one, and
it is the fair test of whether the DAG structure earns its place.

EV-0 (port the adaptive mechanisms onto the linear loop) stays valuable but drops below these two:
if EV-0a closes the chain gap, the mechanisms' apparent value shrinks to what they add on shapes
where the engine is not handicapping itself.

**Process note.** This correction came from a reader asking "shouldn't a graph be *less* suited to
a chain?" — a question about the mechanism, not the numbers. Every number in Addenda 3 and 5 was
correct; the frame around them was not, and no amount of pairing or run-completeness would have
caught it. That is a different failure mode from the five measurement traps recorded above, and
arguably a more dangerous one.
