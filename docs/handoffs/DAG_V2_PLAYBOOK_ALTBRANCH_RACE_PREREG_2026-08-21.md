# Pre-registration: DAG v2 playbook/alt-branch/race-merge vs LangGraph, cheap-only exploratory pass (2026-08-21)

Written **before** any live cell ran. Authorized spend: **$3, cheap models only**, explicit
run-time authorization (not pre-authorized in the plan). Same genre and discipline as
`docs/handoffs/CAPABILITY_SPECTRUM_PREREG_2026-08-15.md`; scaled down per
`/home/muk/.claude/plans/sprightly-squishing-floyd.md` §1.

## The question

The prior cycle (commits `6bc98361`..`d5715608`, plus this cycle's `3422566b` leak fix and
`7afe3286` capture wiring) built three new native DAG v2 mechanisms combined into one arm,
`good_adaptive_playbook`: playbook advice injected into expansion prompts (`strategy_library_*`),
a mechanical A→B alternative-branch fallback with promote-on-fail/promote-on-unverified
(`expansion_alternative_branch_enabled`, `got_alternative_branch_promote_on_*`,
`got_contract_veto_requires_datum_enabled`), and mechanical concurrent race-and-merge winner
selection (`merge_race_winner_selection_enabled`). None of this has run live yet. The question for
this pass:

> Does the combined `good_adaptive_playbook` arm beat plain `good_adaptive` on the shapes it was
> built for (alt-branch disqualify, race-merge), without regressing the shapes it wasn't (chain,
> breadth) — and how does either compare to `langgraph_react` at parity step budget?

This is explicitly a first-look smoke, not the confirmatory ablation run (`good_adaptive_playbook`
already has three single-axis ablation sibling profiles registered —
`good_adaptive_playbook_notes_only`, `..._altbranch_only`, `..._race_only` — none run here; see
"Deferred" below).

## Arms (3)

Verified registered in `agent/app/idea_test_runner.py::_GOT_ARM_PROFILES` before writing this doc
(lines ~459, ~498):

| arm | `IDEA_TEST_EXECUTION_VARIANTS` | `IDEA_TEST_ARM` | what it is |
|---|---|---|---|
| `graph:good_adaptive` | `graph` | `good_adaptive` | baseline — DAG v2 proper, the established winner, no playbook/altbranch/race |
| `graph:good_adaptive_playbook` | `graph` | `good_adaptive_playbook` | this cycle's combined mechanism arm (playbook notes + alt-branch + race-merge, all three axes on) |
| `langgraph_react` | `langgraph_react` | `good_adaptive` (inert profile label; see confound 2) | off-the-shelf `create_react_agent` loop |

## Models (2 tiers, cheap only)

Resolved from actual prior usage in this repo, not guessed — cross-checked against
`scripts/adaptive_ladder_run.py`'s `AXES["capspec_api"]["ladders"]` and the
`capability_spectrum_v2_20260820_*` result filenames on disk (`agent/idea_test_results/`):

| tier | model slug | evidence |
|---|---|---|
| genuinely weak | `meta-llama/llama-3.2-3b-instruct` | `capability_spectrum_v2_20260820_*_meta-llama-llama-3.2-3b-instruct_*.json` (24+ files present) |
| mid-cheap | `openai/gpt-4.1-nano` | `capability_spectrum_v2_20260820_*_openai-gpt-4.1-nano_*.json`; also the "established cheap-tier reference point for every prior result in this repo" per `adaptive_ladder_run.py`'s own comment on `capspec_api` |

(The `capability_spectrum_v2_20260820_*` files also include `google/gemini-2.5-flash`, a third,
somewhat stronger cheap tier — deliberately excluded here to stay inside $3 and match the plan's
"one weak, one mid-cheap" scope.)

## Tasks (4, shape-stratified)

| id | shape | why | weight/difficulty |
|---|---|---|---|
| 122 | alt-branch / disqualify | `expansion_alternative_branch_enabled`'s target shape — Arecibo fame-decoy elimination, FAST survivor | long / 9 |
| 150 | race-merge | `merge_race_winner_selection_enabled`'s target shape — 3 redundant routes to the Hardanger Bridge main span; picked over its sibling 151 per the plan's own fallback instruction ("check task docstrings if needed, or just pick 150" — no separate task-authoring margin report was found on disk, so the plan's explicit default is used) | medium / 6 |
| 065 | chain (regression check) | 3-hop URL-free dependent chain, leak-resistant terminus (Neruda -> Parral, Chile -> elevation); checks the combined arm doesn't regress the shape it wasn't built for | long / 9 |
| 052 | breadth (exploratory) | 6-way fan-out + argmin aggregation; unambiguous margin (Austen 1775 vs runner-up 1821, 46y); exploratory data point, not a target shape for any of the three new axes | long / 8 |

All four are URL-free (search-driven), confirmed by reading each task module's docstring before
this doc was written.

## Reps

**n=1.** Explicitly exploratory — a first-look smoke-and-signal pass, not a confirmatory run. No
p-values will be quoted. 4 tasks x 3 arms x 2 models x 1 rep = **24 cells**.

## Step-budget parity

Checked `agent/app/testing/execution.py:561` (`graph` arm: `idea_settings.get("max_steps")` else
`IDEA_TEST_MAX_STEPS` env else default `"50"`) and `agent/app/testing/execution_langgraph.py:65`
(`langgraph_react`: `IDEA_TEST_LANGGRAPH_MAX_STEPS` env else default `"25"`). Neither
`good_adaptive` nor `good_adaptive_playbook` sets `max_steps` in its profile dict (only `max_burn`
does, to 90 — not used this pass), and this run does not set `IDEA_TEST_EFFORT_TIERS` (an effort
tier would override `max_steps` via `_apply_effort_tier`, confirmed unused/`effort_tier: 0` in the
most recent `capability_spectrum_v2_20260820_*` result files read for this doc). So the `graph`
arm's real default is **50**, not LangGraph's smaller default of **25**.

**Set explicitly for this run, both arms to the same value:**
```
IDEA_TEST_MAX_STEPS=50
IDEA_TEST_LANGGRAPH_MAX_STEPS=50
```
Both raw W/T/L and step/token-normalized win-rate will be reported so a DAG v2 win (if any)
can't be silently explained by "it got more turns" — moot at parity, but reported anyway per the
plan's instruction.

## Full capture

`IDEA_TEST_REPORT_VERBOSITY=3` set for the whole run. Native path (`graph` arm) already supports
this (`ConnectorLLM.set_full_capture`, no code change needed). `langgraph_react`'s support was
added this cycle, commit `7afe3286` — confirmed present before relying on it:
`agent/app/testing/execution_langgraph.py:68` reads `IDEA_TEST_REPORT_VERBOSITY` and passes
`full_capture=report_verbosity >= 3` into `solver.solve(...)` at line 77. Verified live (not just
by reading code) as part of this run's Step 4 report — see the grep-for-`prompt_text`/`messages`
spot-check.

## Stopping rule

Exploratory. Report:
- raw W/T/L per (task, model) across the three arms, using `execution.validation.overall_score`
  (or the task's keystone gate if binary)
- $/cell per arm/model (real spend, not the estimate)
- step/token-normalized win-rate (score per 1k tokens, score per step used) so a raw win can't
  hide behind a bigger budget — moot here since step budgets are matched, reported for completeness
- **no p-values at n=1** — this is a first look, not a confirmatory claim

## Known confounds, recorded up front

1. **Step-budget parity, exact numbers used**: `IDEA_TEST_MAX_STEPS=50` (graph arms),
   `IDEA_TEST_LANGGRAPH_MAX_STEPS=50` (langgraph_react arm). Matched, not the LangGraph default of
   25 vs graph's default of 50 that the 2026-08-15 sweep ran under.
2. **Race-merge "losers still pay" cost asymmetry.** `merge_race_winner_selection_enabled` (task
   150 cell, `good_adaptive_playbook` arm only) runs k concurrent branches and keeps only the
   winner's result — the losing branches' LLM/search calls are real spend that never shows up in
   the winning path's telemetry. Task 150's $/cell will be reported **separately** from the other
   three tasks' $/cell, not pooled into one average, so this asymmetry is visible rather than
   diluted.
3. **`langgraph_react`'s `IDEA_TEST_ARM=good_adaptive` label is inert for every knob that variant
   reads**, except that it preserves `connector_retry_on_failure_enabled: True` (F16 fairness
   invariant in `langgraph_solver.py`) — `baseline`'s profile pins that to `False`, which would
   silently handicap the LangGraph arm alone. Confirmed by reading the arm-profile dicts before
   this run, not assumed.
4. **This pass excludes, by explicit scope decision, and defers to a future better-funded run:**
   - the three single-axis ablation arms (`good_adaptive_playbook_notes_only`,
     `..._altbranch_only`, `..._race_only`) — at $3 across 24 cells they would get near-zero
     coverage each and produce noise, not signal; attribution of which of the three new axes
     drives any observed effect is explicitly out of scope this pass
   - a strong-tier regression check (e.g. `google/gemini-3.1-pro-preview` or similar) — cheap-only
     per the authorization; whether the combined arm helps, hurts, or is inert at the top of the
     capability range is unknown after this pass
   - the branch-scoped-retrieval leak-fix flag (`MemoryConfig.branch_scoped_retrieval_enabled`,
     landed `3422566b`, off by default) as a fourth axis — not included in any arm this pass, so
     this run cannot speak to whether closing the cross-branch memory leak changes race-merge or
     alt-branch outcomes
5. **`infra_failed` cells are not quarantined** by `level_ladder`/`gate_report`/`recovery_curve` —
   checked manually before any cell's result is treated as signal, per the 2026-08-15 pre-reg's
   same caveat.
6. **Local-model context-truncation caveat does not apply here** (no local/Ollama models in this
   pass, both models are API-hosted) — noted only because the 2026-08-15 pre-reg carried it and it
   does NOT carry over; stated explicitly so it isn't silently assumed to still apply.

## What each outcome would mean — stated before the data

| result | reading |
|---|---|
| `good_adaptive_playbook` beats `good_adaptive` on 122 and/or 150, ties on 065/052 | the combined arm's new axes pay on their target shapes without regressing the others — supports building the confirmatory ablation run next |
| `good_adaptive_playbook` ties or loses to `good_adaptive` everywhere | either the mechanisms are inert at this model tier, or $3/n=1 is too little signal to see them — cannot distinguish those two without more data; do not conclude the mechanisms don't work |
| `good_adaptive_playbook` beats `good_adaptive` but also regresses 065/052 | a real overhead/distraction cost from the new axes on shapes they weren't built for — the honest headline is "narrow win, broad cost", worth attributing before shipping default-on |
| `langgraph_react` ties or beats both graph arms at matched step budget on any shape | the step-budget confound from 2026-08-15 is resolved for this shape and the result is not explained by "more turns" |
| any tier shows near-zero scores in all three arms | task/tier mismatch (out of range for the model), not an arm finding — a null result about calibration, not the mechanisms |

## Verification

- Pre-registration doc (this file) written and reviewed before any spend.
- Smoke-clean (no infra-failure spike, no crash, no 0%-success-rate arm) confirmed before
  proceeding to the full 24-cell sweep — reported in Step 4.
- Full-capture presence spot-checked directly in the resulting JSON (grep for `prompt_text` /
  `messages` in one `graph` result and one `langgraph_react` result), not assumed from the flag
  alone.
- Real spend measured via OpenRouter `GET /api/v1/key` `usage` delta (baseline before the run,
  final after), same mechanism `scripts/adaptive_ladder_run.py::openrouter_key_usage` uses for
  `--real-budget`, reproduced manually here since this run drives `idea_test_runner` directly
  (the arms/variants needed don't fit that driver's single-model-per-invocation ladder shape
  cleanly — see the report for the exact commands used instead).
