# Pre-registration: DAG v2 fixes re-validation vs the confounded baseline (2026-08-21)

Written **before** any live cell runs. Authorized spend: **$3, cheap models only**, explicit
run-time authorization required from the user before execution (not pre-authorized by this
doc or by the plan). Same genre/discipline as `DAG_V2_PLAYBOOK_ALTBRANCH_RACE_PREREG_2026-08-21.md`,
whose exact matrix this run reuses.

## The question

The original pass (`dag_v2_playbook_race_20260821`, pre-reg above, results diagnosed in
`DAG_V2_REASONING_DIFF_FINDINGS_2026-08-21.md`) surfaced 3 root-cause bugs, since fixed and
merged on `comment-cleanup`:

| commit | fix | default |
|---|---|---|
| `8b91fa65` | cross-task Chroma link-index contamination (collections scoped to the run) + silent off-list-answer fallback (model's correct off-list URL no longer discarded) | unconditional |
| `910a287a` | grounding gate page-identity relevance signal (`final_require_grounding_page_identity`) | **opt-in, default `False` — NOT enabled this run, see below** |
| `068869e6` | alt-branch/race-group prompt addendum rebalance (worked examples + positive trigger) | always-on when `alternative_branch_enabled=True` (unchanged from original run) |

None of these have been live-validated. The question for this pass:

> Now that the Chroma-contamination and silent-fallback confounds are gone, does task 150's
> `good_adaptive_playbook` × `gpt-4.1-nano` cell actually visit the model-selected URL and score
> above its baseline 0.33? Does task 122's alt-branch gap (0.50 vs 0.80) persist at similar
> magnitude, and does the rebalanced prompt (`068869e6`) still produce zero
> `race_group`/`alternative_of` emission at the 3b tier? Do tasks 065/052 and the
> `langgraph_react` control stay stable (no regression, no infra-driven drift)?

This is a **paired re-run**, not a fresh exploratory pass — the value is a clean diff against
`agent/idea_test_results/dag_v2_playbook_race_20260821_summary.json`, so every parameter below
is held identical to the original except the run_id and the code under test.

## Arms (3) — unchanged

| arm | `IDEA_TEST_EXECUTION_VARIANTS` | `IDEA_TEST_ARM` | what it is |
|---|---|---|---|
| `graph:good_adaptive` | `graph` | `good_adaptive` | baseline |
| `graph:good_adaptive_playbook` | `graph` | `good_adaptive_playbook` | playbook + alt-branch + race-merge |
| `langgraph_react` | `langgraph_react` | `good_adaptive` (inert label) | off-the-shelf `create_react_agent`, stability control |

## Models (2) — unchanged

- `meta-llama/llama-3.2-3b-instruct` (weak tier)
- `openai/gpt-4.1-nano` (mid-cheap tier)
- validation model `gpt-5-mini`

## Tasks (4) — unchanged, same shape stratification

- `122` — alt-branch/disqualify (Arecibo decoy elimination) — checks whether `068869e6`'s
  emission fix moved the needle
- `150` — race-merge (Hardanger Bridge) — checks fix `8b91fa65` directly; this is the flagship
  confound-affected cell
- `065` — chain regression (Neruda elevation) — general-fix side-effect check
- `052` — breadth (6-way fan-out, argmin) — general-fix side-effect check

## Reps

**n=1**, same as the original. 4 tasks x 3 arms x 2 models = **24 cells**. Still exploratory —
no p-values quoted; the comparison being made is a directional paired diff against one specific
prior baseline run, not a statistical claim.

## Step-budget parity — unchanged

```
IDEA_TEST_MAX_STEPS=50
IDEA_TEST_LANGGRAPH_MAX_STEPS=50
```
Both set explicitly, matching the original run's parity fix.

## Full capture — unchanged

`IDEA_TEST_REPORT_VERBOSITY=3` for the whole run, spot-checked in the resulting JSON before
relying on it (same method as the original pre-reg's verification section).

## Grounding page-identity flag — explicit deferral

`final_require_grounding_page_identity` (`agent/app/idea_policies/config.py:376`) is **NOT**
enabled in any arm this run. Turning it on inside `good_adaptive`/`good_adaptive_playbook` here
would make it impossible to attribute any task-150 score movement between the Chroma-contamination
fix and the grounding fix — this run's entire value is being a clean paired diff, and folding in
a second, independently-gated behavior change would corrupt that. The flag also explicitly
requires its own live A/B before defaulting on, per its own design intent — this run doesn't
preempt that. A narrower follow-up run (task 150 + 122 control, 2 models, flag on/off) is the
right vehicle for that question and is deferred to a separate authorization.

## Pre-flight checklist (run before spending anything)

1. **Serper health check** — the original run hit a run-wide 403 outage, a known confound
   unrelated to the arms. Fire one throwaway search call before starting; if still degraded,
   wait or swap connector rather than re-running into the same confound.
2. Confirm `CHROMA_URL=http://localhost:8001` reachable.
3. Extract `OPENROUTER_API_KEY`/`SEARCH_API_KEY` from `services/keys.env` with `tr -d '\r'`
   (CRLF gotcha).
4. `IDEA_TEST_CONCURRENCY=1` (mandatory — shared connectors).
5. `PYTHONPATH=.:services:agent`.
6. New `IDEA_TEST_RUN_ID=dag_v2_fixes_revalidation_20260821` (do not reuse the original run_id —
   this must not overwrite `dag_v2_playbook_race_20260821`'s result files, which are the
   baseline being diffed against).
7. Confirm HEAD includes all 3 fix commits (`8b91fa65`, `910a287a`, `068869e6`) via
   `git merge-base --is-ancestor <sha> HEAD` — record the confirmation in the run report.

## Known confounds carried over from the original run (still apply)

1. Race-merge "losers still pay" cost asymmetry (task 150, `good_adaptive_playbook` arm) —
   reported separately from the other three tasks' $/cell.
2. `langgraph_react`'s `IDEA_TEST_ARM=good_adaptive` label is inert except for preserving
   `connector_retry_on_failure_enabled: True`.
3. `infra_failed` cells are not auto-quarantined by `level_ladder`/`gate_report`/`recovery_curve`
   — check manually.
4. No local/Ollama models in this pass — context-truncation caveat does not apply.

## New confound this run must watch for

5. **Result-file collision risk** — if `IDEA_TEST_RUN_ID` is left unset or accidentally reused,
   this run could silently overwrite or intermix with the baseline files it's meant to diff
   against. Verify the new run_id's output files are distinct from
   `dag_v2_playbook_race_20260821_*` before treating any comparison as valid.

## What each outcome would mean — stated before the data

| result | reading |
|---|---|
| Task 150 `good_adaptive_playbook`×`gpt-4.1-nano` now visits the model-selected URL (Hardanger Bridge, not Brooklyn Bridge), score moves off 0.33 | fix `8b91fa65` confirmed on its flagship target — the diagnosed root cause was real and is resolved |
| Task 150 still visits the wrong page | fix didn't cover this path or a second bug is present — re-diagnose before Part B assumes this confound is closed |
| Task 122 gap (0.50 vs 0.80) persists at similar magnitude, and grep of raw completions still shows zero `race_group`/`alternative_of` emission at the 3b tier | confirms the emission probe's finding in-repo, not just synthetically — direct evidence prompt-tuning has hit its ceiling here; primary go-signal for Part B's structural approach |
| Task 122 now shows nonzero tag emission | the rebalanced prompt (`068869e6`) has more effect live than the local ollama probe suggested — Part B's priority should be reassessed, not assumed |
| Tasks 065/052 regress vs baseline | `8b91fa65` (a general fix, not task-150-specific) has a side effect — investigate before calling the cycle clean |
| `langgraph_react` cells move meaningfully | suspect shared infra (Serper/Chroma) drift rather than the fixes, since none of the 3 fixes touch that execution path |

## Verification

- This pre-reg doc written and reviewed before any spend.
- Smoke-clean check (no infra-failure spike, no 0%-arm) before proceeding to the full sweep.
- Full-capture spot-checked directly in resulting JSON (grep for `prompt_text`/`messages`).
- Real spend measured via OpenRouter `GET /api/v1/key` usage delta, baseline-before/final-after,
  same mechanism as the original run.
- Findings written to a companion doc (`DAG_V2_FIXES_REVALIDATION_FINDINGS_<date>.md`) with
  explicit confirm/refute verdicts against the table above, feeding directly into Part B's
  go/no-go decision per `/home/muk/.claude/plans/cycle-complete-all-structured-treasure.md`.
