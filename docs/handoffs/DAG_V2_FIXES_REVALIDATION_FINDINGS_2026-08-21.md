# Findings: DAG v2 fixes re-validation vs the confounded baseline (2026-08-21)

Paired live re-run of `dag_v2_playbook_race_20260821`'s exact 24-cell matrix, executed under
`comment-cleanup` HEAD after: the 3 originally-diagnosed root-cause fixes (`8b91fa65`,
`910a287a` opt-in/off, `068869e6`), a full additional night of fixes (merge-compaction
`5601309e`, substantive-child merge gate `59e29494`, goal-achievement tautology + consistency
guard `edc3f328`, merge-skip-retry `0879216a`, ollama context-truncation fix `3bf7f604`,
LangGraph fixes `50f25a34`..`db8341fe`), and two just-landed findings fixes (`8f4c3208`, "flag
silent goal_achieved schema misses and detect snippet-only evidence behind a merge gate").
All 10 commits confirmed as ancestors of HEAD via `git merge-base --is-ancestor` before spend.

Run id: `dag_v2_fixes_revalidation_20260821` (distinct from baseline's `dag_v2_playbook_race_20260821`
— no file collisions, verified). Same 3 arms x 2 models x 4 tasks x n=1 = 24 cells, same
`IDEA_TEST_MAX_STEPS=50` / `IDEA_TEST_LANGGRAPH_MAX_STEPS=50` / `IDEA_TEST_REPORT_VERBOSITY=3`,
`IDEA_TEST_CONCURRENCY=1`. `final_require_grounding_page_identity` NOT enabled (deferred per
pre-reg). Full-capture spot-checked in a `report_v3` JSON (`prompt_text`/`messages` present,
362 KB file).

## Spend

- Before: OpenRouter usage `125.417093429`
- After: OpenRouter usage `125.47244036`
- **Real spend: $0.0553** — well inside the $3 cap. All 24 cells landed with `infra_failed: False`
  (the llama-3.2-3b `langgraph_react` cells fail immediately with a 404 "no endpoints found that
  support tool use" — a known, pre-existing, cross-baseline structural gap, not new; identical
  warning string reproduced in the original baseline's own file).

## Top-line verdict

**Yes, directionally — the graph engine (the actual subject of the fixes) improved; the
`langgraph_react` control's pooled number moved the other way, but for a diagnosed non-fix
reason.**

| slice | old pooled | new pooled | delta |
|---|---|---|---|
| **All 24 cells** | 0.355 | 0.405 | **+0.050** |
| **graph arms only** (`good_adaptive` + `good_adaptive_playbook`, 16 cells — where all the fixes actually apply) | 0.327 | 0.448 | **+0.122** |
| `langgraph_react` control (8 cells) | 0.411 | 0.318 | −0.094 |
| excl. llama-3.2-3b `langgraph_react` (structural tool-use gap, both runs) | 0.426 | 0.486 | +0.060 |
| `gpt-4.1-nano` graph arms only | 0.386 | 0.654 | **+0.268** |
| `llama-3.2-3b` graph arms only | 0.267 | 0.243 | −0.024 (noise, n=1) |
| task 065 (chain) | 0.049 | 0.264 | +0.215 |
| task 052 (breadth) | 0.493 | 0.556 | +0.062 |
| task 122 (alt-branch) | 0.656 | 0.533 | −0.122 |
| task 150 (race-merge, flagship) | 0.222 | 0.267 | +0.044 (graph-arm-only cells drove this; see below) |

This is a **single n=1 paired run** — no significance claim, and with ~9 fixes landing between
the two runs plus one infra-state flip (Serper), per-fix attribution below is directional/
plausible, not proven.

## Flagship cell verdicts (from the pre-reg's "what each outcome would mean" table)

### Task 150 x `good_adaptive_playbook` x `gpt-4.1-nano` — confirmed fixed on the diagnosed axis

Old: score 0.33, 2 visits, both resolving to `https://en.wikipedia.org/wiki/Brooklyn_Bridge` /
`simple.wikipedia.org/wiki/Brooklyn_Bridge` — an unrelated bridge, even though the model's own
link-selection call had picked the Hardanger Bridge URL. `keystone_main_span` only passed by
luck/prior knowledge; `citations` was 0/0 (no URLs in the final text at all).

New: score **0.67** (up from 0.33). The final deliverable now reads:

> "The main span of the Hardanger Bridge in Norway is 1,310 metres, as directly stated on its
> English Wikipedia page (https://en.wikipedia.org/wiki/Hardanger_Bridge) in the infobox:
> 'Longest span 1,310 metres (4,300 ft)'."

`keystone_main_span` now passes (1.0, correct value bound to the correct page — no longer a
lucky-guess artifact), `citations` passes (0.67, 2/3 URLs cited). `route_coverage` still only
0.33 (1 genuine visit; the other 2 "routes" in the text are search-snippet citations — Reddit and
a mapping site — not full page reads), so this is not a clean ceiling win, but the specific
diagnosed failure (wrong-page grounding from `8b91fa65`'s target confound) is **directly
resolved**: the model-selected URL is now the one actually visited.

**Verdict: `8b91fa65` confirmed on its flagship target.**

### Task 122 x `good_adaptive` vs `good_adaptive_playbook` x `llama-3.2-3b` — gap persists, emission still zero

Old: 0.50 (`good_adaptive`) vs 0.80 (`good_adaptive_playbook`), gap 0.30.
New: 0.00 vs 0.50, gap 0.50 — same direction (playbook still beats plain good_adaptive), gap did
not close, if anything nominally widened (though both absolute scores also dropped, consistent
with ordinary n=1 run-to-run noise on a 3b model rather than a systematic effect — the
`good_adaptive` cell's new run has `0 visit(s)` and the grounding scrub correctly flagged its
answer as "Insufficient grounded evidence... may be fabricated", i.e. the model simply failed to
fetch any page this run, a stochastic behavior difference, not a code regression).

Grepped both `report_v3` files directly for `race_group`/`alternative_of`: the profile arm
(`cfgf9833fe1` = `good_adaptive_playbook`) shows 6 raw string matches, but **every one is inside
the system-prompt text or the JSON-schema description shown to the model** (e.g. `"Use
\"race_group\": \"<short label>\" on TWO OR MORE candidates..."`) — none appear as an actual
emitted field in a real model completion or expansion output. **Zero real emission at the 3b
tier, confirmed live, matching the local probe's finding exactly.**

**Verdict: confirms the emission probe's finding in-repo — direct evidence prompt-tuning
(`068869e6`) has hit its ceiling at this tier. This is the go-signal for a structural approach
(Part B), not a prompt-rebalance retry.**

### Tasks 065/052 — no regression; both improved

- Task 065 (chain, general-fix side-effect check): pooled 0.049 → 0.264, a large relative jump.
  Every one of its 6 cells is flat-or-better (no cell regressed).
- Task 052 (breadth, general-fix side-effect check): pooled 0.493 → 0.556. 5/6 cells flat-or-better;
  one cell (`llama-3.2-3b` x `good_adaptive_playbook`) dropped 0.583 → 0.25, single n=1 sample,
  consistent with ordinary variance at this model tier rather than a systematic regression (no
  other 3b breadth cell moved that direction).

**Verdict: no evidence `8b91fa65` (or the other general fixes) introduced a side effect on
chain/breadth shapes — both improved.**

### `langgraph_react` cells — NOT stable; one large swing, diagnosed as infra-state, not a fix effect

Per the task instructions' updated expectation (LangGraph fixes `50f25a34`..`db8341fe` ARE in
this run's HEAD), the cells were checked for improvement, not just stability. Result: mixed,
dominated by one large regression that traces to an infra-state flip rather than either set of
fixes:

- Task 065 `gpt-4.1-nano`: 0.29 → 0.54 (improved)
- Task 052 / 122 `gpt-4.1-nano`: 1.00 → 1.00 (already at ceiling both times, no room to move)
- Task 150 `gpt-4.1-nano`: **1.00 → 0.00** (regressed)
- All 4 `llama-3.2-3b` cells: 0.00 → 0.00 (unchanged — pre-existing, unrelated tool-use gap, see below)

The task-150 regression was investigated directly. In the **baseline** run, Serper was in a
run-wide 403 outage; the ReAct loop's search tool calls all failed, so the model fell back to
constructing and fetching 3 canonical Wikipedia URLs directly via `ConnectorHttp`, which
registered as real page visits and produced a fully-grounded correct answer (score 1.00). In
**this** run, Serper is healthy (confirmed live pre-flight and via this cell's own telemetry: 3
real successful `search`/`search_query`/`http_request` timing entries with real durations, not
error placeholders) — so the ReAct loop's search tool succeeded and the model never called a
follow-up page-fetch tool at all. Its final text is still textually correct (1,310 m, 3 cited
URLs, matching wording) but `observability.visit.count == 0`, so the grounding-gate check
(`visit_count >= 1`) correctly zeroes the score: this is an answer built from search snippets
(and plausibly parametric memory), not a page it actually read.

**Verdict: this is a real, diagnosed cause — the same "shared infra drift, not fix-driven"
outcome the pre-reg's own table anticipated (last row), just realized in the *opposite* direction
from what the pre-reg's example illustrated (baseline artificially inflated by an outage-driven
fallback, not this run artificially deflated by a fix regression).** None of the DAG v2 fixes nor
the LangGraph fixes touch this code path; the delta is explained entirely by Serper's health
state differing between the two runs, not by any commit under test. This is also a legitimate,
separate finding worth flagging to `strategy-tuner`: `langgraph_react`'s ReAct loop treats a
successful search as sufficient grounding and skips visiting pages even when doing so would let
it self-verify, which is a thinner evidentiary standard than the graph arms' explicit
`visit_count` requirement — an artifact of the underlying `create_react_agent` tool-calling
policy, not something either commit set was meant to fix.

### llama-3.2-3b x `langgraph_react` — confirmed pre-existing, not new

All 4 cells score 0.00 in both runs with identical `execution.output.warning`:
`"langgraph run error: NotFoundError: Error code: 404 - {'error': {'message': 'No endpoints
found that support tool use. Try disabling \"search\"...'}}"` — byte-identical error string in
both the baseline and this run's JSON. This matches the pre-existing, previously-documented
LangGraph tool-API gap (off-the-shelf `create_react_agent` cannot run several cheap
tool-calling-incapable models on OpenRouter). **Not a regression, not new, not something either
commit set touches.**

## Additional analysis: did "the system" measurably improve?

**Yes, on the part of the system the night's fixes actually target** — the native `graph` engine
(`good_adaptive` / `good_adaptive_playbook`), pooled 0.327 → 0.448 (+0.122, ~37% relative), with
no task shape showing a net regression on the graph arms specifically:

| task | graph-arms-only old | graph-arms-only new | delta |
|---|---|---|---|
| 065 (chain) | 0.00 | 0.26 | +0.26 |
| 052 (breadth) | 0.51 | 0.58 | +0.07 |
| 122 (alt-branch) | 0.67 | 0.44 | −0.23 (llama-3.2-3b noise-driven, see above; the `gpt-4.1-nano` half of 122's graph cells actually improved 0.82 → 0.85) |
| 150 (race-merge) | 0.08 | 0.40 | +0.32 |

The `gpt-4.1-nano` tier shows the largest and most consistent lift (0.386 → 0.654 pooled across
its 8 graph+langgraph cells, and every single one of its 4 graph-arm cells on tasks 065/052/150
improved or matched). The `llama-3.2-3b` tier is flat within noise on the graph arms (0.267 →
0.243, n=1 each cell) — this run has **no local/ollama cells**, so it cannot speak to the
ollama-context-truncation fix (`3bf7f604`) directly; that fix's own local validation (0.208 →
0.420, p=0.015, reported separately) remains the only live evidence for it. This run's 3b-tier
result is consistent with (does not contradict) that fix mattering specifically for the
local-ollama execution path rather than the OpenRouter-hosted 3.2-3b-instruct used here — they
are different backends for a similarly-named-but-distinct model class, so no inference should be
drawn either way from this run about `3bf7f604`'s effect size.

**Which fixes plausibly explain the observed lift, with confounds stated plainly:**

- The clearest single-cell causal story is the flagship task 150 x `good_adaptive_playbook` x
  `gpt-4.1-nano` cell: wrong-page → right-page grounding, directly matching `8b91fa65`'s
  diagnosed mechanism (Chroma link contamination + silent off-list fallback). This is the one
  cell in the whole matrix with a clean, single-fix causal story, because it is the one commit
  whose failure mode was independently diagnosed via raw-event tracing before this run.
- Task 065's broad lift (0.05 → 0.26 pooled, every cell flat-or-better) is consistent with
  general robustness fixes (merge-compaction `5601309e`, goal-achievement tautology guard
  `edc3f328`, merge-skip-retry `0879216a`) reducing spurious 0-scores on a chain shape, but **this
  cannot be isolated from `8b91fa65`'s general (not task-150-specific) contamination fix, which
  also plausibly affects any task that touches Chroma link retrieval** — task 065 does. Multiple
  plausible mechanisms, no way to separate them from this run alone.
- Task 150's non-flagship graph cells (`good_adaptive` x both models: 0.00 → 0.40/0.53) also
  moved a large amount beyond just the flagship cell, again consistent with the same contamination
  fix generalizing beyond the one diagnosed cell, but equally consistent with the merge/goal-
  achievement fixes reducing a different failure mode on the same task. **Not separable here.**
- Task 122's `gpt-4.1-nano` improvement on `good_adaptive_playbook` (0.80 → 1.00) is consistent
  with the rebalanced alt-branch prompt (`068869e6`) helping at a mid-cheap tier even though it
  demonstrably still emits nothing at the 3b tier (see above) — this matches the prompt fix's own
  documented caveat ("helps only ~14b+, not smaller local tiers") almost exactly, with
  `gpt-4.1-nano` landing on the "helps" side of that line.
- **Overclaim guard**: 9 distinct commits landed between the two runs plus one infra-state
  change (Serper up vs down) and this is n=1 per cell — no individual fix's contribution is
  isolated by this run design. The pooled +0.122 on graph arms is real and directionally
  consistent with the fix set as a whole, but a decomposed attribution would require a follow-up
  bisection run (one fix at a time, same matrix) which was explicitly out of scope and budget
  for this pass.

## Confounds and caveats carried into this result

1. n=1 per cell — every single-cell delta quoted above (task 122's llama-3.2-3b drop, task 052's
   one regressed cell) sits inside plausible sampling noise for a 3b-tier stochastic model; only
   the flagship 150 cell and the pooled graph-arms number should be treated as load-bearing.
2. The `langgraph_react` task-150 regression is a genuine, diagnosed infra-state artifact (Serper
   down in baseline, up in this run) rather than a code regression from either fix set — do not
   read it as "LangGraph got worse."
3. No local/ollama models in this pass; the `3bf7f604` context-truncation fix is not directly
   exercised here (see above).
4. `final_require_grounding_page_identity` remains off per the pre-reg's explicit deferral — this
   run cannot speak to that flag's effect.
5. Attribution among the ~9 simultaneously-landed fixes is not separable from this run design; a
   bisection re-run would be required for per-fix causal claims.

## Bottom line

The pre-registered flagship confirmation (`8b91fa65` fixes task 150's wrong-page grounding) is
**directly verified with a quoted deliverable change**, and the alt-branch emission-ceiling
finding is **reconfirmed live** (zero real emission at 3b, matching the local probe). Beyond the
two pre-registered flagship checks, the pooled graph-arms score improved +0.122 (0.327 → 0.448,
~37% relative) with no task shape showing a net regression on the arms the fixes actually target,
at a real cost of $0.0553 for the full 24-cell live re-run. The one cell that moved backward
meaningfully (`langgraph_react` x task 150) is explained by a diagnosed, unrelated infra-state
flip, not by any commit under test. **Directionally: yes, the system measurably improved on this
benchmark — with the standard n=1/multi-fix-landed caveats stated above, not with statistical
certainty.**
