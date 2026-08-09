# Reachable-Tier Composer + Leaf-Extraction Handoff — 2026-08-02

Durable handoff for the NEXT session continuing this thread. **Headline result:** badmodel-lab's
compiled-scaffold reachable tier (tasks 062/064/069/070/072/076/078, `qwen2.5:7b`, R=3) went from
avg **0.473** to avg **0.886** across this session's work — verified live, not just from the score
table (actual `final_deliverable` text spot-checked against ground truth). Everything described here
is **already committed** on `compiled-scaffold-dag` (see §0 — a parallel session/process committed it
mid-session under commit `1aac2ba1`, not under a message that calls out this specific result).

## 0. Current state (verified this session, re-check before trusting)

- Branch `compiled-scaffold-dag`. `git log --oneline -8 -- agent/app/testing/execution_compiled.py`
  shows `1aac2ba1` ("extend the plan library and visit action with grounding follow-through,
  deterministic link picking, infobox extraction, and computed-answer composition") contains
  everything below. `git diff --stat HEAD -- agent/app/testing/execution_compiled.py` was
  empty when this doc was written — **re-run it before assuming that's still true**, per
  `feedback_concurrent_sessions_git_state` (memory) this repo has multiple sessions editing the same
  branch concurrently; three MORE commits landed after `1aac2ba1` this same session on an unrelated
  feature (confidence-based early-exit calibration) — check `git log` fresh, don't trust this doc's
  commit hash staying at HEAD.
- Design docs (still on disk, not deleted, worth reading before extending this work):
  `/home/muk/.claude/plans/plan-so-that-after-flickering-dahl.md` (the session's overall phased plan),
  `/home/muk/.claude/plans/and-filter-count-threshold-design.md` (composer design, validator-traced),
  `/home/muk/.claude/plans/leaf_extraction_retry_design.md` (the leaf-extraction root-cause + fix
  design, live-reproduced evidence).

## 1. What was built (all already committed, per §0)

### Deterministic PAL-style composers (`execution_compiled.py`)
A plan-level `composition` dict (sibling to `leaves`/`aggregation`), read directly in
`_execute_plan()` when `agg_mode == "computed"`: computes the final answer in real Python from typed
leaf facts (zero extra LLM calls) instead of trusting free-text synthesis, falling back to the
original `_aggregate_single` path whenever data doesn't resolve cleanly enough (never worse than
before). Dispatch via `_COMPOSERS = {"and_filter": ..., "argmax": ..., "count_threshold": ...,
"subset_sum": ...}`.

- `_compose_argmax` — task 062 (topographic prominence). Degrades gracefully (≥2 resolved items
  suffice); ties named explicitly.
- `_compose_and_filter` — task 076 (area>X AND depth<Y, unique satisfier). All-or-nothing: every
  item's every constraint must resolve AND exactly one satisfier must result.
- `_compose_count_threshold` — tasks 072/078 (count items exceeding a threshold). All-or-nothing:
  every declared item must resolve (a missing item makes the true count ambiguous).
- `_compose_subset_sum` — task 070 (sum a defined subset). All-or-nothing, same rationale.
- Shared infra: `_compose_value` (handles `"<value> — source: <url>"` / `UNKNOWN` shapes, strips
  thousands-grouping commas before typing — `_coerce_field`'s own regex truncates `"4,148"` to `4`
  otherwise), `_fmt_num`, `_row_citation`, `_COMPARATORS`.
- **NOT built**: an `odd_one_out` composer for 069, or a ratio-argmax composer for 064 (needs
  dual-labelled per-leaf parsing — volume AND area from one leaf's fact string — a harder, distinct
  problem from the others; deliberately deferred, see §3).

### The `_vote_extract` quorum fix
`_run_leaf_thin`'s k-sample majority vote used to let a single, zero-corroboration hallucination win
uncontested whenever the rest of the samples honestly said UNKNOWN. Fixed: at `k>=3`, reject
(propagate a miss) when the leading answer has fewer than 2 corroborating samples
(`top_count < 2`). `k==2` keeps the old "rescue path" (a lone real answer beats a lone UNKNOWN) —
narrower than an earlier, over-aggressive version of this rule that also rejected genuine 2-of-5
corroboration and was walked back after a live regression (see git history / this session's
transcript for the full story if you need the cautionary tale).

### The leaf-extraction source-ask fix (the actual biggest lever — see §2)
`_strip_source_ask()` strips a plan-authored "...and the exact source URL" clause from the
extraction QUESTION only (never from the raw instruction used for `_target_entity`/
`_leaf_search_query`/`_pick_pages`) before it reaches `_vote_extract`. Default ON via
`IDEA_TEST_COMPILED_STRIP_SOURCE_ASK` (pure text transform, zero added LLM cost). Two more levers
shipped as **opt-in infrastructure, default OFF, not yet independently validated**:
`IDEA_TEST_COMPILED_INFOBOX_BLOCK` (restructures a page's infobox into "Label: Value" lines via
`observation.py`'s `extract_infobox_block()`) and `IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY` (one bounded
retry per page with an alternate `_THIN_EXTRACT_SYS_RETRY` prompt, on a quorum-inconclusive vote).

### A citation-URL bug fix, found as a byproduct
`_gathered_source_urls`'s trailing-punctuation stripper was eating a *legitimate* closing paren in
Wikipedia's own parenthesized titles (`Chuck_(season_1)`), silently zeroing citation credit on task
070 in BOTH the free-text and composed paths. Fixed via `_strip_url_trail` (only strips an unbalanced
excess `)`, never a balanced one).

## 2. The root-cause finding (this is the part worth remembering)

`_THIN_EXTRACT_SYS` tells the extraction model "...nothing else... no source", while ~113/151
hand-authored leaf instructions in `idea_tests/*.py` end with "...and the exact source URL" — reused
verbatim as the extraction question. `_run_leaf_thin` already appends the real URL deterministically
after a value is chosen; the model was never supposed to supply one. A weak model resolves the
self-contradiction by abstaining (UNKNOWN) even when told via a system-prompt addendum to ignore the
URL-ask — only removing the clause from the question text worked. Live-verified independently twice:
once by the design-pass agent (10/10 UNKNOWN → 10/10 correct on Sarez Lake's max-depth leaf), once by
a direct reproduction in this session (5/5 UNKNOWN → 5/5 correct, same leaf, same live model).

**This was NOT the hypothesis the session started with.** The original hypothesis (dense infobox
text flattening makes fields hard to parse) is real but secondary — it independently causes a
narrower failure (confusing an adjacent field, e.g. "Average depth" for "Max. depth", on a
generic/under-specified question) but wasn't the dominant driver once leaf instructions already
quote the exact field name (which they do, in this codebase's actual phrasing style).

## 3. Live-verified before/after (compiled `m1_thin`, qwen2.5:7b, reachable tier, R=3)

| test | before (this session's baseline) | after (all fixes above) |
|---|---|---|
| 062 argmax | 0.47 (pre-session) / 0.20 (mid-session regression) | **1.00±0.00** |
| 064 ratio-argmax (no composer) | 0.33 / 0.29±0.16 | **0.80±0.24** |
| 069 odd-one-out (no composer) | 0.72 / 0.60±0.35 | 0.60±0.35 (unchanged) |
| 070 subset_sum | 0.80 / 0.60±0.35 | **0.80±0.00** |
| 072 count_threshold | 0.56 / 0.25±0.02 | **1.00±0.00** |
| 076 and_filter | 0.20 / 0.20±0.00 | **1.00±0.00** |
| 078 count_threshold | 0.23 / 0.57±0.32 | **1.00±0.00** |
| **avg** | 0.473 (pre-session) | **0.886** |

Verified by reading actual `final_deliverable` text (not just scores) — e.g. 062's output is
literally `"Jengish Chokusu has the highest topographic prominence of the 6 peaks compared, at
4,148 m."` (the real ground-truth value), and 072/076/078 show the composer's deterministic render
format with correct real values.

## 4. What's next (per the approved plan's remaining phases)

1. **069 odd_one_out composer** — never built. 069 is stuck at 0.60±0.35 with 2/3 reps missing the
   keystone entirely via free text. Same composer pattern as the others should apply (identify the
   one item whose boolean attribute differs from all others) — needs its own design pass tracing the
   real `test_069_tier5_odd_one_out.py` validators, same rigor as the and_filter/count_threshold
   design doc.
2. **064 ratio-argmax composer — decide now, given new evidence.** The plan's original stance was
   "defer, its shape is harder (dual-labelled per-leaf parsing)". Now that the shared leaf-extraction
   fix alone lifted it from 0.29 to 0.80±0.24, the remaining gap may or may not be worth a bespoke
   composer — check a few more live reps first (064's variance is still high) before deciding whether
   to invest in it.
3. **Calibrate the opt-in levers** (`IDEA_TEST_COMPILED_INFOBOX_BLOCK`, `IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY`)
   — both shipped as infrastructure but their marginal value beyond the source-ask fix is
   unconfirmed. Live-test each independently before considering a nonzero default, per
   `leaf_extraction_retry_design.md` §5's test plan.
4. **AURC/Brier-style calibration-aware secondary metric** for badmodel-lab's scoring (Phase 4 of the
   plan) — orthogonal to the agent work above, addresses the gap that these validators currently give
   zero credit for honest "I don't know" vs. a lucky wrong guess (this tension caused a real
   mid-session regression this session when an anti-fabrication prompt was tried and reverted — see
   the plan file's Context section for the full story if extending aggregation-prompt work again).
5. **Broader suite check**: the source-ask fix is default-ON and the "and the exact source URL"
   phrasing appears in 113/151 `idea_tests/*.py` leaf instructions — this fix's effect is NOT scoped
   to just the reachable tier. Worth a check (not yet done) on whether other compiled-scaffold tasks
   elsewhere in the suite also picked up accuracy from this fix, or whether any test's own validator
   somehow depended on the old (broken) behavior.
6. **Full offline suite** was green at 2307 passed / 18 skipped when this was written — re-run before
   trusting that's still current (`PYTHONPATH=.:services ./.venv/bin/python3 -m pytest agent/tests
   agent/app/idea_tests -q`).

## 5. Cautionary tale worth reading before touching `_vote_extract` or aggregation prompts again

This session shipped a plausible-looking `_vote_extract` quorum threshold, found it caused a real
live regression (an over-broad rejection rule discarding genuine corroborated signal, not just the
diagnosed bug), narrowed it based on live evidence, then separately found and reverted an
anti-fabrication aggregation-prompt addition that *also* looked reasonable but suppressed a
lucky-guess mechanism the (imperfect) validators had been rewarding — netting a score regression with
no compensating gain. Moral: any change to shared voting/aggregation logic needs a live smoke-test
checkpoint read against actual `final_deliverable` text before trusting it, not just an offline-suite
pass. Don't repeat the freehanded-threshold mistake.
