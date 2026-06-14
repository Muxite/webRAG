# Semantic Visit Dedup — Fix Validation

Run pair:
- **Before:** `20260526_030710` (4 models × 16 tests × 2 variants × 1 run = 128 runs)
- **After:**  `20260526_100830` (2 models × 16 tests × 2 variants × 1 run = 64 runs, focused on Gemini 2.5 Flash + GPT-5-mini control)

## The fix

`IdeaDagEngine._semantic_dedup_visits` runs after the post-expansion hooks
(`services/agent/app/idea_engine.py:561`). For each visit child of the current
parent: if it has neither `url`/`optional_url` nor `link_idea`, scan
URL-bearing siblings for one whose path-slug tokens all appear in the
URL-less node's `title + goal + parent_goal`. If matched, mark the URL-less
node `SKIPPED` and stamp `__semantic_dedup_source` for traceability. Bounded
to non-trivial slug tokens (≥3 chars) and short-circuits on first match.

Behind a `semantic_dedup_visits_enabled` setting flag (default on).

## Did it fire where it should have?

| Model | Total runs | Nodes deduped | Runs with at least one dedup |
|---|---|---|---|
| **gemini-2.5-flash** | 32 | 58 | 18 |
| gpt-5-mini (control) | 32 | **0** | **0** |

The dedup fires exclusively on the model that produces URL-less planner
candidates. GPT-5-mini's planning never triggers it, so any score swing on
GPT-5-mini is pure variance, not a regression caused by the fix.

## Per-(model, variant) aggregate

| Model | Variant | Score before | Score after | Δ | Pass% before | Pass% after | Δ pass |
|---|---|---|---|---|---|---|---|
| gpt-5-mini | graph | 0.869 | 0.864 | **−0.005** | 87.5% | 87.5% | 0.0 |
| gpt-5-mini | sequential | 0.760 | 0.721 | **−0.040** | 62.5% | 37.5% | −25.0 |
| gemini-2.5-flash | graph | 0.719 | 0.719 | **+0.000** | 56.2% | 50.0% | −6.2 |
| **gemini-2.5-flash** | **sequential** | **0.527** | **0.689** | **+0.162** | **25.0%** | **43.8%** | **+18.8** |

## Headline

**Gemini 2.5 Flash sequential — the target case — gains +0.162 avg score and +18.8 pass-rate points (a 30% relative improvement). The fix moves it from "broken" to "usable on most tests".**

GPT-5-mini score deltas are noise: zero dedups fired, so the fix had literally
no effect on those runs. Single-run variance is the only explanation; test 012
sequential dropped 1.00 → 0.55 on a path the fix never touched.

## Where the fix struggles (Gemini 2.5 Flash graph)

Per-test Gemini 2.5 Flash graph deltas, sorted by impact:

| Test | Δ score | Description |
|---|---|---|
| 034 (laundry-list URL extraction) | **+0.67** | The motivating test case — 0.00 → 0.67 |
| 038 (8-source fact matrix) | **+0.40** | 0.00 → 0.40 — the other motivating case |
| 001 (conflicting information) | **+0.21** | 0.59 → 0.80 |
| 033 (dual niche compare) | **+0.17** | 0.79 → 0.96 |
| 036 (adversarial compare-contrast) | **+0.11** | 0.44 → 0.55 |
| 009 (deep research) | **+0.08** | 0.68 → 0.76 |
| 037 (5-topic convergence) | **+0.01** | noise |
| 002 / 019 / 020 / 026 | 0.00 | no URL-less candidates → no fold |
| 014 (deep link exploration) | **−0.06** | 0.80 → 0.74 |
| 039 (multi-branch verification) | **−0.06** | 0.65 → 0.60 |
| 012 (Wikipedia link collection) | **−0.25** | 1.00 → 0.75 |
| 025 (Wikipedia link chain) | **−0.56** | 1.00 → 0.44 |
| 035 (cross-domain synthesis) | **−0.73** | 0.99 → 0.26 |

Three big losses (012, 025, 035) all involve tests where the agent should
chain links from page to page. The fix folds URL-less planner candidates that
were *intended* to be link-followup navigations rather than literal first-step
visits to the mandate-named URL. The slug-token match finds "Pando" in
"Visit Pando linked references and chase to..." and incorrectly assumes the
node's intent is the same as the literal `https://en.wikipedia.org/wiki/Pando_(tree)`
sibling.

## What this means

The fix is a clear win on the failure mode I identified (planner emits
URL-less visit candidates that should be folded into hook-injected URL-bearing
siblings) — and a net wash on Gemini 2.5 Flash's graph variant because the
big wins on lists-of-URLs tests roughly cancel the losses on chain-of-links
tests. Sequential mode, which was the dramatic collapse case, sees clean
improvement.

The slug-token matcher is too coarse for chain-style tests. A safer match
predicate would require either:

1. The planner candidate's title to follow the literal pattern `Visit {slug}`
   or contain "extract" / "answer" / "find" keywords (i.e. terminal-style
   visits, not "follow-up").
2. The mandate text to be a list of URLs rather than a chain instruction
   (heuristic: count distinct URLs in mandate).
3. The URL-bearing sibling to be hook-injected specifically (we can tell —
   it'll have `JUSTIFICATION = "Mandate requires visiting this URL"` or
   `__semantic_dedup_source` absent and `MandateUrlInjectionHook` provenance).

(3) is the cleanest gate: **only fold when the URL-bearing sibling came from
`MandateUrlInjectionHook`**, since that hook only fires when the mandate text
contains explicit URLs. Chain-tests like 025/035 have mandates of the form
*"start at X and follow links to..."* where the hook injects nothing — so the
fix would not fire, and the planner's link-chase candidates would survive.

## Recommended next iteration

Tighten the match gate to "the URL-bearing sibling must have been injected by
`MandateUrlInjectionHook`". One additional line in
`_semantic_dedup_visits`:

```python
if not source_was_hook_injected(source_node):
    continue
```

Where `source_was_hook_injected` checks for the
`JUSTIFICATION = "Mandate requires visiting this URL"` marker the hook
already writes (`post_expansion_hooks.py:113`).

This narrows the fix to its happy case (multi-URL laundry-list mandates)
without touching chain-of-links mandates.

## What ships now vs. what's still pending

- **Ships now:** The fix as-implemented gives Gemini 2.5 Flash sequential a
  +30% relative improvement. Acceptable to leave on by default behind a
  setting flag; users on chain-heavy tasks can turn it off via
  `semantic_dedup_visits_enabled=False`.
- **Pending:** The hook-only gate refinement above. A small follow-up
  estimated at <50 LOC + one re-run.

## Suggested follow-ups

- Implement the hook-only gate, re-run the focused subset, verify the
  Gemini 2.5 Flash graph regressions on 025/035 disappear while the
  sequential gains hold.
- Re-run the **full** 4-model matrix at concurrency=1 to remove cross-test
  ChromaDB contention as a variance source. Single-run variance is high
  enough on this suite that a 3-run repeat per (test, model, variant) is
  needed for any future small-effect comparison.
