# Investigation — Gemini 2.5 Flash Collapse on Run 20260526_030710

**TL;DR:** Gemini 2.5 Flash burns **8× more prompt tokens per test** than GPT-5-mini, yet still produces malformed visit candidates and weak self-evaluations. The engine's safety nets (mandate-injection hook, sequential prune-siblings) interact badly with these weaknesses and amplify the collapse. The model itself is the root cause; the engine is the multiplier.

## The numbers

| Model | Graph pass | Sequential pass | Δ |
|---|---|---|---|
| Claude Sonnet 4.6 | 87.5% | 68.8% | −18.7 |
| GPT-5 mini | 87.5% | 62.5% | −25.0 |
| Gemini-3 Flash Preview | 87.5% | 75.0% | −12.5 |
| **Gemini 2.5 Flash** | **56.2%** | **25.0%** | **−31.2** |

Per-test LLM-token usage across all 32 (test × variant) runs:

| Model | LLM calls | Prompt tk | Tk/call | Tk/test |
|---|---:|---:|---:|---:|
| GPT-5 mini | 280 | 308,604 | 1,102 | 9,644 |
| Gemini-3 Flash Preview | 172 | 709,454 | 4,124 | 22,170 |
| Claude Sonnet 4.6 | 242 | 1,515,877 | 6,264 | 47,371 |
| **Gemini 2.5 Flash** | **233** | **2,507,324** | **10,761** | **78,354** |

Gemini 2.5 Flash uses **8.1× more prompt tokens per test than GPT-5-mini** and **3.5× more than its own newer sibling Gemini-3 Flash Preview**. Whatever the smaller model is doing, it's not compressing context efficiently.

## Three compounding weaknesses

### 1. Poor JSON schema adherence (the planner skips URL fields)

On test 034 (laundry-list URL extraction), the engine's `MandateUrlInjectionHook` injects 6 visit nodes (one per URL in the mandate text). The planner is supposed to *also* fill in URLs from the mandate, so dedup folds the duplicate plans together. GPT-5-mini does this correctly — its 6 planned visits each have `url`/`optional_url` populated, dedup with the injected nodes works, and the engine ends up with exactly 6 unique visits → all complete → score 1.00.

Gemini 2.5 Flash produces 5 planner-generated visit candidates **with NO `url` / `optional_url` / `link_idea` fields**:

```json
{
  "action": "visit",
  "goal": "Visit Axolotl Wikipedia page and extract scientific name",
  "is_leaf": true,
  ...
}
```

There's no URL anywhere in `details`. Compared to the same node from GPT-5-mini's planner, which has `"optional_url": "https://en.wikipedia.org/wiki/Axolotl"`. The expansion system prompt explicitly says: *"Copy EXACT URLs from previous results. Never construct or guess URLs."* — Gemini 2.5 Flash ignores it.

These URL-less candidates can't dedup against the hook-injected nodes (different memo keys), so the graph ends up with **11 visits instead of 6**.

### 2. Bloated prompts cause cascading timeouts

The 11 simultaneous visits all attempt to execute in parallel (graph mode's `asyncio.gather`). Each visit, when it has no URL, tries to fall back to semantic link discovery via ChromaDB. With 11 concurrent chroma queries (and 11 concurrent HTTP fetches when the URL-bearing injected nodes do dispatch), the system thrashes.

Aggregate ChromaDB max query time for this run: **55.3s** — well over the `visit_timeout_seconds=20s` cap. Even the hook-injected nodes WITH valid URLs get killed by the per-action timeout because they're competing for the same overloaded backend.

Result on test 034 graph: **all 11 visit nodes fail with `"timeout after 20.0s"`**, 0 of them complete, deliverable says *"I cannot provide accurate answers because the pages were not visited"*. Score: **0.00**.

### 3. Weak self-evaluation collapses sequential mode

In sequential mode the engine generates candidates, the batch-evaluator scores them, the highest-scoring sibling wins, and **the rest are marked SKIPPED**. That prune step is the failure trigger.

Test 026 (deterministic page facts on `example.com` + `iana.org/domains/reserved`):

- `MandateUrlInjectionHook` correctly injects 2 visit nodes for the mandate URLs.
- Gemini 2.5 Flash's planner adds its own visit candidates, including a CiteSeerX node sourced from semantic discovery (not from the mandate text).
- Gemini 2.5 Flash's batch-evaluator scores its own CiteSeerX node higher than the mandate-injected `example.com` / `iana.org` nodes.
- Sequential prune-siblings marks the two correct visits **SKIPPED**.
- The single visited URL (CiteSeerX, completely off-topic) provides no answer to the mandate.

Graph state at the end:
- `visit done`: 1 (the wrong one)
- `visit skipped`: 2 (the correct mandate-injected nodes)

Deliverable: *"The provided execution trail indicates that the requested URLs were not visited. Instead, the system visited CiteSeerX. Therefore, I cannot provide..."*

The mandate-injection safety net works in graph mode (everything runs in parallel) but is defeated by sequential's prune step when the evaluator picks wrong.

## Why graph saves Gemini partially

Graph mode lets every injected node race against the planner's noise. When the planner produces 5 URL-less candidates plus the hook injects 6 URL-bearing ones, all 11 attempt to execute. If the hook's nodes succeed and the planner's nodes fail, the merger still has data to work with — provided the timeout cascade doesn't kill everyone, which is exactly what happens on test 034.

Sequential mode picks one path. If the evaluator picks the noise, the signal is pruned and never recoverable.

## Tests where Gemini 2.5 Flash succeeded

Sanity-check: it's not uniformly broken.

- **Tests 002, 012, 020 (graph and sequential)**: simple single-fact retrievals, the planner gets the URL right because there's only one URL in the mandate.
- **Test 025 (graph 1.00, sequential 0.44)**: Wikipedia link-chain — graph parallelism rescued it; sequential collapse.
- **Test 035 (graph 0.99, sequential 0.32)**: cross-domain synthesis — graph fine; sequential collapse.

Pattern: Gemini 2.5 Flash handles **single-URL mandates** roughly competently, fails on **multi-URL mandates** and on **anything that needs sequencing/dependency between steps**.

## Per-call failure-rate sanity check

| Model | LLM errors / calls | Rate | Verdict |
|---|---|---|---|
| GPT-5 mini | 109 / 280 | 38.9% | High transient-error rate, but retries recover |
| Claude Sonnet 4.6 | 86 / 242 | 35.5% | Same — transient noise, recoverable |
| Gemini-3 Flash Preview | 7 / 172 | 4.1% | Cleanest API |
| **Gemini 2.5 Flash** | 16 / 233 | 6.9% | Low error rate, but the calls that succeed return bad content |

Gemini 2.5 Flash is not failing because of API errors. It's failing because its outputs are worse — quietly, deterministically, on every call.

## Recommendations

Three independent fixes, none of which require changing the engine's plan in major ways.

### 1. Engine: tighten dedup to fold URL-less planner visits into hook-injected ones

When `MandateUrlInjectionHook` has already injected a visit for URL X, the engine should detect a *URL-less* planner visit whose title or goal mentions X's Wikipedia article and either drop it or fill its `optional_url` with X. Today the memo key requires identical action+URL; loosen it to "same action + same intent" by checking title similarity or running the new `GoTOperations.is_duplicate_thought()` over the candidates.

This is the highest-impact fix — it doesn't help Gemini understand the prompt, but it stops the model's noise from spawning 11 nodes when 6 will do.

### 2. Engine: cap parallel visits per step

`asyncio.gather(*leaves)` runs every unblocked sibling concurrently. Add a `parallel_visit_limit` setting (default ~3) so 11 concurrent visits don't thrash ChromaDB. Already-allow-flagged in plan (B6 *Adaptive concurrency* in `IDEA_ENGINE_FEATURES.md`); promote to a Phase 0.5 fix because it bites this run.

### 3. Settings: tier-aware action timeout

`visit_timeout_seconds=20` is fine for GPT-5-mini and Claude-Sonnet but too aggressive when ChromaDB is under contention. Bump it to 60s and rely on the *overall* `final_timeout_seconds=180` to cap runaway costs. Trades some failure-fast benefit for headroom under multi-visit fan-out.

### 4. Model selection: don't ship Gemini 2.5 Flash as a "good cheap option"

It's not. Gemini-3 Flash Preview is in the same tier (free/cheap on OpenRouter), generates 3.5× fewer prompt tokens per test, fails far less catastrophically, and has the *best* sequential pass rate in this run. For users who want a low-cost Gemini option, route them to Gemini-3 Flash Preview.

## Suggested follow-up runs

- **Same matrix with `parallel_visit_limit=3`**: confirm that fix-2 alone lifts Gemini 2.5 Flash graph score by ~20 points (the test-034 family) without hurting the strong models.
- **Same matrix with mandate-prompt addendum disabled for Gemini 2.5 Flash**: isolate whether the mandate-injection-hook safety net is doing more good than the URL-less duplicates do harm.
- **Same matrix with `IDEA_TEST_CONCURRENCY=1`**: removes cross-test backend contention and tells us whether the timeouts are intra-test (test 034's 11 parallel visits) or inter-test (other tests' chroma traffic during test 034).
