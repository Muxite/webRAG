# Model Tier List — Approximate, Evidence-Based

A practical reference for "which model is expected to be better at what," built from three rounds
of live experiments on badmodel-lab's task suite (E1, 2026-08-03; Round 2 and Round 3,
2026-08-04). This is **approximate, not a rigorous statistical study** — sample sizes are R=1-3
per cell, one task tier at a time. Treat it as a starting expectation to update as more data comes
in, not a settled ranking.

**Read the caveats section before trusting any single number here.**

## Approximate tier list

**Tier A — frontier-adjacent, minimal mitigation needed.**
- `openai/gpt-4.1-nano` — consistently excellent everywhere tested (format 0.82–0.89, reachable
  0.97, hard 0.99). No severe weakness found in this data.
- `google/gemini-2.5-flash-lite` — excellent overall (format up to 1.00, hard 0.99, micro 0.83),
  with one sharp, isolated, task-specific miss (see below) rather than a general gap.
- `qwen2.5:14b` (local, $0) — NEW in Round 3. Reachable 0.97 (ties nano's ceiling), hard 0.95
  (beats every local model tested, nearly matches nano/flash-lite's 0.99). The standout finding
  of Round 3: a well-chosen local model, free, can match paid-API capability on this suite.
- `deepseek/deepseek-v4-flash` — RESOLVED in Round 3, moved up from the prior provisional Tier C.
  Reachable-tier score recovered from 0.53 to **0.96** once a real harness bug was fixed (see
  caveats — this was a measurement problem, not a capability gap, confirmed). **Caveat: only the
  reachable tier was re-tested; format/hard/micro numbers in the score table below are still the
  PRE-FIX numbers and should be treated as stale/unreliable rather than a genuine low capability
  read, pending re-test.**

**Tier B — strong for its size, real reproducible gaps.**
- `qwen2.5:7b` (local, $0) — ties or nears Tier A on most reachable/hard tasks, but has two
  distinct, reproducible reasoning gaps (negation/odd-one-out; k-th-ordinal). Worth the full
  mitigation stack, not worth writing off as "weak." Notably, its negation gap does NOT persist
  at 2x scale (`qwen2.5:14b` resolves it cleanly) — see the size-scaling note below.

**Tier C — meaningfully weaker, not a floor.**
- `phi3:mini` (local, $0) — reachable 0.69, best of the small local subjects tested in Round 3's
  gap-fill but clearly below Tier B.

**Tier D — near-floor for this suite without the mitigation stack.**
- `qwen2.5:0.5b` (local, $0) — reachable 0.54, surprisingly not a full floor (strong on 070, 076).
- `llama3.2:1b` (local, $0) — reachable 0.42.
- `tinyllama` (local, $0) — reachable 0.25, the flattest floor, confirms documented expectation.

**Not yet retested** — `qwen2.5:1.5b`, `gemma2:2b`, `llama3.2:3b`, `llama3.1:8b` have historical
data from prior sessions (format/micro tiers) but no `reachable`-tier run under this exact
harness/profile yet; not chased further in Round 3.

## Per-model strengths & weaknesses

### `openai/gpt-4.1-nano` — Tier A
No severe weakness surfaced across format/reachable/hard tiers in ~35 task-runs total (E1 +
Round 2). Cheapest of the API models tested ($0.0002–0.0056/task-run depending on tier). The safe
default recommendation among the models tested here.

### `google/gemini-2.5-flash-lite` — Tier A (one sharp gap)
Strong across the board, including a perfect 1.00 under the `fs1_structured_strict` format
profile. **One clean, reproducible miss**: task 062 (page-only topographic-prominence argmax),
0/3 reps correct, consistently naming the wrong peak with an implausible number ("Mount Gongga...
at 11,949 m"). Every *other* argmax-shaped task (078, hard-tier 077) it nails at 0.93–1.00 — this
reads as a specific fact/page-disambiguation miss on one task, not a general argmax weakness.

### `qwen2.5:14b` (local, free) — Tier A
The Round 3 headline. Reachable 0.97 (021/21 pass), hard 0.95 (14/15 pass) — both essentially at
the paid-API ceiling, and clearly ahead of `qwen2.5:7b` at the same size-doubling that resolved
its negation gap (see below). Zero OOM/crashes/errors across 72 task-runs despite this model
leaving only ~1.4GB VRAM headroom on the test machine's 12GB card — confirmed operationally stable
at this size. One real remaining miss: k-th-ordinal reasoning (task 075, hard tier) — 1 of 3 reps
wrong despite gathering all 6 underlying facts correctly, a ranking/computation slip rather than a
data-gathering one. **Format-tier finding, scoped to this model**: the `fs1_structured_strict`
(strict JSON schema) profile scored *worse* (0.70) than no enforcement at all (`fs0`, 0.74) —
`fs2_thin_assemble` was clearly best (0.89) for `qwen2.5:14b` specifically. Strict schema
enforcement is not a free win even for a capable model. **This does NOT generalize roster-wide** —
see `FORMAT_STRESS_TIER.md` §7 (full 8-model R=12 reconciliation, 2026-08-06): `fs2` wins for
`qwen2.5` (both ends of the size range tested) and `gemma2:2b`, but `fs1` wins for the
llama/phi3/tinyllama models, and `fs0` (unenforced) wins outright for `llama3.2:3b`. Pick the
format profile per model from that table, not from this one cell.

### `qwen2.5:7b` (local, free) — Tier B
Ties gpt-4.1-nano on 5 of 7 E1 reachable-tier tasks. Two real, reproducible gaps:
- **Negation/odd-one-out** (task 069): 0.33 in E1 — self-contradictory across self-consistency
  samples. **Resolved at 2x scale**: `qwen2.5:14b` scores 0.99 on the identical task, clean and
  consistent every rep. This looks like a capacity-limited failure specific to the 7B model, not a
  durable "qwen family" or "small open-weight model" trait.
- **K-th-ordinal reasoning** (hard-tier task 075): the one clear miss in its hard-tier pass.
  **Partially persists at 2x scale** — `qwen2.5:14b` still misses 1 of 3 reps on the same task,
  though it's no longer a clean failure. Scale helps here but doesn't fully resolve it, unlike 069.

### `phi3:mini` (local, free) — Tier C
Reachable 0.69 (13/21 pass) — best of the four small subjects gap-filled in Round 3, but a clear
step below Tier B. Strong on 062/072/078, weak on 064/076.

### `qwen2.5:0.5b`, `llama3.2:1b`, `tinyllama` (local, free) — Tier D
Reachable scores 0.54 / 0.42 / 0.25 respectively. `qwen2.5:0.5b` is the interesting one — not a
flat floor, with real strength on 070 (0.80) and 076 (0.97) despite failing most else. `tinyllama`
is the closest thing to a true floor in this data.

### `deepseek/deepseek-v4-flash` — Tier A on reachable (confirmed); other tiers pending re-test
**The Round 2→3 story**: scored lowest across every tier in Round 2 (reachable 0.53), flagged as
provisional pending a suspected harness bug rather than a real capability gap. Round 3 fixed that
bug and re-tested reachable: score recovered to **0.96** (21/21 pass), matching nano's ceiling.
Task 069 (negation) is the clearest before/after — Round 2 produced self-contradictory outputs
across votes; post-fix, all 3 reps correctly identify the answer with a coherent reason. The one
residual gap (task 070, capped at 0.80) is a citation-completeness issue (0/4 source URLs cited
despite correct computation) — and this exact 0.80 ceiling on 070 recurs across several unrelated
models/sizes (`qwen2.5:14b`, and historically `qwen2.5:7b`/`gemma2:2b`/`llama3.2:3b`), suggesting
it's a composer/task artifact specific to this task's shape under the `m1_thin` profile, not a
per-model weakness — worth a look by whoever next touches that composer.
**Format/hard/micro tier numbers in the score table below are still the pre-fix numbers and were
NOT re-tested in Round 3** — treat them as stale, not as a confirmed weakness, until re-run.

## Score table

| Model | Tier | Mean score | Keystone % | $/task-run | n |
|---|---|---|---|---|---|
| gpt-4.1-nano | format (fs0/fs1/fs2) | 0.82 / 0.89 / 0.89 | 100/100/100 | $0.0002–0.0007 | 9 each |
| gpt-4.1-nano | reachable | 0.97 | 100 | $0.0056 | 21 |
| gpt-4.1-nano | hard | 0.99 | 100 | $0.0050 | 5 |
| gemini-2.5-flash-lite | format (fs0/fs1/fs2) | 0.82 / 1.00 / 0.89 | 100/100/100 | $0.0002–0.0008 | 9 each |
| gemini-2.5-flash-lite | reachable | 0.87 | 86 | $0.0061 | 21 |
| gemini-2.5-flash-lite | hard | 0.99 | 100 | $0.0055 | 5 |
| gemini-2.5-flash-lite | micro | 0.83 | 100 | $0.0007 | 9 |
| qwen2.5:14b (local) | reachable | 0.97 | — | $0 | 21 |
| qwen2.5:14b (local) | hard | 0.95 | — | $0 | 15 |
| qwen2.5:14b (local) | format (fs0/fs1/fs2) | 0.74 / 0.70 / 0.89 | — | $0 | 9 each |
| qwen2.5:14b (local) | micro | 0.83 | — | $0 | 9 |
| qwen2.5:7b (local) | reachable | 0.85 | — | $0 | 21 (E1) |
| qwen2.5:7b (local) | hard | 0.87 | 80 | $0 | 5 |
| phi3:mini (local) | reachable | 0.69 | — | $0 | 21 |
| qwen2.5:0.5b (local) | reachable | 0.54 | — | $0 | 21 |
| llama3.2:1b (local) | reachable | 0.42 | — | $0 | 21 |
| tinyllama (local) | reachable | 0.25 | — | $0 | 21 |
| deepseek-v4-flash | reachable | **0.96 (post-fix)** | 100 | $0.0071 | 21 |
| deepseek-v4-flash | format (fs0/fs1/fs2), pre-fix, unverified | 0.74 / 0.82 / 0.78 | 100/100/89 | $0.0003–0.0014 | 9 each |
| deepseek-v4-flash | hard, pre-fix, unverified | 0.64 | 60 | $0.0037 | 5 |
| deepseek-v4-flash | micro, pre-fix, unverified | 0.61 | 67 | $0.0007 | 9 |

Total live spend across all three rounds: E1 $0.1245 + Round 2 $0.3343 + Round 3 $0.1496 =
**$0.7084**, against a combined $8 authorized ceiling.

## Caveats — read before trusting a number above

1. **Deepseek's Round 2 low scores were confirmed to be a harness measurement problem, fixed in
   Round 3.** `execution_compiled.py::_is_reasoning_model` didn't cover deepseek (OpenRouter bills
   its reasoning tokens inside `completion_tokens`, so it was silently starved to a 24-token
   thin-extraction budget). Fixed (commit `d17de329`) by mirroring the native engine's existing
   correct classification. Reachable-tier re-test confirmed the fix: 0.53 → 0.96. **Format/hard/
   micro were not re-tested** — the fix applies to the same shared token-budget logic across all
   tiers, so those old low numbers are presumed stale rather than confirmed accurate, but this
   hasn't been empirically re-verified. Re-test before drawing conclusions from those specific
   numbers.
2. **`tiers.yaml`'s "hard tier floors even nano" claim didn't hold in this data** — nano, flash-
   lite, and now qwen2.5:14b all score ≥0.95 on the hard tier. Worth a doc update, not chased
   further here.
3. **Task 069 negation reasoning is model/scale-specific, not a price-tier or "cheap model"
   effect.** flash-lite (API) and qwen2.5:14b (local, 2x qwen2.5:7b's size) both handle it cleanly;
   qwen2.5:7b and (pre-fix) deepseek did not. Scale within a model family resolved it once
   (7b→14b); it isn't a durable trait of "smaller" or "cheaper" as a category.
4. **Task 070's citation-completeness ceiling (0.80, 0/4 URLs cited) recurs across unrelated
   models and sizes** (qwen2.5:14b, deepseek post-fix, and historically qwen2.5:7b/gemma2:2b/
   llama3.2:3b) — likely a composer/task artifact specific to this subset-sum shape under the
   `m1_thin` profile, not a real per-model capability signal.
5. **Task m02's `grounding` check has a scoring artifact**, unrelated to model capability: it
   requires an exact URL match against `PAGE_URL`, but Wikipedia's canonical redirect target
   differs from the literal string the task checks for. Reproduces again in Round 3
   (`qwen2.5:14b` gets the correct fact every time but scores 0.0 on grounding, capping m02 at
   0.50 and the format-tier f02 task at 0.67 across all three profiles). Fix candidate:
   `agent/app/idea_tests/test_m02_amsterdam_area.py`'s grounding check should accept the
   known redirect target. Still not fixed.
6. **Small samples.** R=1-3 per cell. Treat single-digit-percentage differences as noise.

## Sources

- E1 (2026-08-03): `badmodel-lab/results/cells.jsonl` run_ids
  `bml__qwen2.5-7b__m1_thin__reachable`, `bml__openai-gpt-4.1-nano__m1_thin__reachable`.
- Round 2 (2026-08-04): run_ids prefixed `bml__openai-gpt-4.1-nano__m1_thin__f*`,
  `bml__deepseek-deepseek-v4-flash__m1_thin__*` (pre-fix), `bml__google-gemini-2.5-flash-lite__
  m1_thin__*`, plus a `qwen2.5:7b` hard-tier cell.
- Round 3 (2026-08-04): the deepseek `reachable`-tier re-test (post-fix, same run_id prefix as
  Round 2, later timestamps), `bml__qwen2.5-14b__*` across reachable/hard/format/micro, and
  gap-fill reachable-tier cells for `tinyllama`/`qwen2.5-0.5b`/`llama3.2-1b`/`phi3-mini`.
- Full writeups: `agent/app/AGENT_CONTINUUM.md`'s E1/E4 sections. Raw result JSONs in
  `agent/idea_test_results/bml__*.json` (gitignored, local to whichever environment ran
  them).
