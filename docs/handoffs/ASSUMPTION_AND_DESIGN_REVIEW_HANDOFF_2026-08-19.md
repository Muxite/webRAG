# Handoff — assumption audit, design reviews, and the thought-log feature (2026-08-19)

**Nothing in this session was committed.** All work is uncommitted in the working tree on branch
`comment-cleanup`. Note other sessions edit this repo concurrently: re-check `git status` and
`git log` before assuming anything here is still true.

Session produced **three review documents** (documentation only, no code, no spend) and **one
implemented feature** (code + tests, uncommitted).

---

## 1. What was written

| Doc | Covers |
|---|---|
| `agent/app/ASSUMPTION_AUDIT.md` | Constants and policies never validated. Retrieval-focused. |
| `agent/app/ENGINE_DESIGN_REVIEW.md` | Design decisions that cost score at execution/finalize. |
| `agent/app/DAG_FORMATION_REVIEW.md` | Decisions that produce bad graph structure. |

All `file:line` citations were verified at the time of writing. Evidence is labelled
VERIFIED / MEASURED / REPORTED / HYPOTHESIS throughout; respect those labels.

---

## 2. The findings that should change what you do next

### 2a. The LangGraph question is already answered, and the target was wrong

`docs/handoffs/CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md` (~500 live cells) shows that
restricted to models LangGraph can run, `langgraph_react` **0.497** vs `graph:baseline`
**0.498** — a tie. The real gap is to this repo's own `seq_react`: **0.516 at 19k prompt tokens
vs `graph:good_adaptive` 0.479 at 58k**, and `seq_react` also wins on **chains**, the DAG's
designed best case (10W/7T/11L).

Do not plan work around "beat LangGraph." Plan it around "the scaffold costs more than it
returns, and the mechanisms are what work."

**Do not simplify toward `seq_react` wholesale either** — LangGraph cannot start 4 of 8 cheap
models, and that population is the project's central thesis. The unmeasured cell worth running
is **`seq_react` + adaptive mechanisms**.

### 2b. `IdeaDag` builds a tree, not a DAG (VERIFIED)

- `merge_nodes()` (the only multi-parent API, `idea_dag.py:96-127`) is called only from
  `idea_dag_log.py:127-128` (a demo) and tests. Never from the engine.
- `path_to_root` (`idea_dag.py:240-251`) follows `parents[0]` only.
- `EXPANSION_JSON_SCHEMA` (`idea_dag_schemas.py:16-59`) has **no `depends_on` field** — the LLM
  gets one graph-level boolean, `execute_all_children`.

A chain task is a fan-in, which a tree cannot express. This is the best structural explanation
for the chain deficit, and it predicts what Q12 already observed (fixing scheduling merely
*relocated* the deficit). Read
`docs/handoffs/RESOLVED_VALUE_CHANNEL_DESIGN_2026-08-16.md` before designing a fix — it may
already contain the intended solution; this session did not evaluate it.

### 2c. Dedup similarity math is wrong (VERIFIED, NOT FIXED — deliberate)

`got_operations.py:391` uses `1.0 - distance` on `mem_*` collections created with no metadata
(`connector_chroma.py:322,492`), so Chroma's default `l2` space returns *squared* euclidean
distance. Computed value is `2s - 1`, so the shipped threshold `0.85` actually demands cosine
**0.925**. The correct conversion already exists at `plan_library/retrieval.py:241-246`
(`similarity_from_distance`) and never propagated.

Left unfixed by explicit decision this cycle. **Fixing it increases dedup firing rate and
invalidates prior measurements through that path** — so fix, then re-measure; do not fold it
into an unrelated experiment.

### 2d. One ordering invariant explains several "dead mechanisms"

> A decision that consumes a score must run after that score exists.

Violated at `idea_engine.py:1153` (`candidates[:max_branching]` before evaluation),
`expansion.py:193-194`, `merge.py:80` (root `goal_achieved` before merge synthesis), and by
`auto_parallel_siblings` skipping batch evaluation — which keeps graphs depth-1, which makes
backtrack's 5-node requirement unreachable (0/261 firings).

**Sequencing consequence: fix this before running experiments E1-E4 from `ASSUMPTION_AUDIT.md`.**
Those A/Bs would otherwise measure mechanisms that never execute and return nulls that read as
"this technique does not help."

### 2e. Untested lead worth one cheap run

`qwen2.5:7b`: graph **0.158** vs LangGraph **0.694**. Bug-shaped, not gradient-shaped.
`connector_llm.py` sets no `num_ctx` (VERIFIED); graph sends 3x the prompt tokens; that model
needed a container-level context workaround that is not in this repo; ollama truncates silently
and drops the head (system prompt + task statement). HYPOTHESIS, untested. One cell, one model.

---

## 3. The feature that was implemented (uncommitted)

**Goal:** make the interactive debugger show every thought, and write the same content to a
readable log.

**Root cause found while building it (VERIFIED):** `connector_llm.py` recorded only counts
(`prompt_chars`, `completion_words`, ...) into `_record_io`. The prompt and completion text were
**never in the payload**, so `set_full_capture(True)` could not recover them — it only stops
summarization of a dict that never held the text. No code path at any verbosity had ever
recorded the real text. Docs claiming verbosity 3 captures "raw LLM prompts/responses" are wrong
about the LLM connector.

### Files

| File | Change |
|---|---|
| `agent/app/connector_llm.py` | records `prompt_text`/`completion_text` (+`messages`), gated on full capture; default payload byte-identical |
| `agent/app/debug_runner.py` | enables full capture + decision detail; `IDEA_DEBUG_FULL_CAPTURE=0` disables |
| `agent/app/interactive/thoughts.py` | **new** `ThoughtBuffer`: watermarks telemetry before a step, slices after, ring-buffered to 20 |
| `agent/app/interactive/thought_log.py` | **new** `ThoughtLog`: readable file sink via `IDEA_THOUGHT_LOG=<path>` |
| `agent/app/interactive/renderer.py` | `thought_card`, `decision_card`, both with `color` flag |
| `agent/app/interactive/controller.py`, `session.py` | `t`/thought and `w`/why commands; buffer wired around every step |
| `agent/app/idea_engine.py` | additive `add_step_observer()`; `on_step` untouched |

New tests: `interactive_thoughts_test.py`, `interactive_renderer_thought_test.py`,
`interactive_thought_log_test.py`, `interactive_session_thought_test.py`,
`connector_llm_full_capture_test.py`.

**Design note:** the terminal view and the log call the *same* renderer functions, the log with
`color=False`. One rendering path, two sinks, so they cannot drift.

### Verification status

**Full offline suite: 5117 passed, 18 skipped, 0 failures.** Verified after every piece landed,
including the follow-up fix below.

Smoke-testing real rendered output caught two defects that all tests missed. Both are now fixed
and re-verified against the original failing case:

1. **`alternatives` rendered as `{}`** when they are plain strings — the renderer assumed dicts,
   and its stubs agreed with the implementation rather than with real data. This is the field
   that shows dropped candidates, i.e. the review-driven part of the feature. Now renders
   `- search span [kept]` / `- visit wiki [dropped]`.
2. **Message roles were destroyed before recording** — `connector_llm.py:~253` joins all message
   contents into one blob and discards `role`, so SYSTEM and USER could not be shown separately.
   Now records the `messages` list under full capture, and the renderer role-labels each entry,
   falling back to the old blob rendering when `messages` is absent.

Lesson worth carrying: green tests did not catch either one. Render real output and read it.

### Two known gaps in the feature

- **`add_step_observer()` has no consumer.** `DebugSession` drives `engine.step()` directly and
  never goes through `_run_loop`, so the hook is working but currently dead code. It is what a
  non-interactive run would use to write a thought log. Keep or wire; do not assume it is used.
- **The debugger does not run the production loop.** `DebugSession` bypasses `_run_loop`, so the
  prune pass, backtrack check and confidence early-exit (`idea_engine.py:351-386`) **never run
  when you step**. What you watch is not what the benchmark runs. Making the stepper faithful is
  separate, larger work.

**Security/hygiene:** thought logs contain raw prompts and page content. `IDEA_THOUGHT_LOG` is
opt-in and off by default; keep log paths gitignored.

---

## 4. Recommended order of work

1. **R1** — test the `qwen2.5:7b` `num_ctx` hypothesis (§2e). One cell. Add `num_ctx` control and
   a served-context log line to `connector_llm.py` regardless of the result: a deployment-
   dependent silent truncation ceiling makes every local-model number non-reproducible.
2. **R2** — establish the evaluation-ordering invariant (§2d). Blocks E1-E4.
3. **R3** — `idea_finalize.py:97` reads truncated `content` while untruncated `content_full` sits
   in the same dict (`:244` already prefers it). One line; re-measure after.
4. **R4** — replace `_validate_goal_achievement` (`merge.py:213-242`). Note it collides with the
   merge judge's AUC 0.288 — "just trust the LLM" is not obviously better and needs measuring.
5. **R5** — per-hop extraction / value channel (§2b). Own cycle.

Then `ASSUMPTION_AUDIT.md`'s E1-E5, which are dict literals in
`idea_test_runner.py::_GOT_ARM_PROFILES` and need no new harness. **No LLM-replay layer exists**,
so a fully $0 engine run is not possible; web fixtures only zero the fetch cost.

---

## 5. Doc hygiene warnings carried forward

- `SYSTEM_STATUS.md:8-12` self-flags stale as of 2026-08-06.
- `EVALUATION_SCORE_PREDICTIVE_POWER.md` and `CONFIDENCE_JUDGE_MISCALIBRATION.md` use different
  corpora; their AUC tables are not directly comparable.
- `badmodel-lab/analyze.py` and `codebench/analyze_code.py` are destructive regenerators that
  have silently rewritten committed CSVs. Check `git status` after running either.
- This session's own reviews are dated 2026-08-19 and cite line numbers in files under
  concurrent edit. Re-verify citations before acting on any single one.
