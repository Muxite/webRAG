# GPU-window handoff: reason-first wiring, live transfer A/B (null), 3 dev cycles (2026-08-20)

Continuation of `PROMPTBENCH_V2_RESULTS_2026-08-19.md`. GPU access (RTX 3060, local Ollama)
was authorized for one night only; that window has now closed (lock released, GPU may not be
available in future sessions unless re-authorized). **Nothing in this handoff is committed —
all work sits uncommitted on `comment-cleanup`.** Full offline suite: **5203 passed, 18
skipped, 0 failed** (`PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest agent/tests
agent/app/idea_tests -q`).

**Spend: ~$0.15-0.25 OpenRouter (of a $10 ceiling). GPU time: local-only, $0.**

---

## What shipped tonight

### 1. Reason-first flags wired into the benchmark harness
Commit `d4f41e11` (2026-08-19) landed three opt-in, default-OFF flags
(`merge_goal_evaluation_first_enabled`, `verify_reason_first_enabled`,
`got_reexpand_followup_reason_first_enabled`) with prompt/schema-level test coverage, but they
weren't reachable from a benchmark run. Added: `IDEA_TEST_MERGE_GOAL_EVAL_FIRST` /
`IDEA_TEST_VERIFY_REASON_FIRST` / `IDEA_TEST_GOT_FOLLOWUP_REASON_FIRST` env overrides and a
`reason_first` arm profile in `agent/app/idea_test_runner.py`'s `_GOT_ARM_PROFILES`. New test:
`agent/tests/idea_test_runner_reason_first_flags_test.py`.

**Caveat found live:** the bare `reason_first` arm profile only sets those three keys and
inherits JSON defaults for everything else, which currently disagree with the `baseline`
profile on two other flags (`expansion_input_output_framing_enabled`,
`final_require_grounding`). A same-model, same-arm-base A/B must layer the three overrides onto
an explicit `IDEA_TEST_ARM=baseline`, not swap to `IDEA_TEST_ARM=reason_first` bare, or the
ordering effect gets confounded with those two flags.

### 2. Live transfer A/B — **null result, not a bug**
8 mixed-shape tasks (`054,085,055,061,146,147,149,122`) x 2 models (`llama3.2:3b` local,
`gpt-4.1-nano` API) x 2 reps, `graph` variant, baseline vs. baseline+reason-first-overrides.
Full numbers in the prior turn's report (not yet a separate doc — worth writing up properly if
this line of work continues). **Headline: no reliable end-to-end lift.** `llama3.2:3b` floors
on 6/8 tasks in both arms; `gpt-4.1-nason` moves slightly negative on average, dominated by one
large single-task swing at n=2. Dominant real failure mode on both weak models: under-visiting
cascades and synthesis-stage confabulation, not verdict ordering — the isolated-prompt-accuracy
result from promptbench v2 does not transfer to this end-to-end slice. This matches promptbench
v2's own caution (§9: "none of it demonstrates transfer to the agent loop").

### 3. Result-filename collision bug found and fixed
`idea_test_runner.py`'s result filename encoded run_id/test_id/model/variant/tier/repeat but
**not** which arm/override combination produced it. Running baseline then reason_first for the
same model under one run_id silently overwrote baseline's JSON — discovered mid-run, worked
around by recovering baseline numbers from captured stdout instead of the JSON files. Fixed:
new `_settings_fingerprint()` hashes the actual effective settings dict and appends an 8-hex
`_cfgXXXXXXXX` tag to every result filename, so any override combination (arm-based or ad-hoc
env flags) is disambiguated regardless of how it was produced. Test:
`agent/tests/idea_test_runner_settings_fingerprint_test.py`. Verified nothing else in the repo
parses filename internals beyond `gate_report.py`'s `{run_id}_*.json` prefix glob.

### 4. Three development cycles (parallel `engine-dev` agents, then a cross-cycle `reviewer` pass)
Sourced from two pre-existing, unread-into-code audits: `agent/app/ASSUMPTION_AUDIT.md` and
`agent/app/ENGINE_DESIGN_REVIEW.md`. Sequencing note preserved from the design review: fixes
that change graph shape/spend were flag-gated default-OFF (the d4f41e11 precedent); pure bug
fixes were left unconditional.

- **Cycle 1 — evaluation-ordering invariant** (`agent/app/idea_engine.py`,
  `agent/app/idea_policies/evaluation.py`, `agent/app/idea_finalize.py`,
  `agent/app/idea_policies/config.py`). Root cause: "a decision that consumes a score ran
  before that score existed" — candidate truncation before evaluation (arrival-order "beam"),
  a batch-scoring cap that dropped overflow candidates instead of scoring them, and
  `auto_parallel_siblings` skipping evaluation for whole batches (the documented cause of
  0/261 backtrack firings, empty early-exit calibration). New flags `beam_after_evaluation` and
  `evaluate_parallel_siblings` (both default OFF) gate the shape-changing parts; the batch-cap
  fix and a `content_full`-vs-truncated-`content` finalize bug are unconditional. 19 new tests.
- **Cycle 2 — Chroma dedup math + dead config** (`agent/app/got_operations.py`,
  `agent/app/idea_memory.py`, `agent/app/plan_library/retrieval.py`,
  `agent/app/idea_policies/candidate_coverage.py`, 3 settings JSONs). Fixed the cosine/L2
  similarity bug (`mem_*` collections default to `l2` space; `similarity = 1 - distance` was
  only correct for `cosine` — shipped `0.85` threshold really meant cosine `0.925`). Belt-and-
  braces fix: request cosine metadata on ensured collections AND read the live collection's
  actual space at the comparison site (`plan_library/retrieval.py`'s existing
  `similarity_from_distance` pattern, ported rather than reinvented). **Dedup firing rate
  changes as a result — any benchmark number through `is_duplicate_thought` before this fix is
  not comparable to one after.** Also deleted `leaf_chroma_results`/`default_semantic_results`
  (confirmed via `git log -S` archaeology: never had a reader, ever). 14 new tests.
- **Cycle 3 — Chroma query-result cache** (`services/shared/connector_config.py`,
  `agent/app/connector_chroma.py`). No LLM-call cache was built (evaluation prompts embed
  per-run node IDs, so a content-hash key would almost never hit, plus temperature sampling is
  in play) — instead, an in-process (deliberately not disk-backed, since collections are
  mutable across runs) query-result cache on `ConnectorChroma.query_chroma`/`add_to_chroma`,
  which also caches embeddings implicitly since Chroma embeds internally during `.query`. Keyed
  by collection + a per-collection write-epoch + query params + embedder identity; invalidated
  on any write/teardown. Default ON via `CHROMA_QUERY_CACHE` env. 20 new tests.
- **Cross-cycle review**: no correctness bugs, seams between cycles hold (specifically checked:
  does Cycle 2's `ensure_collection()` interact badly with Cycle 3's write-epoch cache — no,
  proven independent since the cache stores raw distances and the cosine/L2 conversion happens
  downstream). One pre-existing, out-of-scope duplication flagged for a future cleanup only
  (`strategy_library/retrieval.py`'s private `_read_space()` duplicates the now-public
  `plan_library.retrieval.read_collection_space()`).

---

## Methodology notes for whoever picks this up

**How this session was structured** (worth repeating — it worked well): plan-mode research via
parallel `Explore` agents → write the plan to the plan file → `AskUserQuestion` when scope
genuinely needed a user decision (GPU window, budget) → `ExitPlanMode` → execute via 3 parallel
`engine-dev` agents (one per cycle, each doing its own TDD find→fix→test) → a dedicated
`reviewer` agent scoped ONLY to the touched files (not the whole repo's uncommitted diff, which
has ~90+ files from other concurrent sessions) → a `benchmark` agent for the live run.

**Real friction hit twice, worth knowing in advance:** the `benchmark` subagent kept ending its
turn early with "waiting for the async job to complete," assuming it would be automatically
resumed. **Subagents are not automatically resumed** — only the parent session gets a
notification when a subagent stops, and the parent must explicitly `SendMessage` back in to
resume it. This cost real wall-clock time twice before the subagent was told explicitly to
block with a polling loop (`while pgrep -f idea_test_runner; do sleep 15; done`) inside a single
tool call rather than ending its turn on an assumption. If you dispatch a `benchmark` agent (or
any agent) to babysit a long-running background process, say this explicitly up front.

**Tips on saving context (used this session, worth reusing):**
- Fan out `Explore` agents in parallel for pure research (finding docs, tracing call sites) —
  their tool output never touches the parent's context, only the summary does. Used 4x tonight
  before any code was written.
- Scope a `reviewer`/`code-review` pass to an explicit file list, not "review the diff" — this
  branch always has a large pre-existing uncommitted diff from concurrent sessions
  (`[[feedback_concurrent_sessions_git_state]]`), and an unscoped review would burn tokens
  reviewing other people's in-flight work.
- Give each parallel `engine-dev` agent the exact file/line pointers already gathered during
  research, not "go find the bug" — they land faster and don't re-derive what's already known.
  All three cycle prompts in this session included concrete file:line references up front.
- The plan file (`/home/muk/.claude/plans/*.md`) is a durable, low-token anchor — re-reading it
  costs far less than re-deriving scope from conversation history after a compaction.

---

## Open items for the next session

1. **Nothing is committed.** Files touched: `agent/app/idea_test_runner.py`,
   `agent/app/idea_engine.py`, `agent/app/idea_policies/evaluation.py`,
   `agent/app/idea_finalize.py`, `agent/app/idea_policies/config.py`,
   `agent/app/idea_policies/candidate_coverage.py`, `agent/app/got_operations.py`,
   `agent/app/idea_memory.py`, `agent/app/plan_library/retrieval.py`,
   `agent/app/connector_chroma.py`, `services/shared/connector_config.py`,
   `agent/app/idea_dag_settings.json` (+`.baseline`/`.good_adaptive`), `agent/app/ASSUMPTION_AUDIT.md`,
   plus ~12 new test files under `agent/tests/`. All isolated from the rest of the branch's
   pre-existing uncommitted diff (comment-cleanup pass, unrelated). Consider committing this
   subset specifically rather than everything on the branch.
2. **Part B items 2/3 never ran**: the PromptBench v2 `llama3.2:3b` deferred slice (reopens H4,
   model-size interaction) and the `MODEL_TIER_LIST.md` reachable-tier gap-fill for
   `llama3.2:3b`/`gemma2:2b`/`llama3.1:8b`. Both need GPU access re-authorized.
3. **The reason-first null result** needs a proper write-up (`PROMPTBENCH_V2_RESULTS_2026-08-19.md`-style)
   if this line of work continues, and arguably means the flags should NOT be flipped on by
   default anywhere yet — the isolated-prompt win didn't transfer.
4. Per `ENGINE_DESIGN_REVIEW.md`'s ranked next actions, R1 (qwen2.5:7b `num_ctx` truncation
   hypothesis) and R4/R5 (merge goal-achieved tautology, per-hop extraction step) remain
   unstarted — R2 (this session's Cycle 1) was the prerequisite; it's now done, so the
   assumption-audit's E1/E3/E4 live experiments (dedup ablation, similarity floor, chunk-size
   reconciliation) are now unblocked, not just E1/E5 (Cycle 2 already resolved E5).
