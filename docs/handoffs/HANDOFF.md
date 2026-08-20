# webRAG / Euglena — Session Handoff (2026-08-10)

> Supersedes the 2026-07-10 version below this line's era entirely — that content (the adaptive
> distillation research line, Phases 1-5) is now historical; see
> `agent/app/ADAPTIVE_DISTILLATION_HANDOFF.md` if you need it. **Start here for current state.**

Start here. This is the project-wide pickup doc.

## The single biggest change: there's now a dev-cycle methodology

`docs/DEV_CYCLE.md` replaces the old one-off-`*_HANDOFF.md`-per-session habit with a repeatable
loop: **Plan → Adversarial review → Write/adjust tests → Adjust benchmarks → Implement → Run tests
→ Run benchmarks → Review results → Analyze**, sized Micro/Small/Medium/Large, with a per-stage
tooling map (which subagent/script owns each stage) so nothing needs improvising. A cycle is one of
**feature, bug fix, cleanup, or a branch merge into master** — or a combination. Invoke it via the
`/cycle` skill (`.claude/skills/cycle/SKILL.md`), which reads the doc fresh and helps size + route
whatever you bring to it. **Read `docs/DEV_CYCLE.md` before starting new work** — it's short and
it's the actual source of truth, this handoff just orients you toward it.

Two real cycles have run so far, both fully committed on `master`:

- **Cycle 1** (2026-08-09) — designed the methodology itself, then dogfooded it: shipped a shared
  sandbox-tool dispatcher (`agent/app/sandbox_dispatch.py`) fixing a real crash bug that predated
  the cycle, a cross-benchmark reporting tool (`scripts/unified_bench_report.py`), and 3 fixes to
  unblock the barrage relaunch (`suite59` task set, infra-failure quarantine, arm-symmetric retry).
  Full audit found the barrage's old fix-list handoff was 29/33 already stale-shipped — don't trust
  old handoff docs' checklists without re-diffing against `HEAD`.
- **Cycle 2** (2026-08-09/10) — folded `badmodel-lab/codebench/` (the Docker coding-benchmark
  harness) into a top-level `codebench/` directory, since it was infrastructure, not lab-specific
  mitigation content. Two research passes before implementation caught real structural bugs (bash
  scripts that computed their root-anchor by counting directory levels — silently wrong once the
  directory moved one level shallower; a Python default-path that would've silently returned `[]`
  instead of crashing). Verified live post-move: rebuilt all 5 Docker images, ran a real sandbox→
  grade→record cycle at the new path, confirmed both reporting tools pick it up.

## Versioning: DAG v1 → Compiled v1 → DAG v2 → v3 (planned)

The engine's history now has names, established 2026-08-14 — use them going forward instead of
"the adaptive engine work" / "the compiled thing": **DAG v1** (2026-02–03 native GoT rewrite, not
compiled), **Compiled v1** (2026-06 offline-plan campaign), **DAG v2** (2026-07–present, the
generation everything in this handoff belongs to), **v3** (planned). Full glossary and rationale:
root `README.md#versioning`.

**DAG v2's remaining scope, as decided 2026-08-14:** finish the generation with a hugely expanded,
harder benchmark set than DAG v1's — including some repeated DAG v1 tasks to measure direct
improvement, dropping the sequential-mode comparison arm DAG v1 used, and adding a new arm
comparing the same model run through an off-the-shelf, publicly available agent system in current
use. Codebench and additional tool/capability integrations are deliberately deferred to v3.

**v3's scope:** move past one-shot mandates toward a more continuable, chatbot-like interaction —
stopping mid-run and picking a task back up, rather than only submit-and-wait for a deliverable.

**Open, not yet decided:** whether task continuation (resuming/extending a run rather than
one-shot submit-and-wait — something neither DAG v1 nor DAG v2 handles well today) ships as part
of finishing DAG v2 or waits for v3. Left for whichever turns out more practical once DAG v2's
benchmark work is underway.

**Ladder-rung correction found 2026-08-14, while investigating whether to flip two flags
default-on ahead of the relaunch:** `agent/app/BENCHMARK_SUITE_50.md`'s rung table (still 3 rungs,
`baseline → good_adaptive → full`) is **stale**. `scripts/LADDER_PREREGISTRATION.md` and
`agent/app/TECHNIQUE_INVENTORY.md` (both more recent) record that the old `full` arm — which
bundled k-vote + backtrack + expect-contract + reasoning-effort-discipline + price-tier-tiering —
was **measured net-negative (−0.003 nano, −0.075 deepseek at ~2× cost) and already dropped**. It's
been replaced by a `max_burn` arm (`good_adaptive` + deeper re-expansion + wider hop/beam + the
finalize reconcile chain), which explicitly excludes reasoning-effort-discipline as "a no-op/
wrong-direction" lever in this context, plus price-tier-tiering, backtrack, and expect-contract.
The real current ladder is `baseline → good_adaptive → max_burn`, not the 3-rung table
`BENCHMARK_SUITE_50.md` still shows. Whoever scopes the DAG v2 relaunch's runner config should use
`max_burn`, not `full`, and should NOT flip `native_reasoning_effort_discipline_enabled` to
default-on — that contradicts this more recent finding. `BENCHMARK_SUITE_50.md` itself hasn't been
corrected yet (out of scope for this pass; flagging so the next person doesn't work from the stale
table).

**`expansion_input_output_framing_enabled` flipped default-on 2026-08-14** (live-proven
2026-08-06, see `README.md#versioning`'s DAG v2 section) — `idea_dag_settings.json` and
`ExpansionConfig`'s Python default both now `True`. The `baseline` arm profile
(`idea_test_runner.py::_GOT_ARM_PROFILES`) pins it back to `False` explicitly so the "adaptive
OFF" ladder rung stays byte-identical; 8 offline tests that had hard-coded the old "default off"
assumption were updated to match (full suite green: 4674 passed / 18 skipped).

## DAG v2 relaunch preflight — STOPPED, agenda handed off (2026-08-15)

The relaunch was about to run and was **stopped by an adversarial review**. Four
benchmark-invalidating bugs were found and fixed; one widely-propagated documented claim was
retracted as false. A $0.045 / 13-cell live smoke on `openai/gpt-4.1-nano` produced the first
DAG v2 vs LangGraph numbers. **All of it is uncommitted (37 files).**

- **Evidence:** `docs/handoffs/DAG_V2_PREFLIGHT_2026-08-15.md` — the bugs, the retraction, the
  smoke table, the cheap-relaunch costing (~$20–30 vs ~$60).
- **Agenda:** `docs/handoffs/BENCHMARK_POLICY_HANDOFF_2026-08-15.md` — background, open policy
  questions Q1–Q8, ranked DAG v2 improvement candidates, working agreements.

**Superseded later the same day by a ~525-cell live sweep ($1.35).** Start here instead:

- **`docs/handoffs/CAPABILITY_SPECTRUM_PREREG_2026-08-15.md`** — design + falsification criteria,
  written before any cell ran.
- **`docs/handoffs/CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md`** — 8 models (tinyllama 1.1B →
  cheap APIs) × 5 arms. Headlines: off-the-shelf `create_react_agent` **cannot run 4 of 8 models**
  (no tool-calling endpoint, reproduced on ollama *and* OpenRouter); the active-59 is 56/59 at
  difficulty ≥8 so it cannot measure weak models; the token premium is re-sent context, not
  reasoning (34:1 in:out, and raising LangGraph's budget burned 4.9× the tokens for +0.016).
  Six recorded corrections, including one thesis-supporting result published at n=6 and retracted
  at n=10. **Standing contract adopted: arm comparisons are paired by (model, task) and
  run-complete.**
- **`docs/handoffs/SHAPE_ADAPTATION_HANDOFF_2026-08-15.md`** — the current thread. A causal
  diagnosis of why DAG v2 loses on sequential tasks: `classify_shape` labels the live chain set
  **1/10**, `detect_state_dependencies`' dataflow path has fired **0 times in 476 cells**, so
  `AUTO-PARALLEL` batches chain hops into one step and hop 2 never sees hop 1's answer.
  **The blocker for shape-adaptive work is the sensor, not the strategy layer.** Carries the
  in-flight `auto_parallel_siblings: false` experiment and what to do next.

Worst bug, for context: grounding evidence was sourced from `result["graph"]`, which only 2 of 7
execution variants populate. On a real cell this scored `sequential_react` 0.417 instead of 0.944
against the graph arm's 0.750 — i.e. it would have reported the graph engine winning a comparison
it actually lost. Fixed via telemetry-projected `observability["evidence"]`, with a cross-arm
parity test.

Headline open question: DAG v2 spends 7–13× the tokens while making **fewer** tool calls than a
linear agent, and ties with LangGraph on fan-out/disambiguation shapes at ~1/10th the cost. It
wins clearly on the 3-hop chain shape (0.767 vs 0.575, and +0.48 over its own baseline). Whether
the token premium should be redirected into evidence volume is Q2 and is the cheapest high-value
experiment available.

## Git state

**Work now happens directly on `master`**, not a long-lived feature branch. `master` is the only
local branch as of 2026-08-14; re-run `git branch -a -v` at the start of a branch-merge cycle
(`docs/DEV_CYCLE.md`) to catch anything new since.

| Branch | Where | Status |
|---|---|---|
| `master` | local + `origin/master` | **The one branch anyone should commit to.** ~162 commits ahead of `origin/master`, **not pushed** (push is a separate, explicit decision each time, not a standing default) — `origin/master` hasn't moved since 2026-06-16. |
| `compiled-scaffold-dag` | deleted locally 2026-08-14 | The prior working branch; fast-forward-merged into `master` on 2026-08-09, then deleted once retired. `git branch -d` refused it (its upstream `origin/compiled-scaffold-dag` never received those commits, so git's merge check fails against upstream even though it's a strict ancestor of `master` — confirmed via `git merge-base --is-ancestor` before forcing); deleted with `-D`, safely, since master already contains everything. Recovery SHA if ever needed: `f22a55c0`. |
| `origin/compiled-scaffold-dag` | remote-only, untouched | Stale (last remote commit 2026-07-10, ~1 month behind the local branch it tracked). Left alone — harmless, never needs pushing, wasn't in scope for the 2026-08-14 cleanup. |
| `autoscale`, `autoscale-redux` | deleted, both local (2026-08-10) and remote (2026-08-14) | **Fully gone.** Evaluated 2026-08-10 as fully superseded (see below), local copies deleted then; remote copies (`origin/autoscale`, `origin/autoscale-redux`) pruned 2026-08-14 via `git push origin --delete`. Recovery SHAs if ever needed: `autoscale`=`57f43e54`, `autoscale-redux`=`8f3efd78`. `autoscale`'s real contribution (`services/lambda_autoscaling/lambda_function.py`) was independently redone via a later, cleaner commit (`1fbed65c "autoscale complete"`, 2026-02-07), itself now archived under `services/_legacy-aws/`; its frontend components targeted a `frontend/src/components/` layout that no longer exists post-rebuild. `autoscale-redux` was a strict `git` ancestor of `master` at evaluation time (zero unique commits) — fully superseded, not merged. |

Only `origin/master` (behind) and `origin/compiled-scaffold-dag` (stale, harmless) remain as loose
remote state; neither is acted on automatically by anything in this repo.

## Current test baseline

`PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests` →
**4664 passed, 18 skipped, 0 failed** (as of commit `935ebb22`). Note the `.:services:agent`
PYTHONPATH — `services/agent/` was restructured to a top-level `agent/` directory on 2026-08-09
(concurrent session, `8d45df3a`); the old `PYTHONPATH=services:services/agent` form is stale.

## What's open — candidates for the next cycle, roughly in priority order

**All six items below are resolved**, five with no further action pending and one (item 6) with a
newly-confirmed-necessary follow-up. Kept struck-through, not deleted, for the reasoning trail.
**Both live re-verifications authorized 2026-08-14 (budget: $2, both approved) are now DONE.**

**Candidates for the next (DAG v2) cycle, surfaced by this pass:**
1. **The full barrage relaunch is unblocked** (item 1's fix is live-confirmed) but **not yet
   authorized or run** — running it is the obvious next Medium/Large cycle. Per the 2026-08-14
   versioning/roadmap plan, fold in the expanded/harder benchmark set, repeated DAG v1 tasks, the
   dropped sequential-mode arm, and the `diverse_ground` A/B into this same relaunch rather than
   spending twice — see `README.md#versioning` and `.claude/plans/cheeky-seeking-sedgewick.md`.
2. **Search-leaf query composition bug** — AND-joins multi-entity names into one query instead of
   OR/splitting per-entity, found live on task 084 (fails safely, not a regression). Worth flagging
   to `strategy-tuner`: no "reformulate on repeated zero-result retry" fallback exists for this
   shape.

**Moved to v3-scope, no longer DAG v2 next-cycle candidates (2026-08-14):** codebench's
write-protocol redesign (item 6 below) and its image-freshness guard (see Provenance lesson 4
below) are both codebench-specific — per today's versioning doc, codebench is v3-bucket material,
not DAG v2's benchmark story. Both remain real, still-open items; they just don't compete for the
next DAG v2 cycle's slot anymore. Revisit when v3 work starts.

1. **`good_adaptive` self-loop fix — CONFIRMED LIVE.** Fresh smoke (`reverify_selfloop_20260814`,
   `openai/gpt-5-mini`, tasks 052/084, baseline+good_adaptive, $0.26 of the $2 ceiling spent).
   Task 052/good_adaptive directly re-exercised the fixed re-expansion code path and completed
   cleanly in 12/50 steps (`pending_nodes_count=0`, score 0.2917 vs baseline 0.2083). Task 084
   showed no trace of the old deadlock signature (pre-fix: burned to step 49/50, 1 node stuck
   "pending", `visits=0`; post-fix: both arms stopped cleanly at step 5/50 with 0 pending nodes) —
   confirmed by absence. **The full barrage relaunch can now trust `good_adaptive` numbers.** A
   genuinely new, unrelated failure mode surfaced during this check (see below, not a regression).

6. **codebench syntax-check fix — mechanism confirmed working, but INSUFFICIENT to recover
   scores. The deeper write-protocol fix is now confirmed necessary, not just "still open."**
   First re-run attempt was invalid — caught and corrected mid-session: `codebench-badmodel`'s
   Dockerfile `COPY agent /app/agent`s at BUILD time, not a live mount, and the image hadn't been
   rebuilt since before the fix commit, so the first live rerun silently exercised stale pre-fix
   code. Rebuilt the image (`docker build -f codebench/agents/badmodel/Dockerfile -t
   codebench-badmodel:latest .`) and re-ran (`reverify_syntaxfix_take2_20260814`, qwen2.5:14b,
   the 6 worst-hit tasks: c30/c35/c36/c42/c22/c40) — confirmed the fix's own compile-check *is*
   present and firing in the rebuilt image (verified by extracting the module from the image
   directly). Result: **only 1/6 tasks (c35) fully recovered** (score 1.0, clean compile). **4/6
   (c30, c36, c42, c40) still ended up with a syntax-corrupted final submission** — independently
   confirmed by compiling each raw output file directly (unterminated triple-quote, unindent
   mismatch, unmatched paren — exactly the corruption classes the check targets). The run summaries'
   `"last successful action"` for the still-broken cells is often `read_file`, not `write_file`:
   the model saw the `SyntaxError` observation but never landed a second passing `write_file` to
   the same path before its 10-step budget ran out, and `write_file` doesn't retract the bad
   content it already persisted to disk when the compile-check downgrades it to a failure — so a
   model that can't successfully self-correct just submits whatever was last written, broken or
   not. (c22 scored 0.0 for a different reason this run — a genuine logic failure, `pytest_rc=1`,
   not corruption; but the original corruption on c22 came via `patch_file` in the first
   [invalid-image] rerun, which the syntax-check fix doesn't cover at all — scoped to `write_file`
   only.) **Conclusion: the fix converts a 100%-silent failure into a visible, actionable one (as
   designed), but qwen2.5:14b's self-correction success rate within the existing step budget is too
   low (1/6 here) to call this "recovered."** The invasive fix flagged as a fallback — moving
   `write_file`'s wire format off one-shot full-file JSON strings toward something closer to
   aider's diff/search-replace format (0% corruption on the identical model+tasks) — is the
   confirmed next step, not an optional one. A `patch_file` variant of the same compile-check
   (currently a real gap) should go with it.

**Process lesson from this reverification pass**: codebench live verification is silently invalid
unless the Docker image is rebuilt after any engine code change — nothing catches this
automatically today (no image-freshness check, no CI gate). Worth a cheap guard (e.g. stamp the
image with the git SHA it was built from, warn if it doesn't match `HEAD`) if codebench live
verification becomes routine.

**New failure mode found during the `good_adaptive` reverification (unrelated to that fix, fails
safely, not a regression)**: on task 084's multi-entity no-URL shape, the search-leaf's retry logic
composed one `site:en.wikipedia.org "name1" "name2" ...` query AND-ing all six target lake names
together instead of OR-ing or splitting per-entity — virtually guaranteed 0 results, since no single
page names all six. Both arms retried the *identical* failing query twice rather than reformulating,
then the grounding gate correctly refused to fabricate an answer rather than hallucinate. Confirmed
via a prior pre-fix run of the same task, which instead composed an OR'd query and got 10 results —
this is model-sampling variance in query construction, not a regression from the self-loop fix
(untouched code path). Worth flagging to `strategy-tuner`: no "reformulate on repeated zero-result
retry" fallback exists for this shape.

1. ~~**`good_adaptive`'s self-loop bug**~~ — root-caused and fixed offline 2026-08-10
   (`eb8cc9ac`), **but still needs a live re-verification smoke before the barrage relaunch is
   unblocked** (never spend without explicit go-ahead at execution time). Root cause turned out to
   be TWO independent deadlocks, both silent (no exception, no error field — the run just burns its
   whole step budget returning the same node id every step):
   - **Self-sourced `requires_data`.** Re-expanding a completed leaf (e.g. a search) passes the
     leaf's own `node_id` into `LlmExpansionPolicy._parse_candidates` as `parent_node_id`. If the
     model's follow-up is a bare "visit" with no explicit URL, `_extract_url_from_path_context_with_
     source` walks `graph.path_to_root(parent_node_id)` — INCLUSIVE of the node itself — and on the
     exact "given no URLs, search then visit" shape, the leaf's own just-produced search result is
     the only URL source on that path. The new visit child's `requires_data.source_node_id` gets
     wired back to its own parent, which can only reach `DONE` once that same child finishes —
     `IdeaDagEngine.step()`'s "wait for required data" gate then returns the source node id (itself)
     forever. Fixed in `expansion.py`: skip `requires_data` entirely when the resolved source is the
     node currently being expanded (the URL is already in hand, nothing to wait for).
   - **Merge-creation self-loop for <2 children.** Once leaf #1 was fixed, the SAME test kept
     failing one level up: any ordinary (non-reexpanded) node with a single completed child and no
     merge node hits `_handle_merge_creation`, whose `should_create_merge_node` is a deterministic
     function of child count (`< 2` children -> always `False`) — so `return node_id` on that branch
     re-evaluates identically every step, forever. A near-identical guard already existed in
     `step()` but was scoped only to `_got_reexpanded` nodes (its own comment described this exact
     failure mode); broadened it to apply whenever `should_create_merge_node` is `False`, regardless
     of re-expansion, matching the precedent that guard already set.
   Reproduced offline with a real (non-fake) `LlmExpansionPolicy` + a full `engine.step()` drive
   loop in `agent/tests/reexpand_self_source_deadlock_test.py` — both tests failed against
   pre-fix code (one hanging at each deadlock in turn) and pass now; full suite green (4660
   passed/18 skipped, +2 new). Full detail in `[[project_ladder_benchmark]]` memory.
2. ~~**The QA-lab fold-in**~~ — evaluated 2026-08-11, **no fold-in needed**. Full reasoning in
   `docs/superpowers/specs/2026-08-11-qa-lab-fold-in-evaluation-design.md`: the codebench precedent
   doesn't generalize (codebench was infrastructure that had drifted into a lab-scoped directory;
   `roster.yaml`/`tiers.yaml`/`profiles/`/`cells.jsonl` are genuinely lab-specific *experiment
   configuration*, the intentional "big specific library" half of the capability-continuum
   philosophy). The cross-cutting reporting need this was meant to solve is already solved
   structurally by `scripts/unified_bench_report.py`'s dual-read (no shared directory required),
   and the genuinely unfinished piece (field-name bridging) is already correctly scoped in
   `agent/app/AGENT_CONTINUUM.md`'s own roadmap item 4, in progress. `badmodel-lab/analyze.py`,
   `results/cells.jsonl`, `roster.yaml`, `tiers.yaml`, `profiles/` all stay where they are.
3. ~~**Two small, already-identified cleanup items**~~ — investigated 2026-08-10, **both turn out
   to be "don't touch," not deletions**:
   - `badmodel-lab/localagent/` retirement condition ("once the graph engine matches or exceeds it
     on localagent's own task suite") is explicitly **NOT met yet**: `agent/app/TECHNIQUE_INVENTORY.md`
     itself says, as of the most current status tracking, "`SandboxToolPack` (file/shell tools) —
     architecturally sound, **zero accuracy-lift data yet**." `localagent/RESULTS_P1.md` has real
     per-model Wilson-lower numbers on its 6-task suite (file_count/find/write/memory/web_fact/
     cross_cutting) to compare against — nobody has run the graph engine against that same suite to
     produce the other half of the comparison. Not a quick check; needs its own live/local
     benchmark validation before retirement is justified.
   - `badmodel-lab/playground/pkg/connector_search_searxng.py` is **not stale** — it's actively
     imported by `badmodel-lab/playground/pkg/chat_entrypoint.py`. Main's own docstring's "a twin
     that predates it" was read too literally; the fuller docstring explains it's a *deliberate*
     packaging-boundary duplicate ("kept where it is because the playground image ships its own
     package"), not accidental drift. Deleting it would break the playground Docker image's
     imports.
4. ~~**Branch-merge evaluation** for `autoscale`/`autoscale-redux`~~ — done 2026-08-10, both deleted
   as fully superseded. See Git state above.
5. ~~**~16 `scripts/*.sh` benchmark drivers export the wrong search-provider key.**~~ — fixed
   2026-08-10 (`935ebb22`). All 17 affected drivers (the count was slightly off — 17, not ~16) now
   `export SERPER_KEY="$(keyval SERPER_KEY)"` (or the inline grep/sed form for the 2 drivers that
   don't define `keyval()`) alongside the existing `SEARCH_API_KEY` export. Verified the existing
   `keyval()` helper already strips `services/keys.env`'s double-quote-wrapping correctly (extracted
   a clean 40-char key, no stray quotes) — that half of the concern was already handled, just never
   wired to `SERPER_KEY`. Added `agent/tests/search_key_wiring_test.py` — a regression guard that
   fails if any future `scripts/*.sh` driver sources `SEARCH_API_KEY` from `keys.env` without also
   sourcing `SERPER_KEY`, so this exact gap can't reappear silently. Full suite green (4664/18
   skipped). Serper itself was already confirmed live and working — this was purely a wiring gap.
6. ~~**codebench's JSON-embedded source-code write protocol corrupts ~40% of qwen2.5:14b's
   `badmodel` submissions**~~ — partially addressed offline 2026-08-10 (`d702c697`). Traced the
   corruption to the MODEL's own output: `write_file`'s `content` string is syntactically valid
   JSON (so it parses cleanly) but semantically broken Python (double-escaped newlines, collapsed
   triple-quotes) — invisible to JSON validation, and nothing downstream ever compiled the file
   back, so runs often burned their whole 10-step budget never discovering the submission was dead
   on arrival. Fix: `execution_compiled_code.py`'s leaf loop now `compile()`-checks any `.py` file
   immediately after a successful `write_file` and downgrades the step to a failure with the
   `SyntaxError` message when it doesn't compile — giving the model a chance to self-correct within
   its existing write→run→read-failure→patch→re-run loop, same as it already does for pytest
   failures. Deliberately scoped to `execution_compiled_code.py` (codebench-specific), not the
   shared `SandboxConnector`/`sandbox_dispatch` the native engine's language-agnostic sandbox pack
   also uses. **This does not fix the model's underlying escaping habit** — it converts a silent
   failure into a visible, actionable one. **Live-reverified 2026-08-14** (see the summary at the
   top of this section): confirmed the mechanism fires correctly, but self-correction within the
   existing step budget only succeeded 1/6 times — not enough to call this "recovered." The more
   invasive fix (moving `write_file`'s wire format away from one-shot full-file JSON strings toward
   something like aider's diff/search-replace, which had 0% corruption on the same model+tasks) is
   now the confirmed next step.

**Track 3 of cycle 1 (small filler) is done, committed as `1871a71d`.** Findings, for context on
anything that references them later: `m02`'s zero-variance 0.50 score was a grounding-regex gap
(missed the JSON-escaped rendering of a Wikipedia redirect slug), root-caused and fixed against 48
real replayed result files. Task 088 is now wired to the `ratio_argmax` composer (same opt-in
kill-switch pattern as 084/091). Task 079 was deliberately **not** wired — its numerator unit is
heterogeneous across items (TWh vs GWh) and the composer only supports one global unit, so forcing
it would silently mislabel a converted figure; a negative-case test pins exactly why, so nobody
re-attempts this the naive way later. Item 5 above (the Serper key-wiring gap) was found as a
byproduct of this item's Serper-liveness check.

**Explicitly not authorized by any existing plan**: the full barrage launch (item 1's blocker is now
cleared as of the 2026-08-14 live reverification, but the launch itself is still a separate,
not-yet-requested decision), any codebench live-matrix scale-up past a single smoke cell, and wiring
codebench's LLM-judge for soft-task grading (spends real money per grade, deliberately left for
explicit authorization). Implementing item 6's now-confirmed-necessary write-protocol change is also
not yet authorized — it needs its own Plan stage (this is a Medium-sized change, not a quick follow-
on) before any code gets written.

## How to run things (live = real $)

```bash
export PYTHONPATH=.:services:agent
export IDEA_TEST_CONCURRENCY=1   # MANDATORY (shared connectors)
./.venv/bin/python -m agent.app.idea_test_runner
```
Offline tests (no $): `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests`.

Live benchmark runs are gated by **two distinct, non-interchangeable locks** — see
`docs/DEV_CYCLE.md`'s Parallelism section: the `benchmark` subagent's OpenRouter singleton
(`concurrency=1`) for anything going through `idea_test_runner`/`adaptive_ladder_run.py`, and a
separate local `gpu-lock` (`/home/muk/projects/gpu-lock` — `acquire`/`release`/`status`) for
Ollama-contending local-model work like codebench. They don't contend with each other.

Commit convention (this repo, not the default): a single lowercase line, no punctuation, no body,
no trailer.

## Where the detail lives

- `docs/DEV_CYCLE.md` — the methodology itself; read this, not just this handoff.
- `docs/superpowers/specs/2026-08-08-codebench-tooling-and-benchmark-unification-design.md` and
  `2026-08-09-codebench-fold-in-design.md` — the two design specs behind cycles 1 and 2's codebench
  work (historical/point-in-time by this repo's spec convention, not living docs).
- `agent/app/AGENT_CONTINUUM.md` — the architecture doc framing `badmodel-lab`'s deliberate split
  from main, and what's expected to bridge vs. stay separate.
- `agent/app/COST_BENCHMARK_HANDOFF.md`, `ADAPTIVE_DISTILLATION_HANDOFF.md`, `SYSTEM_STATUS.md`,
  `RESEARCH_NOTES.md` — the prior research line's deep logs (2026-07 era, still accurate for that
  scope, just not current-state).
