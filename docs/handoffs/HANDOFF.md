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

## Git state

**Work now happens directly on `master`**, not a long-lived feature branch. `compiled-scaffold-dag`
(the prior working branch, 122+ commits ahead of its own remote) was fast-forward-merged into
`master` on 2026-08-09 and should be treated as retired — don't keep committing to it. `master` is
currently ~149 commits ahead of `origin/master`, **not pushed** (push is a separate, explicit
decision each time, not a standing default).

`autoscale` and `autoscale-redux` were evaluated (2026-08-10) and deleted — both fully superseded.
`autoscale-redux` was a strict `git` ancestor of `master` (zero unique commits). `autoscale`'s real
contribution (`services/lambda_autoscaling/lambda_function.py`) was independently redone and merged
via a later, cleaner commit (`1fbed65c "autoscale complete"`, 2026-02-07) that's *itself* now
archived under `services/_legacy-aws/` since the project no longer deploys via AWS ECS; its frontend
components (`TaskCard.tsx`, `StatusBar.tsx`, etc.) targeted a `frontend/src/components/` layout that
no longer exists post-rebuild (`frontend/src/app/...`). Deleted SHAs for recovery if ever needed:
`autoscale`=`57f43e54`, `autoscale-redux`=`8f3efd78`.

One more branch ref is worth a follow-up: local `compiled-scaffold-dag` (confirmed a `git` ancestor
of `master` — genuinely merged) still can't `git branch -d` cleanly because its *remote* tracking
ref (`origin/compiled-scaffold-dag`) is stale relative to local `HEAD`, not because the merge is in
question. Needs `-D` (force) or a remote-ref update to clean up; left alone for now since that's a
step past what was evaluated this pass.

## Current test baseline

`PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests` →
**4660 passed, 18 skipped, 0 failed** (as of commit `eb8cc9ac`). Note the `.:services:agent`
PYTHONPATH — `services/agent/` was restructured to a top-level `agent/` directory on 2026-08-09
(concurrent session, `8d45df3a`); the old `PYTHONPATH=services:services/agent` form is stale.

## What's open — candidates for the next cycle, roughly in priority order

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
2. **The QA-lab fold-in** — deliberately deferred out of cycle 2's scope. `badmodel-lab/analyze.py`,
   `results/cells.jsonl`, `roster.yaml`/`tiers.yaml`/`profiles/` are still lab-scoped.
   `agent/app/AGENT_CONTINUUM.md` names specific `cells.jsonl` fields as ones that "may never
   bridge" into main's schema — that tension needs its own Plan stage before deciding what "fold
   in" even means here, not a rushed follow-on to cycle 2.
3. **Two small, already-identified cleanup items**, found during cycle 2's `badmodel-lab`
   inventory but out of scope there: `badmodel-lab/localagent/` (a control-loop precursor
   `AGENT_CONTINUUM.md` already says to retire "once the graph engine matches or exceeds it" — that
   condition may already be true, worth checking) and `badmodel-lab/playground/pkg/
   connector_search_searxng.py` (main's own `connector_search_searxng.py` docstring calls it "a
   twin... that predates" it — an acknowledged stale duplicate).
4. ~~**Branch-merge evaluation** for `autoscale`/`autoscale-redux`~~ — done 2026-08-10, both deleted
   as fully superseded. See Git state above.
5. **~16 `scripts/*.sh` benchmark drivers export the wrong search-provider key.** Found while
   closing out Track 3 (below): `ConnectorConfig.search_provider` defaults to `"serper"`, but only
   `badmodel-lab/run_cell.sh`/`run_adaptive_cell.sh` actually export `SERPER_KEY`. Every
   `scripts/*.sh` driver (including all `barrage_continue*.sh`) still only exports the old,
   documented-stale `SEARCH_API_KEY` (Brave) — so any of those drivers hands the Serper backend a
   Brave key and gets a silent 403 on every search. One-line fix per driver
   (`export SERPER_KEY="$(keyval SERPER_KEY)"`), not yet applied anywhere outside `badmodel-lab`.
   Also: `services/keys.env`'s `SERPER_KEY` value is double-quote-wrapped — a naive parser that
   doesn't strip the quotes will also 403 and look identical to "Serper is down." Serper itself
   **is live and working** (verified with a real request, 200 OK) — this is a wiring gap, not an
   outage.
6. **codebench's JSON-embedded source-code write protocol corrupts ~40% of qwen2.5:14b's `badmodel`
   submissions** (SyntaxError on compile — triple-quote miscounting and double-escaped newlines),
   vs 0% for aider on the identical tasks. Root-caused with file-level examples in
   `docs/handoffs/AGENT_FAILURE_MODES_2026-08-10.md` (2026-08-10 read-only failure-mode analysis) —
   the single most concrete, actionable item that analysis surfaced. Well-scoped bug-fix cycle
   candidate: move `write_file`'s one-shot full-file JSON string toward something closer to aider's
   diff/search-replace format, or add a repair/validation pass before grading.

**Track 3 of cycle 1 (small filler) is done, committed as `1871a71d`.** Findings, for context on
anything that references them later: `m02`'s zero-variance 0.50 score was a grounding-regex gap
(missed the JSON-escaped rendering of a Wikipedia redirect slug), root-caused and fixed against 48
real replayed result files. Task 088 is now wired to the `ratio_argmax` composer (same opt-in
kill-switch pattern as 084/091). Task 079 was deliberately **not** wired — its numerator unit is
heterogeneous across items (TWh vs GWh) and the composer only supports one global unit, so forcing
it would silently mislabel a converted figure; a negative-case test pins exactly why, so nobody
re-attempts this the naive way later. Item 5 above (the Serper key-wiring gap) was found as a
byproduct of this item's Serper-liveness check.

**Explicitly not authorized by any existing plan**: the full barrage launch (blocked on item 1
above), any codebench live-matrix scale-up past a single smoke cell, and wiring codebench's
LLM-judge for soft-task grading (spends real money per grade, deliberately left for explicit
authorization).

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
