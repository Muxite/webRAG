# Fold `badmodel-lab/codebench/` into a top-level `codebench/`

## Context

`badmodel-lab/` was deliberately kept separate from main as a bigger, more specific mitigation
library alongside main's smaller general one (`feedback_capability_continuum_philosophy.md`). That
call is being partially reversed: `badmodel-lab/codebench/` — the Docker-sandboxed coding-benchmark
runner (Dockerfiles, `run_matrix.sh`, grading pipeline, `analyze_code.py`) — is not itself
mitigation-library content. It's benchmark infrastructure that happens to live under a directory
named for the lab. The task *definitions* it exercises already live in main
(`agent/app/idea_code_tests/test_c01..c52_*.py`, validators in `agent/tests/idea_code_test_c*_test.py`)
— only the execution/grading/analysis machinery is still lab-scoped.

This is scoped narrowly to `codebench/`. The QA lab (`analyze.py`, `results/cells.jsonl`,
`roster.yaml`/`tiers.yaml`/`profiles/`) stays in `badmodel-lab/` for now — `AGENT_CONTINUUM.md`
names specific `cells.jsonl` fields as ones that "may never bridge" into main's schema, and that
tension deserves its own cycle rather than being rushed through here. `localagent/` (already flagged
for retirement once the graph engine matches it) and the stale `playground/` SearXNG connector twin
are smaller, separately-actionable items, also out of scope here.

Repo context has shifted since `codebench` was last touched (2026-08-08/09): `services/agent/` was
restructured to a top-level `agent/` directory in the same session this spec is being written in
(commit `8d45df3a`). Every path this spec references was re-verified against that new layout, not
assumed from the pre-restructure inventory.

## Scope

**Moves** (new location: top-level `codebench/`, sibling to `agent/`, `services/`, `badmodel-lab/`,
`docs/`, `scripts/`):
- `agents/{base,badmodel,aider}/` (Dockerfiles + `run_task.sh` scripts)
- `grader/` (the isolated grading container)
- `ollama_proxy/` (the allowlist proxy Dockerfile)
- `materialize_task.py`, `run_agent_sandbox.sh`, `extract_submission.py`, `run_grade.sh`,
  `score_and_record.py`, `analyze_code.py` (+ its test), `run_matrix.sh`, `setup_network.sh`,
  `build_images.sh`
- The gitignored, local-only `tasks/` and `results/` directories, moved as plain files (not
  `git mv` — they were never tracked) so this session's live-run history isn't orphaned.

**Stays in `badmodel-lab/`** (explicit, temporary, documented residual dependency):
- `roster.yaml`, `tiers.yaml` — `codebench`'s scripts keep reading these from
  `badmodel-lab/roster.yaml`/`tiers.yaml` via a repo-root-relative path. Resolved cleanly when the
  QA-lab cycle (separate, later) decides their canonical home — not duplicated here, per the
  root-`shared/`-vs-`services/shared/` lesson this repo already learned once
  (`project_duplicate_shared_modules.md`).

**Unchanged:**
- Task definitions and validators (`agent/app/idea_code_tests/`, `agent/tests/idea_code_test_c*_test.py`)
  — never lived in `badmodel-lab/`, not touched by this move.
- Docker image names/tags (`codebench-base`, `codebench-badmodel`, `codebench-aider`,
  `codebench-grader`, `codebench-ollama-proxy`) — unchanged, this is a path move, not a rebuild-from-
  scratch.

## Migration mechanism

A single `git mv badmodel-lab/codebench codebench` (preserves history — this is an internal move
with no external consumers needing a transition period, so no symlink/shim layer). Then a reference-
update sweep, each site re-verified live at implementation time rather than trusted from this spec's
prose (paths have already drifted once this session):

1. `.gitignore`: `badmodel-lab/codebench/{tasks,results}/` → `codebench/{tasks,results}/`.
2. `-f <path>/Dockerfile` and other repo-root-relative paths inside `build_images.sh`,
   `run_matrix.sh`, `setup_network.sh` themselves.
3. `agents/badmodel/Dockerfile`'s `COPY agent /app/agent` line — already correct post-restructure
   (verified), but re-check nothing else in that Dockerfile still says `badmodel-lab/codebench/...`
   for a path now one level shallower.
4. The `idea_code_test_c*_test.py` files that invoke `materialize_task.py` — **first check whether
   they go through one shared helper/fixture** (likely, given this repo's conventions elsewhere)
   before assuming N separate edits. If there is no shared helper, that's itself worth fixing as
   part of this move rather than editing N call sites twice in two different cycles.
5. `scripts/unified_bench_report.py`'s `DEFAULT_CODE_RUNS` path constant
   (`badmodel-lab/codebench/results/runs.jsonl` → `codebench/results/runs.jsonl`).
6. `.claude/agents/codebench-task-author.md`'s operational commands (`run_matrix.sh` invocation,
   results path).
7. Living architecture docs that cite the old path (`agent/app/AGENT_CONTINUUM.md`,
   `agent/app/TECHNIQUE_INVENTORY.md`, `agent/app/SYSTEM_STATUS.md`, or wherever they actually live
   post-restructure — re-grep, don't assume). The dated spec at
   `docs/superpowers/specs/2026-08-08-codebench-tooling-and-benchmark-unification-design.md` stays
   as-is — historical/point-in-time record by this repo's spec convention, not a living doc.
8. `badmodel-lab/README.md` / `badmodel-lab/HANDOFF.md` get a short note that `codebench/` moved out,
   so a future reader of `badmodel-lab/` isn't left looking for a directory that's no longer there.
9. Any `roster.yaml`/`tiers.yaml` reads inside the moved scripts get their relative path updated to
   point at `../badmodel-lab/roster.yaml` (or an equivalent repo-root-anchored resolution, matching
   how other scripts in this repo already compute `_ROOT`) — not copied, per the Scope section above.

## Testing / verification

- Full offline suite (not codebench-scoped — over 50 files change): `PYTHONPATH=.:services:agent
  ./.venv/bin/python -m pytest -q agent/tests`.
- One cheap local-model live smoke cell post-move (a single task, one local model, badmodel
  agent-kind) to confirm the Docker build context and `run_matrix.sh` actually work end-to-end at
  the new path — a pure path-rename is exactly the kind of change that looks fine in a diff and
  breaks at runtime (Docker build context, `-f` flags, and the `roster.yaml` relative-path read are
  all real ways this specific move could silently break). Needs explicit go-ahead at execution time
  per this repo's live-run convention, even though it's near-$0.
- `grep -rn "badmodel-lab/codebench"` across the repo after the move should return nothing except
  the historical dated spec doc (item 7 above) and, if genuinely still needed, a comment explaining
  why a given reference is intentionally historical.

## Out of scope (explicitly, not deferred silently)

- The QA lab fold-in (`analyze.py`, `cells.jsonl`, `roster.yaml`/`tiers.yaml`'s canonical home) —
  separate future cycle.
- `localagent/` retirement and the stale `playground/` SearXNG connector twin — separate,
  smaller, already-identified items.
- Wiring `code_rubric.py`'s LLM judge into `score_and_record.py` — unrelated pre-existing deferred
  decision, not touched by this move.
