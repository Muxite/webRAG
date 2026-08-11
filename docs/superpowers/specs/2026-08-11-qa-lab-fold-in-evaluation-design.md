# QA-lab fold-in: evaluated, no fold-in needed

## Context

Cycle 2 (2026-08-09) folded `badmodel-lab/codebench/` into a top-level `codebench/` directory,
because it was general Docker-sandboxed-coding *infrastructure* that had drifted into a
lab-scoped directory by accident — the task definitions it exercises already lived in main. That
spec deliberately scoped the QA lab (`badmodel-lab/analyze.py`, `results/cells.jsonl`,
`roster.yaml`/`tiers.yaml`/`profiles/`) OUT, noting `AGENT_CONTINUUM.md` names specific
`cells.jsonl` fields as ones that "may never bridge" into main's schema, and that "this tension
deserves its own cycle rather than being rushed through here." `docs/handoffs/HANDOFF.md` carried
this forward as open item 2 ("The QA-lab fold-in — deliberately deferred out of cycle 2's scope").

This spec is that deferred cycle. The question it answers: does anything here actually need to
fold in, the way codebench did?

## Finding: the codebench precedent does not generalize

Codebench and the QA lab are opposite cases:

- **Codebench** was infrastructure (Dockerfiles, grading pipeline, `run_matrix.sh`) that happened
  to live under a directory named for the lab, with no lab-specific content of its own.
- **`roster.yaml`, `tiers.yaml`, `profiles/`** are the opposite: genuinely lab-specific
  *experiment configuration*. `roster.yaml` names the specific weak/tiny local models
  (`tinyllama`, `qwen2.5:0.5b`, ... `qwen2.5:14b`) this lab exists to test, plus the anchor
  references used to set a ceiling. `tiers.yaml` groups task IDs into the lab's own
  difficulty bands (`sanity`/`micro`/`reachable`/`format`/`hard`), explicitly designed around
  where a sub-1B model gets a gradient at all — a grouping main's own benchmark suite has no
  equivalent need for. `profiles/` holds 17 named `.env` files, each one a specific combination
  of mitigation levers (`a5_native_vote_k_tiered.env`, `fs2_thin_assemble.env`, ...) for the
  lab's own A/B methodology.
- This is exactly the "big, specific library" half of the project's own validated capability-
  continuum philosophy (`feedback_capability_continuum_philosophy.md`): badmodel-lab is supposed
  to carry more, more-specific configuration than main deliberately does. Folding this content
  into a shared top-level location would erase the intentional library-size difference, not fix
  a misplaced-infrastructure problem.
- `analyze.py`/`results/cells.jsonl` are the reader and result format for that lab-specific
  experiment design — they inherit the same reasoning, not a separate one.

## Finding: the actual cross-cutting need is already solved, structurally

The reason a fold-in felt necessary was reporting: being able to see badmodel-lab's results
alongside main's. That's already solved without moving any files — cycle 1 built
`scripts/unified_bench_report.py` specifically because "two threads assumed to need independent
tooling... turned out to share a reporting layer once actually looked at." It reads
`idea_test_results/*.json` and `badmodel-lab/results/cells.jsonl` directly, from their current
locations. No shared directory is required for this to work, and it already does.

## Finding: the remaining real work is already correctly scoped, in `AGENT_CONTINUUM.md`

`AGENT_CONTINUUM.md`'s own roadmap (item 4, "Unify benchmark reporting — in progress, not
closed") already names the right mechanism for the part of this that's genuinely unfinished:
narrow, field-by-field bridging inside `analyze.py` (already done: `model`/`score`/`usd`/`visits`/
`grounding_pass`/`latency_s`; deliberately left unbridged: `completion_tokens`, a "false cousin"
of main's `total_tokens`). It also already names which fields are expected to stay
badmodel-lab-specific by design, not oversight: `place`/`profile`/`leaf_mode`/`tier`/`test_id`.
This is a field-mapping exercise, not a directory move, and it's already under way — there is
nothing this spec needs to add to that roadmap item.

## Decision

No fold-in. `badmodel-lab/analyze.py`, `results/cells.jsonl`, `roster.yaml`, `tiers.yaml`,
`profiles/` all stay exactly where they are. `docs/handoffs/HANDOFF.md` item 2 is closed out
referencing this spec, so a future session doesn't re-open the same question without new
evidence — the trigger for re-opening it would be a genuine change in scope (e.g. main's engine
growing its own per-task-tier experiment methodology that could actually share `roster.yaml`/
`tiers.yaml`'s shape), not a passage of time.

## Non-goals

This spec does not evaluate `localagent/` retirement or the `playground/` SearXNG connector —
both were already investigated separately this session and found to be "don't touch" for
unrelated reasons (see `project_badmodel_lab_cleanup_candidates_rejected` memory). It also does
not propose any change to `AGENT_CONTINUUM.md`'s in-progress field-bridging work in `analyze.py`
— that item is correctly scoped already and is out of scope here.
