# webRAG dev agents

Project-scoped Claude Code subagents that speed up work on the webRAG / Euglena
Graph-of-Thoughts service and its cost-recovery benchmark. Each `*.md` here is one agent:
YAML frontmatter (`name`, `description`, optional `tools` allowlist, optional `model`) registers
it; the body is its system prompt. Invoke by name ("use the `benchmark` agent to…") or let the
main thread delegate (`subagent_type: <name>`). Each spawn starts cold, so the project recipe
(paths, the keys.env/chroma/concurrency rules, the result-JSON schema, commit policy) is baked
into every body.

## Two phases

**Phase 1 — build the system and prove the thesis.** The Graph-of-Thoughts engine
(`services/agent/app/idea_engine.py`, `idea_policies/*`) plus the cost-recovery benchmark: a
discriminating task suite (`idea_tests/test_*.py`), execution variants (graph, sequential_react,
parametric, naive_rag, **graph_compiled**), fixtures, cost instrumentation, and the analysis
scripts. The thesis: a graph/compiled scaffold turns a cheap model's larger token budget into
premium-reference quality at a fraction of the dollar cost. Status: proven live — see
`services/agent/app/COST_BENCHMARK_HANDOFF.md` (Rounds 1–3).

**Phase 2 — use the system to make cheap models stronger (current focus).** Treat the benchmark
as a feedback loop and push *structured work strategy* down to the leaf so weak/cheap models
recover top-tier accuracy: atomic DAG decomposition, the **thin-leaf** micro-prompt executor
(harness owns control flow, the LLM only perceives), and **price-aware anchored voting** (a
dirt-cheap model spends its cheapness on redundant nodes + majority pruning + repeat cycles;
premium models trust one call). Result so far: cheap-model average lifted toward the ceiling at a
fraction of cost. Detail: `COST_BENCHMARK_HANDOFF.md` Round 4 and the
`project_compiled_scaffold_thesis` memory. The agents below exist mainly to run Phase 2.

## The roster (5)

| agent | role | model | spends $? |
|---|---|---|---|
| `task-author` | author + live-verify + harden a new benchmark task & its validators | opus | no |
| `benchmark` | run live matrices (gated, **singleton**) **and** analyze results / diagnose failures | sonnet | live=yes, analysis=no |
| `strategy-tuner` | the Phase-2 loop: one prompt/executor change → gated A/B → keep or revert | opus | yes (via `benchmark`) |
| `engine-dev` | core GoT engine / policy / typed-config work (Phase-1 product) | opus | no |
| `reviewer` | pre-commit gate (byte-compile + offline tests) + git hygiene | sonnet | no |

(Consolidated from an earlier 8: ground-truth verification and validator-hardening folded into
`task-author`; the live runner and results-analyst folded into `benchmark`.)

## Pipeline

```
task-author ──► reviewer(offline gate) ──► benchmark (gated, $)
   ▲                                            │ run + analyze
   │                                            ▼
strategy-tuner ◄──────────── diagnosis / per-cell deltas
engine-dev ──► reviewer ──► (commit)
```

- **Free lane (parallel):** `task-author`, `engine-dev`, and `benchmark` in analysis mode all
  run independently — they only touch code/tests or read existing results.
- **Paid lane (serial):** `benchmark` live runs are a **singleton** (shared connectors,
  `IDEA_TEST_CONCURRENCY=1`) and require explicit spend authorization + a budget. The
  `strategy-tuner ↔ benchmark` A/B loop is the heart of Phase 2; always gate it
  (smoke cell → one model → full matrix) and pin one `IDEA_TEST_RUN_ID` per experiment.
- `reviewer` is the ship gate for every change and enforces the commit policy (branch off
  `master`, exclude the `ideaengine/` + root `shared/` deletions and gitignored artifacts, and
  **no `Co-Authored-By: Claude` trailer**).

## Notes
- `.claude/agents/` is tracked (the rest of `.claude/` is gitignored); edit these files to tune an
  agent's behavior.
- Models are split price-aware on purpose: Opus where reasoning is the value, Sonnet for the
  procedural/cheap agents — the same logic Phase 2 applies to the leaves, applied to the dev team.
