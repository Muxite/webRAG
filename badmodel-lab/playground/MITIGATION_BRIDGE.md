# Why only `a0`-`a4` apply to interactive chat

`badmodel-lab/profiles/` has 13 mitigation profiles across 3 families. The playground's
`BADMODEL_PROFILE` override only works for 5 of them — this document explains why, so
picking the wrong one doesn't look like a broken feature.

## The two families, and why one is structurally inapplicable here

**`m0`-`m4` and `fs0`-`fs2`** (8 profiles) target the *compiled-scaffold* execution path
(`graph_compiled`, driven by `agent.app.testing.execution_compiled`). That path executes
a **pre-authored DAG plan for a known benchmark task id** — the plan itself is data
committed under `agent/compiled_plans/`, keyed by task id. There is no such
thing as a compiled plan for a friend's freeform typed mandate; nothing generates one at
chat time. Setting `BADMODEL_PROFILE` to one of these produces **no behavior change at
all** — not a subtle degradation, a literal no-op — because the env vars these profiles
set (`IDEA_TEST_COMPILED_*`) are only ever read inside the compiled-leaf executor, which
the interactive path never runs.

**`a0`-`a4`** (5 profiles) target the *native adaptive engine* (`IdeaDagEngine` —
`agent/app/idea_engine.py`), where the model itself proposes and expands
Graph-of-Thoughts leaves each turn instead of executing a fixed plan. This is exactly the
engine `Agent.run()` uses whenever `AGENT_USE_IDEA_DAG=1` (always set in this stack's
`docker-compose.yml`) — the same engine `basic_cli.py` drives, compiled-scaffold or not.
These are the only profiles the playground's `BADMODEL_PROFILE` override can meaningfully
change.

## What `a0`-`a4` actually toggle

Each sets one or more `IDEA_TEST_GOT_*` env vars, translated onto the engine's settings
dict by `agent.app.idea_test_runner._apply_got_experiment_overrides` (the same function
`badmodel-lab/run_adaptive_cell.sh` uses for the benchmark harness — the playground
reuses it unmodified rather than reimplementing the translation).

- **`a0_native_baseline`** — every mechanism at its JSON default (off). The control.
- **`a1_native_reexpand`** — the follow-up-detector re-expansion only.
- **`a2_native_good_adaptive`** — reexpand + a decorrelated per-step confidence judge +
  confidence-driven re-expansion.
- **`a3_native_expect_contract`** — a leaf declares a measurable-output-plus-source
  contract, targeting citation discipline specifically.
- **`a4_native_plan_library`** — automatic pre-expansion retrieval from a library of
  pre-authored composition-strategy templates (requires the library synced into the
  benchmark Chroma; not wired into this playground's embedded-Chroma setup, so this
  profile is unlikely to do anything useful here yet — try it, but don't be surprised if
  it behaves identically to `a0`).

No tier ships with a confirmed "best" profile — that research is separate, ongoing work
(see `tier_profiles.yaml`'s own comment for how to update the default once it lands).
Override per-invocation to try one:

```
BADMODEL_PROFILE=a3_native_expect_contract docker compose up badmodel-8gb --build -d
```
