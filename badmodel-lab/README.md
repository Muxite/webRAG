# badmodel-lab

An addendum to the webRAG/Euglena cost-vs-accuracy benchmark: run **super-bad
local LLMs** (0.5B–3B, models that can't reliably emit JSON) through the *same*
scoring harness, apply **mitigations** to make them agentic, and measure how far
each mitigation lifts them. Opus drives the experiment; see `PLAYBOOK.md`.

Nothing here forks the agent — the runner is already provider-abstracted behind an
OpenAI-compatible `base_url`. The lab adds only: env wiring (`run_cell.sh`),
mitigation `profiles/`, an `analyze.py`, JSON parse-failure telemetry
(`agent/app/testing/json_telemetry.py`, env-gated, no-op when off), and a
`micro` tier of tests weak models can actually get a signal on.

## Quickstart

```bash
# 1. dedicated ollama on the GPU (isolated from the yappers project's :11434)
docker compose -f badmodel-lab/docker-compose.yml up -d
./badmodel-lab/pull_roster.sh                       # pull 0.5B–3B subjects

# 2. run one cell = model x mitigation x tier
./badmodel-lab/run_cell.sh qwen2.5:0.5b m0_baseline micro 1   # baseline (JSON path) + telemetry
./badmodel-lab/run_cell.sh qwen2.5:0.5b m1_thin     micro 1   # thin leaf (no JSON) — the big lever

# 3. see the leaderboard, JSON-capability mix, floor/ceiling, and the CSV
./.venv/bin/python badmodel-lab/analyze.py
```

## How it plugs in (confirmed against the code)

| Need | Mechanism |
|---|---|
| Point the agent at a bad model | `LLM_PROVIDER=openai_compatible`, `MODEL_API_URL=http://localhost:11435/v1`, `MODEL_NAME=<tag>` |
| Don't let the JSON preflight drop it | `IDEA_TEST_PREFLIGHT_JSON=0` (`idea_test_runner.py:161`) — it's the subject, not a reject |
| Route a can't-JSON model off the JSON path | `IDEA_TEST_COMPILED_LEAF_MODE=thin` (`execution_compiled.py:602`) — auto-mode sends unpriced models to the react/JSON leaf, so force thin |
| Score it | existing `scripts/gate_report.py --run-id <id>` |
| Cap spend | `IDEA_TEST_USD_CEILING` (default $5 here) |
| See *why* it failed | `IDEA_TEST_JSON_TELEMETRY=1` → parse-failure classes read by `analyze.py` |

## Layout

```
docker-compose.yml   dedicated badmodel-ollama (GPU, host :11435, on euglena_enet)
roster.yaml          subject + anchor models
tiers.yaml           task tiers (sanity/micro/reachable/hard)
pull_roster.sh       pull subjects into badmodel-ollama
run_cell.sh          run ONE (model x profile x tier) cell -> scored result
profiles/*.env       the mitigation ladder (m0..m4)
analyze.py           leaderboard + JSON-capability + floor/ceiling + results/cells_long.csv
PLAYBOOK.md          the Opus experiment protocol
CHART_SPEC.md        LinkedIn 3x2 + 1x1 chart spec (from the design agent)
results/             cells.jsonl (attribution) + cells_long.csv (chart input)
```

## Safety notes

- Uses a **dedicated** ollama container; never pulls into or touches `yappers-ollama`.
- The telemetry module is a no-op unless `IDEA_TEST_JSON_TELEMETRY` is set — the
  main test suite and normal benchmark runs are unaffected.
- Local subjects cost nothing; only OpenRouter *anchor* runs spend, under the ceiling.
