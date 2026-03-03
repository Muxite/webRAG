# Agent Debugger

GDB-style stepping debugger for the GoT agent. Walks the reasoning graph depth-first, pausing at expansions and merges.

## Usage

```bash
docker compose run --profile debug agent-debug
docker compose run --profile debug -e INTERACTIVE_MANDATE="Research quantum computing" agent-debug
docker compose run --profile debug -e INTERACTIVE_TEST_ID=025 agent-debug
```

## Commands

| Command | Description |
|---|---|
| `s` / `step` | Step into branch (expand, then DFS) |
| `n` / `next` | Auto-run branch to merge, then pause |
| `i` / `info` | Show current node; `i nodes` for tree, `i stats` for stats |
| `p` / `print` | Print current node action result |
| `l` / `list` | List children |
| `g` / `graph` | Full ASCII DAG |
| `q` / `quit` | End session |

## Environment

| Variable | Description | Default |
|---|---|---|
| `INTERACTIVE_MANDATE` | Task statement | (none) |
| `INTERACTIVE_TEST_ID` | Load from test module | (none) |
| `INTERACTIVE_MAX_STEPS` | Step cap | `100` |
| `MODEL_NAME` | LLM model | `gpt-5-mini` |
| `INTERACTIVE_LOG_LEVEL` | Log level | `WARNING` |
