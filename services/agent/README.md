# IdeaDAG Agent

DAG-based reasoning agent (Graph-of-Thought) with vector DB integration, parallel execution, and bot-resistant web access.

## Flow

```
Mandate → Root node
  → Expansion (LLM decomposes into 2-5 subproblems)
  → Evaluation (LLM scores candidates 0-1)
  → Selection (best-first or sequential prune)
  → Leaf actions (search / visit / save / think)
  → Merge (LLM synthesizes child results upward)
  → Final synthesis → Deliverable
```

## Execution Modes

| Mode | Branching | Selection | Use Case |
|---|---|---|---|
| `graph` | Parallel, dynamic beam (2-5) | Best-first global | Production — 90.6% pass rate |
| `sequential` | Generate many, keep one | Prune siblings, depth-first | Baseline comparison — 46.9% pass rate |

## Running

```bash
docker compose up -d
docker compose run --profile test idea-test       # full suite: 16 tests × 2 models × 2 variants
docker compose run --profile test visit-test      # visit-focused: 019, 025, 033, 037
docker compose run --profile test general-test    # general: 002, 012
docker compose run --profile debug agent-debug    # GDB-style stepping debugger
```

## Configuration

All tunable parameters live in `app/idea_dag_settings.json`.

| Variable | Description | Default |
|---|---|---|
| `IDEA_TEST_IDS` | Comma-separated test IDs | (all by priority) |
| `IDEA_TEST_MODELS` | Models to test | `gpt-5.2,gpt-5-mini` |
| `IDEA_TEST_EXECUTION_VARIANTS` | `graph`, `sequential`, or both | `graph,sequential` |
| `IDEA_TEST_RUNS` | Repeats per test/model pair | `1` |
| `IDEA_TEST_CONCURRENCY` | Max parallel tests | `4` |
| `IDEA_TEST_MAX_STEPS` | Engine step cap | `75` |
| `IDEA_TEST_REPORT_VERBOSITY` | Detail level 0-3 | `2` |
| `MODEL_NAME` | Default LLM model | `gpt-5-mini` |

## Connectors

| Connector | Purpose |
|---|---|
| `ConnectorLLM` | OpenAI API for expansion, evaluation, merge, finalization |
| `ConnectorSearch` | Brave Search API for web queries |
| `ConnectorHttp` | aiohttp for page fetching with retry logic |
| `ConnectorBrowser` | undetected-chromedriver fallback for bot-protected sites (403/401) |
| `ConnectorChroma` | ChromaDB for vector storage and retrieval |

## Observability

Telemetry is handled by `ConnectorBase` (timing, I/O logging) so action classes stay clean. Every LLM call, HTTP request, search query, and browser fetch is recorded with duration, status, and payload metadata. The test runner emits per-test JSON with scores, costs, token counts, and graph structure metrics. The visualization pipeline generates dashboards from these results.

## Docs

| File | Content |
|---|---|
| [AGENT_ARCHITECTURE.md](app/AGENT_ARCHITECTURE.md) | Node types, execution flow, GoT mechanics, data flow, connectors |
| [DEPLOYMENT.md](app/DEPLOYMENT.md) | Docker services, env vars, prerequisites |
| [AGENT_DEBUG.md](app/AGENT_DEBUG.md) | GDB-style debugger commands and usage |
| [idea_tests/README.md](app/idea_tests/README.md) | Test structure and validation system |
