# Deployment

## Prerequisites

Docker and Docker Compose. `OPENAI_API_KEY` and `SEARCH_API_KEY` must be set in `keys.env`.

## Docker Services

| Service | Profile | Description |
|---|---|---|
| `chroma` | default | ChromaDB vector database |
| `redis` | default | Cache and worker state |
| `rabbitmq` | default | Message queue |
| `agent` | default | Main agent service |
| `idea-test` | test | Full test suite — 16 tests, 2 models (gpt-5.2 + gpt-5-mini), graph + sequential |
| `visit-test` | test | Visit-focused subset — tests 019, 025, 033, 037 with gpt-5-mini |
| `general-test` | test | General subset — tests 002, 012 with gpt-5-mini |
| `agent-test` | test | Unit/integration pytest suite |
| `agent-debug` | debug | GDB-style stepping debugger |

## Commands

```bash
docker compose up -d                                    # start infrastructure
docker compose run --profile test idea-test             # full benchmark
docker compose run --profile test visit-test            # quick visit tests
docker compose run --profile test general-test          # basic fact tests
docker compose run --profile debug agent-debug          # interactive debugger
```

## Dockerfile

The agent image (`agent/.dockerfile`) is based on `python:3.10-slim` with Chrome dependencies for `undetected-chromedriver`:

```
libnss3, libgbm1, libasound2, libatk-bridge2.0-0, libdrm2
```

Pre-downloads SentenceTransformers and ChromaDB ONNX embedding models at build time.

## Test Environment Variables

| Variable | Description | Default |
|---|---|---|
| `IDEA_TEST_IDS` | Comma-separated test IDs | (all by priority) |
| `IDEA_TEST_TOP_N` | Number of top-priority tests | `0` (all) |
| `IDEA_TEST_MODELS` | Comma-separated models | `gpt-5.2,gpt-5-mini` |
| `IDEA_TEST_EXECUTION_VARIANTS` | `graph`, `sequential`, or both | `graph,sequential` |
| `IDEA_TEST_RUNS` | Repeats per model/test pair | `1` |
| `IDEA_TEST_CONCURRENCY` | Max parallel tests | `4` |
| `IDEA_TEST_MAX_STEPS` | Engine step cap per test | `75` |
| `IDEA_TEST_REPORT_VERBOSITY` | Report detail 0–3 | `2` |
| `IDEA_TEST_LOG_LEVEL` | Logging level | `INFO` |
| `IDEA_TEST_VALIDATION_MODEL` | Model used for LLM grading | `gpt-5-mini` |

## Visualization

After running tests, generate plots locally (requires matplotlib):

```bash
cd services/agent
python -m app.testing.idea_test_visualize --latest --core-only
```

Results are stored in `idea_test_results/` as timestamped JSON. Use `--list-runs` to see available runs.
