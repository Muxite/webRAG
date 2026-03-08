# Idea Tests

Priority-ordered test suite for the IdeaDAG agent. 39 tests across categories from basic fact retrieval to massive branching verification.

## Structure

Each `test_XXX_name.py` exports:

| Function | Returns |
|----------|---------|
| `get_test_metadata()` | `{test_id, name, difficulty, category, priority}` |
| `get_task_statement()` | Mandate string for the agent. |
| `get_required_deliverables()` | List of expected output fields. |
| `get_success_criteria()` | Human-readable success conditions. |
| `get_validation_functions()` | List of `{name, fn}` for programmatic checks. |
| `get_llm_validation_function()` | Optional LLM-based grading function. |

## Validation

- **Function checks**: Pattern matching, keyword presence, structural rules, type validation
- **LLM checks**: Semantic grading by `gpt-5-mini` (always uses gpt-5-mini regardless of execution model)
- **Score**: 0.0–1.0 per check; overall = mean of all validation scores
- **Pass threshold**: 0.75 (test passes if overall score ≥ 0.75)
- **Results**: Written to `agent/idea_test_results/` as timestamped JSON files

## Test Categories

| Category | Tests | Purpose |
|---|---|---|
| Baseline | 001, 002, 003 | Basic fact retrieval, conflicting info, multi-query |
| Research | 009, 012, 026 | Deep synthesis, multi-step, multi-source comparison |
| Visit | 019, 020, 025 | Explicit URL visits, GitHub analysis, link chain traversal |
| Branching | 033–037 | Progressive difficulty (5/10 through 9/10) favoring parallel execution |
| Massive Branch | 038 | 8-source fact matrix extraction |
| Verification | 039 | Multi-branch claim fact-checking with confidence scoring |

## Running

### Basic Usage

```bash
# Run specific tests
IDEA_TEST_IDS=019,025 docker compose run --profile test visit-test

# Run full test suite
docker compose run --profile test idea-test

# Benchmark mode (top 8 tests, 3 models, 3 runs each, concurrency 3)
IDEA_TEST_MODE=benchmark docker compose run --profile test idea-test
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IDEA_TEST_IDS` | Comma-separated test IDs (e.g., "019,025,033") | All tests |
| `IDEA_TEST_MODE` | "default" or "benchmark" | "default" |
| `IDEA_TEST_TOP_N` | Number of top-priority tests to run (0 = all) | 0 |
| `IDEA_TEST_RUNS` | Repeats per test/model pair | 1 |
| `IDEA_TEST_CONCURRENCY` | Max parallel task executions | 1 |
| `IDEA_TEST_MODELS` | Comma-separated models (e.g., "gpt-5.2,gpt-5-mini") | `MODEL_NAME` or "gpt-5-mini" |
| `IDEA_TEST_EXECUTION_VARIANTS` | "graph", "sequential", or comma-separated | "graph" |
| `IDEA_TEST_LOG_LEVEL` | Logging level | INFO |

### Test Execution Modes

- **Graph mode**: Parallel branching with best-first selection (default, higher pass rate)
- **Sequential mode**: Generate-many, keep-one, depth-first (baseline comparison)

Both modes can be run in the same test run by setting `IDEA_TEST_EXECUTION_VARIANTS=graph,sequential`.

## Adding Tests

1. Create `test_XXX_name.py` following the export pattern above.
2. Set `priority` in metadata (lower = runs first).
3. Add programmatic validators and/or LLM grading.
4. The runner discovers tests automatically via `discover_test_modules()`.
