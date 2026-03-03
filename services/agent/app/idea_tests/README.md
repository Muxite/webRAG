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

- **Function checks**: pattern matching, keyword presence, structural rules.
- **LLM checks**: semantic grading by `gpt-5-mini`.
- **Score**: 0.0–1.0 per check; overall = mean. Pass threshold = 0.75.

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

```bash
IDEA_TEST_IDS=019,025 docker compose run --profile test visit-test
docker compose run --profile test idea-test    # full suite
```

## Adding Tests

1. Create `test_XXX_name.py` following the export pattern above.
2. Set `priority` in metadata (lower = runs first).
3. Add programmatic validators and/or LLM grading.
4. The runner discovers tests automatically via `discover_test_modules()`.
