# Idea Tests

Priority-ordered test suite for IdeaDAG agent evaluation.

## Test Structure

Each test is a Python module (`test_XXX_name.py`) with:
- `get_test_metadata()`: Test ID, name, difficulty, category
- `get_task_statement()`: Task for agent
- `get_required_deliverables()`: Expected outputs
- `get_success_criteria()`: Success conditions
- `get_validation_functions()`: List of validation functions
- `get_llm_validation_function()`: Optional LLM-based validation

## Priority Order

Tests ordered 1-24 by priority. Lower number = higher priority.

## Validation

- **Function Checks**: Grep-based validation (search for keywords, patterns)
- **LLM Checks**: LLM-based validation for complex criteria
- **Scoring**: 0.0-1.0 score per check, overall score is average
- **Pass Threshold**: 0.75 overall score required to pass

## Running Tests

See main README for environment variables and usage.
