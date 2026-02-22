# IdeaDAG Agent

DAG-based reasoning agent implementing Graph of Thoughts (GoT) architecture for complex problem-solving with automatic vector database integration, parallel execution, and comprehensive testing infrastructure.

## Overview

The IdeaDAG agent breaks complex problems into sub-problems using a directed acyclic graph (DAG) structure. Each branch follows an **Expansion-Merge Pair** pattern: expansion nodes break problems into sub-problems, leaf nodes execute evidence-gathering actions (search, visit, save), and merge nodes synthesize results toward completion.

### Key Features

- **Graph of Thoughts (GoT) Architecture**: Hierarchical problem decomposition with automatic merge synthesis
- **Automatic Vector Database Integration**: ChromaDB queried automatically on every node for context
- **Memory Type Classification**: Separate internal thoughts and observations for granular context control
- **Parallel Execution**: Multi-model testing with asyncio-based parallelization
- **Comprehensive Testing Infrastructure**: 24 priority-ordered tests with validation and visualization
- **Real-time Observability**: Telemetry tracking for LLM calls, searches, visits, and performance metrics

## Architecture

### Core Components

1. **IdeaDAG Engine** (`idea_engine.py`): Main execution engine managing DAG traversal and node execution
2. **Node Types**: Expansion (decompose), Leaf (execute actions), Merge (synthesize)
3. **Policies**: Expansion, evaluation, and merge policies using LLM for decision-making
4. **Memory Manager** (`idea_memory.py`): ChromaDB integration with memory type classification
5. **Agent IO** (`agent_io.py`): Unified interface for LLM, search, HTTP, and ChromaDB operations

### Execution Flow

```
Root Node (Problem)
  ↓
Expansion Node (Break into sub-problems)
  ↓
[Intermediate Layers - can have recursive expansion-merge pairs]
  ↓
Leaf Nodes (Execute actions: search, visit, save)
  ↓
Merge Node (Synthesize results)
  ↓
[Progress toward root for final completion]
```

### Memory System

- **Internal Thoughts**: Agent reasoning, planning, synthesis (from `think`, `save`, merge summaries)
- **Observations**: External data from searches and visits
- **Automatic Retrieval**: Top N internal thoughts + Top M observations on every node
- **Semantic Search**: Vector DB queried with intent for relevant context

## Testing Infrastructure

### Test System

Located in `app/testing/`:
- **Validation System**: Class-based validation with function and LLM checks
- **Test Runner**: Parallel execution with multi-model support
- **Test Modules**: Python-based test definitions with validation functions
- **Visualization**: Matplotlib-based analysis with timestamp tracking

### Test Priority System

Tests ordered by priority (1-24). Environment variable `IDEA_TEST_PRIORITY`:
- `0`: Run all tests
- `N`: Run top N priority tests

### Supported Models

- `gpt-5-mini` (default, also used for validation)
- `gpt-5-nano`
- `gpt-4.1-nano`
- `gpt-4o`

### Environment Variables

- `IDEA_TEST_MODELS`: Comma-separated models to test
- `IDEA_TEST_PRIORITY`: Number of priority tests (0 = all)
- `IDEA_TEST_MAX_PARALLEL`: Max parallel executions (default: 4)
- `IDEA_TEST_LOG_LEVEL`: Logging level (default: INFO)

### Running Tests

```bash
# Run all tests with default model
python -m app.idea_test_runner

# Run top 10 priority tests with multiple models
IDEA_TEST_MODELS=gpt-5-mini,gpt-5-nano python -m app.idea_test_runner
IDEA_TEST_PRIORITY=10 python -m app.idea_test_runner

# Visualize results
python -m app.idea_test_visualize --run-id 20260221_205459
```

## Deployment

### Docker Compose

```bash
# Run agent tests
docker compose up agent-test

# Run idea tests
docker compose up idea-test
```

### Requirements

- Python 3.10+
- ChromaDB (local or remote)
- Redis (for caching)
- RabbitMQ (for messaging)
- OpenAI API key (`OPENAI_API_KEY`)
- Search API key (`SEARCH_API_KEY`)

## Documentation

- **Internal Agent System**: See `app/AGENT_ARCHITECTURE.md` for detailed agent implementation
- **Data Flow**: See `app/DATA_FLOW.md` for memory and data flow details
- **Testing**: See `app/IDEA_TEST_SYSTEM.md` for test system documentation
- **Visualization**: See `app/IDEA_TEST_VISUALIZATION.md` for visualization usage

## Code Organization

- **Core Agent**: `idea_engine.py`, `idea_dag.py`, `idea_policies/`, `idea_memory.py`
- **Testing Infrastructure**: `testing/` (validation, execution, runner, config, utils)
- **Test Definitions**: `idea_tests/` (24 priority-ordered test modules)
- **Visualization**: `idea_test_visualize.py`
- **Connectors**: `connector_*.py` (LLM, search, HTTP, ChromaDB)
- **Interface**: `agent.py`, `interface_agent.py`, `main.py`

## Statistics & Observability

The system collects comprehensive metrics:
- LLM call counts, tokens, durations
- Search/visit counts and data amounts (KB)
- ChromaDB store/retrieve operations
- Validation scores and pass rates
- Timestamp tracking for performance over time

All metrics visualized via `idea_test_visualize.py` with matplotlib diagrams showing performance trends, model comparisons, and validation breakdowns.
