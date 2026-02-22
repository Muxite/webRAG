# IdeaDAG Agent Architecture

DAG-based reasoning system implementing Graph of Thoughts (GoT) for complex problem-solving.

## Core Principles

1. **Expansion-Merge Pair Pattern**: Every branch follows Expansion → [Layers] → Merge
2. **Fewer Steps = Better**: 1 big step >> 2 small steps
3. **Separation of Concerns**: Expansion only expands, merge only merges
4. **Automatic Vector DB**: ChromaDB queried automatically on every node
5. **Memory Type Split**: Internal thoughts vs observations for granular context
6. **Intent-Driven Actions**: Search/visit use intent to query vector DB

## Node Types

### Expansion Node
- Breaks problems into sub-problems
- Creates children with actions (search, visit, save)
- Vector DB provides context automatically

### Leaf Node
- Executes actions: search, visit, save
- Marked with `IS_LEAF=True` or `action` field
- Results stored in vector DB automatically

### Merge Node
- Synthesizes results from completed children
- Always progresses toward root
- Uses LLM to combine results into coherent summary

## Execution Flow

1. **Root**: Initial problem, vector DB queried for context
2. **Expansion**: Break into sub-problems (parallel or sequential)
3. **Intermediate Layers**: Can have recursive expansion-merge pairs
4. **Leaf Execution**: Actions gather evidence (search, visit, save)
5. **Merge**: Synthesize results, progress toward root
6. **Completion**: Final synthesis at root

## Memory System

### Memory Types

- **Internal Thoughts**: Agent reasoning (`think`, `save`, merge summaries)
- **Observations**: External data (search results, visited pages)

### Retrieval Strategy

- Top N internal thoughts + Top M observations per node
- Semantic search with intent for relevant context
- Automatic chunking and storage after actions

## Vector Database Integration

- **Automatic Querying**: Every node queries ChromaDB for context
- **Intent Field**: Search/visit actions can specify intent for targeted retrieval
- **Memory Classification**: Metadata includes `memory_type` (internal_thought/observation)
- **Namespace Isolation**: Test runs use unique namespaces

## Actions

- **search**: Web search with query and optional intent
- **visit**: Fetch URL content with optional intent
- **save**: Store in memory (auto-chunked)

## Failure Handling

- Detailed error tracking with stack traces
- Retry mechanisms with exponential backoff
- Blocked site detection and cooldown
- Root cause analysis in error details
