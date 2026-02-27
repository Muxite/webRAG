# IdeaDAG Agent Architecture

DAG-based reasoning system implementing Graph of Thoughts (GoT) for complex problem-solving.

## Core Principles

1. **Expansion-Merge Pair Pattern**: Every branch follows Expansion → [Layers] → Merge
2. **Fewer Steps = Better**: 1 big step >> 2 small steps
3. **Separation of Concerns**: Expansion only expands, merge only merges
4. **Automatic Vector DB**: ChromaDB queried automatically on every node
5. **Memory Type Split**: Internal thoughts vs observations for granular context
6. **Intent-Driven Actions**: Search/visit use intent to query vector DB
7. **Data Flow Enforcement**: Sequential execution when dependencies exist
8. **Goal Validation**: Merge nodes check if original goals were achieved

## Node Types

### Expansion Node
- Breaks problems into sub-problems
- Creates children with actions (search, visit, save)
- Vector DB provides context automatically
- Propagates original goals to children

### Leaf Node
- Executes actions: search, visit, save
- Marked with `IS_LEAF=True` or `action` field
- Results stored in vector DB automatically
- Large documents trigger chunking sub-problems

### Merge Node
- Synthesizes results from completed children
- Always progresses toward root
- Uses LLM to combine results into coherent summary
- Validates goal achievement against original goal

## Execution Flow

1. **Root**: Initial problem, vector DB queried for context
2. **Expansion**: Break into sub-problems (parallel or sequential)
3. **Intermediate Layers**: Can have recursive expansion-merge pairs
4. **Leaf Execution**: Actions gather evidence (search, visit, save)
5. **Document Chunking**: Large documents (>10K chars) split into chunk-based search sub-problems
6. **Chunk Search**: Search within document chunks for specific content
7. **Merge**: Synthesize results, validate goals, progress toward root
8. **Completion**: Final synthesis at root

## Document Chunking System

### When Chunking Occurs
- Visit action returns document > threshold (default: 10K chars)
- System automatically creates chunk sub-problems
- Each chunk becomes a search node with chunk content

### Chunk Configuration
- **Chunk Size**: 2000 chars (default)
- **Chunk Overlap**: 200 chars (default)
- **Threshold**: 10000 chars (default)

### Chunk-Based Search
- Search nodes with `CHUNK_CONTENT` search within chunk text
- Term matching and snippet extraction
- Can run in parallel for independent chunks
- Sequential when chunk dependencies exist

## Goal Tracking System

### Goal Propagation
- Original goals stored in `GOAL` and `ORIGINAL_GOAL` detail keys
- Goals propagated from parent to children during expansion
- Goals tracked through entire problem-solving process

### Goal Validation
- Merge nodes check if original goal appears in synthesized content
- `GOAL_ACHIEVED` flag stored in merge results
- Warnings logged when goals not fully achieved

## Sequential vs Parallel Execution

### Parallel Execution
- Independent chunks from same document
- Independent actions with no data dependencies
- Set via `execute_all_children: true` in expansion meta

### Sequential Execution
- When data dependencies exist (visit needs URL from search)
- When chunk dependencies exist (chunk depends on previous chunk)
- System automatically detects and enforces sequential order
- Default when dependencies detected

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

- **search**: Web search with query and optional intent (or chunk-based search)
- **visit**: Fetch URL content with optional intent
- **save**: Store in memory (auto-chunked)
- **merge**: Synthesize child results with goal validation

## Evaluation Weights And Penalties

Evaluation prefers certain behaviors and can be tuned via numeric settings in `idea_dag_settings.json`:

- **Unexecuted Action Penalties**:
  - `evaluation_no_action_result_base_score`: Base score assigned when a node has an action but no `action_result`
  - `evaluation_no_action_result_score_cap`: Maximum score cap for nodes with missing `action_result` even if the LLM suggests a higher score

- **Per-Action Weights**:
  - `evaluation_weight_search`, `evaluation_weight_visit`, `evaluation_weight_think`, `evaluation_weight_save`, `evaluation_weight_default`
  - Applied as multiplicative factors on raw LLM scores, then clamped into `[0,1]`
  - Allow the system to prefer or de-emphasize specific action types without changing prompts

## Final Synthesis and Compaction

Before merged results are sent to the final LLM synthesis step, they are compacted via `MergedResultsCompactor` to manage token limits while preserving essential information:

- **Content Truncation**: Action-aware preservation (1500 chars for visits, 400 for others) ensures infobox-style facts remain available
- **Link Preservation**: Visit actions preserve up to 20 links to satisfy link collection requirements
- **Large Field Removal**: `content_full`, `content_with_links`, `links_full` are removed to reduce token usage

## Failure Handling

- Detailed error tracking with stack traces
- Retry mechanisms with exponential backoff
- Blocked site detection and cooldown
- Root cause analysis in error details

## Logging and Debugging

- **LLM Inputs**: Full messages logged before LLM calls
- **LLM Outputs**: Full responses logged after LLM calls
- **Expansion**: Detailed logging of prompt sizes, token estimates, timeouts
- **Merge**: Goal validation results logged
- **Test Results**: Saved to `agent/idea_test_results/` directory (auto-created)
