# Data Flow

## Search Flow

1. Execute `search` with query + optional intent
2. Intent queries ChromaDB for context
3. ConnectorSearch returns title/url/description
4. Stored in ChromaDB as `memory_type="observation"`
5. Future nodes query semantically

## Visit Flow

1. Execute `visit` with URL + optional intent
2. Intent queries ChromaDB for context
3. ConnectorHttp fetches full page (no limit)
4. Full content cleaned; all links extracted
5. LLM gets truncated: 6000 chars, 20 links
6. **ChromaDB stores full content + all links** (chunked if large)
7. Future nodes query semantically (full content available)

## Final Synthesis Compaction

When merged results are sent to the final LLM synthesis step, they are compacted to manage token limits:

- **Content Preservation**:
  - VISIT actions: Up to 1500 characters preserved (ensures infobox-style facts like release dates, creators, versions are not truncated)
  - Other actions: Up to 400 characters preserved

- **Link Preservation**:
  - VISIT actions: Up to 20 links preserved in `links` array (enables link collection tasks requiring 10+ links)
  - All actions: Up to 5 links in `links_sample` for debugging/inspection
  - `links_count` always included for reference

This action-aware compaction ensures that tasks requiring specific facts or link collections have sufficient data available in the final synthesis step.

## Document Chunking Flow

1. Visit action completes with large document (>10K chars by default)
2. System detects document size exceeds threshold
3. Document split into chunks (2000 chars, 200 overlap by default)
4. Each chunk becomes a search sub-problem
5. Chunk-based search nodes search within their assigned chunk
6. Results merged back to parent visit node
7. Goal validation checks if original goal was achieved

## Chunk-Based Search

- When `CHUNK_CONTENT` is present in node details, search operates on chunk text
- Term matching within chunk content
- Returns relevant snippets with context
- Can run in parallel for independent chunks
- Sequential execution enforced when chunk dependencies exist

## Memory Storage

- Auto-chunked: 800 chars, 100 overlap
- Metadata: node_id, action_type, memory_type, success
- Unique namespaces per test run

## Memory Retrieval

- Split: Top N internal thoughts + Top M observations
- Semantic search via intent/topic
- Automatic context for every node

## Goal Tracking and Validation

- Original goals tracked through `GOAL` and `ORIGINAL_GOAL` detail keys
- Goals propagated from parent to children during expansion
- Merge nodes validate goal achievement by checking if goal appears in synthesized content
- `GOAL_ACHIEVED` flag stored in merge results

## Sequential vs Parallel Execution

- **Parallel**: Independent chunks from same document, independent actions
- **Sequential**: When data dependencies exist (visit needs URL from search, chunk depends on previous chunk)
- System automatically detects dependencies and enforces execution order

## Execution Data

- Telemetry: timing, success metrics
- Observability: tokens, searches, visits, durations
- Results: JSON with full trace
- LLM Inputs/Outputs: Full messages and responses logged for debugging