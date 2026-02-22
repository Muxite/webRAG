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

## Memory Storage

- Auto-chunked: 800 chars, 100 overlap
- Metadata: node_id, action_type, memory_type, success
- Unique namespaces per test run

## Memory Retrieval

- Split: Top N internal thoughts + Top M observations
- Semantic search via intent/topic
- Automatic context for every node

## Execution Data

- Telemetry: timing, success metrics
- Observability: tokens, searches, visits, durations
- Results: JSON with full trace
