# Agent Architecture

## Overview

The agent implements a **Graph-of-Thought (GoT)** execution model. A controller (`IdeaDagEngine`) builds a directed acyclic graph of thought nodes, decomposes tasks into subproblems, executes leaf actions (search, visit, think, save), and merges results upward. Two execution modes are supported: `graph` (parallel branching) and `sequential` (generate-many-keep-one, depth-first).

## Core Operations

| Operation | Class | What It Does |
|---|---|---|
| **Generate** | `LlmExpansionPolicy` | Decomposes a parent node into 2–5 child subproblems via LLM |
| **Score** | `LlmBatchEvaluationPolicy` | Rates candidates 0–1 in a single LLM call; best-first selection |
| **Aggregate** | `SimpleMergePolicy` | Synthesizes child results via LLM; propagates upward to root |

## Node Types

| Type | Description |
|---|---|
| Expansion | No action. Decomposes into children via LLM. |
| Leaf | Executes one action: `search`, `visit`, `save`, `think`. |
| Merge | Synthesizes child results upward. |

## Leaf Actions

| Action | Input | Output |
|---|---|---|
| `search` | Query string | Title/URL/snippet results |
| `visit` | URL or semantic link query | Full page content + discovered links |
| `think` | Parent context | Internal reasoning |
| `save` | Documents | Stored in ChromaDB memory |
| `merge` | Child results | LLM synthesis |

### URL Extraction and Dependencies

Visit actions extract URLs from parent search results:
- **Source detection**: Checks sibling nodes with `REQUIRES_DATA` dependency type
- **URL extraction**: Only extracts URLs from `IdeaNodeStatus.DONE` search nodes
- **Dependency setup**: When mandate requires "visit the URL found in search results", dependencies are automatically configured so visit nodes wait for search completion
- **Structured logging**: All URL extraction events are logged with CloudWatch-compatible structured logs for debugging

## Execution Modes

### Graph (parallel branching)

1. Expand node into 2–5 candidates via LLM.
2. Dynamic beam width adapts branching to score quality.
3. Score all children; select best via best-first global selection.
4. Execute leaf actions in parallel where children are independent.
5. Merge results propagate upward recursively.

### Sequential (generate-many, keep-one)

1. Expand node into 2–5 candidates (same LLM call as graph).
2. Score all candidates in a batch evaluation.
3. Select the highest-scoring candidate; mark all siblings as `SKIPPED` (`sequential_prune_siblings`).
4. Proceed depth-first into the selected child only.
5. Resource caps enforced: 80 max nodes, 40k observation chars, 5 links per visit.

Sequential generates the same candidates as graph but explores a single path, making it structurally inferior on tasks requiring parallel information gathering.

## Execution Loop

1. Create root node from mandate; initialize `MemoryManager` and `GoTOperations`.
2. **Mandate enforcement**: After initial expansion, `_enforce_mandate_visit_requirements()` injects missing search/visit nodes if the mandate explicitly requires them (e.g., "visit the URL found in search results").
3. **Main loop** (while steps < max_steps):
   - Expand node if no children.
   - Score children; select best via best-first global selection (graph) or prune-and-pick (sequential).
   - Execute leaf action or trigger merge when all children complete.
   - Merge results propagate upward recursively.
4. **Final synthesis**: merged results + execution trail + raw visit content + ChromaDB context → LLM → deliverable.

## GoT Mechanics

| Feature | Setting | Description |
|---|---|---|
| Thought embedding | `got_embed_on_create = true` | All nodes embedded in ChromaDB at creation |
| Deduplication | `got_dedup_enabled = true` | Rejects candidates with >0.85 similarity |
| Dynamic beam | `got_dynamic_beam_enabled = true` | Branching adapts to scores (min 2, max 5) |
| Pruning | `got_prune_enabled = true` | Removes nodes scoring <0.15 when graph >6 nodes |
| Improve | `got_improve_enabled = false` | Disabled for efficiency |
| Backtracking | `got_backtrack_enabled = false` | Disabled for efficiency |

## Data Flow

```
Mandate → Root
  ├─ Generate → [Search, Visit, Think, Save] children
  │   ├─ Search → URLs + snippets
  │   ├─ Visit → Full page content + links → ChromaDB
  │   ├─ Think → Internal reasoning → ChromaDB
  │   └─ Save → Store findings → ChromaDB
  └─ Merge → Synthesize children → Parent result
      └─ Final LLM → Deliverable
```

## Connectors

| Connector | Layer | Description |
|---|---|---|
| `ConnectorLLM` | Low-level | OpenAI API interaction (chat completions) |
| `ConnectorSearch` | Low-level | Brave Search API queries |
| `ConnectorHttp` | Low-level | aiohttp page fetching with retry and status handling |
| `ConnectorBrowser` | Low-level | undetected-chromedriver fallback for bot-protected sites |
| `ConnectorChroma` | Low-level | ChromaDB connection, add/query operations |
| `AgentIO` | High-level | Unified interface wrapping all connectors with fallback logic and telemetry |

### Browser Fallback

`AgentIO.visit()` and `AgentIO.fetch_url()` first attempt the request via `ConnectorHttp` (aiohttp). If the response is 401/403 or indicates bot blocking, the request is automatically retried through `ConnectorBrowser` (headless Chrome via undetected-chromedriver). The browser is lazily initialized on first use and runs Selenium calls in a thread pool to avoid blocking the async event loop.

## Memory

`MemoryManager` stores content in ChromaDB with two memory types:

| Type | Source | Used For |
|---|---|---|
| `observation` | Search/visit results | RAG context during expansion and final synthesis |
| `internal_thought` | Think/save actions | Internal reasoning context |

Content is chunked (800 chars, 100 overlap) with sentence-boundary splitting.

## Link System

Visit actions store discovered links in per-URL ChromaDB collections. Subsequent visit actions query these collections semantically to find relevant links without re-visiting pages.

## Observability

Telemetry is layered through base classes so action code stays clean:

| Layer | Mechanism | Data |
|---|---|---|
| `ConnectorBase` | `_record_timing()`, `_record_io()` | Every external call: duration, status, payload size |
| `AgentIO` | Method-level telemetry | Visit/search/store/retrieve with fallback tracking |
| `IdeaDagEngine` | Structured logging | Step index, node type, candidate counts, merge results, mandate enforcement |
| `GoTOperations` | Event logging | Embedding, dedup hits, beam width, prune events |
| `MemoryManager` | Operation logging | Chunk counts, retrieval results, namespace isolation |
| `Actions` | Structured logging | URL extraction, dependency checks, error messages (CloudWatch-compatible JSON) |
| Test Runner | JSON output | Per-test: score, pass/fail, cost, tokens, duration, graph depth/branching/nodes |

## Models

| Model | Input $/M | Output $/M |
|---|---|---|
| gpt-5.2 | $1.75 | $14.00 |
| gpt-5-mini | $0.25 | $2.00 |
| gpt-5-nano | $0.05 | $0.40 |

## Key Limits

| Parameter | Value |
|---|---|
| `max_observation_chars` | 100,000 |
| `document_chunk_threshold` | 200,000 |
| `final_max_tokens` | 120,000 |
| `merge_max_tokens` | 100,000 |
| `expansion_max_tokens` | 8,192 |
| `evaluation_max_tokens` | 16,384 |

## File Map

| File | Responsibility |
|---|---|
| `idea_engine.py` | DAG traversal, step loop, merge gating, sequential prune |
| `idea_dag.py` | Node/edge data structure |
| `idea_policies/` | Expansion, evaluation, selection, merge, actions |
| `idea_memory.py` | ChromaDB read/write with chunking |
| `agent_io.py` | Unified connector interface (LLM, search, HTTP, browser, Chroma) |
| `got_operations.py` | Embedding, dedup, beam width, pruning |
| `connector_llm.py` | LLM API interaction |
| `connector_search.py` | Brave Search API |
| `connector_http.py` | aiohttp HTTP fetching |
| `connector_browser.py` | undetected-chromedriver fallback |
| `connector_chroma.py` | ChromaDB connection and operations |
| `connector_base.py` | Base class with timing/IO telemetry |
| `idea_finalize.py` | Final synthesis and deliverable building |
| `idea_dag_settings.json` | All tunable parameters |
| `idea_test_runner.py` | Test orchestration with variant settings |
| `testing/` | Visualization, summary, helpers, core plots |

## Connector Glossary

| Method | Layer | Description |
|---|---|---|
| `ConnectorChroma.add_to_chroma` | Low-level | Adds documents to a ChromaDB collection |
| `ConnectorChroma.query_chroma` | Low-level | Queries ChromaDB for nearest neighbors |
| `ConnectorHttp.request` | Low-level | HTTP GET/POST with retry logic |
| `ConnectorBrowser.fetch_page` | Low-level | Headless Chrome page fetch |
| `ConnectorSearch.query_search` | Low-level | Sends query to Brave Search API |
| `AgentIO.visit` | High-level | HTTP fetch with browser fallback + telemetry |
| `AgentIO.fetch_url` | High-level | URL fetch with browser fallback + telemetry |
| `AgentIO.search` | High-level | Wraps `query_search` + document tracking + telemetry |
| `AgentIO.store_chroma` | High-level | Wraps `add_to_chroma` + metadata prep + telemetry |
| `AgentIO.retrieve_chroma` | High-level | Wraps `query_chroma` + result parsing + telemetry |
