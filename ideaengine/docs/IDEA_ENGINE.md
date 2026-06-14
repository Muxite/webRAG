# Idea Engine — In-Depth Architecture & Behavior

This document is a multi-agent deep dive into webRAG's **Idea Engine**: the Graph-of-Thought (GoT) controller that converts a free-form research mandate into a directed acyclic graph of LLM-driven thoughts, executes search/visit/think/save/merge actions across that graph, and synthesizes a single final deliverable.

It complements (and goes deeper than) [`AGENT_ARCHITECTURE.md`](./AGENT_ARCHITECTURE.md). Every load-bearing claim cites a file and line range so a reader can verify against the source.

---

## 1. What the Idea Engine Is

The engine is not a single function — it is a layered system composed of:

| Layer | Module | Role |
|---|---|---|
| **Orchestrator** | `idea_engine.py` (`IdeaDagEngine`, 1,447 lines) | Step loop, mandate enforcement, action dispatch, merge gating, final synthesis trigger |
| **Graph data model** | `idea_dag.py` (511 lines) | `IdeaNode`, `IdeaNodeStatus`, parent/child wiring, event-log table |
| **Branch model** | `idea_branch_pair.py` (199 lines) | The "expansion → intermediates → merge" pair abstraction |
| **Policies** | `idea_policies/{decomposition,expansion,evaluation,selection,merge,actions}.py` | Pluggable LLM-driven planning and execution |
| **GoT mechanics** | `got_operations.py` | Embedding, dedup, dynamic beam, prune, backtrack |
| **Memory** | `idea_memory.py` (`MemoryManager`, 428 lines) | ChromaDB read/write with sentence-aware chunking |
| **Finalization** | `idea_finalize.py` (`build_final_payload`, 496 lines) | Last-mile LLM call that turns the executed graph into a deliverable |
| **Checkpointing** | `idea_checkpointer.py` (197 lines) | JSON snapshot per step (file or Redis backend) |
| **Settings** | `idea_dag_settings.py` + `idea_dag_settings.json` (274 lines) | All tunable knobs; loader supports env-var overrides for token budgets |
| **Telemetry / analysis** | `idea_dag_log.py`, `idea_graph_analyzer.py`, `idea_graph_visualizer.py` | ASCII DAG, JSON export, post-run quality issue detection |
| **Test harness** | `idea_test_runner.py` (833), `idea_test_abstraction.py` (261), `idea_tests/` (39 scenarios) | Multi-model, multi-variant evaluation matrix |
| **Entry points** | `agent.py`, `interface_agent.py`, `main.py` | RabbitMQ worker, status publishing, engine invocation |

Conceptually the engine is **a step-driven controller that owns a DAG and a memory** and delegates every cognitive decision (decompose, evaluate, select, act, merge, synthesize) to a policy that hits an LLM.

---

## 2. Top-Level Construction & Lifecycle

`IdeaDagEngine.__init__` (`idea_engine.py:31–56`) wires the major collaborators:

| Attribute | Default | Source |
|---|---|---|
| `self.settings` | `load_idea_dag_settings()` | `idea_dag_settings.py:9` |
| `self.io` | required `AgentIO` | injected |
| `self.expansion` | `LlmExpansionPolicy` | `idea_policies/expansion.py` |
| `self.evaluation` | `LlmBatchEvaluationPolicy` | `idea_policies/evaluation.py:224` |
| `self.selection` | `BestScoreSelectionPolicy` | `idea_policies/selection.py:11` |
| `self.decomposition` | `ScoreThresholdDecompositionPolicy` | `idea_policies/decomposition.py:11` |
| `self.merge` | `SimpleMergePolicy` | `idea_policies/merge.py:12` |
| `self.actions` | `LeafActionRegistry` | `idea_policies/actions.py` |
| `self._memory_manager` | lazy `MemoryManager` | `idea_engine.py:63–66` |
| `self._got` | lazy `GoTOperations` | `idea_engine.py:67–71` |
| `self._checkpointer` | env-driven | `idea_engine.py:56` |

`run()` (`idea_engine.py:58–194`) is the public entry. It:

1. Splits the mandate text on `"\n\nTask Statement"` to separate a short summary from the full mandate (line 73).
2. Tries to restore from a checkpoint if one exists (`idea_engine.py:79–101`).
3. Drives the main loop until `max_steps`, node-cap, or quiescence (`idea_engine.py:107–155`).
4. Calls `build_final_payload(...)` (`idea_finalize.py:301`) to produce the deliverable (`idea_engine.py:167–174`).
5. Attaches GoT post-run stats: `dead_ends_detected`, `nodes_pruned`, `nodes_improved`, `parallel_leaves_total` (`idea_engine.py:176–190`).

---

## 3. The DAG Data Model

### IdeaNode (`idea_dag.py:10–24`)

| Field | Purpose |
|---|---|
| `node_id` | UUID |
| `title` | human-readable label |
| `details` | flexible dict — holds `ACTION`, `URL`, `QUERY`, `RESULTS`, `MERGED_RESULTS`, `REQUIRES_DATA`, `ACTION_ATTEMPTS`, `ACTION_COOLDOWN_UNTIL`, `MERGE_SUMMARY`, etc. |
| `parent_id` / `parent_ids` | single + multi-parent support (DAG, not just tree) |
| `status` | `IdeaNodeStatus` enum |
| `children` | list of child IDs |
| `score` | quality 0–1 (LLM-assigned) |
| `memo_key` | dedup memoization |

### Status enum (`idea_policies/base.py:69–78`)

`PENDING → ACTIVE → DONE` is the happy path. `BLOCKED` is set when a node is gated on a `REQUIRES_DATA` source. `SKIPPED` is used when sequential mode prunes siblings or a merge decides the goal is not pursuable. `FAILED` is terminal.

### Dependency edges

There is no separate edge type for data flow. Instead, nodes carry `DetailKey.REQUIRES_DATA` describing what they need from a peer:

```python
{
  "type": "urls_from_search" | "urls_from_visit" | "url_from_think" | "chunk_from_visit",
  "source_node_id": "<peer node id>"
}
```

`_has_required_data()` (`idea_engine.py:363–421`) validates the source is `DONE` *and* that the expected payload is non-empty (`RESULTS`, `LINKS`/`LINKS_FULL`, `URL`, or `CONTENT_FULL` depending on type).

### Branch pairs (`idea_branch_pair.py`)

Every fan-out follows the pattern `expansion_node → [intermediate_nodes] → merge_node`. `find_branch_pair()` (lines 123–173) locates the pair a given node belongs to; `needs_expansion()` and `needs_merge()` drive the loop's decisions about *when* to create children versus *when* to collapse them.

---

## 4. The Step Loop

`step()` (`idea_engine.py:196–305`) routes the *current* node to one handler:

| Condition | Handler | Line |
|---|---|---|
| Root with no children at step 0 | `_handle_expansion_node()` | 212 |
| Node is a merge node | `_handle_merge_node()` | 223 |
| Node is leaf or carries an action | `_handle_leaf_node()` | 230 |
| Node has no children | `_handle_expansion_node()` | 234 |
| All children terminal + merges present | mark DONE, bubble to parent | 254–264 |
| All leaves complete, no merges yet | create merge via `find_branch_pair()` | 272–276 |
| Child blocked by `REQUIRES_DATA` | execute source node first | 290–302 |
| Otherwise | `_handle_intermediate_node()` | 304 |

The loop is wrapped in three guards:

- **Step budget** — `while steps < max_steps` (`idea_engine.py:107`); the caller sets `max_ticks`, default flows from `agent.py:194`.
- **Node cap** — `max_total_nodes = 500` (`idea_engine.py:198`, settings line 3).
- **Quiescence** — if `step()` returns `None`, the loop breaks (`idea_engine.py:153–155`); a pending-nodes warning fires if there are still executable, unfinished nodes (`idea_engine.py:158–174`).

Two early validations help debug deadlocks: at step 1 it forces re-expansion of the root if it is still childless (`idea_engine.py:134–146`), and at step 3 it warns if zero action nodes have been created (`idea_engine.py:148–151`).

Every step optionally writes a checkpoint (`idea_engine.py:119–132`); every five steps it runs GoT pruning (`idea_engine.py:114–117`).

---

## 5. Decomposition, Expansion, Evaluation, Selection

### Decomposition gate (`idea_policies/decomposition.py:11–56`)

`ScoreThresholdDecompositionPolicy.should_decompose()` returns `True` when a node has no action *and* its score is below `decomposition_threshold` (default 0.5) *and* it does not already have a complete set of children. This is the upstream decision that *whether* to expand at all.

### Expansion (`idea_policies/expansion.py:23–103`)

`LlmExpansionPolicy` is the **single biggest LLM call** in the engine and the source of branching:

- System prompt is templated with `{allowed_actions}` and `{max_children}` (defaults `["search","visit","save","think","merge"]` and 5 — settings lines 9, 28).
- User prompt is a JSON blob describing the up-to-5-ancestor path, parent goal, blocked sites, prior errors, retrieved memories, and the event log (`expansion.py:225–339`).
- Call uses `json_mode=True` (`expansion.py:76`); the response is parsed by `_parse_candidates()` (line 500) into candidate dicts.
- **URL extraction** (`expansion.py:565–616`) — when the LLM proposes a `visit` action without a URL, the policy proactively scans inline `[link: URL]` markers and ancestor search-result snippets, and if the URL came from a search node it stamps `REQUIRES_DATA = {"type": "urls_from_search", "source_node_id": ...}` so dependency-resolution works correctly at execution time.
- Token caps come from settings: `expansion_max_tokens=8192`, `expansion_temperature=0.4`.

### Evaluation (`idea_policies/evaluation.py:70–429`)

Two implementations:

- `LlmEvaluationPolicy` (per-node) scores a single candidate against its path context. It enforces a hard penalty: nodes with an action but no `action_result` are capped at `no_action_result_score_cap=0.2`. `EvaluationWeights` then applies action-specific multipliers (search/visit/think/save).
- `LlmBatchEvaluationPolicy` (the default, lines 224–429) scores up to `evaluation_batch_max_candidates=5` candidates in a single LLM call. It builds an internal map of `simple_id (1, 2, 3 …) → real UUID` and parses a JSON response shaped `{"scores":[{"id":"1","score":0.85}, ...]}` (line 362). All weighting and penalties apply per candidate. Token cap `evaluation_max_tokens=16384`, temp `0.2`.

### Selection (`idea_policies/selection.py:11–28`)

`BestScoreSelectionPolicy.select()` returns the highest-scored child of a given parent. In **best-first global** mode (`settings["best_first_global"]`, `idea_engine.py:909`) the engine instead calls `_select_best_global()` (`idea_engine.py:1408–1428`) which scans the entire graph and lets a higher-scoring sibling under a *different* parent jump the queue.

### Dynamic beam width

`GoTOperations.compute_dynamic_beam_width()` (`got_operations.py:300–344`) narrows the beam when score p25/p75 spread is small (the LLM is converging) and widens it when scores are spread out. Bound by `got_beam_min=2`, `got_beam_max=5`.

---

## 6. Leaf Actions

Implemented in `idea_policies/actions.py`. Each action receives the node, the graph, and `AgentIO`, and returns a dict that `ActionResultExtractor` knows how to read.

| Action | Class | Highlights |
|---|---|---|
| `search` | `SearchLeafAction` (166–260) | Calls `io.search()`; can also drive chunk-search for oversized documents (lines 182–185). |
| `visit` | `VisitLeafAction` (263–1448) | Fetches a page via `AgentIO.visit` (with browser fallback), extracts links, stores them as ChromaDB memories. URL resolution priority: explicit `url` → `_extract_url_from_think_node` (1632–1663) → `_extract_url_from_parents` (553–630) → `_extract_url_from_sibling_results` (665–746). When `REQUIRES_DATA = {"type":"urls_from_search","source_node_id":X}` is set, it pulls results from the source node's `RESULTS` array (360–398). Semantic link discovery is done by querying ChromaDB with the `link_idea` text (801–863). |
| `think` | `ThinkLeafAction` (1451–1573) | Extracts URLs from `REQUIRES_DATA` source nodes (1452–1498); stores its reasoning as `internal_thought` memory. |
| `save` | `SaveLeafAction` (1576–1610) | Wraps `io.store_chroma()` with metadata. |
| `merge` | `MergeLeafAction` (1613–1777) | LLM-driven synthesis using `merge_system_prompt` and `merge_user_prompt`. Expects JSON `{"goal_achieved":bool,"goal_evaluation":str,"missing_requirements":[…]}`. Sets parent `DONE` if `goal_achieved`. |

`action_constants.py` (468 lines) collects the JSON keys, error types, and reusable builders (`ActionResultBuilder`, `PromptBuilder`, `NodeDetailsExtractor`, `ActionResultExtractor`) so action code stays uniform.

### Action dispatch lifecycle (`idea_engine.py:942–1020`)

```
dedup check      → graph.has_executed_action()       (line 950)
site block check → graph.is_site_blocked(url)        (line 962)
action lookup    → enum + allowed_actions validation (line 980)
attempt tracking → details[ACTION_ATTEMPTS]++        (line 994)
execute          → action.execute(graph, id, io)     (line 1004)
sanitize         → primitives + nested dicts/lists   (line 1005)
memory write     → memory_manager.write_node_result  (line 1008)
mark executed    → graph.mark_action_executed()      (line 1017)
```

### Retry & cooldown

`ActionResultExtractor.is_retryable(result)` drives a soft-retry policy: when retryable and `attempts <= max_retries` (default `action_max_retries=2`), the node is marked `BLOCKED` with `ACTION_COOLDOWN_UNTIL = step_index + backoff` (`idea_engine.py:1070–1078`). When the cooldown elapses, the node becomes ready again via `_is_action_ready()` (`idea_engine.py:1387–1393`). Visit nodes whose call "succeeded" but returned empty content are forcibly flipped to `FAILED` (`idea_engine.py:1042–1063`) so they don't poison downstream merges.

---

## 7. Mandate Enforcement

The engine treats the user's mandate as an authority that can *demand* certain actions exist in the graph, regardless of what the expansion LLM proposes. Two layers:

### A. URL extraction from mandate text (`idea_engine.py:556–615`)

`_enforce_visit_nodes_for_mandate_urls()`:

- Regex: `r'https?://[^\s<>"{}|\\^`\[\]]+'` (line 572).
- `_clean_extracted_url()` (lines 725–738) trims trailing punctuation and balances parentheses for Wikipedia-style URLs.
- For each URL not already covered by an existing VISIT child, it injects a child with `ACTION=VISIT`, `URL=<extracted>`, `IS_LEAF=True`, `JUSTIFICATION="Mandate requires visiting this URL"`.

### B. Phrase-based requirements (`idea_engine.py:617–723`)

`_enforce_mandate_visit_requirements()` looks for phrases like *"must visit"*, *"visit the URL"*, *"must search"*, *"search for"* (lines 638–645). If the requirement isn't satisfied, it injects a SEARCH or VISIT node — and when both are required, it wires the visit's `REQUIRES_DATA` to the search node so order is correct:

```python
REQUIRES_DATA = {
  "type": "urls_from_search",
  "source_node_id": <search_node_id>
}
```

This is what lets the engine reliably honor mandates like *"Search for X, then visit the top result"* even if the planner's first decomposition doesn't get it right.

There is also a **mandate addendum** (`mandate_addendum.py`) injected into LLM prompts via `effective_mandate(...)` (`agent.py:196, 215`) to warn the planner that *"web visits may be blocked here"* when `AGENT_MANDATE_ADDENDUM_ENABLED` is set.

---

## 8. Merge Gating & Synthesis

`SimpleMergePolicy` (`idea_policies/merge.py:12–240`) defines four ordered predicates:

| Method | Returns true when |
|---|---|
| `are_children_ready_to_merge()` | all children terminal (`DONE`/`FAILED`/`BLOCKED`/`SKIPPED`) |
| `should_create_merge_node()` | ≥2 ready children and no merge child yet |
| `create_merge_node()` | adds a merge child and copies `MERGED_RESULTS` |
| `merge()` | populates `MERGED_RESULTS` + `MERGE_SUMMARY` on the parent |

The actual merge **LLM call** lives in `MergeLeafAction.execute()` (`actions.py:1613–1777`), which reads `MERGED_RESULTS`, fills `merge_system_prompt`/`merge_user_prompt` templates, and parses a JSON envelope. Token cap is `merge_max_tokens=100000` (settings line ~140), temperature 0.3.

The engine wires this together in `_handle_merge_creation()` (`idea_engine.py:740–784`). If the merge LLM responds with `goal_achieved=False`, the merge can be marked `SKIPPED` via the `merge_should_skip` flag (line 760) — i.e., the engine acknowledges the branch didn't accomplish its sub-goal rather than pretending it did.

---

## 9. Memory Layer

`MemoryManager` (`idea_memory.py`) is the only file-level abstraction over ChromaDB used by the engine:

- One collection per run: `mem_{namespace_hash}` (`idea_memory.py:34–35`); the raw namespace is stored in metadata (line 213) so cross-run bleed is impossible.
- Two memory types stored in the same collection: `"observation"` (from search/visit results) and `"internal_thought"` (from think/save). Memory type is auto-detected from the originating action (lines 203–208).
- **Chunking**: 800-char chunks with 100-char overlap; the splitter searches the trailing 20% of each chunk for a sentence-boundary character (`.`, `!`, `?`, `\n\n`) so chunks don't shred sentences (`idea_memory.py:146–153`).
- Parallel write fires for >20 chunks (lines 259–265).
- API surface used by the engine:
  - `retrieve_relevant_memories(query, …)` — vector search with optional memory_type filter
  - `retrieve_memories_split(query, …)` — returns `{"internal_thoughts":[…], "observations":[…]}`
  - `write_memory(...)` and the higher-level `write_node_result(node, action_result)`
  - `format_memories_for_llm(...)` — formats results with a 2000-char budget by default

Links discovered during a visit are *not* stored in per-URL collections (despite the README's wording); they are embedded into the standard observation memory inline (`idea_memory.py:246–253`).

---

## 10. Graph-of-Thought Mechanics

`GoTOperations` (`got_operations.py`) is the layer of "graph-of-thought" smarts that sits above raw planning:

| Operation | Method | Behavior | Defaults |
|---|---|---|---|
| Embed | `embed_thought()` (25–61), `embed_children()` (63–95) | Writes node title/goal/action into memory as `internal_thought` | `got_embed_on_create=true` |
| Dedup | `is_duplicate_thought()` (226–265), `_adaptive_dedup_threshold()` (204–224) | Queries memory; rejects candidates above similarity threshold | static 0.85 default; adaptive based on fanout |
| Beam | `compute_dynamic_beam_width()` (300–344) | Narrows on tight p25/p75 score spread, widens on broad spread | `[got_beam_min=2, got_beam_max=5]` |
| Prune | `identify_prune_candidates()` (346–385), `prune_nodes()` (387–402) | Removes low-scoring nodes once the graph is large enough; skips root, done, failed, skipped | trigger >6 nodes, score < 0.15 (or adaptive σ) |
| Backtrack | `should_backtrack()` (404–430), `find_backtrack_target()` (432–442) | Detects 3+ consecutive low-score nodes; targets nearest parent with score ≥ 0.3 | `got_backtrack_enabled=false` |
| Improve | `try_improve_node()` (97–202) | LLM-driven refinement of low-score nodes | `got_improve_enabled=false` |
| Hybrid retrieve | `hybrid_retrieve()` (444–490) | Combines ChromaDB vector hits with graph-path context | — |
| Model routing | `get_model_for_operation()` (492–529) | Lets scoring and generation use different models | — |

Notably, **improve and backtrack are off by default** — the engine relies on dedup + dynamic beam + prune for quality, and on the dependency-edge + REQUIRES_DATA system for correctness.

---

## 11. Final Synthesis

When the loop exits, `build_final_payload()` (`idea_finalize.py:301–496`) produces the deliverable. Inputs:

1. **Merged root result** — extracted from the root node's `MERGED_RESULTS`, falling back to a leaf collection if empty (lines 309–318).
2. **Raw visit content** — `_collect_all_visit_content()` (66–105) gathers every successful VISIT's content, capped at 80,000 chars total (line 72).
3. **Memory context** — `_retrieve_final_chroma_context()` (218–298) issues parallel ChromaDB queries against the mandate, merge summaries, node titles, and discovered URLs, then dedups by memory id and caps at 80,000 chars.
4. **Node summary table** — `_build_node_summary_table()` (155–215) lists every node with action/status/outcome.
5. **Event log** — ancestor decision trail from `idea_dag.build_event_log_table()` (`idea_dag.py:396–512`).
6. **Mandate** — the original user query.

The final prompt is built either from a custom template (`idea_finalize.py:373–405`) or from `FinalPromptBuilder`. A *Runtime capability* clause (lines 376–381) tells the LLM what tools the agent had access to. Token budget is `final_max_tokens=120000` (settings line ~205), and the timeout adapts to the prompt size: base 180s + 1s per 60 chars, capped at 600s (lines 435–440).

Success criteria (lines 460–485):

- `goal_achieved` — set if root or any merge node reports `goal_achieved=True`.
- `has_critical_failures` — true if any SEARCH/VISIT/MERGE actions failed.
- `success = deliverable_non_empty AND (goal_achieved OR no_critical_failures)`, relaxed by `final_allow_partial_success`.

If the LLM call fails, a degraded fallback deliverable is assembled from graph data so the caller never gets nothing (lines 452–453).

---

## 12. Sequential vs Graph Mode

The engine supports two execution shapes, governed by `mode` in settings and by data-dependency detection:

### Graph (parallel branching)

- Expand into 2–5 candidates → score all → best-first global selection → execute leaves in parallel via `asyncio.gather()` (`idea_engine.py:821–867`).
- `_parallel_leaves_total` counts parallel executions (line 844).
- Per-action timeout converts to `FAILED` (lines 847–855).

### Sequential (generate-many, keep-one)

- Same candidate generation, but `_reorder_for_sequential()` (`idea_engine.py:1100–1165`) enforces data-flow ordering: visits without URLs defer to a search sibling; think/save/merge prioritize data-producing siblings.
- Non-selected siblings are marked `SKIPPED` when `sequential_prune_siblings=True` (lines 931–938).
- Resource caps tightened: 80 max nodes, 40k observation chars, 5 links per visit (applied by the test runner — `idea_test_runner.py:239–256`).

### Forced sequential

Two detectors short-circuit parallelism even when mode is graph:

- `_detect_state_dependencies()` (`idea_engine.py:1167–1203`) — search+visit pair without URL.
- `_detect_chunk_dependencies()` (`idea_engine.py:1302–1329`) — ordered chunk processing.

When either fires, `execute_all=False` (line 815) and the engine falls back to one-at-a-time even within graph mode.

---

## 13. Settings Surface

`idea_dag_settings.json` is the single source of truth (274 lines). Grouped:

**Execution control** (lines 2–10): `max_branching=5`, `max_total_nodes=500`, `enable_idea_dag=true`, `evaluation_strategy="best_first"`, `selection_strategy="best_score"`, `min_score_threshold=0.0`, `allow_unscored_selection=true`.

**Document & memory** (13–26): `max_observation_chars=100000`, `document_chunk_threshold=200000`, `document_chunk_size=4000`, `document_chunk_overlap=400`, `visit_link_query_top_k=10`, `max_links_per_visit=20`.

**Action execution** (28–42): `allowed_actions=["search","visit","save","think","merge"]`, `action_max_retries=2`, `action_timeout_seconds=20`, `search_timeout_seconds=15`, `visit_timeout_seconds=20`, `final_timeout_seconds=180`.

**Model & token budgets** (49–207):

| Phase | Model | Max tokens | Temp |
|---|---|---|---|
| Expansion | `expansion_model` (empty → inherit) | 8,192 | 0.4 |
| Evaluation | `evaluation_model` | 16,384 | 0.2 |
| Merge | `merge_model` | 100,000 | 0.3 |
| Final | `final_model` | 120,000 | 0.3 |

Token budgets are overridable via env vars `IDEA_DAG_EXPANSION_MAX_TOKENS`, `IDEA_DAG_EVALUATION_MAX_TOKENS`, `IDEA_DAG_MERGE_MAX_TOKENS`, `IDEA_DAG_FINAL_MAX_TOKENS` (`idea_dag_settings.py:16–32`).

**GoT toggles** (252–273):

| Setting | Default | Effect |
|---|---|---|
| `got_embed_on_create` | true | every node embedded at creation |
| `got_dedup_enabled` / `got_dedup_similarity_threshold` | true / 0.85 | reject near-duplicate candidates |
| `got_dynamic_beam_enabled` / `got_beam_min` / `got_beam_max` | true / 2 / 5 | adaptive branching |
| `got_prune_enabled` / `got_prune_score_threshold` / `got_prune_min_nodes_before_prune` | true / 0.15 / 6 | prune low-score branches once graph is large |
| `got_improve_enabled` | false | refinement loop disabled |
| `got_backtrack_enabled` | false | no backwards recovery |

**Prompt overrides** (166–207) — the JSON embeds *complete* system prompts per phase. The expansion prompt enforces visit requirements ("If mandate says 'must visit' … you MUST create a visit action node"), and the final prompt requires the LLM to include verbatim quotes from raw visit content to prove facts came from the actual page.

`idea_settings.py` (the 20-line file) is unrelated — it loads from a non-existent `idea_graph_settings.json` and appears to be legacy or unused.

---

## 14. Checkpointing, Logging, Analysis

### Checkpointer (`idea_checkpointer.py`)

Abstract `Checkpointer` interface (lines 18–60) with two backends:

- `FileCheckpointer` (62–115) — writes `.checkpoints/{run_id}/{step:04d}.json` plus a `latest.json` symlink (86–91).
- `RedisCheckpointer` (117–171) — single key `euglena:checkpoint:{run_id}` with 86,400s TTL (lines 126, 146).

Snapshot is JSON: `{run_id, step_index, saved_at, snapshot}` (80–85, 139–144). Enabled via `IDEA_CHECKPOINT_ENABLED`. Used for crash recovery and replay.

### DAG event log (`idea_dag.py:396–512`)

`build_event_log_table()` produces an in-prompt table of ancestor decisions: `[status] action — title (summary)`. Each row shows why a node was created, the URL or query, the result size, and any error. This is the "agent reasoning trail" injected into expansion, merge, and final prompts so the LLM can see what already happened on the path it is operating on. Cap: 20 events per path.

### ASCII / JSON DAG (`idea_dag_log.py`, `idea_graph_visualizer.py`)

- `idea_dag_to_ascii()` (`idea_dag_log.py:55–82`) — asciidag tree for terminal output.
- `idea_dag_data()` (`idea_dag_log.py:85–107`) and `idea_graph_data()` (`idea_graph_visualizer.py:19–44`) — `{nodes:[…], edges:[…]}` JSON for the frontend.

There is **no Mermaid or Graphviz output** — it's ASCII + JSON only.

### Graph analyzer (`idea_graph_analyzer.py`)

Post-run quality detector. Computes `total_nodes`, `action_counts`, `think_ratio`, `unique_titles`, `duplicate_title_count`, `action_diversity` (`idea_graph_analyzer.py:81–86, 144`). Flags four issue types (lines 89–152):

1. Excessive think actions (>50% high severity, >30% medium).
2. Repeated node titles.
3. Similar long titles (>50 chars) — typically duplicate merge nodes.
4. Low action diversity (<2 distinct actions in >3 nodes).

Each issue carries `{severity, type, message, recommendation, examples}`.

---

## 15. Test Harness

`idea_test_runner.py` (833 lines) drives the eval matrix over the 39 scenarios in `idea_tests/` (`test_001_…` to `test_039_…`):

- **Test selection** — `TEST_PRIORITY_ORDER` ranks tests by difficulty (lines 116–149). `IDEA_TEST_TOP_N` selects top-N by priority; `IDEA_TEST_IDS` overrides explicitly. Benchmark mode (lines 668–674) uses `select_benchmark_test_files()` (289–351) to pick a balanced 8-test subset that includes visit-heavy scenarios.
- **Variants** — each test is run in both `graph` and `sequential` modes. Sequential applies tighter caps: beam disabled, max nodes 80, observations capped at 40k (lines 239–256).
- **Matrix execution** — `for test × model × variant × repeat`, gated by `asyncio.Semaphore(max_parallel)` (lines 764–772).
- **Output** — per-test JSON with `{test_metadata, model, execution_variant, validation, timestamp, …}` (line 468); summary JSON aggregates everything; console table prints test ID, model, variant, pass/fail, score, visit count, node count, duration (lines 502–541).

`idea_test_abstraction.py` (261 lines) defines `IdeaTestModule` (30–72), `run_test_execution()` (75–187), and `run_complete_test()` (192–249). `idea_test_utils.py` (59 lines) is text helpers.

`preflight_llm_test.py` (128 lines) is a standalone pytest that loads settings, builds an `IdeaDagEngine`, runs one step with a tiny mandate, and verifies expansion produced children — a smoke test for the LLM + settings combination.

---

## 16. End-to-End Entry Points

```
RabbitMQ task
     │
     ▼
InterfaceAgent._handle_task()           interface_agent.py:340
     │  parses payload, special modes (visit / skip / normal)
     ▼
Agent(mandate, …).run()                 agent.py:169–359
     │  if AGENT_USE_IDEA_DAG=1:
     ▼
IdeaDagEngine(io, model_name, settings).run(effective_mandate)
     │                                  idea_engine.py:58
     │  loop: step() → expand / score / select / act / merge
     │                                  idea_engine.py:107–155
     ▼
build_final_payload(io, settings, graph, mandate, model, memory)
     │                                  idea_finalize.py:301
     ▼
{final_deliverable, success, goal_achieved, has_failures, got_stats}
     │
     ▼
InterfaceAgent publishes COMPLETED / ERROR  interface_agent.py:625–648
```

`main.py` (109 lines) wraps this with a tiny aiohttp service that exposes `/health` and `/version` on port 8081 and supervises the `InterfaceAgent` worker loop.

---

## 17. Non-Obvious Behaviors Worth Knowing

These are the gotchas a reader should know before debugging or extending the engine:

1. **URL cleanup is regex-aware of Wikipedia.** `_clean_extracted_url()` (`idea_engine.py:725–738`) balances parentheses so `…/Apple_(company)` survives intact instead of being truncated to `…/Apple_(company`.
2. **Action dedup is global, not parent-scoped.** `graph.has_executed_action()` (`idea_engine.py:950–959`) lets a second node reuse a prior identical search/visit result; this avoids redundant API calls but means two siblings can effectively share evidence.
3. **Large documents auto-split.** Docs >200,000 chars create chunked search sub-problems with `REQUIRES_DATA` edges back to the original visit node (`idea_engine.py:1205–1278`).
4. **Emergency root expansion.** If the root is still childless after step 1, the engine forces a re-expansion with detailed logging (`idea_engine.py:134–146`) — useful for debugging planner stalls.
5. **Best-first global breaks parent locality.** With `best_first_global=true` (`idea_engine.py:909`), a higher-scoring node under a different parent can preempt the current branch's next step.
6. **Mandate addendum is invisible to the user.** It's appended at LLM-call assembly time (`mandate_addendum.py`, `agent.py:196,215`), so users never see warnings like "web visits may be blocked here" in their own copy.
7. **Visit "success" with empty body is treated as failure.** `idea_engine.py:1042–1063` flips such results to FAILED so downstream merges don't synthesize hallucinations from nothing.
8. **`idea_settings.py` is dead code.** It loads a non-existent `idea_graph_settings.json` and returns `{}`. The live loader is `idea_dag_settings.load_idea_dag_settings()`.
9. **Improve and backtrack are intentionally off.** The defaults choose dedup + dynamic beam + prune over the more expensive GoT options.
10. **Final prompt is adaptive.** Timeout scales with prompt size up to 600s (`idea_finalize.py:435–440`), and a degraded fallback deliverable is built from graph data if the final LLM call dies (`idea_finalize.py:452–453`).

---

## 18. File Map

| File | Lines | Responsibility |
|---|---|---|
| `idea_engine.py` | 1,447 | Step loop, mandate enforcement, action dispatch, merge gating, final-synthesis trigger |
| `idea_dag.py` | 511 | `IdeaNode`, status enum, event-log table |
| `idea_branch_pair.py` | 199 | Expansion / intermediates / merge pair abstraction |
| `idea_policies/base.py` | — | Abstract policy interfaces, `IdeaNodeStatus`, `DetailKey` |
| `idea_policies/decomposition.py` | 56 | "Should I expand this node?" decision |
| `idea_policies/expansion.py` | ~700 | LLM-driven decomposition into candidate children + URL extraction |
| `idea_policies/evaluation.py` | 429 | Per-node and batch scoring policies |
| `idea_policies/selection.py` | 28 | Best-score selection |
| `idea_policies/merge.py` | 240 | Merge gating + node creation |
| `idea_policies/actions.py` | 1,777 | Leaf actions (search/visit/think/save/merge) |
| `idea_policies/action_constants.py` | 468 | Result keys, error types, builders, extractors |
| `idea_memory.py` | 428 | `MemoryManager` — ChromaDB read/write, chunking |
| `got_operations.py` | ~530 | Embedding, dedup, beam, prune, backtrack, improve |
| `idea_finalize.py` | 496 | `build_final_payload` — last-mile LLM call |
| `idea_checkpointer.py` | 197 | File + Redis checkpoint backends |
| `idea_dag_log.py` | 133 | ASCII + JSON DAG visualization |
| `idea_graph_analyzer.py` | 215 | Post-run quality metrics + issue detection |
| `idea_graph_visualizer.py` | 108 | Graph JSON for frontend |
| `idea_dag_settings.py` | 34 | Settings loader + env overrides |
| `idea_dag_settings.json` | 274 | Canonical config |
| `idea_settings.py` | 20 | (legacy / unused) |
| `mandate_addendum.py` | 37 | System-note injection into LLM prompts |
| `idea_test_runner.py` | 833 | Multi-test, multi-model, multi-variant matrix |
| `idea_test_abstraction.py` | 261 | Test module loading + execution pipeline |
| `idea_test_utils.py` | 59 | Text helpers |
| `preflight_llm_test.py` | 128 | Smoke test for expansion + settings |
| `interface_agent.py` | 702 | RabbitMQ worker, status publishing |
| `agent.py` | 584 | Agent wrapper that constructs and runs `IdeaDagEngine` |
| `main.py` | 109 | Service entry: health endpoints + worker lifecycle |
| `idea_tests/` | 39 files | Test scenarios (basic retrieval → extreme multi-source synthesis) |

---

## 19. Mental Model Summary

The Idea Engine is best understood as **a controller that owns a graph and a memory** and outsources every cognitive decision to an LLM-backed policy:

- *Plan* via `LlmExpansionPolicy` (decompose into 2–5 candidates).
- *Judge* via `LlmBatchEvaluationPolicy` (score candidates 0–1).
- *Pick* via `BestScoreSelectionPolicy` (or best-first global).
- *Act* via the leaf action registry (search / visit / think / save).
- *Synthesize* via `MergeLeafAction` upward and `build_final_payload` at the root.
- *Discipline* the whole thing with mandate enforcement, REQUIRES_DATA edges, dedup, dynamic beam, prune, retries, cooldowns, and checkpointing.

The system's "intelligence budget" is spent in three places: the **expansion call** (planning), the **batch evaluation call** (judgment), and the **final synthesis call** (writing the answer). Everything else — graph management, dependency resolution, mandate enforcement, telemetry — is deterministic Python whose job is to keep those three calls grounded and accountable.
