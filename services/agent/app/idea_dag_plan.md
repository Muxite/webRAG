## Idea Graph Plan (Graph of Thoughts Inspired)

### Goals
- Replace the linear tick history with an idea graph that supports expansion, evaluation, and leaf decomposition.
- Use memoization (via Chroma) to avoid recomputing subproblems.
- Keep observability and data lineage first-class (inputs/outputs, timing, token usage).

### Phase 1: Core Idea Graph Model
- Define node lifecycle states and scoring strategy.
- Extend node metadata to capture rationale, evidence, and memoization keys.
- Add graph traversal helpers for selecting leaves and merging results.
- Add serialization schema for storage/telemetry export.

### Phase 2: Expansion (Idea Generation)
- Introduce an expansion policy interface to generate candidate child nodes from a parent.
- Store expansion prompts/outputs in node details for traceability.
- Add limits: branching factor, depth, and duplicate pruning by memoization key.

### Phase 3: Evaluation (Idea Selection)
- Introduce evaluation policies to score candidates based on path context.
- Support both local scores (node-level) and path scores (accumulated).
- Add selection strategy options: best score, beam search, diversity-aware.

### Phase 4: Leaf Decomposition and Merge
- Add a decomposition policy to split complex nodes into subproblems.
- Add a merge policy to combine child results into parent summaries.
- Define stopping criteria for leaf nodes (depth, confidence, or completion).

### Phase 5: Memoization and Reuse
- Define memoization key derivation rules (problem text, context, tool outputs).
- Store resolved subproblem artifacts in Chroma with retrieval hooks.
- When memo hits are found, attach prior results and skip recomputation.

### Phase 6: Agent Integration (Future)
- Replace tick loop with idea graph traversal loop.
- Each cycle: expand -> evaluate -> select -> act -> record -> merge.
- Align LLM calls with node-level operations and record per-node usage.

### Phase 7: Observability and Benchmarks
- Log per-node expansion/evaluation timings and inputs/outputs.
- Record graph snapshots and final merge artifacts in traces.
- Add benchmarks comparing linear vs idea graph strategies.
