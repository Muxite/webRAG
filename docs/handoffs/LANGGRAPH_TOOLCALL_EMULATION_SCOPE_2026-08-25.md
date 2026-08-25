# langgraph_react tool-calling emulation for weak/non-tool-calling models — scope (not built)

2026-08-25. Phase 2 of the constrained-decoding session (Phase 1: `run_policy_constrained_decoding_enabled`,
shipped in `ef801010`). This document scopes the work precisely enough to build in a future session;
nothing here is implemented yet.

## Why

Direction for this project: stay focused on the lower end of the model-capability spectrum that can
still tool-call — quantized local models in the 6-16GB VRAM class (qwen2.5:0.5b/1.5b/7b/14b,
llama3.2:1b/3b, gemma2:2b, phi3:mini, tinyllama, llama3.1:8b) — not frontier-adjacent models, and not a
no-tool-calling pivot. Today only the native `graph` engine can run the very bottom of that roster
(`tinyllama`, `phi3:mini`): `langgraph_react` hard-fails on them because LangGraph's
`create_react_agent` calls `model.bind_tools(tools)` internally, which unconditionally attaches an
HTTP `tools` parameter Ollama rejects with a 400 for models with no tool-calling template (documented
in `scripts/adaptive_ladder_run.py`'s `capspec_local` axis comment, "probed 2026-08-15"). Building an
emulation shim would let `langgraph_react` run the same weak-model roster the native engine already
does, giving a much richer 3-way comparison space at exactly the tier this project targets.

## What's already confirmed (this session's research, not re-derive)

- **`sequential_react`** (`agent/app/testing/execution_sequential.py`) needs NO shim: it parses actions
  via plain `json_mode=True` + `json.loads`, never sends an HTTP `tools` parameter. It should already
  be compatible with `tinyllama`/`phi3:mini` as-is. **Verify live, don't build** — a few core24 cells
  against those two models, $0, before assuming this.
- **`langgraph_react`** (`agent/app/langgraph_solver.py`) has no fallback today. `_build_llm` (line
  ~856-870) builds a raw `ChatOpenAI`; `solve()` (line ~908) calls
  `create_react_agent(llm, tools, prompt=system_prompt, pre_model_hook=...)` — all tool-call
  dispatch/`ToolMessage` construction lives inside `langgraph.prebuilt`, not this repo. A 400 from a
  non-tool-calling model is caught by a generic `except Exception` (line ~931), logged, and stored as
  `state.run_error` — recorded as a generic failure, not distinguished as "this model can't tool-call."
  No test simulates this today.
- Tools are declared via `_make_tools`/`_make_sandbox_tools` as LangChain `@tool`-decorated plain
  async functions — callable directly, without going through `bind_tools`.
- `model_tiers.py` tracks price-tier and reasoning-model-family only; nothing about tool-calling
  support or quantization level. **Quantization is a confirmed blind spot** in this project's telemetry
  more broadly (see the "DAG v2 Reconstruction Spec" artifact's §21 for detail): `ollama show <model>`
  reports both a quantization tag and a tool-calling capability flag right now, for free, and nothing
  captures either into any result JSON, trace, or run-id.

## The design

**Capability detection**: a `supports_native_tool_calling(model_name, provider) -> bool` predicate
(new, small — `model_tiers.py` or a new `model_capabilities.py`). Start with a hardcoded deny-list
seeded from what's already empirically known (`tinyllama`, `phi3:mini`) — zero new runtime risk. Only
add a cheap one-time preflight probe (send a trivial `tools=`-bearing request, catch the 400, cache
the result — mirroring this repo's existing `IDEA_TEST_PREFLIGHT_JSON`/search-provider preflight
pattern) if the roster grows past what's already known. While researching this, also record the
model's quantization (`ollama show <model>` reports it) alongside — closing the blind spot above at
the same time, since the plumbing to query per-model capability is already being built here.

**The shim itself**, in `langgraph_solver.py`: when the target model lacks native tool-calling, skip
`create_react_agent(llm, tools, ...)` entirely and run a manual loop that:

1. Prompts with a text description of the same tools already declared via `_make_tools`/
   `_make_sandbox_tools`, using a compact action format — reuse `sequential_react`'s already-proven
   `{"thought": ..., "action": ..., "args": {...}}` JSON scheme rather than inventing a new one.
2. Parses the response through Phase 1's constrained-decoding path (`json_repair.repair_malformed_json`
   with a `json_schema`, gated the same way — reuse, don't duplicate).
3. Dispatches by calling the existing `@tool`-decorated functions directly (they're plain async
   functions under the decorator).
4. Constructs synthetic `AIMessage(tool_calls=[...])`/`ToolMessage(...)` objects and appends them to
   the same message list `_msg_tool_calls`/`_extract_usage`/`_final_answer` already consume — so none
   of `langgraph_solver.py`'s existing post-processing, telemetry, or analysis code needs to change,
   only the inner model-call loop.
5. Every tool-calling-capable model keeps going through the existing `create_react_agent` path,
   completely unchanged — purely additive, gated on the capability check.

## Order of work (when picked up)

1. Live-verify `sequential_react` against `tinyllama`/`phi3:mini` first (cheap, might already be done).
2. `supports_native_tool_calling` predicate + tests (offline, no model calls needed to test the
   predicate itself).
3. The manual-loop shim in `langgraph_solver.py`, gated, with a new
   `agent/tests/langgraph_toolcall_emulation_test.py` mirroring `malformed_json_repair_test.py`'s
   scripted-IO style — assert: tool-calling-capable models are byte-identical (still go through
   `create_react_agent`); non-tool-calling models get the manual loop; the manual loop's messages are
   shape-compatible with `_msg_tool_calls`/`_extract_usage`/`_final_answer`.
4. Live smoke check on `tinyllama`/`phi3:mini` via `langgraph_react` (a few core24 cells, $0),
   confirming no more 400s and a real (even if weak) score instead of an infra failure.
5. Only then: a real 3-way comparison (`graph` vs `langgraph_react` vs `sequential_react`) across the
   FULL weak-model roster including the bottom tier, which is the actual payoff this shim exists for.

## Non-goals

- Not touching the native `graph` engine's action dispatch — that's Phase 1, already shipped.
- Not building a general-purpose LangChain tool-calling emulation library — this is scoped to what
  `langgraph_solver.py` already declares as tools, nothing more.
- Not attempting `sequential_react` changes unless live verification (step 1) proves it's actually
  broken, which the code review found no evidence of.
