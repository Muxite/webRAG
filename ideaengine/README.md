# ideaengine

> Graph-of-Thought DAG executor for LLM research tasks.

A Python library that turns a research mandate into a structured deliverable
by planning, scoring, acting (search / visit / think / save), and merging
across a directed acyclic graph of LLM-driven thoughts.

Extracted from [webRAG](https://github.com/) where it powers the agent's
research pipeline. Provider-agnostic (any OpenAI-compatible LLM, Anthropic,
OpenRouter), works with ChromaDB for memory and Brave for search out of the
box.

## Status

`v0.1.0-alpha` — extracted, importable, unit-tested. The flat namespace is a
deliberate v0.1 choice; v0.2 will introduce a proper sub-package layout
(`ideaengine.core`, `ideaengine.policies`, `ideaengine.backends`) once the
public API has settled in.

## Install

```bash
pip install ideaengine                 # core
pip install "ideaengine[chroma]"       # + ChromaDB vector store
pip install "ideaengine[anthropic]"    # + Anthropic Messages backend
pip install "ideaengine[browser]"      # + headless Chrome fallback for visits
pip install "ideaengine[testing]"      # + pytest harness
pip install "ideaengine[all]"          # everything except testing/benchmarks
```

## Hello world

```python
import asyncio
from ideaengine import IdeaDagEngine, AgentIO, load_idea_dag_settings

async def main():
    settings = load_idea_dag_settings()
    io = AgentIO(
        connector_llm=...,      # your LLM connector (OpenAI-compatible)
        connector_search=...,   # your search connector (Brave)
        connector_http=...,     # your HTTP connector
        connector_chroma=...,   # your ChromaDB connector
    )
    engine = IdeaDagEngine(io=io, settings=settings)
    result = await engine.run("What is the capital of Australia?")
    print(result["final_deliverable"])

asyncio.run(main())
```

For the no-deps smoke construction (no real LLM calls):

```python
from ideaengine import IdeaDagEngine, default_contract_registry
from ideaengine.idea_policies.data_contracts import URLS_FROM_SEARCH
# Inspect the engine's wiring without dispatching anything:
print(sorted(default_contract_registry().names()))
# ['chunk_from_visit', 'url_from_think', 'urls_from_search', 'urls_from_visit']
```

## What's in the box

### Core engine

| Module | Purpose |
|---|---|
| `IdeaDagEngine` | Step-driven controller that owns the DAG and runs the planner / evaluator / actions / merger loop |
| `IdeaDag` / `IdeaNode` | The graph data model |
| `MemoryManager` | ChromaDB-backed memory with sentence-aware chunking |
| `GoTOperations` | Graph-of-Thought mechanics: embedding, dedup, dynamic beam, pruning, backtrack (gated), improve (gated) |
| `build_final_payload` | Final synthesis call that turns the executed graph into a deliverable |
| `Solver`, `IdeaEngineSolver` | Solver protocol for comparison harnesses (LangGraph / LangChain adapters slot here) |

### Policies (pluggable)

`ideaengine.idea_policies.*`: `expansion`, `evaluation`, `selection`,
`decomposition`, `merge`, `actions`, plus the data-contract registry
(`data_contracts.py`) and the post-expansion hook protocol
(`post_expansion_hooks.py`) for mandate-enforcement extension points.

### Action packs

Built-in: `search`, `visit`, `think`, `save`, `merge` (the web-research pack).

Eleven additional opt-in plugins under `ideaengine.idea_policies.extra_actions`:

| | Action | Network? |
|---|---|---|
| 1 | `wikipedia_summary` | yes |
| 2 | `arxiv_search` | yes |
| 3 | `github_repo_info` | yes |
| 4 | `hacker_news_top` | yes |
| 5 | `pypi_package_info` | yes |
| 6 | `open_meteo_weather` | yes |
| 7 | `url_metadata` | yes |
| 8 | `regex_extract` | no |
| 9 | `json_path` | no |
| 10 | `unit_convert` | no |
| 11 | `datetime_now` | no |

All seven network actions use free public APIs — no keys required. See
[`docs/PLUGINS.md`](docs/PLUGINS.md) for how to add your own.

### Backends

| | Default impl | Provider-agnostic? |
|---|---|---|
| LLM | `OpenAICompatibleBackend`, `AnthropicMessagesBackend`, `OpenRouterBackend` | yes |
| Search | `BraveSearchBackend` | no (Brave-only; ABC pending v0.2) |
| Vector store | `ChromaConnector` | no (Chroma-only; ABC pending v0.2) |
| HTTP | `ConnectorHttp` + optional `ConnectorBrowser` fallback | yes |

### Prompts

Externalized to disk under `ideaengine/prompts/defaults/` (one `.md` file
per template). The settings.json copies still win for back-compat. Override
with `EngineConfig(prompts={...})` or by providing a custom directory to
`load_default_prompts(directory=...)`.

## Documentation

- [`docs/IDEA_ENGINE.md`](docs/IDEA_ENGINE.md) — Architecture deep-dive (19 sections)
- [`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md) — Compact reference
- [`docs/IDEA_ENGINE_FEATURES.md`](docs/IDEA_ENGINE_FEATURES.md) — Strategic feature backlog
- [`docs/PLUGINS.md`](docs/PLUGINS.md) — Plugin authoring guide + plug-in catalog
- [`docs/benchmarks/`](docs/benchmarks/) — Benchmark methodology + findings

## Testing

```bash
pip install "ideaengine[testing]"
pytest tests/
```

71+ unit tests, no network or chroma required (importlib-bypass pattern
keeps the package's transitive deps out of the test surface).

## License

Apache License 2.0. See [LICENSE](LICENSE).
