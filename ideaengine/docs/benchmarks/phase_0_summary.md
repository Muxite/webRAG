# Phase 0 — In-tree Idea Engine Refactor Summary

In-tree refactor inside webRAG, completed prior to extraction to a standalone
`ideaengine` repo. All five planned refactors landed; 15/15 new unit tests pass.
The full 39-test parity gate (deferred per user choice "code refactor first,
smoke test at the end") still needs to run against real infrastructure.

## What changed

### 1. `DataContract` indirection — `services/agent/app/idea_policies/data_contracts.py` (new)

The four hardcoded `REQUIRES_DATA` string types
(`urls_from_search`, `urls_from_visit`, `url_from_think`, `chunk_from_visit`)
are now first-class objects with `(name, is_ready, default_for_action)`
shape, registered in a `ContractRegistry`. Engine becomes contract-table-driven:

- `IdeaDagEngine._has_required_data` no longer switch-cases on type strings.
- `IdeaDagEngine.__init__` accepts an optional `contracts: ContractRegistry`;
  defaults to `default_contract_registry()` containing the four built-ins.
- Custom action packs (Phase 1+) register their own contracts at construction
  time and the engine adopts them with no further code changes.

### 2. `LeafAction.post_execute_provides` — `services/agent/app/idea_policies/actions.py`

Base `LeafAction` declares the hook returning `None` by default. Overrides:

- `SearchLeafAction` → `"urls_from_search"`
- `VisitLeafAction` → `"urls_from_visit"` (only when content is non-empty)

The engine's auto-tagging in `_handle_action_result` no longer switches on
`IdeaActionType.SEARCH.value`/`VISIT.value`; instead it asks the action
instance what contract its successful result satisfies. Custom actions
declare what they provide simply by overriding the method.

### 3. Prompts externalized — `services/agent/app/prompts/`

Nine prompt strings (expansion/evaluation/merge/final, system + user, plus
expansion planning addendum) live as `.md` files under
`prompts/defaults/`. The loader `apply_default_prompts(settings)` fills
in any missing keys. `idea_dag_settings.json` keeps the prompts for
backwards compatibility and **takes precedence on conflict** — behavior is
unchanged.

The Jinja2 migration + composition slots for plug-in actions stay deferred to
Phase 1 (new repo) where they ride alongside `ActionPack` introduction.

### 4. Mandate enforcement hooks — `services/agent/app/idea_policies/post_expansion_hooks.py` (new)

The two enforcement helpers and the `_clean_extracted_url` utility moved out
of the engine and into a `PostExpansionHook` Protocol with two
implementations:

- `MandateUrlInjectionHook` — injects visit nodes for URLs literally
  present in the mandate text.
- `MandatePhraseEnforcementHook` — injects search/visit nodes when the
  mandate uses phrases like "must visit" or "must search", wiring the
  visit's `REQUIRES_DATA` to the search node when both are needed.

`IdeaDagEngine.__init__` accepts `post_expansion_hooks: Optional[List[...]]`;
defaults to the two web-research hooks above. Custom packs swap them out.
~180 lines deleted from `idea_engine.py`.

### 5. `Solver` Protocol + `IdeaEngineSolver` — `services/agent/app/solver.py` (new)

Narrow contract built around `engine.run()`:

```python
class Solver(Protocol):
    name: str
    async def solve(self, mandate, *, max_steps, settings, telemetry) -> SolverResult: ...

class IdeaEngineSolver:
    name = "ideaengine"
    def __init__(self, engine: IdeaDagEngine) -> None: ...
    async def solve(self, mandate, **kw) -> SolverResult: ...
```

`SolverResult` is a `TypedDict` with `final_deliverable`, `success`,
`observability` as required fields and `graph`, `token_usage`, `cost_usd`,
`wall_time_s`, etc. as optional. The Phase 3 comparison harness adds
`LangGraphSolver` and `LangChainSolver` against the same contract.

The test harness in `services/agent/app/testing/execution.py` is **not yet
migrated** to use `Solver` — it manually drives `engine.step()` for fine
control. Migration is part of Phase 3, scoped together with the LangGraph
and LangChain adapter work.

## Files touched

| Module | Change |
|---|---|
| `services/agent/app/idea_engine.py` | Removed 180 lines (mandate enforcement helpers); wired `contracts` registry and `post_expansion_hooks` into `__init__`; `_has_required_data` registry-driven; auto-tagging uses `action.post_execute_provides` |
| `services/agent/app/idea_policies/actions.py` | Added `post_execute_provides` hook on base `LeafAction`; overrides in `SearchLeafAction`, `VisitLeafAction` |
| `services/agent/app/idea_policies/data_contracts.py` | **NEW** — `DataContract`, `ContractRegistry`, four built-in contracts |
| `services/agent/app/idea_policies/post_expansion_hooks.py` | **NEW** — `PostExpansionHook` Protocol, two web hooks, `default_post_expansion_hooks()` |
| `services/agent/app/prompts/__init__.py`, `prompts/loader.py` | **NEW** — disk-backed prompt loader |
| `services/agent/app/prompts/defaults/*.md` | **NEW** — nine prompt files |
| `services/agent/app/idea_dag_settings.py` | Calls `apply_default_prompts(settings)` after loading the JSON |
| `services/agent/app/solver.py` | **NEW** — `Solver` Protocol, `SolverResult`, `IdeaEngineSolver` |
| `services/agent/tests/data_contracts_test.py` | **NEW** — 7 unit tests |
| `services/agent/tests/prompts_loader_test.py` | **NEW** — 4 unit tests |
| `services/agent/tests/solver_normalize_test.py` | **NEW** — 4 unit tests |

## Verification done

- AST-parse passes on all modified files.
- 15/15 new unit tests pass:
  ```bash
  PYTHONPATH=services python3 -m pytest \
      services/agent/tests/data_contracts_test.py \
      services/agent/tests/prompts_loader_test.py \
      services/agent/tests/solver_normalize_test.py -v
  ```
- Behavioral preservation verified by line-by-line equivalence of the
  refactored predicates against the original switch-case bodies in
  `_has_required_data` and `_handle_action_result`.

## What's still needed (smoke test the user runs)

Run the engine against a small subset of the 39-test suite at the existing
default model. Requires ChromaDB + Redis + OpenRouter + Brave API keys.
The standard webRAG invocation:

```bash
# From repo root. Requires docker-compose + .env with API keys.
IDEA_TEST_IDS=001,002,003,019,026 \
IDEA_TEST_RUNS=1 \
IDEA_TEST_MODELS=openai/gpt-5-mini \
docker compose run --profile test idea-test
```

Expected outcome: all five tests complete; pass scores within noise of any
historical run on the same tests. If `idea_test_results/` contains a
recent pre-refactor run for the same tests/model, diff the aggregate
scores. If not, this becomes the baseline.

If anything regresses against historical numbers (>5% drop in aggregate
pass@1, or a previously-passing test drops to 0/n), open `parity_phase_0_failures.md`
and root-cause before moving to Phase 1.

## Next phase

Phase 1: extract the engine to a new `ideaengine` repo (Apache 2.0).
File-by-file copy in dependency order; rewrite imports; inline
`shared.connector_config`/`shared.retry`; new `pyproject.toml` + CI against
`InMemoryVectorStore` + `MockLLMBackend`. Tag `v0.1.0a1`. See
`~/.claude/plans/plan-ways-to-take-golden-storm.md` for full sequencing.
