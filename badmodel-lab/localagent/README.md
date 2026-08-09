# localagent — local-model agent orchestrator (badmodel-lab)

A **separate layer** that treats a weak local LLM as a *noisy semantic controller* and wraps every fragile
interface in deterministic code. It **reuses IdeaEngine primitives by import** (connectors, local MiniLM
embeddings, HTML cleaning, `json_telemetry.schema_check`) and makes **zero changes to `idea_engine.py` or the
production path**. Named `localagent` (not `agent`) to avoid colliding with the repo's `agent`
package. Full design: `/home/muk/.claude/plans/plan-future-work-*.md`.

## The loop (doctrine)

```
OBSERVE (typed state) → Router (pick 1 of ≤8 state-legal actions) → Slot-filler (typed slots, entity ids)
   → Validator (deterministic, typed errors) → Repairer (≤2, one field) → execute tool → summarize → update state
```

The model only makes small, high-accuracy choices (a closed action enum, a few typed slots); it never authors
JSON, paths, or UUIDs. `RunResult` is the truth; the `narrator` stream is cosmetic.

## Modules

| Area | Files |
|---|---|
| Core loop | `state.py` (typed blackboard) · `actions.py` (action IR + group-gated registry) · `ir.py` (parse/route/validate/repair) · `loop.py` · `catalog.py` |
| Tools | `tools/files.py` · `tools/shell.py` (read-only allow-listed DSL) · `tools/web.py` (read-only; SearXNG+fetch adapters) · `tools/memory.py` (`FileMemoryStore` + `VectorMemoryStore`/`ChromaBackend`) |
| Verify / tasks | `verify.py` (evidence-first, schema, fs assertions) · `agent_tasks/suite.py` (the mix suite + validators) |
| Run / measure | `llm.py` (`ScriptedLLM`/`OllamaLLM`) · `runner.py` (per-run metrics + traces) · `analyze_agent.py` (Wilson-CI floor + latency-ratio) · `narrator.py` |
| Sandbox | `docker/` (hardened per-run container, `run_sandbox.sh`, `containment_check.sh`) |

## Status

- **P0a + P0b: DONE, unit-tested (no LLM, no sandbox).** `pytest badmodel-lab/localagent/tests/ -q` → **31 passed**.
  Covers the IR (parse/route/validate/typed-repair, group gating, core-always-legal), the loop (end-to-end
  file+memory, cross-session recall, slot & route repair, narration, budget-stop), the shell/web/vector-memory
  tools, the verifiers, the **whole mix suite solved by scripted models through the real tools**, and the
  analyzer math (Wilson floor, latency baseline).
- **Sandbox** image + hardened `run_sandbox.sh` + `containment_check.sh` written (build/verify in P1).
- **Wired, live-validated in P1:** `OllamaLLM`, the SearXNG/`observation.clean` web adapters, and the
  `ChromaBackend` vector memory. Their logic is exercised via fakes/`InMemoryBackend`; live services attach in P1.

## Run

- Tests: `./.venv/bin/python -m pytest badmodel-lab/localagent/tests/ -q`
- Live (P1): `./.venv/bin/python badmodel-lab/localagent/run_agent.py --model gemma2:2b --goal "…" --workdir /tmp/agent_work`
- Sandboxed (P1): `docker build -f badmodel-lab/localagent/docker/Dockerfile -t localagent-sandbox badmodel-lab && badmodel-lab/localagent/docker/run_sandbox.sh gemma2:2b "…"`
