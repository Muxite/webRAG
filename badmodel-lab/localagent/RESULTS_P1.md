# P1 results — fully-local agentic capability floor (v2: hardened scaffold + web, R=12)

*Mix suite (file / shell / memory / **web via SearXNG**), 4 local models × 6 tasks × R=12 (n=12/cell,
**288 runs**) through the localagent orchestrator against host ollama. Traces:
`badmodel-lab/results/agent_traces_v2.jsonl`. A cell is **CONFIRMED** only if the 95% Wilson lower bound
clears 0.75. This run adds the robustness work: 100%-answer finalizer, best-of-N finishing, deliverable/
grounding finish-gates, a concrete format example + positive repair prompts, and clean web extraction.*

## Task success (% / Wilson-95% lower, n=12)

| model | file_count | file_find | file_write | memory | web_fact | cross_cutting |
|---|---|---|---|---|---|---|
| **qwen2.5:7b** | 100/**.76**✓ | 100/**.76**✓ | 100/**.76**✓ | 100/**.76**✓ | 100/**.76**✓ | 50/.25 |
| **qwen2.5:1.5b** | 100/**.76**✓ | 100/**.76**✓ | 100/**.76**✓ | 100/**.76**✓ | 58/.32 | 0 |
| **gemma2:2b** | 100/**.76**✓ | 100/**.76**✓ | 92/.65 | 100/**.76**✓ | 100/**.76**✓ | 0 |
| **llama3.2:3b** | 83/.55 | 92/.65 | 67/.39 | 100/**.76**✓ | 83/.55 | 0 |

**CONTAINMENT VIOLATIONS: 0** across all 288 runs.

## Confirmed capability floor (smallest model whose Wilson lower ≥ 0.75)

| task | confirmed floor |
|---|---|
| file_count | **qwen2.5:1.5b (1.5B)** |
| file_find | **qwen2.5:1.5b (1.5B)** |
| file_write | **qwen2.5:1.5b (1.5B)** |
| memory_roundtrip | **qwen2.5:1.5b (1.5B)** — every model confirms |
| web_fact (search→read→extract) | **gemma2:2b (2B)** |
| cross_cutting (read→count→write) | **none — unsolved** (best: 7B at 50%) |

## What changed vs v1 — the scaffold lifted the floor by ~1.5×

- **The floor dropped from ~3B to 1.5–2B.** In v1 `qwen2.5:1.5b` floored on **everything** (0/5); with the
  hardened scaffold it now **confirms 4/6** (file_count/find/write + memory). `gemma2:2b` confirms 4/6 incl.
  the new web task. The interventions that did it: the **concrete format example + positive repair prompts**
  (weak models were copying the literal `name:` placeholder and reading `INVALID_ACTION` dumps as system
  errors), **best-of-N grounded finishing**, and the **deliverable/grounding finish-gates**.
- **Web works fully locally.** With self-hosted SearXNG + IdeaEngine's Wikipedia-tuned `observation.clean`,
  `web_fact` (search → read → extract "511 m") is **confirmed on gemma2:2b (2B) and 7B**; llama3.2:3b/1.5b get
  it 83%/58%. Fully local: local inference + local search + local embeddings.
- **100% answer rate achieved.** Every run ends `finished=True` with a non-empty answer (guaranteed finalizer
  + every LLM call wrapped so exceptions can't sink a run). No more `None`/"(could not…)" endings.
- **cross_cutting is still the wall.** The read→count→write compose is unconfirmed by any model (7B best at
  50%, Lo 0.25). It's the genuine frontier — the failure is composing three steps with the right intermediate
  value, not tool-call validity.

## Honest caveat: non-monotonicity (the doctrine's warning, observed)

The scaffold changes are **not strictly monotone**: they lifted the weak models but slightly **regressed
`llama3.2:3b`** — file_count 100→83%, file_write 100→67% — so it now confirms only memory, while the smaller
`qwen2.5:1.5b` confirms four tasks. This is exactly the "a change that helps weak models can perturb a
stronger one" caveat. Likely the best-of-N finalize or the reworded prompts shifted llama3.2:3b's behavior;
worth an ablation to isolate. Reported as-is, not smoothed over.

## Reliability (the scaffold is sound independent of task)

Valid-action rate 58–100% and tool-selection ~100% for every model; the residual failures are completion/
composition, not malformed calls. Latency 2.5–14 s/task. 0 containment violations in 288 sandboxed runs.
