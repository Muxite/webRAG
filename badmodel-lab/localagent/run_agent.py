#!/usr/bin/env python3
"""CLI entrypoint: run ONE task through the local-agent orchestrator against a real
local model. $0 to import; only touches the model when actually invoked (P1 step).

    ./.venv/bin/python badmodel-lab/agent/run_agent.py \
        --model gemma2:2b --goal "list the files here and tell me how many there are" \
        --workdir /tmp/agent_work --memory /tmp/agent_mem.jsonl --identity me

Streams cosmetic narration to stdout as it goes (see narrator.py); prints the truthful
step trace at the end. NOTE: this runs the tools on the HOST until the sandbox image is
built (P1) — point --workdir at a throwaway dir. The tools already confine to --workdir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # badmodel-lab/ on path

from localagent.catalog import build_default_registry, build_default_tools   # noqa: E402
from localagent.llm import OllamaLLM                                          # noqa: E402
from localagent.loop import run_task                                         # noqa: E402
from localagent.narrator import Narrator, StdoutSink                         # noqa: E402
from localagent.state import AgentState                                      # noqa: E402
from localagent.tools.base import ToolContext                                # noqa: E402
from localagent.tools.memory import FileMemoryStore                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ollama tag, e.g. gemma2:2b")
    ap.add_argument("--goal", required=True, help="the free-form task for the agent")
    ap.add_argument("--workdir", default="/tmp/agent_work", help="sandbox scratch dir (confined)")
    ap.add_argument("--memory", default="/tmp/agent_mem.jsonl", help="durable memory JSONL")
    ap.add_argument("--identity", default="default", help="stable memory identity key")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base url (default $MODEL_API_URL)")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-calls", type=int, default=40)
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(workdir=wd, memory=FileMemoryStore(Path(args.memory), identity=args.identity))
    state = AgentState(task_goal=args.goal)
    state.budget.max_calls = args.max_calls
    llm = OllamaLLM(model=args.model, base_url=args.base_url)
    narr = Narrator(sinks=[StdoutSink()], filler_every=3)

    res = run_task(state, build_default_registry(), build_default_tools(), llm, ctx,
                   narrator=narr, max_steps=args.max_steps)

    print("\n" + "=" * 72)
    print(f"FINAL: {res.final_answer}")
    print(f"steps={len(res.steps)} calls={state.budget.used_calls} finished={state.finished}")
    print("--- truth trace ---")
    for i, s in enumerate(res.steps, 1):
        rep = f" (repairs={s.repairs})" if s.repairs else ""
        print(f"  {i:2}. {s.action or '?':<12} ok={s.ok!s:<5} {s.summary}{rep}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
