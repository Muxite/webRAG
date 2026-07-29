#!/usr/bin/env python3
"""Run the mix suite × local models on the host ollama and write JSONL traces for the
capability-floor study. Offline tasks (file/shell/memory) need no network; the web task
is skipped unless --web and a SearXNG url are given.

  ./.venv/bin/python badmodel-lab/localagent/run_suite.py \
      --models gemma2:2b,qwen2.5:1.5b --reps 4 --out badmodel-lab/results/agent_traces.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # badmodel-lab/

from localagent.agent_tasks.suite import build_suite            # noqa: E402
from localagent.catalog import build_default_registry, build_default_tools  # noqa: E402
from localagent.llm import OllamaLLM                            # noqa: E402
from localagent.runner import metrics_for, run_task_once, write_trace  # noqa: E402
from localagent.tools.memory import FileMemoryStore             # noqa: E402
from localagent.tools.web import (WebReadTool, httpx_fetch_fn,  # noqa: E402
                                  observation_extract_fn, searxng_search_fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of ollama tags")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--tasks", default=None, help="comma list of task ids (default: all offline)")
    ap.add_argument("--out", default="badmodel-lab/results/agent_traces.jsonl")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--searxng", default=None, help="SearXNG base url to enable the web task")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-calls", type=int, default=25)
    ap.add_argument("--finalize-n", type=int, default=3, help="best-of-N candidates at finish")
    ap.add_argument("--workroot", default="/tmp/localagent_runs")
    args = ap.parse_args()

    reg = build_default_registry()
    suite = build_suite()
    want = set(args.tasks.split(",")) if args.tasks else None
    web_tool = None
    if args.searxng:
        web_tool = WebReadTool(searxng_search_fn(args.searxng), httpx_fetch_fn(), observation_extract_fn())
    tasks = [t for t in suite if (want is None or t.id in want) and (not t.needs_web or web_tool)]

    out = Path(args.out)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    workroot = Path(args.workroot)
    print(f"models={models} tasks={[t.id for t in tasks]} reps={args.reps} -> {out}")

    for model in models:
        llm = OllamaLLM(model=model, base_url=args.base_url)
        tools = build_default_tools(web_tool=web_tool)
        for task in tasks:
            for rep in range(1, args.reps + 1):
                wd = workroot / model.replace(":", "-") / task.id / str(rep)
                mem = FileMemoryStore(wd.parent / "mem.jsonl", identity=f"{model}:{task.id}:{rep}")
                if mem.path.exists():
                    mem.path.unlink()  # fresh memory per rep
                try:
                    res, ctx, latency = run_task_once(task, reg, tools, llm, workdir=wd,
                                                      memory=mem, max_steps=args.max_steps,
                                                      finalize_n=args.finalize_n)
                    res.state.budget.max_calls = args.max_calls
                    m = metrics_for(task, res, ctx, reg, model=model, rep=rep, latency_s=latency)
                except Exception as exc:  # noqa: BLE001 — one bad run must not kill the sweep
                    m = {"task_id": task.id, "model": model, "rep": rep, "success": False,
                         "error": str(exc)[:200], "finished": False, "n_steps": 0,
                         "latency_s": None, "containment_ok": True}
                write_trace(out, m)
                flag = "ok " if m.get("success") else "MISS"
                print(f"  [{flag}] {model:<14} {task.id:<16} r{rep} "
                      f"steps={m.get('n_steps')} {m.get('latency_s')}s "
                      f"ans={str(m.get('final_answer', m.get('error','')))[:60]!r}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
