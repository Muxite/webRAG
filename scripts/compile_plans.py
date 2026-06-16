#!/usr/bin/env python3
"""
Pre-author the compiled-plan cache — the "paid offline" step of the compiled-scaffold thesis.

The ``graph_compiled`` arm executes a static DAG plan authored by a *strong* model. This helper
authors those plans ONCE with the author model and writes them to ``compiled_plans/<hash>.json``,
so the benchmark's cheap-model runtime path is pure execution with zero planning cost (cache hit).
That cached artifact is the offline cost, accounted separately from runtime dollars.

It calls only the LLM (no web tools), so it is cheap and deterministic-ish. Re-running is a no-op
unless ``--force`` (plans are keyed by a hash of the mandate).

Usage::

    # author plans for the cross-shape suite with the reference model
    OPENROUTER_API_KEY=... LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 \\
      PYTHONPATH=services:services/agent ./.venv/bin/python scripts/compile_plans.py \\
      --tests 050,051,052,053,054 --author-model google/gemini-3.1-pro-preview

    # show the cached plan structure without authoring (no LLM, no network)
    PYTHONPATH=services:services/agent ./.venv/bin/python scripts/compile_plans.py \\
      --tests 050,051,052,053,054 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Mirror the runner's import roots so this works from a plain checkout.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "services", _ROOT / "services" / "agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.connector_config import ConnectorConfig  # noqa: E402
from agent.app.connector_llm import ConnectorLLM  # noqa: E402
from agent.app.connector_search import ConnectorSearch  # noqa: E402
from agent.app.connector_http import ConnectorHttp  # noqa: E402
from agent.app.connector_chroma import ConnectorChroma  # noqa: E402
from agent.app.agent_io import AgentIO  # noqa: E402
from agent.app.telemetry import TelemetrySession  # noqa: E402
from agent.app.testing.test_module import IdeaTestModule  # noqa: E402
from agent.app.testing import scaffold_compiler  # noqa: E402
from agent.app.testing.compiled_plan import plan_structure  # noqa: E402


def _load_modules() -> dict:
    tests_dir = _ROOT / "services" / "agent" / "app" / "idea_tests"
    by_id = {}
    for f in sorted(tests_dir.glob("test_*.py")):
        m = IdeaTestModule(f)
        by_id[m.metadata.get("test_id")] = m
    return by_id


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default="050,051,052,053,054", help="Comma-separated test_ids")
    ap.add_argument("--author-model", default=scaffold_compiler.DEFAULT_AUTHOR_MODEL)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--force", action="store_true", help="Re-author even if cached")
    ap.add_argument("--dry-run", action="store_true", help="Only print cached plan structure; no LLM")
    ap.add_argument("--show", action="store_true", help="Print the full authored plan JSON")
    args = ap.parse_args()

    ids = [t.strip() for t in args.tests.split(",") if t.strip()]
    by_id = _load_modules()

    io = None
    connectors = []
    if not args.dry_run:
        config = ConnectorConfig()
        cl, cs, ch, cc = ConnectorLLM(config), ConnectorSearch(config), ConnectorHttp(config), ConnectorChroma(config)
        connectors = [cl, cs, ch, cc]

    rc = 0
    try:
        for tid in ids:
            m = by_id.get(tid)
            if m is None:
                print(f"[{tid}] no such test"); rc = 1; continue
            mandate = m.get_task_statement()

            if args.dry_run:
                cached = scaffold_compiler.load_cached_plan(mandate)
                if cached is None:
                    print(f"[{tid}] NOT CACHED ({scaffold_compiler.cached_plan_path(mandate)})")
                else:
                    s = plan_structure(cached)
                    print(f"[{tid}] cached leaves={s['leaf_count']} edges={s['edge_count']} "
                          f"waves={s['wave_widths']} edges={s['edges']}")
                    if args.show:
                        print(json.dumps(cached, indent=2))
                continue

            tel = TelemetrySession(enabled=True, mandate=mandate, correlation_id=f"compile_{tid}", trace_path=None)
            io = AgentIO(connector_llm=connectors[0], connector_search=connectors[1],
                         connector_http=connectors[2], connector_chroma=connectors[3],
                         telemetry=tel, collection_name="scaffold_compiler")
            try:
                plan, info = await scaffold_compiler.compile_plan(
                    mandate, author_model=args.author_model, agent_io=io,
                    max_tokens=args.max_tokens, force=args.force,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{tid}] compile FAILED: {exc}"); rc = 1; continue
            s = info.get("structure") or plan_structure(plan)
            print(f"[{tid}] cache={info['cache']} author={args.author_model} "
                  f"leaves={s['leaf_count']} edges={s['edge_count']} waves={s['wave_widths']} "
                  f"-> {info['path']}")
            if args.show:
                print(json.dumps(plan, indent=2))
    finally:
        for c in connectors:
            for closer in ("aclose", "close"):
                fn = getattr(c, closer, None)
                if callable(fn):
                    try:
                        res = fn()
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception:  # noqa: BLE001
                        pass
                    break
    return rc


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
