#!/usr/bin/env python3
"""
Prewarm web fixtures for the cost-vs-accuracy benchmark — for $0 (no LLM tokens).

Strict-replay (``IDEA_TEST_FIXTURES=replay_strict``) makes every tooling rung and
model see *identical* evidence, which is what makes the cost comparison fair. But the
cache has to be populated first. Doing that with a full ``record`` run wastes LLM
tokens just to fill fixtures. This helper instead drives only the *tool* layer —
``AgentIO.visit`` for each URL named in a task mandate, plus one ``AgentIO.search`` of
the mandate — which records the same fixture keys the executors will later replay,
with no model calls.

Usage::

    # populate fixtures for the pilot tasks
    PYTHONPATH=services:services/agent SEARCH_API_KEY=... \\
      ./.venv/bin/python scripts/prewarm_fixtures.py --tests 048,049

    # see what it WOULD fetch, no network
    PYTHONPATH=services:services/agent ./.venv/bin/python scripts/prewarm_fixtures.py --tests 048,049 --dry-run

After prewarming, run the matrix with ``IDEA_TEST_FIXTURES=replay_strict`` so any
cache miss fails loudly instead of silently going live.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import List

# Mirror the runner's import roots so this works from a plain checkout.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "services", _ROOT / "services" / "agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_URL_RE = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')


def _mandate_urls(mandate: str) -> List[str]:
    """URLs named in a mandate (same extraction the naive_rag / minimal runners use)."""
    urls: List[str] = []
    for u in _URL_RE.findall(mandate):
        cleaned = u.rstrip('.,);]')
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _load_tasks(test_ids: List[str]):
    """Return [(test_id, mandate, [urls])] for the requested tasks."""
    from agent.app.testing.runner import discover_test_modules
    from agent.app.testing.test_module import IdeaTestModule

    by_id = {}
    for p in discover_test_modules():
        m = IdeaTestModule(p)
        by_id[str(m.metadata.get("test_id"))] = m
    out = []
    for tid in test_ids:
        m = by_id.get(tid)
        if m is None:
            print(f"  [skip] test id {tid} not found", file=sys.stderr)
            continue
        mandate = m.get_task_statement()
        out.append((tid, mandate, _mandate_urls(mandate)))
    return out


async def _prewarm(tasks, do_search: bool) -> int:
    from shared.connector_config import ConnectorConfig
    from agent.app.connector_llm import ConnectorLLM
    from agent.app.connector_search import ConnectorSearch
    from agent.app.connector_http import ConnectorHttp
    from agent.app.connector_chroma import ConnectorChroma
    from agent.app.agent_io import AgentIO
    from agent.app.telemetry import TelemetrySession

    config = ConnectorConfig()
    agent_io = AgentIO(
        connector_llm=ConnectorLLM(config),
        connector_search=ConnectorSearch(config),
        connector_http=ConnectorHttp(config),
        connector_chroma=ConnectorChroma(config),
        telemetry=TelemetrySession(enabled=False, mandate="", correlation_id="prewarm", trace_path=None),
        collection_name="prewarm",
    )

    misses = 0
    for tid, mandate, urls in tasks:
        print(f"[{tid}] {len(urls)} url(s)")
        for u in urls:
            try:
                content = await agent_io.visit(u, timeout_seconds=30)
                ok = bool(content)
                print(f"    visit {'ok ' if ok else 'EMPTY'} {u}")
                misses += 0 if ok else 1
            except Exception as exc:  # noqa: BLE001
                print(f"    visit FAIL {u}: {exc}")
                misses += 1
        if do_search:
            try:
                res = await agent_io.search(mandate, count=8, timeout_seconds=20) or []
                print(f"    search ok ({len(res)} results)")
            except Exception as exc:  # noqa: BLE001
                print(f"    search FAIL: {exc}")
                misses += 1
    return misses


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Prewarm benchmark web fixtures (no LLM cost)")
    ap.add_argument("--tests", required=True, help="Comma-separated test ids, e.g. 048,049")
    ap.add_argument("--no-search", action="store_true", help="Skip recording the mandate search")
    ap.add_argument("--dry-run", action="store_true", help="List planned fetches; no network")
    args = ap.parse_args(argv)

    test_ids = [t.strip() for t in args.tests.split(",") if t.strip()]
    tasks = _load_tasks(test_ids)
    if not tasks:
        print("No matching tasks.", file=sys.stderr)
        return 2

    if args.dry_run:
        print("DRY RUN — would fetch:")
        for tid, _mandate, urls in tasks:
            print(f"[{tid}]")
            for u in urls:
                print(f"    GET {u}")
            if not args.no_search:
                print(f"    SEARCH <mandate of {tid}>")
        return 0

    mode = (os.environ.get("IDEA_TEST_FIXTURES") or "").strip().lower()
    if mode not in ("record", "replay"):
        os.environ["IDEA_TEST_FIXTURES"] = "record"
        print("IDEA_TEST_FIXTURES not set to record/replay; forcing 'record' for prewarm.")

    misses = asyncio.run(_prewarm(tasks, do_search=not args.no_search))
    if misses:
        print(f"\nDONE with {misses} miss/failure(s) — fix before running replay_strict.", file=sys.stderr)
        return 1
    print("\nDONE — fixtures recorded. Run the matrix with IDEA_TEST_FIXTURES=replay_strict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
