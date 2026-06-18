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

URL sources (compose freely):
  * ``--tests 048,049``     — records each mandate's named URLs + the mandate search.
  * ``--from-run <run_id>`` — harvests the pages a PRIOR run actually visited from its result
    JSON. This is the only way to prewarm **search-driven** tasks (050-054, 046/047), which name
    no URLs in the mandate: do one permissive discovery pass, then prewarm its discovered pages
    so the scored matrix can run in strict replay with identical, frozen evidence for every model.
  * ``--urls a,b`` / ``--urls-file f`` — explicit page URLs.

Usage::

    # mandate-named URLs (URL-bearing tasks)
    PYTHONPATH=services:services/agent SEARCH_API_KEY=... \\
      ./.venv/bin/python scripts/prewarm_fixtures.py --tests 048,049

    # freeze a discovery run's discovered pages (search-driven tasks)
    PYTHONPATH=services:services/agent SEARCH_API_KEY=... \\
      ./.venv/bin/python scripts/prewarm_fixtures.py --from-run xshape_20260615_164736

    # see what it WOULD fetch, no network
    PYTHONPATH=services:services/agent ./.venv/bin/python scripts/prewarm_fixtures.py --from-run <id> --dry-run

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


def _clean_url(raw: str) -> str:
    """Trim trailing punctuation a URL picks up when scanned out of free text / markdown.

    Handles sentence tails ("...wiki/X." / "...wiki/X):") and the closing paren of a markdown
    ``[text](url)`` wrapper, WITHOUT eating a paren that is part of the path itself
    (``.../Beloved_(novel)`` stays intact — that paren is balanced)."""
    u = str(raw or "").strip()
    trail = ".,;:'\"<>"
    while u and u[-1] in trail:
        u = u[:-1]
    while u.endswith(")") and u.count("(") < u.count(")"):
        u = u[:-1]
        while u and u[-1] in trail:
            u = u[:-1]
    return u


def _mandate_urls(mandate: str) -> List[str]:
    """URLs named in a mandate (same extraction the naive_rag / minimal runners use)."""
    urls: List[str] = []
    for u in _URL_RE.findall(mandate):
        cleaned = _clean_url(u)
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


# Hosts that are tool *endpoints*, not content pages — never prewarm these as visits.
_ENDPOINT_HOSTS = ("api.search.brave.com", "openrouter.ai", "googleapis.com")


def _is_page_url(url: str) -> bool:
    """True for a real content page worth visiting (not a search/LLM API endpoint)."""
    low = url.lower()
    return low.startswith(("http://", "https://")) and not any(h in low for h in _ENDPOINT_HOSTS)


def _urls_from_result(d: dict) -> List[str]:
    """Page URLs the agent actually engaged in one result file.

    Search-driven tasks (050-054, 046/047) name no mandate URLs — the pages are discovered at
    runtime — so the *only* way to prewarm them for strict replay is to harvest what a prior run
    visited. Prefers the structured ``execution.output.sources`` (added by the evidence line),
    falling back to a URL scan of the deliverable / action_summary so it also works on older
    result files that predate ``sources``."""
    out: List[str] = []
    seen = set()

    def _add(raw: str) -> None:
        u = _clean_url(raw)
        if u and _is_page_url(u) and u not in seen:
            seen.add(u)
            out.append(u)

    output = ((d.get("execution") or {}).get("output")) or {}
    for src in (output.get("sources") or []):
        if isinstance(src, dict):
            _add(src.get("url", ""))
    for field in ("final_deliverable", "action_summary"):
        for m in _URL_RE.findall(str(output.get(field) or "")):
            _add(m)
    return out


def _harvest_run_urls(results_dir: Path, run_id: str) -> List[str]:
    """Union of page URLs across every result file with the given run-id prefix."""
    import json
    urls: List[str] = []
    seen = set()
    prefix = f"{run_id}_"
    for p in sorted(results_dir.glob(f"{run_id}_*.json")):
        if p.name.endswith("_summary.json") or "_report_" in p.name or not p.name.startswith(prefix):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for u in _urls_from_result(d):
            if u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def _results_dir() -> Path:
    for cand in (_ROOT / "services" / "agent" / "idea_test_results", _ROOT / "agent" / "idea_test_results"):
        if cand.is_dir():
            return cand
    return _ROOT / "services" / "agent" / "idea_test_results"


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


async def _prewarm(tasks) -> int:
    """Visit every page (and optionally record the mandate search) for each task.

    ``tasks`` is a list of ``(label, mandate, urls, do_search)``; ``do_search`` records the
    mandate search only for ``--tests`` tasks (harvested/explicit URL sets carry no mandate)."""
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
    for label, mandate, urls, do_search in tasks:
        print(f"[{label}] {len(urls)} url(s)")
        for u in urls:
            try:
                content = await agent_io.visit(u, timeout_seconds=30)
                ok = bool(content)
                print(f"    visit {'ok ' if ok else 'EMPTY'} {u}")
                misses += 0 if ok else 1
            except Exception as exc:  # noqa: BLE001
                print(f"    visit FAIL {u}: {exc}")
                misses += 1
        if do_search and mandate:
            try:
                res = await agent_io.search(mandate, count=8, timeout_seconds=20) or []
                print(f"    search ok ({len(res)} results)")
            except Exception as exc:  # noqa: BLE001
                print(f"    search FAIL: {exc}")
                misses += 1
    return misses


def _build_tasks(args) -> List[tuple]:
    """Assemble the (label, mandate, urls, do_search) prewarm task list from all sources.

    Sources compose: ``--tests`` (mandate URLs + mandate search), ``--from-run`` (URLs harvested
    from a prior run's results — the only way to cover search-driven tasks), and explicit
    ``--urls`` / ``--urls-file``."""
    tasks: List[tuple] = []
    if args.tests:
        test_ids = [t.strip() for t in args.tests.split(",") if t.strip()]
        for tid, mandate, urls in _load_tasks(test_ids):
            tasks.append((tid, mandate, urls, not args.no_search))
    for run_id in [r.strip() for r in args.from_run.split(",") if r.strip()]:
        harvested = _harvest_run_urls(_results_dir(), run_id)
        if harvested:
            tasks.append((f"from-run:{run_id}", "", harvested, False))
        else:
            print(f"  [warn] no page URLs harvested from run-id {run_id!r}", file=sys.stderr)
    explicit: List[str] = []
    if args.urls:
        explicit += [_clean_url(u) for u in args.urls.split(",") if u.strip()]
    if args.urls_file:
        explicit += [_clean_url(ln) for ln in Path(args.urls_file).read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#")]
    explicit = [u for u in dict.fromkeys(explicit) if _is_page_url(u)]
    if explicit:
        tasks.append(("urls", "", explicit, False))
    return tasks


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Prewarm benchmark web fixtures (no LLM cost)")
    ap.add_argument("--tests", default="", help="Comma-separated test ids, e.g. 048,049 (records mandate URLs + the mandate search)")
    ap.add_argument("--from-run", dest="from_run", default="",
                    help="Harvest actually-visited page URLs from prior run result JSON (comma-separated "
                         "run-id prefixes). The only way to prewarm search-driven tasks (050-054, 046/047) "
                         "for strict replay.")
    ap.add_argument("--urls", default="", help="Comma-separated explicit page URLs to record")
    ap.add_argument("--urls-file", dest="urls_file", default="", help="File with one page URL per line (# comments allowed)")
    ap.add_argument("--no-search", action="store_true", help="Skip recording the mandate search (for --tests)")
    ap.add_argument("--dry-run", action="store_true", help="List planned fetches; no network")
    args = ap.parse_args(argv)

    if not (args.tests or args.from_run or args.urls or args.urls_file):
        print("Nothing to do: pass at least one of --tests / --from-run / --urls / --urls-file.", file=sys.stderr)
        return 2

    tasks = _build_tasks(args)
    if not tasks or all(not t[2] for t in tasks):
        print("No URLs to prewarm from the given sources.", file=sys.stderr)
        return 2

    if args.dry_run:
        print("DRY RUN — would fetch:")
        for label, mandate, urls, do_search in tasks:
            print(f"[{label}] ({len(urls)} url(s))")
            for u in urls:
                print(f"    GET {u}")
            if do_search and mandate:
                print(f"    SEARCH <mandate of {label}>")
        return 0

    mode = (os.environ.get("IDEA_TEST_FIXTURES") or "").strip().lower()
    if mode not in ("record", "replay", "replay_strict"):
        os.environ["IDEA_TEST_FIXTURES"] = "record"
        print("IDEA_TEST_FIXTURES not set to record/replay[_strict]; forcing 'record' for prewarm.")
    elif mode == "replay_strict":
        # Acts as a $0 cache-completeness CHECK: every harvested page must already be cached,
        # else it's a miss — surface it before spending on the strict scored pass.
        print("replay_strict: verifying every page is already cached (misses = gaps to fix).")

    misses = asyncio.run(_prewarm(tasks))
    if misses:
        print(f"\nDONE with {misses} miss/failure(s) — fix before running replay_strict.", file=sys.stderr)
        return 1
    print("\nDONE — fixtures recorded. Run the matrix with IDEA_TEST_FIXTURES=replay_strict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
