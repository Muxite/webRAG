#!/usr/bin/env python3
"""Build a frozen search corpus for deterministic $0 replay.

Two sources, so the first corpus costs nothing:

* **Harvest (free).** Every stored cell under ``agent/idea_test_results`` already carries the
  pages the agent fetched (``execution.output.pages``, the ``store_page`` shape) and the URLs it
  cited (``execution.output.extractions[].source_url``). Hundreds of cells exist; harvesting turns
  spend that already happened into a reusable evidence universe.
* **Live top-up (paid).** ``scripts/prewarm_fixtures.py`` already drives search and visits through
  ``AgentIO``; run it first to record pages this corpus lacks, then harvest again.

Output is one ``documents.jsonl`` under the corpus directory, loaded directly by
``agent.app.connector_search_corpus.load_documents`` -- no translation step between builder and
backend, so the two cannot drift apart.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/build_corpus.py \
        --results-dir agent/idea_test_results --out agent/idea_test_results/corpus/core_long24
"""
import argparse
import asyncio
import json
import os
import re
import logging
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

#: Canonical per-cell result files. ``*_summary.json`` reflects only the last cell of a
#: multi-invocation run and ``*.jsonl`` are traces; counting either inflates any tally over the
#: results directory (it once produced a throughput figure 2.1x too high).
RESULT_GLOB = "*_r[0-9]*.json"
#: Page body kept per document. Enough to rank on, bounded so a corpus stays reviewable.
MAX_TEXT_CHARS = 20000
#: Opening characters used to decide two pages are the same content. Long enough that distinct
#: articles diverge, short enough that different truncations of one page still agree.
CONTENT_KEY_CHARS = 400


_logger = logging.getLogger(__name__)

def _cell_output(payload: Any) -> Dict[str, Any]:
    """The ``execution.output`` dict of a result payload, or ``{}`` when absent."""
    if not isinstance(payload, dict):
        return {}
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        return {}
    output = execution.get("output")
    return output if isinstance(output, dict) else {}


def iter_result_files(results_dir: str) -> Iterable[Path]:
    """Canonical per-cell result JSONs under ``results_dir``, sorted for deterministic output."""
    root = Path(results_dir)
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob(RESULT_GLOB)
                  if path.is_file() and not path.name.endswith("_summary.json"))


def _content_key(text: str) -> str:
    """Identity of a page by its opening body, whitespace-normalised.

    URL canonicalisation is not enough. ``canonicalize_url`` deliberately keeps the query string,
    so the same article reached bare and with a tracking parameter stays two URLs -- measured on
    the real corpus, where one USGS release appeared twice and burned two of three result slots.
    Two documents that begin with identical prose are one page however they were reached.
    """
    return " ".join(str(text or "").split())[:CONTENT_KEY_CHARS]


def _better_url(left: str, right: str) -> str:
    """The cleaner of two URLs for the same content: fewer parameters, then shorter."""
    return min((left, right), key=lambda url: (url.count("?") + url.count("&"), len(url)))


def harvest_documents(results_dir: str, max_text_chars: int = MAX_TEXT_CHARS) -> List[Dict[str, str]]:
    """Collect one document per distinct page from every stored cell.

    Identity is applied twice, because neither key alone is sufficient:

    * **Canonical URL** (``evidence_store.canonicalize_url``) folds fragments and host casing.
    * **Content prefix** folds the same page reached through different query strings.

    A page seen in several cells keeps the **longest** stored text -- cells truncate to different
    budgets, and the most complete copy is the one worth freezing -- and the cleanest URL.

    A corrupt or unreadable cell is skipped rather than fatal: an unattended build cannot stop to
    ask what to do about one bad file.

    :param results_dir: directory of per-cell result JSONs.
    :param max_text_chars: per-document body cap.
    :returns: documents in first-seen order, each ``{url, title, description, text}``.
    """
    from agent.app.evidence_store import canonicalize_url

    by_url: Dict[str, Dict[str, str]] = {}
    for path in iter_result_files(results_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        pages = _cell_output(payload).get("pages")
        if not isinstance(pages, list):
            continue
        for page in pages:
            if not isinstance(page, dict):
                continue
            url = canonicalize_url(page.get("url"))
            if not url:
                continue
            text = str(page.get("text") or "")[:max_text_chars]
            existing = by_url.get(url)
            if existing is None:
                by_url[url] = {"url": url, "title": str(page.get("title") or "").strip(),
                               "description": text[:300], "text": text}
            elif len(text) > len(existing["text"]):
                existing.update({"text": text, "description": text[:300]})

    by_content: Dict[str, Dict[str, str]] = {}
    for doc in by_url.values():
        key = _content_key(doc["text"])
        if not key:
            # No body to compare on; keep it, keyed uniquely so it cannot swallow others.
            by_content[f"\x00{doc['url']}"] = doc
            continue
        existing = by_content.get(key)
        if existing is None:
            by_content[key] = doc
        else:
            existing["url"] = _better_url(existing["url"], doc["url"])
            if len(doc["text"]) > len(existing["text"]):
                existing.update({"text": doc["text"], "description": doc["text"][:300]})
    return list(by_content.values())


def dedupe_documents(raw_docs: Iterable[Dict[str, Any]],
                     max_text_chars: int = MAX_TEXT_CHARS) -> List[Dict[str, str]]:
    """Collapse a flat document list to one entry per distinct page.

    The same two-stage identity :func:`harvest_documents` uses -- canonical URL, then content
    prefix -- generalised so it can merge documents from *any* source (harvested, live-fetched,
    or a mix of both) rather than only the stored-cell shape. Unlike harvested pages, a live
    document usually carries a real search-result ``description``; that is kept rather than
    overwritten with a text prefix, and only backfilled with one when the source gave none.

    :param raw_docs: dicts with ``url``, and optionally ``title``/``description``/``text``.
    :param max_text_chars: per-document body cap.
    :returns: documents in first-seen order, each ``{url, title, description, text}``.
    """
    from agent.app.evidence_store import canonicalize_url

    by_url: Dict[str, Dict[str, str]] = {}
    for raw in raw_docs:
        url = canonicalize_url(raw.get("url"))
        if not url:
            continue
        text = str(raw.get("text") or "")[:max_text_chars]
        description = str(raw.get("description") or "").strip() or text[:300]
        doc = {"url": url, "title": str(raw.get("title") or "").strip(),
               "description": description, "text": text}
        existing = by_url.get(url)
        if existing is None or len(doc["text"]) > len(existing["text"]):
            by_url[url] = doc

    by_content: Dict[str, Dict[str, str]] = {}
    for doc in by_url.values():
        key = _content_key(doc["text"])
        if not key:
            by_content[f"\x00{doc['url']}"] = doc
            continue
        existing = by_content.get(key)
        if existing is None:
            by_content[key] = doc
        elif len(doc["text"]) > len(existing["text"]):
            existing["url"] = _better_url(existing["url"], doc["url"])
            existing.update({"text": doc["text"], "description": doc["description"]})
        else:
            existing["url"] = _better_url(existing["url"], doc["url"])
    return list(by_content.values())


# ---------------------------------------------------------------------------
# --live mode: derive queries from a mandate, live-search, visit, build documents
# ---------------------------------------------------------------------------

#: Sentence-ish boundaries: a mandate is written prose, not one search query, so its own
#: punctuation is the cheapest deterministic signal for "here is a distinct sub-question".
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
#: Runs of Title-Case words -- a cheap, dependency-free stand-in for named-entity extraction,
#: good enough to add a query variant that is shorter and more targeted than a full sentence.
_PROPER_NOUN_RUN = re.compile(r"(?:[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)+)")


def derive_queries(mandate: str, count: int) -> List[str]:
    """Distinct, deterministic search queries derived from one task mandate.

    Leads with the mandate itself (the query most likely to surface the primary source), then
    adds its sentence-level sub-questions, then any multi-word proper-noun runs found in it --
    each a plausible narrower query a human would try next. No randomness and no model call, so
    the same mandate always yields the same queries and a live run is reproducible.

    :param mandate: the task's full instruction text.
    :param count: maximum queries to return.
    :returns: up to ``count`` distinct queries, mandate first.
    """
    mandate = str(mandate or "").strip()
    if not mandate or count <= 0:
        return []
    seen_lower = set()
    queries: List[str] = []

    def _add(candidate: str) -> None:
        candidate = " ".join(candidate.split()).strip()
        low = candidate.lower()
        if candidate and low not in seen_lower:
            seen_lower.add(low)
            queries.append(candidate)

    _add(mandate)
    for sentence in _SENTENCE_SPLIT.split(mandate):
        if len(queries) >= count:
            break
        if len(sentence.strip()) > 8:
            _add(sentence)
    for match in _PROPER_NOUN_RUN.findall(mandate):
        if len(queries) >= count:
            break
        _add(match)
    return queries[:count]


async def live_harvest(
    mandates: List[Tuple[str, str]],
    query_search: Callable[[str, int], Awaitable[Optional[List[Dict[str, str]]]]],
    visit: Callable[[str], Awaitable[str]],
    *,
    queries_per_task: int = 5,
    visits_per_query: int = 5,
    max_searches: Optional[int] = None,
    max_text_chars: int = MAX_TEXT_CHARS,
    queries_override: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[Dict[str, str]], int, bool]:
    """Live-search each task's derived queries and visit its top results.

    A hard, pre-flight budget check -- decided before ``query_search`` is ever awaited -- means
    the actual number of paid calls can never exceed ``max_searches``; the run stops cleanly
    (not mid-call) the moment the cap would be crossed. Documents are *not* deduplicated here --
    that is :func:`dedupe_documents`'s job, run once over the combined harvested + live set so
    the same page found by two different queries collapses to one entry.

    :param mandates: ``(task_id, mandate_text)`` pairs.
    :param query_search: ``async (query, count) -> results|None``, the paid call.
    :param visit: ``async (url) -> text``, a free page fetch.
    :param queries_override: optional ``{task_id: [queries]}`` bypassing :func:`derive_queries`
        for that task -- used by tests to pin exact query text.
    :returns: ``(raw_documents, searches_used, budget_exhausted)``.
    """
    raw_docs: List[Dict[str, str]] = []
    searches_used = 0
    budget_exhausted = False
    visited_urls: set = set()

    for task_id, mandate in mandates:
        if budget_exhausted:
            break
        queries = (queries_override or {}).get(task_id)
        if queries is None:
            queries = derive_queries(mandate, queries_per_task)
        for query in queries:
            if max_searches is not None and searches_used >= max_searches:
                budget_exhausted = True
                break
            # A failed search skips this QUERY; it must never abort the harvest. The sibling
            # `visit` below has always been guarded, and the asymmetry had real money on it: one
            # transient "Request failed after 3 attempts" on the FIRST query of a 22-task build
            # discarded every document already paid for. The attempt still counts against
            # `max_searches`, so a dead key or an exhausted quota burns the budget down and stops
            # rather than retrying forever.
            try:
                results = await query_search(query, visits_per_query) or []
            except Exception as exc:  # noqa: BLE001 -- one bad search must not abort the run
                _logger.warning(f"[BUILD-CORPUS] search failed for {query[:60]!r}: {exc}")
                results = []
            searches_used += 1
            for row in list(results)[:visits_per_query]:
                url = str((row or {}).get("url") or "")
                if not url or url in visited_urls:
                    continue
                visited_urls.add(url)
                try:
                    text = await visit(url)
                except Exception:  # noqa: BLE001 -- one bad fetch must not abort the run
                    continue
                if not text:
                    continue
                raw_docs.append({
                    "url": url,
                    "title": str((row or {}).get("title") or ""),
                    "description": str((row or {}).get("description") or ""),
                    "text": str(text)[:max_text_chars],
                })
    return raw_docs, searches_used, budget_exhausted


def load_mandates(test_ids: List[str]) -> List[Tuple[str, str]]:
    """``(task_id, mandate)`` pairs for the given task ids.

    Reuses ``prewarm_fixtures``' task-module resolution rather than re-implementing mandate
    lookup: both scripts need the same "task id -> statement text" step, and it already handles
    module discovery correctly.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import prewarm_fixtures  # local import: keeps pure-unit tests free of the agent import graph

    return [(tid, mandate) for tid, mandate, _urls in prewarm_fixtures._load_tasks(test_ids)]


async def _run_live(test_ids: List[str], out_dir: str, results_dir: str, queries_per_task: int,
                    visits_per_query: int, max_searches: Optional[int]) -> int:
    """Drive a real ``--live`` recording pass: load mandates, search, visit, merge, write."""
    from shared.connector_config import ConnectorConfig
    from agent.app.connector_llm import ConnectorLLM
    from agent.app.connector_search import create_search_backend
    from agent.app.connector_http import ConnectorHttp
    from agent.app.connector_chroma import ConnectorChroma
    from agent.app.agent_io import AgentIO
    from agent.app.telemetry import TelemetrySession

    mandates = load_mandates(test_ids)
    if not mandates:
        print("no mandates resolved for the given --tests; nothing to do", file=sys.stderr)
        return 2

    config = ConnectorConfig()
    agent_io = AgentIO(
        connector_llm=ConnectorLLM(config),
        connector_search=create_search_backend(config),
        connector_http=ConnectorHttp(config),
        connector_chroma=ConnectorChroma(config),
        telemetry=TelemetrySession(enabled=False, mandate="", correlation_id="build_corpus_live",
                                   trace_path=None),
        collection_name="build_corpus_live",
    )

    async def _search(query: str, count: int) -> Optional[List[Dict[str, str]]]:
        return await agent_io.search(query, count=count, timeout_seconds=20)

    async def _visit(url: str) -> str:
        return await agent_io.visit(url, timeout_seconds=30)

    live_docs, searches_used, exhausted = await live_harvest(
        mandates, _search, _visit,
        queries_per_task=queries_per_task, visits_per_query=visits_per_query,
        max_searches=max_searches)

    harvested = harvest_documents(results_dir)
    merged = dedupe_documents(harvested + live_docs)
    target = write_corpus(out_dir, merged)
    total_chars = sum(len(doc["text"]) for doc in merged)
    print(f"live: {searches_used} search(es) used"
         f"{' (budget exhausted)' if exhausted else ''}, ~${searches_used * 0.001:.3f} estimated")
    print(f"live-fetched {len(live_docs)} document(s); merged with {len(harvested)} harvested "
         f"-> {len(merged)} document(s), {total_chars} chars -> {target}")
    print(f"replay with: SEARCH_PROVIDER=corpus LEDGER_CORPUS_DIR={out_dir}")
    return 0


def write_corpus(out_dir: str, documents: List[Dict[str, str]]) -> Path:
    """Write ``documents.jsonl`` under ``out_dir``, creating it if needed.

    :returns: the path written.
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "documents.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, sort_keys=True) + "\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="agent/idea_test_results",
                        help="directory of per-cell result JSONs to harvest")
    parser.add_argument("--out", required=True, help="corpus directory to write")
    parser.add_argument("--live", action="store_true",
                        help="also record fresh evidence with a real search+visit pass "
                             "(spends money -- see --max-searches)")
    parser.add_argument("--tests", default="",
                        help="--live only: comma-separated task ids to record evidence for")
    parser.add_argument("--max-searches", type=int, default=None,
                        help="--live only: hard cap on paid query_search calls")
    parser.add_argument("--queries-per-task", type=int, default=5,
                        help="--live only: search queries derived per task mandate")
    parser.add_argument("--visits-per-query", type=int, default=5,
                        help="--live only: top-N result pages visited per query")
    args = parser.parse_args()

    if args.live:
        test_ids = [t.strip() for t in args.tests.split(",") if t.strip()]
        if not test_ids:
            print("--live requires --tests <comma-separated task ids>", file=sys.stderr)
            return 2
        return asyncio.run(_run_live(
            test_ids, args.out, args.results_dir, args.queries_per_task,
            args.visits_per_query, args.max_searches))

    documents = harvest_documents(args.results_dir)
    target = write_corpus(args.out, documents)
    total_chars = sum(len(doc["text"]) for doc in documents)
    print(f"harvested {len(documents)} document(s), {total_chars} chars -> {target}")
    print(f"replay with: SEARCH_PROVIDER=corpus LEDGER_CORPUS_DIR={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
