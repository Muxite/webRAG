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
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

#: Canonical per-cell result files. ``*_summary.json`` reflects only the last cell of a
#: multi-invocation run and ``*.jsonl`` are traces; counting either inflates any tally over the
#: results directory (it once produced a throughput figure 2.1x too high).
RESULT_GLOB = "*_r[0-9]*.json"
#: Page body kept per document. Enough to rank on, bounded so a corpus stays reviewable.
MAX_TEXT_CHARS = 20000
#: Opening characters used to decide two pages are the same content. Long enough that distinct
#: articles diverge, short enough that different truncations of one page still agree.
CONTENT_KEY_CHARS = 400


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
    args = parser.parse_args()

    documents = harvest_documents(args.results_dir)
    target = write_corpus(args.out, documents)
    total_chars = sum(len(doc["text"]) for doc in documents)
    print(f"harvested {len(documents)} document(s), {total_chars} chars -> {target}")
    print(f"replay with: SEARCH_PROVIDER=corpus LEDGER_CORPUS_DIR={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
