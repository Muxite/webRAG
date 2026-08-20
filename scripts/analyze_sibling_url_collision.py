#!/usr/bin/env python3
"""How often does a batch of sibling visits fetch the SAME page? $0, offline.

Unit of analysis: a **sibling visit batch** -- one parent's ``children`` restricted to
executed ``visit`` leaves that recorded a URL. Two siblings landing on one URL is paid for
twice (fetch, embed, judge) and read twice by the merge, which sees the same evidence
duplicated rather than the two different pages the plan asked for.

The interesting column is not the rate but the **cause**, which the recorded details make
separable without re-running anything:

* ``fallback`` -- neither duplicate declared a URL. The URL came from the resolution cascade
  in ``VisitLeafAction.execute`` (think node -> ``_extract_url_from_parents`` ->
  ``_extract_url_from_sibling_results`` -> the Chroma link query), which is sibling-blind:
  it ranks the same pool the same way for every sibling, so every URL-less sibling of a
  fan-out is handed the same top hit. This is the class ``action.visit_sibling_url_dedup``
  addresses.
* ``declared`` -- every duplicate carried the same explicit ``url``. The planner wrote it,
  typically a sequential chain whose later hops were planned before the data that would
  name their page existed, so each hop repeats the one URL it does know (the seed).
* ``mixed`` / ``hijacked`` -- one declared and one resolved onto it, or the fetched URL is
  not the declared one at all (the declared visit failed and a link scavenge took over).

Also reported: the language of the visited Wikipedia hosts, because the traced example
(``eps2_on_070``) had four English-task siblings all reading ``hr.wikipedia.org``.

Analysis only: reads result JSON and prints. Writes nothing the engine reads.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python \\
      scripts/analyze_sibling_url_collision.py
    ./.venv/bin/python scripts/analyze_sibling_url_collision.py --examples 5
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR_DEFAULT = REPO_ROOT / "agent" / "idea_test_results"

WIKI_HOST_RE = re.compile(r"^([a-z\-]{2,3})\.(?:m\.)?wikipedia\.org$")


def _visit_url(details: Dict[str, Any]) -> Optional[str]:
    result = details.get("action_result")
    url = result.get("url") if isinstance(result, dict) else None
    return str(url or details.get("visit_url") or "") or None


def _declared_url(details: Dict[str, Any]) -> Optional[str]:
    for key in ("url", "optional_url", "link"):
        value = details.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value.strip()
    return None


def sibling_visit_batches(payload: Any) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Every parent whose executed visit children recorded >=2 URLs, with those children."""
    if not isinstance(payload, dict):
        return []
    execution = payload.get("execution")
    graph = execution.get("graph") if isinstance(execution, dict) else None
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, dict):
        return []
    out = []
    for parent_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        visits = []
        for child_id in node.get("children") or []:
            child = nodes.get(child_id)
            details = child.get("details") if isinstance(child, dict) else None
            if not isinstance(details, dict) or details.get("action") != "visit":
                continue
            url = _visit_url(details)
            if not url:
                continue
            visits.append({
                "node_id": child_id,
                "title": (child.get("title") or details.get("goal") or "")[:70],
                "url": url,
                "declared": _declared_url(details),
            })
        if len(visits) >= 2:
            out.append((parent_id, visits))
    return out


def classify(members: List[Dict[str, Any]], url: str) -> str:
    declared = [m["declared"] for m in members]
    if all(d == url for d in declared):
        return "declared"
    if all(d is None for d in declared):
        return "fallback"
    if any(d not in (None, url) for d in declared):
        return "hijacked"
    return "mixed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR_DEFAULT))
    parser.add_argument("--pattern", default="**/*.json")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args(argv)

    batches: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    for path in sorted(Path(args.results_dir).glob(args.pattern)):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001. A corrupt or non-result JSON is simply skipped
            continue
        for parent_id, visits in sibling_visit_batches(payload):
            batches.append((path.name, parent_id, visits))

    if not batches:
        print("no sibling visit batches found")
        return 1

    duplicated = [b for b in batches if len({v["url"] for v in b[2]}) < len(b[2])]
    print(f"sibling visit batches (>=2 executed visits with a URL): {len(batches)}")
    print(f"batches with a duplicate URL: {len(duplicated)} "
          f"({len(duplicated) / len(batches):.1%})")

    severity = Counter()
    all_identical = 0
    groups: List[Tuple[str, str, str, List[Dict[str, Any]]]] = []
    for source, parent_id, visits in duplicated:
        counts = Counter(v["url"] for v in visits)
        severity[max(counts.values())] += 1
        if len(counts) == 1:
            all_identical += 1
        for url, count in counts.items():
            if count > 1:
                groups.append((source, parent_id, url, [v for v in visits if v["url"] == url]))
    print(f"  max siblings on one URL: "
          f"{', '.join(f'{k}x:{v}' for k, v in sorted(severity.items()))}")
    print(f"  batches where EVERY sibling hit the same URL: {all_identical} "
          f"({all_identical / len(duplicated):.1%} of duplicated batches)")

    print()
    print("=== cause of the duplicate (per duplicated URL group) ===")
    by_cause: Dict[str, List[Tuple[str, str, str, List[Dict[str, Any]]]]] = defaultdict(list)
    for group in groups:
        by_cause[classify(group[3], group[2])].append(group)
    for cause, items in sorted(by_cause.items(), key=lambda kv: -len(kv[1])):
        print(f"{len(items):4} {cause} ({len(items) / len(groups):.1%})")

    print()
    print("=== visited Wikipedia host language ===")
    for label, urls in (
        ("all sibling visits", [v["url"] for _, _, vs in batches for v in vs]),
        ("duplicated visits", [v["url"] for g in groups for v in g[3]]),
    ):
        langs = Counter()
        for url in urls:
            match = WIKI_HOST_RE.match((urlparse(url).netloc or "").lower())
            if match:
                langs[match.group(1)] += 1
        total = sum(langs.values())
        non_en = total - langs.get("en", 0)
        print(f"{label}: n={len(urls)} wikipedia={total} non-English={non_en} "
              f"({non_en / total:.2%} of wikipedia hits)" if total else f"{label}: no wikipedia")
        if non_en:
            print("   ", {k: v for k, v in langs.most_common() if k != "en"})

    print()
    print("=== most-duplicated URLs ===")
    for url, count in Counter(v["url"] for g in groups for v in g[3]).most_common(10):
        print(f"{count:4}  {url[:100]}")

    if args.examples:
        print()
        print("=== examples (whole batch on one URL) ===")
        shown = 0
        for source, parent_id, url, members in groups:
            if len(members) < 3 or shown >= args.examples:
                continue
            shown += 1
            print(f"\n{source} parent={parent_id[:8]} -> {url}")
            for member in members:
                print(f"    {member['title']!r} declared={member['declared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
