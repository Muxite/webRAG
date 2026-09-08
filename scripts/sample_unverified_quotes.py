#!/usr/bin/env python3
"""Stratified sampler of UNVERIFIED extraction quotes, for the L4 hand audit.

Phase 0d of ``docs/superpowers/plans/2026-09-08-ledger-dag-replan.md``. Stored cells carry, at
``execution.output.extractions[]``, one record per extracted (entity, field) pair with a
``quote`` the model claimed to have copied from a page and a ``quote_verified`` flag saying
whether the host could actually locate that quote in the stored page text. This script draws a
seeded, proportionally stratified sample of the ``quote_verified == False`` records and writes
them to a CSV together with the surrounding stored page text, so a human can label each one:

  * ``paraphrase`` -- the page supports the claim but the model reworded it,
  * ``offset_miss`` -- the exact string IS on the page and the verifier missed it,
  * ``unsupported`` -- the page does not support the quote at all.

The consequence declared in the plan: if ``paraphrase`` dominates, an off-the-shelf NLI
cross-encoder (no training) is queued ahead of any further numeric-operand work.

Cell discovery reuses ``compare_arms._result_files_for_id`` (with ``bench_common.results_dir()``
as the default directory) rather than adding a seventh loader to the repo; ``*_summary.json``
and ``*_report_v<N>.json`` siblings are skipped, since both match the loose run-id glob and
carry a different schema.

Usage (from the repo root)::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/sample_unverified_quotes.py \
        --n 200 --seed 1 --out /tmp/unverified_quotes.csv

Offline only: reads stored JSON, no model calls, no network.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common  # noqa: E402
import compare_arms as ca  # noqa: E402

#: Campaigns that carry ledger extractions with per-quote verification (the evidence_loop-bearing
#: runs). Overridable with --prefixes.
DEFAULT_PREFIXES: List[str] = ["ledgernum22r3", "ledgerfinal01", "bughunt01", "gpu0831b"]

#: Characters of stored page text to show on either side of the located quote.
DEFAULT_RADIUS = 300

#: Fallback window when the quote cannot be located on the page at all.
PAGE_HEAD_CHARS = 600

#: Minimum fuzzy-match length before a locate is believed (below this a common word like "the"
#: would "locate" the quote anywhere on the page and centre the window on noise).
_MIN_FUZZY_CHARS = 12
_MIN_FUZZY_FRACTION = 0.4

CSV_COLUMNS: Tuple[str, ...] = (
    "cell_file", "test_id", "model", "variant", "entity", "field", "value", "unit",
    "quote", "quote_fail_reason", "source_url", "page_id", "window_method", "page_window",
    # Left empty on purpose -- these three are the human auditor's columns.
    "label", "supporting_span", "notes",
)

_WS = re.compile(r"\s+")


# ==============================================================================================
# Loading
# ==============================================================================================

def discover_cell_files(prefixes: Sequence[str], results_dir: Optional[str] = None) -> List[str]:
    """Result-cell JSON paths for ``prefixes``, with summaries and reports removed.

    :param prefixes: run-id prefixes (e.g. ``["gpu0831b"]``).
    :param results_dir: directory to glob; defaults to :func:`bench_common.results_dir`.
    :returns: sorted, de-duplicated list of cell file paths.
    :raises: nothing.
    """
    rd = results_dir or str(bench_common.results_dir())
    found: set = set()
    for prefix in prefixes:
        prefix = str(prefix).strip()
        if prefix:
            found.update(ca._result_files_for_id(prefix, rd, exact=False))
    keep = []
    for path in sorted(found):
        base = os.path.basename(path)
        if base.endswith("_summary.json") or "_report_" in base:
            continue
        keep.append(path)
    return keep


def _pages_by_id(output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for page in output.get("pages") or []:
        if isinstance(page, dict) and page.get("page_id"):
            out[str(page["page_id"])] = page
    return out


def candidates_from_cell(path: str, data: Dict[str, Any],
                         radius: int = DEFAULT_RADIUS) -> List[Dict[str, Any]]:
    """Every ``quote_verified is False`` extraction record in one loaded cell, as audit rows."""
    output = ((data.get("execution") or {}).get("output") or {})
    if not isinstance(output, dict):
        return []
    pages = _pages_by_id(output)
    model = data.get("model")
    variant = data.get("execution_variant")
    test_id = ((data.get("test_metadata") or {}).get("test_id")) or "?"
    rows = []
    for rec in output.get("extractions") or []:
        if not isinstance(rec, dict) or rec.get("quote_verified") is not False:
            continue
        page = pages.get(str(rec.get("page_id") or "")) or {}
        window, method = locate_window(str(page.get("text") or ""),
                                       str(rec.get("quote") or ""), radius=radius)
        rows.append({
            "cell_file": os.path.basename(path),
            "test_id": str(test_id),
            "model": str(model or "?"),
            "variant": str(variant or "?"),
            "entity": str(rec.get("entity") or ""),
            "field": str(rec.get("field") or ""),
            "value": str(rec.get("value") or ""),
            "unit": str(rec.get("unit") or ""),
            "quote": str(rec.get("quote") or ""),
            "quote_fail_reason": str(rec.get("quote_fail_reason") or ""),
            "source_url": str(rec.get("source_url") or ""),
            "page_id": str(rec.get("page_id") or ""),
            "window_method": method,
            "page_window": window,
            "label": "", "supporting_span": "", "notes": "",
        })
    return rows


def collect_candidates(prefixes: Sequence[str], results_dir: Optional[str] = None,
                       radius: int = DEFAULT_RADIUS) -> List[Dict[str, Any]]:
    """All unverified-quote audit rows across every cell under ``prefixes``.

    Streams one file at a time and never keeps a parsed cell alive after extracting its rows, so
    a multi-GB results dir costs one cell of memory. Unreadable files are skipped silently (a
    half-written cell from an interrupted campaign must not abort a sampling run).
    """
    rows: List[Dict[str, Any]] = []
    for path in discover_cell_files(prefixes, results_dir):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            rows.extend(candidates_from_cell(path, data, radius=radius))
    return rows


# ==============================================================================================
# Quote location
# ==============================================================================================

def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """Lowercased, whitespace-collapsed copy of ``text`` plus norm-index -> original-index map."""
    out: List[str] = []
    idx: List[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space or not out:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    return "".join(out), idx


def locate_window(page_text: str, quote: str, radius: int = DEFAULT_RADIUS) -> Tuple[str, str]:
    """A +/-``radius`` character window of ``page_text`` around the best locate of ``quote``.

    Four strategies, cheapest first: exact substring, whitespace/case-normalized substring, a
    ``difflib`` longest-common-block (paraphrases and truncations still centre roughly right),
    and finally the head of the page.

    :returns: ``(window, method)`` where method is one of ``exact``, ``normalized``, ``fuzzy``,
        ``page_head`` (quote not locatable) or ``no_page`` (no stored text for the page id).
    :raises: nothing.
    """
    if not page_text:
        return "", "no_page"
    if not quote:
        return page_text[:PAGE_HEAD_CHARS], "page_head"

    start = page_text.find(quote)
    if start >= 0:
        return _slice(page_text, start, start + len(quote), radius), "exact"

    norm_text, idx_map = _normalize_with_map(page_text)
    norm_quote, _ = _normalize_with_map(quote)
    if norm_quote:
        pos = norm_text.find(norm_quote)
        if pos >= 0:
            first = idx_map[pos]
            last = idx_map[min(pos + len(norm_quote), len(idx_map)) - 1] + 1
            return _slice(page_text, first, last, radius), "normalized"

        match = SequenceMatcher(None, norm_text, norm_quote, autojunk=False).find_longest_match(
            0, len(norm_text), 0, len(norm_quote))
        threshold = max(_MIN_FUZZY_CHARS, int(_MIN_FUZZY_FRACTION * len(norm_quote)))
        if match.size >= threshold:
            first = idx_map[match.a]
            last = idx_map[min(match.a + match.size, len(idx_map)) - 1] + 1
            return _slice(page_text, first, last, radius), "fuzzy"

    return page_text[:PAGE_HEAD_CHARS], "page_head"


def _slice(text: str, start: int, end: int, radius: int) -> str:
    return text[max(0, start - radius):min(len(text), end + radius)]


# ==============================================================================================
# Stratification
# ==============================================================================================

def stratum_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """The stratum a row belongs to: ``(model, execution_variant)``."""
    return (str(row.get("model") or "?"), str(row.get("variant") or "?"))


def stratum_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    """``Counter`` of :func:`stratum_key` over ``rows``."""
    return Counter(stratum_key(r) for r in rows)


def allocate(sizes: Dict[Tuple[str, str], int], n: int) -> Dict[Tuple[str, str], int]:
    """Largest-remainder proportional allocation of ``n`` draws over strata of given ``sizes``.

    Capped at each stratum's population, with the shortfall from any exhausted stratum
    redistributed to strata that still have room. Deterministic: ties break on the sorted
    stratum key, never on dict order.
    """
    total = sum(sizes.values())
    if total <= 0 or n <= 0:
        return {k: 0 for k in sizes}
    if n >= total:
        return dict(sizes)

    keys = sorted(sizes)
    exact = {k: n * sizes[k] / total for k in keys}
    alloc = {k: min(int(exact[k]), sizes[k]) for k in keys}
    # Distribute what integer truncation and capping left over, biggest fractional part first.
    order = sorted(keys, key=lambda k: (-(exact[k] - int(exact[k])), k))
    while sum(alloc.values()) < n:
        progressed = False
        for k in order:
            if sum(alloc.values()) >= n:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                progressed = True
        if not progressed:  # every stratum is at capacity
            break
    return alloc


#: Stable within-stratum ordering, so the seeded draw does not depend on filesystem order.
_SORT_FIELDS = ("cell_file", "entity", "field", "value", "quote")


def _sort_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(str(row.get(f, "")) for f in _SORT_FIELDS)


def stratified_sample(rows: Sequence[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Seeded, proportionally stratified sample of ``rows`` by ``(model, variant)``.

    Each stratum is sampled with its own ``Random(f"{seed}:{key}")`` so the draw for one stratum
    does not depend on how many rows another stratum happened to contribute -- reruns with the
    same seed are byte-identical, and a different seed gives a different sample.
    """
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(stratum_key(row), []).append(row)
    alloc = allocate({k: len(v) for k, v in buckets.items()}, n)

    picked: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        k = alloc.get(key, 0)
        if k <= 0:
            continue
        pool = sorted(buckets[key], key=_sort_key)
        rng = Random(f"{seed}:{key[0]}:{key[1]}")
        picked.extend(rng.sample(pool, min(k, len(pool))))
    return picked


# ==============================================================================================
# Output
# ==============================================================================================

def write_csv(rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    """Write ``rows`` to ``out_path`` with exactly :data:`CSV_COLUMNS`, in order."""
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES),
                    help="comma-separated run-id prefixes (default: the evidence_loop campaigns)")
    ap.add_argument("--n", type=int, default=200, help="sample size (default 200)")
    ap.add_argument("--seed", type=int, default=1, help="sampling seed (default 1)")
    ap.add_argument("--out", default="", help="CSV path (default: no file, counts only)")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                    help=f"page-text window radius in chars (default {DEFAULT_RADIUS})")
    ap.add_argument("--results-dir", default="",
                    help="results directory (default: bench_common.results_dir())")
    args = ap.parse_args(argv)

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    rows = collect_candidates(prefixes, args.results_dir or None, radius=args.radius)
    print(f"candidate rows (quote_verified == False): {len(rows)} "
          f"across {len({r['cell_file'] for r in rows})} cells, prefixes={','.join(prefixes)}")
    if not rows:
        return 0

    picked = stratified_sample(rows, args.n, args.seed)
    pop, sample = stratum_counts(rows), stratum_counts(picked)
    print(f"\n{'model':<22}{'variant':<26}{'population':>11}{'sampled':>9}")
    print("-" * 68)
    for key in sorted(pop):
        print(f"{key[0]:<22}{key[1]:<26}{pop[key]:>11}{sample.get(key, 0):>9}")
    print("-" * 68)
    print(f"{'TOTAL':<48}{sum(pop.values()):>11}{sum(sample.values()):>9}")
    print("\nwindow methods: " + ", ".join(
        f"{m}={c}" for m, c in sorted(Counter(r["window_method"] for r in picked).items())))

    if args.out:
        write_csv(picked, args.out)
        print(f"\nwrote {len(picked)} rows to {args.out} "
              f"(label / supporting_span / notes left empty for the auditor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
