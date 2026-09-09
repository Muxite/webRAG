#!/usr/bin/env python3
"""Replay ``LedgerToolkit.host_derive`` over stored result cells -- Phase 2e of
``docs/superpowers/plans/2026-09-08-ledger-dag-replan.md``.

This is the plan's go/no-go for mint03, and it is deliberately a *replay*: every cell already on
disk carries the pages the run fetched (``execution.output.pages[]`` = ``{page_id, url,
content_hash, chars, text}``), so the host-side derivation can be recomputed from exactly the
evidence that run had, with no model, no network and no GPU. ``host_derive`` is model-invisible
(a finish-time host hook that reads the mandate and the pages, never the answer), which is why a
cell run with the mechanism OFF is still a valid replay subject: the mechanism's inputs were
recorded regardless of whether the hook was installed.

For each cell, and for each ranker under test, the script:

  1. builds a fresh :class:`~agent.app.ledger_tools.LedgerToolkit`;
  2. ``register_page(url, text)`` for every stored page, in stored order;
  3. calls ``host_derive(get_task_statement())`` with the REAL task module's mandate;
  4. scores the outcome three ways --
     ``host_value_correct`` (vs the module's ``DERIVED`` / ``WINNER`` ground truth: the PRIMARY
     metric), ``host_agrees`` (vs the model's own deliverable, with units: secondary), and the
     chain ``certified`` signal already computed by ``ledger_risk_coverage.classify_cell``.

Two negative controls run side by side: the **availability-only baseline** (accept every cell
whose reason is ``computed``, i.e. what the mechanism would deliver with no agreement filter) and
the **document-order ranker** (``operand_attribution.document_order_ranker``), whose constant
score is not on the hand rule's scale -- it is therefore driven with ``min_score=0.0``.

Every rate in the report uses the FULL stratum as the denominator, with availability shown
beside it, never as the denominator: "45% of computed cells" and "9% of cells" are different
claims and the second one is the one a campaign endpoint can be written against.

Usage (from the repo root)::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/host_derive_replay.py
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/host_derive_replay.py --limit 20
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/host_derive_replay.py \
        --prefixes mint02 --rankers hand_rule --out-dir /tmp/hd

``--prefetch`` (the availability drive, ``docs/handoffs/AVAILABILITY_DRIVE_HANDOFF_2026-09-08.md``)
runs ``agent.app.host_prefetch.host_prefetch`` ONCE per cell before the rankers, so the cells the
model never fetched a page for -- mint03's ``no_pages`` bucket, which the plain replay SKIPS --
become replay subjects. Prefetched pages are registered on every ranker's kit with
``source=host_prefetch``, and every row records whether a selected slot landed on one
(``prefetch_used``). It goes to the network (search + page visits) through the web-fixture cache:
``IDEA_TEST_FIXTURES`` is forced to ``replay`` (load-or-fetch-and-save) when unset, so a rerun
is $0 and deterministic; ``record`` is never forced because it never loads and would re-bill.
The report then states BOTH denominators: every cell including ``no_pages``, and the
stored-page cells the plain replay is defined over.

Outputs (under ``--out-dir``): ``rows.jsonl`` (one row per (cell, ranker), the full record
including per-slot operand choices), ``summary.json`` (every table the report prints), and
``report.txt`` (the printed report). Deterministic: files are processed in sorted order and
nothing here samples.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "agent"), str(_REPO_ROOT / "services")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import ledger_risk_coverage as LRC  # noqa: E402

#: Ranker registry: name -> (factory, default min_score). ``document_order``'s score is a constant
#: rank-derived number, NOT a probability on the hand rule's scale, so the 0.93 floor calibrated
#: for the hand rule would reject everything it proposes; the plan specifies ``min_score=0.0``
#: for it, which is what makes it a fair ablation of the RANKING rather than of the floor.
RANKERS: Dict[str, Tuple[str, float]] = {
    "hand_rule": ("default_ranker", None),
    "document_order": ("document_order_ranker", 0.0),
}

DEFAULT_PREFIXES = "mint01,mint02,ladder03"
DEFAULT_OUT_DIR = "agent/idea_test_results/host_derive_replay"

#: The accept signals compared at every operating point. ``availability_only`` is the pre-declared
#: baseline (accept iff the mechanism produced a number at all); ``chain_certified`` is the frozen
#: 5-clause chain the host is being measured against; ``union``/``intersection`` show whether the
#: two are reading the same cells.
SIGNALS: Tuple[str, ...] = ("host_certified", "availability_only", "chain_certified",
                            "union", "intersection")

#: Cap on forensics lines in the PRINTED report; the JSONL always carries them all.
FORENSICS_PRINT_CAP = 40


# ---------------------------------------------------------------------------------------------
# Cell-level replay
# ---------------------------------------------------------------------------------------------

#: Where a cell's fetched pages can live. ``langgraph_react`` writes ``execution.output.pages``;
#: ``sequential_react`` writes ONLY ``execution.output.evidence_graph.pages`` and leaves
#: ``output.pages`` empty/absent. Reading just the first location silently classified every
#: sequential_react cell as ``no_pages`` -- see the doc's "§15 Correction".
PAGE_SOURCES: Tuple[str, ...] = ("output", "evidence_graph", "none")

#: ``page_source`` value for a cell that had NO stored pages and was replayed only because
#: ``--prefetch`` fetched some. Distinct from the two storage locations above on purpose: an
#: availability number on these cells is a claim about the prefetcher, not about the run.
PREFETCHED_PAGE_SOURCE = "prefetched"

#: The ``source`` tag prefetched pages carry on a kit (mirrors ``host_prefetch.PREFETCH_SOURCE``;
#: read from the module when it is importable so the two can never drift silently).
PREFETCH_SOURCE_FALLBACK = "host_prefetch"


def cell_pages(raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """``(pages, page_source)`` for one loaded cell dict.

    Precedence is ``output.pages`` (non-empty), else ``evidence_graph.pages`` (non-empty), else
    ``([], "none")``. An EMPTY ``output.pages`` list must fall through rather than shadow the
    evidence-graph list, which is exactly the sequential_react shape on disk.

    Both lists carry the same record fields -- ``page_id, url, content_hash, chars, text``
    (plus ``stored_chars``/``truncated``) -- verified across mint01/mint02/mint03/ladder03, so
    no key mapping is needed and the two are interchangeable inputs to ``register_page``. Where
    both lists exist the ``output`` copy wins, which keeps every previously-replayed
    langgraph_react cell byte-identical to the pre-correction run.
    """
    output = ((raw.get("execution") or {}).get("output")) or {}
    pages = output.get("pages") or []
    if pages:
        return list(pages), "output"
    graph = output.get("evidence_graph")
    graph_pages = (graph.get("pages") or []) if isinstance(graph, dict) else []
    if graph_pages:
        return list(graph_pages), "evidence_graph"
    return [], "none"


def skip_reason(raw: Dict[str, Any], *, allow_no_pages: bool = False) -> Optional[str]:
    """Why this cell cannot be replayed, or ``None`` when it can.

    ``allow_no_pages`` is the ``--prefetch`` switch: a cell with no stored pages is exactly the
    cell the prefetcher exists for, so under prefetch it is a replay subject, not a skip.

    Precedence is fixed so the counters partition the skipped set: ``infra_failed`` (the run did
    not really happen), then ``no_run_config`` (arm undeterminable -- ``run_config`` is null on
    some older campaigns), then ``no_pages`` (nothing to register, so ``host_derive`` could only
    ever return ``no_pages``). "Has pages" is decided by :func:`cell_pages`, i.e. across BOTH
    storage locations.
    """
    if bool(raw.get("infra_failed")):
        return "infra_failed"
    if not (raw.get("run_config") or {}):
        return "no_run_config"
    if not allow_no_pages and not cell_pages(raw)[0]:
        return "no_pages"
    return None


def campaign_prefix(name: str, prefixes: Sequence[str]) -> str:
    """Which of ``prefixes`` a cell FILENAME belongs to (``""`` if none match)."""
    for pref in prefixes:
        if name.startswith(f"{pref}_"):
            return pref
    return ""


def _make_ranker(name: str):
    from agent.app import operand_attribution as OA
    factory, _floor = RANKERS[name]
    return getattr(OA, factory)()


#: Set by ``--min-score`` to override the hand rule's floor for a sweep. The floor encodes "an
#: entry whose label says nothing about the field is never an operand" (see
#: ``ledger_tools._HOST_DERIVE_MIN_SCORE``), so lowering it deliberately buys availability with
#: correctness -- an exploratory operating-point curve, never a default.
_MIN_SCORE_OVERRIDE: Optional[float] = None


def _min_score_for(name: str) -> float:
    from agent.app.ledger_tools import _HOST_DERIVE_MIN_SCORE
    floor = RANKERS[name][1]
    if floor is None and _MIN_SCORE_OVERRIDE is not None:
        return float(_MIN_SCORE_OVERRIDE)
    return float(_HOST_DERIVE_MIN_SCORE if floor is None else floor)


def task_statement(test_id: str) -> Optional[str]:
    """The real task module's mandate text, or ``None`` when the module will not load.

    Reuses ``ledger_risk_coverage._load_task_module`` -- the same cached real-package import the
    ground-truth scorer already uses -- rather than adding another module loader.
    """
    module = LRC._load_task_module(test_id)
    if module is None:
        return None
    try:
        return module.get_task_statement()
    except Exception:  # noqa: BLE001 -- a broken module must not abort the replay
        return None


# ---------------------------------------------------------------------------------------------
# --prefetch support
# ---------------------------------------------------------------------------------------------

def prefetch_source() -> str:
    """The ``source`` tag prefetched pages are registered under."""
    try:
        from agent.app.host_prefetch import PREFETCH_SOURCE  # lazy: Lane-2 module, optional
        return str(PREFETCH_SOURCE)
    except Exception:  # noqa: BLE001 -- module absent or mid-landing: use the agreed literal
        return PREFETCH_SOURCE_FALLBACK


def register_page_compat(toolkit: Any, url: str, text: str, **kwargs: Any) -> str:
    """``toolkit.register_page`` with the provenance kwargs when the toolkit accepts them.

    The kwargs (``source``/``max_chars``/``structured``) are the extended signature; a toolkit
    predating it takes ``(url, text)`` only. Falling back keeps the page registered either way --
    provenance is then simply not recorded on the artifact, which the replay row still carries
    through its own ``prefetched_page_ids``.
    """
    if not kwargs:
        return str(toolkit.register_page(url, text))
    try:
        return str(toolkit.register_page(url, text, **kwargs))
    except TypeError:
        return str(toolkit.register_page(url, text))


def register_stored_pages(toolkit: Any, pages: Sequence[Dict[str, Any]]) -> None:
    for page in pages:
        toolkit.register_page(str(page.get("url") or ""), str(page.get("text") or ""))


def run_prefetch(toolkit: Any, statement: str, prefetcher: Any
                 ) -> Tuple[Dict[str, Any], List[Tuple[str, str, Any]]]:
    """Run ``prefetcher(toolkit, statement)`` once and capture every page it registered.

    The capture is an INSTANCE-level wrapper around ``toolkit.register_page`` (delegating to the
    bound original), so it records ``(url, text, kwargs)`` exactly as the prefetcher handed
    them over -- independent of how the toolkit stores pages internally, and without a second
    accessor that could disagree with the kit's own ``registered_urls()``/artifact view. The
    summary the prefetcher returns is passed through untouched; an exception becomes an
    ``{"error": ...}`` summary rather than aborting the cell.
    """
    captured: List[Tuple[str, str, Any]] = []
    original = toolkit.register_page

    def capturing_register_page(url: str, text: str, **kwargs: Any) -> str:
        # The WHOLE kwargs dict, not just `structured`: the prefetcher also passes
        # `infobox_chars` (the rendered-infobox boundary), and dropping it here would silently
        # re-register the page without it and re-admit the body-prose-as-infobox rows.
        captured.append((str(url or ""), str(text or ""), dict(kwargs)))
        return register_page_compat(_Original(original), url, text, **kwargs)

    toolkit.register_page = capturing_register_page
    try:
        summary = prefetcher(toolkit, statement)
    except Exception as exc:  # noqa: BLE001 -- a prefetch failure is a row fact, not a crash
        summary = {"error": f"{type(exc).__name__}: {exc}", "registered": len(captured)}
    finally:
        del toolkit.register_page  # restore the class method
    if not isinstance(summary, dict):
        summary = {"error": f"prefetcher returned {type(summary).__name__}",
                   "registered": len(captured)}
    return summary, captured


class _Original:
    """Adapter so :func:`register_page_compat` can call a captured bound method."""

    def __init__(self, bound: Any) -> None:
        self.register_page = bound


def make_live_prefetcher(loop: asyncio.AbstractEventLoop, *, http: Any, search: Any) -> Any:
    """A ``prefetcher(kit, mandate) -> summary`` bound to real connectors on ``loop``.

    ``host_prefetch`` is imported lazily and per call so the replay's plain mode never depends
    on it and a missing module surfaces as a per-cell ``error`` summary, not an import error at
    startup.
    """
    def prefetcher(kit: Any, mandate: str) -> Dict[str, Any]:
        from agent.app.host_prefetch import host_prefetch
        return loop.run_until_complete(host_prefetch(kit, mandate, http=http, search=search))
    return prefetcher


def slug_coverage(entity: Any, url: str) -> Optional[float]:
    """Fraction of ``entity``'s significant tokens the URL slug of ``url`` carries.

    Same tokeniser and slug rule as ``LedgerToolkit._host_derive_slug_coverage`` so the forensics
    can run over a stored row's ``url`` without a toolkit. ``None`` when the entity has no
    significant token to match on.
    """
    from agent.app.operand_attribution import _tokens
    wanted = _significant(entity)
    if not wanted:
        return None
    return len(wanted & set(_tokens(_slug_of(url)))) / len(wanted)


def replay_cell(path: Path, raw: Dict[str, Any], ranker_names: Sequence[str],
                prefixes: Sequence[str], *, prefetcher: Any = None) -> List[Dict[str, Any]]:
    """One JSONL row per ranker for a single cell (never raises).

    With ``prefetcher`` set (``--prefetch``), it is called ONCE per cell -- against a scratch kit
    holding the stored pages, so it sees what the run had and does not refetch it -- and the
    pages it registered are re-registered on every ranker's kit after the stored ones, tagged
    with the prefetch source. Rankers must not share the prefetch call: a second call would
    repeat the network round-trips and could register a different page set per ranker.

    The toolkit is rebuilt per ranker because ``host_derive`` MINTS nodes into the toolkit's
    graph: reusing one across rankers would let the first ranker's nodes exist while the second
    ran, and any de-duplication inside the graph would then be measuring the wrong thing.
    """
    from agent.app.ledger_tools import LedgerToolkit

    classified = LRC.classify_cell(path, raw)
    output = ((raw.get("execution") or {}).get("output")) or {}
    pages, page_source = cell_pages(raw)
    deliverable = output.get("final_deliverable") or ""
    final_numbers = LRC.extract_numbers(
        LRC.strip_citation_markers(LRC.strip_urls(deliverable))
    )
    test_id = classified["test_id"]
    statement = task_statement(test_id)

    prefetch_summary: Optional[Dict[str, Any]] = None
    prefetched: List[Tuple[str, str, Any]] = []
    if prefetcher is not None:
        if statement is not None:
            scratch = LedgerToolkit()
            register_stored_pages(scratch, pages)
            prefetch_summary, prefetched = run_prefetch(scratch, statement, prefetcher)
        else:
            prefetch_summary = {"error": "no_task_module", "registered": 0}
        if not pages:
            page_source = PREFETCHED_PAGE_SOURCE

    rows: List[Dict[str, Any]] = []
    for ranker_name in ranker_names:
        base: Dict[str, Any] = {
            "file": path.name,
            "prefix": campaign_prefix(path.name, prefixes),
            "test_id": test_id,
            "model": classified["model"],
            "host": classified["host"],
            "arm": classified["arm"],
            "split": classified["split"],
            "score": classified["score"],
            "wrong_05": classified["wrong_05"],
            "certified": classified["certified"],
            "ranker": ranker_name,
            "n_pages": len(pages),
            "page_source": page_source,
        }
        if prefetcher is not None:
            base.update({"prefetch": prefetch_summary, "n_prefetched": len(prefetched),
                         "prefetch_used": False, "prefetched_page_ids": []})
        if statement is None:
            rows.append({**base, "reason": "no_task_module", "value": None, "unit": "",
                         "winner_entity": None, "agrees_final": False, "agrees_any": False,
                         "agrees_entity": None, "unit_status": "unassessed",
                         "value_correct": None, "value_close": None, "host_certified": False,
                         "availability_only": False, "n_entries": 0, "slot_reasons": [],
                         "slots": []})
            continue
        toolkit = LedgerToolkit()
        register_stored_pages(toolkit, pages)
        prefetched_ids: List[str] = []
        for url, text, kwargs in prefetched:
            prefetched_ids.append(register_page_compat(
                toolkit, url, text, source=prefetch_source(), max_chars=len(text),
                structured=kwargs.get("structured"),
                infobox_chars=kwargs.get("infobox_chars")))
        hd = toolkit.host_derive(statement, ranker=_make_ranker(ranker_name),
                                 min_score=_min_score_for(ranker_name))
        if prefetcher is not None:
            used = {str(s.get("page_id")) for s in (hd.get("slots") or [])
                    if s.get("reason") == "selected"}
            base["prefetched_page_ids"] = prefetched_ids
            base["prefetch_used"] = bool(used & set(prefetched_ids))
        agreement = LRC.host_agrees(hd, deliverable, final_numbers, test_id)
        detail = LRC.host_value_correct_detail(hd, test_id)
        host_certified = bool(agreement["available"] and agreement["agrees_final"]
                              and not agreement["wrong_by_unit"])
        rows.append({
            **base,
            "reason": hd.get("reason"),
            "operation": hd.get("operation"),
            "mode": hd.get("mode"),
            "value": hd.get("value"),
            "unit": hd.get("unit"),
            "winner_entity": hd.get("winner_entity"),
            "agrees_final": bool(agreement["agrees_final"]),
            "agrees_any": bool(agreement["agrees_any"]),
            "agrees_entity": agreement["agrees_entity"],
            "unit_status": agreement["unit_status"],
            "value_correct": detail["correct"],
            "value_close": detail["close"],
            "value_kind": detail["kind"],
            "host_certified": host_certified,
            "availability_only": hd.get("reason") == "computed",
            "n_entries": hd.get("n_entries"),
            "slot_reasons": [s.get("reason") for s in (hd.get("slots") or [])],
            "slots": hd.get("slots") or [],
        })
    return rows


def classify_for_replay(path: Path, prefixes: Sequence[str], *, prefetching: bool
                        ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """``(cell or None, skip-counter keys)`` for one file -- the bookkeeping half of :func:`replay`.

    Split out so the serial and parallel paths cannot drift: a worker process classifies a cell
    with exactly this function, and the parent applies exactly these keys, in file order.
    """
    prefix = campaign_prefix(path.name, prefixes) or "?"
    keys = [f"{prefix}|files"]
    raw = LRC.load_cell(path)
    if raw is None:
        return None, keys + ["unreadable", f"{prefix}|unreadable"]
    reason = skip_reason(raw, allow_no_pages=prefetching)
    if reason is not None:
        return None, keys + [reason, f"{prefix}|{reason}"]
    source = cell_pages(raw)[1]
    if prefetching and source == "none":
        source = PREFETCHED_PAGE_SOURCE
    return raw, keys + ["replayed", f"{prefix}|replayed",
                        f"pages_{source}", f"{prefix}|pages_{source}"]


#: Per-process state for the worker pool: built once per process by :func:`_worker_init`.
_WORKER: Dict[str, Any] = {}


def _worker_init(ranker_names: Sequence[str], prefixes: Sequence[str], prefetching: bool,
                 results_dir: str, fixtures_mode: str, min_score: Optional[float] = None) -> None:
    """Build this worker's OWN connectors and event loop.

    Connectors and an asyncio loop are not picklable, so the pool cannot be handed a prefetcher
    -- each process constructs its own against the same on-disk web-fixture cache. The cache is
    content-addressed and this runs under ``IDEA_TEST_FIXTURES=replay`` (load-or-fetch-and-save),
    so concurrent workers reading it race only to write identical bytes.
    """
    os.environ["IDEA_TEST_FIXTURES"] = fixtures_mode
    global _MIN_SCORE_OVERRIDE
    _MIN_SCORE_OVERRIDE = min_score  # a spawned worker does not inherit module globals
    _WORKER.update(ranker_names=list(ranker_names), prefixes=list(prefixes),
                   prefetching=prefetching, results_dir=results_dir, prefetcher=None)
    if not prefetching:
        return
    from shared.connector_config import ConnectorConfig
    from agent.app.connector_http import ConnectorHttp
    from agent.app.connector_search import create_search_backend

    config = ConnectorConfig()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _WORKER["loop"] = loop
    _WORKER["http"] = ConnectorHttp(config)
    _WORKER["prefetcher"] = make_live_prefetcher(loop, http=_WORKER["http"],
                                                 search=create_search_backend(config))


def _worker_replay(name: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Replay one cell by FILE NAME (paths rebuilt here, so nothing path-like crosses the pipe)."""
    path = Path(_WORKER["results_dir"]) / name
    raw, keys = classify_for_replay(path, _WORKER["prefixes"],
                                    prefetching=_WORKER["prefetching"])
    if raw is None:
        return [], keys
    rows = replay_cell(path, raw, _WORKER["ranker_names"], _WORKER["prefixes"],
                       prefetcher=_WORKER["prefetcher"])
    return rows, keys


def replay(files: Sequence[Path], ranker_names: Sequence[str],
           prefixes: Sequence[str], *, prefetcher: Any = None
           ) -> Tuple[List[Dict[str, Any]], Counter]:
    """Replay every file; returns ``(rows, skip_counter)``.

    With ``prefetcher`` set, ``no_pages`` cells are replayed (counted under
    ``pages_prefetched`` / ``<prefix>|pages_prefetched``) instead of skipped.

    ``skip_counter`` also carries ``files``, ``unreadable``, ``replayed`` and one
    ``pages_<source>`` key per :data:`PAGE_SOURCES` entry so the report can
    state its own denominator without recomputing it, plus a per-campaign copy of every one of
    those keys under ``"<prefix>|<key>"`` -- which campaign lost its cells is the first question
    a large skip count raises, and it cannot be recovered from the pooled number.
    """
    rows: List[Dict[str, Any]] = []
    skips: Counter = Counter()
    skips["files"] = len(files)
    for path in files:
        raw, keys = classify_for_replay(path, prefixes, prefetching=prefetcher is not None)
        for key in keys:
            skips[key] += 1
        if raw is None:
            continue
        rows.extend(replay_cell(path, raw, ranker_names, prefixes, prefetcher=prefetcher))
    return rows, skips


def replay_parallel(files: Sequence[Path], ranker_names: Sequence[str], prefixes: Sequence[str],
                    *, workers: int, prefetching: bool, results_dir: Path
                    ) -> Tuple[List[Dict[str, Any]], Counter]:
    """:func:`replay` across ``workers`` processes, byte-identical to the serial path.

    Cells are independent by construction -- each builds a fresh ``LedgerToolkit`` and shares no
    state -- so the only thing parallelism can disturb is ORDER, and order is load-bearing here
    (the report's forensics print "first N shown"). ``Pool.imap`` with ``chunksize=1`` yields
    results in submission order, so rows and skip keys are applied in exactly the file order the
    serial loop would use. ``--workers 1`` must therefore be indistinguishable from no flag, and
    a parallel run's ``summary.json`` must match a serial one modulo ``meta.wall_seconds``.
    """
    import multiprocessing

    rows: List[Dict[str, Any]] = []
    skips: Counter = Counter()
    skips["files"] = len(files)
    names = [path.name for path in files]
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=workers, initializer=_worker_init,
                      initargs=(list(ranker_names), list(prefixes), prefetching,
                                str(results_dir),
                                os.environ.get("IDEA_TEST_FIXTURES", "replay"),
                                _MIN_SCORE_OVERRIDE)) as pool:
        for cell_rows, keys in pool.imap(_worker_replay, names, chunksize=1):
            rows.extend(cell_rows)
            for key in keys:
                skips[key] += 1
    return rows, skips


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------

def _rate(num: int, den: int) -> Optional[float]:
    return None if den <= 0 else num / den


def availability(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Reason histogram plus ``computed`` rate over ALL rows given (the stratum denominator)."""
    by_reason = Counter(str(r.get("reason")) for r in rows)
    computed = by_reason.get("computed", 0)
    return {"n": len(rows), "computed": computed, "rate": _rate(computed, len(rows)),
            "by_reason": dict(sorted(by_reason.items()))}


def value_correct(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """``host_value_correct`` among COMPUTED rows, plus the argmax entity split.

    ``n`` is the whole stratum, ``computed`` the availability numerator and ``assessed`` the rows
    where the task module supplied ground truth -- three different denominators, all reported, so
    a rate can never be read against the wrong one.
    """
    computed = [r for r in rows if r.get("reason") == "computed"]
    assessed = [r for r in computed if r.get("value_correct") is not None]
    correct = [r for r in assessed if r.get("value_correct")]
    argmax = [r for r in assessed if r.get("value_kind") == "argmax"]
    argmax_correct = [r for r in argmax if r.get("value_correct")]
    close = [r for r in computed if r.get("value_close") is True]
    return {
        "n": len(rows), "computed": len(computed), "assessed": len(assessed),
        "correct": len(correct), "rate": _rate(len(correct), len(assessed)),
        "argmax_assessed": len(argmax), "argmax_correct": len(argmax_correct),
        "argmax_rate": _rate(len(argmax_correct), len(argmax)),
        "value_close": len(close),
    }


def operating_point(rows: Sequence[Dict[str, Any]], signal: str) -> Dict[str, Any]:
    """Coverage / risk / exact-binomial upper bound for one accept signal.

    Denominator is ``len(rows)`` -- ALL cells in the stratum -- per the plan's standing rule that
    conditional-on-available coverage is never an endpoint. ``risk`` is the wrong-answer rate
    AMONG ACCEPTED cells (``wrong_05``: overall_score < 0.5), and ``risk_cp_upper`` its
    Clopper-Pearson 95% upper limit, which is what makes a 0/3 accept read as "risk <= 71%"
    rather than "0% risk".
    """
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal: {signal}")
    accepted = [r for r in rows if accepts(r, signal)]
    wrong = [r for r in accepted if r.get("wrong_05")]
    return {
        "signal": signal, "n": len(rows), "accepted": len(accepted), "wrong": len(wrong),
        "coverage": _rate(len(accepted), len(rows)),
        "risk": _rate(len(wrong), len(accepted)),
        "risk_cp_upper": LRC.clopper_pearson_upper(len(wrong), len(accepted)),
    }


def accepts(row: Dict[str, Any], signal: str) -> bool:
    """Whether ``signal`` accepts ``row``. ``union``/``intersection`` combine host and chain."""
    host = bool(row.get("host_certified"))
    chain = bool(row.get("certified"))
    if signal == "host_certified":
        return host
    if signal == "availability_only":
        return bool(row.get("availability_only"))
    if signal == "chain_certified":
        return chain
    if signal == "union":
        return host or chain
    if signal == "intersection":
        return host and chain
    raise ValueError(f"unknown signal: {signal}")


def host_vs_chain(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Every :data:`SIGNALS` operating point on one stratum."""
    return {signal: operating_point(rows, signal) for signal in SIGNALS}


def _group(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key))].append(row)
    return dict(sorted(out.items()))


def _by_ranker(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return _group(rows, "ranker")


def forensics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every computed row whose value is WRONG, with the operands it read.

    These are the rows a mechanism fix has to explain: the host was confident enough to mint a
    number and the number disagrees with the task module's ground truth.
    """
    out = []
    for row in rows:
        if row.get("reason") == "computed" and row.get("value_correct") is False:
            out.append({
                "file": row["file"], "ranker": row["ranker"], "test_id": row["test_id"],
                "model": row["model"], "host": row["host"],
                "page_source": row.get("page_source"), "value": row.get("value"),
                "unit": row.get("unit"), "winner_entity": row.get("winner_entity"),
                "slots": [
                    {"index": s.get("index"), "entity": s.get("entity"),
                     "field_phrase": s.get("field_phrase"), "url": s.get("url"),
                     "reason": s.get("reason"), "score": s.get("score"),
                     "entry": s.get("entry")}
                    for s in (row.get("slots") or [])
                ],
            })
    return out


_PARENTHETICAL_RE = re.compile(r"\(([^()]*)\)")


def _significant(entity: Any) -> set:
    """The entity's identifying tokens -- the toolkit's own rule when importable."""
    try:
        from agent.app.ledger_tools import _significant_tokens  # re-exported from operand_attribution
    except Exception:  # noqa: BLE001
        from agent.app.operand_attribution import _significant_tokens
    return set(_significant_tokens(entity))


def _slug_of(url: str) -> str:
    from urllib.parse import unquote, urlparse
    return unquote(urlparse(str(url or "")).path).rsplit("/", 1)[-1].replace("_", " ")


def slug_qualifiers(url: str) -> List[str]:
    """Parenthetical qualifiers in the URL slug, e.g. ``["1974–2001"]`` for
    ``.../Foo_(1974–2001)``: a disambiguated article the entity name itself did not ask for."""
    return [q.strip() for q in _PARENTHETICAL_RE.findall(_slug_of(url)) if q.strip()]


def prefetched_wrong_page(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Selected slots that landed on a PREFETCHED page whose URL slug does not name the slot's
    entity -- the prefetcher fetched *a* page, not necessarily the entity's page.

    Flagged unless the slug covers ALL of the entity's significant tokens: partial coverage is
    exactly the failure seen live ("Burj_Azizi" for "Burj Khalifa" shares "burj",
    "Jin_Mao_Tower" for "Shanghai Tower" shares "tower"). Also flagged when the slug carries a
    parenthetical qualifier the entity does not ("Foo_(1974–2001)" for "Foo"): that is a
    disambiguated sibling article, not the entity's own. Each entry names the entity, the URL
    and the slot's field phrase so the fix can be aimed at the resolver query.
    """
    out = []
    for row in rows:
        prefetched_ids = set(row.get("prefetched_page_ids") or [])
        if not prefetched_ids:
            continue
        for slot in row.get("slots") or []:
            if slot.get("reason") != "selected" or str(slot.get("page_id")) not in prefetched_ids:
                continue
            entity = slot.get("entity")
            url = str(slot.get("url") or "")
            cover = slug_coverage(entity, url)
            entity_qualifiers = {q.strip().lower() for q in _PARENTHETICAL_RE.findall(str(entity or ""))}
            extra_qualifiers = [q for q in slug_qualifiers(url) if q.lower() not in entity_qualifiers]
            full_cover = cover is None or cover >= 1.0
            if full_cover and not extra_qualifiers:
                continue
            out.append({"file": row["file"], "ranker": row["ranker"], "test_id": row["test_id"],
                        "model": row["model"], "host": row["host"],
                        "entity": entity, "field_phrase": slot.get("field_phrase"), "url": url,
                        "page_id": slot.get("page_id"), "slug_coverage": cover,
                        "slug_qualifiers": extra_qualifiers,
                        "flag": ("partial_slug_coverage" if not full_cover
                                 else "slug_qualifier_not_in_entity"),
                        "value_correct": row.get("value_correct"),
                        "reason": row.get("reason")})
    return out


def build_summary(rows: Sequence[Dict[str, Any]], *, skips: Counter, prefixes: Sequence[str],
                  ranker_names: Sequence[str], results_dir: str, out_dir: str,
                  wall_seconds: float, prefetch: bool = False) -> Dict[str, Any]:
    """Every table the report prints, as plain JSON (the gate artifact for Phase 3).

    ``prefetch`` adds the ``--prefetch`` tables (both denominators, the ``by_prefetch_used``
    splits and the ``prefetched_wrong_page`` forensics); when False the summary is byte-identical
    to the pre-prefetch script's.
    """
    by_ranker = _by_ranker(rows)
    dev = [r for r in rows if r.get("split") == "dev"]
    holdout = [r for r in rows if r.get("split") == "holdout"]

    def points(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {name: host_vs_chain(sub) for name, sub in _by_ranker(subset).items()}

    dev_derive = [r for r in dev if r.get("arm") == "derive"]
    holdout_derive = [r for r in holdout if r.get("arm") == "derive"]

    per_model = {}
    for name, sub in _by_ranker(dev_derive).items():
        per_model[name] = {model: host_vs_chain(msub)
                           for model, msub in _group(sub, "model").items()}

    # Host stratification is a §15 requirement, not a nicety: before the page-source fix every
    # replayed row was langgraph_react, so a pooled number and a langgraph number were the same
    # number. Now that sequential_react rows exist, every headline has to be readable per host.
    per_host = {name: {host: host_vs_chain(hsub) for host, hsub in _group(sub, "host").items()}
                for name, sub in _by_ranker(dev_derive).items()}
    per_host_all = {name: {host: host_vs_chain(hsub)
                           for host, hsub in _group(sub, "host").items()}
                    for name, sub in _by_ranker(rows).items()}

    summary = {
        "meta": {
            "prefixes": list(prefixes), "rankers": list(ranker_names),
            "results_dir": results_dir, "out_dir": out_dir,
            "wall_seconds": round(float(wall_seconds), 2),
            "rel_tol": LRC.REL_TOL,
            "holdout_test_ids": sorted(LRC.HOLDOUT_TEST_IDS),
            "min_score": {name: _min_score_for(name) for name in ranker_names},
        },
        "counts": {
            "files": skips.get("files", 0), "replayed": skips.get("replayed", 0),
            "rows": len(rows),
            "skipped": {k: v for k, v in sorted(skips.items())
                        if k in ("infra_failed", "no_run_config", "no_pages", "unreadable")},
            # Which storage location each replayed cell's pages came from. Before the §15
            # correction this was implicitly {"output": everything}: evidence_graph-only cells
            # were counted as ``no_pages`` and never replayed at all.
            "page_source": {source: skips.get(f"pages_{source}", 0)
                            for source in PAGE_SOURCES if source != "none"},
            "by_prefix": {
                prefix: {key: skips.get(f"{prefix}|{key}", 0)
                         for key in ("files", "replayed", "infra_failed", "no_run_config",
                                     "no_pages", "unreadable", "pages_output",
                                     "pages_evidence_graph")}
                for prefix in prefixes
            },
        },
        "availability": {
            "pooled": {name: availability(sub) for name, sub in by_ranker.items()},
            "by_arm": {name: {arm: availability(sub2)
                              for arm, sub2 in _group(sub, "arm").items()}
                       for name, sub in by_ranker.items()},
            "by_model": {name: {model: availability(sub2)
                                for model, sub2 in _group(sub, "model").items()}
                         for name, sub in by_ranker.items()},
            "by_test_id": {name: {tid: availability(sub2)
                                  for tid, sub2 in _group(sub, "test_id").items()}
                           for name, sub in by_ranker.items()},
            "by_host": {name: {host: availability(sub2)
                               for host, sub2 in _group(sub, "host").items()}
                        for name, sub in by_ranker.items()},
            "by_page_source": {name: {src: availability(sub2)
                                      for src, sub2 in _group(sub, "page_source").items()}
                               for name, sub in by_ranker.items()},
        },
        "value_correct": {
            "pooled": {name: value_correct(sub) for name, sub in by_ranker.items()},
            "by_model": {name: {model: value_correct(sub2)
                                for model, sub2 in _group(sub, "model").items()}
                         for name, sub in by_ranker.items()},
            "by_test_id": {name: {tid: value_correct(sub2)
                                  for tid, sub2 in _group(sub, "test_id").items()}
                           for name, sub in by_ranker.items()},
            "by_host": {name: {host: value_correct(sub2)
                               for host, sub2 in _group(sub, "host").items()}
                        for name, sub in by_ranker.items()},
        },
        "operating_points": {
            "dev_derive_on": points(dev_derive),
            "dev_all_arms": points(dev),
            "holdout_derive_on": points(holdout_derive),
            "holdout_all_arms": points(holdout),
            "dev_derive_on_by_model": per_model,
            "dev_derive_on_by_host": per_host,
            "all_by_host": per_host_all,
        },
        "forensics": forensics(rows),
    }
    if prefetch:
        stored_rows = [r for r in rows if r.get("page_source") != PREFETCHED_PAGE_SOURCE]
        summary["meta"]["prefetch"] = True
        summary["meta"]["prefetch_source"] = prefetch_source()
        summary["counts"]["page_source"][PREFETCHED_PAGE_SOURCE] = \
            skips.get(f"pages_{PREFETCHED_PAGE_SOURCE}", 0)
        for prefix in prefixes:
            summary["counts"]["by_prefix"][prefix][f"pages_{PREFETCHED_PAGE_SOURCE}"] = \
                skips.get(f"{prefix}|pages_{PREFETCHED_PAGE_SOURCE}", 0)
        # The two denominators, stated once and named: every rate elsewhere in this summary is
        # over ALL replayed cells (incl. the no_pages bucket); ``stored_page_cells`` is the plain
        # replay's population, so the two runs are comparable on it.
        summary["counts"]["denominators"] = {
            "all_cells_incl_no_pages": skips.get("replayed", 0),
            "stored_page_cells": skips.get("replayed", 0)
                                 - skips.get(f"pages_{PREFETCHED_PAGE_SOURCE}", 0),
            "rows_all": len(rows), "rows_stored_page_cells": len(stored_rows),
        }
        summary["availability"]["by_prefetch_used"] = {
            name: {used: availability(sub2) for used, sub2 in _group(sub, "prefetch_used").items()}
            for name, sub in by_ranker.items()}
        summary["availability"]["stored_page_cells"] = {
            name: availability(sub) for name, sub in _by_ranker(stored_rows).items()}
        summary["value_correct"]["by_prefetch_used"] = {
            name: {used: value_correct(sub2)
                   for used, sub2 in _group(sub, "prefetch_used").items()}
            for name, sub in by_ranker.items()}
        summary["value_correct"]["stored_page_cells"] = {
            name: value_correct(sub) for name, sub in _by_ranker(stored_rows).items()}
        summary["prefetch"] = {
            "rows_prefetch_used": sum(1 for r in rows if r.get("prefetch_used")),
            "cells_with_prefetched_pages": len({r["file"] for r in rows
                                                if r.get("n_prefetched")}),
            "status_counts": _prefetch_status_counts(rows),
            "errors": sorted({str((r.get("prefetch") or {}).get("error"))
                              for r in rows if (r.get("prefetch") or {}).get("error")}),
        }
        summary["prefetched_wrong_page"] = prefetched_wrong_page(rows)
    return summary


def _prefetch_status_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Entity-level ``status`` histogram from the per-cell prefetch summaries (one per cell,
    not per row: the summary is shared across a cell's rankers)."""
    seen = set()
    counts: Counter = Counter()
    for row in rows:
        if row["file"] in seen:
            continue
        seen.add(row["file"])
        for ent in ((row.get("prefetch") or {}).get("entities") or []):
            counts[str(ent.get("status"))] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def _pct(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _point_line(label: str, point: Dict[str, Any]) -> str:
    return (f"  {label:<20}{point['n']:>6}{point['accepted']:>10}{point['wrong']:>7}"
            f"{_pct(point['coverage']):>11}{_pct(point['risk']):>10}"
            f"{_pct(point['risk_cp_upper']):>12}")


_POINT_HEAD = (f"  {'signal':<20}{'n':>6}{'accepted':>10}{'wrong':>7}{'coverage':>11}"
               f"{'risk':>10}{'cp_upper':>12}")


def _availability_block(title: str, table: Dict[str, Dict[str, Any]]) -> List[str]:
    out = [title, f"  {'key':<28}{'n':>6}{'computed':>10}{'rate':>9}  top reasons"]
    for key, row in table.items():
        reasons = ", ".join(f"{k}:{v}" for k, v in sorted(row["by_reason"].items(),
                                                          key=lambda kv: -kv[1])[:4])
        out.append(f"  {key:<28}{row['n']:>6}{row['computed']:>10}{_pct(row['rate']):>9}  "
                   f"{reasons}")
    return out


def format_report(summary: Dict[str, Any]) -> str:
    """The printed Phase-2e report. Every rate's denominator is the full stratum."""
    meta, counts = summary["meta"], summary["counts"]
    out: List[str] = []
    out.append("=" * 100)
    out.append("host_derive replay (Phase 2e) -- offline, $0, no model calls")
    out.append("=" * 100)
    out.append(f"prefixes={','.join(meta['prefixes'])}  rankers={','.join(meta['rankers'])}  "
               f"min_score={meta['min_score']}  wall={meta['wall_seconds']}s")
    out.append(f"files={counts['files']}  replayed={counts['replayed']}  rows={counts['rows']}  "
               f"skipped={counts['skipped']}")
    out.append(f"page_source={counts.get('page_source', {})}  "
               "(evidence_graph = sequential_react cells, invisible before the §15 correction)")
    if meta.get("prefetch"):
        den = counts.get("denominators", {})
        out.append(f"PREFETCH ON (source={meta.get('prefetch_source')}): denominators -- "
                   f"all cells incl. no_pages = {den.get('all_cells_incl_no_pages')} "
                   f"({den.get('rows_all')} rows); stored-page cells = "
                   f"{den.get('stored_page_cells')} ({den.get('rows_stored_page_cells')} rows). "
                   "Every rate below is over ALL cells unless the block says stored_page_cells.")
    out.append(f"  {'campaign':<16}{'files':>7}{'replayed':>10}{'src:output':>12}"
               f"{'src:ev_graph':>14}{'no_pages':>10}{'infra_failed':>14}{'no_run_config':>15}")
    for prefix, row in counts.get("by_prefix", {}).items():
        out.append(f"  {prefix:<16}{row['files']:>7}{row['replayed']:>10}"
                   f"{row.get('pages_output', 0):>12}{row.get('pages_evidence_graph', 0):>14}"
                   f"{row['no_pages']:>10}{row['infra_failed']:>14}{row['no_run_config']:>15}")

    out.append("")
    out.append("(a) AVAILABILITY -- denominator is every replayed cell in the stratum")
    for name, row in summary["availability"]["pooled"].items():
        out.append(f"  pooled[{name}]: {row['computed']}/{row['n']} computed "
                   f"({_pct(row['rate']).strip()})")
        for reason, n in sorted(row["by_reason"].items(), key=lambda kv: -kv[1]):
            out.append(f"      {reason:<36}{n:>6}{_pct(_rate(n, row['n'])):>9}")
    for name in meta["rankers"]:
        out.append("")
        out += _availability_block(f"  availability by arm [{name}]",
                                   summary["availability"]["by_arm"].get(name, {}))
        out += _availability_block(f"  availability by model [{name}]",
                                   summary["availability"]["by_model"].get(name, {}))
        out += _availability_block(f"  availability by test_id [{name}]",
                                   summary["availability"]["by_test_id"].get(name, {}))
        out += _availability_block(f"  availability by host [{name}]",
                                   summary["availability"]["by_host"].get(name, {}))
        out += _availability_block(f"  availability by page_source [{name}]",
                                   summary["availability"]["by_page_source"].get(name, {}))
        if meta.get("prefetch"):
            out += _availability_block(
                f"  availability by prefetch_used [{name}]",
                summary["availability"].get("by_prefetch_used", {}).get(name, {}))
            out += _availability_block(
                f"  availability, stored-page cells only [{name}]",
                {"stored_page_cells": summary["availability"]["stored_page_cells"][name]}
                if name in summary["availability"].get("stored_page_cells", {}) else {})

    out.append("")
    out.append("(b) host_value_correct AMONG COMPUTED (primary metric)")
    for name in meta["rankers"]:
        pooled = summary["value_correct"]["pooled"].get(name, {})
        out.append(f"  pooled[{name}]: correct {pooled.get('correct')}/"
                   f"{pooled.get('assessed')} assessed "
                   f"({_pct(pooled.get('rate')).strip()}); computed={pooled.get('computed')} "
                   f"of n={pooled.get('n')}; argmax entity-correct "
                   f"{pooled.get('argmax_correct')}/{pooled.get('argmax_assessed')} "
                   f"({_pct(pooled.get('argmax_rate')).strip()})")
        splits = [("by model", "by_model"), ("by test_id", "by_test_id"), ("by host", "by_host")]
        if meta.get("prefetch"):
            splits.append(("by prefetch_used", "by_prefetch_used"))
        for label, key in splits:
            out.append(f"  {label} [{name}]")
            out.append(f"    {'key':<28}{'n':>6}{'computed':>10}{'assessed':>10}"
                       f"{'correct':>9}{'rate':>9}")
            for key2, row in summary["value_correct"][key].get(name, {}).items():
                out.append(f"    {key2:<28}{row['n']:>6}{row['computed']:>10}"
                           f"{row['assessed']:>10}{row['correct']:>9}{_pct(row['rate']):>9}")

    out.append("")
    out.append("(c) OPERATING POINTS -- dev split (the plan's 2e gate stratum)")
    for stratum in ("dev_derive_on", "dev_all_arms"):
        for name in meta["rankers"]:
            table = summary["operating_points"][stratum].get(name)
            if not table:
                continue
            out.append(f"  host_vs_chain [{stratum} | {name}]")
            out.append(_POINT_HEAD)
            for signal in SIGNALS:
                out.append(_point_line(signal, table[signal]))

    out.append("")
    out.append("(d) SEALED READOUT ONLY -- holdout split (213/217/221). Reported, not decided on.")
    for stratum in ("holdout_derive_on", "holdout_all_arms"):
        for name in meta["rankers"]:
            table = summary["operating_points"][stratum].get(name)
            if not table:
                continue
            out.append(f"  host_vs_chain [{stratum} | {name}]")
            out.append(_POINT_HEAD)
            for signal in SIGNALS:
                out.append(_point_line(signal, table[signal]))

    out.append("")
    out.append("(e) PER-MODEL operating points [dev, derive-on]")
    for name in meta["rankers"]:
        for model, table in summary["operating_points"]["dev_derive_on_by_model"] \
                .get(name, {}).items():
            out.append(f"  {model} [{name}]")
            out.append(_POINT_HEAD)
            for signal in SIGNALS:
                out.append(_point_line(signal, table[signal]))

    out.append("")
    out.append("(f) NEGATIVE CONTROLS side by side [dev, derive-on]")
    out.append(f"  {'ranker':<16}{'signal':<20}{'n':>6}{'accepted':>10}{'coverage':>11}"
               f"{'risk':>10}{'cp_upper':>12}")
    for name in meta["rankers"]:
        table = summary["operating_points"]["dev_derive_on"].get(name, {})
        for signal in ("host_certified", "availability_only"):
            point = table.get(signal)
            if not point:
                continue
            out.append(f"  {name:<16}{signal:<20}{point['n']:>6}{point['accepted']:>10}"
                       f"{_pct(point['coverage']):>11}{_pct(point['risk']):>10}"
                       f"{_pct(point['risk_cp_upper']):>12}")

    out.append("")
    out.append("(g) PER-HOST operating points -- sequential_react vs langgraph_react")
    for stratum in ("dev_derive_on_by_host", "all_by_host"):
        for name in meta["rankers"]:
            for host, table in summary["operating_points"].get(stratum, {}) \
                    .get(name, {}).items():
                out.append(f"  {host} [{stratum} | {name}]")
                out.append(_POINT_HEAD)
                for signal in SIGNALS:
                    out.append(_point_line(signal, table[signal]))

    if meta.get("prefetch"):
        pf = summary.get("prefetch", {})
        out.append("")
        out.append("(h) PREFETCH -- entity statuses across cells (one summary per cell)")
        out.append(f"  cells_with_prefetched_pages={pf.get('cells_with_prefetched_pages')}  "
                   f"rows_prefetch_used={pf.get('rows_prefetch_used')}  "
                   f"status_counts={pf.get('status_counts')}")
        if pf.get("errors"):
            out.append(f"  errors ({len(pf['errors'])} distinct): "
                       + "; ".join(pf["errors"][:FORENSICS_PRINT_CAP]))
        wrong = summary.get("prefetched_wrong_page", [])
        out.append(f"  prefetched_wrong_page ({len(wrong)} selected slots on a prefetched page "
                   f"whose slug does not fully name the entity or carries a foreign "
                   f"qualifier; first "
                   f"{min(len(wrong), FORENSICS_PRINT_CAP)} shown)")
        for row in wrong[:FORENSICS_PRINT_CAP]:
            cover = row["slug_coverage"]
            out.append(f"    {row['test_id']} {row['ranker']:<14}{row['model']:<16}"
                       f"{row['flag']} entity={row['entity']!r} "
                       f"field={row['field_phrase']!r} "
                       f"slug_cover={'n/a' if cover is None else f'{cover:.2f}'} "
                       f"qualifiers={row['slug_qualifiers']} "
                       f"value_correct={row['value_correct']} url={row['url']}")

    rows = summary["forensics"]
    out.append("")
    out.append(f"FORENSICS -- computed but value_correct=False ({len(rows)} rows; "
               f"first {min(len(rows), FORENSICS_PRINT_CAP)} shown, all in rows.jsonl)")
    for row in rows[:FORENSICS_PRINT_CAP]:
        slots = "; ".join(
            f"{s['entity']}<-{(s.get('entry') or {}).get('label', '')}="
            f"{(s.get('entry') or {}).get('value', '')}"
            f"{(s.get('entry') or {}).get('unit', '')}[{s.get('reason')}]"
            for s in row["slots"]
        )
        out.append(f"  {row['test_id']} {row['ranker']:<14}{row['model']:<16}"
                   f"{str(row.get('host')):<18}value={row['value']}{row['unit']} "
                   f"winner={row['winner_entity']} | {slots}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def write_outputs(out_dir: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any],
                  report: str) -> Dict[str, Path]:
    """Write ``rows.jsonl`` / ``summary.json`` / ``report.txt``; returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"rows": out_dir / "rows.jsonl", "summary": out_dir / "summary.json",
             "report": out_dir / "report.txt"}
    with open(paths["rows"], "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True, default=str),
                                encoding="utf-8")
    paths["report"].write_text(report + "\n", encoding="utf-8")
    return paths


def _force_replay_fixtures() -> None:
    """Force ``IDEA_TEST_FIXTURES=replay`` unless it is already an explicit mode.

    ``record`` never loads the cache and would re-bill, so it is never forced -- but an explicit
    setting is always respected. Shared by the serial and parallel prefetch paths so both make
    the same promise about spend.
    """
    mode = (os.environ.get("IDEA_TEST_FIXTURES") or "").strip().lower()
    if mode not in ("record", "replay", "replay_strict"):
        os.environ["IDEA_TEST_FIXTURES"] = "replay"
        print("IDEA_TEST_FIXTURES not set to record/replay[_strict]; forcing 'replay' for "
              "--prefetch (load-or-fetch-and-save).", file=sys.stderr)


def _replay_with_prefetch(files: Sequence[Path], ranker_names: Sequence[str],
                          prefixes: Sequence[str]) -> Tuple[List[Dict[str, Any]], Counter]:
    """:func:`replay` behind live connectors and one event loop for the whole run.

    Fixture mode follows ``scripts/prewarm_fixtures.py``: ``replay`` is forced only when
    ``IDEA_TEST_FIXTURES`` is unset/unknown. ``record`` never loads the cache and would re-bill,
    so it is never forced -- but an explicit setting is respected. Connectors are the prewarm
    set minus LLM/Chroma/AgentIO: ``host_prefetch`` reads the mandate and pages, never a model.
    """
    _force_replay_fixtures()
    from shared.connector_config import ConnectorConfig
    from agent.app.connector_http import ConnectorHttp
    from agent.app.connector_search import create_search_backend

    config = ConnectorConfig()
    search = create_search_backend(config)  # raises on an unknown provider: no silent paid path
    http = ConnectorHttp(config)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        prefetcher = make_live_prefetcher(loop, http=http, search=search)
        return replay(files, ranker_names, prefixes, prefetcher=prefetcher)
    finally:
        try:
            loop.run_until_complete(http.__aexit__(None, None, None))
        except Exception:  # noqa: BLE001 -- session teardown must not mask the result
            pass
        asyncio.set_event_loop(None)
        loop.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefixes", default=DEFAULT_PREFIXES,
                    help=f"comma-separated campaign prefixes (default: {DEFAULT_PREFIXES})")
    ap.add_argument("--results-dir", default=None,
                    help="stored-cell directory (default: the main tree's idea_test_results)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--rankers", default=",".join(RANKERS),
                    help=f"comma-separated ranker names from {sorted(RANKERS)}")
    ap.add_argument("--limit", type=int, default=0,
                    help="replay only the first N cell files (smoke runs); 0 = all")
    ap.add_argument("--min-score", type=float, default=None,
                    help="override the hand rule's 0.93 floor (exploratory availability/"
                         "correctness sweep; the control ranker keeps its own 0.0)")
    ap.add_argument("--workers", type=int, default=1,
                    help="replay cells across N processes (default 1 = serial). Cells are "
                         "independent, so this is a pure speedup: --workers N must produce the "
                         "same summary as --workers 1, modulo meta.wall_seconds.")
    ap.add_argument("--prefetch", action="store_true",
                    help="run agent.app.host_prefetch once per cell (live search + visits via the "
                         "web-fixture cache; IDEA_TEST_FIXTURES forced to 'replay' when unset) "
                         "and replay no_pages cells too")
    args = ap.parse_args(argv)

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    ranker_names = [r.strip() for r in args.rankers.split(",") if r.strip()]
    unknown = [r for r in ranker_names if r not in RANKERS]
    if unknown:
        print(f"unknown ranker(s): {unknown}; known: {sorted(RANKERS)}", file=sys.stderr)
        return 2
    results_dir = Path(args.results_dir) if args.results_dir else LRC._default_results_dir()

    files = LRC.discover_cell_files(results_dir, ",".join(prefixes))
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    started = time.time()
    global _MIN_SCORE_OVERRIDE
    _MIN_SCORE_OVERRIDE = args.min_score
    workers = max(1, int(args.workers))
    if workers > 1:
        if args.prefetch:
            _force_replay_fixtures()
        rows, skips = replay_parallel(files, ranker_names, prefixes, workers=workers,
                                      prefetching=bool(args.prefetch), results_dir=results_dir)
    elif args.prefetch:
        rows, skips = _replay_with_prefetch(files, ranker_names, prefixes)
    else:
        rows, skips = replay(files, ranker_names, prefixes)
    wall = time.time() - started

    summary = build_summary(rows, skips=skips, prefixes=prefixes, ranker_names=ranker_names,
                            results_dir=str(results_dir), out_dir=str(args.out_dir),
                            wall_seconds=wall, prefetch=bool(args.prefetch))
    report = format_report(summary)
    paths = write_outputs(Path(args.out_dir), rows, summary, report)
    print(report)
    print("")
    for key, path in paths.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
