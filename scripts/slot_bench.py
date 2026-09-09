#!/usr/bin/env python3
"""The slot bench -- ``host_derive``'s mechanism measured over SLOTS instead of over CELLS.

Why this exists
---------------
``host_derive`` is model-invisible: a finish-time hook that reads the mandate and the pages and
never the model's answer. So a campaign cell varies the *pages* a model happened to fetch, but the
*slots* -- ``(entity, field_phrase)`` pairs -- come from the mandate and are identical in every
cell of a task. Measured on the mint04 launch replay: 261 forensic rows collapsed to 24 distinct
``(task, entity, field_phrase)`` problems, and the 859-cell ``--prefetch`` replay spent 47 minutes
to answer them.

This bench asks the same mechanism questions directly. For each tier-5 task it builds an EMPTY
toolkit, lets ``host_prefetch`` supply the pages (so page reach is the host's own, not a model's),
runs the real ``host_derive`` over the real mandate, and scores the result with the same
``ledger_risk_coverage.host_value_correct_detail`` the campaign KPI uses. 12 tasks, 36 slots, one
resolution each, all served from the warm web-fixture cache.

What it is NOT
--------------
It is not a replacement for ``scripts/host_derive_replay.py``. The replay measures the mechanism
over pages a MODEL chose, carries the ``document_order`` loosening control, and reports the
availability/correctness numbers a campaign endpoint is written against. Those remain the gate
before any live campaign. This bench is the edit loop that runs between them.

Correctness is never traded here either: a task whose host computes a WRONG value is a hard
failure (non-zero exit) no matter what happens to availability.

Usage (from the repo root)::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/slot_bench.py
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/slot_bench.py --tests 210,219
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/slot_bench.py --json out.json
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import host_derive_replay as HDR  # noqa: E402
from scripts import ledger_risk_coverage as LRC  # noqa: E402

#: The tier-5 derived-arithmetic suite. 210-217 are two-operand; 218-221 are 5-entity argmax.
DEFAULT_TESTS = ["210", "211", "212", "213", "214", "215", "216", "217",
                 "218", "219", "220", "221"]

#: Which key of an argmax task's ``ENTITIES`` row holds each operand, in the mandate's own order
#: ("TWO numbers ... the LENGTH ... and its DRAINAGE BASIN SIZE"). Written out per task rather
#: than guessed from units, because two operands of one slot can share a unit (220: both metres).
#: Diagnostics only -- the pass/fail verdict comes from the module's DERIVED/WINNER ground truth.
ARGMAX_OPERAND_KEYS: Dict[str, Sequence[str]] = {
    "218": ("length_km", "basin_km2"),
    "219": ("height_m", "width_m"),
    "220": ("total_m", "span_m"),
    "221": ("height_m", "floors"),
}


def task_module(test_id: str) -> Any:
    """The real task module for ``test_id``, imported the way the replay imports it."""
    matches = sorted(glob.glob(str(_ROOT / "agent" / "app" / "idea_tests"
                                   / f"test_{test_id}_*.py")))
    if not matches:
        return None
    return importlib.import_module("agent.app.idea_tests." + Path(matches[0]).stem)


def expected_operands(module: Any, test_id: str) -> Dict[str, List[Any]]:
    """``{entity_name: [expected operand values]}`` straight out of the task module.

    Two-operand tasks export ``OP_A``/``OP_B`` (whose ``label`` may carry a parenthetical the
    mandate's slot does not, so entities are keyed by the module's own label and matched loosely
    downstream). Argmax tasks export ``ENTITIES`` with per-field keys named in
    :data:`ARGMAX_OPERAND_KEYS`. Missing ground truth is an empty dict, never an exception: this
    is a diagnostic column, not the verdict.
    """
    out: Dict[str, List[Any]] = {}
    try:
        entities = getattr(module, "ENTITIES", None)
        if entities:
            keys = ARGMAX_OPERAND_KEYS.get(test_id, ())
            for row in entities:
                values = [row[k] for k in keys if k in row]
                if values:
                    out[str(row.get("name") or "")] = values
            return out
        for name in ("OP_A", "OP_B"):
            op = getattr(module, name, None)
            if isinstance(op, dict) and "value" in op:
                out.setdefault(str(op.get("label") or name), []).append(op["value"])
    except Exception:  # noqa: BLE001 -- a diagnostic must never take the bench down
        return {}
    return out


def bench_task(test_id: str, *, ranker_name: str, prefetcher: Any) -> Dict[str, Any]:
    """Prefetch and derive one task from an empty toolkit; never raises.

    The toolkit starts EMPTY on purpose. In a campaign cell the host inherits whatever pages the
    model fetched, which flatters or starves the mechanism depending on the model; here its reach
    is entirely its own, which is the property the availability drive is trying to move.
    """
    from agent.app.ledger_tools import LedgerToolkit

    statement = HDR.task_statement(test_id)
    if statement is None:
        return {"test_id": test_id, "error": "no_task_module"}

    module = task_module(test_id)
    started = time.perf_counter()
    toolkit = LedgerToolkit()
    prefetch_summary, prefetched = HDR.run_prefetch(toolkit, statement, prefetcher)
    for url, text, entries in prefetched:
        HDR.register_page_compat(toolkit, url, text, source=HDR.prefetch_source(),
                                 max_chars=len(text), structured=entries)
    hd = toolkit.host_derive(statement, ranker=HDR._make_ranker(ranker_name),
                             min_score=HDR._min_score_for(ranker_name))
    detail = LRC.host_value_correct_detail(hd, test_id)

    slots = []
    for slot in (hd.get("slots") or []):
        entry = slot.get("entry") or {}
        slots.append({
            "entity": slot.get("entity"),
            "field_phrase": slot.get("field_phrase"),
            "reason": slot.get("reason"),
            "score": slot.get("score"),
            "label": entry.get("label"),
            "section": entry.get("section"),
            "source": entry.get("source"),
            "value": entry.get("value"),
            "unit": entry.get("unit"),
            "url": slot.get("url"),
        })
    return {
        "test_id": test_id,
        "ranker": ranker_name,
        "reason": hd.get("reason"),
        "operation": hd.get("operation"),
        "value": hd.get("value"),
        "unit": hd.get("unit"),
        "winner_entity": hd.get("winner_entity"),
        "value_correct": detail["correct"],
        "value_kind": detail["kind"],
        "computed": hd.get("reason") == "computed",
        "n_entries": hd.get("n_entries"),
        "expected_operands": expected_operands(module, test_id) if module else {},
        "prefetch_status": [e.get("status") for e in (prefetch_summary.get("entities") or [])],
        "prefetch_urls": [e.get("url") for e in (prefetch_summary.get("entities") or [])],
        "slots": slots,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def verdict(row: Dict[str, Any]) -> str:
    """``WRONG`` / ``computed`` / ``refused(<reason>)`` -- ``WRONG`` dominates everything."""
    if row.get("error"):
        return f"error({row['error']})"
    if row.get("computed") and row.get("value_correct") is False:
        return "WRONG"
    if row.get("computed"):
        return "computed" if row.get("value_correct") else "computed(unscored)"
    return f"refused({row.get('reason')})"


def format_report(rows: Sequence[Dict[str, Any]], wall: float) -> str:
    out: List[str] = []
    computed = [r for r in rows if r.get("computed")]
    wrong = [r for r in rows if verdict(r) == "WRONG"]
    out.append("=" * 96)
    out.append(f"SLOT BENCH -- {len(rows)} tasks, "
               f"{sum(len(r.get('slots') or []) for r in rows)} slots, {wall:.1f}s")
    out.append("=" * 96)
    out.append(f"  availability : {len(computed)}/{len(rows)} tasks computed")
    out.append(f"  correctness  : {sum(1 for r in computed if r.get('value_correct'))}"
               f"/{len(computed)} of computed correct"
               + ("   *** WRONG VALUES PRESENT ***" if wrong else ""))
    out.append("")
    for row in rows:
        mark = verdict(row)
        head = (f"  {row['test_id']}  {mark:<34} value={row.get('value')} "
                f"{row.get('unit') or ''}")
        if row.get("winner_entity"):
            head += f" winner={row['winner_entity']}"
        out.append(head)
        for slot in (row.get("slots") or []):
            score = slot.get("score")
            score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            label = slot.get("label") or "(no label)"
            section = f"[{slot['section']}]" if slot.get("section") else ""
            out.append(f"       {str(slot.get('entity'))[:28]:<28} "
                       f"{str(slot.get('reason')):<16} score={score_s:<7} "
                       f"{section}{label!r}={slot.get('value')}{slot.get('unit') or ''}")
        expected = row.get("expected_operands") or {}
        if expected and not row.get("computed"):
            out.append(f"       expected: {expected}")
    out.append("")
    return "\n".join(out)


_POOL: Dict[str, Any] = {}


def _pool_init(ranker_name: str, fixtures_mode: str) -> None:
    """One event loop and one set of connectors per worker process (neither is picklable)."""
    os.environ["IDEA_TEST_FIXTURES"] = fixtures_mode
    from shared.connector_config import ConnectorConfig
    from agent.app.connector_http import ConnectorHttp
    from agent.app.connector_search import create_search_backend

    config = ConnectorConfig()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    http = ConnectorHttp(config)
    _POOL.update(ranker_name=ranker_name, loop=loop, http=http,
                 prefetcher=HDR.make_live_prefetcher(loop, http=http,
                                                     search=create_search_backend(config)))


def _pool_bench(test_id: str) -> Dict[str, Any]:
    return bench_task(test_id, ranker_name=_POOL["ranker_name"], prefetcher=_POOL["prefetcher"])


def run_bench(test_ids: Sequence[str], *, ranker_name: str, workers: int) -> List[Dict[str, Any]]:
    """Bench every task, in ``test_ids`` order, across ``workers`` processes.

    Tasks share nothing -- each builds its own toolkit and prefetches its own pages -- so this is
    a pure speedup and ``--workers 1`` stays the serial path exactly. Results are returned in
    submission order (``Pool.imap``), so the report reads the same either way.
    """
    if workers <= 1:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            from shared.connector_config import ConnectorConfig
            from agent.app.connector_http import ConnectorHttp
            from agent.app.connector_search import create_search_backend

            config = ConnectorConfig()
            http = ConnectorHttp(config)
            prefetcher = HDR.make_live_prefetcher(loop, http=http,
                                                  search=create_search_backend(config))
            return [bench_task(t, ranker_name=ranker_name, prefetcher=prefetcher)
                    for t in test_ids]
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=min(workers, len(test_ids)), initializer=_pool_init,
                      initargs=(ranker_name, os.environ.get("IDEA_TEST_FIXTURES", "replay"))
                      ) as pool:
        return list(pool.imap(_pool_bench, list(test_ids)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tests", default=",".join(DEFAULT_TESTS),
                    help=f"comma-separated test ids (default: {','.join(DEFAULT_TESTS)})")
    ap.add_argument("--ranker", default="hand_rule",
                    help=f"ranker name from {sorted(HDR.RANKERS)} (default: hand_rule)")
    ap.add_argument("--json", default=None, help="also write the rows as JSON to this path")
    ap.add_argument("--workers", type=int, default=0,
                    help="bench tasks across N processes (default: one per task). 1 = serial.")
    args = ap.parse_args(argv)

    if args.ranker not in HDR.RANKERS:
        print(f"unknown ranker {args.ranker!r}; known: {sorted(HDR.RANKERS)}", file=sys.stderr)
        return 2
    test_ids = [t.strip() for t in args.tests.split(",") if t.strip()]

    # Same fixture contract as the replay: `record` would re-bill, so it is never forced, but an
    # explicit setting is respected.
    mode = (os.environ.get("IDEA_TEST_FIXTURES") or "").strip().lower()
    if mode not in ("record", "replay", "replay_strict"):
        os.environ["IDEA_TEST_FIXTURES"] = "replay"
        print("IDEA_TEST_FIXTURES not set; forcing 'replay' (load-or-fetch-and-save).",
              file=sys.stderr)

    workers = args.workers if args.workers and args.workers > 0 else len(test_ids)
    started = time.time()
    rows = run_bench(test_ids, ranker_name=args.ranker, workers=workers)
    wall = time.time() - started

    print(format_report(rows, wall))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.json}")

    wrong = [r for r in rows if verdict(r) == "WRONG"]
    if wrong:
        print(f"FAIL: {len(wrong)} task(s) computed a WRONG value: "
              f"{[r['test_id'] for r in wrong]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
