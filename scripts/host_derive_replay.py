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

Outputs (under ``--out-dir``): ``rows.jsonl`` (one row per (cell, ranker), the full record
including per-slot operand choices), ``summary.json`` (every table the report prints), and
``report.txt`` (the printed report). Deterministic: files are processed in sorted order and
nothing here samples.
"""
from __future__ import annotations

import argparse
import json
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


def skip_reason(raw: Dict[str, Any]) -> Optional[str]:
    """Why this cell cannot be replayed, or ``None`` when it can.

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
    if not cell_pages(raw)[0]:
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


def _min_score_for(name: str) -> float:
    from agent.app.ledger_tools import _HOST_DERIVE_MIN_SCORE
    floor = RANKERS[name][1]
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


def replay_cell(path: Path, raw: Dict[str, Any], ranker_names: Sequence[str],
                prefixes: Sequence[str]) -> List[Dict[str, Any]]:
    """One JSONL row per ranker for a single cell (never raises).

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
        if statement is None:
            rows.append({**base, "reason": "no_task_module", "value": None, "unit": "",
                         "winner_entity": None, "agrees_final": False, "agrees_any": False,
                         "agrees_entity": None, "unit_status": "unassessed",
                         "value_correct": None, "value_close": None, "host_certified": False,
                         "availability_only": False, "n_entries": 0, "slot_reasons": [],
                         "slots": []})
            continue
        toolkit = LedgerToolkit()
        for page in pages:
            toolkit.register_page(str(page.get("url") or ""), str(page.get("text") or ""))
        hd = toolkit.host_derive(statement, ranker=_make_ranker(ranker_name),
                                 min_score=_min_score_for(ranker_name))
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


def replay(files: Sequence[Path], ranker_names: Sequence[str],
           prefixes: Sequence[str]) -> Tuple[List[Dict[str, Any]], Counter]:
    """Replay every file; returns ``(rows, skip_counter)``.

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
        prefix = campaign_prefix(path.name, prefixes) or "?"
        skips[f"{prefix}|files"] += 1
        raw = LRC.load_cell(path)
        if raw is None:
            skips["unreadable"] += 1
            skips[f"{prefix}|unreadable"] += 1
            continue
        reason = skip_reason(raw)
        if reason is not None:
            skips[reason] += 1
            skips[f"{prefix}|{reason}"] += 1
            continue
        skips["replayed"] += 1
        skips[f"{prefix}|replayed"] += 1
        source = cell_pages(raw)[1]
        skips[f"pages_{source}"] += 1
        skips[f"{prefix}|pages_{source}"] += 1
        rows.extend(replay_cell(path, raw, ranker_names, prefixes))
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


def build_summary(rows: Sequence[Dict[str, Any]], *, skips: Counter, prefixes: Sequence[str],
                  ranker_names: Sequence[str], results_dir: str, out_dir: str,
                  wall_seconds: float) -> Dict[str, Any]:
    """Every table the report prints, as plain JSON (the gate artifact for Phase 3)."""
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

    return {
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
        for label, key in (("by model", "by_model"), ("by test_id", "by_test_id"),
                           ("by host", "by_host")):
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
    rows, skips = replay(files, ranker_names, prefixes)
    wall = time.time() - started

    summary = build_summary(rows, skips=skips, prefixes=prefixes, ranker_names=ranker_names,
                            results_dir=str(results_dir), out_dir=str(args.out_dir),
                            wall_seconds=wall)
    report = format_report(summary)
    paths = write_outputs(Path(args.out_dir), rows, summary, report)
    print(report)
    print("")
    for key, path in paths.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
