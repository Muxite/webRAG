"""Env-gated retrieval-quality log — was the RETRIEVED template the right one?

No-op unless ``IDEA_TEST_PLAN_LIBRARY_RETRIEVAL_LOG`` is truthy. One JSONL row per retrieval
attempt (``{run_id}_plan_retrievals.jsonl``), keyed by ``retrieval_id`` (uuid4, minted in
``retrieval.PlanLibrary.retrieve``).

Deliberately a SEPARATE file from ``testing/contract_log.py``, which answers a different
question. "Did the library return the closest template?" and "did the leaf that template
authored deliver its contract?" are independent failure modes — a perfect retrieval can be
followed by bad execution, and a bad retrieval can still stumble into a right answer. One
merged stream would hide exactly that distinction; two streams join offline on
``retrieval_id`` / ``template_id`` whenever the combined view IS the question.

Every row carries an unfilled ``human_review`` block. ``scripts/review_plan_retrievals.py``
(later) pages through unreviewed rows for labelling, and that labelled set is what re-tunes
``retrieval.py``'s threshold constants against real traffic — the bootstrap eval set
(``scripts/eval_plan_library_retrieval.py``) is only its day-one stand-in.

Built exactly like ``json_telemetry`` / ``contract_log``: pure observability, so it is
env-gated rather than wired into the JSON-settings/typed-config chain reserved for
behavior-changing knobs, it shares ``json_telemetry``'s ``set_task`` so one call in the runner
tags every stream, and every error in here is swallowed — telemetry can never perturb a run.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.app.testing import json_telemetry as _json_telemetry

# plan_library/ -> app/ -> agent/ ; results live at agent/idea_test_results/
_RES = Path(__file__).resolve().parents[2] / "idea_test_results"

ENV_FLAG = "IDEA_TEST_PLAN_LIBRARY_RETRIEVAL_LOG"
ENV_PATH = "IDEA_TEST_PLAN_LIBRARY_RETRIEVAL_LOG_PATH"

#: How many ranked results to keep per row. Only the top few ever drive a decision.
_MAX_RESULTS = 10

#: Review vocabulary. ``bad_top1_should_have_been_<template_id>`` is a prefix, not a literal —
#: the corrective label carries the id the reviewer expected, which is what makes the reviewed
#: set usable as a labelled eval set later.
LABEL_GOOD_TOP1 = "good_top1"
LABEL_BAD_TOP1_PREFIX = "bad_top1_should_have_been_"
LABEL_CORRECTLY_NO_MATCH = "correctly_no_match"
LABEL_FALSE_POSITIVE_APPLIED = "false_positive_applied"

# Same global as the contract log and the JSON telemetry — never a second copy that can drift.
set_task = _json_telemetry.set_task


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "") not in ("", "0", "false", "False")


def _run_id() -> str:
    return os.environ.get("IDEA_TEST_RUN_ID", "adhoc")


def _path() -> Path:
    override = os.environ.get(ENV_PATH)
    if override:
        return Path(override)
    return _RES / f"{_run_id()}_plan_retrievals.jsonl"


def _write(entry: Dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _results(result: Any) -> List[Dict[str, Any]]:
    candidates = list(getattr(result, "candidates", None) or [])[:_MAX_RESULTS]
    out: List[Dict[str, Any]] = []
    for rank, match in enumerate(candidates, 1):
        row = match.as_dict() if hasattr(match, "as_dict") else dict(match or {})
        row["rank"] = rank
        out.append(row)
    return out


def record_retrieval(
    result: Any,
    *,
    call_site: str,
    slot_fill: Optional[str] = None,
    applied_template_id: Optional[str] = None,
    node_id: Optional[str] = None,
    slot_values: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Optional[str]:
    """Log one retrieval attempt; returns its ``retrieval_id`` (or ``None`` when disabled).

    ``result`` is a ``retrieval.RetrievalResult``. ``slot_fill`` is the fill outcome
    (``"filled"`` / ``"failed"`` / ``"not_attempted"``) and ``applied_template_id`` the
    template that actually reached the engine — the pair is what separates "matched" from
    "used", which the decision alone cannot express once a slot-fill failure has downgraded
    the decision to no-match.

    A no-match is logged just like a hit: a retrieval that correctly found nothing is exactly
    as informative when the library's coverage is being audited.
    """
    if not enabled():
        return None
    retrieval_id = getattr(result, "retrieval_id", None)
    try:
        entry: Dict[str, Any] = {
            "event": "plan_library_retrieval",
            "t": round(time.time(), 3),
            "run_id": _run_id(),
            "task_id": _json_telemetry.current_task(),
            "retrieval_id": retrieval_id,
            "call_site": call_site,
            "node_id": node_id,
            "query_text": getattr(result, "query_text", ""),
            "archetype_hint": getattr(result, "archetype_hint", None),
            "results": _results(result),
            "decision": getattr(result, "decision", None),
            "reason": getattr(result, "reason", ""),
            "matched_template_id": getattr(result, "template_id", None),
            "similarity": getattr(result, "similarity", None),
            "margin_over_second": getattr(result, "margin_over_second", None),
            "applied_template_id": applied_template_id,
            "slot_fill": slot_fill or "not_attempted",
            "slot_values": slot_values,
            "error": error,
            # Filled in later by scripts/review_plan_retrievals.py, never by the engine.
            "human_review": {
                "reviewed": False,
                "label": None,
                "reviewer": None,
                "notes": None,
            },
        }
        _write(entry)
    except Exception:  # noqa: BLE001 — telemetry must never affect a run
        return retrieval_id
    return retrieval_id


def read_events(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every row of a retrieval log (``[]`` when the file is absent) — for the review CLI."""
    target = Path(path) if path else _path()
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
