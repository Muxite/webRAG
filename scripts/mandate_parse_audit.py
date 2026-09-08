#!/usr/bin/env python3
"""Audit what the mandate parsers actually see on the real derivation suite (tasks 210-221).

Phase 0a of ``docs/superpowers/plans/2026-09-08-ledger-dag-replan.md``. For each task module
``agent/app/idea_tests/test_2{10..21}_*.py`` this loads the module as a real package module
(so module-level constants such as ``OP_A``/``OP_B``/``ENTITIES`` keep object identity with any
other importer -- the same pattern ``scripts/rescore_results.py:38-46`` uses), calls
``get_task_statement()``, and runs the parsers a host-side ``host_derive`` hook would depend on:

  * ``agent.app.answer_numbers.mandate_demanded_operation`` -> which two-operand arithmetic the
    mandate demands, if any (``{"operation", "absolute", "reason"}``);
  * ``agent.app.idea_policies.candidate_coverage.extract_named_candidates`` -> the enumerated
    entity names the mandate lists (the LEGACY path, unchanged and still blind to 210-217);
  * ``agent.app.mandate_slots.parse_slots`` -> the Phase-2a slot parser: the ``(entity, field
    phrase)`` pairs of BOTH roster shapes (``n_slots`` / ``slot_fields``).

The point of the audit is the *joint* availability: a host-side derivation needs an operation
AND >= 2 operand slots from the same mandate. Two joint-availability lines are printed, one per
entity parser, and the contrast between them is the Phase-2a result: the legacy line is 0/12
(210-217 have an operation but no entities; 218-221 have entities but the ``argmax_phrasing``
veto suppresses the operation), the slot line is 12/12. The table is pinned by
``agent/tests/mandate_parse_audit_test.py``.

Usage (from the repo root)::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/mandate_parse_audit.py
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/mandate_parse_audit.py --json
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/mandate_parse_audit.py --ids 210,218

No network, no model calls, no result-file reads: pure module import + regex.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from agent.app.answer_numbers import mandate_demanded_operation
from agent.app.idea_policies.candidate_coverage import extract_named_candidates
from agent.app.mandate_slots import parse_slots, slot_field_phrases

#: The derivation-layer instrument cluster: 210-217 are two-operand arithmetic tasks,
#: 218-221 are argmax-over-a-computed-quantity tasks. Not part of suite59.
DERIVATION_SUITE_IDS: List[str] = [str(i) for i in range(210, 222)]

_TESTS_DIR = os.path.join("agent", "app", "idea_tests")
_TESTS_PKG = "agent.app.idea_tests"

#: How many extracted entity names to carry in the printed/JSON row (the audit is about the
#: COUNT; the first few names are only there to make a wrong parse recognisable at a glance).
_ENTITY_PREVIEW = 3

#: How many DISTINCT slot field phrases to carry per row. Two is the informative number: it is
#: what separates a 210-217 pair asking two different fields from one asking the same field of
#: two entities, and it shows a 218-221 roster sharing ONE phrase across five slots.
_FIELD_PREVIEW = 2

#: Field phrases are full mandate wording; the table shows a prefix of each.
_FIELD_COL_CHARS = 44


def _module_name_for(test_id: str) -> Optional[str]:
    """Dotted module name for ``test_<test_id>_*.py``, or ``None`` when no module matches."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for base in (_TESTS_DIR, os.path.join(here, _TESTS_DIR)):
        matches = sorted(glob.glob(os.path.join(base, f"test_{test_id}_*.py")))
        if matches:
            return f"{_TESTS_PKG}." + os.path.basename(matches[0])[:-3]
    return None


def audit_module(test_id: str) -> Dict[str, Any]:
    """Parse one task module's mandate with both parsers.

    :param test_id: 3-digit test id, e.g. ``"210"``.
    :returns: a row dict with keys ``test_id``, ``module``, ``operation``, ``absolute``,
        ``reason``, ``n_entities``, ``entities`` (first :data:`_ENTITY_PREVIEW` names),
        ``n_slots``, ``slot_fields`` (first :data:`_FIELD_PREVIEW` DISTINCT field phrases),
        ``slot_entities``, ``mandate_chars`` and ``error`` (``None`` on success; the exception
        text otherwise, with every parser field left at its empty value).
    :raises: nothing -- an unimportable module or a raising ``get_task_statement()`` is
        reported in ``error`` so one bad module cannot hide the rest of the table.
    """
    row: Dict[str, Any] = {
        "test_id": str(test_id), "module": _module_name_for(str(test_id)) or "",
        "operation": None, "absolute": False, "reason": "", "n_entities": 0,
        "entities": [], "n_slots": 0, "slot_fields": [], "slot_entities": [],
        "mandate_chars": 0, "error": None,
    }
    if not row["module"]:
        row["error"] = f"no module matching test_{test_id}_*.py"
        return row
    try:
        mod = importlib.import_module(row["module"])
        mandate = mod.get_task_statement()
        op = mandate_demanded_operation(mandate)
        entities = extract_named_candidates(mandate)
        slots = parse_slots(mandate)
    except Exception as exc:  # noqa: BLE001 -- report, never abort the audit
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["operation"] = op.get("operation")
    row["absolute"] = bool(op.get("absolute"))
    row["reason"] = str(op.get("reason") or "")
    row["n_entities"] = len(entities)
    row["entities"] = list(entities[:_ENTITY_PREVIEW])
    row["n_slots"] = len(slots)
    row["slot_fields"] = slot_field_phrases(slots)[:_FIELD_PREVIEW]
    row["slot_entities"] = [s.entity for s in slots[:_ENTITY_PREVIEW]]
    row["mandate_chars"] = len(str(mandate or ""))
    return row


def audit_modules(ids: Sequence[str]) -> List[Dict[str, Any]]:
    """:func:`audit_module` over ``ids``, in the order given. Pure; safe to call from tests."""
    return [audit_module(str(i)) for i in ids]


def _shape_is_derivable(row: Dict[str, Any]) -> bool:
    """Whether the mandate names a host-derivable SHAPE: a two-operand operation, or the argmax
    over a per-entity computed quantity that ``mandate_demanded_operation`` reports as
    ``argmax_phrasing`` (its deliberate "not a two-operand op" answer, not a parse failure)."""
    return row["operation"] is not None or row["reason"] == "argmax_phrasing"


def format_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Render the audit rows as a fixed-width table (the human-readable Phase-0a artifact)."""
    head = (f"{'test':<6}{'operation':<18}{'reason':<17}{'n_ent':>6}{'n_slots':>8}"
            f"  slot_fields[:{_FIELD_PREVIEW}]")
    out = [head, "-" * len(head)]
    for r in rows:
        if r["error"]:
            out.append(f"{r['test_id']:<6}{'ERROR':<18}{r['error'][:60]}")
            continue
        op = str(r["operation"]) + (" (abs)" if r["absolute"] else "")
        fields = " | ".join(f[:_FIELD_COL_CHARS] for f in r["slot_fields"]) or "-"
        out.append(f"{r['test_id']:<6}{op:<18}{r['reason']:<17}{r['n_entities']:>6}"
                   f"{r['n_slots']:>8}  {fields}")
    ok_rows = [r for r in rows if not r["error"]]
    legacy = [r for r in ok_rows if r["operation"] is not None and r["n_entities"] >= 2]
    by_slots = [r for r in ok_rows if _shape_is_derivable(r) and r["n_slots"] >= 2]
    out.append("-" * len(head))
    out.append(f"joint availability (operation AND >=2 entity slots): "
               f"{len(legacy)}/{len(rows)}")
    out.append(f"joint_availability_slots (operation or argmax_phrasing AND >=2 parsed slots): "
               f"{len(by_slots)}/{len(rows)}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", default=",".join(DERIVATION_SUITE_IDS),
                    help="comma-separated test ids (default: the 210-221 derivation suite)")
    ap.add_argument("--json", action="store_true", help="emit JSON rows instead of the table")
    args = ap.parse_args(argv)

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    rows = audit_modules(ids)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(format_table(rows))
    return 1 if any(r["error"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
