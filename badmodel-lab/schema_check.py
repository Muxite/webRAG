#!/usr/bin/env python3
"""Unit test for the schema-aware validity check used by the format-stress tier.

The function itself is CANONICAL in the harness at
``agent.app.testing.json_telemetry.schema_check`` (importable, used by the
structured_json/structured_thin aggregation and by analyze.py). This file imports
that exact function and exercises the real weak-model failure modes, so the lab and
the harness can never drift. No LLM, no network.

    ./.venv/bin/python badmodel-lab/schema_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the harness importable when run standalone (mirrors run_cell.sh's PYTHONPATH).
_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "services", _REPO / "services" / "agent"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from agent.app.testing.json_telemetry import schema_check  # noqa: E402

_SCHEMA = {"entity": "string", "value": "number", "unit": "string",
           "source_url": "string", "is_estimate": "boolean"}

_CASES = [
    # (label, raw, expect_parsed, expect_klass)
    ("clean object",
     '{"entity":"Quesnel Lake","value":511,"unit":"m",'
     '"source_url":"https://en.wikipedia.org/wiki/Quesnel_Lake","is_estimate":false}',
     True, "schema_valid"),
    ("empty object (parses, no fields) — the blind-spot the metric must catch",
     "{}", True, "schema_partial"),
    ("stringified number — the classic weak-model miss",
     '{"entity":"Quesnel Lake","value":"511 m","unit":"m",'
     '"source_url":"u","is_estimate":false}', True, "schema_partial"),
    ("bool-as-string is_estimate",
     '{"entity":"e","value":511,"unit":"m","source_url":"u","is_estimate":"false"}',
     True, "schema_partial"),
    ("number where boolean required — is_estimate=0 is not bool",
     '{"entity":"e","value":511,"unit":"m","source_url":"u","is_estimate":0}',
     True, "schema_partial"),
    ("missing a required key (no is_estimate)",
     '{"entity":"e","value":511,"unit":"m","source_url":"u"}', True, "schema_partial"),
    ("extra prose key (additionalProperties)",
     '{"entity":"e","value":511,"unit":"m","source_url":"u","is_estimate":false,'
     '"note":"I read this from the infobox"}', True, "schema_partial"),
    ("prose, ignored the format",
     "The maximum depth of Quesnel Lake is 511 m.", False, "not_json"),
    ("fenced json (right idea, wrong wrapper) — not parseable as-is",
     '```json\n{"entity":"e","value":511,"unit":"m","source_url":"u","is_estimate":false}\n```',
     False, "not_json"),
    ("bare number (parses, not an object)",
     "511", True, "schema_partial"),
    ("empty completion",
     "", False, "not_json"),
    ("integer value accepted as number",
     '{"entity":"e","value":56,"unit":"km2","source_url":"u","is_estimate":true}',
     True, "schema_valid"),
    ("float value accepted as number",
     '{"entity":"e","value":56.6,"unit":"km2","source_url":"u","is_estimate":true}',
     True, "schema_valid"),
]


def _selftest() -> int:
    fails = 0
    for label, raw, exp_parsed, exp_klass in _CASES:
        r = schema_check(raw, _SCHEMA)
        ok = (r["parsed_ok"] == exp_parsed) and (r["klass"] == exp_klass)
        fails += 0 if ok else 1
        detail = ""
        if r["klass"] == "schema_partial":
            bits = []
            if r["missing"]:
                bits.append(f"missing={r['missing']}")
            if r["mistyped"]:
                bits.append(f"mistyped={r['mistyped']}")
            if r["extra"]:
                bits.append(f"extra={r['extra']}")
            detail = "  (" + "; ".join(bits) + ")" if bits else ""
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:<52} parsed={r['parsed_ok']!s:<5} {r['klass']}{detail}")
    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(_CASES)-fails}/{len(_CASES)} cases")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
