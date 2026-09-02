#!/usr/bin/env python3
"""Sub-problem harness: measure `extract_decision`'s recovery rate against the fault corpus.

The fault corpus already exists — `agent/idea_test_results/*_json_telemetry.jsonl`, written by
`agent.app.testing.json_telemetry.record`. Every record has a `class` (one of `empty`,
`valid_json`, `fenced_json`, `refusal`, `truncated_json`, `malformed_json`, `prose`), a `raw_head`
(the completion, capped at 300 chars), `model`, `task_id`, `phase`, `parsed_ok`. The FAULT
classes — the ones a repair to `extract_decision` could plausibly recover — are `prose`,
`truncated_json`, `malformed_json`, `fenced_json`.

No agent, no model, no GPU: this loads JSONL off disk and calls a pure function. Runs in seconds.
See `docs/LEDGER_METHODOLOGY.md` §4.3 ("sub-problem harness — test the module without the agent").

CAUTION: `raw_head` is capped at 300 chars, so recovery on `truncated_json`/`malformed_json`
(and, to a lesser extent, `fenced_json`) is an UNDERSTATEMENT of what `extract_decision` would
recover against the model's actual, uncapped completion — a fixer that would have found a closing
brace past char 300 never gets the chance here. `prose` recovery is unaffected by the cap (a
prose completion recovers, or fails to, from its opening characters).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from agent.app.prompted_tools import extract_decision  # noqa: E402

#: The classes a JSON-repair fix could plausibly recover. `valid_json` already parses;
#: `refusal`/`empty` are not format defects a parser can fix.
FAULT_CLASSES = ("prose", "truncated_json", "malformed_json", "fenced_json")

_DEFAULT_GLOB = "agent/idea_test_results/*_json_telemetry.jsonl"


def find_corpus_files(root: Optional[Path] = None) -> List[Path]:
    """Every `*_json_telemetry.jsonl` fault-corpus file under `root` (repo root by default)."""
    base = root if root is not None else Path(__file__).resolve().parents[1]
    return sorted(Path(p) for p in glob.glob(str(base / _DEFAULT_GLOB)))


def iter_records(paths: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    """Every JSON object across `paths`, one per line. Malformed lines are skipped (best-effort,
    matching `json_telemetry.record`'s own "never break the caller" stance)."""
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(rec, dict):
                yield rec


def load_fault_records(
    paths: Optional[Iterable[Path]] = None, *, only_class: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Every record whose `class` is one of :data:`FAULT_CLASSES` (optionally narrowed further to
    `only_class`)."""
    paths = list(paths) if paths is not None else find_corpus_files()
    wanted = FAULT_CLASSES if only_class is None else (only_class,)
    return [r for r in iter_records(paths) if r.get("class") in wanted]


# --------------------------------------------------------------------------- unrecovered shapes


#: Ordered so the most specific/diagnostic shape wins when a raw_head matches more than one.
_SHAPE_PATTERNS: List[tuple] = [
    ("action_colon", re.compile(r'(^|\n)\s*ACTION\s*:', re.MULTILINE)),
    ("kv_line", re.compile(
        r'(^|\n)\s*action\s*[:=]\s*"?[A-Za-z_][\w\-]*"?\s*(\n|\bargs\s*=)', re.IGNORECASE,
    )),
    ("curly_quotes", re.compile("[“”‘’]")),
    ("python_literal", re.compile(r'\b(True|False|None)\b')),
]


def classify_unrecovered_shape(raw_head: str) -> str:
    """A coarse label for why a still-unrecovered `raw_head` didn't parse — used only for the
    harness's reporting breakdown, never by `extract_decision` itself."""
    text = raw_head or ""
    for label, pattern in _SHAPE_PATTERNS:
        if pattern.search(text):
            return label
    return "other_prose"


# --------------------------------------------------------------------------- measurement


def measure_recovery(
    records: Iterable[Dict[str, Any]], *, extract_fn: Callable[[Optional[str]], Any] = extract_decision,
) -> Dict[str, Any]:
    """Run `extract_fn` over every record's `raw_head` and tally recovery per class + overall.

    :returns: ``{"per_class": {class: {"recovered": n, "total": n}}, "overall": {...},
        "unrecovered_shapes": Counter, "unrecovered_ids": [...]}``. `unrecovered_ids` holds
        ``(class, task_id, model)`` for every record that stayed unrecovered, so a caller can
        diff two runs and prove nothing regressed.
    """
    per_class: Dict[str, Dict[str, int]] = {}
    shapes: Counter = Counter()
    unrecovered_ids: List[tuple] = []
    recovered_ids: List[tuple] = []

    for rec in records:
        cls = rec.get("class")
        per_class.setdefault(cls, {"recovered": 0, "total": 0})
        per_class[cls]["total"] += 1
        raw_head = rec.get("raw_head", "")
        ident = (cls, rec.get("task_id"), rec.get("model"), raw_head)
        extraction = extract_fn(raw_head)
        recovered = getattr(extraction, "value", None) is not None
        if recovered:
            per_class[cls]["recovered"] += 1
            recovered_ids.append(ident)
        else:
            shapes[classify_unrecovered_shape(raw_head)] += 1
            unrecovered_ids.append(ident)

    total = sum(v["total"] for v in per_class.values())
    recovered_total = sum(v["recovered"] for v in per_class.values())
    return {
        "per_class": per_class,
        "overall": {"recovered": recovered_total, "total": total},
        "unrecovered_shapes": shapes,
        "unrecovered_ids": unrecovered_ids,
        "recovered_ids": recovered_ids,
    }


def diff_no_regression(before: Dict[str, Any], after: Dict[str, Any]) -> List[tuple]:
    """Every id that was recovered `before` but is NOT recovered `after` — empty means no
    previously-recovered fault regressed. Compares on the full identity tuple (raw_head included)
    so it is robust to corpus files changing between runs."""
    before_recovered = set(before["recovered_ids"])
    after_recovered = set(after["recovered_ids"])
    return sorted(before_recovered - after_recovered)


# --------------------------------------------------------------------------- CLI


def _format_report(report: Dict[str, Any]) -> str:
    lines = []
    overall = report["overall"]
    pct = (overall["recovered"] / overall["total"] * 100) if overall["total"] else 0.0
    lines.append(
        f"overall: {overall['recovered']}/{overall['total']} ({pct:.1f}%)"
    )
    for cls in FAULT_CLASSES:
        stats = report["per_class"].get(cls, {"recovered": 0, "total": 0})
        if stats["total"] == 0:
            continue
        cpct = stats["recovered"] / stats["total"] * 100
        lines.append(f"  {cls}: {stats['recovered']}/{stats['total']} ({cpct:.1f}%)")
    if report["unrecovered_shapes"]:
        lines.append("unrecovered shapes:")
        for shape, count in report["unrecovered_shapes"].most_common():
            lines.append(f"  {shape}: {count}")
    return "\n".join(lines)


def _json_safe(report: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(report)
    out["unrecovered_shapes"] = dict(report["unrecovered_shapes"])
    out["unrecovered_ids"] = [list(t) for t in report["unrecovered_ids"]]
    out["recovered_ids"] = [list(t) for t in report["recovered_ids"]]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--class", dest="only_class", choices=FAULT_CLASSES, default=None,
                         help="restrict to one fault class")
    parser.add_argument("--limit", type=int, default=None,
                         help="only measure the first N matching records")
    args = parser.parse_args(argv)

    records = load_fault_records(only_class=args.only_class)
    if args.limit is not None:
        records = records[: args.limit]

    report = measure_recovery(records)
    if args.json:
        print(json.dumps(_json_safe(report), indent=2))
    else:
        print(_format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
