"""Run the variant-agnostic audit layer (agent/app/testing/audit_layer.py) over real stored
result cells and print a per-variant recoverability/audit table. Offline, $0 — reads only files
already on disk under agent/idea_test_results/.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

from agent.app.testing import audit_layer as al  # noqa: E402

PATTERNS = [
    "agent/idea_test_results/lvl_el_*.json",
    "agent/idea_test_results/lvl_sr_*.json",
    "agent/idea_test_results/night_a1_g_*.json",
    "agent/idea_test_results/night_a1_sr_*.json",
    "agent/idea_test_results/night_a1_lg_*.json",
]


def cell_files():
    for pattern in PATTERNS:
        for path in sorted(glob.glob(pattern)):
            if path.endswith("_summary.json") or path.endswith(".jsonl"):
                continue
            yield path


def main():
    by_variant = defaultdict(lambda: {
        "n": 0, "recoverable": 0, "fetched_source": Counter(),
        "cited_total": 0, "cited_never_fetched_total": 0, "fetched_never_cited_total": 0,
        "quote_applicable": 0, "quotes_total": 0, "quotes_true": 0, "quotes_false": 0,
        "quotes_none": 0,
    })
    for path in cell_files():
        try:
            result = json.loads(Path(path).read_text())
        except Exception as exc:
            print(f"SKIP {path}: {exc}", file=sys.stderr)
            continue
        report = al.audit_result(result)
        variant = report["variant"] or Path(path).name
        stats = by_variant[variant]
        stats["n"] += 1
        stats["cited_total"] += len(report["cited"])
        if report["recoverable"]:
            stats["recoverable"] += 1
            stats["fetched_source"][report["fetched_source"]] += 1
            stats["cited_never_fetched_total"] += len(report["cited_never_fetched"])
            stats["fetched_never_cited_total"] += len(report["fetched_never_cited"])
        if report["quote_audit_applicable"]:
            stats["quote_applicable"] += 1
            for row in report["quotes"]:
                stats["quotes_total"] += 1
                if row["verified"] is True:
                    stats["quotes_true"] += 1
                elif row["verified"] is False:
                    stats["quotes_false"] += 1
                else:
                    stats["quotes_none"] += 1

    print(f"{'variant':<16}{'n':>5}{'fetch-recov':>13}{'src':>26}{'cited':>8}"
          f"{'cited!fetch':>13}{'fetch!cited':>13}{'quote-appl':>11}{'quotes(T/F/?)':>16}")
    for variant, s in sorted(by_variant.items()):
        src = ",".join(f"{k}:{v}" for k, v in s["fetched_source"].items()) or "-"
        quotes = f"{s['quotes_true']}/{s['quotes_false']}/{s['quotes_none']}"
        print(f"{variant:<16}{s['n']:>5}{s['recoverable']:>7}/{s['n']:<5}{src:>26}"
              f"{s['cited_total']:>8}{s['cited_never_fetched_total']:>13}"
              f"{s['fetched_never_cited_total']:>13}{s['quote_applicable']:>7}/{s['n']:<3}"
              f"{quotes:>16}")


if __name__ == "__main__":
    main()
