#!/usr/bin/env python
"""Render example DAGs (real cached compiled plans + a synthetic runtime graph) to PNGs.

Used to eyeball the visualizer's layout/sizing. Writes square >=1920px images.

    PYTHONPATH=services:services/agent ./.venv/bin/python scripts/render_dag_examples.py [out_dir] [side_px]
"""
import json
import os
import sys

from agent.app.testing.dag_visualizer import render_plan_dag, render_runtime_dag

PLANS_DIR = "services/agent/compiled_plans"
EXAMPLES = {
    "fanout6": ("c7b7b03a215c4fe0.json", "Pure fan-out — deepest of 6 lakes (test 053)",
                "Which of 6 lakes is deepest by max depth?"),
    "mixed3": ("2fe3a9d53a0fce8a.json", "Mixed DAG — 2 independent + 1 dependent (test 054)",
               "MA university of Beloved's author; author of Old Man & the Sea"),
    "chain3": ("b9ec94e88a68d814.json", "Dependent chain — author → university → founding year",
               "Founding year of the university of Things Fall Apart's author"),
    "wide12": ("bcef73fe20dd1509.json", "Two-wave DAG — 6 authors × birth-year (argmin, test 052)",
               "Which of 6 novels has the earliest-born author?"),
    "chain2": ("8ee76c256d660ee6.json", "Short chain — author → master's university",
               "Where did Beloved's author earn her master's?"),
}


def _synthetic_runtime_graph():
    """A small but realistic IdeaDag.to_dict() shape: expand -> visits -> merge -> finalize."""
    nodes = {
        "root": {"title": "Compare CPU vulnerabilities", "children": ["e1"], "status": "done"},
        "e1": {"title": "Expand: plan sub-questions", "children": ["v1", "v2", "v3"],
               "status": "done", "details": {"action": "expand"}},
        "v1": {"title": "Visit: Spectre (Wikipedia)", "children": ["m1"], "status": "done",
               "details": {"action": "visit"}, "score": 0.9},
        "v2": {"title": "Visit: Meltdown (Wikipedia)", "children": ["m1"], "status": "done",
               "details": {"action": "visit"}, "score": 0.8},
        "v3": {"title": "Visit: Retbleed advisory", "children": ["v4"], "status": "done",
               "details": {"action": "visit"}, "score": 0.7},
        "v4": {"title": "Visit: linked CVE detail", "children": ["m1"], "status": "done",
               "details": {"action": "visit"}, "score": 0.6},
        "m1": {"title": "Merge: reconcile mechanisms", "children": ["f1"], "status": "done",
               "details": {"action": "merge"}},
        "f1": {"title": "Finalize: cited comparison", "children": [], "status": "done",
               "details": {"action": "synthesize"}, "score": 0.95},
    }
    return {"nodes": nodes}


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dag_examples"
    side = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, (fn, title, mandate) in EXAMPLES.items():
        path = os.path.join(PLANS_DIR, fn)
        if not os.path.exists(path):
            print(f"skip {name}: {path} missing")
            continue
        plan = json.load(open(path))
        out = os.path.join(out_dir, f"plan_{name}.png")
        render_plan_dag(plan, title=title, mandate=mandate, side_px=side, out_path=out)
        written.append(out)
        print("wrote", out)
    out = os.path.join(out_dir, "runtime_graph.png")
    render_runtime_dag(_synthetic_runtime_graph(), side_px=side, out_path=out)
    written.append(out)
    print("wrote", out)
    print("\n".join(written))


if __name__ == "__main__":
    main()
