#!/usr/bin/env python3
"""Chain-shape test: does the graph engine's advantage appear where it should?

The premise: on a CHAIN, a later step must condition on an earlier step's result. That is the one
shape where a graph/DAG structure has a mechanistic reason to beat a linear ReAct loop. Wave 3
found seq_react matches graph:good_adaptive overall at 1/3 the input cost — so if the graph
structure is buying anything at all, it should show up here and nowhere else.

Reports paired-by-(model,task), run-complete — the contract this session's three unpaired-mean
traps forced.
"""
import collections
import glob
import json
import os
import re
import statistics

ARM_RE = re.compile(
    r"^(cs[\w.]+?)_(baseline|good_adaptive)_rep(\d+)_(\d{3})_.*?"
    r"_(graph|langgraph_react|sequential_react)(?:_t\d+)?_r\d+\.json$")
CHAIN_TASKS = {"135", "136", "137", "138", "139", "046", "047", "065", "093", "134"}
PRICES = {"meta-llama/llama-3.2-1b-instruct": (0.027, 0.201),
          "openai/gpt-4.1-nano": (0.10, 0.40),
          "google/gemini-2.5-flash-lite": (0.10, 0.40)}


def load():
    rows = []
    for p in glob.glob("agent/idea_test_results/cs*.json"):
        b = os.path.basename(p)
        m = ARM_RE.match(b)
        if not m:
            continue
        runid, arm, rep, task, variant = m.groups()
        if task not in CHAIN_TASKS:
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if j.get("infra_failed"):
            continue
        s = (j.get("validation") or {}).get("overall_score")
        if s is None:
            continue
        o = (j.get("execution") or {}).get("observability") or {}
        c = o.get("cost") or {}
        a = (f"graph:{arm}" if variant == "graph"
             else ("langgraph@60" if "lg60" in runid else "langgraph") if variant == "langgraph_react"
             else "seq_react")
        pin, pout = PRICES.get(j.get("model"), (0.0, 0.0))
        pt, ct = int(c.get("prompt_tokens") or 0), int(c.get("completion_tokens") or 0)
        rows.append(dict(model=j.get("model"), arm=a, task=task, score=float(s),
                         visits=int((o.get("visit") or {}).get("count") or 0),
                         pt=pt, ct=ct, cost=(pt * pin + ct * pout) / 1e6))
    return rows


def main():
    rows = load()
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        idx[(r["model"], r["task"])][r["arm"]].append(r)

    def paired(a, b):
        P = [(statistics.mean(x["score"] for x in v[a]), statistics.mean(x["score"] for x in v[b]))
             for v in idx.values() if a in v and b in v]
        if not P:
            return None
        w = sum(1 for x, y in P if x > y + 1e-9)
        t = sum(1 for x, y in P if abs(x - y) <= 1e-9)
        return len(P), statistics.mean(p[0] for p in P), statistics.mean(p[1] for p in P), w, t, len(P) - w - t

    print(f"CHAIN-SHAPE ONLY — tasks {','.join(sorted(t for t in CHAIN_TASKS))}")
    print(f"cells: {len(rows)}\n")
    print(f"{'A':22s}{'B':22s}{'n':>4}{'A':>8}{'B':>8}{'W/T/L':>12}")
    print("-" * 76)
    for a, b in [("graph:good_adaptive", "seq_react"),
                 ("graph:good_adaptive", "graph:baseline"),
                 ("graph:baseline", "seq_react"),
                 ("graph:good_adaptive", "langgraph")]:
        r = paired(a, b)
        print(f"{a:22s}{b:22s}" + (f"{r[0]:>4}{r[1]:>8.3f}{r[2]:>8.3f}{f'{r[3]}/{r[4]}/{r[5]}':>12}"
                                   if r else "   — no overlap"))

    print(f"\n{'arm':22s}{'n':>4}{'score':>8}{'visits':>8}{'prompt':>10}{'compl':>8}{'$/cell':>10}")
    print("-" * 70)
    for a in ["graph:baseline", "graph:good_adaptive", "seq_react", "langgraph", "langgraph@60"]:
        rs = [r for r in rows if r["arm"] == a]
        if not rs:
            continue
        print(f"{a:22s}{len(rs):>4}{statistics.mean(r['score'] for r in rs):>8.3f}"
              f"{statistics.mean(r['visits'] for r in rs):>8.1f}"
              f"{statistics.mean(r['pt'] for r in rs):>10,.0f}"
              f"{statistics.mean(r['ct'] for r in rs):>8,.0f}"
              f"{statistics.mean(r['cost'] for r in rs):>10.5f}")

    print("\nper-model paired: graph:good_adaptive vs seq_react")
    for mdl in sorted({r["model"] for r in rows}):
        P = [(statistics.mean(x["score"] for x in v["graph:good_adaptive"]),
              statistics.mean(x["score"] for x in v["seq_react"]))
             for (m2, t), v in idx.items()
             if m2 == mdl and "graph:good_adaptive" in v and "seq_react" in v]
        if not P:
            continue
        w = sum(1 for x, y in P if x > y + 1e-9)
        t_ = sum(1 for x, y in P if abs(x - y) <= 1e-9)
        print(f"  {mdl:36s} n={len(P):2d} {statistics.mean(p[0] for p in P):.3f} vs "
              f"{statistics.mean(p[1] for p in P):.3f}  {w}/{t_}/{len(P)-w-t_}")


if __name__ == "__main__":
    main()
