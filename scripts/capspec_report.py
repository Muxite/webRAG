#!/usr/bin/env python3
"""Capability-spectrum report: DAG v2 (graph) vs off-the-shelf LangGraph across a model ladder.

Reads the flat result JSONs written by `adaptive_ladder_run.py` for the `capspec_*` axes and
answers the question the sweep was run to answer: *at which point on the model-capability curve,
if any, does DAG v2's structure earn its cost against an agent loop anyone can pip install?*

Three things this does that the existing reporters (`level_ladder.py`, `gate_report.py`) do not:

1. **Recomputes cost from raw tokens.** ``observability.cost.usd`` is ``$0`` for any model absent
   from ``model_costs.MODEL_PRICING`` (the OpenRouter price cache is only consulted for known
   slugs), which makes the cheapest, weakest models look infinitely cost-efficient. Prices below
   are the live OpenRouter list rates as of 2026-08-15, applied to the token counts the runs
   actually recorded.
2. **Separates "did not run" from "ran and scored 0".** A model with no tool-calling template is
   rejected by the LangGraph arm at the API layer before inference. That is a categorically
   different outcome from a run that happened and failed, and averaging them together is the
   single easiest way to draw a wrong conclusion from this sweep.
3. **Reports visits alongside score and cost.** The prior smoke's one clear linear-agent win came
   from *looking more*, not reasoning better, so evidence volume is a primary column here.
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "agent", "idea_test_results")

# Live OpenRouter list prices, USD per 1M tokens, fetched 2026-08-15. Local models are free at
# the margin (electricity is not being accounted for) and are pinned at 0.
PRICES = {
    "meta-llama/llama-3.2-1b-instruct": (0.027, 0.201),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
}
LOCAL_MODELS = {"tinyllama:latest", "phi3:mini", "qwen2.5:0.5b", "llama3.2:3b", "qwen2.5:7b"}

# tier ordering, weakest first — the x-axis of the whole question
TIERS = [
    ("tinyllama:latest", "local 1.1B  (no tool template)"),
    ("phi3:mini", "local 3.8B  (no tool template)"),
    ("qwen2.5:0.5b", "local 0.5B"),
    ("llama3.2:3b", "local 3B"),
    ("qwen2.5:7b", "local 7B"),
    ("meta-llama/llama-3.2-1b-instruct", "API 1B      ($0.027/$0.201)"),
    ("openai/gpt-4.1-nano", "API nano    ($0.10/$0.40)"),
    ("google/gemini-2.5-flash-lite", "API flash-lite ($0.10/$0.40)"),
]
TIER_ORDER = {m: i for i, (m, _) in enumerate(TIERS)}
TIER_LABEL = dict(TIERS)

# Wave 1 = the active-59 hard tier (difficulty 9-10). Wave 2 = in-range tasks (difficulty 3-6),
# added because the hard tier floors every weak model at 0 and cannot discriminate the arms there.
SHAPES = {"134": "chain", "135": "chain", "122": "fan-out", "140": "disambig", "128": "conflicting",
          "012": "breadth-links", "021": "extraction", "022": "doc-extraction",
          "023": "sequential", "049": "two-page-combine",
          # chain-shape follow-up axis (2026-08-15): chain WITHOUT fan-out, the one shape where a
          # later step must condition on an earlier one — i.e. where graph structure should pay.
          "136": "chain", "137": "chain", "138": "chain", "139": "chain",
          "046": "chain", "047": "chain", "065": "chain", "093": "chain"}
WAVE = {"134": "hard", "135": "hard", "122": "hard", "140": "hard", "128": "hard",
        "012": "easy", "021": "easy", "022": "easy", "023": "easy", "049": "easy",
        "136": "hard", "137": "hard", "138": "hard", "139": "hard",
        "046": "hard", "047": "hard", "065": "hard", "093": "hard"}
DIFFICULTY = {"134": 10, "135": 10, "122": 9, "140": 10, "128": 9,
              "012": 3, "021": 3, "022": 5, "023": 6, "049": 4,
              "136": 10, "137": 10, "138": 10, "139": 10,
              "046": 7, "047": 10, "065": 9, "093": 10}

# run_id prefix -> (variant label). csapi_g/csloc_g carry the arm in the run_id itself.
# Parse the execution VARIANT off the filename suffix rather than the run-id prefix: run-ids vary
# per wave (csapi_g / cseasy_g / csw3sr / csw3lg60) but the suffix the runner writes is stable.
ARM_RE = re.compile(
    r"^(cs[\w.]+?)_(baseline|good_adaptive)_rep(\d+)_(\d{3})_.*?"
    r"_(graph|langgraph_react|sequential_react)(?:_t\d+)?_r\d+\.json$")


def load_rows(prefixes):
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        base = os.path.basename(path)
        if base.endswith("_summary.json") or "_report_" in base:
            continue
        if not any(base.startswith(p) for p in prefixes):
            continue
        m = ARM_RE.match(base)
        if not m:
            continue
        runid, arm, rep, task, variant = m.groups()
        try:
            d = json.load(open(path))
        except Exception:
            continue
        val = d.get("validation") or {}
        score = val.get("overall_score")
        if score is None:
            continue
        # observability lives under `execution`; `infra_failed` is hoisted to the top level
        obs = ((d.get("execution") or {}).get("observability")) or {}
        cost_blob = obs.get("cost") or {}
        model = (d.get("model") or "").strip()
        if not model:
            continue
        pin, pout = PRICES.get(model, (0.0, 0.0))
        p_tok = int(cost_blob.get("prompt_tokens") or 0)
        c_tok = int(cost_blob.get("completion_tokens") or 0)
        true_cost = (p_tok * pin + c_tok * pout) / 1e6
        rows.append({
            "model": model,
            "arm": (f"graph:{arm}" if variant == "graph"
                    else ("langgraph@60" if "lg60" in runid else "langgraph") if variant == "langgraph_react"
                    else "seq_react"),
            "task": task,
            "shape": SHAPES.get(task, "?"),
            "wave": WAVE.get(task, "?"),
            "difficulty": DIFFICULTY.get(task, 0),
            "rep": int(rep),
            "score": float(score),
            "passed": bool(val.get("overall_passed")),
            "visits": int((obs.get("visit") or {}).get("count") or 0),
            "searches": int((obs.get("search") or {}).get("count") or 0),
            "prompt_tok": p_tok,
            "completion_tok": c_tok,
            "cost": true_cost,
            "reported_cost": float(cost_blob.get("usd") or 0.0),
            "infra_failed": bool(d.get("infra_failed")),
            "warning": (((d.get("execution") or {}).get("output")) or {}).get("warning", ""),
        })
    return rows


def agg(vals):
    vals = list(vals)
    if not vals:
        return None
    return statistics.mean(vals)


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cs",
                    help="comma-separated result-filename prefixes; default covers every capspec run-id")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    rows = load_rows([p for p in args.prefix.split(",") if p])
    if not rows:
        print("no matching result files", file=sys.stderr)
        return 1

    infra = [r for r in rows if r["infra_failed"]]
    if infra:
        print(f"!! {len(infra)} infra-quarantined cells EXCLUDED from every table below")
        rows = [r for r in rows if not r["infra_failed"]]

    arms = [a for a in ["graph:baseline", "graph:good_adaptive", "langgraph",
                        "langgraph@60", "seq_react"]
            if any(r["arm"] == a for r in rows)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["arm"])].append(r)

    print(f"\n{'='*104}\nSCORE by model tier x arm   (mean overall_score; n in parens)\n{'='*104}")
    print(f"{'tier':32s} " + "".join(f"{a:>22s}" for a in arms) + f"{'v2−LG':>10s}")
    for model, label in TIERS:
        if not any((model, a) in by for a in arms):
            continue
        cells = []
        for a in arms:
            rs = by.get((model, a), [])
            cells.append(f"{fmt(agg(r['score'] for r in rs), 3)} (n={len(rs)})" if rs else "not run")
        ga, lg = agg(r["score"] for r in by.get((model, "graph:good_adaptive"), [])), \
                 agg(r["score"] for r in by.get((model, "langgraph"), []))
        delta = f"{ga-lg:+.3f}" if (ga is not None and lg is not None) else "—"
        print(f"{label:32s} " + "".join(f"{c:>22s}" for c in cells) + f"{delta:>10s}")

    print(f"\n{'='*104}\nCOST (recomputed from tokens, USD/cell) and EVIDENCE (visits/cell)\n{'='*104}")
    print(f"{'tier':32s} " + "".join(f"{a+' $':>16s}{a+' vis':>14s}" for a in arms))
    for model, label in TIERS:
        if not any((model, a) in by for a in arms):
            continue
        cells = ""
        for a in arms:
            rs = by.get((model, a), [])
            c = agg(r["cost"] for r in rs)
            v = agg(r["visits"] for r in rs)
            cells += f"{('$'+format(c,'.5f')) if c is not None else '—':>16s}{fmt(v,1):>14s}"
        print(f"{label:32s} {cells}")

    print(f"\n{'='*104}\nSHAPE-STRATIFIED (mean score; the pooled number hides this)\n{'='*104}")
    shapes = sorted({r["shape"] for r in rows})
    print(f"{'shape':14s}{'tier':32s} " + "".join(f"{a:>22s}" for a in arms))
    for shape in shapes:
        for model, label in TIERS:
            sub = [r for r in rows if r["shape"] == shape and r["model"] == model]
            if not sub:
                continue
            cells = []
            for a in arms:
                rs = [r for r in sub if r["arm"] == a]
                cells.append(f"{fmt(agg(r['score'] for r in rs),3)} (n={len(rs)})" if rs else "not run")
            print(f"{shape:14s}{label:32s} " + "".join(f"{c:>22s}" for c in cells))

    print(f"\n{'='*104}\nPOOLED by arm across ALL models\n{'='*104}")
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        if not rs:
            continue
        tot_cost = sum(r["cost"] for r in rs)
        mean_s = agg(r["score"] for r in rs)
        print(f"  {a:22s} n={len(rs):3d}  score={fmt(mean_s,3)}  "
              f"visits={fmt(agg(r['visits'] for r in rs),1)}  "
              f"searches={fmt(agg(r['searches'] for r in rs),1)}  "
              f"$/cell={fmt(tot_cost/len(rs),5)}  total=${tot_cost:.4f}")

    print(f"\n{'='*104}\nZERO-SCORE RATE (floor effect check — a suite too hard to discriminate)\n{'='*104}")
    for model, label in TIERS:
        line = f"{label:32s} "
        any_row = False
        for a in arms:
            rs = by.get((model, a), [])
            if rs:
                any_row = True
                z = sum(1 for r in rs if r["score"] == 0.0)
                line += f"{f'{z}/{len(rs)} zero':>22s}"
            else:
                line += f"{'not run':>22s}"
        if any_row:
            print(line)


    print(f"\n{'='*104}\nWAVE-STRATIFIED: hard suite (difficulty 9-10) vs in-range tasks (difficulty 3-6)\n{'='*104}")
    print("The active-59 is 56/59 at difficulty >=8. Weak models floor at 0 there, so the arms")
    print("cannot be distinguished. The 'easy' wave exists to give them dynamic range.\n")
    for wave in ("hard", "easy"):
        sub = [r for r in rows if r["wave"] == wave]
        if not sub:
            continue
        print(f"-- {wave} wave (tasks: {','.join(sorted({r['task'] for r in sub}))}) --")
        print(f"{'tier':32s} " + "".join(f"{a:>22s}" for a in arms) + f"{'v2-base':>10s}")
        for model, label in TIERS:
            ms = [r for r in sub if r["model"] == model]
            if not ms:
                continue
            cells = []
            for a in arms:
                rs = [r for r in ms if r["arm"] == a]
                cells.append(f"{fmt(agg(r['score'] for r in rs),3)} (n={len(rs)})" if rs else "not run")
            b = agg(r["score"] for r in ms if r["arm"] == "graph:baseline")
            g = agg(r["score"] for r in ms if r["arm"] == "graph:good_adaptive")
            lift = f"{g-b:+.3f}" if (b is not None and g is not None) else "—"
            print(f"{label:32s} " + "".join(f"{c:>22s}" for c in cells) + f"{lift:>10s}")
        print()

    print(f"{'='*104}\nTOKEN ECONOMICS — is the premium reasoning (completion) or context (prompt)?\n{'='*104}")
    print(f"{'tier':30s}{'arm':22s}{'prompt':>10s}{'compl':>8s}{'ratio':>8s}{'visits':>8s}{'score':>7s}")
    for model, label in TIERS:
        for a in arms:
            rs = by.get((model, a), [])
            if not rs:
                continue
            pt = agg(r["prompt_tok"] for r in rs); ct = agg(r["completion_tok"] for r in rs)
            print(f"{label[:29]:30s}{a:22s}{pt:>10,.0f}{ct:>8,.0f}{pt/max(ct,1):>8.1f}"
                  f"{agg(r['visits'] for r in rs):>8.1f}{agg(r['score'] for r in rs):>7.3f}")

    print(f"\n{'='*104}\nISO-COST view (Q1): score per $0.01 spent — the thesis's own framing\n{'='*104}")
    print(f"{'tier':32s} " + "".join(f"{a:>22s}" for a in arms))
    for model, label in TIERS:
        if model not in PRICES:
            continue  # local models are $0 at the margin; iso-cost is undefined
        line = f"{label:32s} "
        for a in arms:
            rs = by.get((model, a), [])
            if not rs:
                line += f"{'not run':>22s}"; continue
            c = sum(r["cost"] for r in rs); sc = sum(r["score"] for r in rs)
            line += f"{(f'{sc/(c/0.01):.2f}' if c > 0 else 'n/a'):>22s}"
        print(line)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json_out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
