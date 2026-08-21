"""Aggregate rows written by ``langgraph_repeat_visit_probe.py``: a paired pre/post outcome table
plus what each run did immediately AFTER its first repeat visit.

    python3 scripts/langgraph_repeat_visit_analyze.py scripts/_probe_out/repeat_visit_*.json
"""
import json, sys, collections
from math import comb


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def p(x):
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)
    obs = p(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12)


def outcome(r):
    if r["correct"]:
        return "correct"
    if r["fabricated"]:
        return "fabricated"
    if r["abstained"]:
        return "abstained"
    return "other"


def after_repeat(r):
    """The tool calls following the FIRST repeat visit."""
    visits, seen, idx = [], set(), None
    for i, e in enumerate(r["tool_log"]):
        if e["tool"] != "visit":
            continue
        if e["arg"] in seen:
            idx = i
            break
        seen.add(e["arg"])
    if idx is None:
        return None
    repeated_url = r["tool_log"][idx]["arg"]
    tail = r["tool_log"][idx + 1:]
    next_visits = [e["arg"] for e in tail if e["tool"] == "visit"]
    return {"repeated_url": repeated_url, "n_after": len(tail),
            "revisited_again": repeated_url in next_visits,
            "new_url_after": any(u != repeated_url for u in next_visits),
            "only_searched_after": bool(tail) and not next_visits}


def main():
    rows = [r for path in sys.argv[1:] for r in json.load(open(path))]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r.get("stimulus", "?"), r["temperature"], r["arm"])].append(r)

    def redundant(r):
        """Visits to a URL already fetched in this run (the wasted fetches the fix aims at)."""
        seen, n = set(), 0
        for u in r["visits"]:
            n += u in seen
            seen.add(u)
        return n


    def max_same(r):
        return max(collections.Counter(r["visits"]).values(), default=0)


    print(f"{'model':<14}{'stim':<8}{'T':<5}{'arm':<6}{'n':<4}{'rep%':<7}{'visits':<8}{'redund':<8}"
          f"{'maxsame':<9}{'correct':<9}{'fabric':<8}{'abstain':<8}")
    for k in sorted(by):
        rs = by[k]
        n = len(rs)
        outs = collections.Counter(outcome(r) for r in rs)
        print(f"{k[0]:<14}{k[1]:<8}{k[2]:<5}{k[3]:<6}{n:<4}"
              f"{sum(r['repeat_visit'] for r in rs)/n:<7.2f}"
              f"{sum(len(r['visits']) for r in rs)/n:<8.1f}"
              f"{sum(redundant(r) for r in rs)/n:<8.2f}"
              f"{max(max_same(r) for r in rs):<9}"
              f"{outs['correct']:<9}{outs['fabricated']:<8}{outs['abstained']:<8}")

    print("\n-- runs that DID repeat a visit --")
    for arm in ("pre", "post"):
        rs = [r for r in rows if r["arm"] == arm and r["repeat_visit"]]
        info = [after_repeat(r) for r in rs]
        outs = collections.Counter(outcome(r) for r in rs)
        print(f"{arm:<5} n={len(rs):<3} marker_delivered={sum(r['repeat_marker_delivered'] for r in rs):<3}"
              f" fabricated={outs['fabricated']:<3} abstained={outs['abstained']:<3} correct={outs['correct']:<3}"
              f" | after the repeat: revisited_again={sum(i['revisited_again'] for i in info)}"
              f" new_url={sum(i['new_url_after'] for i in info)}"
              f" search_only={sum(i['only_searched_after'] for i in info)}"
              f" nothing_left={sum(1 for i in info if i['n_after'] == 0)}")

    pre = [r for r in rows if r["arm"] == "pre" and r["repeat_visit"]]
    post = [r for r in rows if r["arm"] == "post" and r["repeat_visit"]]
    a = sum(outcome(r) == "fabricated" for r in pre)
    c = sum(outcome(r) == "fabricated" for r in post)
    if pre and post:
        print(f"\nfabrication | repeat-visit: pre {a}/{len(pre)} vs post {c}/{len(post)}  "
              f"Fisher p={fisher(a, len(pre)-a, c, len(post)-c):.3f}")
    apre = [r for r in rows if r["arm"] == "pre"]
    apost = [r for r in rows if r["arm"] == "post"]
    a2 = sum(outcome(r) == "fabricated" for r in apre)
    c2 = sum(outcome(r) == "fabricated" for r in apost)
    print(f"fabrication | all runs:      pre {a2}/{len(apre)} vs post {c2}/{len(apost)}  "
          f"Fisher p={fisher(a2, len(apre)-a2, c2, len(apost)-c2):.3f}")

if __name__ == "__main__":
    main()
