#!/usr/bin/env python3
"""Estimate $ and wall-clock to run the validated benchmark suite, from EMPIRICAL per-arm costs
measured on the live `ladder` run (2026-07-22). Update PER_ARM as more cells complete.

Usage: ./.venv/bin/python scripts/benchmark_cost_estimate.py [--n 64] [--jobs 8]
"""
import argparse

# (usd_per_run, secs_per_run) — empirical means from the live ladder; reference is historical (gemini react).
PER_ARM = {
    "baseline":      (0.0423, 286),
    "good_adaptive": (0.0809, 623),
    "full":          (0.1305, 657),
    "reference":     (0.1500, 400),   # gemini-3.1-pro react; refine when ladder reference cells finish
}

SCENARIOS = {
    # name: (list of (arm, reps))
    "A: full ladder + ref (R=5/3)":      [("baseline", 5), ("good_adaptive", 5), ("full", 5), ("reference", 3)],
    "B: headline A/B + ref (R=5/3)":     [("baseline", 5), ("good_adaptive", 5), ("reference", 3)],
    "C: full ladder + ref (R=3/3)":      [("baseline", 3), ("good_adaptive", 3), ("full", 3), ("reference", 3)],
    "D: headline A/B only (R=5)":        [("baseline", 5), ("good_adaptive", 5)],
    "E: full ladder no ref (R=5)":       [("baseline", 5), ("good_adaptive", 5), ("full", 5)],
}


def per_task(scenario):
    usd = sum(PER_ARM[a][0] * r for a, r in scenario)
    secs = sum(PER_ARM[a][1] * r for a, r in scenario)  # serial chain per task (per-task serialized)
    return usd, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8, help="parallel tasks (capped by #tasks)")
    ap.add_argument("--ns", default="40,50,64,80", help="candidate valid-task counts")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]

    print(f"Empirical per-arm: " + ", ".join(f"{a}=${c:.3f}/{s}s" for a, (c, s) in PER_ARM.items()))
    print(f"Parallel tasks (jobs) = {args.jobs}; per-task arms run serially (chroma mem is task-keyed)\n")
    for name, sc in SCENARIOS.items():
        u, s = per_task(sc)
        cells = sum(r for _, r in sc)
        print(f"### {name}   [{cells} cells/task, ${u:.2f}/task, {s/3600:.1f}h/task serial]")
        print(f"    {'N tasks':>8} {'$ total':>10} {'wall-clock':>12}")
        for n in ns:
            waves = -(-n // args.jobs)  # ceil
            hours = waves * s / 3600.0
            print(f"    {n:>8} {'$'+format(n*u, '.0f'):>10} {format(hours, '.1f')+'h':>12}")
        print()


if __name__ == "__main__":
    main()
