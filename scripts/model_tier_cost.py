#!/usr/bin/env python3
"""Per-model-tier cost estimate for the 4-tier benchmark design (2026-07-22).

Re-prices the EMPIRICAL per-arm token workload (measured on the live gpt-5-mini ladder) under each
tier model's OpenRouter pricing and its assigned ladder depth / burn budget. Because output tokens
dominate, cheap-output models can burn far more for the same $ — which is the whole point of giving
different tiers different burn extents.

Assumption: token COUNTS are ~model-independent for the same (task, arm) — true to ~±30%; a more
verbose/thinky model emits more. Burn multipliers scale the adaptive/full arms' tokens to represent
"allowed to burn more" (more k-vote samples + deeper re-expansion). Tune MODELS and rerun.
"""
N_TASKS = 50  # the deduped suite50

# (in_tokens, out_tokens) per run, empirical from the live gpt-5-mini ladder; 'react' is an estimate.
BASE = {
    "react":         (14000, 15000),   # sonnet sequential (no burn) — ESTIMATE, no live react tokens yet
    "baseline":      (13762, 18863),
    "good_adaptive": (47348, 35196),
    "full":          (56835, 56253),
}

# price = ($/1M in, $/1M out). arms = ladder rungs this tier runs. burn = token multiplier on the
# ADAPTIVE/FULL arms (baseline & react never scaled). reps = replicates.
# FINAL axis (decided 2026-07-22): sonnet ref (react, R3) + nano full ladder (R5) + deepseek full+2.5x
# burn (R5). gemini mid-tier dropped (same price as nano, dominated by deepseek on the frontier).
MODELS = {
    "1 good/strong  · claude-sonnet-5":   dict(price=(2.00, 10.00), arms=["react"],                             burn=1.0, reps=3),
    "3 budget/nonag · gpt-4.1-nano":      dict(price=(0.10, 0.40),  arms=["baseline", "good_adaptive", "full"], burn=1.0, reps=5),
    "4 super budget · deepseek-v4-flash": dict(price=(0.098, 0.196),arms=["baseline", "good_adaptive", "full"], burn=2.5, reps=5),
}


def run_cost(arm, price, burn):
    pin, pout = price
    it, ot = BASE[arm]
    mult = burn if arm in ("good_adaptive", "full") else 1.0
    return (it * mult * pin + ot * mult * pout) / 1e6


def main():
    print(f"Per-model cost for the {N_TASKS}-task suite. Output tokens dominate, so cheap-output = cheap burn.\n")
    print(f"{'tier · model':<40} {'arms (reps)':<26} {'$/run(arms)':<26} {'$/task':>7} {f'${N_TASKS}-suite':>10}")
    print("-" * 118)
    rows = []
    for name, c in MODELS.items():
        per_arm = {a: run_cost(a, c["price"], c["burn"]) for a in c["arms"]}
        per_task = sum(per_arm[a] * c["reps"] for a in c["arms"])
        total = per_task * N_TASKS
        arms_str = "+".join(a[:4] for a in c["arms"]) + f" (R{c['reps']}" + (f",×{c['burn']}burn" if c["burn"] != 1 else "") + ")"
        cost_str = " ".join(f"{a[:4]}=${per_arm[a]:.3f}" for a in c["arms"])
        print(f"{name:<40} {arms_str:<26} {cost_str:<26} {'$'+format(per_task,'.2f'):>7} {'$'+format(total,'.0f'):>10}")
        rows.append((name, per_task, total, c))
    print("-" * 118)
    # headline totals for a chosen axis (drop the ALT gemini-full)
    axis = [r for r in rows if "ALT" not in r[0]]
    grand = sum(r[2] for r in axis)
    print(f"\n4-tier axis total ({N_TASKS} tasks, sonnet ref + 3 cheap ladders): ${grand:.0f}")
    print("(swap gemini-flash-lite→full flash to see the ALT row; full flash's $2.50 output makes mid-tier burn ~5x pricier.)")


if __name__ == "__main__":
    main()
