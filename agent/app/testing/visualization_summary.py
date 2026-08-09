"""
Summary statistics functions for visualization.

Provides overall stats, per-model/per-variant breakdowns, and
best-graph / best-sequential leaderboards.
"""

from typing import Dict, Any, List
from collections import defaultdict
import numpy as np

from .visualization_helpers import _system_label, _extract_graph_metrics


def _variant_model_key(result: Dict[str, Any]) -> tuple:
    """Return (model, variant) for a result."""
    model = str(result.get("model", "unknown"))
    variant = str(result.get("execution_variant", "graph"))
    return model, variant


def calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics from test results.
    :param results: Test results.
    :return: Summary statistics dict.
    """
    from agent.app.model_costs import estimate_cost, estimate_cost_from_total

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("validation", {}).get("overall_passed", False))
    scores = [r.get("validation", {}).get("overall_score", 0.0) for r in results]
    avg_score = float(np.mean(scores)) if scores else 0.0
    median_score = float(np.median(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0

    total_duration = sum(r.get("execution", {}).get("duration_seconds", 0) for r in results)
    total_tokens = sum(r.get("execution", {}).get("observability", {}).get("llm", {}).get("total_tokens", 0) for r in results)
    total_searches = sum(r.get("execution", {}).get("observability", {}).get("search", {}).get("count", 0) for r in results)
    total_visits = sum(r.get("execution", {}).get("observability", {}).get("visit", {}).get("count", 0) for r in results)
    total_visit_kb = sum(r.get("execution", {}).get("observability", {}).get("visit", {}).get("kilobytes", 0.0) for r in results)
    total_search_kb = sum(r.get("execution", {}).get("observability", {}).get("search", {}).get("kilobytes", 0.0) for r in results)

    deliverable_lens = []
    for r in results:
        deliv = r.get("final_output", {}).get("final_deliverable", "")
        if not deliv:
            deliv = r.get("output", {}).get("final_deliverable", "")
        deliverable_lens.append(len(str(deliv)) if deliv else 0)
    avg_deliverable_len = float(np.mean(deliverable_lens)) if deliverable_lens else 0.0

    test_ids = set(r.get("test_metadata", {}).get("test_id", "unknown") for r in results)
    models = set(_system_label(r) for r in results)

    # --- Cost calculation per system label ---
    total_cost = 0.0
    model_costs: Dict[str, float] = defaultdict(float)
    model_test_counts: Dict[str, int] = defaultdict(int)
    for r in results:
        raw_model = str(r.get("model", "unknown"))
        label = _system_label(r)
        obs = r.get("execution", {}).get("observability", {})
        llm = obs.get("llm", {})
        input_tokens = llm.get("prompt", {}).get("tokens", 0)
        output_tokens = llm.get("completion", {}).get("tokens", 0)
        if input_tokens > 0 or output_tokens > 0:
            cost = estimate_cost(raw_model, input_tokens, output_tokens)
        else:
            cost = estimate_cost_from_total(raw_model, llm.get("total_tokens", 0))
        if cost is not None:
            total_cost += cost
            model_costs[label] += cost
        model_test_counts[label] += 1

    # --- Validation checks ---
    target_checks = ["user_quotes", "synthesis_quality", "source_urls", "source_coverage", "languages_count", "historical_quote", "data_numbers"]
    check_stats = {}
    for check_name in target_checks:
        check_scores = []
        check_passed = 0
        check_total = 0
        for result in results:
            validation = result.get("validation", {})
            grep_validations = validation.get("grep_validations", [])
            for check in grep_validations:
                if check.get("check", "") == check_name:
                    check_scores.append(check.get("score", 0.0))
                    check_total += 1
                    if check.get("passed", False):
                        check_passed += 1
        check_stats[check_name] = {
            "success_rate": check_passed / check_total if check_total > 0 else 0.0,
            "avg_score": float(np.mean(check_scores)) if check_scores else 0.0,
            "total": check_total,
        }

    per_model_cost_summary = {}
    for label in sorted(model_costs.keys()):
        count = model_test_counts[label]
        per_model_cost_summary[label] = {
            "total_cost": model_costs[label],
            "avg_cost": model_costs[label] / count if count > 0 else 0.0,
            "test_count": count,
        }

    # --- Per-model/variant breakdown ---
    mv_scores: Dict[tuple, List[float]] = defaultdict(list)
    mv_passed: Dict[tuple, int] = defaultdict(int)
    mv_count: Dict[tuple, int] = defaultdict(int)
    mv_duration: Dict[tuple, float] = defaultdict(float)
    mv_tokens: Dict[tuple, int] = defaultdict(int)
    mv_visits: Dict[tuple, int] = defaultdict(int)
    mv_searches: Dict[tuple, int] = defaultdict(int)
    mv_cost: Dict[tuple, float] = defaultdict(float)
    mv_depth: Dict[tuple, List[float]] = defaultdict(list)
    mv_branching: Dict[tuple, List[float]] = defaultdict(list)
    mv_nodes: Dict[tuple, List[int]] = defaultdict(list)

    for r in results:
        model, variant = _variant_model_key(r)
        key = (model, variant)
        score = r.get("validation", {}).get("overall_score", 0.0)
        passed = r.get("validation", {}).get("overall_passed", False)
        exec_data = r.get("execution", {})
        obs = exec_data.get("observability", {})

        mv_scores[key].append(score)
        if passed:
            mv_passed[key] += 1
        mv_count[key] += 1
        mv_duration[key] += exec_data.get("duration_seconds", 0)
        mv_tokens[key] += obs.get("llm", {}).get("total_tokens", 0)
        mv_visits[key] += obs.get("visit", {}).get("count", 0)
        mv_searches[key] += obs.get("search", {}).get("count", 0)

        gm = _extract_graph_metrics(r)
        mv_depth[key].append(gm["max_depth"])
        mv_branching[key].append(gm["avg_branching"])
        mv_nodes[key].append(gm["total_nodes"])

        llm = obs.get("llm", {})
        inp = llm.get("prompt", {}).get("tokens", 0)
        out = llm.get("completion", {}).get("tokens", 0)
        if inp > 0 or out > 0:
            c = estimate_cost(model, inp, out)
        else:
            c = estimate_cost_from_total(model, llm.get("total_tokens", 0))
        if c is not None:
            mv_cost[key] += c

    leaderboard = []
    for key in sorted(mv_count.keys()):
        model, variant = key
        n = mv_count[key]
        sc = mv_scores[key]
        entry = {
            "model": model,
            "variant": variant,
            "count": n,
            "passed": mv_passed[key],
            "pass_rate": mv_passed[key] / n if n > 0 else 0.0,
            "avg_score": float(np.mean(sc)) if sc else 0.0,
            "median_score": float(np.median(sc)) if sc else 0.0,
            "std_score": float(np.std(sc)) if sc else 0.0,
            "min_score": float(np.min(sc)) if sc else 0.0,
            "max_score": float(np.max(sc)) if sc else 0.0,
            "avg_duration": mv_duration[key] / n if n > 0 else 0.0,
            "total_tokens": mv_tokens[key],
            "avg_tokens": mv_tokens[key] / n if n > 0 else 0.0,
            "avg_visits": mv_visits[key] / n if n > 0 else 0.0,
            "avg_searches": mv_searches[key] / n if n > 0 else 0.0,
            "total_cost": mv_cost[key],
            "avg_cost": mv_cost[key] / n if n > 0 else 0.0,
            "avg_depth": float(np.mean(mv_depth[key])) if mv_depth[key] else 0.0,
            "avg_branching": float(np.mean(mv_branching[key])) if mv_branching[key] else 0.0,
            "avg_nodes": float(np.mean(mv_nodes[key])) if mv_nodes[key] else 0.0,
        }
        leaderboard.append(entry)

    # Sort by avg_score desc then pass_rate desc
    leaderboard.sort(key=lambda e: (e["avg_score"], e["pass_rate"]), reverse=True)

    graph_board = [e for e in leaderboard if e["variant"] == "graph"]
    sequential_board = [e for e in leaderboard if e["variant"] == "sequential"]

    # --- Per-test breakdown (across all models/variants) ---
    test_scores: Dict[str, List[float]] = defaultdict(list)
    test_passed: Dict[str, int] = defaultdict(int)
    test_count: Dict[str, int] = defaultdict(int)
    for r in results:
        tid = r.get("test_metadata", {}).get("test_id", "unknown")
        score = r.get("validation", {}).get("overall_score", 0.0)
        test_scores[tid].append(score)
        test_count[tid] += 1
        if r.get("validation", {}).get("overall_passed", False):
            test_passed[tid] += 1

    test_difficulty = []
    for tid in sorted(test_count.keys()):
        n = test_count[tid]
        sc = test_scores[tid]
        test_difficulty.append({
            "test_id": tid,
            "count": n,
            "pass_rate": test_passed[tid] / n if n > 0 else 0.0,
            "avg_score": float(np.mean(sc)) if sc else 0.0,
            "std_score": float(np.std(sc)) if sc else 0.0,
        })
    test_difficulty.sort(key=lambda e: e["avg_score"])

    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
        "avg_score": avg_score,
        "median_score": median_score,
        "std_score": std_score,
        "total_duration": total_duration,
        "avg_duration": total_duration / total_tests if total_tests > 0 else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / total_tests if total_tests > 0 else 0.0,
        "total_searches": total_searches,
        "avg_searches": total_searches / total_tests if total_tests > 0 else 0.0,
        "total_visits": total_visits,
        "avg_visits": total_visits / total_tests if total_tests > 0 else 0.0,
        "total_visit_kb": total_visit_kb,
        "total_search_kb": total_search_kb,
        "avg_deliverable_len": avg_deliverable_len,
        "unique_tests": len(test_ids),
        "unique_models": len(models),
        "test_ids": sorted(test_ids),
        "models": sorted(models),
        "check_stats": check_stats,
        "total_cost_usd": total_cost,
        "avg_cost_usd": total_cost / total_tests if total_tests > 0 else 0.0,
        "model_costs": per_model_cost_summary,
        "leaderboard": leaderboard,
        "graph_leaderboard": graph_board,
        "sequential_leaderboard": sequential_board,
        "test_difficulty": test_difficulty,
    }


def _print_leaderboard(entries: List[Dict[str, Any]], title: str):
    """
    Print a ranked leaderboard of model/variant combos.
    :param entries: Sorted leaderboard entries.
    :param title: Section title.
    """
    from agent.app.model_costs import format_cost

    if not entries:
        return
    print(f"\n  {title}")
    print(f"  {'Rank':>4}  {'Model':>14}  {'Variant':>10}  {'Avg':>6}  {'Med':>6}  {'Std':>5}  {'Pass%':>6}  {'Runs':>4}  {'AvgCost':>8}  {'AvgTime':>7}  {'Depth':>5}  {'Branch':>6}  {'Nodes':>5}")
    print(f"  {'----':>4}  {'-'*14:>14}  {'-'*10:>10}  {'-----':>6}  {'-----':>6}  {'-----':>5}  {'-----':>6}  {'----':>4}  {'-------':>8}  {'------':>7}  {'-----':>5}  {'------':>6}  {'-----':>5}")
    for rank, e in enumerate(entries, 1):
        marker = " *" if rank == 1 else "  "
        cost_str = format_cost(e["avg_cost"])
        print(
            f"{marker}{rank:>2}  {e['model']:>14}  {e['variant']:>10}  "
            f"{e['avg_score']:>6.3f}  {e['median_score']:>6.3f}  {e['std_score']:>5.3f}  "
            f"{e['pass_rate']:>5.1%}  {e['count']:>4}  "
            f"{cost_str:>8}  {e['avg_duration']:>6.1f}s  "
            f"{e.get('avg_depth', 0):>5.1f}  {e.get('avg_branching', 0):>6.1f}  {e.get('avg_nodes', 0):>5.0f}"
        )


def print_summary(stats: Dict[str, Any]):
    """
    Print comprehensive summary statistics.
    :param stats: Summary statistics dict.
    """
    from agent.app.model_costs import format_cost

    print("\n" + "="*100)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*100)
    print(f"Total Runs: {stats['total_tests']}")
    print(f"Passed: {stats['passed_tests']} ({stats['pass_rate']:.1%})")
    print(f"Score Statistics:")
    print(f"  Average: {stats['avg_score']:.3f}")
    print(f"  Median: {stats['median_score']:.3f}")
    print(f"  Std Dev: {stats['std_score']:.3f}")
    print(f"\nExecution Metrics:")
    print(f"  Total Duration: {stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f} min)")
    print(f"  Average Duration: {stats['avg_duration']:.1f}s per test")
    print(f"  Total Tokens: {stats['total_tokens']:,}")
    print(f"  Average Tokens: {stats['avg_tokens']:,.0f} per test")
    print(f"\nEstimated Costs:")
    print(f"  Total Cost: {format_cost(stats.get('total_cost_usd'))}")
    print(f"  Average Cost per Test: {format_cost(stats.get('avg_cost_usd'))}")
    model_costs = stats.get("model_costs", {})
    for label, cost_data in sorted(model_costs.items()):
        tc = format_cost(cost_data.get("total_cost"))
        ac = format_cost(cost_data.get("avg_cost"))
        n = cost_data.get("test_count", 0)
        print(f"  {label}: total {tc}, avg {ac} ({n} runs)")
    print(f"\nActions:")
    print(f"  Total Searches: {stats['total_searches']} (avg: {stats['avg_searches']:.1f} per test)")
    print(f"  Total Visits: {stats['total_visits']} (avg: {stats['avg_visits']:.1f} per test)")
    print(f"\nCoverage:")
    print(f"  Unique Tests: {stats['unique_tests']} ({', '.join(stats['test_ids'])})")
    print(f"  Unique Systems: {stats['unique_models']} ({', '.join(stats['models'])})")

    # --- LEADERBOARD: Best Graph ---
    graph_board = stats.get("graph_leaderboard", [])
    _print_leaderboard(graph_board, "BEST GRAPH CONFIGURATION")
    if graph_board:
        best = graph_board[0]
        print(f"\n  >>> BEST GRAPH: {best['model']} — avg {best['avg_score']:.3f}, "
              f"pass {best['pass_rate']:.0%}, {format_cost(best['avg_cost'])}/run")

    # --- LEADERBOARD: Best Sequential ---
    seq_board = stats.get("sequential_leaderboard", [])
    _print_leaderboard(seq_board, "BEST SEQUENTIAL CONFIGURATION")
    if seq_board:
        best = seq_board[0]
        print(f"\n  >>> BEST SEQUENTIAL: {best['model']} — avg {best['avg_score']:.3f}, "
              f"pass {best['pass_rate']:.0%}, {format_cost(best['avg_cost'])}/run")

    # --- OVERALL LEADERBOARD ---
    leaderboard = stats.get("leaderboard", [])
    _print_leaderboard(leaderboard, "OVERALL LEADERBOARD (all model+variant combos)")
    if leaderboard:
        best = leaderboard[0]
        print(f"\n  >>> OVERALL BEST: {best['model']} [{best['variant']}] — avg {best['avg_score']:.3f}, "
              f"pass {best['pass_rate']:.0%}, {format_cost(best['avg_cost'])}/run")

    # --- Graph vs Sequential comparison ---
    graph_results = [e for e in leaderboard if e["variant"] == "graph"]
    seq_results = [e for e in leaderboard if e["variant"] == "sequential"]
    if graph_results and seq_results:
        graph_avg = np.mean([e["avg_score"] for e in graph_results])
        seq_avg = np.mean([e["avg_score"] for e in seq_results])
        graph_pass = np.mean([e["pass_rate"] for e in graph_results])
        seq_pass = np.mean([e["pass_rate"] for e in seq_results])
        print(f"\n  Graph vs Sequential (averaged across models):")
        print(f"    Graph:      avg score {graph_avg:.3f}, avg pass rate {graph_pass:.1%}")
        print(f"    Sequential: avg score {seq_avg:.3f}, avg pass rate {seq_pass:.1%}")
        diff = graph_avg - seq_avg
        winner = "Graph" if diff > 0 else "Sequential" if diff < 0 else "Tied"
        print(f"    Winner: {winner} (delta: {abs(diff):.3f})")

    # --- GRAPH STRUCTURE STATS ---
    if leaderboard:
        graph_entries = [e for e in leaderboard if e["variant"] == "graph"]
        seq_entries = [e for e in leaderboard if e["variant"] == "sequential"]
        print(f"\n  GRAPH STRUCTURE (avg per model+variant)")
        print(f"  {'System':<30} {'Depth':>6} {'Branch':>7} {'Nodes':>6} {'Leaf%':>6}")
        print(f"  {'-'*30:<30} {'-'*6:>6} {'-'*7:>7} {'-'*6:>6} {'-'*6:>6}")
        for e in leaderboard:
            label = f"{e['model']} [{e['variant']}]"
            nodes = e.get("avg_nodes", 0)
            depth = e.get("avg_depth", 0)
            branch = e.get("avg_branching", 0)
            # leaf% approximation: if branching ~1 and depth > 0 then few internals
            # Just show available data
            print(f"  {label:<30} {depth:>6.1f} {branch:>7.1f} {nodes:>6.0f}")

        if graph_entries and seq_entries:
            g_depth = np.mean([e.get("avg_depth", 0) for e in graph_entries])
            s_depth = np.mean([e.get("avg_depth", 0) for e in seq_entries])
            g_branch = np.mean([e.get("avg_branching", 0) for e in graph_entries])
            s_branch = np.mean([e.get("avg_branching", 0) for e in seq_entries])
            g_nodes = np.mean([e.get("avg_nodes", 0) for e in graph_entries])
            s_nodes = np.mean([e.get("avg_nodes", 0) for e in seq_entries])
            print(f"\n  Graph vs Sequential — Structure:")
            print(f"    Graph:      depth {g_depth:.1f}, branching {g_branch:.1f}, nodes {g_nodes:.0f}")
            print(f"    Sequential: depth {s_depth:.1f}, branching {s_branch:.1f}, nodes {s_nodes:.0f}")

    # --- Hardest / Easiest tests ---
    test_difficulty = stats.get("test_difficulty", [])
    if test_difficulty:
        print(f"\n  TEST DIFFICULTY RANKING (hardest first):")
        print(f"  {'Test':>6}  {'AvgScore':>8}  {'Pass%':>6}  {'Std':>5}  {'Runs':>4}")
        print(f"  {'------':>6}  {'--------':>8}  {'-----':>6}  {'-----':>5}  {'----':>4}")
        for e in test_difficulty:
            print(f"  {e['test_id']:>6}  {e['avg_score']:>8.3f}  {e['pass_rate']:>5.1%}  {e['std_score']:>5.3f}  {e['count']:>4}")

    # --- DATA COVERAGE & RESPONSE QUALITY ---
    leaderboard = stats.get("leaderboard", [])
    if leaderboard:
        print(f"\n  DATA COVERAGE & RESPONSE QUALITY")
        total_data_kb = stats.get("total_visit_kb", 0) + stats.get("total_search_kb", 0)
        avg_deliverable_len = stats.get("avg_deliverable_len", 0)
        if total_data_kb > 0:
            print(f"  Total data gathered: {total_data_kb:.1f} KB")
        if avg_deliverable_len > 0:
            print(f"  Avg response length: {avg_deliverable_len:,.0f} chars")
        print(f"  Avg visits per test: {stats.get('avg_visits', 0):.1f}")
        print(f"  Avg searches per test: {stats.get('avg_searches', 0):.1f}")
        if stats.get('avg_visits', 0) > 0 and avg_deliverable_len > 0:
            chars_per_visit = avg_deliverable_len / stats['avg_visits']
            print(f"  Response chars per visit: {chars_per_visit:,.0f}")

    # --- AGGREGATE IMPACT METRICS (resume-ready) ---
    graph_entries = [e for e in leaderboard if e["variant"] == "graph"]
    seq_entries = [e for e in leaderboard if e["variant"] == "sequential"]

    if graph_entries and seq_entries:
        g_avg = np.mean([e["avg_score"] for e in graph_entries])
        s_avg = np.mean([e["avg_score"] for e in seq_entries])
        g_pass = np.mean([e["pass_rate"] for e in graph_entries])
        s_pass = np.mean([e["pass_rate"] for e in seq_entries])
        g_cost = np.mean([e["avg_cost"] for e in graph_entries])
        s_cost = np.mean([e["avg_cost"] for e in seq_entries])
        g_tok = np.mean([e["avg_tokens"] for e in graph_entries])
        s_tok = np.mean([e["avg_tokens"] for e in seq_entries])
        g_dur = np.mean([e["avg_duration"] for e in graph_entries])
        s_dur = np.mean([e["avg_duration"] for e in seq_entries])

        def _pct(a, b):
            return ((a - b) / b * 100) if b != 0 else 0.0

        print(f"\n  GRAPH VS SEQUENTIAL — PERCENTAGE DELTAS")
        print(f"  {'Metric':<20} {'Graph':>10} {'Sequential':>12} {'Delta':>10} {'% Change':>10}")
        print(f"  {'-'*20:<20} {'-'*10:>10} {'-'*12:>12} {'-'*10:>10} {'-'*10:>10}")

        rows = [
            ("Avg Score", f"{g_avg:.3f}", f"{s_avg:.3f}", f"{g_avg - s_avg:+.3f}", f"{_pct(g_avg, s_avg):+.1f}%"),
            ("Pass Rate", f"{g_pass:.1%}", f"{s_pass:.1%}", f"{(g_pass - s_pass)*100:+.1f}pp", ""),
            ("Avg Cost", format_cost(g_cost), format_cost(s_cost), f"{g_cost - s_cost:+.4f}", f"{_pct(g_cost, s_cost):+.1f}%"),
            ("Avg Tokens", f"{g_tok:,.0f}", f"{s_tok:,.0f}", f"{g_tok - s_tok:+,.0f}", f"{_pct(g_tok, s_tok):+.1f}%"),
            ("Avg Duration", f"{g_dur:.1f}s", f"{s_dur:.1f}s", f"{g_dur - s_dur:+.1f}s", f"{_pct(g_dur, s_dur):+.1f}%"),
        ]
        for name, gv, sv, delta, pct in rows:
            print(f"  {name:<20} {gv:>10} {sv:>12} {delta:>10} {pct:>10}")

    # --- PER-MODEL PERCENTAGE COMPARISON ---
    if len(leaderboard) >= 2:
        print(f"\n  MODEL EFFICIENCY RANKINGS")
        print(f"  {'System':<30} {'Score/$':>10} {'Score/1Mtok':>12} {'$/point':>10}")
        print(f"  {'-'*30:<30} {'-'*10:>10} {'-'*12:>12} {'-'*10:>10}")
        for e in leaderboard:
            label = f"{e['model']} [{e['variant']}]"
            spd = e["avg_score"] / e["avg_cost"] if e["avg_cost"] > 0 else 0
            spt = e["avg_score"] / (e["avg_tokens"] / 1_000_000) if e["avg_tokens"] > 0 else 0
            cpp = e["avg_cost"] / e["avg_score"] if e["avg_score"] > 0 else 0
            print(f"  {label:<30} {spd:>10.1f} {spt:>12.2f} {format_cost(cpp):>10}")

    # --- AGGREGATE IMPACT NUMBERS ---
    total_runs = stats["total_tests"]
    if total_runs > 0:
        print(f"\n  AGGREGATE IMPACT NUMBERS")
        print(f"  Total test runs:           {total_runs}")
        print(f"  Overall accuracy:          {stats['avg_score']:.1%}")
        print(f"  Overall pass rate:         {stats['pass_rate']:.0%}")
        print(f"  Avg tokens per test:       {stats['avg_tokens']:,.0f}")
        print(f"  Avg cost per test:         {format_cost(stats.get('avg_cost_usd'))}")
        total_cost = stats.get("total_cost_usd", 0)
        if total_cost > 0 and stats["avg_score"] > 0:
            cost_per_point = total_cost / (stats["avg_score"] * total_runs)
            print(f"  Cost per accuracy point:   {format_cost(cost_per_point)}")
        total_tok = stats.get("total_tokens", 0)
        if total_tok > 0:
            score_per_mtok = (stats["avg_score"] * total_runs) / (total_tok / 1_000_000)
            print(f"  Score-points per 1M tok:   {score_per_mtok:.1f}")
        if stats["avg_visits"] > 0:
            print(f"  Score per visit:           {stats['avg_score'] / stats['avg_visits']:.3f}")
        if stats["avg_searches"] > 0:
            visit_search_ratio = stats["avg_visits"] / stats["avg_searches"]
            print(f"  Visit/search ratio:        {visit_search_ratio:.2f}")

    # --- Validation checks ---
    check_stats = stats.get("check_stats", {})
    if check_stats:
        print(f"\nSpecific Validation Checks:")
        zero_failures = []
        for check_name, check_data in check_stats.items():
            if check_data["total"] > 0:
                sr = check_data["success_rate"]
                score = check_data["avg_score"]
                total = check_data["total"]
                status = "[FAILED]" if sr == 0.0 else "[PARTIAL]" if sr < 0.5 else "[PASSED]"
                print(f"  {status} {check_name}: {sr:.1%} success rate, {score:.2f} avg score ({total} occurrences)")
                if sr == 0.0:
                    zero_failures.append(check_name)

        if zero_failures:
            print(f"\n[WARNING] Checks with 0% Success Rate: {', '.join(zero_failures)}")

    print("="*100)
