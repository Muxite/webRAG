"""
Main orchestration for visualization.
"""

from pathlib import Path
from typing import Optional

from .visualization_data import load_test_results, list_available_runs
from .visualization_plots import (
    plot_validation_scores,
    plot_pass_rates,
    plot_execution_metrics,
    plot_token_efficiency,
    plot_time_vs_score,
    plot_tokens_and_actions_dual_axis,
    plot_comprehensive_performance,
    plot_cost_analysis,
    plot_graph_vs_sequential,
    plot_model_head_to_head,
    plot_score_heatmap,
    plot_executive_dashboard,
    plot_token_breakdown,
    plot_action_effectiveness,
    plot_per_test_model_comparison,
    plot_graph_structure,
    plot_data_coverage,
    plot_cost_efficiency_frontier,
)
from .visualization_core import generate_core_plots
from .visualization_summary import calculate_summary_stats, print_summary

def generate_all_plots(results_dir: Path, output_dir: Optional[Path] = None,
                       run_id_filter: Optional[str] = None, core_only: bool = False):
    """
    Generate visualization plots from test results.
    :param results_dir: Directory containing test result JSON files.
    :param output_dir: Output directory (defaults to results_dir/plots or results_dir/plots_{run_id}).
    :param run_id_filter: Optional run ID to filter by.
    :param core_only: When True, only produce the 4 consolidated core images.
    """
    if output_dir is None:
        if run_id_filter:
            output_dir = results_dir / f"plots_{run_id_filter}"
        else:
            output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = load_test_results(results_dir, run_id_filter=run_id_filter)
    
    if not results:
        print(f"No test results found in {results_dir}")
        if run_id_filter:
            print(f"  (filtered by run_id: {run_id_filter})")
        return
    
    run_info = ""
    if run_id_filter:
        run_info = f" (run_id: {run_id_filter})"
    
    print(f"Generating plots from {len(results)} test result(s){run_info}...")
    
    test_ids = set()
    models = set()
    variants = set()
    for result in results:
        test_ids.add(result.get("test_metadata", {}).get("test_id", "unknown"))
        models.add(result.get("model", "unknown"))
        variants.add(result.get("execution_variant", "graph"))
    
    print(f"Tests: {len(test_ids)} ({', '.join(sorted(test_ids))})")
    print(f"Models: {len(models)} ({', '.join(sorted(models))})")
    print(f"Variants: {len(variants)} ({', '.join(sorted(variants))})")

    print("\n--- Core Plots (4 consolidated images) ---")
    generate_core_plots(results, output_dir)

    if core_only:
        summary_stats = calculate_summary_stats(results)
        print_summary(summary_stats)
        return

    print("\n--- Detailed Plots ---")
    plot_validation_scores(results, output_dir)
    print("[OK] validation_scores.png")
    
    plot_pass_rates(results, output_dir)
    print("[OK] pass_rates.png")
    
    plot_execution_metrics(results, output_dir)
    print("[OK] execution_metrics.png")
    
    plot_token_efficiency(results, output_dir)
    print("[OK] token_efficiency.png")
    
    plot_time_vs_score(results, output_dir)
    print("[OK] time_vs_score.png")
    
    plot_tokens_and_actions_dual_axis(results, output_dir)
    print("[OK] tokens_and_actions_dual_axis.png")
    
    plot_comprehensive_performance(results, output_dir)
    print("[OK] comprehensive_distributions.png")
    print("[OK] comprehensive_correlations.png")
    print("[OK] comprehensive_summary.png")
    
    plot_cost_analysis(results, output_dir)
    print("[OK] cost_analysis.png")

    # --- New advanced plots ---
    try:
        plot_graph_vs_sequential(results, output_dir)
        print("[OK] graph_vs_sequential.png")
    except Exception:
        print("[SKIP] graph_vs_sequential.png (need both graph+sequential data)")

    try:
        plot_model_head_to_head(results, output_dir)
        print("[OK] model_head_to_head.png")
    except Exception:
        print("[SKIP] model_head_to_head.png (need 2+ models)")

    plot_score_heatmap(results, output_dir)
    print("[OK] score_heatmap.png")

    plot_executive_dashboard(results, output_dir)
    print("[OK] executive_dashboard.png")

    plot_token_breakdown(results, output_dir)
    print("[OK] token_breakdown.png")

    plot_action_effectiveness(results, output_dir)
    print("[OK] action_effectiveness.png")

    plot_per_test_model_comparison(results, output_dir)
    print("[OK] per_test_model_comparison.png")

    plot_graph_structure(results, output_dir)
    print("[OK] graph_structure.png")

    plot_data_coverage(results, output_dir)
    print("[OK] data_coverage.png")

    try:
        plot_cost_efficiency_frontier(results, output_dir)
        print("[OK] cost_efficiency_frontier.png")
    except Exception:
        print("[SKIP] cost_efficiency_frontier.png (need cost data)")

    print(f"\nAll plots saved to: {output_dir}")
    
    summary_stats = calculate_summary_stats(results)
    print_summary(summary_stats)


def main():
    """Main entry point for visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from idea test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots from default results directory
  python -m app.testing.idea_test_visualize
  
  # Generate plots from specific run (collects all files with that timestamp)
  python -m app.testing.idea_test_visualize --run-id 20260221_205459
  
  # Use most recent run automatically
  python -m app.testing.idea_test_visualize --latest
  
  # List available runs
  python -m app.testing.idea_test_visualize --list-runs
  
  # Custom results and output directories
  python -m app.testing.idea_test_visualize --results-dir ./results --output-dir ./plots
        """
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="idea_test_results",
        help="Directory containing test result JSON files (default: idea_test_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (defaults to results_dir/plots)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Filter results by run ID (e.g., 20260221_205459). Use 'latest' for most recent run.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent run ID automatically",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available run IDs and exit",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only generate the 4 core summary images (skip detailed plots)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        # Resolve results directory relative to the agent root (../.. from this file)
        # This matches where idea_test_runner and execution write: agent/idea_test_results
        results_dir = Path(__file__).resolve().parent.parent.parent / args.results_dir
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    if args.list_runs:
        runs = list_available_runs(results_dir)
        if runs:
            print(f"Available run IDs in {results_dir}:")
            print(f"{'Run ID':<20} {'Tests':<10} {'Latest File'}")
            print("-" * 70)
            for run_info in runs:
                latest = run_info["latest_file"][:40] + "..." if len(run_info["latest_file"]) > 40 else run_info["latest_file"]
                print(f"{run_info['run_id']:<20} {run_info['count']:<10} {latest}")
            print(f"\nTotal: {len(runs)} run(s)")
            if runs:
                print(f"Most recent: {runs[0]['run_id']} ({runs[0]['count']} test(s))")
        else:
            print(f"No test results found in {results_dir}")
        return
    
    run_id_filter = args.run_id
    if args.latest or (run_id_filter and run_id_filter.lower() == "latest"):
        runs = list_available_runs(results_dir)
        if runs:
            run_id_filter = runs[0]["run_id"]
            print(f"Using latest run ID: {run_id_filter} ({runs[0]['count']} test(s))")
        else:
            print("Error: No test results found to determine latest run")
            return
    
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent / args.output_dir
    
    generate_all_plots(results_dir, output_dir, run_id_filter=run_id_filter,
                       core_only=args.core_only)