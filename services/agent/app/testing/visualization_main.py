"""
Main orchestration for visualization.

Simplified API for generating benchmark visualizations from test results.
Focuses on core plots needed for documentation.

Quick Start:
    # Generate the 3 plots needed for docs (auto-detects latest run)
    from agent.app.testing.visualization_main import generate_docs_plots
    mappings = generate_docs_plots(results_dir="idea_test_results")
    
    # Or use the script:
    python scripts/generate_benchmark_plots.py

The system generates 4 core images:
- core_p1_executive.png: Executive summary with scores, pass rates, KPIs
- core_p2_heatmap.png: Test × Model heatmap showing per-test performance
- core_p3_efficiency.png: Efficiency dashboard (cost, time, tokens, graph vs sequential)
- core_p4_details.png: Actions & structure (searches/visits, graph metrics)

For documentation, only the first 3 are typically used.
"""

from pathlib import Path
from typing import Optional

from .visualization_data import load_test_results, list_available_runs
from .visualization_core import generate_core_plots
from .visualization_summary import calculate_summary_stats, print_summary

def generate_docs_plots(results_dir: Path, output_dir: Optional[Path] = None,
                       run_id_filter: Optional[str] = None) -> dict:
    """
    Generate the 3 core plots needed for documentation.
    
    Returns a dict mapping source filenames to destination filenames:
    {
        "core_p1_executive.png": "executive_summary.png",
        "core_p2_heatmap.png": "score_heatmap.png",
        "core_p3_efficiency.png": "efficiency_dashboard.png",
    }
    
    :param results_dir: Directory containing test result JSON files.
    :param output_dir: Output directory (defaults to results_dir/plots_latest).
    :param run_id_filter: Optional run ID to filter by. If None, uses latest run.
    :return: Dict mapping source -> dest filenames for copying to docs.
    """
    if output_dir is None:
        output_dir = results_dir / "plots_latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect latest run if not specified
    if run_id_filter is None:
        runs = list_available_runs(results_dir)
        if runs:
            run_id_filter = runs[0]["run_id"]
            print(f"Using latest run ID: {run_id_filter} ({runs[0]['count']} test(s))")
        else:
            raise ValueError(f"No test results found in {results_dir}")
    
    results = load_test_results(results_dir, run_id_filter=run_id_filter)
    
    if not results:
        raise ValueError(f"No test results found in {results_dir} (run_id: {run_id_filter})")
    
    print(f"Generating core plots from {len(results)} test result(s)...")
    generate_core_plots(results, output_dir)
    
    return {
        "core_p1_executive.png": "executive_summary.png",
        "core_p2_heatmap.png": "score_heatmap.png",
        "core_p3_efficiency.png": "efficiency_dashboard.png",
    }


def generate_all_plots(results_dir: Path, output_dir: Optional[Path] = None,
                       run_id_filter: Optional[str] = None, core_only: bool = True):
    """
    Generate visualization plots from test results.
    
    For documentation use, prefer generate_docs_plots() which is simpler.
    
    :param results_dir: Directory containing test result JSON files.
    :param output_dir: Output directory (defaults to results_dir/plots or results_dir/plots_{run_id}).
    :param run_id_filter: Optional run ID to filter by.
    :param core_only: When True, only produce the 4 consolidated core images (default: True).
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

    # Detailed plots removed - use core_only=True for simplicity
    # If you need detailed plots, import from visualization_plots directly
    print("\nNote: Detailed plots are not generated by default.")
    print("Use core_only=True (default) for documentation plots, or import visualization_plots directly.")
    
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
        default=True,
        help="Only generate the 4 core summary images (default: True)",
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