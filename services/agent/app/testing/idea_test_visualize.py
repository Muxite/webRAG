"""
Visualization system for idea test results.

This module re-exports functionality from the split visualization modules
for backward compatibility.
"""

import sys
from pathlib import Path

_file = Path(__file__).resolve()
_agent_dir = _file.parent.parent.parent

if __name__ == "__main__":
    if str(_agent_dir) not in sys.path:
        sys.path.insert(0, str(_agent_dir))
    import importlib.util
    _testing_dir = _file.parent
    
    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(f"app.testing.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "app.testing"
        mod.__name__ = f"app.testing.{name}"
        sys.modules[f"app.testing.{name}"] = mod
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    
    mod_helpers = _load_module("visualization_helpers", _testing_dir / "visualization_helpers.py")
    mod_data = _load_module("visualization_data", _testing_dir / "visualization_data.py")
    mod_summary = _load_module("visualization_summary", _testing_dir / "visualization_summary.py")
    mod_plots = _load_module("visualization_plots", _testing_dir / "visualization_plots.py")
    mod_main = _load_module("visualization_main", _testing_dir / "visualization_main.py")
    
    generate_all_plots = mod_main.generate_all_plots
    main = mod_main.main
    load_test_results = mod_data.load_test_results
    list_available_runs = mod_data.list_available_runs
    extract_run_id_from_filename = mod_data.extract_run_id_from_filename
    calculate_summary_stats = mod_summary.calculate_summary_stats
    print_summary = mod_summary.print_summary
    plot_validation_scores = mod_plots.plot_validation_scores
    plot_pass_rates = mod_plots.plot_pass_rates
    plot_execution_metrics = mod_plots.plot_execution_metrics
    plot_validation_breakdown = mod_plots.plot_validation_breakdown
    plot_difficulty_vs_performance = mod_plots.plot_difficulty_vs_performance
    plot_token_efficiency = mod_plots.plot_token_efficiency
    plot_time_vs_score = mod_plots.plot_time_vs_score
    plot_model_comparison = mod_plots.plot_model_comparison
    plot_check_success_rates = mod_plots.plot_check_success_rates
    plot_observability_timeline = mod_plots.plot_observability_timeline
    plot_score_distributions = mod_plots.plot_score_distributions
    plot_tokens_and_actions_dual_axis = mod_plots.plot_tokens_and_actions_dual_axis
    plot_specific_validation_checks = mod_plots.plot_specific_validation_checks
    plot_comprehensive_performance = mod_plots.plot_comprehensive_performance
    _system_label = mod_helpers._system_label
    _format_tokens = mod_helpers._format_tokens
    _get_system_colors = mod_helpers._get_system_colors
    _get_difficulty_colormap = mod_helpers._get_difficulty_colormap
else:
    from .visualization_main import generate_all_plots, main
    from .visualization_data import load_test_results, list_available_runs, extract_run_id_from_filename
    from .visualization_summary import calculate_summary_stats, print_summary
    from .visualization_plots import (
        plot_validation_scores,
        plot_pass_rates,
        plot_execution_metrics,
        plot_validation_breakdown,
        plot_difficulty_vs_performance,
        plot_token_efficiency,
        plot_time_vs_score,
        plot_model_comparison,
        plot_check_success_rates,
        plot_observability_timeline,
        plot_score_distributions,
        plot_tokens_and_actions_dual_axis,
        plot_specific_validation_checks,
        plot_comprehensive_performance,
    )
    from .visualization_helpers import (
        _system_label,
        _format_tokens,
        _get_system_colors,
        _get_difficulty_colormap,
    )

__all__ = [
    "generate_all_plots",
    "main",
    "load_test_results",
    "list_available_runs",
    "extract_run_id_from_filename",
    "calculate_summary_stats",
    "print_summary",
    "plot_validation_scores",
    "plot_pass_rates",
    "plot_execution_metrics",
    "plot_validation_breakdown",
    "plot_difficulty_vs_performance",
    "plot_token_efficiency",
    "plot_time_vs_score",
    "plot_model_comparison",
    "plot_check_success_rates",
    "plot_observability_timeline",
    "plot_score_distributions",
    "plot_tokens_and_actions_dual_axis",
    "plot_specific_validation_checks",
    "plot_comprehensive_performance",
    "_system_label",
    "_format_tokens",
    "_get_system_colors",
    "_get_difficulty_colormap",
]

if __name__ == "__main__":
    main()
