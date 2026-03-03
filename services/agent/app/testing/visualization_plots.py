"""
Plot functions for visualization.
"""

from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from .visualization_helpers import (
    _system_label, _get_system_colors, _get_difficulty_colormap,
    _format_tokens, _extract_graph_metrics,
)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 24,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 36,
})

def plot_check_success_rates(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot success rates for validation checks.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    check_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    
    for result in results:
        validation = result.get("validation", {})
        grep_validations = validation.get("grep_validations", [])
        for check in grep_validations:
            check_name = check.get("check", "unknown")
            check_stats[check_name]["total"] += 1
            if check.get("passed", False):
                check_stats[check_name]["passed"] += 1
    
    check_names = sorted(check_stats.keys())
    success_rates = [check_stats[name]["passed"] / check_stats[name]["total"] if check_stats[name]["total"] > 0 else 0.0 for name in check_names]
    
    fig, ax = plt.subplots(figsize=(14, max(12, len(check_names) * 0.8)))
    
    colors = ["green" if sr >= 0.75 else "orange" if sr >= 0.5 else "red" for sr in success_rates]
    ax.barh(check_names, success_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    
    ax.set_xlabel("Success Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("Validation Check", fontsize=14, fontweight="bold")
    ax.set_title("Validation Check Success Rates", fontsize=24, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=16)
    
    for i, (name, rate) in enumerate(zip(check_names, success_rates)):
        ax.text(rate + 0.02, i, f"{rate:.1%}", va="center", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "check_success_rates.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comprehensive_performance(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot comprehensive performance metrics split into separate files.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    models = sorted(set(_system_label(r) for r in results))
    model_colors = _get_system_colors(models, "tab10")
    
    model_scores = defaultdict(list)
    model_tokens = defaultdict(list)
    model_durations = defaultdict(list)
    model_searches = defaultdict(list)
    model_visits = defaultdict(list)
    
    for result in results:
        model = _system_label(result)
        validation = result.get("validation", {})
        obs = result.get("execution", {}).get("observability", {})
        
        score = validation.get("overall_score", 0.0)
        tokens = obs.get("llm", {}).get("total_tokens", 0)
        duration = result.get("execution", {}).get("duration_seconds", 0)
        searches = obs.get("search", {}).get("count", 0)
        visits = obs.get("visit", {}).get("count", 0)
        
        model_scores[model].append(score)
        model_tokens[model].append(tokens)
        model_durations[model].append(duration)
        model_searches[model].append(searches)
        model_visits[model].append(visits)
    
    fig1, axes1 = plt.subplots(1, 3, figsize=(24, 9))
    fig1.suptitle("Performance Distributions by System", fontsize=14, fontweight="bold")
    
    x_positions = np.arange(len(models))
    width = 0.6
    
    for i, model in enumerate(models):
        scores = model_scores[model]
        axes1[0].scatter([i] * len(scores), scores, alpha=0.6, s=80, color=model_colors[model], edgecolors="black", linewidth=0.5)
        if scores:
            axes1[0].axhline(np.mean(scores), color=model_colors[model], linestyle="--", alpha=0.7, linewidth=2)
    axes1[0].set_ylabel("Score", fontsize=14)
    axes1[0].set_title("Score Distribution", fontsize=14, fontweight="bold")
    axes1[0].set_xticks(x_positions)
    axes1[0].set_xticklabels(models, fontsize=14)
    axes1[0].set_ylim(0, 1.0)
    axes1[0].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Pass Threshold")
    axes1[0].legend(fontsize=14)
    axes1[0].grid(alpha=0.3)
    
    for i, model in enumerate(models):
        tokens = model_tokens[model]
        axes1[1].scatter([i] * len(tokens), tokens, alpha=0.6, s=80, color=model_colors[model], edgecolors="black", linewidth=0.5)
    axes1[1].set_ylabel("Total Tokens", fontsize=14)
    axes1[1].set_title("Token Usage", fontsize=14, fontweight="bold")
    axes1[1].set_xticks(x_positions)
    axes1[1].set_xticklabels(models, fontsize=14)
    axes1[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_tokens(x)))
    axes1[1].grid(alpha=0.3)
    
    for i, model in enumerate(models):
        durations = model_durations[model]
        axes1[2].scatter([i] * len(durations), durations, alpha=0.6, s=80, color=model_colors[model], edgecolors="black", linewidth=0.5)
    axes1[2].set_ylabel("Duration (seconds)", fontsize=14)
    axes1[2].set_title("Execution Duration", fontsize=14, fontweight="bold")
    axes1[2].set_xticks(x_positions)
    axes1[2].set_xticklabels(models, fontsize=14)
    axes1[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 9))
    fig2.suptitle("Performance Correlations", fontsize=14, fontweight="bold")
    
    for model in models:
        scores = model_scores[model]
        tokens = model_tokens[model]
        axes2[0].scatter(tokens, scores, alpha=0.7, s=120, color=model_colors[model], label=model, edgecolors="black", linewidth=0.5)
    axes2[0].set_xlabel("Total Tokens", fontsize=14)
    axes2[0].set_ylabel("Score", fontsize=14)
    axes2[0].set_title("Score vs Token Usage", fontsize=14, fontweight="bold")
    axes2[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_tokens(x)))
    axes2[0].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes2[0].legend(fontsize=14, loc="best")
    axes2[0].grid(alpha=0.3)
    
    for model in models:
        durations = model_durations[model]
        scores = model_scores[model]
        axes2[1].scatter(durations, scores, alpha=0.7, s=120, color=model_colors[model], label=model, edgecolors="black", linewidth=0.5)
    axes2[1].set_xlabel("Duration (seconds)", fontsize=14)
    axes2[1].set_ylabel("Score", fontsize=14)
    axes2[1].set_title("Score vs Duration", fontsize=14, fontweight="bold")
    axes2[1].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes2[1].legend(fontsize=14, loc="best")
    axes2[1].grid(alpha=0.3)
    
    for model in models:
        searches = model_searches[model]
        visits = model_visits[model]
        axes2[2].scatter(searches, visits, alpha=0.7, s=120, color=model_colors[model], label=model, edgecolors="black", linewidth=0.5)
    axes2[2].set_xlabel("Search Count", fontsize=14)
    axes2[2].set_ylabel("Visit Count", fontsize=14)
    axes2[2].set_title("Searches vs Visits", fontsize=14, fontweight="bold")
    axes2[2].legend(fontsize=14, loc="best")
    axes2[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    fig3, axes3 = plt.subplots(1, 3, figsize=(24, 9))
    fig3.suptitle("System Summary Metrics", fontsize=14, fontweight="bold")
    
    model_avg_scores = [np.mean(model_scores[m]) if model_scores[m] else 0.0 for m in models]
    model_avg_tokens = [np.mean(model_tokens[m]) if model_tokens[m] else 0.0 for m in models]
    scatter1 = axes3[0].scatter(model_avg_tokens, model_avg_scores, s=250, c=range(len(models)), cmap="viridis", alpha=0.8, edgecolors="black", linewidth=2)
    for i, model in enumerate(models):
        axes3[0].annotate(model, (model_avg_tokens[i], model_avg_scores[i]), fontsize=14, fontweight="bold", ha="center", va="bottom")
    axes3[0].set_xlabel("Average Tokens", fontsize=14)
    axes3[0].set_ylabel("Average Score", fontsize=14)
    axes3[0].set_title("Model Efficiency", fontsize=14, fontweight="bold")
    axes3[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_tokens(x)))
    axes3[0].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes3[0].grid(alpha=0.3)
    
    model_avg_durations = [np.mean(model_durations[m]) if model_durations[m] else 0.0 for m in models]
    scatter2 = axes3[1].scatter(model_avg_durations, model_avg_scores, s=250, c=range(len(models)), cmap="plasma", alpha=0.8, edgecolors="black", linewidth=2)
    for i, model in enumerate(models):
        axes3[1].annotate(model, (model_avg_durations[i], model_avg_scores[i]), fontsize=14, fontweight="bold", ha="center", va="bottom")
    axes3[1].set_xlabel("Average Duration (seconds)", fontsize=14)
    axes3[1].set_ylabel("Average Score", fontsize=14)
    axes3[1].set_title("Speed vs Performance", fontsize=14, fontweight="bold")
    axes3[1].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes3[1].grid(alpha=0.3)
    
    model_pass_rates = []
    for model in models:
        passed = sum(1 for s in model_scores[model] if s >= 0.75)
        total = len(model_scores[model])
        model_pass_rates.append(passed / total if total > 0 else 0.0)
    
    bars = axes3[2].bar(x_positions, model_pass_rates, width, color=[model_colors[m] for m in models], alpha=0.8, edgecolor="black", linewidth=2)
    axes3[2].set_ylabel("Pass Rate", fontsize=14)
    axes3[2].set_title("Pass Rate by System", fontsize=14, fontweight="bold")
    axes3[2].set_xticks(x_positions)
    axes3[2].set_xticklabels(models, fontsize=14)
    axes3[2].set_ylim(0, 1.0)
    axes3[2].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Target")
    axes3[2].legend(fontsize=14)
    axes3[2].grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, model_pass_rates):
        axes3[2].text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate:.1%}", ha="center", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_difficulty_vs_performance(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot test difficulty vs performance.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    difficulty_scores = defaultdict(list)
    
    for result in results:
        metadata = result.get("test_metadata", {})
        difficulty_str = metadata.get("difficulty_level", "0/10")
        try:
            difficulty = float(difficulty_str.split("/")[0])
        except:
            difficulty = 0.0
        
        validation = result.get("validation", {})
        score = validation.get("overall_score", 0.0)
        difficulty_scores[difficulty].append(score)
    
    difficulties = sorted(difficulty_scores.keys())
    avg_scores = [np.mean(difficulty_scores[d]) for d in difficulties]
    std_scores = [np.std(difficulty_scores[d]) for d in difficulties]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    ax.errorbar(difficulties, avg_scores, yerr=std_scores, fmt="o-", capsize=5, capthick=2, alpha=0.7)
    ax.axhline(y=0.75, color="red", linestyle="--", label="Pass Threshold (75%)")
    
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Average Validation Score")
    ax.set_title("Test Difficulty vs Performance")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "difficulty_vs_performance.png", dpi=150)
    plt.close()


def plot_execution_metrics(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot execution metrics (duration, tokens, etc.).
    :param results: Test results.
    :param output_dir: Output directory.
    """
    metrics_data = {
        "duration": [],
        "tokens": [],
        "llm_calls": [],
        "searches": [],
        "visits": [],
        "test_ids": [],
    }
    
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        execution = result.get("execution", {})
        observability = execution.get("observability", {})
        
        metrics_data["test_ids"].append(test_id)
        metrics_data["duration"].append(execution.get("duration_seconds", 0))
        metrics_data["tokens"].append(observability.get("llm", {}).get("total_tokens", 0))
        metrics_data["llm_calls"].append(observability.get("llm", {}).get("calls", 0))
        metrics_data["searches"].append(observability.get("search", {}).get("count", 0))
        metrics_data["visits"].append(observability.get("visit", {}).get("count", 0))
    
    fig, axes = plt.subplots(2, 3, figsize=(26, 15))
    axes = axes.flatten()
    
    metrics = [
        ("duration", "Duration (seconds)", "Duration"),
        ("tokens", "Total Tokens", "Token Usage"),
        ("llm_calls", "LLM Calls", "LLM Call Count"),
        ("searches", "Search Count", "Search Actions"),
        ("visits", "Visit Count", "Visit Actions"),
    ]
    
    for idx, (key, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        test_ids = metrics_data["test_ids"]
        values = metrics_data[key]
        
        ax.scatter(test_ids, values, alpha=0.6, s=50)
        ax.set_xlabel("Test ID")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
    
    axes[5].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "execution_metrics.png", dpi=150)
    plt.close()


def plot_model_comparison(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot model comparison.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    model_metrics = defaultdict(lambda: {
        "scores": [],
        "durations": [],
        "tokens": [],
        "pass_count": 0,
        "total_count": 0,
    })
    
    for result in results:
        model = _system_label(result)
        execution = result.get("execution", {})
        validation = result.get("validation", {})
        observability = execution.get("observability", {})
        
        model_metrics[model]["scores"].append(validation.get("overall_score", 0.0))
        model_metrics[model]["durations"].append(execution.get("duration_seconds", 0))
        model_metrics[model]["tokens"].append(observability.get("llm", {}).get("total_tokens", 0))
        model_metrics[model]["total_count"] += 1
        if validation.get("overall_passed", False):
            model_metrics[model]["pass_count"] += 1
    
    models = sorted(model_metrics.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 15))
    
    avg_scores = [np.mean(model_metrics[m]["scores"]) for m in models]
    pass_rates = [model_metrics[m]["pass_count"] / model_metrics[m]["total_count"] if model_metrics[m]["total_count"] > 0 else 0.0 for m in models]
    avg_durations = [np.mean(model_metrics[m]["durations"]) for m in models]
    avg_tokens = [np.mean(model_metrics[m]["tokens"]) for m in models]
    
    axes[0, 0].bar(models, avg_scores, alpha=0.7, color="steelblue")
    axes[0, 0].set_ylabel("Average Score")
    axes[0, 0].set_title("Average Validation Score by System")
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].grid(axis="y", alpha=0.3)
    
    axes[0, 1].bar(models, pass_rates, alpha=0.7, color="green")
    axes[0, 1].set_ylabel("Pass Rate")
    axes[0, 1].set_title("Pass Rate by System")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    axes[1, 0].bar(models, avg_durations, alpha=0.7, color="orange")
    axes[1, 0].set_ylabel("Average Duration (seconds)")
    axes[1, 0].set_title("Average Execution Time by System")
    axes[1, 0].grid(axis="y", alpha=0.3)
    
    axes[1, 1].bar(models, avg_tokens, alpha=0.7, color="purple")
    axes[1, 1].set_ylabel("Average Tokens")
    axes[1, 1].set_title("Average Token Usage by System")
    axes[1, 1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()


def plot_observability_timeline(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot observability metrics over time using timestamps.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    from datetime import datetime
    
    time_data = []
    
    for result in results:
        timestamp_str = result.get("timestamp", "")
        if not timestamp_str:
            run_id = result.get("execution", {}).get("telemetry", {}).get("correlation_id", "")
            if "_" in run_id:
                parts = run_id.split("_")
                if len(parts) >= 3:
                    try:
                        timestamp_str = f"{parts[-2]}_{parts[-1]}"
                        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    except:
                        continue
                else:
                    continue
            else:
                continue
        else:
            try:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except:
                continue
        
        execution = result.get("execution", {})
        validation = result.get("validation", {})
        observability = execution.get("observability", {})
        model = _system_label(result)
        
        time_data.append({
            "timestamp": dt,
            "score": validation.get("overall_score", 0.0),
            "tokens": observability.get("llm", {}).get("total_tokens", 0),
            "duration": execution.get("duration_seconds", 0),
            "model": model,
        })
    
    if not time_data:
        return
    
    time_data.sort(key=lambda x: x["timestamp"])
    
    timestamps = [d["timestamp"] for d in time_data]
    scores = [d["score"] for d in time_data]
    models = sorted(set(d["model"] for d in time_data))
    
    fig, axes = plt.subplots(2, 1, figsize=(24, 15))
    fig.suptitle("Performance Over Time", fontsize=14, fontweight="bold")
    
    model_colors = _get_system_colors(models, "tab10")
    
    for model in models:
        model_times = [d["timestamp"] for d in time_data if d["model"] == model]
        model_scores = [d["score"] for d in time_data if d["model"] == model]
        axes[0].plot(model_times, model_scores, marker="o", linestyle="-", linewidth=2, markersize=6, 
                    label=model, color=model_colors[model], alpha=0.7)
    
    axes[0].axhline(y=0.75, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Pass Threshold")
    axes[0].set_ylabel("Validation Score", fontsize=14, fontweight="bold")
    axes[0].set_title("Score Over Time by System", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)
    
    for model in models:
        model_times = [d["timestamp"] for d in time_data if d["model"] == model]
        model_tokens = [d["tokens"] for d in time_data if d["model"] == model]
        axes[1].plot(model_times, model_tokens, marker="s", linestyle="-", linewidth=2, markersize=6,
                    label=model, color=model_colors[model], alpha=0.7)
    
    axes[1].set_xlabel("Timestamp", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Total Tokens", fontsize=14, fontweight="bold")
    axes[1].set_title("Token Usage Over Time by System", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=14)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "observability_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_pass_rates(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot pass rates by test, sorted by success rate.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    test_pass_rates = defaultdict(lambda: {"passed": 0, "total": 0})
    
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        validation = result.get("validation", {})
        passed = validation.get("overall_passed", False)
        test_pass_rates[test_id]["total"] += 1
        if passed:
            test_pass_rates[test_id]["passed"] += 1
    
    test_ids_with_rates = [
        (tid, test_pass_rates[tid]["passed"] / test_pass_rates[tid]["total"] if test_pass_rates[tid]["total"] > 0 else 0.0)
        for tid in test_pass_rates.keys()
    ]
    test_ids_sorted = [tid for tid, _ in sorted(test_ids_with_rates, key=lambda x: x[1], reverse=True)]
    pass_rates = [test_pass_rates[tid]["passed"] / test_pass_rates[tid]["total"] if test_pass_rates[tid]["total"] > 0 else 0.0 for tid in test_ids_sorted]
    
    fig, ax = plt.subplots(figsize=(22, 13))
    
    colors = ["green" if pr >= 0.75 else "orange" if pr >= 0.5 else "red" for pr in pass_rates]
    bars = ax.bar(test_ids_sorted, pass_rates, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    ax.axhline(y=0.75, color="blue", linestyle="--", linewidth=2, label="Pass Threshold (75%)")
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate:.1%}", 
               ha="center", va="bottom", fontsize=14, fontweight="bold")
    
    ax.set_xlabel("Test ID (sorted by success rate)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Pass Rate", fontsize=14, fontweight="bold")
    ax.set_title("Pass Rates by Test (Sorted by Success Rate)", fontsize=24, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distributions(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot score distributions showing granular performance, not just pass/fail.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(32, 26))
    fig.suptitle("Score Distributions - Granular Performance Analysis", fontsize=14, fontweight="bold")
    
    all_scores = [r.get("validation", {}).get("overall_score", 0.0) for r in results]
    models = sorted(set(_system_label(r) for r in results))
    test_ids = sorted(set(r.get("test_metadata", {}).get("test_id", "unknown") for r in results))
    
    axes[0, 0].hist(all_scores, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].axvline(np.mean(all_scores), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(all_scores):.2f}")
    axes[0, 0].axvline(0.75, color="orange", linestyle="--", linewidth=2, label="Pass Threshold: 0.75")
    axes[0, 0].set_xlabel("Overall Score", fontsize=14)
    axes[0, 0].set_ylabel("Frequency", fontsize=14)
    axes[0, 0].set_title("Overall Score Distribution", fontsize=14, fontweight="bold")
    axes[0, 0].legend(fontsize=14)
    axes[0, 0].grid(alpha=0.3)
    
    model_scores = defaultdict(list)
    for result in results:
        model = _system_label(result)
        score = result.get("validation", {}).get("overall_score", 0.0)
        model_scores[model].append(score)
    
    model_means = [np.mean(model_scores[m]) for m in models]
    model_stds = [np.std(model_scores[m]) for m in models]
    system_colors = _get_system_colors(models, "Set3")
    colors = [system_colors[m] for m in models]
    bars = axes[0, 1].bar(models, model_means, yerr=model_stds, capsize=5, alpha=0.8, color=colors, edgecolor="black")
    axes[0, 1].axhline(0.75, color="red", linestyle="--", linewidth=2, label="Pass Threshold")
    axes[0, 1].set_ylabel("Average Score", fontsize=14)
    axes[0, 1].set_title("Score by System (with std dev)", fontsize=14, fontweight="bold")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].legend(fontsize=14)
    axes[0, 1].grid(axis="y", alpha=0.3)
    for i, (bar, mean) in enumerate(zip(bars, model_means)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, mean + model_stds[i] + 0.02, f"{mean:.2f}", ha="center", fontsize=14, fontweight="bold")
    
    test_scores = defaultdict(list)
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        score = result.get("validation", {}).get("overall_score", 0.0)
        test_scores[test_id].append(score)
    
    test_means = [np.mean(test_scores[tid]) for tid in test_ids]
    test_stds = [np.std(test_scores[tid]) for tid in test_ids]
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(test_ids)))
    bars2 = axes[1, 0].barh(test_ids, test_means, xerr=test_stds, capsize=5, alpha=0.8, color=colors2, edgecolor="black")
    axes[1, 0].axvline(0.75, color="red", linestyle="--", linewidth=2, label="Pass Threshold")
    axes[1, 0].set_xlabel("Average Score", fontsize=14)
    axes[1, 0].set_ylabel("Test ID", fontsize=14)
    axes[1, 0].set_title("Score by Test (with std dev)", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlim(0, 1.0)
    axes[1, 0].legend(fontsize=14)
    axes[1, 0].grid(axis="x", alpha=0.3)
    for i, (bar, mean) in enumerate(zip(bars2, test_means)):
        axes[1, 0].text(mean + test_stds[i] + 0.02, bar.get_y() + bar.get_height()/2, f"{mean:.2f}", va="center", fontsize=14, fontweight="bold")
    
    score_ranges = [(0.0, 0.5, "Fail (0-0.5)"), (0.5, 0.75, "Partial (0.5-0.75)"), (0.75, 1.0, "Pass (0.75-1.0)")]
    range_counts = []
    range_labels = []
    for low, high, label in score_ranges:
        count = sum(1 for s in all_scores if low <= s < high)
        range_counts.append(count)
        range_labels.append(label)
    
    colors3 = ["red", "orange", "green"]
    wedges, texts, autotexts = axes[1, 1].pie(range_counts, labels=range_labels, autopct="%1.1f%%", colors=colors3, startangle=90, textprops={"fontsize": 11})
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    axes[1, 1].set_title("Score Range Distribution", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_specific_validation_checks(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot specific validation checks that are failing (0% success rate).
    :param results: Test results.
    :param output_dir: Output directory.
    """
    target_checks = ["user_quotes", "synthesis_quality", "source_urls", "source_coverage", "languages_count", "historical_quote", "data_numbers"]
    
    check_data = defaultdict(lambda: {"scores": [], "passed": 0, "total": 0})
    
    for result in results:
        validation = result.get("validation", {})
        grep_validations = validation.get("grep_validations", [])
        llm_validation = validation.get("llm_validation", {})
        
        for check in grep_validations:
            check_name = check.get("check", "")
            if check_name in target_checks:
                score = check.get("score", 0.0)
                passed = check.get("passed", False)
                check_data[check_name]["scores"].append(score)
                check_data[check_name]["total"] += 1
                if passed:
                    check_data[check_name]["passed"] += 1
        
        if llm_validation:
            details = llm_validation.get("details", {})
            if "synthesis_quality" in str(details):
                score = llm_validation.get("score", 0.0)
                passed = llm_validation.get("passed", False)
                check_data["synthesis_quality"]["scores"].append(score)
                check_data["synthesis_quality"]["total"] += 1
                if passed:
                    check_data["synthesis_quality"]["passed"] += 1
    
    found_checks = [c for c in target_checks if check_data[c]["total"] > 0]
    
    if not found_checks:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(28, 34))
    fig.suptitle("Specific Validation Checks - Performance Analysis", fontsize=14, fontweight="bold")
    
    check_names = found_checks
    success_rates = [check_data[c]["passed"] / check_data[c]["total"] if check_data[c]["total"] > 0 else 0.0 for c in check_names]
    avg_scores = [np.mean(check_data[c]["scores"]) if check_data[c]["scores"] else 0.0 for c in check_names]
    total_counts = [check_data[c]["total"] for c in check_names]
    
    colors = ["red" if sr == 0.0 else "orange" if sr < 0.5 else "green" for sr in success_rates]
    
    bars = axes[0, 0].barh(check_names, success_rates, color=colors, alpha=0.8, edgecolor="black")
    axes[0, 0].axvline(0.75, color="blue", linestyle="--", linewidth=2, label="Pass Threshold (75%)")
    axes[0, 0].set_xlabel("Success Rate", fontsize=14)
    axes[0, 0].set_ylabel("Validation Check", fontsize=14)
    axes[0, 0].set_title("Success Rate by Check Type", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlim(0, 1.0)
    axes[0, 0].legend(fontsize=14)
    axes[0, 0].grid(axis="x", alpha=0.3)
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        axes[0, 0].text(rate + 0.02, bar.get_y() + bar.get_height()/2, f"{rate:.1%}", va="center", fontsize=14, fontweight="bold")
    
    bars = axes[0, 1].barh(check_names, avg_scores, color=colors, alpha=0.8, edgecolor="black")
    axes[0, 1].axvline(0.75, color="blue", linestyle="--", linewidth=2, label="Pass Threshold (75%)")
    axes[0, 1].set_xlabel("Average Score", fontsize=14)
    axes[0, 1].set_ylabel("Validation Check", fontsize=14)
    axes[0, 1].set_title("Average Score by Check Type", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlim(0, 1.0)
    axes[0, 1].legend(fontsize=14)
    axes[0, 1].grid(axis="x", alpha=0.3)
    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        axes[0, 1].text(score + 0.02, bar.get_y() + bar.get_height()/2, f"{score:.2f}", va="center", fontsize=14, fontweight="bold")
    
    bars = axes[1, 0].barh(check_names, total_counts, color="steelblue", alpha=0.8, edgecolor="black")
    axes[1, 0].set_xlabel("Total Occurrences", fontsize=14)
    axes[1, 0].set_ylabel("Validation Check", fontsize=14)
    axes[1, 0].set_title("Check Occurrence Count", fontsize=14, fontweight="bold")
    axes[1, 0].grid(axis="x", alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, total_counts)):
        axes[1, 0].text(count + max(total_counts)*0.02, bar.get_y() + bar.get_height()/2, f"{count}", va="center", fontsize=14, fontweight="bold")
    
    zero_failures = [c for c, sr in zip(check_names, success_rates) if sr == 0.0]
    if zero_failures:
        axes[1, 1].text(0.5, 0.5, f"Checks with 0% Success Rate:\n\n" + "\n".join(f"• {c}" for c in zero_failures), 
                       ha="center", va="center", fontsize=14, fontweight="bold", 
                       bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8))
    else:
        axes[1, 1].text(0.5, 0.5, "No checks with 0% success rate", 
                       ha="center", va="center", fontsize=14, fontweight="bold",
                       bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "specific_validation_checks.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_vs_score(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot execution time vs validation score.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    test_scores = defaultdict(list)
    
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        validation = result.get("validation", {})
        score = validation.get("overall_score", 0.0)
        test_scores[test_id].append(score)
    
    avg_scores_by_test = {test_id: np.mean(scores) for test_id, scores in test_scores.items()}
    sorted_tests = sorted(avg_scores_by_test.items(), key=lambda x: x[1])
    test_to_color_index = {test_id: idx for idx, (test_id, _) in enumerate(sorted_tests)}
    
    durations = []
    scores = []
    test_ids = []
    color_indices = []
    
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        execution = result.get("execution", {})
        validation = result.get("validation", {})
        
        duration = execution.get("duration_seconds", 0)
        score = validation.get("overall_score", 0.0)
        
        if duration > 0:
            durations.append(duration)
            scores.append(score)
            test_ids.append(test_id)
            color_indices.append(test_to_color_index.get(test_id, 0))
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    max_color_idx = max(color_indices) if color_indices else 1
    difficulty_cmap = _get_difficulty_colormap()
    reversed_indices = [max_color_idx - idx for idx in color_indices]
    scatter = ax.scatter(durations, scores, alpha=0.6, s=100, c=reversed_indices, cmap=difficulty_cmap, vmin=0, vmax=max_color_idx)
    
    ax.set_xlabel("Execution Time (seconds)")
    ax.set_ylabel("Validation Score")
    ax.set_title("Time Efficiency (Duration vs Score)")
    ax.grid(alpha=0.3)
    ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.5, label="Pass Threshold")
    
    cbar = plt.colorbar(scatter, label="Test Difficulty (by avg score)")
    cbar.set_ticks(range(0, max_color_idx + 1, max(1, max_color_idx // 10)))
    cbar.set_ticklabels([sorted_tests[i][0] if i < len(sorted_tests) else "" for i in range(0, max_color_idx + 1, max(1, max_color_idx // 10))])
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_score.png", dpi=150)
    plt.close()


def plot_token_efficiency(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot token usage vs validation score.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    test_scores = defaultdict(list)
    
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        validation = result.get("validation", {})
        score = validation.get("overall_score", 0.0)
        test_scores[test_id].append(score)
    
    avg_scores_by_test = {test_id: np.mean(scores) for test_id, scores in test_scores.items()}
    sorted_tests = sorted(avg_scores_by_test.items(), key=lambda x: x[1])
    test_to_color_index = {test_id: idx for idx, (test_id, _) in enumerate(sorted_tests)}
    
    tokens = []
    scores = []
    test_ids = []
    color_indices = []
    
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        execution = result.get("execution", {})
        observability = execution.get("observability", {})
        validation = result.get("validation", {})
        
        token_count = observability.get("llm", {}).get("total_tokens", 0)
        score = validation.get("overall_score", 0.0)
        
        if token_count > 0:
            tokens.append(token_count)
            scores.append(score)
            test_ids.append(test_id)
            color_indices.append(test_to_color_index.get(test_id, 0))
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    max_color_idx = max(color_indices) if color_indices else 1
    difficulty_cmap = _get_difficulty_colormap()
    reversed_indices = [max_color_idx - idx for idx in color_indices]
    scatter = ax.scatter(tokens, scores, alpha=0.6, s=100, c=reversed_indices, cmap=difficulty_cmap, vmin=0, vmax=max_color_idx)
    
    ax.set_xlabel("Total Tokens Used")
    ax.set_ylabel("Validation Score")
    ax.set_title("Token Efficiency (Tokens vs Score)")
    ax.grid(alpha=0.3)
    ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.5, label="Pass Threshold")
    
    cbar = plt.colorbar(scatter, label="Test Difficulty (by avg score)")
    cbar.set_ticks(range(0, max_color_idx + 1, max(1, max_color_idx // 10)))
    cbar.set_ticklabels([sorted_tests[i][0] if i < len(sorted_tests) else "" for i in range(0, max_color_idx + 1, max(1, max_color_idx // 10))])
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "token_efficiency.png", dpi=150)
    plt.close()


def plot_tokens_and_actions_dual_axis(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot tokens vs searches/visits using dual y-axis.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(26, 20))
    fig.suptitle("Token Usage vs Actions & Data Amounts - Comprehensive Analysis", fontsize=14, fontweight="bold")
    
    test_ids = sorted(set(r.get("test_metadata", {}).get("test_id", "unknown") for r in results))
    models = sorted(set(_system_label(r) for r in results))
    
    test_tokens = defaultdict(list)
    test_searches = defaultdict(list)
    test_visits = defaultdict(list)
    test_search_data = defaultdict(list)
    test_visit_data = defaultdict(list)
    
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        obs = result.get("execution", {}).get("observability", {})
        tokens = max(0, obs.get("llm", {}).get("total_tokens", 0))
        searches = max(0, obs.get("search", {}).get("count", 0))
        visits = max(0, obs.get("visit", {}).get("count", 0))
        search_kb = max(0.0, obs.get("search", {}).get("kilobytes", 0.0))
        visit_kb = max(0.0, obs.get("visit", {}).get("kilobytes", 0.0))
        
        test_tokens[test_id].append(tokens)
        test_searches[test_id].append(searches)
        test_visits[test_id].append(visits)
        test_search_data[test_id].append(search_kb)
        test_visit_data[test_id].append(visit_kb)
    
    test_ids_sorted = sorted(test_ids, key=lambda tid: np.mean(test_tokens[tid]) if test_tokens[tid] else 0, reverse=True)
    
    avg_tokens = [np.mean(test_tokens[tid]) if test_tokens[tid] else 0 for tid in test_ids_sorted]
    avg_searches = [np.mean(test_searches[tid]) if test_searches[tid] else 0 for tid in test_ids_sorted]
    avg_visits = [np.mean(test_visits[tid]) if test_visits[tid] else 0 for tid in test_ids_sorted]
    avg_search_data = [np.mean(test_search_data[tid]) if test_search_data[tid] else 0.0 for tid in test_ids_sorted]
    avg_visit_data = [np.mean(test_visit_data[tid]) if test_visit_data[tid] else 0.0 for tid in test_ids_sorted]
    
    ax1 = axes[0, 0]
    ax2 = ax1.twinx()
    
    x = np.arange(len(test_ids_sorted))
    width = 0.25
    
    bars1 = ax1.bar(x - width, avg_tokens, width, label="Tokens", alpha=0.8, color="steelblue", edgecolor="black")
    bars2 = ax2.bar(x, avg_searches, width, label="Search Count", alpha=0.8, color="coral", edgecolor="black")
    bars3 = ax2.bar(x + width, avg_visits, width, label="Visit Count", alpha=0.8, color="mediumseagreen", edgecolor="black")
    
    ax1.set_xlabel("Test ID (sorted by token usage)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Tokens", fontsize=14, color="steelblue", fontweight="bold")
    ax2.set_ylabel("Action Count", fontsize=14, color="black", fontweight="bold")
    ax1.set_title("Tokens vs Searches vs Visits by Test", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_ids_sorted, rotation=45, ha="right", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=14)
    ax2.legend(loc="upper right", fontsize=14)
    ax1.grid(alpha=0.3)
    
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width, avg_tokens, width, label="Tokens", alpha=0.8, color="steelblue", edgecolor="black")
    bars2 = ax2.bar(x, avg_search_data, width, label="Search Data (KB)", alpha=0.8, color="orange", edgecolor="black")
    bars3 = ax2.bar(x + width, avg_visit_data, width, label="Visit Data (KB)", alpha=0.8, color="purple", edgecolor="black")
    
    ax1.set_xlabel("Test ID (sorted by token usage)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Tokens", fontsize=14, color="steelblue", fontweight="bold")
    ax2.set_ylabel("Data Amount (KB)", fontsize=14, color="black", fontweight="bold")
    ax1.set_title("Tokens vs Search/Visit Data Amounts by Test", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_ids_sorted, rotation=45, ha="right", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=14)
    ax2.legend(loc="upper right", fontsize=14)
    ax1.grid(alpha=0.3)
    
    model_tokens = defaultdict(list)
    model_searches = defaultdict(list)
    model_visits = defaultdict(list)
    model_search_data = defaultdict(list)
    model_visit_data = defaultdict(list)
    
    for result in results:
        model = _system_label(result)
        obs = result.get("execution", {}).get("observability", {})
        tokens = max(0, obs.get("llm", {}).get("total_tokens", 0))
        searches = max(0, obs.get("search", {}).get("count", 0))
        visits = max(0, obs.get("visit", {}).get("count", 0))
        search_kb = max(0.0, obs.get("search", {}).get("kilobytes", 0.0))
        visit_kb = max(0.0, obs.get("visit", {}).get("kilobytes", 0.0))
        
        model_tokens[model].append(tokens)
        model_searches[model].append(searches)
        model_visits[model].append(visits)
        model_search_data[model].append(search_kb)
        model_visit_data[model].append(visit_kb)
    
    avg_tokens_model = [np.mean(model_tokens[m]) if model_tokens[m] else 0 for m in models]
    avg_searches_model = [np.mean(model_searches[m]) if model_searches[m] else 0 for m in models]
    avg_visits_model = [np.mean(model_visits[m]) if model_visits[m] else 0 for m in models]
    avg_search_data_model = [np.mean(model_search_data[m]) if model_search_data[m] else 0.0 for m in models]
    avg_visit_data_model = [np.mean(model_visit_data[m]) if model_visit_data[m] else 0.0 for m in models]
    
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    
    x = np.arange(len(models))
    bars1 = ax1.bar(x - width, avg_tokens_model, width, label="Tokens", alpha=0.8, color="steelblue", edgecolor="black")
    bars2 = ax2.bar(x, avg_searches_model, width, label="Search Count", alpha=0.8, color="coral", edgecolor="black")
    bars3 = ax2.bar(x + width, avg_visits_model, width, label="Visit Count", alpha=0.8, color="mediumseagreen", edgecolor="black")
    
    ax1.set_xlabel("System", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Tokens", fontsize=14, color="steelblue", fontweight="bold")
    ax2.set_ylabel("Action Count", fontsize=14, color="black", fontweight="bold")
    ax1.set_title("Tokens vs Searches vs Visits by System", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=14)
    ax2.legend(loc="upper right", fontsize=14)
    ax1.grid(alpha=0.3)
    
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width, avg_tokens_model, width, label="Tokens", alpha=0.8, color="steelblue", edgecolor="black")
    bars2 = ax2.bar(x, avg_search_data_model, width, label="Search Data (KB)", alpha=0.8, color="orange", edgecolor="black")
    bars3 = ax2.bar(x + width, avg_visit_data_model, width, label="Visit Data (KB)", alpha=0.8, color="purple", edgecolor="black")
    
    ax1.set_xlabel("System", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Tokens", fontsize=14, color="steelblue", fontweight="bold")
    ax2.set_ylabel("Data Amount (KB)", fontsize=14, color="black", fontweight="bold")
    ax1.set_title("Tokens vs Search/Visit Data Amounts by System", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=14)
    ax2.legend(loc="upper right", fontsize=14)
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "tokens_and_actions_dual_axis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_validation_breakdown(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot validation breakdown by check type.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    check_scores = defaultdict(list)
    check_names = set()
    
    for result in results:
        validation = result.get("validation", {})
        grep_validations = validation.get("grep_validations", [])
        for check in grep_validations:
            check_name = check.get("check", "unknown")
            check_names.add(check_name)
            score = check.get("score", 0.0)
            check_scores[check_name].append(score)
    
    check_names = sorted(check_names)
    scores = [np.mean(check_scores[name]) if check_scores[name] else 0.0 for name in check_names]
    
    fig, ax = plt.subplots(figsize=(14, max(12, len(check_names) * 0.8)))
    
    colors = ["green" if s >= 0.75 else "orange" if s >= 0.5 else "red" for s in scores]
    ax.barh(check_names, scores, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    
    ax.set_xlabel("Average Score", fontsize=14, fontweight="bold")
    ax.set_ylabel("Validation Check", fontsize=14, fontweight="bold")
    ax.set_title("Validation Check Performance", fontsize=24, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=16)
    
    for i, (name, score) in enumerate(zip(check_names, scores)):
        ax.text(score + 0.02, i, f"{score:.2f}", va="center", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "validation_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_validation_scores(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot validation scores by test and model, sorted by success rate.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    test_scores = defaultdict(lambda: defaultdict(list))
    test_pass_rates = defaultdict(lambda: {"passed": 0, "total": 0})
    
    for result in results:
        test_id = result.get("test_metadata", {}).get("test_id", "unknown")
        model = _system_label(result)
        validation = result.get("validation", {})
        score = validation.get("overall_score", 0.0)
        test_scores[test_id][model].append(score)
        test_pass_rates[test_id]["total"] += 1
        if validation.get("overall_passed", False):
            test_pass_rates[test_id]["passed"] += 1
    
    test_ids_with_rates = [
        (tid, test_pass_rates[tid]["passed"] / test_pass_rates[tid]["total"] if test_pass_rates[tid]["total"] > 0 else 0.0)
        for tid in test_scores.keys()
    ]
    test_ids_sorted = [tid for tid, _ in sorted(test_ids_with_rates, key=lambda x: x[1], reverse=True)]
    models = sorted(set(model for scores in test_scores.values() for model in scores.keys()))
    
    fig, ax = plt.subplots(figsize=(24, 13))
    
    x = np.arange(len(test_ids_sorted))
    width = 0.35
    
    system_colors = _get_system_colors(models, "Set3")
    for i, model in enumerate(models):
        scores = [np.mean(test_scores[tid][model]) if model in test_scores[tid] else 0.0 for tid in test_ids_sorted]
        bars = ax.bar(x + i * width, scores, width, label=model, alpha=0.8, color=system_colors[model], edgecolor="black", linewidth=1)
        for j, (bar, score) in enumerate(zip(bars, scores)):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, score + 0.02, f"{score:.2f}", 
                       ha="center", va="bottom", fontsize=14, fontweight="bold")
    
    ax.set_xlabel("Test ID (sorted by success rate)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Validation Score", fontsize=14, fontweight="bold")
    ax.set_title("Validation Scores by Test and System (Sorted by Success Rate)", fontsize=24, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(test_ids_sorted, fontsize=14)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Pass Threshold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "validation_scores.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost_analysis(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot estimated USD costs per model/variant using model_costs pricing.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    from agent.app.model_costs import MODEL_PRICING, estimate_cost, estimate_cost_from_total, format_cost

    models = sorted(set(_system_label(r) for r in results))
    model_colors = _get_system_colors(models, "tab10")

    model_costs = defaultdict(list)
    model_scores = defaultdict(list)
    model_tokens = defaultdict(list)

    for result in results:
        label = _system_label(result)
        raw_model = str(result.get("model", "unknown"))
        obs = result.get("execution", {}).get("observability", {})
        llm = obs.get("llm", {})
        input_tokens = llm.get("prompt", {}).get("tokens", 0)
        output_tokens = llm.get("completion", {}).get("tokens", 0)
        total_tokens = llm.get("total_tokens", 0)
        score = result.get("validation", {}).get("overall_score", 0.0)

        if input_tokens > 0 or output_tokens > 0:
            cost = estimate_cost(raw_model, input_tokens, output_tokens)
        else:
            cost = estimate_cost_from_total(raw_model, total_tokens)
        if cost is not None:
            model_costs[label].append(cost)
        model_scores[label].append(score)
        model_tokens[label].append(total_tokens)

    fig, axes = plt.subplots(1, 3, figsize=(28, 11))
    fig.suptitle("Cost Analysis by System", fontsize=14, fontweight="bold")

    x = np.arange(len(models))
    avg_costs = [np.mean(model_costs[m]) if model_costs[m] else 0.0 for m in models]
    colors = [model_colors.get(m, (0.5, 0.5, 0.5, 1.0)) for m in models]

    bars = axes[0].bar(x, avg_costs, color=colors, alpha=0.8, edgecolor="black", linewidth=1)
    for i, (m, c) in enumerate(zip(models, avg_costs)):
        axes[0].text(i, c + max(avg_costs) * 0.02, format_cost(c), ha="center", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=14)
    axes[0].set_ylabel("Avg Cost per Test (USD)", fontsize=14)
    axes[0].set_title("Average Cost per Test", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    for m in models:
        costs = model_costs.get(m, [])
        scores = model_scores.get(m, [])
        min_len = min(len(costs), len(scores))
        if min_len > 0:
            axes[1].scatter(
                costs[:min_len], scores[:min_len],
                alpha=0.7, s=120, color=model_colors.get(m),
                label=m, edgecolors="black", linewidth=0.5,
            )
    axes[1].set_xlabel("Cost (USD)", fontsize=14)
    axes[1].set_ylabel("Score", fontsize=14)
    axes[1].set_title("Score vs Cost", fontsize=14, fontweight="bold")
    axes[1].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes[1].legend(fontsize=14, loc="best")
    axes[1].grid(alpha=0.3)

    avg_scores = [np.mean(model_scores[m]) if model_scores[m] else 0.0 for m in models]
    efficiency = [s / c if c > 0 else 0.0 for s, c in zip(avg_scores, avg_costs)]
    bars2 = axes[2].bar(x, efficiency, color=colors, alpha=0.8, edgecolor="black", linewidth=1)
    for i, e in enumerate(efficiency):
        axes[2].text(i, e + max(efficiency) * 0.02 if efficiency else 0, f"{e:.1f}", ha="center", fontsize=14, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, fontsize=14)
    axes[2].set_ylabel("Score / USD", fontsize=14)
    axes[2].set_title("Cost Efficiency (Score per Dollar)", fontsize=14, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "cost_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_graph_vs_sequential(results: List[Dict[str, Any]], output_dir: Path):
    """
    Side-by-side comparison of graph vs sequential with percentage deltas.
    Shows accuracy gain, token usage difference, cost difference, and speed.
    """
    from agent.app.model_costs import estimate_cost, estimate_cost_from_total, format_cost

    graph_data: Dict[str, list] = defaultdict(list)
    seq_data: Dict[str, list] = defaultdict(list)

    for r in results:
        variant = str(r.get("execution_variant", "graph"))
        model = str(r.get("model", "unknown"))
        obs = r.get("execution", {}).get("observability", {})
        llm = obs.get("llm", {})
        score = r.get("validation", {}).get("overall_score", 0.0)
        passed = r.get("validation", {}).get("overall_passed", False)
        duration = r.get("execution", {}).get("duration_seconds", 0)
        tokens = llm.get("total_tokens", 0)
        inp = llm.get("prompt", {}).get("tokens", 0)
        out = llm.get("completion", {}).get("tokens", 0)
        cost = estimate_cost(model, inp, out) if (inp or out) else estimate_cost_from_total(model, tokens)
        cost = cost or 0.0
        visits = obs.get("visit", {}).get("count", 0)
        searches = obs.get("search", {}).get("count", 0)
        bucket = graph_data if variant == "graph" else seq_data

        bucket["scores"].append(score)
        bucket["passed"].append(passed)
        bucket["durations"].append(duration)
        bucket["tokens"].append(tokens)
        bucket["costs"].append(cost)
        bucket["visits"].append(visits)
        bucket["searches"].append(searches)

    if not graph_data["scores"] or not seq_data["scores"]:
        # Need both variants for comparison
        return

    # Aggregate metrics
    metrics = {}
    for label, data in [("Graph", graph_data), ("Sequential", seq_data)]:
        metrics[label] = {
            "avg_score": np.mean(data["scores"]),
            "pass_rate": sum(1 for p in data["passed"] if p) / len(data["passed"]),
            "avg_duration": np.mean(data["durations"]),
            "avg_tokens": np.mean(data["tokens"]),
            "avg_cost": np.mean(data["costs"]),
            "avg_visits": np.mean(data["visits"]),
            "avg_searches": np.mean(data["searches"]),
            "total_runs": len(data["scores"]),
        }

    g = metrics["Graph"]
    s = metrics["Sequential"]

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    fig.suptitle("Graph vs Sequential — Head-to-Head Comparison", fontsize=14, fontweight="bold", y=0.98)

    # --- 1. Score comparison bar with % delta ---
    ax = axes[0, 0]
    labels = ["Graph", "Sequential"]
    scores = [g["avg_score"], s["avg_score"]]
    colors = ["#2196F3", "#FF9800"]
    bars = ax.bar(labels, scores, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5, width=0.5)
    for bar, sc in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, sc + 0.02, f"{sc:.3f}", ha="center", fontsize=14, fontweight="bold")
    delta_score = g["avg_score"] - s["avg_score"]
    pct = (delta_score / s["avg_score"] * 100) if s["avg_score"] > 0 else 0
    sign = "+" if delta_score >= 0 else ""
    ax.set_title(f"Avg Score ({sign}{pct:.1f}% {'graph advantage' if delta_score>=0 else 'seq advantage'})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5, label="Pass threshold")
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # --- 2. Pass rate comparison ---
    ax = axes[0, 1]
    pass_rates = [g["pass_rate"], s["pass_rate"]]
    bars = ax.bar(labels, pass_rates, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5, width=0.5)
    for bar, pr in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, pr + 0.02, f"{pr:.1%}", ha="center", fontsize=14, fontweight="bold")
    delta_pr = g["pass_rate"] - s["pass_rate"]
    sign = "+" if delta_pr >= 0 else ""
    ax.set_title(f"Pass Rate ({sign}{delta_pr*100:.1f}pp delta)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # --- 3. Token usage comparison ---
    ax = axes[0, 2]
    token_vals = [g["avg_tokens"], s["avg_tokens"]]
    bars = ax.bar(labels, token_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5, width=0.5)
    for bar, t in zip(bars, token_vals):
        ax.text(bar.get_x() + bar.get_width()/2, t + max(token_vals)*0.02, _format_tokens(t), ha="center", fontsize=14, fontweight="bold")
    delta_t = g["avg_tokens"] - s["avg_tokens"]
    pct_t = (delta_t / s["avg_tokens"] * 100) if s["avg_tokens"] > 0 else 0
    sign = "+" if delta_t >= 0 else ""
    ax.set_title(f"Avg Tokens ({sign}{pct_t:.1f}%)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- 4. Cost comparison ---
    ax = axes[1, 0]
    cost_vals = [g["avg_cost"], s["avg_cost"]]
    bars = ax.bar(labels, cost_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5, width=0.5)
    for bar, c in zip(bars, cost_vals):
        ax.text(bar.get_x() + bar.get_width()/2, c + max(cost_vals)*0.02, format_cost(c), ha="center", fontsize=14, fontweight="bold")
    delta_c = g["avg_cost"] - s["avg_cost"]
    pct_c = (delta_c / s["avg_cost"] * 100) if s["avg_cost"] > 0 else 0
    sign = "+" if delta_c >= 0 else ""
    ax.set_title(f"Avg Cost ({sign}{pct_c:.1f}%)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- 5. Speed comparison ---
    ax = axes[1, 1]
    dur_vals = [g["avg_duration"], s["avg_duration"]]
    bars = ax.bar(labels, dur_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5, width=0.5)
    for bar, d in zip(bars, dur_vals):
        ax.text(bar.get_x() + bar.get_width()/2, d + max(dur_vals)*0.02, f"{d:.0f}s", ha="center", fontsize=14, fontweight="bold")
    delta_d = g["avg_duration"] - s["avg_duration"]
    pct_d = (delta_d / s["avg_duration"] * 100) if s["avg_duration"] > 0 else 0
    sign = "+" if delta_d >= 0 else ""
    ax.set_title(f"Avg Duration ({sign}{pct_d:.1f}%)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- 6. Efficiency: score per dollar and score per 1000 tokens ---
    ax = axes[1, 2]
    eff_cost_g = g["avg_score"] / g["avg_cost"] if g["avg_cost"] > 0 else 0
    eff_cost_s = s["avg_score"] / s["avg_cost"] if s["avg_cost"] > 0 else 0
    eff_tok_g = g["avg_score"] / (g["avg_tokens"] / 1000) if g["avg_tokens"] > 0 else 0
    eff_tok_s = s["avg_score"] / (s["avg_tokens"] / 1000) if s["avg_tokens"] > 0 else 0

    x = np.arange(2)
    w = 0.3
    bars1 = ax.bar(x - w/2, [eff_cost_g, eff_cost_s], w, label="Score / $", color=["#2196F3", "#FF9800"], alpha=0.85, edgecolor="black")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, [eff_tok_g*1000, eff_tok_s*1000], w, label="Score / 1M tokens", color=["#90CAF9", "#FFE0B2"], alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("Score per Dollar", fontsize=14, color="#1565C0")
    ax2.set_ylabel("Score per 1M Tokens", fontsize=14, color="#E65100")
    ax.set_title("Efficiency Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=14)
    ax2.legend(loc="upper right", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "graph_vs_sequential.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_head_to_head(results: List[Dict[str, Any]], output_dir: Path):
    """
    Head-to-head comparison across all models with percentage deltas,
    radar-style multi-metric view, and cost-performance quadrant.
    """
    from agent.app.model_costs import estimate_cost, estimate_cost_from_total, format_cost

    model_data: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        model = str(r.get("model", "unknown"))
        obs = r.get("execution", {}).get("observability", {})
        llm = obs.get("llm", {})
        score = r.get("validation", {}).get("overall_score", 0.0)
        passed = r.get("validation", {}).get("overall_passed", False)
        duration = r.get("execution", {}).get("duration_seconds", 0)
        tokens = llm.get("total_tokens", 0)
        inp = llm.get("prompt", {}).get("tokens", 0)
        out = llm.get("completion", {}).get("tokens", 0)
        cost = estimate_cost(model, inp, out) if (inp or out) else estimate_cost_from_total(model, tokens)
        visits = obs.get("visit", {}).get("count", 0)
        searches = obs.get("search", {}).get("count", 0)

        model_data[model]["scores"].append(score)
        model_data[model]["passed"].append(passed)
        model_data[model]["durations"].append(duration)
        model_data[model]["tokens"].append(tokens)
        model_data[model]["costs"].append(cost or 0.0)
        model_data[model]["visits"].append(visits)
        model_data[model]["searches"].append(searches)
        model_data[model]["input_tokens"].append(inp)
        model_data[model]["output_tokens"].append(out)

    models = sorted(model_data.keys())
    if len(models) < 2:
        return

    # Compute aggregates
    agg = {}
    for m in models:
        d = model_data[m]
        agg[m] = {
            "avg_score": np.mean(d["scores"]),
            "pass_rate": sum(1 for p in d["passed"] if p) / len(d["passed"]) if d["passed"] else 0,
            "avg_duration": np.mean(d["durations"]),
            "avg_tokens": np.mean(d["tokens"]),
            "avg_cost": np.mean(d["costs"]),
            "avg_visits": np.mean(d["visits"]),
            "avg_searches": np.mean(d["searches"]),
            "avg_input": np.mean(d["input_tokens"]),
            "avg_output": np.mean(d["output_tokens"]),
            "runs": len(d["scores"]),
            "score_per_dollar": np.mean(d["scores"]) / np.mean(d["costs"]) if np.mean(d["costs"]) > 0 else 0,
            "score_per_1k_tokens": np.mean(d["scores"]) / (np.mean(d["tokens"])/1000) if np.mean(d["tokens"]) > 0 else 0,
        }

    n_models = len(models)
    color_list = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"][:n_models]

    fig = plt.figure(figsize=(32, 26))
    fig.suptitle("Model Head-to-Head Comparison", fontsize=14, fontweight="bold", y=0.98)

    # --- 1. Score + Pass Rate grouped bar ---
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(n_models)
    w = 0.35
    score_vals = [agg[m]["avg_score"] for m in models]
    pr_vals = [agg[m]["pass_rate"] for m in models]
    bars_s = ax1.bar(x - w/2, score_vals, w, label="Avg Score", color=color_list, alpha=0.85, edgecolor="black")
    bars_p = ax1.bar(x + w/2, pr_vals, w, label="Pass Rate", color=color_list, alpha=0.45, edgecolor="black", hatch="//")
    for bar, v in zip(bars_s, score_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=14, fontweight="bold")
    for bar, v in zip(bars_p, pr_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.0%}", ha="center", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=14)
    ax1.set_ylim(0, 1.15)
    ax1.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax1.set_title("Score & Pass Rate", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=14)
    ax1.grid(axis="y", alpha=0.3)

    # --- 2. Cost vs Score (Pareto) ---
    ax2 = fig.add_subplot(2, 3, 2)
    for i, m in enumerate(models):
        ax2.scatter(agg[m]["avg_cost"], agg[m]["avg_score"], s=300, color=color_list[i],
                   edgecolors="black", linewidth=2, zorder=5, label=m)
        ax2.annotate(m, (agg[m]["avg_cost"], agg[m]["avg_score"]),
                    textcoords="offset points", xytext=(10, 10), fontsize=14, fontweight="bold")
    # Pareto frontier
    pts = sorted([(agg[m]["avg_cost"], agg[m]["avg_score"]) for m in models])
    pareto = [pts[0]]
    for pt in pts[1:]:
        if pt[1] >= pareto[-1][1]:
            pareto.append(pt)
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax2.plot(px, py, "g--", linewidth=2, alpha=0.7, label="Pareto frontier")
    ax2.axhline(0.75, color="red", linestyle="--", alpha=0.4)
    ax2.set_xlabel("Avg Cost ($)", fontsize=14)
    ax2.set_ylabel("Avg Score", fontsize=14)
    ax2.set_title("Cost-Performance Frontier", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=14)
    ax2.grid(alpha=0.3)

    # --- 3. Efficiency bars (score/$ and score/1M tokens) ---
    ax3 = fig.add_subplot(2, 3, 3)
    spd = [agg[m]["score_per_dollar"] for m in models]
    spt = [agg[m]["score_per_1k_tokens"] * 1000 for m in models]  # score per 1M
    bars_d = ax3.bar(x - w/2, spd, w, label="Score / $", color=color_list, alpha=0.85, edgecolor="black")
    ax3b = ax3.twinx()
    bars_t = ax3b.bar(x + w/2, spt, w, label="Score / 1M tok", color=color_list, alpha=0.4, edgecolor="black", hatch="\\\\")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=14)
    ax3.set_ylabel("Score per Dollar", fontsize=14, color="#1565C0")
    ax3b.set_ylabel("Score per 1M Tokens", fontsize=14, color="#E65100")
    ax3.set_title("Efficiency Rankings", fontsize=14, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=14)
    ax3b.legend(loc="upper right", fontsize=14)
    ax3.grid(axis="y", alpha=0.3)

    # --- 4. Input vs Output token breakdown ---
    ax4 = fig.add_subplot(2, 3, 4)
    inp_vals = [agg[m]["avg_input"] for m in models]
    out_vals = [agg[m]["avg_output"] for m in models]
    bars_in = ax4.bar(x, inp_vals, 0.5, label="Input tokens", color="#64B5F6", edgecolor="black")
    bars_out = ax4.bar(x, out_vals, 0.5, bottom=inp_vals, label="Output tokens", color="#EF5350", edgecolor="black")
    for i, (inp, out) in enumerate(zip(inp_vals, out_vals)):
        total = inp + out
        if total > 0:
            ax4.text(i, total + max(v1+v2 for v1,v2 in zip(inp_vals,out_vals))*0.02,
                    f"{_format_tokens(total)}\n({inp/(total)*100:.0f}% in)", ha="center", fontsize=14, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, fontsize=14)
    ax4.set_ylabel("Tokens", fontsize=14)
    ax4.set_title("Token Breakdown (Input vs Output)", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=14)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: _format_tokens(v)))
    ax4.grid(axis="y", alpha=0.3)

    # --- 5. Percentage deltas from cheapest model ---
    ax5 = fig.add_subplot(2, 3, 5)
    cheapest = min(models, key=lambda m: agg[m]["avg_cost"])
    metric_names = ["Score", "Pass Rate", "Cost", "Tokens", "Duration"]
    metric_keys = ["avg_score", "pass_rate", "avg_cost", "avg_tokens", "avg_duration"]
    y_pos = np.arange(len(metric_names))

    for i, m in enumerate(models):
        if m == cheapest:
            continue
        deltas = []
        for key in metric_keys:
            base = agg[cheapest][key]
            val = agg[m][key]
            if base > 0:
                pct = ((val - base) / base) * 100
            else:
                pct = 0
            deltas.append(pct)
        bar_colors = ["green" if (d > 0 and k in ["avg_score", "pass_rate"]) or
                       (d < 0 and k in ["avg_cost", "avg_tokens", "avg_duration"])
                       else "red" for d, k in zip(deltas, metric_keys)]
        offset = (i - len(models)/2) * 0.15
        bars = ax5.barh(y_pos + offset, deltas, 0.12, label=f"{m} vs {cheapest}",
                       color=color_list[i], alpha=0.8, edgecolor="black")
        for bar, d in zip(bars, deltas):
            sign = "+" if d >= 0 else ""
            ax5.text(bar.get_width() + (1 if d >= 0 else -1), bar.get_y() + bar.get_height()/2,
                    f"{sign}{d:.1f}%", va="center", fontsize=11, fontweight="bold")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(metric_names, fontsize=14)
    ax5.axvline(0, color="black", linewidth=1)
    ax5.set_xlabel("% Difference", fontsize=14)
    ax5.set_title(f"% Change vs Cheapest ({cheapest})", fontsize=14, fontweight="bold")
    ax5.legend(fontsize=14)
    ax5.grid(axis="x", alpha=0.3)

    # --- 6. Actions per model (searches + visits stacked) ---
    ax6 = fig.add_subplot(2, 3, 6)
    search_vals = [agg[m]["avg_searches"] for m in models]
    visit_vals = [agg[m]["avg_visits"] for m in models]
    bars_s = ax6.bar(x, search_vals, 0.5, label="Avg Searches", color="#FFA726", edgecolor="black")
    bars_v = ax6.bar(x, visit_vals, 0.5, bottom=search_vals, label="Avg Visits", color="#26A69A", edgecolor="black")
    for i, (s, v) in enumerate(zip(search_vals, visit_vals)):
        ax6.text(i, s + v + 0.1, f"{s+v:.1f}", ha="center", fontsize=14, fontweight="bold")
    ax6.set_xticks(x)
    ax6.set_xticklabels(models, fontsize=14)
    ax6.set_ylabel("Action Count", fontsize=14)
    ax6.set_title("Avg Actions per Test", fontsize=14, fontweight="bold")
    ax6.legend(fontsize=14)
    ax6.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "model_head_to_head.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_heatmap(results: List[Dict[str, Any]], output_dir: Path):
    """
    Heatmap of avg scores: test (rows) x system (columns).
    Gives an instant view of which model+variant does best on which test.
    """
    test_model_scores: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        tid = r.get("test_metadata", {}).get("test_id", "unknown")
        label = _system_label(r)
        score = r.get("validation", {}).get("overall_score", 0.0)
        test_model_scores[tid][label].append(score)

    tests = sorted(test_model_scores.keys())
    systems = sorted(set(label for scores in test_model_scores.values() for label in scores))

    if not tests or not systems:
        return

    matrix = np.zeros((len(tests), len(systems)))
    for i, t in enumerate(tests):
        for j, s in enumerate(systems):
            sc = test_model_scores[t].get(s, [])
            matrix[i, j] = np.mean(sc) if sc else np.nan

    fig, ax = plt.subplots(figsize=(max(10, len(systems)*2.5), max(8, len(tests)*0.5)))

    # Use green->red colormap (high=green, low=red)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(systems)))
    ax.set_yticks(np.arange(len(tests)))
    ax.set_xticklabels(systems, fontsize=14, rotation=30, ha="right")
    ax.set_yticklabels(tests, fontsize=11)

    # Annotate each cell
    for i in range(len(tests)):
        for j in range(len(systems)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "—"
                color = "gray"
            else:
                text = f"{val:.2f}"
                color = "white" if val < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_title("Score Heatmap: Test x System", fontsize=24, fontweight="bold")
    ax.set_xlabel("System (model [variant])", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test ID", fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Avg Score", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / "score_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_executive_dashboard(results: List[Dict[str, Any]], output_dir: Path):
    """
    Single-page executive dashboard with resume-ready KPIs.
    Big numbers, percentages, and impact metrics at a glance.
    """
    from agent.app.model_costs import estimate_cost, estimate_cost_from_total, format_cost

    total_runs = len(results)
    passed = sum(1 for r in results if r.get("validation", {}).get("overall_passed", False))
    scores = [r.get("validation", {}).get("overall_score", 0.0) for r in results]
    avg_score = np.mean(scores) if scores else 0
    pass_rate = passed / total_runs if total_runs > 0 else 0
    total_tokens = sum(r.get("execution", {}).get("observability", {}).get("llm", {}).get("total_tokens", 0) for r in results)
    total_duration = sum(r.get("execution", {}).get("duration_seconds", 0) for r in results)
    total_visits = sum(r.get("execution", {}).get("observability", {}).get("visit", {}).get("count", 0) for r in results)
    total_searches = sum(r.get("execution", {}).get("observability", {}).get("search", {}).get("count", 0) for r in results)

    total_cost = 0.0
    for r in results:
        model = str(r.get("model", "unknown"))
        llm = r.get("execution", {}).get("observability", {}).get("llm", {})
        inp = llm.get("prompt", {}).get("tokens", 0)
        out = llm.get("completion", {}).get("tokens", 0)
        c = estimate_cost(model, inp, out) if (inp or out) else estimate_cost_from_total(model, llm.get("total_tokens", 0))
        total_cost += (c or 0.0)

    avg_cost = total_cost / total_runs if total_runs > 0 else 0
    avg_tokens = total_tokens / total_runs if total_runs > 0 else 0
    avg_duration = total_duration / total_runs if total_runs > 0 else 0

    # Graph vs sequential split
    graph_scores = [r.get("validation", {}).get("overall_score", 0.0) for r in results if r.get("execution_variant") == "graph"]
    seq_scores = [r.get("validation", {}).get("overall_score", 0.0) for r in results if r.get("execution_variant") == "sequential"]
    graph_avg = np.mean(graph_scores) if graph_scores else 0
    seq_avg = np.mean(seq_scores) if seq_scores else 0

    # Unique models & tests
    unique_models = set(r.get("model", "?") for r in results)
    unique_tests = set(r.get("test_metadata", {}).get("test_id", "?") for r in results)

    # Best and worst test
    test_scores: Dict[str, list] = defaultdict(list)
    for r in results:
        tid = r.get("test_metadata", {}).get("test_id", "?")
        test_scores[tid].append(r.get("validation", {}).get("overall_score", 0.0))
    best_test = max(test_scores.items(), key=lambda kv: np.mean(kv[1]))[0] if test_scores else "—"
    worst_test = min(test_scores.items(), key=lambda kv: np.mean(kv[1]))[0] if test_scores else "—"
    best_test_score = np.mean(test_scores[best_test]) if best_test in test_scores else 0
    worst_test_score = np.mean(test_scores[worst_test]) if worst_test in test_scores else 0

    fig = plt.figure(figsize=(30, 22))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Executive Dashboard — GoT Agent Performance", fontsize=14, fontweight="bold", y=0.97)

    # Helper to draw a KPI card
    def draw_kpi(ax, value_str, label, sublabel="", color="#1565C0", fontsize=36):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=True, facecolor="white",
                                    edgecolor="#E0E0E0", linewidth=2, transform=ax.transAxes, zorder=0))
        ax.text(0.5, 0.65, value_str, ha="center", va="center", fontsize=fontsize,
               fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.30, label, ha="center", va="center", fontsize=14,
               fontweight="bold", color="#424242", transform=ax.transAxes)
        if sublabel:
            ax.text(0.5, 0.12, sublabel, ha="center", va="center", fontsize=14,
                   color="#757575", transform=ax.transAxes)

    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3, left=0.05, right=0.95, top=0.92, bottom=0.02)

    # Row 1: Big KPIs
    ax = fig.add_subplot(gs[0, 0])
    draw_kpi(ax, f"{avg_score:.1%}", "Average Accuracy", f"across {total_runs} test runs",
             color="#2E7D32" if avg_score >= 0.75 else "#C62828")

    ax = fig.add_subplot(gs[0, 1])
    draw_kpi(ax, f"{pass_rate:.0%}", "Pass Rate", f"{passed}/{total_runs} tests passed",
             color="#2E7D32" if pass_rate >= 0.75 else "#C62828")

    ax = fig.add_subplot(gs[0, 2])
    draw_kpi(ax, format_cost(avg_cost), "Avg Cost / Test", f"Total: {format_cost(total_cost)}",
             color="#1565C0")

    ax = fig.add_subplot(gs[0, 3])
    draw_kpi(ax, f"{avg_duration:.0f}s", "Avg Time / Test", f"Total: {total_duration/60:.1f} min",
             color="#6A1B9A")

    # Row 2: Token & Action KPIs
    ax = fig.add_subplot(gs[1, 0])
    draw_kpi(ax, _format_tokens(avg_tokens), "Avg Tokens / Test", f"Total: {_format_tokens(total_tokens)}",
             color="#E65100", fontsize=36)

    ax = fig.add_subplot(gs[1, 1])
    draw_kpi(ax, f"{total_searches}", "Total Searches", f"Avg {total_searches/total_runs:.1f} per test" if total_runs else "",
             color="#00838F", fontsize=36)

    ax = fig.add_subplot(gs[1, 2])
    draw_kpi(ax, f"{total_visits}", "Total Visits", f"Avg {total_visits/total_runs:.1f} per test" if total_runs else "",
             color="#00838F", fontsize=36)

    ax = fig.add_subplot(gs[1, 3])
    score_dollar = avg_score / avg_cost if avg_cost > 0 else 0
    draw_kpi(ax, f"{score_dollar:.1f}", "Score per Dollar", "Higher = more efficient",
             color="#2E7D32" if score_dollar > 10 else "#E65100", fontsize=36)

    # Row 3: Graph vs Sequential & model counts
    ax = fig.add_subplot(gs[2, 0:2])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=True, facecolor="white",
                                edgecolor="#E0E0E0", linewidth=2, transform=ax.transAxes))
    ax.text(0.5, 0.85, "Graph vs Sequential", ha="center", va="center",
           fontsize=24, fontweight="bold", color="#212121", transform=ax.transAxes)
    if graph_scores and seq_scores:
        delta = graph_avg - seq_avg
        winner = "Graph" if delta > 0 else "Sequential" if delta < 0 else "Tied"
        delta_pct = abs(delta / seq_avg * 100) if seq_avg > 0 else 0
        ax.text(0.25, 0.55, f"Graph\n{graph_avg:.3f}", ha="center", va="center",
               fontsize=14, fontweight="bold", color="#2196F3", transform=ax.transAxes)
        ax.text(0.5, 0.55, "vs", ha="center", va="center",
               fontsize=14, color="#9E9E9E", transform=ax.transAxes)
        ax.text(0.75, 0.55, f"Sequential\n{seq_avg:.3f}", ha="center", va="center",
               fontsize=14, fontweight="bold", color="#FF9800", transform=ax.transAxes)
        winner_color = "#2196F3" if winner == "Graph" else "#FF9800" if winner == "Sequential" else "#9E9E9E"
        ax.text(0.5, 0.18, f"Winner: {winner} (+{delta_pct:.1f}%)",
               ha="center", va="center", fontsize=14, fontweight="bold",
               color=winner_color, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Single variant only", ha="center", va="center",
               fontsize=14, color="#9E9E9E", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=True, facecolor="white",
                                edgecolor="#E0E0E0", linewidth=2, transform=ax.transAxes))
    ax.text(0.5, 0.85, "Test Coverage", ha="center", va="center",
           fontsize=24, fontweight="bold", color="#212121", transform=ax.transAxes)
    ax.text(0.5, 0.55, f"{len(unique_tests)} tests × {len(unique_models)} models",
           ha="center", va="center", fontsize=14, fontweight="bold", color="#1565C0",
           transform=ax.transAxes)
    ax.text(0.5, 0.25,
           f"Hardest: {worst_test} ({worst_test_score:.2f})  |  Easiest: {best_test} ({best_test_score:.2f})",
           ha="center", va="center", fontsize=14, color="#616161", transform=ax.transAxes)

    # Row 4: Score distribution mini-histogram + cumulative pass rate
    ax = fig.add_subplot(gs[3, 0:2])
    ax.hist(scores, bins=20, color="#42A5F5", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(scores), color="#C62828", linestyle="--", linewidth=2, label=f"Mean: {np.mean(scores):.3f}")
    ax.axvline(0.75, color="#FF6F00", linestyle="--", linewidth=2, label="Pass: 0.75")
    ax.set_xlabel("Score", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)

    # Cumulative pass rate across score thresholds
    ax = fig.add_subplot(gs[3, 2:4])
    thresholds = np.linspace(0, 1, 50)
    cum_pass = [sum(1 for s in scores if s >= t) / len(scores) * 100 for t in thresholds]
    ax.fill_between(thresholds, cum_pass, alpha=0.3, color="#2196F3")
    ax.plot(thresholds, cum_pass, color="#1565C0", linewidth=2.5)
    ax.axvline(0.75, color="red", linestyle="--", linewidth=2, label="Pass threshold")
    pass_at_75 = sum(1 for s in scores if s >= 0.75) / len(scores) * 100
    ax.axhline(pass_at_75, color="#4CAF50", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(0.77, pass_at_75 + 2, f"{pass_at_75:.1f}%", fontsize=14, fontweight="bold", color="#4CAF50")
    ax.set_xlabel("Score Threshold", fontsize=14)
    ax.set_ylabel("% Tests Passing", fontsize=14)
    ax.set_title("Cumulative Pass Rate by Threshold", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)

    plt.savefig(output_dir / "executive_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_token_breakdown(results: List[Dict[str, Any]], output_dir: Path):
    """
    Detailed token breakdown: input vs output per system and per test,
    with I/O ratio analysis.
    """
    systems = sorted(set(_system_label(r) for r in results))
    system_colors = _get_system_colors(systems, "tab10")

    sys_input: Dict[str, list] = defaultdict(list)
    sys_output: Dict[str, list] = defaultdict(list)
    test_input: Dict[str, list] = defaultdict(list)
    test_output: Dict[str, list] = defaultdict(list)

    for r in results:
        label = _system_label(r)
        tid = r.get("test_metadata", {}).get("test_id", "?")
        llm = r.get("execution", {}).get("observability", {}).get("llm", {})
        inp = llm.get("prompt", {}).get("tokens", 0)
        out = llm.get("completion", {}).get("tokens", 0)
        sys_input[label].append(inp)
        sys_output[label].append(out)
        test_input[tid].append(inp)
        test_output[tid].append(out)

    fig, axes = plt.subplots(2, 2, figsize=(28, 20))
    fig.suptitle("Token Breakdown Analysis", fontsize=14, fontweight="bold")

    # --- 1. Input vs Output by system (stacked) ---
    ax = axes[0, 0]
    x = np.arange(len(systems))
    avg_in = [np.mean(sys_input[s]) for s in systems]
    avg_out = [np.mean(sys_output[s]) for s in systems]
    ax.bar(x, avg_in, 0.5, label="Input (prompt)", color="#64B5F6", edgecolor="black")
    ax.bar(x, avg_out, 0.5, bottom=avg_in, label="Output (completion)", color="#EF5350", edgecolor="black")
    for i, (inp, out) in enumerate(zip(avg_in, avg_out)):
        total = inp + out
        ratio = inp / total * 100 if total > 0 else 0
        ax.text(i, total + max(a+b for a,b in zip(avg_in, avg_out))*0.02,
               f"{_format_tokens(total)}\n{ratio:.0f}% in", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("Tokens", fontsize=14)
    ax.set_title("Avg Token Split by System", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: _format_tokens(v)))
    ax.grid(axis="y", alpha=0.3)

    # --- 2. I/O ratio by system ---
    ax = axes[0, 1]
    ratios = []
    for s in systems:
        total_in = sum(sys_input[s])
        total_out = sum(sys_output[s])
        ratios.append(total_in / total_out if total_out > 0 else 0)
    bars = ax.bar(x, ratios, 0.5, color=[system_colors[s] for s in systems], alpha=0.85, edgecolor="black")
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, r + max(ratios)*0.02, f"{r:.1f}x", ha="center", fontsize=14, fontweight="bold")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1:1 ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("Input/Output Ratio", fontsize=14)
    ax.set_title("Input:Output Token Ratio", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # --- 3. Token split by test (top 15 by total) ---
    ax = axes[1, 0]
    test_ids = sorted(test_input.keys())
    test_avg_in = {t: np.mean(test_input[t]) for t in test_ids}
    test_avg_out = {t: np.mean(test_output[t]) for t in test_ids}
    test_total = {t: test_avg_in[t] + test_avg_out[t] for t in test_ids}
    top_tests = sorted(test_ids, key=lambda t: test_total[t], reverse=True)[:15]

    y = np.arange(len(top_tests))
    in_vals = [test_avg_in[t] for t in top_tests]
    out_vals = [test_avg_out[t] for t in top_tests]
    ax.barh(y, in_vals, 0.6, label="Input", color="#64B5F6", edgecolor="black")
    ax.barh(y, out_vals, 0.6, left=in_vals, label="Output", color="#EF5350", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(top_tests, fontsize=14)
    ax.set_xlabel("Tokens", fontsize=14)
    ax.set_title("Token Split by Test (Top 15)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: _format_tokens(v)))
    ax.grid(axis="x", alpha=0.3)

    # --- 4. Output tokens vs Score scatter ---
    ax = axes[1, 1]
    for s in systems:
        s_results = [r for r in results if _system_label(r) == s]
        out_tokens = [r.get("execution", {}).get("observability", {}).get("llm", {}).get("completion", {}).get("tokens", 0) for r in s_results]
        s_scores = [r.get("validation", {}).get("overall_score", 0.0) for r in s_results]
        ax.scatter(out_tokens, s_scores, alpha=0.7, s=100, color=system_colors[s],
                  label=s, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Output Tokens", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Output Tokens", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: _format_tokens(v)))
    ax.legend(fontsize=14, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "token_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_action_effectiveness(results: List[Dict[str, Any]], output_dir: Path):
    """
    Action pipeline effectiveness: search-to-visit ratio, visits per score point,
    action ROI, and data throughput analysis.
    """
    systems = sorted(set(_system_label(r) for r in results))
    system_colors = _get_system_colors(systems, "tab10")

    sys_data: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        label = _system_label(r)
        obs = r.get("execution", {}).get("observability", {})
        score = r.get("validation", {}).get("overall_score", 0.0)
        searches = obs.get("search", {}).get("count", 0)
        visits = obs.get("visit", {}).get("count", 0)
        search_kb = obs.get("search", {}).get("kilobytes", 0.0)
        visit_kb = obs.get("visit", {}).get("kilobytes", 0.0)
        duration = r.get("execution", {}).get("duration_seconds", 0)

        sys_data[label]["scores"].append(score)
        sys_data[label]["searches"].append(searches)
        sys_data[label]["visits"].append(visits)
        sys_data[label]["search_kb"].append(search_kb)
        sys_data[label]["visit_kb"].append(visit_kb)
        sys_data[label]["duration"].append(duration)

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    fig.suptitle("Action Pipeline Effectiveness", fontsize=14, fontweight="bold")

    x = np.arange(len(systems))

    # --- 1. Search-to-Visit ratio ---
    ax = axes[0, 0]
    ratios = []
    for s in systems:
        total_s = sum(sys_data[s]["searches"])
        total_v = sum(sys_data[s]["visits"])
        ratios.append(total_v / total_s if total_s > 0 else 0)
    bars = ax.bar(x, ratios, 0.5, color=[system_colors[s] for s in systems], alpha=0.85, edgecolor="black")
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, r + max(ratios)*0.02 if ratios else 0,
               f"{r:.2f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("Visits / Searches", fontsize=14)
    ax.set_title("Visit-to-Search Ratio", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- 2. Score per visit (ROI) ---
    ax = axes[0, 1]
    spv = []
    for s in systems:
        avg_s = np.mean(sys_data[s]["scores"]) if sys_data[s]["scores"] else 0
        avg_v = np.mean(sys_data[s]["visits"]) if sys_data[s]["visits"] else 0
        spv.append(avg_s / avg_v if avg_v > 0 else 0)
    bars = ax.bar(x, spv, 0.5, color=[system_colors[s] for s in systems], alpha=0.85, edgecolor="black")
    for bar, v in zip(bars, spv):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(spv)*0.02 if spv else 0,
               f"{v:.2f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("Score / Avg Visits", fontsize=14)
    ax.set_title("Score per Visit (Visit ROI)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- 3. Data throughput (KB per second) ---
    ax = axes[0, 2]
    throughputs = []
    for s in systems:
        total_kb = sum(sys_data[s]["search_kb"]) + sum(sys_data[s]["visit_kb"])
        total_sec = sum(sys_data[s]["duration"])
        throughputs.append(total_kb / total_sec if total_sec > 0 else 0)
    bars = ax.bar(x, throughputs, 0.5, color=[system_colors[s] for s in systems], alpha=0.85, edgecolor="black")
    for bar, v in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(throughputs)*0.02 if throughputs else 0,
               f"{v:.1f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("KB / second", fontsize=14)
    ax.set_title("Data Throughput", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- 4. Visits vs Score scatter ---
    ax = axes[1, 0]
    for s in systems:
        ax.scatter(sys_data[s]["visits"], sys_data[s]["scores"], alpha=0.7, s=100,
                  color=system_colors[s], label=s, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Visit Count", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Visit Count", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=14, loc="best")
    ax.grid(alpha=0.3)

    # --- 5. Searches vs Score scatter ---
    ax = axes[1, 1]
    for s in systems:
        ax.scatter(sys_data[s]["searches"], sys_data[s]["scores"], alpha=0.7, s=100,
                  color=system_colors[s], label=s, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Search Count", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Search Count", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=14, loc="best")
    ax.grid(alpha=0.3)

    # --- 6. Data volume stacked (search KB + visit KB) ---
    ax = axes[1, 2]
    avg_search_kb = [np.mean(sys_data[s]["search_kb"]) for s in systems]
    avg_visit_kb = [np.mean(sys_data[s]["visit_kb"]) for s in systems]
    ax.bar(x, avg_search_kb, 0.5, label="Search Data (KB)", color="#FFA726", edgecolor="black")
    ax.bar(x, avg_visit_kb, 0.5, bottom=avg_search_kb, label="Visit Data (KB)", color="#26A69A", edgecolor="black")
    for i, (sk, vk) in enumerate(zip(avg_search_kb, avg_visit_kb)):
        ax.text(i, sk + vk + 0.5, f"{sk+vk:.1f} KB", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("Avg Data (KB)", fontsize=14)
    ax.set_title("Avg Data Volume per Test", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "action_effectiveness.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_test_model_comparison(results: List[Dict[str, Any]], output_dir: Path):
    """
    Per-test breakdown showing how each model performs on each test.
    Grouped bar chart + a delta chart showing which model gains the most.
    """
    test_model_scores: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    test_model_tokens: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        tid = r.get("test_metadata", {}).get("test_id", "?")
        label = _system_label(r)
        score = r.get("validation", {}).get("overall_score", 0.0)
        tokens = r.get("execution", {}).get("observability", {}).get("llm", {}).get("total_tokens", 0)
        test_model_scores[tid][label].append(score)
        test_model_tokens[tid][label].append(tokens)

    tests = sorted(test_model_scores.keys())
    systems = sorted(set(label for scores in test_model_scores.values() for label in scores))

    if len(tests) < 1 or len(systems) < 1:
        return

    system_colors = _get_system_colors(systems, "tab10")

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(tests)*1.5), 14))
    fig.suptitle("Per-Test Model Comparison", fontsize=14, fontweight="bold")

    # --- 1. Grouped bars: score per test per system ---
    ax = axes[0]
    n_sys = len(systems)
    width = 0.8 / n_sys
    x = np.arange(len(tests))

    for i, s in enumerate(systems):
        scores = [np.mean(test_model_scores[t].get(s, [0])) for t in tests]
        offset = (i - n_sys/2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=s, color=system_colors[s],
                     alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=14)
    ax.set_ylabel("Avg Score", fontsize=14)
    ax.set_title("Score by Test × System", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5, label="Pass threshold")
    ax.legend(fontsize=14, ncol=min(n_sys, 4), loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # --- 2. Token efficiency per test: score / 1k tokens ---
    ax = axes[1]
    for i, s in enumerate(systems):
        efficiencies = []
        for t in tests:
            avg_s = np.mean(test_model_scores[t].get(s, [0]))
            avg_t = np.mean(test_model_tokens[t].get(s, [1]))
            efficiencies.append(avg_s / (avg_t / 1000) if avg_t > 0 else 0)
        offset = (i - n_sys/2 + 0.5) * width
        ax.bar(x + offset, efficiencies, width, label=s, color=system_colors[s],
              alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=14)
    ax.set_ylabel("Score / 1K Tokens", fontsize=14)
    ax.set_title("Token Efficiency by Test × System", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14, ncol=min(n_sys, 4), loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "per_test_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_graph_structure(results: List[Dict[str, Any]], output_dir: Path):
    """
    Visualize graph structure metrics: depth, branching, node count,
    and leaf-to-internal ratio per system and per test.
    """
    systems = sorted(set(_system_label(r) for r in results))
    system_colors = _get_system_colors(systems, "tab10")

    sys_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    test_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        label = _system_label(r)
        tid = r.get("test_metadata", {}).get("test_id", "?")
        gm = _extract_graph_metrics(r)
        score = r.get("validation", {}).get("overall_score", 0.0)

        for key in ["total_nodes", "max_depth", "avg_depth", "avg_branching",
                     "max_branching", "leaf_count", "internal_count"]:
            sys_metrics[label][key].append(gm[key])
            test_metrics[tid][key].append(gm[key])
        sys_metrics[label]["scores"].append(score)
        test_metrics[tid]["scores"].append(score)

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    fig.suptitle("Graph Structure Analysis", fontsize=14, fontweight="bold")

    x = np.arange(len(systems))

    # --- 1. Avg depth & max depth per system ---
    ax = axes[0, 0]
    avg_d = [np.mean(sys_metrics[s]["avg_depth"]) for s in systems]
    max_d = [np.mean(sys_metrics[s]["max_depth"]) for s in systems]
    w = 0.3
    ax.bar(x - w/2, avg_d, w, label="Avg Depth", color="#42A5F5", edgecolor="black")
    ax.bar(x + w/2, max_d, w, label="Max Depth", color="#1565C0", edgecolor="black")
    for i, (a, m) in enumerate(zip(avg_d, max_d)):
        ax.text(i - w/2, a + 0.05, f"{a:.1f}", ha="center", fontsize=14, fontweight="bold")
        ax.text(i + w/2, m + 0.05, f"{m:.1f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("Depth", fontsize=14)
    ax.set_title("Graph Depth by System", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # --- 2. Avg branching per system ---
    ax = axes[0, 1]
    avg_b = [np.mean(sys_metrics[s]["avg_branching"]) for s in systems]
    max_b = [np.mean(sys_metrics[s]["max_branching"]) for s in systems]
    ax.bar(x - w/2, avg_b, w, label="Avg Branching", color="#66BB6A", edgecolor="black")
    ax.bar(x + w/2, max_b, w, label="Max Branching", color="#2E7D32", edgecolor="black")
    for i, (a, m) in enumerate(zip(avg_b, max_b)):
        ax.text(i - w/2, a + 0.05, f"{a:.1f}", ha="center", fontsize=14, fontweight="bold")
        ax.text(i + w/2, m + 0.05, f"{m:.1f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("Branching Factor", fontsize=14)
    ax.set_title("Branching Factor by System", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # --- 3. Total nodes per system ---
    ax = axes[0, 2]
    avg_n = [np.mean(sys_metrics[s]["total_nodes"]) for s in systems]
    leaf_n = [np.mean(sys_metrics[s]["leaf_count"]) for s in systems]
    int_n = [np.mean(sys_metrics[s]["internal_count"]) for s in systems]
    ax.bar(x, int_n, 0.5, label="Internal", color="#FFB74D", edgecolor="black")
    ax.bar(x, leaf_n, 0.5, bottom=int_n, label="Leaf", color="#4DB6AC", edgecolor="black")
    for i, (n, l) in enumerate(zip(avg_n, leaf_n)):
        ax.text(i, n + 0.2, f"{n:.1f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=14, rotation=20, ha="right")
    ax.set_ylabel("Node Count", fontsize=14)
    ax.set_title("Avg Nodes (Leaf vs Internal)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # --- 4. Depth vs Score scatter ---
    ax = axes[1, 0]
    for s in systems:
        ax.scatter(sys_metrics[s]["max_depth"], sys_metrics[s]["scores"],
                  alpha=0.7, s=100, color=system_colors[s], label=s,
                  edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Max Depth", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Graph Depth", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=14, loc="best")
    ax.grid(alpha=0.3)

    # --- 5. Branching vs Score scatter ---
    ax = axes[1, 1]
    for s in systems:
        ax.scatter(sys_metrics[s]["avg_branching"], sys_metrics[s]["scores"],
                  alpha=0.7, s=100, color=system_colors[s], label=s,
                  edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Avg Branching Factor", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Branching Factor", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=14, loc="best")
    ax.grid(alpha=0.3)

    # --- 6. Node count vs Score scatter ---
    ax = axes[1, 2]
    for s in systems:
        ax.scatter(sys_metrics[s]["total_nodes"], sys_metrics[s]["scores"],
                  alpha=0.7, s=100, color=system_colors[s], label=s,
                  edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Total Nodes", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Node Count", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=14, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "graph_structure.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_data_coverage(results: List[Dict[str, Any]], output_dir: Path):
    """
    Data coverage analysis: how much data was gathered vs score.
    Shows visit/search volume, response length, and data-to-score correlations.
    """
    from agent.app.model_costs import estimate_cost, estimate_cost_from_total, format_cost

    systems = sorted(set(_system_label(r) for r in results))
    system_colors = _get_system_colors(systems, "tab10")

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Data Coverage & Response Quality Analysis", fontsize=14, fontweight="bold")

    scores = []
    visits = []
    searches = []
    visit_kb = []
    search_kb = []
    deliverable_lens = []
    total_tokens = []
    labels = []

    for r in results:
        score = r.get("validation", {}).get("overall_score", 0.0)
        obs = r.get("execution", {}).get("observability", {})
        label = _system_label(r)

        v = obs.get("visit", {}).get("count", 0)
        s = obs.get("search", {}).get("count", 0)
        v_kb = obs.get("visit", {}).get("kilobytes", 0.0)
        s_kb = obs.get("search", {}).get("kilobytes", 0.0)
        tok = obs.get("llm", {}).get("total_tokens", 0)

        deliv = r.get("final_output", {}).get("final_deliverable", "")
        if not deliv:
            deliv = r.get("output", {}).get("final_deliverable", "")
        deliv_len = len(str(deliv)) if deliv else 0

        scores.append(score)
        visits.append(v)
        searches.append(s)
        visit_kb.append(v_kb)
        search_kb.append(s_kb)
        deliverable_lens.append(deliv_len)
        total_tokens.append(tok)
        labels.append(label)

    scores = np.array(scores)
    visits = np.array(visits)
    searches = np.array(searches)
    visit_kb = np.array(visit_kb)
    search_kb = np.array(search_kb)
    deliverable_lens = np.array(deliverable_lens)
    total_tokens = np.array(total_tokens)

    # 1. Visits vs Score
    ax = axes[0, 0]
    for sys in systems:
        mask = [l == sys for l in labels]
        if any(mask):
            ax.scatter(np.array(visits)[mask], np.array(scores)[mask],
                      alpha=0.7, s=80, color=system_colors[sys], label=sys,
                      edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Visit Count", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Visit Count", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5, label="Pass threshold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)

    # 2. Total data gathered (visit+search KB) vs Score
    ax = axes[0, 1]
    total_data = visit_kb + search_kb
    for sys in systems:
        mask = [l == sys for l in labels]
        if any(mask):
            ax.scatter(np.array(total_data)[mask], np.array(scores)[mask],
                      alpha=0.7, s=80, color=system_colors[sys], label=sys,
                      edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Total Data Gathered (KB)", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Data Volume", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)

    # 3. Deliverable length vs Score
    ax = axes[0, 2]
    for sys in systems:
        mask = [l == sys for l in labels]
        if any(mask):
            ax.scatter(np.array(deliverable_lens)[mask], np.array(scores)[mask],
                      alpha=0.7, s=80, color=system_colors[sys], label=sys,
                      edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Final Response Length (chars)", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Response Length", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)

    # 4. Searches vs Visits (bubble = score)
    ax = axes[1, 0]
    for sys in systems:
        mask = [l == sys for l in labels]
        if any(mask):
            s_vals = np.array(scores)[mask]
            sizes = s_vals * 150 + 20
            ax.scatter(np.array(searches)[mask], np.array(visits)[mask],
                      s=sizes, alpha=0.6, color=system_colors[sys], label=sys,
                      edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Search Count", fontsize=14)
    ax.set_ylabel("Visit Count", fontsize=14)
    ax.set_title("Search vs Visit (bubble size = score)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)

    # 5. Token usage vs Score with trend line
    ax = axes[1, 1]
    for sys in systems:
        mask = np.array([l == sys for l in labels])
        if any(mask):
            t = np.array(total_tokens)[mask]
            s = np.array(scores)[mask]
            ax.scatter(t, s, alpha=0.7, s=80, color=system_colors[sys], label=sys,
                      edgecolors="black", linewidth=0.5)
    if len(total_tokens) > 2 and np.std(total_tokens) > 0:
        z = np.polyfit(total_tokens, scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(total_tokens.min(), total_tokens.max(), 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.6, linewidth=2, label="Trend")
    ax.set_xlabel("Total Tokens", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Token Usage (with trend)", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)

    # 6. Per-system average data coverage bar chart
    ax = axes[1, 2]
    sys_avg_visits = []
    sys_avg_searches = []
    sys_avg_data_kb = []
    sys_avg_deliv = []
    for sys in systems:
        mask = [l == sys for l in labels]
        if any(mask):
            sys_avg_visits.append(np.mean(visits[mask]))
            sys_avg_searches.append(np.mean(searches[mask]))
            sys_avg_data_kb.append(np.mean(total_data[mask]))
            sys_avg_deliv.append(np.mean(deliverable_lens[mask]) / 100)
        else:
            sys_avg_visits.append(0)
            sys_avg_searches.append(0)
            sys_avg_data_kb.append(0)
            sys_avg_deliv.append(0)

    x = np.arange(len(systems))
    w = 0.2
    ax.bar(x - 1.5*w, sys_avg_visits, w, label="Avg Visits", color="#2196F3", edgecolor="black")
    ax.bar(x - 0.5*w, sys_avg_searches, w, label="Avg Searches", color="#FF9800", edgecolor="black")
    ax.bar(x + 0.5*w, [d/10 for d in sys_avg_data_kb], w, label="Data KB/10", color="#4CAF50", edgecolor="black")
    ax.bar(x + 1.5*w, sys_avg_deliv, w, label="Response len/100", color="#9C27B0", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11, rotation=20, ha="right")
    ax.set_ylabel("Metric Value", fontsize=14)
    ax.set_title("Avg Data Coverage per System", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "data_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost_efficiency_frontier(results: List[Dict[str, Any]], output_dir: Path):
    """
    Cost-efficiency frontier: Pareto front of score vs cost per model/variant.
    Shows which configurations give the best score-to-cost ratio.
    """
    from agent.app.model_costs import estimate_cost, estimate_cost_from_total, format_cost

    systems = sorted(set(_system_label(r) for r in results))
    system_colors = _get_system_colors(systems, "tab10")

    sys_scores: Dict[str, List[float]] = defaultdict(list)
    sys_costs: Dict[str, List[float]] = defaultdict(list)

    for r in results:
        label = _system_label(r)
        score = r.get("validation", {}).get("overall_score", 0.0)
        model = str(r.get("model", "unknown"))
        obs = r.get("execution", {}).get("observability", {})
        llm = obs.get("llm", {})
        inp = llm.get("prompt", {}).get("tokens", 0)
        out = llm.get("completion", {}).get("tokens", 0)
        if inp > 0 or out > 0:
            cost = estimate_cost(model, inp, out)
        else:
            cost = estimate_cost_from_total(model, llm.get("total_tokens", 0))
        if cost is not None:
            sys_scores[label].append(score)
            sys_costs[label].append(cost)

    fig, axes = plt.subplots(1, 2, figsize=(28, 13))
    fig.suptitle("Cost-Efficiency Analysis", fontsize=14, fontweight="bold")

    # 1. Scatter: individual runs score vs cost
    ax = axes[0]
    for sys in systems:
        if sys in sys_scores and sys_scores[sys]:
            ax.scatter(sys_costs[sys], sys_scores[sys],
                      alpha=0.6, s=80, color=system_colors[sys], label=sys,
                      edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Cost per Run ($)", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Score vs Cost (individual runs)", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5, label="Pass threshold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)

    # 2. Average score vs average cost per system (efficiency frontier)
    ax = axes[1]
    avg_costs = []
    avg_scores = []
    eff_labels = []
    for sys in systems:
        if sys in sys_scores and sys_scores[sys]:
            ac = np.mean(sys_costs[sys])
            asc = np.mean(sys_scores[sys])
            avg_costs.append(ac)
            avg_scores.append(asc)
            eff_labels.append(sys)
            ax.scatter(ac, asc, s=200, color=system_colors[sys],
                      edgecolors="black", linewidth=1.5, zorder=5)
            ax.annotate(sys, (ac, asc), fontsize=14, fontweight="bold",
                       xytext=(8, 8), textcoords="offset points",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Draw Pareto frontier
    if len(avg_costs) > 1:
        points = sorted(zip(avg_costs, avg_scores), key=lambda p: p[0])
        frontier_x = [points[0][0]]
        frontier_y = [points[0][1]]
        best_score = points[0][1]
        for cx, sy in points[1:]:
            if sy >= best_score:
                frontier_x.append(cx)
                frontier_y.append(sy)
                best_score = sy
        if len(frontier_x) > 1:
            ax.plot(frontier_x, frontier_y, "--", color="#4CAF50", linewidth=2.5,
                   alpha=0.7, label="Efficiency Frontier", zorder=3)
            ax.fill_between(frontier_x, frontier_y, alpha=0.05, color="#4CAF50")

    ax.set_xlabel("Average Cost per Run ($)", fontsize=14)
    ax.set_ylabel("Average Score", fontsize=14)
    ax.set_title("Cost-Efficiency Frontier", fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=14, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "cost_efficiency_frontier.png", dpi=150, bbox_inches="tight")
    plt.close()
