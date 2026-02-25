"""
Plot functions for visualization.
"""

from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from .visualization_helpers import _system_label, _get_system_colors, _get_difficulty_colormap, _format_tokens

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
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
    
    ax.set_xlabel("Success Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("Validation Check", fontsize=13, fontweight="bold")
    ax.set_title("Validation Check Success Rates", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=11)
    
    for i, (name, rate) in enumerate(zip(check_names, success_rates)):
        ax.text(rate + 0.02, i, f"{rate:.1%}", va="center", fontsize=10, fontweight="bold")
    
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
    
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle("Performance Distributions by System", fontsize=16, fontweight="bold")
    
    x_positions = np.arange(len(models))
    width = 0.6
    
    for i, model in enumerate(models):
        scores = model_scores[model]
        axes1[0].scatter([i] * len(scores), scores, alpha=0.6, s=80, color=model_colors[model], edgecolors="black", linewidth=0.5)
        if scores:
            axes1[0].axhline(np.mean(scores), color=model_colors[model], linestyle="--", alpha=0.7, linewidth=2)
    axes1[0].set_ylabel("Score", fontsize=13)
    axes1[0].set_title("Score Distribution", fontsize=14, fontweight="bold")
    axes1[0].set_xticks(x_positions)
    axes1[0].set_xticklabels(models, fontsize=11)
    axes1[0].set_ylim(0, 1.0)
    axes1[0].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Pass Threshold")
    axes1[0].legend(fontsize=10)
    axes1[0].grid(alpha=0.3)
    
    for i, model in enumerate(models):
        tokens = model_tokens[model]
        axes1[1].scatter([i] * len(tokens), tokens, alpha=0.6, s=80, color=model_colors[model], edgecolors="black", linewidth=0.5)
    axes1[1].set_ylabel("Total Tokens", fontsize=13)
    axes1[1].set_title("Token Usage", fontsize=14, fontweight="bold")
    axes1[1].set_xticks(x_positions)
    axes1[1].set_xticklabels(models, fontsize=11)
    axes1[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_tokens(x)))
    axes1[1].grid(alpha=0.3)
    
    for i, model in enumerate(models):
        durations = model_durations[model]
        axes1[2].scatter([i] * len(durations), durations, alpha=0.6, s=80, color=model_colors[model], edgecolors="black", linewidth=0.5)
    axes1[2].set_ylabel("Duration (seconds)", fontsize=13)
    axes1[2].set_title("Execution Duration", fontsize=14, fontweight="bold")
    axes1[2].set_xticks(x_positions)
    axes1[2].set_xticklabels(models, fontsize=11)
    axes1[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle("Performance Correlations", fontsize=16, fontweight="bold")
    
    for model in models:
        scores = model_scores[model]
        tokens = model_tokens[model]
        axes2[0].scatter(tokens, scores, alpha=0.7, s=120, color=model_colors[model], label=model, edgecolors="black", linewidth=0.5)
    axes2[0].set_xlabel("Total Tokens", fontsize=13)
    axes2[0].set_ylabel("Score", fontsize=13)
    axes2[0].set_title("Score vs Token Usage", fontsize=14, fontweight="bold")
    axes2[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_tokens(x)))
    axes2[0].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes2[0].legend(fontsize=10, loc="best")
    axes2[0].grid(alpha=0.3)
    
    for model in models:
        durations = model_durations[model]
        scores = model_scores[model]
        axes2[1].scatter(durations, scores, alpha=0.7, s=120, color=model_colors[model], label=model, edgecolors="black", linewidth=0.5)
    axes2[1].set_xlabel("Duration (seconds)", fontsize=13)
    axes2[1].set_ylabel("Score", fontsize=13)
    axes2[1].set_title("Score vs Duration", fontsize=14, fontweight="bold")
    axes2[1].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes2[1].legend(fontsize=10, loc="best")
    axes2[1].grid(alpha=0.3)
    
    for model in models:
        searches = model_searches[model]
        visits = model_visits[model]
        axes2[2].scatter(searches, visits, alpha=0.7, s=120, color=model_colors[model], label=model, edgecolors="black", linewidth=0.5)
    axes2[2].set_xlabel("Search Count", fontsize=13)
    axes2[2].set_ylabel("Visit Count", fontsize=13)
    axes2[2].set_title("Searches vs Visits", fontsize=14, fontweight="bold")
    axes2[2].legend(fontsize=10, loc="best")
    axes2[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle("System Summary Metrics", fontsize=16, fontweight="bold")
    
    model_avg_scores = [np.mean(model_scores[m]) if model_scores[m] else 0.0 for m in models]
    model_avg_tokens = [np.mean(model_tokens[m]) if model_tokens[m] else 0.0 for m in models]
    scatter1 = axes3[0].scatter(model_avg_tokens, model_avg_scores, s=250, c=range(len(models)), cmap="viridis", alpha=0.8, edgecolors="black", linewidth=2)
    for i, model in enumerate(models):
        axes3[0].annotate(model, (model_avg_tokens[i], model_avg_scores[i]), fontsize=10, fontweight="bold", ha="center", va="bottom")
    axes3[0].set_xlabel("Average Tokens", fontsize=13)
    axes3[0].set_ylabel("Average Score", fontsize=13)
    axes3[0].set_title("Model Efficiency", fontsize=14, fontweight="bold")
    axes3[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_tokens(x)))
    axes3[0].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes3[0].grid(alpha=0.3)
    
    model_avg_durations = [np.mean(model_durations[m]) if model_durations[m] else 0.0 for m in models]
    scatter2 = axes3[1].scatter(model_avg_durations, model_avg_scores, s=250, c=range(len(models)), cmap="plasma", alpha=0.8, edgecolors="black", linewidth=2)
    for i, model in enumerate(models):
        axes3[1].annotate(model, (model_avg_durations[i], model_avg_scores[i]), fontsize=10, fontweight="bold", ha="center", va="bottom")
    axes3[1].set_xlabel("Average Duration (seconds)", fontsize=13)
    axes3[1].set_ylabel("Average Score", fontsize=13)
    axes3[1].set_title("Speed vs Performance", fontsize=14, fontweight="bold")
    axes3[1].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes3[1].grid(alpha=0.3)
    
    model_pass_rates = []
    for model in models:
        passed = sum(1 for s in model_scores[model] if s >= 0.75)
        total = len(model_scores[model])
        model_pass_rates.append(passed / total if total > 0 else 0.0)
    
    bars = axes3[2].bar(x_positions, model_pass_rates, width, color=[model_colors[m] for m in models], alpha=0.8, edgecolor="black", linewidth=2)
    axes3[2].set_ylabel("Pass Rate", fontsize=13)
    axes3[2].set_title("Pass Rate by System", fontsize=14, fontweight="bold")
    axes3[2].set_xticks(x_positions)
    axes3[2].set_xticklabels(models, fontsize=11)
    axes3[2].set_ylim(0, 1.0)
    axes3[2].axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Target")
    axes3[2].legend(fontsize=10)
    axes3[2].grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, model_pass_rates):
        axes3[2].text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate:.1%}", ha="center", fontsize=11, fontweight="bold")
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
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
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Performance Over Time", fontsize=16, fontweight="bold")
    
    model_colors = _get_system_colors(models, "tab10")
    
    for model in models:
        model_times = [d["timestamp"] for d in time_data if d["model"] == model]
        model_scores = [d["score"] for d in time_data if d["model"] == model]
        axes[0].plot(model_times, model_scores, marker="o", linestyle="-", linewidth=2, markersize=6, 
                    label=model, color=model_colors[model], alpha=0.7)
    
    axes[0].axhline(y=0.75, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Pass Threshold")
    axes[0].set_ylabel("Validation Score", fontsize=12, fontweight="bold")
    axes[0].set_title("Score Over Time by System", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)
    
    for model in models:
        model_times = [d["timestamp"] for d in time_data if d["model"] == model]
        model_tokens = [d["tokens"] for d in time_data if d["model"] == model]
        axes[1].plot(model_times, model_tokens, marker="s", linestyle="-", linewidth=2, markersize=6,
                    label=model, color=model_colors[model], alpha=0.7)
    
    axes[1].set_xlabel("Timestamp", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Total Tokens", fontsize=12, fontweight="bold")
    axes[1].set_title("Token Usage Over Time by System", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
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
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ["green" if pr >= 0.75 else "orange" if pr >= 0.5 else "red" for pr in pass_rates]
    bars = ax.bar(test_ids_sorted, pass_rates, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    ax.axhline(y=0.75, color="blue", linestyle="--", linewidth=2, label="Pass Threshold (75%)")
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate:.1%}", 
               ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    ax.set_xlabel("Test ID (sorted by success rate)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Pass Rate", fontsize=13, fontweight="bold")
    ax.set_title("Pass Rates by Test (Sorted by Success Rate)", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distributions(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot score distributions showing granular performance, not just pass/fail.
    :param results: Test results.
    :param output_dir: Output directory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Score Distributions - Granular Performance Analysis", fontsize=16, fontweight="bold")
    
    all_scores = [r.get("validation", {}).get("overall_score", 0.0) for r in results]
    models = sorted(set(_system_label(r) for r in results))
    test_ids = sorted(set(r.get("test_metadata", {}).get("test_id", "unknown") for r in results))
    
    axes[0, 0].hist(all_scores, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].axvline(np.mean(all_scores), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(all_scores):.2f}")
    axes[0, 0].axvline(0.75, color="orange", linestyle="--", linewidth=2, label="Pass Threshold: 0.75")
    axes[0, 0].set_xlabel("Overall Score", fontsize=12)
    axes[0, 0].set_ylabel("Frequency", fontsize=12)
    axes[0, 0].set_title("Overall Score Distribution", fontsize=14, fontweight="bold")
    axes[0, 0].legend(fontsize=11)
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
    axes[0, 1].set_ylabel("Average Score", fontsize=12)
    axes[0, 1].set_title("Score by System (with std dev)", fontsize=14, fontweight="bold")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(axis="y", alpha=0.3)
    for i, (bar, mean) in enumerate(zip(bars, model_means)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, mean + model_stds[i] + 0.02, f"{mean:.2f}", ha="center", fontsize=10, fontweight="bold")
    
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
    axes[1, 0].set_xlabel("Average Score", fontsize=12)
    axes[1, 0].set_ylabel("Test ID", fontsize=12)
    axes[1, 0].set_title("Score by Test (with std dev)", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlim(0, 1.0)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(axis="x", alpha=0.3)
    for i, (bar, mean) in enumerate(zip(bars2, test_means)):
        axes[1, 0].text(mean + test_stds[i] + 0.02, bar.get_y() + bar.get_height()/2, f"{mean:.2f}", va="center", fontsize=9, fontweight="bold")
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 24))
    fig.suptitle("Specific Validation Checks - Performance Analysis", fontsize=16, fontweight="bold")
    
    check_names = found_checks
    success_rates = [check_data[c]["passed"] / check_data[c]["total"] if check_data[c]["total"] > 0 else 0.0 for c in check_names]
    avg_scores = [np.mean(check_data[c]["scores"]) if check_data[c]["scores"] else 0.0 for c in check_names]
    total_counts = [check_data[c]["total"] for c in check_names]
    
    colors = ["red" if sr == 0.0 else "orange" if sr < 0.5 else "green" for sr in success_rates]
    
    bars = axes[0, 0].barh(check_names, success_rates, color=colors, alpha=0.8, edgecolor="black")
    axes[0, 0].axvline(0.75, color="blue", linestyle="--", linewidth=2, label="Pass Threshold (75%)")
    axes[0, 0].set_xlabel("Success Rate", fontsize=12)
    axes[0, 0].set_ylabel("Validation Check", fontsize=12)
    axes[0, 0].set_title("Success Rate by Check Type", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlim(0, 1.0)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(axis="x", alpha=0.3)
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        axes[0, 0].text(rate + 0.02, bar.get_y() + bar.get_height()/2, f"{rate:.1%}", va="center", fontsize=10, fontweight="bold")
    
    bars = axes[0, 1].barh(check_names, avg_scores, color=colors, alpha=0.8, edgecolor="black")
    axes[0, 1].axvline(0.75, color="blue", linestyle="--", linewidth=2, label="Pass Threshold (75%)")
    axes[0, 1].set_xlabel("Average Score", fontsize=12)
    axes[0, 1].set_ylabel("Validation Check", fontsize=12)
    axes[0, 1].set_title("Average Score by Check Type", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlim(0, 1.0)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(axis="x", alpha=0.3)
    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        axes[0, 1].text(score + 0.02, bar.get_y() + bar.get_height()/2, f"{score:.2f}", va="center", fontsize=10, fontweight="bold")
    
    bars = axes[1, 0].barh(check_names, total_counts, color="steelblue", alpha=0.8, edgecolor="black")
    axes[1, 0].set_xlabel("Total Occurrences", fontsize=12)
    axes[1, 0].set_ylabel("Validation Check", fontsize=12)
    axes[1, 0].set_title("Check Occurrence Count", fontsize=14, fontweight="bold")
    axes[1, 0].grid(axis="x", alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, total_counts)):
        axes[1, 0].text(count + max(total_counts)*0.02, bar.get_y() + bar.get_height()/2, f"{count}", va="center", fontsize=10, fontweight="bold")
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Token Usage vs Actions & Data Amounts - Comprehensive Analysis", fontsize=16, fontweight="bold")
    
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
    
    ax1.set_xlabel("Test ID (sorted by token usage)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Tokens", fontsize=12, color="steelblue", fontweight="bold")
    ax2.set_ylabel("Action Count", fontsize=12, color="black", fontweight="bold")
    ax1.set_title("Tokens vs Searches vs Visits by Test", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_ids_sorted, rotation=45, ha="right", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width, avg_tokens, width, label="Tokens", alpha=0.8, color="steelblue", edgecolor="black")
    bars2 = ax2.bar(x, avg_search_data, width, label="Search Data (KB)", alpha=0.8, color="orange", edgecolor="black")
    bars3 = ax2.bar(x + width, avg_visit_data, width, label="Visit Data (KB)", alpha=0.8, color="purple", edgecolor="black")
    
    ax1.set_xlabel("Test ID (sorted by token usage)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Tokens", fontsize=12, color="steelblue", fontweight="bold")
    ax2.set_ylabel("Data Amount (KB)", fontsize=12, color="black", fontweight="bold")
    ax1.set_title("Tokens vs Search/Visit Data Amounts by Test", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_ids_sorted, rotation=45, ha="right", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)
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
    
    ax1.set_xlabel("System", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Tokens", fontsize=12, color="steelblue", fontweight="bold")
    ax2.set_ylabel("Action Count", fontsize=12, color="black", fontweight="bold")
    ax1.set_title("Tokens vs Searches vs Visits by System", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width, avg_tokens_model, width, label="Tokens", alpha=0.8, color="steelblue", edgecolor="black")
    bars2 = ax2.bar(x, avg_search_data_model, width, label="Search Data (KB)", alpha=0.8, color="orange", edgecolor="black")
    bars3 = ax2.bar(x + width, avg_visit_data_model, width, label="Visit Data (KB)", alpha=0.8, color="purple", edgecolor="black")
    
    ax1.set_xlabel("System", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Tokens", fontsize=12, color="steelblue", fontweight="bold")
    ax2.set_ylabel("Data Amount (KB)", fontsize=12, color="black", fontweight="bold")
    ax1.set_title("Tokens vs Search/Visit Data Amounts by System", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)
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
    
    ax.set_xlabel("Average Score", fontsize=13, fontweight="bold")
    ax.set_ylabel("Validation Check", fontsize=13, fontweight="bold")
    ax.set_title("Validation Check Performance", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=11)
    
    for i, (name, score) in enumerate(zip(check_names, scores)):
        ax.text(score + 0.02, i, f"{score:.2f}", va="center", fontsize=10, fontweight="bold")
    
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
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(test_ids_sorted))
    width = 0.35
    
    system_colors = _get_system_colors(models, "Set3")
    for i, model in enumerate(models):
        scores = [np.mean(test_scores[tid][model]) if model in test_scores[tid] else 0.0 for tid in test_ids_sorted]
        bars = ax.bar(x + i * width, scores, width, label=model, alpha=0.8, color=system_colors[model], edgecolor="black", linewidth=1)
        for j, (bar, score) in enumerate(zip(bars, scores)):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, score + 0.02, f"{score:.2f}", 
                       ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xlabel("Test ID (sorted by success rate)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Validation Score", fontsize=13, fontweight="bold")
    ax.set_title("Validation Scores by Test and System (Sorted by Success Rate)", fontsize=15, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(test_ids_sorted, fontsize=11)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.75, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Pass Threshold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "validation_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

