"""
Summary statistics functions for visualization.
"""

from typing import Dict, Any, List
from collections import defaultdict
import numpy as np

from .visualization_helpers import _system_label

def calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics from test results.
    :param results: Test results.
    :return: Summary statistics dict.
    """
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("validation", {}).get("overall_passed", False))
    scores = [r.get("validation", {}).get("overall_score", 0.0) for r in results]
    avg_score = np.mean(scores) if scores else 0.0
    median_score = np.median(scores) if scores else 0.0
    std_score = np.std(scores) if scores else 0.0
    
    total_duration = sum(r.get("execution", {}).get("duration_seconds", 0) for r in results)
    total_tokens = sum(r.get("execution", {}).get("observability", {}).get("llm", {}).get("total_tokens", 0) for r in results)
    total_searches = sum(r.get("execution", {}).get("observability", {}).get("search", {}).get("count", 0) for r in results)
    total_visits = sum(r.get("execution", {}).get("observability", {}).get("visit", {}).get("count", 0) for r in results)
    
    test_ids = set(r.get("test_metadata", {}).get("test_id", "unknown") for r in results)
    models = set(_system_label(r) for r in results)
    
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
            "avg_score": np.mean(check_scores) if check_scores else 0.0,
            "total": check_total,
        }
    
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
        "unique_tests": len(test_ids),
        "unique_models": len(models),
        "test_ids": sorted(test_ids),
        "models": sorted(models),
        "check_stats": check_stats,
    }


def print_summary(stats: Dict[str, Any]):
    """
    Print comprehensive summary statistics.
    :param stats: Summary statistics dict.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*80)
    print(f"Total Tests: {stats['total_tests']}")
    print(f"Passed: {stats['passed_tests']} ({stats['pass_rate']:.1%})")
    print(f"Score Statistics:")
    print(f"  Average: {stats['avg_score']:.3f}")
    print(f"  Median: {stats['median_score']:.3f}")
    print(f"  Std Dev: {stats['std_score']:.3f}")
    print(f"\nExecution Metrics:")
    print(f"  Total Duration: {stats['total_duration']:.1f}s")
    print(f"  Average Duration: {stats['avg_duration']:.1f}s per test")
    print(f"  Total Tokens: {stats['total_tokens']:,}")
    print(f"  Average Tokens: {stats['avg_tokens']:,.0f} per test")
    print(f"\nActions:")
    print(f"  Total Searches: {stats['total_searches']} (avg: {stats['avg_searches']:.1f} per test)")
    print(f"  Total Visits: {stats['total_visits']} (avg: {stats['avg_visits']:.1f} per test)")
    print(f"\nCoverage:")
    print(f"  Unique Tests: {stats['unique_tests']} ({', '.join(stats['test_ids'])})")
    print(f"  Unique Systems: {stats['unique_models']} ({', '.join(stats['models'])})")
    
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
    
    print("="*80)
