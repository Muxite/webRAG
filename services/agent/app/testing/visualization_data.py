"""
Data loading functions for visualization.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict


def load_test_results(results_dir: Path, pattern: str = "*.json", run_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load test result JSON files, optionally filtered by run ID.
    :param results_dir: Directory containing results.
    :param pattern: File pattern to match.
    :param run_id_filter: Optional run ID to filter by (e.g., "20260221_205459"). Matches files starting with this timestamp.
    :return: List of test results.
    """
    results = []
    files_loaded = 0
    files_skipped = 0
    
    for result_file in sorted(results_dir.glob(pattern)):
        if "summary" in result_file.name or "_report_" in result_file.name:
            continue
        
        if run_id_filter:
            if not result_file.name.startswith(run_id_filter):
                files_skipped += 1
                continue
        
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            results.append(data)
            files_loaded += 1
        except Exception as exc:
            print(f"Warning: Failed to load {result_file.name}: {exc}")
            files_skipped += 1
    
    if run_id_filter:
        print(f"Loaded {files_loaded} result file(s) for run_id: {run_id_filter}")
        if files_skipped > 0:
            print(f"  (skipped {files_skipped} file(s) not matching run_id)")
    
    return results


def extract_run_id_from_filename(filename: str) -> str:
    """
    Extract run ID from filename.
    :param filename: Filename.
    :return: Run ID.
    """
    match = re.match(r"(\d{8}_\d{6})_", filename)
    return match.group(1) if match else "unknown"


def list_available_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """
    List available run IDs from result files with metadata.
    :param results_dir: Directory containing results.
    :return: List of run info dicts with run_id, count, and latest file.
    """
    run_data = defaultdict(lambda: {"count": 0, "files": []})
    
    for result_file in sorted(results_dir.glob("*.json")):
        if "summary" in result_file.name:
            continue
        match = re.match(r"(\d{8}_\d{6})_", result_file.name)
        if match:
            run_id = match.group(1)
            run_data[run_id]["count"] += 1
            run_data[run_id]["files"].append(result_file.name)
    
    runs = []
    for run_id in sorted(run_data.keys(), reverse=True):
        runs.append({
            "run_id": run_id,
            "count": run_data[run_id]["count"],
            "latest_file": run_data[run_id]["files"][-1] if run_data[run_id]["files"] else None,
        })
    
    return runs
