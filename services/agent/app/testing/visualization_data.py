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
    
    # Result files can live either at the top level (legacy layout) or nested
    # in {run_id}_{test_id}_{model_short}/ subdirectories (current layout).
    # Match either, and resolve run_id against filename OR parent directory.
    candidates = list(results_dir.glob(pattern)) + list(results_dir.rglob(pattern))
    seen_paths: set = set()
    deduped = []
    for f in candidates:
        if f in seen_paths:
            continue
        seen_paths.add(f)
        deduped.append(f)

    for result_file in sorted(deduped):
        if "summary" in result_file.name or "_report_" in result_file.name:
            continue

        if run_id_filter:
            name_match = result_file.name.startswith(run_id_filter)
            parent_match = result_file.parent.name.startswith(run_id_filter)
            if not (name_match or parent_match):
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

    # Look at both top-level and nested JSON files.
    candidates = list(results_dir.glob("*.json")) + list(results_dir.rglob("*.json"))
    seen_paths: set = set()
    for result_file in sorted(candidates):
        if result_file in seen_paths:
            continue
        seen_paths.add(result_file)
        if "summary" in result_file.name:
            continue
        run_id = None
        match = re.match(r"(\d{8}_\d{6})_", result_file.name)
        if match:
            run_id = match.group(1)
        else:
            parent_match = re.match(r"(\d{8}_\d{6})_", result_file.parent.name)
            if parent_match:
                run_id = parent_match.group(1)
        if run_id:
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
