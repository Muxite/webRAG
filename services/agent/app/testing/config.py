"""
Test configuration and utilities.
"""

import os
from typing import List
from pathlib import Path
import re

MODEL_CANDIDATES = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-nano",
]

MODEL_ALIASES = {
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-4.1-nano": "gpt-4.1-nano",
}

VALIDATION_MODEL = "gpt-5-mini"

TEST_PRIORITY_ORDER = [
    "025", "014", "002", "019", "020", "009", "012", "026", "001", "021", "004", "013", "022",
    "005", "006", "007", "008", "010", "011", "015", "016",
    "017", "018", "023", "024",
]


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model identifiers to canonical names.
    :param model_name: Raw model string.
    :return: Canonical model name.
    """
    candidate = model_name.strip()
    return MODEL_ALIASES.get(candidate, candidate)


def _split_model_values(raw: str) -> List[str]:
    """
    Split model env values into normalized tokens.
    :param raw: Raw env string.
    :return: Parsed model tokens.
    """
    text = (raw or "").strip()
    if not text:
        return []
    normalized = re.sub(r"[;\n\r\t]+", ",", text)
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) == 1 and " " in parts[0]:
        parts = [part.strip() for part in re.split(r"\s+", parts[0])]
    values: List[str] = []
    seen = set()
    for part in parts:
        if not part:
            continue
        model = normalize_model_name(part)
        if not model or model in seen:
            continue
        values.append(model)
        seen.add(model)
    return values


def load_models_from_env() -> List[str]:
    """
    Load test models from environment.
    :return: List of model names.
    """
    raw = os.environ.get("IDEA_TEST_MODELS", "").strip()
    if not raw:
        default_model = os.environ.get("MODEL_NAME", "").strip()
        if default_model:
            parsed_default = _split_model_values(default_model)
            if parsed_default:
                return parsed_default
        return [MODEL_CANDIDATES[0]]
    parsed = _split_model_values(raw)
    return parsed if parsed else [MODEL_CANDIDATES[0]]


def extract_test_id(test_file: Path) -> str:
    """
    Extract test ID from test file name.
    :param test_file: Test file path.
    :return: Test ID string.
    """
    stem = test_file.stem
    if stem.startswith("test_"):
        remaining = stem.replace("test_", "", 1)
        parts = remaining.split("_", 1)
        return parts[0] if parts else ""
    return ""


def filter_test_files_by_priority(test_files: List[Path], priority_count: int = 0) -> List[Path]:
    """
    Filter test files by priority order.
    :param test_files: List of test file paths.
    :param priority_count: Number of priority tests to run (0 = all).
    :return: Filtered list of test files in priority order.
    """
    import logging
    _logger = logging.getLogger(__name__)
    
    test_id_to_file = {extract_test_id(f): f for f in test_files}
    
    if priority_count == 0:
        ordered = []
        for test_id in TEST_PRIORITY_ORDER:
            if test_id in test_id_to_file:
                ordered.append(test_id_to_file[test_id])
        for test_file in test_files:
            test_id = extract_test_id(test_file)
            if test_id not in TEST_PRIORITY_ORDER:
                ordered.append(test_file)
        _logger.info(f"Running all {len(ordered)} tests in priority order")
        return ordered
    
    filtered = []
    for test_id in TEST_PRIORITY_ORDER[:priority_count]:
        if test_id in test_id_to_file:
            filtered.append(test_id_to_file[test_id])
    
    _logger.info(f"Priority mode: Running top {len(filtered)} priority tests: {[extract_test_id(f) for f in filtered]}")
    return filtered
