from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_idea_dag_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load settings for the idea DAG system.
    Environment variables can override max_tokens settings:
    - IDEA_DAG_EXPANSION_MAX_TOKENS
    - IDEA_DAG_EVALUATION_MAX_TOKENS
    - IDEA_DAG_FINAL_MAX_TOKENS
    - IDEA_DAG_MERGE_MAX_TOKENS
    :param path: Optional path to settings file.
    :returns: Settings dictionary.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "idea_dag_settings.json"
    with path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)
    
    # Override max_tokens from environment variables if set
    env_overrides = {
        "expansion_max_tokens": os.environ.get("IDEA_DAG_EXPANSION_MAX_TOKENS"),
        "evaluation_max_tokens": os.environ.get("IDEA_DAG_EVALUATION_MAX_TOKENS"),
        "final_max_tokens": os.environ.get("IDEA_DAG_FINAL_MAX_TOKENS"),
        "merge_max_tokens": os.environ.get("IDEA_DAG_MERGE_MAX_TOKENS"),
    }
    
    for key, env_value in env_overrides.items():
        if env_value is not None:
            env_value = env_value.strip()
            if env_value:
                try:
                    # Parse as integer, or None if "null" or empty
                    if env_value.lower() in ("null", "none", ""):
                        settings[key] = None
                    else:
                        settings[key] = int(env_value)
                except (ValueError, TypeError):
                    # Invalid value, keep JSON default
                    pass
    
    return settings
