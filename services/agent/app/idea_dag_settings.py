from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_idea_dag_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parent / "idea_dag_settings.json"
    with path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)
    
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
                    if env_value.lower() in ("null", "none", ""):
                        settings[key] = None
                    else:
                        settings[key] = int(env_value)
                except (ValueError, TypeError):
                    pass
    
    return settings
