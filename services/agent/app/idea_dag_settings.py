from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_idea_dag_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load settings for the idea DAG system.
    :param path: Optional path to settings file.
    :returns: Settings dictionary.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "idea_dag_settings.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
