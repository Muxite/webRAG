from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_idea_graph_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load idea graph settings from JSON.
    :param path: Optional path override.
    :returns: Settings dictionary.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "idea_graph_settings.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload or {})
