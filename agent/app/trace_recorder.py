import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TraceRecorder:
    """
    Append-only JSONL recorder for agent traces.
    """
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        self._file.close()

    def record(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "ts": time.time(),
            "event": event,
            "payload": payload or {},
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()
