import json
from pathlib import Path
from typing import Any, Dict


class BenchmarkWriter:
    """
    Append-only JSONL writer for benchmark summaries.
    """
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        self._file.close()

    def append(self, payload: Dict[str, Any]) -> None:
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()
