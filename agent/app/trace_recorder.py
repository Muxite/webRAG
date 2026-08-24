import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

_UNSAFE_PATH_CHARS = re.compile(r"[\\/:\s]+")


def sanitize_path_component(value: str) -> str:
    """Make a string safe to embed as (part of) a single filesystem path component.

    Benchmark/trace filenames interpolate raw, config-supplied identifiers straight
    into a path string (e.g. an OpenRouter model id like ``openai/gpt-5-mini`` or an
    Ollama tag like ``qwen2.5:7b``). Left unsanitized, a ``/`` (or ``\\`` on Windows)
    is silently interpreted as a path separator: ``Path.mkdir(parents=True)`` creates
    an unintended subdirectory instead of raising, so the intended file is never
    written where callers expect it and an empty orphan directory is left behind
    once any downstream cleanup unlinks just the file. This collapses path
    separators, colons and whitespace into a single ``_`` so the result is always
    one path component, while staying human-readable and still unambiguously
    identifying the source value (no hashing).

    :param value: Raw identifier to sanitize (e.g. a model id).
    :return: A string safe to use as one path component; ``"unknown"`` if empty.
    """
    if not value:
        return "unknown"
    safe = _UNSAFE_PATH_CHARS.sub("_", value).strip("_")
    return safe or "unknown"


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
