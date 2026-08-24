import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

_UNSAFE_WITH_COLON = re.compile(r"[\\/:\s]+")
_UNSAFE_NO_COLON = re.compile(r"[\\/\s]+")


def sanitize_path_component(
    value: str, *, replacement: str = "-", preserve_colon: bool = True
) -> str:
    """Make a string safe to embed as (part of) a single filesystem path component.

    Benchmark/trace/result filenames interpolate raw, config-supplied identifiers
    straight into a path string (e.g. an OpenRouter model id like
    ``openai/gpt-5-mini`` or an Ollama tag like ``qwen2.5:7b``). Left unsanitized, a
    ``/`` (or ``\\`` on Windows) is silently interpreted as a path separator:
    ``Path.mkdir(parents=True)`` creates an unintended subdirectory instead of
    raising, so the intended file is never written where callers expect it and an
    empty orphan directory is left behind once any downstream cleanup unlinks just
    the file. This collapses path separators and whitespace into a single
    ``replacement`` character so the result is always one path component, while
    staying human-readable and still unambiguously identifying the source value
    (no hashing).

    ``:`` is NOT a path separator on Linux/macOS (only ``/`` is; the historical
    macOS Finder-display quirk doesn't apply to raw filesystem calls, and ``:`` is
    a fully legal filename byte on ext4/APFS/most Linux/macOS filesystems). It is
    reserved on Windows and in NTFS alternate-data-stream syntax, but this codebase
    only ever runs on Linux, so colon-bearing model ids (``qwen2.5:7b``) are safe to
    leave untouched by default -- doing so keeps trace and result filenames using
    the identical spelling of a model id. Pass ``preserve_colon=False`` for a
    stricter/portable sanitization if ever needed.

    :param value: Raw identifier to sanitize (e.g. a model id).
    :param replacement: Character(s) substituted for each run of unsafe characters.
        Defaults to ``"-"`` to match the result-JSON filename convention, where
        ``_`` is reserved as the field delimiter in the filename template.
    :param preserve_colon: When ``True`` (default), ``:`` is left untouched. When
        ``False``, ``:`` is treated as unsafe too (matches the original, stricter
        trace-only sanitization).
    :return: A string safe to use as one path component; ``"unknown"`` if empty.
    """
    if not value:
        return "unknown"
    pattern = _UNSAFE_NO_COLON if preserve_colon else _UNSAFE_WITH_COLON
    safe = pattern.sub(replacement, value).strip(replacement)
    return safe or "unknown"


def sanitize_model_component(value: str) -> str:
    """Sanitize a model id for a RESULT-JSON filename: ``/`` and ``\\`` -> ``-``,
    whitespace -> ``-``, ``:`` left untouched. Thin, self-documenting wrapper around
    :func:`sanitize_path_component`'s defaults, kept so call sites don't need to
    remember the ``replacement``/``preserve_colon`` argument shape.
    """
    return sanitize_path_component(value, replacement="-", preserve_colon=True)


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
