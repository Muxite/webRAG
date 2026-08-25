import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

_UNSAFE_WITH_COLON = re.compile(r"[\\/:\s]+")
_UNSAFE_NO_COLON = re.compile(r"[\\/\s]+")
_TRUTHY = {"1", "true", "yes", "on"}


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


def traces_retained() -> bool:
    """Whether per-cell JSONL traces survive a successful run.

    Every execution variant used to end with an unconditional ``trace_path.unlink()`` on the
    success path, so a trace existed only when the run had *crashed*. That is why a 96-cell
    baseline yielded 0 recoverable traces and its forensics had to be reconstructed from
    result-graph topology. Retention is opt-in rather than default-on so ordinary runs keep
    their current disk footprint.

    :returns: True when ``IDEA_TEST_KEEP_TRACES`` is set to a truthy value.
    """
    return os.environ.get("IDEA_TEST_KEEP_TRACES", "").strip().lower() in _TRUTHY


def llm_io_capture_enabled(report_verbosity: int = 1) -> bool:
    """Whether connectors capture raw prompt/completion text.

    Full capture already existed but was welded to ``IDEA_TEST_REPORT_VERBOSITY >= 3``, which
    also inflates every other artifact -- so getting replayable LLM I/O meant paying for
    maximum verbosity everywhere. ``IDEA_TEST_CAPTURE_LLM_IO`` turns capture on by itself; the
    text lands in the JSONL trace, while ``testing.utils.slim_telemetry_raw`` keeps it out of
    the result JSON.

    :param report_verbosity: The effective ``IDEA_TEST_REPORT_VERBOSITY``, still honoured so
        existing verbosity-3 workflows are unchanged.
    :returns: True when raw text should be captured.
    """
    if os.environ.get("IDEA_TEST_CAPTURE_LLM_IO", "").strip().lower() in _TRUTHY:
        return True
    return report_verbosity >= 3


def build_trace_path(
    results_dir: Path,
    run_stamp: str,
    test_id: str,
    model_name: str,
    variant: str,
    cell_tag: str = "",
) -> Path:
    """Build the JSONL trace path for one benchmark cell.

    ``cell_tag`` carries the effort tier / settings fingerprint / repeat index that the result
    JSON's own filename carries (``idea_test_runner``). Without it, repeats and A/B conditions
    of the same cell collided on one path -- and since :class:`TraceRecorder` opens in APPEND
    mode, retaining traces under colliding names would interleave concurrent cells into a
    single corrupt file. Naming and retention therefore have to change together.

    :param results_dir: Directory the trace is written into.
    :param run_stamp: Run identifier, shared with the result JSON.
    :param test_id: Benchmark task id.
    :param model_name: Execution model id (may embed ``/`` or ``:``).
    :param variant: Execution variant name.
    :param cell_tag: Disambiguating suffix, e.g. ``"_t2_cfgef66f4d7_r3"``.
    :returns: The trace path.
    """
    safe_model = sanitize_path_component(model_name)
    return results_dir / f"{run_stamp}_{test_id}_{safe_model}_{variant}{cell_tag}.jsonl"


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
