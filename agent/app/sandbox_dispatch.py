"""One dispatcher for every sandbox action, shared by the native engine and codebench.

``idea_policies/extra_actions/sandbox_tools.py`` (the native ``LeafAction`` pack) and
``testing/execution_compiled_code.py`` (the ``graph_compiled_code`` leaf loop) both turn a
model-authored ``{action, args}`` object into a :class:`agent.app.connector_sandbox.SandboxConnector`
call. They used to do that translation twice, and the two copies had already drifted: the path-alias
tables overlapped but were not equal (the native pack took ``dir``/``directory``, codebench took
``target``/``file_path``), and codebench's loop wired only four of the native pack's actions, so an
agent working inside the real Docker sandbox had no way to search or inspect files by pattern. This
module is the ONE place that translation happens; both call sites are now thin adapters over it.

What each call site EXPOSES is still deliberately different, and that difference is not drift:

* **codebench** advertises all of :data:`ACTION_NAMES`. It runs inside the network-isolated
  codebench container, whose entire purpose is executing the agent's code.
* **the native pack** advertises EIGHT actions — ``read_file``, ``write_file``, ``list_dir`` and the
  five read-only shell actions (``count_lines``, ``word_count``, ``head_file``, ``disk_usage``,
  ``find_files``). It has NO ``patch_file``: the native engine can only full-overwrite a file today.
  And it deliberately has no ``run_python`` / ``run_pytest`` / ``search_web`` — handing arbitrary
  code execution to a general web-research engine is a materially bigger capability decision than
  unifying a dispatcher, and re-homing the dispatch logic here does not make it. Nothing in this
  module installs anything; ``SandboxToolPack.ACTION_CLASSES`` is still the only list that decides
  what the engine can reach, and it is still opt-in behind ``ToolsConfig.sandbox_pack_enabled``.

**Result shape.** Every action returns the connector's own dict — ``{"ok": bool, "action": str,
...}`` plus whatever that action produced (``path``, ``output``, ``error``, ``bytes``, ``entries``,
``stdout``, ``stderr``, ``exit_code``, ``occurrences``, ``matched``, ``results``, ``timed_out``) —
and never raises: a missing slot, an escaping path, a malformed argument and a crashed subprocess
all come back as a failed result the model can read and correct. Each call site adapts that one
shape to its own contract (the native pack maps ``ok`` -> ``success`` in
``SandboxLeafAction._from_sandbox``; codebench's ``_observation()`` reads ``ok``/``output``/``error``
straight off it), so a field dropped here silently breaks both — which is what
``tests/sandbox_dispatch_test.py`` pins.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import edge
    from agent.app.connector_sandbox import SandboxConnector


#: Arg keys accepted for "the path to work on", most specific first — the UNION of the two tables
#: that used to live in the two call sites. A weak model names this slot inconsistently; accepting
#: its synonyms costs nothing and saves a wasted step.
PATH_KEYS: Tuple[str, ...] = ("path", "relpath", "file", "filename", "dir", "directory",
                              "target", "file_path")
#: Arg keys carrying the TEXT of a file to write.
CONTENT_KEYS: Tuple[str, ...] = ("content", "text", "body", "code", "data")
#: Arg keys carrying a python program (or the path of one) for ``run_python``.
CODE_KEYS: Tuple[str, ...] = ("code", "script", "source", "content", "path", "file")
#: Arg keys carrying a web-search query.
QUERY_KEYS: Tuple[str, ...] = ("query", "q", "search", "text")
#: ``patch_file``'s two snippet slots.
OLD_KEYS: Tuple[str, ...] = ("old", "old_text", "find")
NEW_KEYS: Tuple[str, ...] = ("new", "new_text", "replace")
#: ``count_lines``' optional "count only matching lines" pattern.
PATTERN_KEYS: Tuple[str, ...] = ("pattern", "match")
#: ``find_files``' filename glob.
NAME_KEYS: Tuple[str, ...] = ("name", "glob", "pattern", "filename")
#: ``head_file``'s line count.
LINES_KEYS: Tuple[str, ...] = ("lines", "n", "count")

#: ``head_file`` defaults and clamp — a head that returns a whole file is not a head.
_HEAD_DEFAULT_LINES = 10
_HEAD_MAX_LINES = 200


def _value(args: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    """First non-empty value among ``keys``, stripped, or ``None``.

    Used for the slots where surrounding whitespace is never meaningful and an empty value is the
    same as an absent one: paths, patterns, globs, queries, line counts, program text.
    """
    for key in keys:
        value = args.get(key)
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value)
        if text.strip():
            return text.strip()
    return None


def _verbatim(args: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    """Value of the first PRESENT key among ``keys``, untouched, or ``None`` when none is present.

    File content is not a slot that may be stripped or coalesced: ``""`` is a legitimate write (an
    empty ``__init__.py``), a trailing newline is part of the file, and "the key is missing" is a
    different failure ("say what to write") from "the key is empty".
    """
    for key in keys:
        value = args.get(key)
        if value is None:
            continue
        return value if isinstance(value, str) else json.dumps(value)
    return None


def _fail(action: str, error: str, **extra: Any) -> Dict[str, Any]:
    """A failed result in the connector's shape (so both adapters map it like any other)."""
    return {"ok": False, "action": action, "error": error, **extra}


def _missing(action: str, slot: str, hint: str = "str") -> Dict[str, Any]:
    return _fail(action, f"missing '{slot}' detail ({hint})")


def _confined_file(sandbox: "SandboxConnector", action: str, args: Dict[str, Any]):
    """``(workdir-relative path, None)`` for an existing confined FILE, else ``(None, failure)``.

    Distinguishes the two refusals a weak model must react to differently: a path that escapes the
    workdir (rephrase it) versus one that simply is not there (make it first).
    """
    relpath = _value(args, PATH_KEYS)
    if relpath is None:
        return None, _missing(action, "path")
    target = sandbox.confine(relpath)
    if target is None:
        return None, _fail(action, f"path escapes the workdir: {relpath}", path=relpath)
    if not target.is_file():
        return None, _fail(action, f"no such file in the workdir: {relpath}", path=relpath)
    return sandbox.rel(target), None


async def _readonly(sandbox: "SandboxConnector", action: str, argv: Sequence[str],
                    path: Optional[str] = None) -> Dict[str, Any]:
    """Run one allow-listed read-only command and stamp the caller's confined path onto the result.

    ``run_readonly`` builds its result from the argv, which for these actions never carries the
    path slot in a recoverable form, so the caller's already-confined ``path`` is authoritative.
    """
    result = await sandbox.run_readonly(action, list(argv))
    if path is not None:
        result["path"] = path
    return result


# --- file actions ------------------------------------------------------------------------------
async def _read_file(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    relpath = _value(args, PATH_KEYS)
    if relpath is None:
        return _missing("read_file", "path")
    return await sandbox.read_file(relpath)


async def _write_file(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    relpath = _value(args, PATH_KEYS)
    if relpath is None:
        return _missing("write_file", "path")
    content = _verbatim(args, CONTENT_KEYS)
    if content is None:
        return _missing("write_file", "content")
    return await sandbox.write_file(relpath, content)


async def _list_dir(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    return await sandbox.list_dir(_value(args, PATH_KEYS) or "")


async def _patch_file(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    relpath = _value(args, PATH_KEYS)
    if relpath is None:
        return _missing("patch_file", "path")
    return await sandbox.patch_file(relpath, _value(args, OLD_KEYS) or "",
                                    _verbatim(args, NEW_KEYS) or "")


# --- read-only shell actions -------------------------------------------------------------------
async def _count_lines(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    rel, refusal = _confined_file(sandbox, "count_lines", args)
    if refusal is not None:
        return refusal
    pattern = _value(args, PATTERN_KEYS)
    argv = ["grep", "-c", "--", pattern, rel] if pattern else ["wc", "-l", rel]
    return await _readonly(sandbox, "count_lines", argv, path=rel)


async def _word_count(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    rel, refusal = _confined_file(sandbox, "word_count", args)
    if refusal is not None:
        return refusal
    return await _readonly(sandbox, "word_count", ["wc", "-w", rel], path=rel)


async def _head_file(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    rel, refusal = _confined_file(sandbox, "head_file", args)
    if refusal is not None:
        return refusal
    try:
        lines = int(_value(args, LINES_KEYS) or _HEAD_DEFAULT_LINES)
    except (TypeError, ValueError):
        lines = _HEAD_DEFAULT_LINES
    lines = max(1, min(_HEAD_MAX_LINES, lines))
    return await _readonly(sandbox, "head_file", ["head", "-n", str(lines), rel], path=rel)


async def _disk_usage(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    relpath = _value(args, PATH_KEYS) or "."
    target = sandbox.confine(relpath)
    if target is None:
        return _fail("disk_usage", f"path escapes the workdir: {relpath}", path=relpath)
    if not target.exists():
        return _fail("disk_usage", f"no such path under the workdir: {relpath}", path=relpath)
    rel = sandbox.rel(target)
    return await _readonly(sandbox, "disk_usage", ["du", "-sh", rel], path=rel)


async def _find_files(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = _value(args, NAME_KEYS)
    if pattern is None:
        return _missing("find_files", "name", "a filename glob, e.g. '*.py'")
    return await _readonly(sandbox, "find_files", ["find", ".", "-name", pattern, "-type", "f"])


# --- execution + web actions ---------------------------------------------------------------------
async def _run_python(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    return await sandbox.run_python(_value(args, CODE_KEYS) or "")


async def _run_pytest(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    return await sandbox.run_pytest(_value(args, PATH_KEYS) or "")


async def _search_web(sandbox: "SandboxConnector", args: Dict[str, Any]) -> Dict[str, Any]:
    return await sandbox.search_web(_value(args, QUERY_KEYS) or "")


#: Every sandbox action, by name. Order is the order they are advertised in.
_HANDLERS = {
    "write_file": _write_file,
    "read_file": _read_file,
    "list_dir": _list_dir,
    "patch_file": _patch_file,
    "run_python": _run_python,
    "run_pytest": _run_pytest,
    "search_web": _search_web,
    "count_lines": _count_lines,
    "word_count": _word_count,
    "head_file": _head_file,
    "disk_usage": _disk_usage,
    "find_files": _find_files,
}

#: The full sandbox vocabulary. A call site may advertise a SUBSET of this (see the module
#: docstring) — being dispatchable here is not the same as being reachable from a run.
ACTION_NAMES: Tuple[str, ...] = tuple(_HANDLERS)


async def dispatch_sandbox_action(
    sandbox: "SandboxConnector",
    action: str,
    args: Dict[str, Any],
    *,
    vocabulary: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run ONE sandbox action and return its result dict. Never raises.

    :param sandbox: The workdir-confined connector every action delegates to. Confinement, the
        read-only command allow-list and the byte/file budgets live there, not here — this module
        only decides which method to call and which arg alias fills which slot.
    :param action: The action name (case/whitespace-insensitive).
    :param args: The model-authored argument object; a non-dict is treated as empty.
    :param vocabulary: Verbs to name in the "unknown action" message, so each call site's hint
        matches what IT actually offers (codebench adds ``finish``, which is a loop verb rather
        than a sandbox action). Defaults to :data:`ACTION_NAMES`.
    """
    if not isinstance(args, dict):
        args = {}
    name = str(action or "").strip().lower()
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "action": action or "(none)",
                "error": f"unknown action; use {'/'.join(vocabulary or ACTION_NAMES)}"}
    return await handler(sandbox, args)
