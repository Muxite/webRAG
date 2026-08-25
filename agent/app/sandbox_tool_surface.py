"""One sandbox capability surface, shared by every arm.

The native engine could already work a sandbox filesystem through
:class:`~agent.app.idea_policies.extra_actions.sandbox_tools.SandboxToolPack`, but the flat arms
(``sequential_react``, ``langgraph_react``) had no sandbox surface at all, and the codebench
matrix drives only the *compiled* variant. So a closed-environment task was measurable on one
arm — which cannot test any claim of the form "the DAG does better than a linear agent here",
because there is nothing to compare against.

This module is the single place that says WHAT the surface is (:data:`PARITY_ACTIONS`) and how
one call is turned into a text observation (:func:`run_sandbox_action`). The translation from
``{action, args}`` to a connector call already had exactly one home,
:mod:`agent.app.sandbox_dispatch`, after two copies of it drifted; this adds a second consumer
of that dispatcher rather than a third copy of the translation.

**The capability boundary is deliberate and is NOT widened here.** The surface is the same eight
verbs the native pack exposes: three file actions plus five read-only shell actions. There is no
``run_python`` / ``run_pytest`` / ``search_web``, even though the shared dispatcher can reach
them — handing arbitrary code execution to a web-research agent is a materially bigger decision
than giving three arms the same file surface, and proximity is not authorization. Confinement,
the read-only command allow-list and the byte/file budgets all live in
:class:`~agent.app.connector_sandbox.SandboxConnector`, not here.

Reaching any of this still requires a run that actually carries a ``connector_sandbox``; a plain
web-research run has none, and every helper here degrades to a clean refusal rather than an
exception.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

from agent.app.sandbox_dispatch import dispatch_sandbox_action

#: The eight verbs every arm gets. Kept identical to ``SandboxToolPack.ACTION_CLASSES`` -- a
#: divergence here would mean the arms are being compared on different capabilities, which is
#: the confound the whole exercise exists to avoid. Pinned by
#: ``agent/tests/sandbox_tool_surface_test.py``.
PARITY_ACTIONS: Tuple[str, ...] = (
    "read_file",
    "write_file",
    "list_dir",
    "count_lines",
    "word_count",
    "head_file",
    "disk_usage",
    "find_files",
)

#: One prompt line per verb: what it does and which slots it fills. Deliberately terse -- these
#: go into a weak model's system prompt, where every token competes with the task itself.
ACTION_HINTS: Dict[str, str] = {
    "read_file": 'read_file {"path": "<workdir-relative file>"} — read a text file.',
    "write_file": 'write_file {"path": "<file>", "content": "<text>"} — create or OVERWRITE a '
                  "file (there is no partial patch; write the whole file).",
    "list_dir": 'list_dir {"path": "<dir, default \'.\'>"} — list a directory.',
    "count_lines": 'count_lines {"path": "<file>"} — count lines in a file.',
    "word_count": 'word_count {"path": "<file>"} — count words in a file.',
    "head_file": 'head_file {"path": "<file>", "lines": <n>} — read the first n lines.',
    "disk_usage": 'disk_usage {"path": "<path, default \'.\'>"} — size on disk.',
    "find_files": 'find_files {"name": "<glob, e.g. *.txt>"} — find files by name pattern.',
}

#: Returned when the run has no workdir. A refusal, never an exception -- an arm that asks for a
#: file in a run that has no sandbox should read an observation and move on.
NO_SANDBOX = "NO SANDBOX: this run has no workdir, so filesystem actions are unavailable."

_MAX_OUTPUT_CHARS = 4000


def sandbox_menu(actions: Sequence[str] = PARITY_ACTIONS) -> str:
    """Render the prompt block describing the sandbox verbs.

    :param actions: Which verbs to advertise; defaults to the full parity set.
    :returns: A newline-joined menu, or ``""`` when nothing is advertised.
    """
    lines = [f"- {ACTION_HINTS[name]}" for name in actions if name in ACTION_HINTS]
    if not lines:
        return ""
    return (
        "FILE ACTIONS (a working directory is available to you):\n"
        + "\n".join(lines)
        + "\nPaths are relative to the working directory; anything outside it is refused."
    )


def format_sandbox_result(action: str, result: Dict[str, Any]) -> str:
    """Turn a connector result dict into the text observation a flat arm consumes.

    The native engine keeps the dict (its leaf results are structured); the ReAct-shaped arms
    read a scratchpad, so the same information has to survive as prose without losing the
    fields a task's validator might key on.

    :param action: The verb that ran.
    :param result: The connector's own result dict.
    :returns: A single observation string.
    """
    if not isinstance(result, dict):
        return f"{action.upper()} ERROR: malformed result"
    if not result.get("ok"):
        return f"{action.upper()} ERROR: {result.get('error') or 'failed'}"

    for key in ("output", "content", "stdout"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return f"{action.upper()} OK:\n{value[:_MAX_OUTPUT_CHARS]}"

    entries = result.get("entries")
    if isinstance(entries, list):
        return f"{action.upper()} OK:\n" + "\n".join(str(e) for e in entries)[:_MAX_OUTPUT_CHARS]

    results = result.get("results")
    if isinstance(results, list):
        return f"{action.upper()} OK:\n" + "\n".join(str(r) for r in results)[:_MAX_OUTPUT_CHARS]

    payload = {k: v for k, v in result.items() if k not in ("ok", "action")}
    if not payload:
        return f"{action.upper()} OK"
    return f"{action.upper()} OK: {json.dumps(payload, default=str)[:_MAX_OUTPUT_CHARS]}"


async def run_sandbox_action(
    sandbox: Optional[Any], action: str, args: Optional[Dict[str, Any]],
) -> str:
    """Run one sandbox verb and return its observation text. Never raises.

    :param sandbox: The run's ``SandboxConnector``, or ``None`` when it has no workdir.
    :param action: One of :data:`PARITY_ACTIONS`.
    :param args: Model-authored slots.
    :returns: The observation.
    """
    if sandbox is None:
        return NO_SANDBOX
    if action not in PARITY_ACTIONS:
        return (
            f"UNKNOWN FILE ACTION '{action}'. Available: {', '.join(PARITY_ACTIONS)}."
        )
    try:
        result = await dispatch_sandbox_action(
            sandbox, action, args if isinstance(args, dict) else {},
            vocabulary=PARITY_ACTIONS,
        )
    except Exception as exc:  # noqa: BLE001 - a tool call must never kill the run
        return f"{action.upper()} ERROR: {exc}"
    return format_sandbox_result(action, result)
