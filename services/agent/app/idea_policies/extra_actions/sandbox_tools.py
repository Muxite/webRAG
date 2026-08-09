"""Workdir-confined file + read-only shell leaf actions (opt-in, never on by default).

Ported from ``badmodel-lab/localagent/tools/{files,shell}.py`` — the smaller control loop's
tool set — into this engine's ``LeafAction`` shape. What is ported is the CAPABILITY SURFACE and
its narrow-intent discipline, not the confinement code: every filesystem/subprocess call here
delegates to :class:`agent.app.connector_sandbox.SandboxConnector`, which already implements the
same ``confine()`` (and has been through codebench's security review, where real
sandbox-escape/symlink-exfiltration bugs were found and fixed). Re-implementing confinement in a
second place is exactly how the two copies drift.

The same reasoning applies one level up: the ``{action, args} -> connector method`` translation
lives in :mod:`agent.app.sandbox_dispatch`, shared with codebench's ``graph_compiled_code`` leaf
loop, so the two call sites cannot drift on which argument aliases they accept. Each action class
below is therefore a NAME, a docstring (which is its prompt line) and an args hint; the work is
the shared dispatcher's.

What this pack exposes is EIGHT actions: ``read_file``, ``write_file``, ``list_dir`` and the five
read-only shell actions. Two absences are deliberate and should not be read as oversights:

* there is no ``patch_file`` — the native engine can only full-overwrite a file today. The
  connector and the shared dispatcher both support patching (codebench uses it); adding the leaf
  action here would be a capability ADDITION, not part of unifying the dispatcher.
* there is no ``run_python`` / ``run_pytest`` / ``search_web``. The shared dispatcher can reach
  them, but exposing arbitrary code execution to a general web-research engine is a materially
  bigger decision than de-duplicating a dispatcher, and proximity is not authorization.

Two properties this module is built to keep:

* **Narrow intent.** The model never authors a command string or an ``op`` argument. It picks one
  named action (``count_lines``, ``head_file``, ...) and fills typed slots; code assembles the
  argv from a fixed read-only allow-list (``SandboxConnector.READONLY_COMMANDS``: wc/grep/du/find/
  head — no writes, no deletes, no network) with no shell, no pipes, no glob expansion.
* **No ambient authority.** These actions are a materially higher-stakes capability class than the
  web-research actions, so they are reachable only when THREE things line up: the pack is
  installed in the registry, its names are in ``allowed_actions``, AND the run's ``AgentIO``
  actually carries a ``connector_sandbox``. A plain web-research run satisfies none of them, and
  a run missing only the last gets a clean "no sandbox in this run" observation rather than any
  filesystem access. Nothing installs this pack by default — the caller does, deliberately::

      engine.install_action_pack(SandboxToolPack(settings=engine.settings))

Every action's path slot is resolved through ``SandboxConnector.confine``, so ``..`` traversal, an
absolute path, and a symlink pointing out of the workdir are all refused as an observation (never
an exception, never a partial read).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.base import fail, ok
from agent.app.sandbox_dispatch import dispatch_sandbox_action


class SandboxLeafAction(LeafAction):
    """Base for the sandbox actions: resolve the run's sandbox, or refuse cleanly.

    Subclasses are declarations — a ``name``, an ``args_hint`` and a docstring the expansion menu
    reads. None of them touches the filesystem, or even the argument object: :meth:`run` hands the
    action name and the node's details straight to the shared dispatcher.
    """

    #: Set on the subclass; kept only for the failure message when no sandbox is wired.
    name = "sandbox"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        sandbox = getattr(io, "connector_sandbox", None)
        if sandbox is None:
            # The capability gate that matters most: a run that was never handed a workdir has no
            # filesystem access at all, whatever the plan says.
            return fail(self.name, "no sandbox workdir is available in this run", retryable=False)
        return await self.run(sandbox, dict(node.details or {}))

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch this action's slots through the shared dispatcher and adapt the result."""
        return self._from_sandbox(await dispatch_sandbox_action(sandbox, self.name, details))

    def _from_sandbox(self, result: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        """Map a ``SandboxConnector`` result dict onto the leaf action-result shape.

        The connector speaks ``ok``; leaf results speak ``success``. Everything else (path,
        output, error, exit code) rides along unchanged so the observation keeps its detail.
        """
        payload = {key: value for key, value in (result or {}).items()
                   if key not in ("ok", "action") and not (key == "path" and value is None)}
        payload.update(extra)  # merged, not double-splatted: both sides may carry ``path``
        if result.get("ok"):
            return ok(self.name, **payload)
        failure = fail(self.name, str(result.get("error") or "sandbox action failed"))
        failure.update({key: value for key, value in payload.items() if key != "error"})
        return failure


# --- file actions --------------------------------------------------------------------------


class ReadFileAction(SandboxLeafAction):
    """Read a text file from the sandbox workdir.

    Confinement: the path is resolved under the workdir by ``SandboxConnector.confine``; anything
    that escapes it (``..``, an absolute path, a symlink out) is refused, not read. A file over
    the configured ``max_file_bytes`` is refused before it is read at all.

    Reads from node details:
      - `path` (str): workdir-relative file path (required).

    Returns `{path, bytes, output}` (content truncated for the observation).
    """

    name = "read_file"
    args_hint = "details={path}"


class WriteFileAction(SandboxLeafAction):
    """Write a text file into the sandbox workdir (create or overwrite).

    Confinement: the path is resolved under the workdir by ``SandboxConnector.confine``, so a
    write can only ever land inside it; the connector's own byte/file-count budgets still apply.

    Reads from node details:
      - `path` (str): workdir-relative file path (required).
      - `content` (str): the text to write.

    Returns `{path, bytes, output}`.
    """

    name = "write_file"
    args_hint = "details={path, content}"


class ListDirAction(SandboxLeafAction):
    """List one directory inside the sandbox workdir.

    Confinement: the path is resolved under the workdir by ``SandboxConnector.confine``; a path
    that escapes it is refused, so no directory outside the workdir can be enumerated.

    Reads from node details:
      - `path` (str): workdir-relative directory, default the workdir root.

    Returns `{path, entries, output}` (long listings are capped, with the overflow counted).
    """

    name = "list_dir"
    args_hint = "details={path?}"


# --- read-only shell actions ---------------------------------------------------------------


class CountLinesAction(SandboxLeafAction):
    """Count lines in a workdir file, or lines matching a pattern (``wc -l`` / ``grep -c``).

    Reads from node details:
      - `path` (str): workdir-relative file (required).
      - `pattern` (str): optional — count MATCHING lines instead of all lines.

    Returns `{path, stdout, exit_code, output}`.
    """

    name = "count_lines"
    args_hint = "details={path, pattern?}"


class WordCountAction(SandboxLeafAction):
    """Count words in a workdir file (``wc -w``).

    Reads from node details:
      - `path` (str): workdir-relative file (required).

    Returns `{path, stdout, exit_code, output}`.
    """

    name = "word_count"
    args_hint = "details={path}"


class HeadFileAction(SandboxLeafAction):
    """Show the first N lines of a workdir file (``head -n``).

    Reads from node details:
      - `path` (str): workdir-relative file (required).
      - `lines` (int): how many lines, default 10, capped at 200.

    Returns `{path, stdout, exit_code, output}`.
    """

    name = "head_file"
    args_hint = "details={path, lines?}"


class DiskUsageAction(SandboxLeafAction):
    """Report the size of a workdir directory or file (``du -sh``).

    Reads from node details:
      - `path` (str): workdir-relative directory, default the workdir root.

    Returns `{path, stdout, exit_code, output}`.
    """

    name = "disk_usage"
    args_hint = "details={path?}"


class FindFilesAction(SandboxLeafAction):
    """Find files by name inside the sandbox workdir (``find . -name <glob> -type f``).

    The glob is passed as a single argv element, so the shell never sees it — no expansion, no
    metacharacter injection — and the search root is always the workdir.

    Reads from node details:
      - `name` (str): filename glob, e.g. `*.py` (required).

    Returns `{stdout, exit_code, output}` — one matching path per line.
    """

    name = "find_files"
    args_hint = "details={name: \"<filename glob, e.g. *.py>\"}"


class SandboxToolPack:
    """The workdir file + read-only shell actions, as one installable bundle.

    Deliberately NOT part of ``ExtraActionPack`` and never installed by default: those are
    read-only public-API lookups, these touch a filesystem. Install it only for a run that has a
    ``SandboxConnector`` on its ``AgentIO``::

        engine.install_action_pack(SandboxToolPack(settings=engine.settings))

    Use ``install_action_pack(..., allow=False)`` to register the classes without permitting them
    (dispatch stays closed until the names are added to ``allowed_actions``).

    This list — not :data:`agent.app.sandbox_dispatch.ACTION_NAMES` — is what the engine can reach.
    See this module's docstring for why ``patch_file`` and the execution actions are not on it.
    """

    name = "sandbox_tools"

    ACTION_CLASSES: List[Type[LeafAction]] = [
        ReadFileAction,
        WriteFileAction,
        ListDirAction,
        CountLinesAction,
        WordCountAction,
        HeadFileAction,
        DiskUsageAction,
        FindFilesAction,
    ]

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self.settings = dict(settings or {})

    def build_instances(self) -> Dict[str, LeafAction]:
        """Instantiate each action class with the pack's shared settings."""
        return {cls.name: cls(settings=self.settings) for cls in self.ACTION_CLASSES}

    def names(self) -> List[str]:
        return [cls.name for cls in self.ACTION_CLASSES]
