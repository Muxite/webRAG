"""Workdir-confined file + read-only shell leaf actions (opt-in, never on by default).

Ported from ``badmodel-lab/localagent/tools/{files,shell}.py`` — the smaller control loop's
tool set — into this engine's ``LeafAction`` shape. What is ported is the CAPABILITY SURFACE and
its narrow-intent discipline, not the confinement code: every filesystem/subprocess call here
delegates to :class:`agent.app.connector_sandbox.SandboxConnector`, which already implements the
same ``confine()`` (and has been through codebench's security review, where real
sandbox-escape/symlink-exfiltration bugs were found and fixed). Re-implementing confinement in a
second place is exactly how the two copies drift.

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
  filesystem access. Nothing installs this pack by default — the caller does, deliberately:

      engine.install_action_pack(SandboxToolPack(settings=engine.settings))

Every action's path slot is resolved through ``SandboxConnector.confine``, so ``..`` traversal, an
absolute path, and a symlink pointing out of the workdir are all refused as an observation (never
an exception, never a partial read).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.base import fail, ok


#: Detail keys accepted for "the path to work on", most specific first. A weak model names this
#: slot inconsistently; accepting its synonyms costs nothing and saves a wasted step.
_PATH_KEYS = ("path", "file", "relpath", "filename", "dir", "directory")


def _detail(details: Dict[str, Any], *keys: str) -> Optional[str]:
    """First non-empty string among ``keys`` in ``details`` (else ``None``)."""
    for key in keys:
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
            return str(value).strip()
    return None


class SandboxLeafAction(LeafAction):
    """Base for the sandbox actions: resolve the run's sandbox, or refuse cleanly.

    Subclasses implement :meth:`run` and never touch the filesystem themselves.
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
        raise NotImplementedError()

    def _path(self, details: Dict[str, Any], default: Optional[str] = None) -> Optional[str]:
        return _detail(details, *_PATH_KEYS) or default

    def _confined(self, sandbox: Any, details: Dict[str, Any], default: Optional[str] = None):
        """``(relpath, resolved)`` for this node's path slot, or ``(relpath, None)`` on escape."""
        relpath = self._path(details, default)
        if relpath is None:
            return None, None
        return relpath, sandbox.confine(relpath)

    def _confined_file(self, sandbox: Any, details: Dict[str, Any]):
        """``(workdir-relative path, None)`` for an existing confined FILE, else ``(None, error)``.

        Distinguishes the two refusals a weak model must react to differently: a path that
        escapes the workdir (rephrase it) versus one that simply is not there (make it first).
        """
        relpath, target = self._confined(sandbox, details)
        if relpath is None:
            return None, fail(self.name, "missing 'path' detail (str)")
        if target is None:
            return None, fail(self.name, f"path escapes the workdir: {relpath}")
        if not target.is_file():
            return None, fail(self.name, f"no such file in the workdir: {relpath}")
        return sandbox.rel(target), None

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

    async def _readonly(self, sandbox: Any, argv: List[str], **extra: Any) -> Dict[str, Any]:
        """Run one allow-listed read-only command through the sandbox and map its result."""
        return self._from_sandbox(await sandbox.run_readonly(self.name, argv), **extra)


# --- file actions --------------------------------------------------------------------------


class ReadFileAction(SandboxLeafAction):
    """Read a text file from the sandbox workdir.

    Confinement: the path is resolved under the workdir by ``SandboxConnector.confine``; anything
    that escapes it (``..``, an absolute path, a symlink out) is refused, not read.

    Reads from node details:
      - `path` (str): workdir-relative file path (required).

    Returns `{path, bytes, output}` (content truncated for the observation).
    """

    name = "read_file"
    args_hint = "details={path}"

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        relpath = self._path(details)
        if relpath is None:
            return fail(self.name, "missing 'path' detail (str)")
        return self._from_sandbox(await sandbox.read_file(relpath))


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

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        relpath = self._path(details)
        if relpath is None:
            return fail(self.name, "missing 'path' detail (str)")
        content = details.get("content")
        if content is None:
            content = details.get("text")
        if content is None:
            return fail(self.name, "missing 'content' detail (str)")
        return self._from_sandbox(await sandbox.write_file(relpath, str(content)))


class ListDirAction(SandboxLeafAction):
    """List one directory inside the sandbox workdir.

    Confinement: the path is resolved under the workdir by ``SandboxConnector.confine``; a path
    that escapes it is refused, so no directory outside the workdir can be enumerated.

    Reads from node details:
      - `path` (str): workdir-relative directory, default the workdir root.

    Returns `{path, entries, output}`.
    """

    name = "list_dir"
    args_hint = "details={path?}"

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        return self._from_sandbox(await sandbox.list_dir(self._path(details, "") or ""))


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

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        rel, refusal = self._confined_file(sandbox, details)
        if refusal is not None:
            return refusal
        pattern = _detail(details, "pattern", "match")
        argv = ["grep", "-c", "--", pattern, rel] if pattern else ["wc", "-l", rel]
        return await self._readonly(sandbox, argv, path=rel)


class WordCountAction(SandboxLeafAction):
    """Count words in a workdir file (``wc -w``).

    Reads from node details:
      - `path` (str): workdir-relative file (required).

    Returns `{path, stdout, exit_code, output}`.
    """

    name = "word_count"
    args_hint = "details={path}"

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        rel, refusal = self._confined_file(sandbox, details)
        if refusal is not None:
            return refusal
        return await self._readonly(sandbox, ["wc", "-w", rel], path=rel)


class HeadFileAction(SandboxLeafAction):
    """Show the first N lines of a workdir file (``head -n``).

    Reads from node details:
      - `path` (str): workdir-relative file (required).
      - `lines` (int): how many lines, default 10, capped at 200.

    Returns `{path, stdout, exit_code, output}`.
    """

    name = "head_file"
    args_hint = "details={path, lines?}"

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        rel, refusal = self._confined_file(sandbox, details)
        if refusal is not None:
            return refusal
        try:
            lines = int(_detail(details, "lines", "n", "count") or 10)
        except (TypeError, ValueError):
            lines = 10
        lines = max(1, min(200, lines))
        return await self._readonly(sandbox, ["head", "-n", str(lines), rel], path=rel)


class DiskUsageAction(SandboxLeafAction):
    """Report the size of a workdir directory or file (``du -sh``).

    Reads from node details:
      - `path` (str): workdir-relative directory, default the workdir root.

    Returns `{path, stdout, exit_code, output}`.
    """

    name = "disk_usage"
    args_hint = "details={path?}"

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        relpath, target = self._confined(sandbox, details, ".")
        if target is None:
            return fail(self.name, f"path escapes the workdir: {relpath}")
        if not target.exists():
            return fail(self.name, f"no such path under the workdir: {relpath}")
        rel = sandbox.rel(target)
        return await self._readonly(sandbox, ["du", "-sh", rel], path=rel)


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

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        pattern = _detail(details, "name", "glob", "pattern", "filename")
        if pattern is None:
            return fail(self.name, "missing 'name' detail (a filename glob, e.g. '*.py')")
        return await self._readonly(sandbox, ["find", ".", "-name", pattern, "-type", "f"])


class SandboxToolPack:
    """The workdir file + read-only shell actions, as one installable bundle.

    Deliberately NOT part of ``ExtraActionPack`` and never installed by default: those are
    read-only public-API lookups, these touch a filesystem. Install it only for a run that has a
    ``SandboxConnector`` on its ``AgentIO``::

        engine.install_action_pack(SandboxToolPack(settings=engine.settings))

    Use ``install_action_pack(..., allow=False)`` to register the classes without permitting them
    (dispatch stays closed until the names are added to ``allowed_actions``).
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
