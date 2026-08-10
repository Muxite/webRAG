"""Sandbox connector — the CODE-domain counterpart of ``connector_http``/``connector_search``.

The web-research arms reach the world through ``search``/``visit``; the coding arm
(``testing/execution_compiled_code.py``, execution variant ``graph_compiled_code``) reaches it
through a writable workdir: read/write/patch files, list a directory, run python, run pytest —
plus ``search_web``, which is NOT a new backend but a straight delegation to the existing
``ConnectorSearch`` (pointed at the sandboxed SearXNG by
``connector_search_searxng.ConnectorSearchXNG``).

Same shape as the other connectors: a ``ConnectorBase`` subclass wrapping one I/O surface, every
action awaitable so the leaf loop can dispatch on it uniformly, and every action returning a small
dict (never raising) so a weak model's bad argument becomes an OBSERVATION it can recover from
rather than an exception that sinks the leaf.

Two bounds, both DEFENCE IN DEPTH rather than the security boundary (the container is that:
read-only rootfs, dropped caps, tmpfs ``/work``, an outer wall-clock kill):
  * :meth:`SandboxConnector.confine` — every relpath is resolved under the workdir root and
    anything that escapes it (``..`` traversal, an absolute path, a symlink pointing out) is
    refused. This guards OUR file abstractions only; it does nothing about a ``run_python`` body
    that calls ``open()`` itself, which is why the agent image also strips every canonical-test
    package (see ``codebench/agents/badmodel/Dockerfile``).
  * :class:`agent.app.idea_policies.config.SandboxActionConfig` — per-action subprocess timeouts
    and the per-leaf write budget (``max_file_bytes`` / ``max_files_per_leaf``).
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from shared.connector_config import ConnectorConfig

from agent.app.connector_base import ConnectorBase
from agent.app.idea_policies.config import SandboxActionConfig

# Cap on the stdout/stderr text an action feeds back into the leaf loop's observation. A failing
# pytest run can emit tens of kB; a weak model's context (and its attention) cannot absorb that,
# and the useful signal — the summary line and the first traceback — is at the ends, so we keep
# the head and the tail and mark the elision explicitly.
OUTPUT_CHAR_CAP = 4000

# Cap on the entries one ``list_dir`` reports back. Same bounded-observation reasoning as
# OUTPUT_CHAR_CAP: a node_modules-sized directory is thousands of names, which buries the useful
# signal and costs a weak model its whole context — and unlike stdout there is no "both ends" to
# preserve, so the first N (sorted) plus an explicit count of what was dropped is the honest view.
DIR_ENTRY_CAP = 200

# A ``run_python`` argument is treated as a FILE when it looks like a path to an existing file
# under the workdir; anything else is treated as an inline program.
_PY_SUFFIXES = (".py",)

#: Executables :meth:`SandboxConnector.run_readonly` will run. Ported from
#: ``badmodel-lab/localagent/tools/shell.py``'s allow-list: every one of them READS — nothing here
#: writes, deletes, or reaches the network, so no argv built from it can mutate the workdir. The
#: allow-list is what makes the surface safe to expose as leaf actions; adding a writing command to
#: it would silently widen every caller's blast radius, so don't.
READONLY_COMMANDS = frozenset({"wc", "grep", "du", "find", "head"})


def _encoding_fault(text: str) -> Optional[str]:
    """Why ``text`` cannot be encoded to UTF-8, or ``None`` when it can.

    Every string that reaches this connector came out of a model-authored JSON object, and
    ``json.loads('{"content": "\\ud800"}')` SUCCEEDS — the parser is happy to hand back a lone
    surrogate that no UTF-8 encoder will take. Writing it (``Path.write_text``) or passing it to
    ``exec`` raises ``UnicodeEncodeError``, which is a ``UnicodeError``/``ValueError`` and so slips
    past an ``except OSError``. On the engine's non-parallel path that took down the whole run
    rather than one leaf. Checking first turns it into an ordinary observation, and — matching this
    codebase's "degrade honestly" convention — refuses rather than silently rewriting the content
    with replacement characters the model never asked for.
    """
    try:
        text.encode("utf-8")
    except UnicodeError:
        return "contains characters that cannot be encoded (a lone surrogate such as '\\ud800')"
    return None


def _argv_fault(token: str) -> Optional[str]:
    """Why ``token`` cannot be passed to ``create_subprocess_exec``, or ``None`` when it can.

    Same class of bug as :func:`_encoding_fault` one layer down, and reachable from every
    subprocess action — inline ``run_python`` code, a ``grep`` pattern, a ``find`` glob. An
    embedded NUL is the second case: ``os.fsencode`` raises a bare ``ValueError("embedded null
    byte")``, which no caller here caught either.
    """
    if "\x00" in token:
        return "contains a null byte"
    return _encoding_fault(token)


def _truncate(text: str, cap: int = OUTPUT_CHAR_CAP) -> str:
    """Clip ``text`` to ``cap`` chars keeping both ends (head + tail), marking the elision.

    Truncating only the head loses pytest's summary line; only the tail loses the first failing
    assertion. Both matter to the model, so keep both.
    """
    text = text or ""
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = max(0, cap - head)
    dropped = len(text) - head - tail
    return f"{text[:head]}\n...[{dropped} chars omitted]...\n{text[len(text) - tail:]}" if tail else text[:head]


class SandboxConnector(ConnectorBase):
    """Bounded, workdir-confined filesystem + subprocess access for the coding arm.

    :param workdir: The sandbox root. Every relpath resolves under it; created if absent.
    :param connector_config: Shared connector configuration (telemetry/logging plumbing only —
        may be ``None`` in tests, exactly like the other connectors it never reads it eagerly).
    :param limits: Typed action limits; defaults to :class:`SandboxActionConfig` defaults.
    :param connector_search: Existing search connector reused verbatim by :meth:`search_web`;
        ``None`` disables web search (the action then returns a clean "not available" observation).
    """

    def __init__(
        self,
        workdir: Union[str, Path],
        connector_config: Optional[ConnectorConfig] = None,
        limits: Optional[SandboxActionConfig] = None,
        connector_search: Optional[Any] = None,
    ) -> None:
        super().__init__(connector_config)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.limits = limits or SandboxActionConfig()
        self.connector_search = connector_search
        # Per-leaf write budget (max_files_per_leaf). Distinct relpaths, so rewriting the same
        # file while iterating on a failing test never exhausts it — the cap is on FILE COUNT.
        self._leaf_files_written: set = set()
        self.actions_count = 0

    # --- confinement -----------------------------------------------------------------------
    def confine(self, relpath: str) -> Optional[Path]:
        """Resolve ``relpath`` under the workdir, or ``None`` if it escapes.

        Mirrors ``badmodel-lab/localagent/tools/base.py``'s ``confine()``: resolve BOTH sides and
        require the target to be the root itself or a descendant of it. ``Path.resolve()`` also
        collapses ``..`` and follows symlinks, so a symlink planted inside the workdir that points
        outside resolves to its real (outside) location and is refused. Returns ``None`` instead
        of raising — the caller turns that into an observation, never an exception.
        """
        try:
            base = self.workdir.resolve()
            target = (base / str(relpath or "")).resolve()
            if target == base or base in target.parents:
                return target
            return None
        except Exception:  # noqa: BLE001 — a malformed path is a refusal, not a crash
            return None

    def rel(self, path: Path) -> str:
        """``path`` as a workdir-relative string (best effort; absolute if somehow outside)."""
        try:
            return str(path.resolve().relative_to(self.workdir.resolve()))
        except Exception:  # noqa: BLE001
            return str(path)

    def reset_leaf_budget(self) -> None:
        """Start a new leaf's write budget (``max_files_per_leaf`` counts per leaf, not per run)."""
        self._leaf_files_written = set()

    @property
    def files_written_this_leaf(self) -> List[str]:
        """Relpaths this leaf has written so far, sorted — HARNESS-OBSERVED, not model-reported.

        The leaf loop folds this into what a dependent leaf sees for ``{dep_id}``, so a downstream
        step learns the files that actually exist even when the upstream model's own closing
        summary was vague, wrong, or never written at all. Reset by :meth:`reset_leaf_budget`, so
        read it BEFORE starting the next leaf.
        """
        return sorted(self._leaf_files_written)

    def _denied(self, action: str, relpath: str) -> Dict[str, Any]:
        self._record_io(direction="out", operation=f"sandbox_{action}",
                        payload={"path": str(relpath)}, error="path escapes workdir")
        return {"ok": False, "action": action, "path": str(relpath),
                "error": f"path {relpath!r} escapes the workdir; use a path inside it"}

    def _done(self, action: str, started_at: float, result: Dict[str, Any]) -> Dict[str, Any]:
        self.actions_count += 1
        self._record_timing(name=f"sandbox_{action}", started_at=started_at, success=bool(result.get("ok")),
                            payload={"path": result.get("path"), "action": action},
                            error=None if result.get("ok") else str(result.get("error") or ""))
        self._record_io(direction="out", operation=f"sandbox_{action}",
                        payload={"path": result.get("path"), "ok": result.get("ok")})
        return result

    # --- file actions ----------------------------------------------------------------------
    async def write_file(self, relpath: str, content: str) -> Dict[str, Any]:
        """Create/overwrite a workdir file. Enforces ``max_file_bytes`` and ``max_files_per_leaf``."""
        started = time.perf_counter()
        target = self.confine(relpath)
        if target is None:
            return self._denied("write_file", relpath)
        text = "" if content is None else str(content)
        fault = _encoding_fault(text)
        if fault is not None:
            return self._done("write_file", started, {
                "ok": False, "action": "write_file", "path": str(relpath),
                "error": f"content {fault}; re-send it as plain text"})
        size = len(text.encode("utf-8", "replace"))
        if size > int(self.limits.max_file_bytes):
            return self._done("write_file", started, {
                "ok": False, "action": "write_file", "path": str(relpath), "bytes": size,
                "error": f"file is {size} bytes, over the {self.limits.max_file_bytes}-byte limit",
            })
        rel = self.rel(target)
        if (rel not in self._leaf_files_written
                and len(self._leaf_files_written) >= int(self.limits.max_files_per_leaf)):
            return self._done("write_file", started, {
                "ok": False, "action": "write_file", "path": rel,
                "error": (f"per-leaf file budget exhausted "
                          f"({self.limits.max_files_per_leaf} files); edit an existing file instead"),
            })
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except (OSError, UnicodeError) as exc:  # UnicodeError: see _encoding_fault (backstop)
            return self._done("write_file", started, {
                "ok": False, "action": "write_file", "path": rel, "error": f"write failed: {exc}"})
        self._leaf_files_written.add(rel)
        return self._done("write_file", started, {
            "ok": True, "action": "write_file", "path": rel, "bytes": size,
            "output": f"wrote {rel} ({size} bytes, {text.count(chr(10)) + 1} lines)",
        })

    async def read_file(self, relpath: str, max_chars: int = OUTPUT_CHAR_CAP) -> Dict[str, Any]:
        """Read a workdir file back (truncated to ``max_chars`` for the observation).

        Refuses BEFORE reading anything when the file is over ``max_file_bytes``, the same
        pre-flight ``write_file``/``patch_file`` already do. ``write_file``'s own budget does not
        cover this: a file can enter the workdir through the starter fixture, a ``run_python`` body
        calling ``open()``, or the task image itself, and materializing all of it in memory just to
        throw away everything past ``max_chars`` is cost with no reader.
        """
        started = time.perf_counter()
        target = self.confine(relpath)
        if target is None:
            return self._denied("read_file", relpath)
        rel = self.rel(target)
        if not target.is_file():
            return self._done("read_file", started, {
                "ok": False, "action": "read_file", "path": rel, "error": f"no such file: {rel}"})
        try:
            size = target.stat().st_size
        except OSError as exc:
            return self._done("read_file", started, {
                "ok": False, "action": "read_file", "path": rel, "error": f"read failed: {exc}"})
        if size > int(self.limits.max_file_bytes):
            return self._done("read_file", started, {
                "ok": False, "action": "read_file", "path": rel, "bytes": size,
                "error": f"file is {size} bytes, over the {self.limits.max_file_bytes}-byte limit; "
                         f"read part of it instead (head_file, or grep it with count_lines)"})
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return self._done("read_file", started, {
                "ok": False, "action": "read_file", "path": rel, "error": f"read failed: {exc}"})
        return self._done("read_file", started, {
            "ok": True, "action": "read_file", "path": rel, "bytes": len(text),
            "output": _truncate(text, max_chars)})

    async def list_dir(self, relpath: str = "") -> Dict[str, Any]:
        """List one workdir directory (``""`` = the root). Directories are marked with a ``/``.

        Capped at :data:`DIR_ENTRY_CAP` entries; the overflow is reported as a count rather than
        dropped silently, so the model knows it is looking at a partial listing.
        """
        started = time.perf_counter()
        target = self.confine(relpath)
        if target is None:
            return self._denied("list_dir", relpath)
        rel = self.rel(target) if target != self.workdir.resolve() else "."
        if not target.is_dir():
            return self._done("list_dir", started, {
                "ok": False, "action": "list_dir", "path": rel, "error": f"not a directory: {rel}"})
        found: List[str] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            if child.name == "__pycache__":
                continue
            found.append(f"{child.name}/" if child.is_dir() else child.name)
        entries = found[:DIR_ENTRY_CAP]
        dropped = len(found) - len(entries)
        result = {
            "ok": True, "action": "list_dir", "path": rel, "entries": entries,
            "total_entries": len(found), "truncated": bool(dropped),
            "output": "\n".join(entries) if entries else "(empty)",
        }
        if dropped:
            result["output"] += f"\n...and {dropped} more"
        return self._done("list_dir", started, result)

    async def patch_file(self, relpath: str, old: str, new: str) -> Dict[str, Any]:
        """Replace the FIRST exact occurrence of ``old`` with ``new`` in a workdir file.

        Deliberately a find/replace, not a diff-apply engine: a weak model cannot author a valid
        unified diff, but it can quote the line it wants changed. A miss (``old`` absent, or
        ambiguous because it occurs more than once) is reported as an actionable observation
        instead of being silently applied somewhere the model did not mean.
        """
        started = time.perf_counter()
        target = self.confine(relpath)
        if target is None:
            return self._denied("patch_file", relpath)
        rel = self.rel(target)
        if not target.is_file():
            return self._done("patch_file", started, {
                "ok": False, "action": "patch_file", "path": rel, "error": f"no such file: {rel}"})
        old_text = "" if old is None else str(old)
        new_text = "" if new is None else str(new)
        if not old_text:
            return self._done("patch_file", started, {
                "ok": False, "action": "patch_file", "path": rel,
                "error": "'old' must be a non-empty snippet to replace"})
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return self._done("patch_file", started, {
                "ok": False, "action": "patch_file", "path": rel, "error": f"read failed: {exc}"})
        fault = _encoding_fault(old_text) or _encoding_fault(new_text)
        if fault is not None:
            return self._done("patch_file", started, {
                "ok": False, "action": "patch_file", "path": rel,
                "error": f"content {fault}; re-send the snippet as plain text"})
        occurrences = content.count(old_text)
        if occurrences == 0:
            return self._done("patch_file", started, {
                "ok": False, "action": "patch_file", "path": rel, "occurrences": 0,
                "error": "'old' snippet not found in the file; read_file it and quote the exact text"})
        patched = content.replace(old_text, new_text, 1)
        size = len(patched.encode("utf-8", "replace"))
        if size > int(self.limits.max_file_bytes):
            return self._done("patch_file", started, {
                "ok": False, "action": "patch_file", "path": rel, "bytes": size,
                "error": f"patched file would be {size} bytes, over the "
                         f"{self.limits.max_file_bytes}-byte limit"})
        try:
            target.write_text(patched, encoding="utf-8")
        except (OSError, UnicodeError) as exc:  # UnicodeError: see _encoding_fault (backstop)
            return self._done("patch_file", started, {
                "ok": False, "action": "patch_file", "path": rel, "error": f"write failed: {exc}"})
        note = "" if occurrences == 1 else f" (matched {occurrences}x, replaced the first)"
        return self._done("patch_file", started, {
            "ok": True, "action": "patch_file", "path": rel, "occurrences": occurrences,
            "output": f"patched {rel}{note}"})

    def exists(self, relpath: str) -> bool:
        """True iff ``relpath`` resolves inside the workdir AND exists (used by ``_compose_submit``)."""
        target = self.confine(relpath)
        return bool(target is not None and target.exists())

    def file_size(self, relpath: str) -> Optional[int]:
        """Byte size of a confined workdir file, or ``None`` when it is missing/unreadable."""
        target = self.confine(relpath)
        if target is None or not target.is_file():
            return None
        try:
            return target.stat().st_size
        except OSError:
            return None

    # --- subprocess actions ----------------------------------------------------------------
    def _subprocess_env(self) -> Dict[str, str]:
        """Child env: inherited, minus bytecode caching (``__pycache__`` would pollute the
        extracted submission) and with the workdir first on ``PYTHONPATH`` so a module written
        at the root imports from anywhere under it."""
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        root = str(self.workdir.resolve())
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
        return env

    @staticmethod
    def _kill_process_group(proc: Any) -> None:
        """SIGKILL a timed-out child's WHOLE process group, not just the direct child.

        The child is started with ``start_new_session=True``, so it leads its own process group and
        everything it spawns inherits that group id — including a process that double-forks and
        lets its parent exit to orphan itself, which a bare ``proc.kill()`` leaves running (and
        untracked) for the rest of the container's life, long past the timeout we just declared.
        Killing the group reaps those leftovers too. A grandchild that calls ``setsid()`` itself
        still escapes the group; the container's outer wall-clock kill and its pids cgroup are the
        backstop for that, exactly as for every other bound in this module.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (OSError, AttributeError):  # already gone, or no POSIX process groups
            pass
        try:
            proc.kill()
        except (OSError, ValueError):  # noqa: BLE001 — killing a dead child is not an error
            pass

    async def _run(self, action: str, argv: List[str], timeout_s: float,
                   path: Optional[str] = None) -> Dict[str, Any]:
        """Run ``argv`` with CWD = the workdir, capturing truncated stdout/stderr.

        A timeout kills the child's whole process group (see :meth:`_kill_process_group`) and is
        reported as a normal (failed) result — an infinite loop in generated code is an ordinary
        outcome here, not an infrastructure error.

        Malformed argv is the other "ordinary outcome" that used to escape as an exception. Every
        token here is model-authored (inline ``run_python`` code, a ``grep`` pattern, a ``find``
        glob), and two shapes of it kill the launch rather than the program: a lone surrogate
        (``UnicodeEncodeError``) and an embedded NUL (a bare ``ValueError``). Both are checked
        up-front — precisely, so a genuine ``ValueError`` from a real bug still surfaces instead of
        being reported to a model as its own mistake.
        """
        started = time.perf_counter()
        self._record_io(direction="in", operation=f"sandbox_{action}",
                        payload={"argv": argv[:3], "timeout_s": timeout_s, "path": path})
        for token in argv:
            fault = _argv_fault(str(token))
            if fault is not None:
                return self._done(action, started, {
                    "ok": False, "action": action, "path": path,
                    "error": f"could not start: an argument {fault}; re-send it as plain text"})
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=str(self.workdir), env=self._subprocess_env(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # own process group, so a timeout can kill the whole tree
            )
        except (OSError, UnicodeError) as exc:  # UnicodeError: see _argv_fault (backstop)
            return self._done(action, started, {
                "ok": False, "action": action, "path": path, "error": f"could not start: {exc}"})
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=max(1.0, float(timeout_s)))
        except asyncio.TimeoutError:
            self._kill_process_group(proc)
            await asyncio.gather(proc.wait(), return_exceptions=True)
            return self._done(action, started, {
                "ok": False, "action": action, "path": path, "timed_out": True, "exit_code": None,
                "error": f"timed out after {timeout_s}s (killed)"})
        stdout = _truncate((out_b or b"").decode("utf-8", "replace"))
        stderr = _truncate((err_b or b"").decode("utf-8", "replace"))
        code = proc.returncode
        return self._done(action, started, {
            "ok": code == 0, "action": action, "path": path, "exit_code": code,
            "stdout": stdout, "stderr": stderr,
            "output": _truncate(f"exit_code={code}\nSTDOUT:\n{stdout or '(empty)'}\n"
                                f"STDERR:\n{stderr or '(empty)'}"),
        })

    async def run_python(self, code_or_file: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Run python in the workdir: a ``.py`` path that exists there runs as a FILE, anything
        else runs as an inline program (``python -c``)."""
        text = "" if code_or_file is None else str(code_or_file)
        timeout = self.limits.run_python_timeout_seconds if timeout_s is None else timeout_s
        if not text.strip():
            return {"ok": False, "action": "run_python", "error": "nothing to run: 'code' was empty"}
        candidate = text.strip()
        if candidate.endswith(_PY_SUFFIXES) and "\n" not in candidate:
            target = self.confine(candidate)
            if target is None:
                return self._denied("run_python", candidate)
            if not target.is_file():
                return {"ok": False, "action": "run_python", "path": candidate,
                        "error": f"no such file: {candidate}"}
            return await self._run("run_python", [sys.executable, self.rel(target)], timeout,
                                   path=self.rel(target))
        return await self._run("run_python", [sys.executable, "-c", text], timeout)

    def _argv_escapes(self, argv: List[str]) -> Optional[str]:
        """The first ``argv`` token that looks like a path OUT of the workdir, else ``None``.

        Belt-and-suspenders behind the callers, which confine their own path slots before building
        argv: refuses an absolute token that does not resolve under the workdir, and any token
        carrying a ``..`` path segment (which ``cwd``-relative execution would otherwise honour).
        A pattern-shaped argument that genuinely needs ``..`` or a leading ``/`` is refused too —
        fail-closed is the right trade for a surface whose whole point is that it cannot reach
        outside the workdir.
        """
        for token in argv[1:]:
            text = str(token)
            if any(segment == ".." for segment in text.split("/")):
                return text
            if text.startswith("/") and self.confine(text) is None:
                return text
        return None

    async def run_readonly(self, op: str, argv: List[str],
                           timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Run ONE allow-listed READ-ONLY command (``wc``/``grep``/``du``/``find``/``head``).

        The model never authors ``argv``: a caller (see
        ``idea_policies/extra_actions/sandbox_tools.py``) picks a narrow intent and fills typed
        slots, and code assembles the exact argument vector — no ``shell=True``, no pipes, no
        metacharacter expansion. Two refusals guard it: :data:`READONLY_COMMANDS` (so a mutating
        command cannot be constructed at all) and :meth:`_argv_escapes` (so no argument reaches
        outside the workdir). Runs with ``cwd`` = the workdir under
        ``sandbox_shell_timeout_seconds``, like every other subprocess action here.

        ``grep``'s exit code 1 means "no match", which is a RESULT and not a failure, so it comes
        back ``ok`` with ``matched: False``.
        """
        argv = [str(part) for part in (argv or [])]
        if not argv:
            return {"ok": False, "action": op, "error": "no command to run"}
        if argv[0] not in READONLY_COMMANDS:
            return {"ok": False, "action": op,
                    "error": f"command {argv[0]!r} is not read-only allow-listed "
                             f"({sorted(READONLY_COMMANDS)})"}
        escaping = self._argv_escapes(argv)
        if escaping is not None:
            return self._denied(op, escaping)
        timeout = self.limits.shell_timeout_seconds if timeout_s is None else timeout_s
        result = await self._run(op, argv, timeout)
        if argv[0] == "grep":
            result["matched"] = result.get("exit_code") == 0
            if result.get("exit_code") == 1:
                result["ok"] = True
        return result

    async def run_pytest(self, target_relpath: str = "", timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Run pytest inside the workdir on ``target_relpath`` (``""`` = the whole workdir)."""
        timeout = self.limits.run_pytest_timeout_seconds if timeout_s is None else timeout_s
        rel = str(target_relpath or "").strip()
        target = self.confine(rel)
        if target is None:
            return self._denied("run_pytest", rel)
        if rel and not target.exists():
            return {"ok": False, "action": "run_pytest", "path": rel,
                    "error": f"no such test path: {rel}"}
        arg = self.rel(target) if rel else "."
        return await self._run("run_pytest", [sys.executable, "-m", "pytest", arg, "-q"],
                               timeout, path=arg)

    # --- web action ------------------------------------------------------------------------
    async def search_web(self, query: str, count: int = 5) -> Dict[str, Any]:
        """Web search, delegated verbatim to the injected ``ConnectorSearch``.

        No new backend: whichever search connector the caller wired (in the codebench container
        that is ``ConnectorSearchXNG``, pointed at the sandbox-reachable SearXNG) does the work,
        keeping the coding arm on exactly the same search surface as the QA arms.
        """
        started = time.perf_counter()
        text = str(query or "").strip()
        if not text:
            return {"ok": False, "action": "search_web", "error": "empty query"}
        if self.connector_search is None:
            return {"ok": False, "action": "search_web",
                    "error": "web search is not available in this sandbox"}
        try:
            results = await self.connector_search.query_search(text, count=count) or []
        except Exception as exc:  # noqa: BLE001 — a search outage must not sink the leaf
            self.logger.warning(f"sandbox search_web failed: {exc}")
            return self._done("search_web", started, {
                "ok": False, "action": "search_web", "error": f"search failed: {exc}"})
        lines = [
            f"{i}. {str(r.get('title', ''))[:120]}\n   {r.get('url', '')}\n   "
            f"{str(r.get('description', ''))[:200]}"
            for i, r in enumerate(results[:count], 1)
        ]
        return self._done("search_web", started, {
            "ok": True, "action": "search_web", "results": results[:count],
            "output": "\n".join(lines) if lines else "(no results)"})
