"""A one-off Python calculator leaf action (opt-in, never on by default).

WHY this exists. The composers in ``testing/execution_compiled.py``
(``_compose_argmax``/``_compose_ratio_argmax``/``_compose_count_threshold``/``_compose_subset_sum``/
``_compose_and_filter``) exist because a weak model that has already RETRIEVED the right facts still
picks the wrong answer out of them: it writes the correct comparison in its own free text and then
concludes something else. Each composer patches one task SHAPE, in hand-authored code, after the
fact. This action closes the same loop from the other side — it lets the model reach for a
calculator mid-reasoning ("let me just compute this") instead of eyeballing six heights, so a shape
nobody wrote a composer for is still computable. The composers stay: they are the deterministic
safety net that does not depend on the model choosing to use a tool. This is the optional
alternative path, not a replacement.

Deliberately its OWN pack, separate from :class:`SandboxToolPack`. A caller who wants "just a
calculator" gets exactly that — no file writes, no shell — and a caller who wants both installs
both::

    engine.install_action_pack(CalculatorToolPack(settings=engine.settings))

Same three-way gate and the same no-ambient-authority posture as the file/shell pack: the pack must
be installed, ``run_python`` must be in ``allowed_actions``, AND the run's ``AgentIO`` must carry a
``connector_sandbox``. A plain web-research run satisfies none of them, and a run missing only the
last gets a clean "no sandbox in this run" OBSERVATION rather than an exception. Nothing here
re-implements execution, timeouts, confinement or output truncation: every one of those comes from
:meth:`agent.app.connector_sandbox.SandboxConnector.run_python`, which codebench's security review
already went over. The base class is imported from ``sandbox_tools`` for the same reason — one
capability gate, not two copies of it that can drift apart.

Bounds: ``sandbox_calculator_timeout_seconds`` (10s, tighter than the coding arm's 15s ``run_python``
budget — see :class:`agent.app.idea_policies.config.SandboxActionConfig`) and the connector's own
``OUTPUT_CHAR_CAP`` truncation of stdout/stderr.
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any, Dict, List, Optional, Type

from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.base import fail
# The sandbox gate ("resolve the run's sandbox or refuse cleanly") is shared, not re-implemented:
# a second copy is how the two would drift into disagreeing about what "no sandbox" means.
from agent.app.idea_policies.extra_actions.sandbox_tools import SandboxLeafAction

#: Detail keys accepted for "the code to run", most specific first. Mirrors the alias list the
#: codebench leaf loop already accepts (``execution_compiled_code._CODE_KEYS``) plus the synonyms a
#: weak model reaches for; naming the slot differently should not cost a step.
_CODE_KEYS = ("code", "script", "source", "snippet", "python", "program",
              "expression", "expr", "content", "path", "file")

#: A successful snippet with empty stdout is the single most common weak-model miss here: it
#: computes the answer and never prints it. Say so in the observation instead of handing back a
#: blank result the model has to diagnose.
_NO_OUTPUT_HINT = "(the snippet ran but printed nothing — print(...) the value you want to see)"


def _strip_fence(text: str) -> str:
    """Drop a surrounding markdown code fence (```python ... ```), if there is one.

    Models trained to emit fenced code do it inside JSON string slots too; the fence is never
    valid Python, so accepting it costs one string operation and saves a wasted step.
    """
    if not text.lstrip().startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _code(details: Dict[str, Any]) -> Optional[str]:
    """The snippet to run from this node's details, cleaned up, or ``None`` if no slot carries one.

    Cleanup order matters: de-fence, then ``dedent``, and only THEN strip. Stripping first would
    remove the leading indentation of line ONE and leave ``dedent`` nothing in common to remove, so
    a uniformly indented block would still raise ``IndentationError`` on its second line. A list of
    lines — another shape models emit — is joined rather than rejected.
    """
    for key in _CODE_KEYS:
        value = details.get(key)
        if value is None or isinstance(value, dict):
            continue
        if isinstance(value, (list, tuple)):
            value = "\n".join(str(line) for line in value)
        text = textwrap.dedent(_strip_fence(str(value).strip("\r\n"))).strip()
        if text:
            return text
    return None


def _as_program(code: str) -> str:
    """``print(...)`` a BARE EXPRESSION so ``max(heights)`` works like a calculator.

    The check is ``ast.parse(..., mode="eval")``, not a heuristic: if the whole snippet is a valid
    Python expression it can only be meant as "tell me this value", and running it unwrapped would
    print nothing at all. Anything that is not a valid expression (a statement, an assignment, a
    multi-line program) runs verbatim, so this can never rewrite real code into a syntax error —
    and an expression that ALREADY prints is left alone, or ``print(2 + 40)`` would come back as
    "42\\nNone".
    """
    try:
        tree = ast.parse(code, mode="eval")
    except (SyntaxError, ValueError):  # not an expression (or has embedded NULs) — run as written
        return code
    call = tree.body
    if isinstance(call, ast.Call):
        func = call.func
        printer = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if printer in ("print", "pprint"):
            return code
    return f"print({code})"


class RunPythonAction(SandboxLeafAction):
    """Run a short Python snippet (e.g. to compare/compute over facts already gathered) and see its printed output.

    The snippet runs through ``SandboxConnector.run_python``: workdir CWD, its own process group,
    killed at ``sandbox_calculator_timeout_seconds``, stdout/stderr truncated for the observation.
    A crash (bad syntax, ``NameError``, division by zero) is a FAILED OBSERVATION carrying the
    traceback, never an exception — the model reads the error and fixes its snippet.

    Reads from node details:
      - `code` (str): the snippet (required). A bare expression is printed for you; a single-line
        `*.py` path that exists in the workdir is run as that file instead.

    Returns `{exit_code, stdout, stderr, output}`.
    """

    name = "run_python"
    args_hint = "details={code}"

    async def run(self, sandbox: Any, details: Dict[str, Any]) -> Dict[str, Any]:
        code = _code(details)
        if code is None:
            return fail(self.name, "missing 'code' detail (a short Python snippet that prints its result)")
        # File mode is the connector's own rule (a single-line existing ``*.py`` path runs as a
        # file); leave such a slot untouched, since ``pkg.py`` also parses as an expression and
        # auto-printing it would turn a file run into a NameError.
        program = code if (code.endswith(".py") and "\n" not in code) else _as_program(code)
        result = await sandbox.run_python(program, timeout_s=self._cfg.sandbox.calculator_timeout_seconds)
        mapped = self._from_sandbox(result)
        if mapped.get("success") and not str(mapped.get("stdout") or "").strip():
            mapped["output"] = f"{str(mapped.get('output') or '').rstrip()}\n{_NO_OUTPUT_HINT}".strip()
        return mapped


class CalculatorToolPack:
    """The one-off Python calculator, as its own installable bundle.

    Separate from :class:`SandboxToolPack` on purpose: "let the model compute" and "let the model
    read and write files" are different asks, and a caller should be able to grant the first
    without the second. Like that pack it is never installed by default and is not part of the
    curated ``ExtraActionPack`` (those are read-only public-API lookups; this runs a subprocess)::

        engine.install_action_pack(CalculatorToolPack(settings=engine.settings))

    Use ``install_action_pack(..., allow=False)`` to register the class without permitting it
    (dispatch stays closed until the name is added to ``allowed_actions``).
    """

    name = "calculator_tools"

    ACTION_CLASSES: List[Type[LeafAction]] = [RunPythonAction]

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self.settings = dict(settings or {})

    def build_instances(self) -> Dict[str, LeafAction]:
        """Instantiate each action class with the pack's shared settings."""
        return {cls.name: cls(settings=self.settings) for cls in self.ACTION_CLASSES}

    def names(self) -> List[str]:
        return [cls.name for cls in self.ACTION_CLASSES]
