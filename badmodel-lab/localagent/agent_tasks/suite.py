"""The v1 mix suite — small graded agentic tasks across file / shell / web-read /
memory, each issued as a FREE-FORM request the router must interpret, each scored by
DETERMINISTIC validators (no LLM judge). A task also declares which action groups are
enabled, so a weak model only ever sees the relevant handful of actions.

Validators reuse verify.py (evidence-first, filesystem assertions). Web tasks are
marked needs_web (run in P1 with the real web tool); everything else runs offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set

from ..verify import Verdict, contains_number, fail, file_contains, file_exists, ok

# validator signature: (RunResult, ToolContext) -> Verdict   (typed loosely to avoid import cycle)
Validator = Callable[[object, object], Verdict]
SetupFn = Callable[[Path, object], None]     # (workdir, memory_store) -> None


@dataclass
class Task:
    id: str
    goal: str
    groups: Set[str]
    validators: List[Validator]
    setup: Optional[SetupFn] = None
    expected_tool: Optional[str] = None      # for the tool-selection metric
    needs_web: bool = False
    # optional (state, ctx) -> Verdict gate that must pass before FINISH is accepted; used for
    # side-effect tasks (a file must exist) so the model can't stop before doing the job.
    finish_gate: Optional[Callable[[object, object], Verdict]] = None

    def validate(self, res, ctx) -> List[Verdict]:
        return [v(res, ctx) for v in self.validators]

    def success(self, res, ctx) -> bool:
        return all(v.passed for v in self.validate(res, ctx))


# --- validator builders --------------------------------------------------------
def _finished(res) -> bool:
    return bool(getattr(res, "final_answer", None)) and getattr(res.state, "finished", False)


def answer_has_number(n: str) -> Validator:
    def _v(res, ctx):
        if not _finished(res):
            return fail("did not finish")
        return ok(f"answer has {n}") if contains_number(res.final_answer, n) else fail(f"answer lacks {n}")
    return _v


def answer_has(sub: str) -> Validator:
    def _v(res, ctx):
        if not _finished(res):
            return fail("did not finish")
        return ok(f"answer has {sub!r}") if sub.lower() in res.final_answer.lower() \
            else fail(f"answer lacks {sub!r}")
    return _v


def wrote_file(rel: str, needle: str) -> Validator:
    return lambda res, ctx: file_contains(ctx.workdir, rel, needle)


def used_action(name: str) -> Validator:
    def _v(res, ctx):
        return ok(f"used {name}") if any(getattr(s, "action", None) == name and s.ok
                                         for s in res.steps) else fail(f"never used {name}")
    return _v


def gate_used(action_name: str):
    """Finish-gate: a given action must have succeeded before finishing (e.g. WEB_READ — don't
    answer a web question without actually reading a page). This is the grounding gate."""
    def _g(state, ctx):
        if action_name in getattr(state, "executed_actions", set()):
            return ok(f"{action_name} done")
        return fail(f"{action_name} not done yet — do that before finishing")
    return _g


def gate_wrote(rel: str):
    """Finish-gate: the deliverable file must exist and be non-empty. Checks EXISTENCE, not the
    value, so it forces completion of a side-effect task without leaking the correct answer."""
    def _g(state, ctx):
        p = Path(ctx.workdir) / rel
        if p.is_file() and p.read_text(errors="replace").strip():
            return ok(f"{rel} written")
        return fail(f"{rel} not written yet — do that before finishing")
    return _g


# --- setups --------------------------------------------------------------------
def _setup_lines(n: int, name: str = "data.txt"):
    def _s(wd: Path, _mem):
        (wd / name).write_text("\n".join(f"line {i}" for i in range(1, n + 1)) + "\n")
    return _s


def _setup_needle(wd: Path, _mem):
    (wd / "sub").mkdir(exist_ok=True)
    (wd / "sub" / "needle.txt").write_text("found me")


def _setup_none(wd: Path, _mem):
    pass


# --- the suite -----------------------------------------------------------------
def build_suite() -> List[Task]:
    return [
        Task("file_count", "How many lines are in data.txt? Reply with the number.",
             groups={"shell", "core"}, setup=_setup_lines(7),
             validators=[answer_has_number("7")], expected_tool="shell"),

        Task("file_write", "Create a file named out.txt whose contents are the word BANANA.",
             groups={"file", "core"}, setup=_setup_none,
             validators=[wrote_file("out.txt", "BANANA")], expected_tool="file",
             finish_gate=gate_wrote("out.txt")),

        Task("file_find", "There is a file named needle.txt somewhere here. Confirm it exists.",
             groups={"shell", "core"}, setup=_setup_needle,
             validators=[answer_has("needle.txt")], expected_tool="shell"),

        Task("memory_roundtrip",
             "Remember that the API key label is ZEBRA. Then tell me the API key label.",
             groups={"memory", "core"}, setup=_setup_none,
             validators=[answer_has("ZEBRA"), used_action("RECALL")], expected_tool="memory"),

        Task("cross_cutting",
             "Count the lines in data.txt and save that number into result.txt.",
             groups={"file", "shell", "core"}, setup=_setup_lines(5),
             validators=[wrote_file("result.txt", "5")], expected_tool="file",
             finish_gate=gate_wrote("result.txt")),

        # runs in P1 with the real web tool (SearXNG + fetch)
        Task("web_fact",
             "Find and report the maximum depth of Quesnel Lake in metres.",
             groups={"web", "core"}, setup=_setup_none, needs_web=True,
             validators=[answer_has_number("511"), used_action("WEB_READ")], expected_tool="web",
             finish_gate=gate_used("WEB_READ")),
    ]
