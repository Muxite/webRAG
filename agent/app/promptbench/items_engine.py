"""Two families shaped like real engine decisions: ``followup`` and ``goal_achieved``.

Both are built from candidate sets, and both need only the candidates' NAMES plus
the task statement. That is what lets them reach all 28 one-survivor sets --
including the nine ``select`` must drop for want of per-candidate descriptions,
whose discriminating data lives on Wikipedia and not in this repo.

``followup``   mirrors ``GoTOperations.check_needs_followup``
``goal_achieved`` mirrors ``MergeLeafAction``'s ``goal_achieved`` boolean

WHY THESE TWO
-------------
``goal_achieved`` is the highest-value target in the engine that this bench can
reach. ``CONFIDENCE_JUDGE_MISCALIBRATION.md` measures the merge step at AUC
**0.288 [0.21, 0.37]** -- not uninformative but *anti*-predictive, its interval
excluding 0.5 in the wrong direction -- and quotes a merge scoring confidence 1.0
on an output that did not exist, every clause of its reason praising the mandate
pasted into its own prompt.

So the negative here is built to be exactly that failure: a long, fluent,
methodologically confident synthesis that never names an answer. The item asks
whether the model can tell a non-answer from an answer, which is a capability the
shipped engine demonstrably lacks today.

NEITHER FAMILY REQUIRES KNOWING THE SURVIVOR
--------------------------------------------
Deliberate. A ``goal_achieved`` negative that named the *wrong* candidate would be
judgeable only where per-candidate descriptions exist, making the family's
difficulty depend on which cluster an item came from -- and would duplicate what
``select`` already measures. Completeness of the answer is a different question
from correctness of it, and it is the one merge is actually asked.

``followup``'s judgement is a set difference: the statement enumerates the
candidates, the prompt lists which sub-tasks already exist and which just
finished, and the question is whether anything remains uncovered. The engine asks
the same thing, against ``existing_sibling_tasks``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from agent.app.promptbench.availability import Item, Label

FOLLOWUP_CHOICES = ("YES", "NO")
GOAL_CHOICES = ("ACHIEVED", "NOT_ACHIEVED")

STATEMENT_CHARS = 900


def _statement(spec: Dict[str, Any]) -> str:
    return " ".join((spec.get("statement") or "").split())[:STATEMENT_CHARS]


def _survivor(spec: Dict[str, Any]) -> Dict[str, Any]:
    return next(c for c in spec["candidates"] if c["survivor"])


def _usable(spec: Dict[str, Any]) -> bool:
    cands = spec.get("candidates") or []
    if len(cands) < 3:
        return False
    if sum(1 for c in cands if c["survivor"]) != 1:
        return False
    if not all((c.get("name") or "").strip() for c in cands):
        return False
    return bool(_statement(spec))


def _rotation(spec: Dict[str, Any], n: int) -> int:
    """A per-module offset, so the split point is not always the same index.

    Without it every positive item would stop at candidate 1 and a model could be
    right for a positional reason. ``hashlib`` rather than ``hash`` because the
    latter is salted per interpreter run and the fixture must be reproducible.
    """
    return int(hashlib.sha256(spec["module"].encode()).hexdigest(), 16) % n


def _task_title(name: str) -> str:
    return f"Check whether {name} satisfies the stated requirement"


# ---------------------------------------------------------------------------
# followup
# ---------------------------------------------------------------------------

def build_followup_items(specs: Sequence[Dict[str, Any]]) -> List[Item]:
    items: List[Item] = []
    for spec in specs:
        if not _usable(spec):
            continue
        names = [c["name"] for c in spec["candidates"]]
        stmt = _statement(spec)

        # NO: every candidate is accounted for -- one just finished, the rest
        # already have sibling tasks. Nothing concrete is left to open.
        completed = names[-1]
        siblings = [_task_title(n) for n in names[:-1]]
        items.append(Item(
            item_id=f"followup-{spec['test_id'] or spec['module']}-covered",
            cluster=spec["module"],
            runtime={
                "statement": stmt,
                "completed_task": _task_title(completed),
                "completed_result": (
                    f"Opened the page for {completed} and read the figure the task asks for. "
                    "The value was recorded."
                ),
                "siblings": "\n".join(f"- {s}" for s in siblings),
                "choices": list(FOLLOWUP_CHOICES),
            },
            posthoc={"test_id": spec["test_id"], "polarity": "NO",
                     "n_candidates": len(names), "n_covered": len(names)},
            label=Label(value="NO",
                        derived_from=f"task_module.{spec['module']}.CANDIDATES.coverage"),
        ))

        # YES: the sibling set plus the finished task leave named candidates from
        # the statement untouched, so a concrete follow-up exists.
        cut = 1 + _rotation(spec, len(names) - 2)      # 1 .. len-2 covered as siblings
        completed = names[cut]
        siblings = [_task_title(n) for n in names[:cut]]
        items.append(Item(
            item_id=f"followup-{spec['test_id'] or spec['module']}-uncovered",
            cluster=spec["module"],
            runtime={
                "statement": stmt,
                "completed_task": _task_title(completed),
                "completed_result": (
                    f"Opened the page for {completed} and read the figure the task asks for. "
                    "The value was recorded."
                ),
                "siblings": "\n".join(f"- {s}" for s in siblings),
                "choices": list(FOLLOWUP_CHOICES),
            },
            posthoc={"test_id": spec["test_id"], "polarity": "YES",
                     "n_candidates": len(names), "n_covered": cut + 1},
            label=Label(value="YES",
                        derived_from=f"task_module.{spec['module']}.CANDIDATES.coverage"),
        ))
    return items


# ---------------------------------------------------------------------------
# goal_achieved
# ---------------------------------------------------------------------------

def _preamble(spec: Dict[str, Any]) -> str:
    """The methodological prose both polarities share, verbatim.

    Shared so the two arms differ in ONE respect: whether an answer is named. An
    earlier draft wrote the negative long and fluent and left the positive terse,
    which made the synthesis 2.4x longer on NOT_ACHIEVED -- a model could then
    score well above chance on length alone, without reading a word.
    """
    names = [c["name"] for c in spec["candidates"]]
    return (
        "The approach follows the task's structure closely. Each of the "
        f"{len(names)} candidates ({', '.join(names)}) was identified from the task "
        "statement, the relevant page was located for each, and the attribute the task "
        "specifies was identified as the basis for comparison. The methodology is sound "
        "and matches the requirement as stated: locate each candidate, read the "
        "attribute, compare across the set."
    )


def _achieved_synthesis(spec: Dict[str, Any]) -> str:
    survivor = _survivor(spec)
    return (
        f"{_preamble(spec)} The comparison across the recorded values has been carried "
        f"out, and {survivor['name']} is the candidate that satisfies the requirement. "
        f"ANSWER: {survivor['name']}."
    )


def _stalled_synthesis(spec: Dict[str, Any]) -> str:
    """Fluent, confident, methodologically complete -- and it names no answer.

    Modelled on the verbatim merge failure in ``CONFIDENCE_JUDGE_MISCALIBRATION.md``
    §3, where a merge scored confidence 1.0 on an output that did not exist and
    every clause of its reason praised the mandate pasted into its own prompt.
    """
    return (
        f"{_preamble(spec)} The comparison across the recorded values still needs to be "
        "carried out before the requirement can be reported as met, so no candidate is "
        "put forward here."
    )


def build_goal_achieved_items(specs: Sequence[Dict[str, Any]]) -> List[Item]:
    items: List[Item] = []
    for spec in specs:
        if not _usable(spec):
            continue
        stmt = _statement(spec)
        for synth, truth, tag in (
            (_achieved_synthesis(spec), "ACHIEVED", "answered"),
            (_stalled_synthesis(spec), "NOT_ACHIEVED", "stalled"),
        ):
            items.append(Item(
                item_id=f"goal-{spec['test_id'] or spec['module']}-{tag}",
                cluster=spec["module"],
                runtime={
                    "statement": stmt,
                    "synthesis": synth,
                    "choices": list(GOAL_CHOICES),
                },
                posthoc={"test_id": spec["test_id"], "polarity": truth,
                         "synthesis_chars": len(synth)},
                label=Label(value=truth,
                            derived_from=f"task_module.{spec['module']}.CANDIDATES.survivor"),
            ))
    return items


def census(specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [s for s in specs if _usable(s)]
    fu = build_followup_items(specs)
    ga = build_goal_achieved_items(specs)
    out: Dict[str, Any] = {
        "candidate_sets_seen": sum(1 for s in specs if s.get("candidates")),
        "usable_sets": len(usable),
        "dropped": [s["module"] for s in specs if s.get("candidates") and not _usable(s)],
    }
    for name, items, positive in (("followup", fu, "YES"), ("goal_achieved", ga, "ACHIEVED")):
        pos = sum(1 for i in items if i.posthoc["polarity"] == positive)
        out[f"{name}_items"] = len(items)
        out[f"{name}_clusters"] = len({i.cluster for i in items})
        out[f"{name}_positive"] = pos
        out[f"{name}_negative"] = len(items) - pos
    # A length cue would let a model score without reading. Reported, not asserted:
    # the test suite is where this becomes a gate.
    if ga:
        ach = [i.posthoc["synthesis_chars"] for i in ga if i.posthoc["polarity"] == "ACHIEVED"]
        notach = [i.posthoc["synthesis_chars"] for i in ga if i.posthoc["polarity"] != "ACHIEVED"]
        out["goal_achieved_mean_chars"] = {
            "ACHIEVED": round(sum(ach) / len(ach)),
            "NOT_ACHIEVED": round(sum(notach) / len(notach)),
        }
    return out


def main() -> int:
    path = Path("agent/tests/fixtures/promptbench/task_specs.json")
    specs = json.loads(path.read_text())["specs"]
    print(json.dumps(census(specs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
