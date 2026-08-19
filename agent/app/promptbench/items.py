"""Build benchmark items from the frozen task-spec fixture.

Two families, both constructed so that the negative population is guaranteed
rather than hoped for:

``select``  -- N described candidates, exactly one satisfies the task's stated
               constraint. One item per candidate set. Label is the survivor's
               name.
``verify``  -- one candidate at a time: does THIS candidate satisfy the
               constraint? Built as a balanced pair per set (the survivor, plus
               one non-survivor drawn deterministically), so positives and
               negatives are equal by construction. This is the family that
               makes a precision-side number defined at all.

The cluster for leave-one-cluster-out is the source task module: items drawn
from the same module share a topic, a statement and an author, and are not
independent.

LEAK GUARD
----------
These items are only meaningful if the prompt does not contain its own answer.
A task statement that names the survivor would make the item measure reading
rather than judgement, so ``_leaks`` drops those sets and the census records
how many were dropped. This is checked on the rendered runtime fields, not on
the label object -- the label is never touched here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from agent.app.promptbench.availability import Item, Label

FIXTURE = Path("agent/tests/fixtures/promptbench/task_specs.json")

STATEMENT_CHARS = 900
VERIFY_CHOICES = ("SATISFIES", "VIOLATES")


def load_specs(path: Path | str = FIXTURE) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return [s for s in data["specs"] if s.get("candidates")]


def _statement_excerpt(spec: Dict[str, Any]) -> str:
    text = " ".join((spec.get("statement") or "").split())
    return text[:STATEMENT_CHARS]


def _mentions(statement: str, name: str) -> bool:
    tokens = [t for t in re.split(r"\W+", name) if len(t) > 3]
    if not tokens:
        return False
    return all(re.search(rf"\b{re.escape(t)}\b", statement, re.IGNORECASE) for t in tokens)


def _leaks(statement: str, spec: Dict[str, Any]) -> bool:
    """True when the statement singles the survivor out.

    These statements enumerate EVERY candidate with its description -- that is
    the task's own candidate list, not a leak, and an earlier version of this
    guard rejected 18 of 19 sets by treating it as one. The contaminating shape
    is *asymmetric* mention: the survivor named where the others are not. If
    every candidate is named, the statement tells you the options and not the
    answer, which is exactly what it should do.
    """
    cands = spec["candidates"]
    survivor = next(c for c in cands if c["survivor"])
    if not _mentions(statement, survivor["name"]):
        return False
    losers = [c for c in cands if not c["survivor"]]
    named_losers = sum(1 for c in losers if _mentions(statement, c["name"]))
    return named_losers == 0


def _usable(spec: Dict[str, Any]) -> bool:
    stmt = _statement_excerpt(spec)
    if not stmt:
        return False
    cands = spec["candidates"]
    if not all(c.get("desc") for c in cands):
        return False
    survivor = next((c for c in cands if c["survivor"]), None)
    if survivor is None:
        return False
    return not _leaks(stmt, spec)


def _pick_negative(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically choose one non-survivor, without Python's hash seed."""
    losers = [c for c in spec["candidates"] if not c["survivor"]]
    digest = hashlib.sha256(spec["module"].encode()).hexdigest()
    return losers[int(digest, 16) % len(losers)]


def _render_candidates(cands: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(f"{i + 1}. {c['name']} -- {c['desc']}" for i, c in enumerate(cands))


def build_select_items(specs: Sequence[Dict[str, Any]]) -> List[Item]:
    items: List[Item] = []
    for spec in specs:
        if not _usable(spec):
            continue
        cands = spec["candidates"]
        survivor = next(c for c in cands if c["survivor"])
        items.append(Item(
            item_id=f"select-{spec['test_id']}",
            cluster=spec["module"],
            runtime={
                "statement": _statement_excerpt(spec),
                "candidates": _render_candidates(cands),
                "choices": [c["name"] for c in cands],
            },
            posthoc={"test_id": spec["test_id"], "n_candidates": len(cands)},
            label=Label(value=survivor["name"], derived_from=f"task_module.{spec['module']}.CANDIDATES.survivor"),
        ))
    return items


def build_verify_items(specs: Sequence[Dict[str, Any]]) -> List[Item]:
    items: List[Item] = []
    for spec in specs:
        if not _usable(spec):
            continue
        survivor = next(c for c in spec["candidates"] if c["survivor"])
        negative = _pick_negative(spec)
        stmt = _statement_excerpt(spec)
        for cand, truth in ((survivor, "SATISFIES"), (negative, "VIOLATES")):
            items.append(Item(
                item_id=f"verify-{spec['test_id']}-{cand['key'] or cand['name'][:8]}",
                cluster=spec["module"],
                runtime={
                    "statement": stmt,
                    "candidate": f"{cand['name']} -- {cand['desc']}",
                    "choices": list(VERIFY_CHOICES),
                },
                posthoc={"test_id": spec["test_id"], "polarity": truth},
                label=Label(value=truth, derived_from=f"task_module.{spec['module']}.CANDIDATES.survivor"),
            ))
    return items


def census(specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dropped_leak, dropped_desc = [], []
    for spec in specs:
        if not all(c.get("desc") for c in spec["candidates"]):
            dropped_desc.append(spec["module"])
        elif _leaks(_statement_excerpt(spec), spec):
            dropped_leak.append(spec["module"])
    select = build_select_items(specs)
    verify = build_verify_items(specs)
    pos = sum(1 for i in verify if i.posthoc["polarity"] == "SATISFIES")
    return {
        "candidate_sets_seen": len(specs),
        "dropped_undescribed": dropped_desc,
        "dropped_statement_leak": dropped_leak,
        "select_items": len(select),
        "select_clusters": len({i.cluster for i in select}),
        "verify_items": len(verify),
        "verify_clusters": len({i.cluster for i in verify}),
        "verify_positive": pos,
        "verify_negative": len(verify) - pos,
    }


def main() -> int:
    specs = load_specs()
    print(json.dumps(census(specs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
