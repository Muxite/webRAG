"""c53 — bin rebalancing: a closed-environment REASONING task on a container filesystem.

Distinct in kind from c01..c52, which are all code-*writing* tasks. Here the agent writes no
program: it reads four "container" files, works out which items to move so every container holds
exactly the same total weight, and writes the containers back. The deliverable IS the final
filesystem state.

**Why this shape.** The web suite's failure modes all trace to the environment rather than the
reasoning: site flakiness, unbounded page context, and answers a model can recall instead of
derive (a live case had an arm answer a 7-part question from search snippets with 0 page visits).
A sealed container removes all three. The ground truth here is CONSTRUCTED, not discovered, so
there is nothing to recall and nothing to fabricate — a wrong assignment fails arithmetic that the
grader redoes from the files themselves.

**Instance provenance (do not regenerate casually).** Weights come from
``random.Random(20260824)``, filtered so that:

  * total is 108 and divides evenly by 4 containers -> target 27 per container;
  * **descending first-fit greedy FAILS** — the natural heuristic ("move the heaviest thing that
    fits") does not reach a solution, so a model that pattern-matches instead of searching lands
    in a dead end rather than stumbling into the answer;
  * a solution nevertheless exists, confirmed by a SECOND, independently-written solver
    (memoized over sorted remaining capacities) rather than by the same search that found it;
  * exactly 14 distinct assignments satisfy it under symmetry breaking — enough that the task is
    a real search and not a knife-edge, few enough that guessing is hopeless.

Both reference solvers live in this module and are re-run by
``agent/tests/idea_code_test_c53_test.py``, so the claims above are re-derived on every test run
rather than trusted from this docstring.

**No code execution is required or available.** The native engine's sandbox pack and the flat
arms' shared surface expose file verbs only — no ``run_python``. The agent has to do the
arithmetic itself and write intermediate results down, which is the capability under test.

**Staged**, in the sense the harness supports: one canonical check per stage, each asserting that
stage's persisted artifact. There is no mid-run gate, so a later stage is attempted even if an
earlier one failed and the instructions tell the agent to re-derive rather than trust an upstream
summary.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

CONTAINERS: Tuple[str, ...] = ("alpha", "bravo", "charlie", "delta")

#: item -> weight. See the module docstring for how this instance was selected.
WEIGHTS: Dict[str, int] = {
    "bolt": 18, "cog": 6, "dowel": 3, "flange": 5, "gasket": 6, "hinge": 9,
    "jig": 19, "key": 15, "latch": 15, "nut": 3, "pin": 4, "rivet": 5,
}

#: The deliberately lopsided starting arrangement (52 / 14 / 30 / 12 against a target of 27).
#: No container starts at target, so every one of them has to change.
START: Dict[str, List[str]] = {
    "alpha": ["bolt", "jig", "key"],
    "bravo": ["cog", "dowel", "flange"],
    "charlie": ["gasket", "hinge", "latch"],
    "delta": ["nut", "pin", "rivet"],
}

TOTAL = sum(WEIGHTS.values())
TARGET = TOTAL // len(CONTAINERS)

VISIBLE_TEST_PATH = "tests/test_rebalance.py"


# --- reference solvers (two, deliberately independent) ---------------------------------------

def solve_exhaustive(items: List[Tuple[str, int]], n_bins: int, target: int) -> List[Tuple[int, ...]]:
    """Enumerate every assignment whose bins each sum to ``target``.

    Symmetry-broken over equal partial sums so permutations of identical bins are not counted
    as distinct solutions.

    :returns: One tuple of bin indices per solution, in item order.
    """
    solutions: List[Tuple[int, ...]] = []
    n = len(items)

    def rec(i: int, sums: List[int], assign: List[int]) -> None:
        if i == n:
            if all(s == target for s in sums):
                solutions.append(tuple(assign))
            return
        if len(solutions) > 400:
            return
        weight = items[i][1]
        seen = set()
        for b in range(n_bins):
            if sums[b] + weight > target or sums[b] in seen:
                continue
            seen.add(sums[b])
            sums[b] += weight
            assign.append(b)
            rec(i + 1, sums, assign)
            sums[b] -= weight
            assign.pop()

    rec(0, [0] * n_bins, [])
    return solutions


def solve_capacity_dp(items: List[Tuple[str, int]], n_bins: int, target: int) -> bool:
    """Independent feasibility check, memoized over SORTED remaining capacities.

    Deliberately a different formulation from :func:`solve_exhaustive` -- it answers only
    "is this solvable", by a different recursion, so agreement between the two is evidence
    rather than a restatement.
    """
    n = len(items)

    @lru_cache(maxsize=None)
    def rec(i: int, caps: Tuple[int, ...]) -> bool:
        if i == n:
            return all(c == 0 for c in caps)
        weight = items[i][1]
        for j, cap in enumerate(caps):
            if cap >= weight:
                nxt = list(caps)
                nxt[j] = cap - weight
                if rec(i + 1, tuple(sorted(nxt))):
                    return True
        return False

    return rec(0, tuple([target] * n_bins))


def greedy_first_fit_solves(items: List[Tuple[str, int]], n_bins: int, target: int) -> bool:
    """Descending first-fit — the heuristic a non-reasoning model reaches for.

    This MUST return False for the shipped instance: if the obvious greedy worked, the task
    would not discriminate reasoning from pattern-matching.
    """
    caps = [target] * n_bins
    for _, weight in sorted(items, key=lambda pair: -pair[1]):
        for j in range(n_bins):
            if caps[j] >= weight:
                caps[j] -= weight
                break
        else:
            return False
    return all(c == 0 for c in caps)


def items_list() -> List[Tuple[str, int]]:
    """The instance as ``(name, weight)`` pairs, in a fixed order."""
    return [(name, WEIGHTS[name]) for name in sorted(WEIGHTS)]


# --- harness contract ------------------------------------------------------------------------

def get_test_metadata() -> dict:
    return {"test_id": "c53", "title": "bin-rebalance-closed-env", "category": "hard"}


def get_visibility() -> str:
    return "visible"


def get_task_statement() -> str:
    lines = [
        "You are working in a directory containing four container files:",
        "",
    ]
    for name in CONTAINERS:
        lines.append(f"  containers/{name}.txt")
    lines += [
        "",
        "Each line of a container file is an item and its weight, separated by a space, e.g.:",
        "",
        "  bolt 18",
        "",
        "GOAL: move items between containers so that EVERY container holds items summing to "
        f"EXACTLY {TARGET}.",
        "",
        "Rules:",
        "  - You may not split, invent, rename, duplicate or discard an item. Every item that "
        "starts somewhere must end up in exactly one container.",
        "  - An item's weight never changes.",
        "  - A container may hold any number of items, including a different number than it "
        "started with.",
        "",
        "Produce these files when you are done:",
        f"  1. containers/<name>.txt for each of the four containers, rewritten to their FINAL "
        "contents, same 'item weight' line format.",
        "  2. totals.txt — one line per container, 'name total', listing each container's final "
        "total weight.",
        "  3. moves.txt — one line per item you moved, in the form 'item from->to' "
        "(e.g. 'bolt alpha->delta'). List only items that actually changed container.",
        "",
        "Work it out step by step and write your intermediate reasoning to a scratch file if it "
        "helps — you have no calculator and no way to run code, so be careful with the "
        "arithmetic and check each container's total by re-adding it before you finish.",
    ]
    return "\n".join(lines)


def get_sandbox_fixture() -> dict:
    """The starting containers, plus the visible test file."""
    fixture = {
        f"containers/{name}.txt": "".join(
            f"{item} {WEIGHTS[item]}\n" for item in START[name]
        )
        for name in CONTAINERS
    }
    fixture[VISIBLE_TEST_PATH] = _TEST_FILE_CONTENT
    return fixture


KEYSTONE_TEST_IDS = [f"{VISIBLE_TEST_PATH}::test_every_container_hits_the_target"]


_TEST_FILE_CONTENT = '''"""Canonical checks for c53. The grader re-derives every total from the files."""
import os
import re

CONTAINERS = ("alpha", "bravo", "charlie", "delta")
WEIGHTS = {
    "bolt": 18, "cog": 6, "dowel": 3, "flange": 5, "gasket": 6, "hinge": 9,
    "jig": 19, "key": 15, "latch": 15, "nut": 3, "pin": 4, "rivet": 5,
}
TARGET = 27
START = {
    "alpha": ["bolt", "jig", "key"],
    "bravo": ["cog", "dowel", "flange"],
    "charlie": ["gasket", "hinge", "latch"],
    "delta": ["nut", "pin", "rivet"],
}


def _read(name):
    path = os.path.join("containers", name + ".txt")
    if not os.path.exists(path):
        return None
    entries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                entries.append((parts[0], int(parts[1])))
            except ValueError:
                continue
    return entries


def _all():
    return {name: _read(name) for name in CONTAINERS}


def test_every_container_file_exists_and_parses():
    """Stage 1: the containers survived as readable files."""
    for name, entries in _all().items():
        assert entries is not None, "missing containers/%s.txt" % name
        assert entries, "containers/%s.txt is empty" % name


def test_weights_were_not_altered():
    """An item's weight is fixed; rewriting one is a way to fake a total."""
    for name, entries in _all().items():
        for item, weight in (entries or []):
            assert item in WEIGHTS, "unknown item %r in %s" % (item, name)
            assert weight == WEIGHTS[item], "weight of %r changed in %s" % (item, name)


def test_items_are_conserved():
    """Stage 2: nothing invented, duplicated or dropped."""
    seen = []
    for entries in _all().values():
        seen.extend(item for item, _ in (entries or []))
    assert sorted(seen) == sorted(WEIGHTS), "item multiset changed"


def test_every_container_hits_the_target():
    """KEYSTONE: the actual rebalancing, re-added from the files by the grader."""
    for name, entries in _all().items():
        total = sum(w for _, w in (entries or []))
        assert total == TARGET, "container %s totals %d, expected %d" % (name, total, TARGET)


def test_totals_file_matches_the_containers():
    """Stage 3: the reported totals agree with the filesystem."""
    assert os.path.exists("totals.txt"), "missing totals.txt"
    reported = {}
    with open("totals.txt") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    reported[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    for name, entries in _all().items():
        assert name in reported, "totals.txt does not mention %s" % name
        assert reported[name] == sum(w for _, w in (entries or [])), \\
            "totals.txt disagrees with containers/%s.txt" % name


def test_moves_file_reconciles_start_to_finish():
    """Stage 4: the move log explains how the final state was reached."""
    assert os.path.exists("moves.txt"), "missing moves.txt"
    moved = {}
    with open("moves.txt") as fh:
        for line in fh:
            m = re.match(r"^\\s*(\\S+)\\s+(\\S+)\\s*->\\s*(\\S+)\\s*$", line)
            if m:
                moved[m.group(1)] = (m.group(2), m.group(3))

    origin = {item: name for name, items in START.items() for item in items}
    final = {item: name for name, entries in _all().items() for item, _ in (entries or [])}
    for item, dest in final.items():
        if origin.get(item) != dest:
            assert item in moved, "%s changed container but is not in moves.txt" % item
            assert moved[item] == (origin[item], dest), "moves.txt misreports %s" % item
'''


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        # Informational only (written into meta.json). This task's deliverable is filesystem
        # state, not an importable module, so there is no callable entrypoint.
        "entrypoint": {"module": None, "functions": []},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Three leaves: survey, rebalance, report.

    Leaf 1 is genuinely parallel work (each container's total is independent of the others);
    leaf 2 is the merge that needs all of them at once. That is the shape the graph engine's
    fan-out-then-merge machinery exists for, and the reason this task is worth running across
    arms rather than only on the compiled scaffold.

    Leaks nothing: no leaf names a final assignment, and the target is already in the prompt.
    """
    return {
        "leaves": [
            {
                "id": "survey",
                "instruction": (
                    "Read every file in containers/ and write survey.txt listing, for each "
                    "container, its items with weights and its current total. Add each "
                    "container's total by hand and double-check it — every later step depends "
                    "on these numbers being right."
                ),
                "expect": "survey.txt lists all four containers with their items and totals",
                "depends_on": [],
            },
            {
                "id": "rebalance",
                "instruction": (
                    "Using the containers themselves as the source of truth (re-read them; do "
                    "not trust an earlier summary), work out an assignment of every item to a "
                    f"container such that each container's items sum to exactly {TARGET}. Note "
                    "that repeatedly placing the heaviest remaining item into the first "
                    "container it fits does NOT work here — you will need to backtrack. Then "
                    "rewrite each containers/<name>.txt with its final contents, one 'item "
                    "weight' per line. {survey}"
                ),
                "expect": f"every containers/<name>.txt sums to exactly {TARGET}",
                "depends_on": ["survey"],
            },
            {
                "id": "report",
                "instruction": (
                    "Re-read every containers/<name>.txt as it now stands. Write totals.txt "
                    "with one 'name total' line per container, and moves.txt with one "
                    "'item from->to' line for each item whose container changed from the "
                    "original arrangement. {rebalance}"
                ),
                "expect": "totals.txt and moves.txt agree with the container files",
                "depends_on": ["rebalance"],
            },
        ],
        "aggregation": (
            "Confirm every container totals exactly "
            f"{TARGET}, that totals.txt matches, and that moves.txt explains the changes."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {
            "op": "submit_files",
            "files": [f"containers/{name}.txt" for name in CONTAINERS] + ["totals.txt", "moves.txt"],
        },
    }
