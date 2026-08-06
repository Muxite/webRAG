"""
Test 202: Self-contained reasoning — 5-bay constraint-satisfaction (ordering/assignment)
Level: reasoning   Weight: long   Difficulty: 6/10   Grounding required: NO

This task belongs to the self-contained "general reasoning" category, NOT the web-grounded
suite: there is no page to visit, no URL, no search. Every fact needed to reach the answer is
stated in ``get_task_statement()`` itself, so the usual ``visit.count > 0`` keystone gate is
structurally inapplicable (``get_test_metadata()["grounding_required"] = False``). It is meant
to run through the ``parametric`` (no-tools) execution variant.
``scripts/validator_lint.py`` therefore reports one EXPECTED ``[GATE]`` finding for this file
(the keystone scores without grounding — by design, there is nothing to ground against); the
``[LLM]`` bar is NOT relaxed: validation here is 100% deterministic.

Anti-memorisation design
------------------------
Recognisable named puzzles (river crossing, Monty Hall, the "Einstein"/zebra grid, etc.) are
unusable here because their ANSWER may sit in parametric memory. So both the narrative (a
generic depot inspection-bay roster) and the parameters (five invented technician names, a
procedurally-searched clue set) are novel: the clue set below was found by randomised search
over a clue-template pool (kinds: immediately-left / not-adjacent / numeric-gap / excluded-bay),
keeping only sets that are (a) satisfied by exactly ONE of the 5! = 120 permutations and
(b) MINIMAL — deleting any single clue leaves the answer under-determined (>1 solution), so no
clue is decorative and the solver must use all five.

Ground truth (reference-solver verified, never hand-derived)
-----------------------------------------------------------
    Bay 1: Delphine   Bay 2: Marisol   Bay 3: Corwin   Bay 4: Halvard   Bay 5: Teodor
Solver A (``_solve_bruteforce`` below, in this module): exhaustive scan of all 120 permutations
against the CLUES predicates -> exactly 1 survivor.
Solver B (``_solve_by_backtracking`` in
``services/agent/tests/test_202_reasoning_bay_assignment_validators_test.py``): an independent
implementation — seat-by-seat DFS with clue predicates re-written by hand FROM THE ENGLISH
STATEMENT (not from this module's spec), so a mis-transcription between the prose and the
predicates cannot hide from the cross-check. Both solvers, and the uniqueness + per-clue
minimality assertions, are re-runnable in that test file.

Discrimination: the keystone demands the WHOLE arrangement (1/120 by guessing), while the
un-gated ``position_coverage`` / ``clue_consistency`` diagnostics measure how far a wrong
answer got — a random permutation scores ~1/5 expected on coverage, so the score stays bimodal.
"""

from typing import Dict, Any, List
import re
from itertools import permutations
from agent.app.idea_test_utils import extract_final_text


# --- puzzle parameters (single source of truth for statement, solvers and validators) -------
N_BAYS = 5
# Roster order as PRESENTED (alphabetical) — deliberately NOT the solution order, so "answer =
# the order they were listed in" earns nothing.
ENTITIES: List[str] = ["Corwin", "Delphine", "Halvard", "Marisol", "Teodor"]

# The unique arrangement, index i -> bay i+1. Verified by the two reference solvers (see above).
SOLUTION: List[str] = ["Delphine", "Marisol", "Corwin", "Halvard", "Teodor"]

# Clues: prose + predicate over ``pos`` (name -> bay number, 1-based). The statement is BUILT
# from these strings, so prose and predicate can never drift apart.
CLUES: List[Dict[str, Any]] = [
    {"text": "Marisol works immediately to the left of Corwin (Marisol's bay number is exactly "
             "one less than Corwin's).",
     "check": lambda pos: pos["Marisol"] + 1 == pos["Corwin"]},
    {"text": "Delphine does not work in bay 5.",
     "check": lambda pos: pos["Delphine"] != 5},
    {"text": "Corwin's bay number and Teodor's bay number differ by exactly 2.",
     "check": lambda pos: abs(pos["Corwin"] - pos["Teodor"]) == 2},
    {"text": "Corwin works immediately to the left of Halvard (Corwin's bay number is exactly "
             "one less than Halvard's).",
     "check": lambda pos: pos["Corwin"] + 1 == pos["Halvard"]},
    {"text": "Marisol and Teodor do not work in adjacent bays (their bay numbers are not "
             "consecutive).",
     "check": lambda pos: abs(pos["Marisol"] - pos["Teodor"]) != 1},
]

# Secondary deliverable: the bays read RIGHT to LEFT (bay 5 -> bay 1), first letter of each name.
SHIFT_CODE = "".join(name[0] for name in reversed(SOLUTION)).upper()   # -> "THCMD"


def _solve_bruteforce(clues: List[Dict[str, Any]] = None) -> List[List[str]]:
    """Reference solver A: exhaustive scan of every permutation of the roster.

    :param clues: Clue list to satisfy (defaults to the task's own ``CLUES``).
    :return: Every arrangement (list indexed bay-1) satisfying all clues.
    """
    clues = CLUES if clues is None else clues
    found: List[List[str]] = []
    for perm in permutations(ENTITIES):
        pos = {name: i + 1 for i, name in enumerate(perm)}
        if all(c["check"](pos) for c in clues):
            found.append(list(perm))
    return found


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "202",
        "test_name": "Reasoning: 5-bay constraint satisfaction (unique assignment)",
        "difficulty_level": "6/10",
        "category": "Self-contained Constraint Satisfaction",
        # Deliberately outside the web ladder's LEVEL_ORDER (micro/integration/navigation/graph):
        # these rows must not be averaged into a grounding ladder they cannot participate in.
        "level": "reasoning",
        "weight": "long",
        "grounding_required": False,
    }


def get_task_statement() -> str:
    clue_lines = "\n".join(f"  ({i}) {c['text']}" for i, c in enumerate(CLUES, 1))
    roster = ", ".join(ENTITIES)
    return (
        "This is a pure reasoning task. Do NOT search the web and do NOT open any page — every "
        "fact you need is written below, and no external source contains this scenario.\n\n"
        f"A maintenance depot has {N_BAYS} inspection bays standing in one straight row, "
        f"numbered 1 to {N_BAYS} from left to right. On the morning shift exactly one technician "
        f"works in each bay, and every technician works in exactly one bay. The {N_BAYS} "
        f"technicians on shift are: {roster}.\n\n"
        "The shift log records these facts:\n"
        f"{clue_lines}\n\n"
        f"Exactly one assignment of the {N_BAYS} technicians to the {N_BAYS} bays satisfies all "
        "of the facts at once. Work it out.\n\n"
        "Report BOTH of the following:\n"
        f"(a) The full assignment, as {N_BAYS} separate lines in exactly this form:\n"
        "    Bay 1: <technician>\n"
        "    Bay 2: <technician>\n"
        "    ... and so on through "
        f"Bay {N_BAYS}.\n"
        "(b) The shift code: read the bays from RIGHT to LEFT (bay "
        f"{N_BAYS} first, bay 1 last) and write the FIRST LETTER of each technician's name in "
        f"that order as a single {N_BAYS}-letter string. (Format example only, using names that "
        "are not on this roster: if the right-to-left reading were Quill, Renn, Sabra, ... the "
        "code would begin 'QRS...'.)"
    )


def get_required_deliverables() -> List[str]:
    return [
        f"The technician in each of bays 1-{N_BAYS} (the full unique assignment)",
        f"The {N_BAYS}-letter shift code (first letters, bay {N_BAYS} down to bay 1)",
    ]


def get_success_criteria() -> List[str]:
    return [
        "All five bays assigned to the correct technician (the unique satisfying arrangement)",
        "The reported arrangement violates none of the five stated facts",
        f"The shift code is reported correctly ({N_BAYS} letters, right-to-left reading)",
        "No web sources used or needed (self-contained reasoning)",
    ]


# --- answer parsing -------------------------------------------------------------------------
# Layout-tolerant, priority-tiered extraction of the reported arrangement. Tiers are tried in
# order and only fill bays still missing, so an explicit "Bay 3: Corwin" answer always beats a
# looser signal (e.g. a numbered restatement of the CLUES, which is exactly the false-positive
# that would otherwise mark a CORRECT answer wrong).
_NAME_ALT = "|".join(sorted(ENTITIES, key=len, reverse=True))
_SLOT_WORD = r"(?:bay|seat|slot|position|station)"
# Horizontal whitespace only: a label and its name must sit on the SAME line, otherwise
# "Bay 1: Delphine\nMarisol ..." would let bay 1 swallow the next row's name.
_H = r"[^\S\n]"
# Junk allowed between a bay label and its name: spaces, separators, markdown/JSON decoration.
# Deliberately excludes newline, comma, "." and every letter/digit — so an intervening WORD
# (as in a restated clue, "... in bay 5. Marisol is ...") can never glue a label to a name.
_SEP = r"[ \t:=\-–—>\)\]\|\*_`\"'~]*"
# Optional prefix junk on the number itself: "bay_1", "bay #1", "**bay** 1".
_NUMPRE = r"[ \t_\-\*`\"']*#?[ \t]*"
# Lines that are restated clues / roster echoes, never answer rows.
_CLUEISH = re.compile(
    r"immediately|adjacent|consecutive|differ|does not|do not|neither|either|"
    r"left of|right of|roster|technicians are|clue|fact\s*\(",
    re.IGNORECASE,
)


def _scan_labeled_forward(text: str) -> Dict[int, str]:
    """Tier 1 — "Bay 3: Corwin", "bay #3 - Corwin", "**Bay 3:** Corwin", "| Bay 3 | Corwin |",
    '"bay_3": "Corwin"'."""
    out: Dict[int, str] = {}
    for i in range(1, N_BAYS + 1):
        rx = re.compile(
            rf"{_SLOT_WORD}{_NUMPRE}{i}\b{_SEP}"
            rf"(?:is{_H}+|holds{_H}+|has{_H}+)?{_SEP}({_NAME_ALT})\b",
            re.IGNORECASE,
        )
        hits = rx.findall(text)
        if hits:
            out[i] = hits[-1].capitalize()
    return out


def _scan_labeled_reverse(text: str) -> Dict[int, str]:
    """Tier 2 — "Corwin: bay 3", "Corwin sits in bay 3", "Corwin (bay 3)".

    The connector alternation deliberately admits no negation, so a restated clue such as
    "Delphine does not work in bay 5" cannot register as an assignment.
    """
    out: Dict[int, str] = {}
    rx = re.compile(
        rf"\b({_NAME_ALT})\b[ \t,:=\-–—>\(\[\*_`\"'~]*"
        rf"(?:is{_H}+|sits{_H}+|works{_H}+|stands{_H}+|occupies{_H}+)?"
        rf"(?:in{_H}+|at{_H}+)?[ \t,:=\-–—>\(\[\*_`\"'~]*{_SLOT_WORD}{_NUMPRE}(\d+)\b",
        re.IGNORECASE,
    )
    for name, num in rx.findall(text):
        n = int(num)
        if 1 <= n <= N_BAYS:
            out[n] = name.capitalize()
    return out


def _scan_bare_numbered(text: str) -> Dict[int, str]:
    """Tier 3 — bare numbered/table rows: "3. Corwin", "3) Corwin", "| 3 | Corwin |",
    "- **3.** Corwin"."""
    out: Dict[int, str] = {}
    for line in text.splitlines():
        if _CLUEISH.search(line):
            continue
        m = re.match(
            rf"[ \t\*\-\+\|`\"']*{_SLOT_WORD}?{_NUMPRE}(\d+)[ \t\*_`\"']*"
            rf"[\.\):\-–—\|][ \t\*_`\"'\|]*({_NAME_ALT})\b",
            line, re.IGNORECASE,
        )
        if m:
            n = int(m.group(1))
            if 1 <= n <= N_BAYS:
                out[n] = m.group(2).capitalize()
    return out


def _scan_ordered_line(text: str) -> Dict[int, str]:
    """Tier 4 — a single line naming every technician exactly once, read left to right."""
    best: Dict[int, str] = {}
    for line in text.splitlines():
        if _CLUEISH.search(line) or re.search(r"\d", line):
            continue
        names = [m.group(0).capitalize()
                 for m in re.finditer(rf"\b({_NAME_ALT})\b", line, re.IGNORECASE)]
        if len(names) == N_BAYS and set(names) == set(ENTITIES):
            best = {i + 1: name for i, name in enumerate(names)}
    return best


def _parse_arrangement(text: str) -> Dict[int, str]:
    """
    Extract the reported bay -> technician mapping from free-form report text.
    :param text: The agent's final deliverable text.
    :return: Mapping of bay number to technician name (may be partial).
    """
    mapping: Dict[int, str] = {}
    for scan in (_scan_labeled_forward, _scan_labeled_reverse,
                 _scan_bare_numbered, _scan_ordered_line):
        if len(mapping) == N_BAYS:
            break
        for pos, name in scan(text).items():
            mapping.setdefault(pos, name)
    return mapping


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE predicate: every bay assigned to the correct technician.

    NOTE — no grounding clause on purpose: this task is self-contained
    (``grounding_required=False``), so there is no page-read to gate on and
    ``scripts/validator_lint.py`` correctly reports a ``[GATE]`` finding here. That finding is
    the honest signal for this category and must not be silenced. The anti-guessing work is
    carried by the answer space instead: all 5 bays must be right at once (1 in 120).
    """
    parsed = _parse_arrangement(extract_final_text(result))
    return all(parsed.get(i + 1) == name for i, name in enumerate(SOLUTION))


def validate_keystone_arrangement(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the complete, uniquely-determined bay assignment."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_arrangement", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Correct unique arrangement " +
                       ", ".join(f"bay {i + 1}={n}" for i, n in enumerate(SOLUTION)))
                      if passed else "Bay assignment missing or does not match the unique solution"}


def validate_position_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Un-gated diagnostic: how many bays were assigned correctly (partial credit).

    Deliberately NOT short-circuited on the keystone — it is the axis that shows whether a
    failed run reasoned most of the way to the answer or produced noise. Expected value for a
    random permutation is exactly 1 correct bay (0.2), so it cannot masquerade as success.
    """
    parsed = _parse_arrangement(extract_final_text(result))
    correct = sum(1 for i, name in enumerate(SOLUTION) if parsed.get(i + 1) == name)
    return {"check": "position_coverage", "passed": correct == N_BAYS, "score": correct / N_BAYS,
            "reason": f"{correct}/{N_BAYS} bays assigned correctly "
                      f"({len(parsed)}/{N_BAYS} bays reported at all)"}


def validate_clue_consistency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Un-gated diagnostic: how many of the stated facts the REPORTED arrangement satisfies.

    Because the clue set admits exactly one solution, any wrong-but-complete answer must break
    at least one fact; an incomplete answer cannot evaluate its clues at all (they count as
    violated). This separates "arithmetic slip on one constraint" from "ignored the constraints".
    """
    parsed = _parse_arrangement(extract_final_text(result))
    pos = {name: bay for bay, name in parsed.items()}
    satisfied = 0
    broken: List[int] = []
    for idx, clue in enumerate(CLUES, 1):
        try:
            ok = bool(clue["check"](pos))
        except KeyError:            # a technician the report never placed
            ok = False
        if ok:
            satisfied += 1
        else:
            broken.append(idx)
    n = len(CLUES)
    return {"check": "clue_consistency", "passed": satisfied == n, "score": satisfied / n,
            "reason": f"{satisfied}/{n} stated facts satisfied by the reported arrangement"
                      + (f" (violates {broken})" if broken else "")}


_SHIFT_CODE_RE = re.compile(r"\b" + r"[\s\-,.]{0,3}".join(SHIFT_CODE) + r"\b", re.IGNORECASE)


def validate_shift_code(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary (SHORT-CIRCUITS TO 0 without the keystone): the right-to-left letter code.

    Gating it on the keystone is what keeps scores bimodal: a run that never solved the puzzle
    cannot bank a consolation point for a code it could only have guessed.
    """
    if not _keystone_ok(result, observability):
        return {"check": "shift_code", "passed": False, "score": 0.0,
                "reason": "Arrangement keystone absent -> shift code not credited"}
    ok = bool(_SHIFT_CODE_RE.search(extract_final_text(result)))
    return {"check": "shift_code", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": f"Shift code {SHIFT_CODE} reported" if ok
                      else f"Shift code {SHIFT_CODE} missing/incorrect"}


def get_validation_functions() -> List[callable]:
    return [validate_keystone_arrangement, validate_position_coverage,
            validate_clue_consistency, validate_shift_code]


def get_llm_validation_function() -> callable:
    return None
