"""
Test 203: Self-contained reasoning — 6-slot constraint-satisfaction (temporal ordering)
Level: reasoning   Weight: long   Difficulty: 7/10   Grounding required: NO

Sibling of test 202 (spatial 5-bay assignment); this instance varies EVERY axis: six entities
instead of five, a temporal running-order instead of a left-to-right row, a different narrative
(a workshop's afternoon speaking order), a different, independently-searched clue set including
an "either first or last" end constraint, and a numeric secondary deliverable (a sum of slot
numbers) instead of a letter code.

Self-contained category, NOT the web-grounded suite: no URL, no search, no page to visit —
``get_task_statement()`` carries every fact needed, so the suite's ``visit.count > 0`` keystone
gate is structurally inapplicable (``get_test_metadata()["grounding_required"] = False``) and
the task runs under the ``parametric`` (no-tools) execution variant.
``scripts/validator_lint.py`` reports one EXPECTED ``[GATE]`` finding for this file (a keystone
that scores without grounding — by design); the ``[LLM]`` bar is NOT relaxed: validation is
100% deterministic, ``get_llm_validation_function()`` returns None.

Anti-memorisation design
------------------------
No recognisable named puzzle is used (their ANSWERS may be memorised): the narrative is generic
and the parameters are procedurally varied — six invented speaker names and a clue set found by
randomised search over a clue-template pool (kinds: immediately-before / not-consecutive /
numeric-gap / excluded-slot / end-slot), keeping only sets that are (a) satisfied by exactly ONE
of the 6! = 720 orderings and (b) MINIMAL — deleting any single clue leaves the ordering
under-determined (>1 solution), so every clue must actually be used.

Ground truth (reference-solver verified, never hand-derived)
-----------------------------------------------------------
    Slot 1: Fennick  Slot 2: Verity  Slot 3: Kolbein
    Slot 4: Rasmus   Slot 5: Jules   Slot 6: Ondine
Solver A (``_solve_bruteforce`` below, in this module): exhaustive scan of all 720 orderings
against the CLUES predicates -> exactly 1 survivor.
Solver B (``_solve_by_backtracking`` in
``services/agent/tests/test_203_reasoning_slot_ordering_validators_test.py``): an independent
implementation — slot-by-slot DFS whose clue predicates were re-written by hand FROM THE ENGLISH
STATEMENT (not from this module's spec), so a prose/predicate mis-transcription cannot hide.
Both solvers plus the uniqueness and per-clue minimality assertions are re-runnable there.

Discrimination: the keystone demands the WHOLE ordering (1/720 by guessing) while the un-gated
``position_coverage`` / ``clue_consistency`` diagnostics report how far a failed run actually
got — a random ordering places exactly 1 speaker correctly in expectation (~0.17 coverage).
"""

from typing import Dict, Any, List
import re
from itertools import permutations
from agent.app.idea_test_utils import extract_final_text


# --- puzzle parameters (single source of truth for statement, solvers and validators) -------
N_SLOTS = 6
# Roster as PRESENTED (alphabetical) — not the solution order.
ENTITIES: List[str] = ["Fennick", "Jules", "Kolbein", "Ondine", "Rasmus", "Verity"]

# The unique ordering, index i -> slot i+1. Verified by the two reference solvers (see above).
SOLUTION: List[str] = ["Fennick", "Verity", "Kolbein", "Rasmus", "Jules", "Ondine"]

# Clues: prose + predicate over ``pos`` (name -> slot number, 1-based). The task statement is
# BUILT from these strings, so prose and predicate cannot drift apart.
CLUES: List[Dict[str, Any]] = [
    {"text": "Fennick speaks either in the first slot or in the last slot.",
     "check": lambda pos: pos["Fennick"] in (1, N_SLOTS)},
    {"text": "Kolbein does not speak in slot 6.",
     "check": lambda pos: pos["Kolbein"] != 6},
    {"text": "Fennick speaks immediately before Verity (Fennick's slot number is exactly one "
             "less than Verity's).",
     "check": lambda pos: pos["Fennick"] + 1 == pos["Verity"]},
    {"text": "Kolbein's slot number and Ondine's slot number differ by exactly 3.",
     "check": lambda pos: abs(pos["Kolbein"] - pos["Ondine"]) == 3},
    {"text": "Ondine and Rasmus do not speak in consecutive slots (their slot numbers differ by "
             "more than 1).",
     "check": lambda pos: abs(pos["Ondine"] - pos["Rasmus"]) != 1},
]

# Secondary deliverable: the sum of the slot numbers of these three speakers.
SUM_SPEAKERS: List[str] = ["Kolbein", "Rasmus", "Ondine"]
SLOT_SUM = sum(SOLUTION.index(name) + 1 for name in SUM_SPEAKERS)     # -> 13


def _solve_bruteforce(clues: List[Dict[str, Any]] = None) -> List[List[str]]:
    """Reference solver A: exhaustive scan of every ordering of the roster.

    :param clues: Clue list to satisfy (defaults to the task's own ``CLUES``).
    :return: Every ordering (list indexed slot-1) satisfying all clues.
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
        "test_id": "203",
        "test_name": "Reasoning: 6-slot constraint satisfaction (unique running order)",
        "difficulty_level": "7/10",
        "category": "Self-contained Constraint Satisfaction",
        # Outside the web ladder's LEVEL_ORDER on purpose: a no-grounding task must not be
        # averaged into the micro/integration/navigation/graph grounding ladder.
        "level": "reasoning",
        "weight": "long",
        "grounding_required": False,
    }


def get_task_statement() -> str:
    clue_lines = "\n".join(f"  ({i}) {c['text']}" for i, c in enumerate(CLUES, 1))
    roster = ", ".join(ENTITIES)
    trio = ", ".join(SUM_SPEAKERS[:-1]) + " and " + SUM_SPEAKERS[-1]
    return (
        "This is a pure reasoning task. Do NOT search the web and do NOT open any page — every "
        "fact you need is written below, and no external source contains this scenario.\n\n"
        f"A one-afternoon workshop has {N_SLOTS} speaking slots, numbered 1 to {N_SLOTS} in the "
        f"order they happen (slot 1 is the earliest talk, slot {N_SLOTS} the latest). Exactly one "
        f"speaker fills each slot, and each speaker speaks exactly once. The {N_SLOTS} speakers "
        f"are: {roster}.\n\n"
        "The organiser's notes record these facts:\n"
        f"{clue_lines}\n\n"
        f"Exactly one running order of the {N_SLOTS} speakers satisfies all of the facts at "
        "once. Work it out.\n\n"
        "Report BOTH of the following:\n"
        f"(a) The full running order, as {N_SLOTS} separate lines in exactly this form:\n"
        "    Slot 1: <speaker>\n"
        "    Slot 2: <speaker>\n"
        f"    ... and so on through Slot {N_SLOTS}.\n"
        f"(b) The check total: add together the slot numbers of {trio}, and state that sum."
    )


def get_required_deliverables() -> List[str]:
    return [
        f"The speaker in each of slots 1-{N_SLOTS} (the full unique running order)",
        f"The check total: the sum of the slot numbers of {', '.join(SUM_SPEAKERS)}",
    ]


def get_success_criteria() -> List[str]:
    return [
        "All six slots filled with the correct speaker (the unique satisfying ordering)",
        "The reported ordering violates none of the five stated facts",
        "The check total is reported correctly",
        "No web sources used or needed (self-contained reasoning)",
    ]


# --- answer parsing -------------------------------------------------------------------------
# Layout-tolerant, priority-tiered extraction of the reported ordering. Tiers are tried in order
# and only fill slots still missing, so an explicit "Slot 3: Kolbein" answer always outranks a
# looser signal (e.g. a numbered restatement of the CLUES — exactly the false positive that
# would otherwise mark a CORRECT answer wrong).
_NAME_ALT = "|".join(sorted(ENTITIES, key=len, reverse=True))
_SLOT_WORD = r"(?:slot|position|place|talk)"
# Horizontal whitespace only: a label and its name must sit on the SAME line, otherwise
# "Slot 1: Fennick\nVerity ..." would let slot 1 swallow the next row's name.
_H = r"[^\S\n]"
# Junk allowed between a slot label and its name: spaces, separators, markdown/JSON decoration.
# Excludes newline, comma, "." and all letters/digits, so an intervening WORD (as in a restated
# fact, "... in slot 6. Ondine is ...") can never glue a label to a name.
_SEP = r"[ \t:=\-–—>\)\]\|\*_`\"'~]*"
# Optional prefix junk on the number itself: "slot_1", "slot #1", "**slot** 1".
_NUMPRE = r"[ \t_\-\*`\"']*#?[ \t]*"
# Lines that are restated facts / roster echoes, never answer rows.
_CLUEISH = re.compile(
    r"immediately|consecutive|adjacent|differ|does not|do not|neither|either|"
    r"before|after|roster|speakers are|clue|fact\s*\(",
    re.IGNORECASE,
)


def _scan_labeled_forward(text: str) -> Dict[int, str]:
    """Tier 1 — "Slot 3: Kolbein", "slot #3 - Kolbein", "**Slot 3:** Kolbein",
    "| Slot 3 | Kolbein |", '"slot_3": "Kolbein"'."""
    out: Dict[int, str] = {}
    for i in range(1, N_SLOTS + 1):
        rx = re.compile(
            rf"{_SLOT_WORD}{_NUMPRE}{i}\b{_SEP}"
            rf"(?:is{_H}+|has{_H}+|goes{_H}+to{_H}+)?{_SEP}({_NAME_ALT})\b",
            re.IGNORECASE,
        )
        hits = rx.findall(text)
        if hits:
            out[i] = hits[-1].capitalize()
    return out


def _scan_labeled_reverse(text: str) -> Dict[int, str]:
    """Tier 2 — "Kolbein: slot 3", "Kolbein speaks in slot 3", "Kolbein (slot 3)".

    The connector alternation admits no negation, so a restated fact such as "Kolbein does not
    speak in slot 6" cannot register as an assignment.
    """
    out: Dict[int, str] = {}
    rx = re.compile(
        rf"\b({_NAME_ALT})\b[ \t,:=\-–—>\(\[\*_`\"'~]*"
        rf"(?:is{_H}+|speaks{_H}+|presents{_H}+|talks{_H}+|occupies{_H}+|fills{_H}+)?"
        rf"(?:in{_H}+|at{_H}+)?[ \t,:=\-–—>\(\[\*_`\"'~]*{_SLOT_WORD}{_NUMPRE}(\d+)\b",
        re.IGNORECASE,
    )
    for name, num in rx.findall(text):
        n = int(num)
        if 1 <= n <= N_SLOTS:
            out[n] = name.capitalize()
    return out


def _scan_bare_numbered(text: str) -> Dict[int, str]:
    """Tier 3 — bare numbered/table rows: "3. Kolbein", "3) Kolbein", "| 3 | Kolbein |",
    "- **3.** Kolbein"."""
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
            if 1 <= n <= N_SLOTS:
                out[n] = m.group(2).capitalize()
    return out


def _scan_ordered_line(text: str) -> Dict[int, str]:
    """Tier 4 — a single line naming every speaker exactly once, read left to right."""
    best: Dict[int, str] = {}
    for line in text.splitlines():
        if _CLUEISH.search(line) or re.search(r"\d", line):
            continue
        names = [m.group(0).capitalize()
                 for m in re.finditer(rf"\b({_NAME_ALT})\b", line, re.IGNORECASE)]
        if len(names) == N_SLOTS and set(names) == set(ENTITIES):
            best = {i + 1: name for i, name in enumerate(names)}
    return best


def _parse_arrangement(text: str) -> Dict[int, str]:
    """
    Extract the reported slot -> speaker mapping from free-form report text.
    :param text: The agent's final deliverable text.
    :return: Mapping of slot number to speaker name (may be partial).
    """
    mapping: Dict[int, str] = {}
    for scan in (_scan_labeled_forward, _scan_labeled_reverse,
                 _scan_bare_numbered, _scan_ordered_line):
        if len(mapping) == N_SLOTS:
            break
        for pos, name in scan(text).items():
            mapping.setdefault(pos, name)
    return mapping


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE predicate: every slot filled with the correct speaker.

    No grounding clause on purpose: the task is self-contained (``grounding_required=False``),
    so there is no page-read to gate on and ``scripts/validator_lint.py`` correctly reports a
    ``[GATE]`` finding here — the honest signal for this category, not to be silenced.
    Anti-guessing is carried by the answer space instead: all six slots must be right at once
    (1 in 720).
    """
    parsed = _parse_arrangement(extract_final_text(result))
    return all(parsed.get(i + 1) == name for i, name in enumerate(SOLUTION))


def validate_keystone_ordering(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the complete, uniquely-determined running order."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_ordering", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Correct unique running order " +
                       ", ".join(f"slot {i + 1}={n}" for i, n in enumerate(SOLUTION)))
                      if passed else "Running order missing or does not match the unique solution"}


def validate_position_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Un-gated diagnostic: how many slots were filled correctly (partial credit).

    Deliberately NOT short-circuited on the keystone — it measures how far a failed run got.
    A random ordering places 1 speaker correctly in expectation (~0.17), so partial credit here
    can never masquerade as a solve.
    """
    parsed = _parse_arrangement(extract_final_text(result))
    correct = sum(1 for i, name in enumerate(SOLUTION) if parsed.get(i + 1) == name)
    return {"check": "position_coverage", "passed": correct == N_SLOTS, "score": correct / N_SLOTS,
            "reason": f"{correct}/{N_SLOTS} slots filled correctly "
                      f"({len(parsed)}/{N_SLOTS} slots reported at all)"}


def validate_clue_consistency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Un-gated diagnostic: how many stated facts the REPORTED ordering satisfies.

    The clue set admits exactly one ordering, so any wrong-but-complete answer must break at
    least one fact; an unplaced speaker makes its facts unevaluable (counted as violated).
    """
    parsed = _parse_arrangement(extract_final_text(result))
    pos = {name: slot for slot, name in parsed.items()}
    satisfied = 0
    broken: List[int] = []
    for idx, clue in enumerate(CLUES, 1):
        try:
            ok = bool(clue["check"](pos))
        except KeyError:            # a speaker the report never placed
            ok = False
        if ok:
            satisfied += 1
        else:
            broken.append(idx)
    n = len(CLUES)
    return {"check": "clue_consistency", "passed": satisfied == n, "score": satisfied / n,
            "reason": f"{satisfied}/{n} stated facts satisfied by the reported ordering"
                      + (f" (violates {broken})" if broken else "")}


# The check total is outside this puzzle's natural number vocabulary (slot numbers run 1-6,
# there are 6 speakers and 5 facts), so a bare occurrence is a reliable signal — and it is only
# ever consulted once the full ordering is already correct (keystone gate below).
_SLOT_SUM_RE = re.compile(r"\b" + str(SLOT_SUM) + r"\b")


def validate_check_total(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary (SHORT-CIRCUITS TO 0 without the keystone): the summed slot numbers.

    Gating on the keystone keeps the score bimodal: a run that never solved the ordering cannot
    bank a consolation point for a total it could only have guessed.
    """
    if not _keystone_ok(result, observability):
        return {"check": "check_total", "passed": False, "score": 0.0,
                "reason": "Ordering keystone absent -> check total not credited"}
    ok = bool(_SLOT_SUM_RE.search(extract_final_text(result)))
    return {"check": "check_total", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": f"Check total {SLOT_SUM} reported" if ok
                      else f"Check total {SLOT_SUM} missing/incorrect"}


def get_validation_functions() -> List[callable]:
    return [validate_keystone_ordering, validate_position_coverage,
            validate_clue_consistency, validate_check_total]


def get_llm_validation_function() -> callable:
    return None
