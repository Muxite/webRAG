"""
Test 207: Self-contained reasoning — 6-clue ordering puzzle, necessity question (drum tests)
Level: micro   Weight: short   Difficulty: 7/10   grounding_required: False

NO web access, no page visit, no entity lookup: the mandate contains every premise needed.
Like test_206 this replaces the web suite's ``visit.count > 0`` grounding gate with
SELF-CONTAINMENT plus ANSWER-SPACE NOVELTY (an invented rope-works with invented drum names,
never a recognizable named logic puzzle whose *answer* could be memorized), and it is
deliberately a different SHAPE from 206: a finite-domain ordering puzzle answered by case
elimination, not a propositional implication chain.

THE PUZZLE
    Five drums — Kestrel (K), Tarnbeck (T), Bramling (B), Padgett (P), Sorrel (S) — occupy the
    five test slots 1..5, one drum per slot.
        clue 1:  K != 2
        clue 2:  |K - T| != 1
        clue 3:  S != 2                <- DECOY: true in every consistent order, and needed
                                          for NOTHING (it is implied by the other five)
        clue 4:  B == K + 1
        clue 5:  B != 5
        clue 6:  T not in {1, 5}
    QUESTION: must B come before S?

FORCED ANSWER: **YES**, by case elimination:
    clue 4 => K in {1,2,3,4}; clue 1 kills K=2; clue 5 (B != 5) kills K=4  => K in {1,3}.
    K=3 => B=4, so clue 2 bars T from slots 2 and 4, and clue 6 bars T from 1 and 5 —
           T has nowhere to go, so K=3 is impossible.
    Therefore K=1 and B=2. Slots 3, 4, 5 hold T, P, S in some arrangement, so S is in slot
    3, 4 or 5 — strictly after B. Hence B precedes S in EVERY consistent order.

WHY IT DISCRIMINATES
  * The full order is deliberately NOT unique — four different orders satisfy all six clues.
    An agent that tries to "solve the grid" and read the answer off a single solution has
    nothing to read off; the question is about what is NECESSARY across all four. That is the
    reasoning step cheap models most often skip (they commit to the first consistent order
    they stumble into and answer from it).
  * No proper subset of clues {1, 2, 4, 5, 6} forces the answer — verified exhaustively over
    all 2^6 clue subsets, the unique minimal entailing set is exactly those five. Dropping any
    one of them admits a counterexample order in which Sorrel precedes Bramling.
  * Clue 3 is an attractor: it is the only clue that mentions Sorrel at all, so it looks
    directly relevant to a question about Sorrel, while doing no work whatsoever.
  * Padgett's slot, and Tarnbeck's, stay genuinely open, so the validators never punish an
    answer for saying so — only the queried relation is forced.

GROUND TRUTH PROVENANCE (this category's analog of "verified against live <source>")
  Solver A — ``reference_solve()`` below: exhaustive enumeration of all 120 permutations,
  filtered by the clue predicates, checking (i) at least two consistent orders exist,
  (ii) B precedes S in every one of them, (iii) which clues are load-bearing (drop-one
  counterexample search) and (iv) the unique minimal entailing clue subset over all 2^6 subsets.
  Solver B — an INDEPENDENTLY IMPLEMENTED solver in
  ``agent/tests/test_207_reasoning_drum_test_order_validators_test.py``: a
  depth-first slot-filling search whose clues are re-typed from the natural-language text over
  a slot-ordered tuple (a different representation AND a different search strategy). That test
  asserts A and B produce the identical model set, that ANSWER/DRIVER_CLUES below match what
  the solvers compute, and re-derives all of it on every offline test run.
  Both solvers run and agree: ANSWER = "yes", DRIVER_CLUES = (1, 2, 4, 5, 6), decoy = (3,), and
  the four consistent orders are
      K B T P S | K B T S P | K B P T S | K B S T P.
"""

from typing import Any, Dict, List, Optional, Sequence, Set
import itertools
import re

from agent.app.idea_test_utils import extract_final_text


DRUMS: List[str] = ["Kestrel", "Tarnbeck", "Bramling", "Padgett", "Sorrel"]
N_SLOTS = len(DRUMS)

# --- The puzzle: single source of truth for the statement, the solver and the validators ---
CLUES: List[Dict[str, Any]] = [
    {"n": 1, "text": "Kestrel is not tested in slot 2.", "encoding": "K != 2"},
    {"n": 2, "text": "Kestrel and Tarnbeck are never tested in consecutive slots (in either "
                     "order).", "encoding": "|K - T| != 1"},
    {"n": 3, "text": "Sorrel is not tested in slot 2.", "encoding": "S != 2"},
    {"n": 4, "text": "Bramling is tested in the slot immediately after Kestrel.",
     "encoding": "B == K + 1"},
    {"n": 5, "text": "Bramling is not tested in slot 5.", "encoding": "B != 5"},
    {"n": 6, "text": "Tarnbeck is tested neither in slot 1 nor in slot 5.",
     "encoding": "T not in {1, 5}"},
]

ANSWER = "yes"                          # keystone: Bramling must be tested before Sorrel
DRIVER_CLUES = (1, 2, 4, 5, 6)          # the unique minimal set of clues that forces ANSWER
DECOY_CLUES = (3,)                      # true in every consistent order, needed for nothing


# ---------------------------------------------------------------------------
# Reference solver A (exhaustive permutation enumeration) — the ground truth generator.
# ---------------------------------------------------------------------------
_CLUE_PREDICATES = {
    1: lambda o: o["Kestrel"] != 2,
    2: lambda o: abs(o["Kestrel"] - o["Tarnbeck"]) != 1,
    3: lambda o: o["Sorrel"] != 2,
    4: lambda o: o["Bramling"] == o["Kestrel"] + 1,
    5: lambda o: o["Bramling"] != 5,
    6: lambda o: o["Tarnbeck"] not in (1, 5),
}


def _orders(active: Sequence[int]) -> List[Dict[str, int]]:
    """All {drum: slot} assignments satisfying every clue in ``active``."""
    out: List[Dict[str, int]] = []
    for perm in itertools.permutations(DRUMS):
        o = {name: i + 1 for i, name in enumerate(perm)}
        if all(_CLUE_PREDICATES[c](o) for c in active):
            out.append(o)
    return out


def _bramling_before_sorrel(o: Dict[str, int]) -> bool:
    return o["Bramling"] < o["Sorrel"]


def reference_solve() -> Dict[str, Any]:
    """
    Brute-force the puzzle from the clue encodings — never hand-derived.
    :return: ``{"answer", "drivers", "orders", "open_drums", "minimal_entailing_sets"}``;
             ``answer`` is "yes" iff Bramling precedes Sorrel in EVERY consistent order,
             ``drivers`` are the clues whose removal admits a counterexample order.
    """
    allc = [c["n"] for c in CLUES]
    orders = _orders(allc)
    forced_yes = bool(orders) and all(_bramling_before_sorrel(o) for o in orders)
    forced_no = bool(orders) and not any(_bramling_before_sorrel(o) for o in orders)
    drivers = tuple(c for c in allc
                    if not all(_bramling_before_sorrel(o) for o in _orders([x for x in allc if x != c])))
    entailing = [set(s) for k in range(len(allc) + 1) for s in itertools.combinations(allc, k)
                 if _orders(list(s)) and all(_bramling_before_sorrel(o) for o in _orders(list(s)))]
    minimal = [sorted(s) for s in entailing if not any(t < s for t in entailing)]
    return {
        "answer": "yes" if forced_yes else ("no" if forced_no else "undetermined"),
        "drivers": drivers,
        "orders": [" ".join(sorted(o, key=o.get)) for o in orders],
        "open_drums": tuple(d for d in DRUMS if len({o[d] for o in orders}) > 1),
        "minimal_entailing_sets": minimal,
    }


# ---------------------------------------------------------------------------
# Answer parsing.
# ---------------------------------------------------------------------------
# Primary parse is the mandated ``ANSWER: YES`` / ``ANSWER: NO`` line. Conflicting markers parse
# to None (hedging both ways is never correct); ``\b`` after the verdict keeps "the answer is
# not settled" from parsing as "no".
_ANSWER_MARKER_RX = re.compile(
    r"\b(?:final\s+|the\s+)?answers?\s*(?:is|are|:|=|-|—)+\s*\**\s*[\[(]?\s*"
    r"(yes|no|true|false)\b",
    re.IGNORECASE,
)
# Prose fallback, used ONLY when no explicit marker is present. A report that asserts BOTH
# directions (e.g. echoes the question and then contradicts it) parses to None rather than
# being charitably resolved.
_ASSERTS_YES_RX = re.compile(
    r"bramling[^.]{0,60}\b(?:must|always|necessarily|is\s+necessarily|is\s+always)\b"
    r"[^.]{0,60}before[^.]{0,40}sorrel"
    r"|bramling[^.]{0,60}before[^.]{0,40}sorrel[^.]{0,60}\bin\s+(?:every|all|each)\b",
    re.IGNORECASE,
)
_ASSERTS_NO_RX = re.compile(
    r"bramling[^.]{0,80}\b(?:need\s+not|does\s+not\s+have\s+to|not\s+necessarily|is\s+not\s+"
    r"necessarily|could\s+come\s+after|could\s+be\s+tested\s+after)\b[^.]{0,60}sorrel"
    r"|sorrel[^.]{0,60}\b(?:could|might|may|can)\b[^.]{0,60}before[^.]{0,40}bramling",
    re.IGNORECASE,
)


def _parse_answer(text: str) -> Optional[str]:
    """
    :param text: The agent's final deliverable.
    :return: "yes" / "no", or None when absent, unparseable or self-contradictory.
    """
    verdicts = {"yes" if v.lower() in ("yes", "true") else "no"
                for v in _ANSWER_MARKER_RX.findall(text or "")}
    if len(verdicts) == 1:
        return verdicts.pop()
    if verdicts:
        return None                      # contradicts itself -> no credit
    says_yes = bool(_ASSERTS_YES_RX.search(text or ""))
    says_no = bool(_ASSERTS_NO_RX.search(text or ""))
    if says_yes != says_no:
        return "yes" if says_yes else "no"
    return None


# ---------------------------------------------------------------------------
# Citation parsing ("which clues did the derivation actually use").
# ---------------------------------------------------------------------------
_CLUE_WORD = r"(?:clues?|facts?|rules?|premises?|statements?|constraints?|conditions?)"
_NUM_WORD = r"(?:\d{1,2}|one|two|three|four|five|six)"
_SEP = r"(?:\s*(?:,|;|&|\+|and|or|to|through|thru|-|–|—)\s*)"
_CLUE_LIST_RX = re.compile(
    rf"{_CLUE_WORD}\b\s*(?:used|applied|needed|invoked|required)?\s*[:#=–—-]?\s*"
    rf"(#?{_NUM_WORD}(?:{_SEP}#?{_NUM_WORD})*)",
    re.IGNORECASE,
)
_TOKEN_RX = re.compile(rf"({_NUM_WORD})|({_SEP})", re.IGNORECASE)
_ORDINAL_RX = re.compile(
    rf"(?:the\s+)?(first|second|third|fourth|fifth|sixth)\s+{_CLUE_WORD}"
    rf"|{_CLUE_WORD}\s+(first|second|third|fourth|fifth|sixth)\b",
    re.IGNORECASE,
)
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
_RANGE_SEP_RX = re.compile(r"to|through|thru|-|–|—", re.IGNORECASE)


def _as_int(tok: str) -> Optional[int]:
    tok = tok.strip().lstrip("#")
    if tok.isdigit():
        return int(tok)
    return _WORD_NUM.get(tok.lower())


def _cited_clues(text: str, max_n: int = len(CLUES)) -> Set[int]:
    """
    Harvest the clue numbers a report actually cites. Accepts the mandated
    ``FACTS USED: 1, 2, 4, 5, 6`` line, inline prose ("by clues 4 and 5 ..."), ranges
    ("clues 4-6", "facts 1 through 6"), spelled-out numbers and ordinals ("the second clue").
    Every harvest is anchored on a clue-word so that the SLOT numbers this puzzle is full of
    ("slot 2", "slots 3, 4 and 5") are never mistaken for citations.
    :param text: The agent's final deliverable.
    :param max_n: Highest legal clue number; anything outside 1..max_n is discarded.
    :return: Set of cited clue numbers.
    """
    cited: Set[int] = set()
    for listing in _CLUE_LIST_RX.findall(text or ""):
        prev: Optional[int] = None
        pending_range = False
        for num_tok, sep_tok in _TOKEN_RX.findall(listing):
            if sep_tok:
                pending_range = bool(_RANGE_SEP_RX.fullmatch(sep_tok.strip()))
                continue
            val = _as_int(num_tok)
            if val is None:
                continue
            cited.add(val)
            if pending_range and prev is not None:
                cited.update(range(min(prev, val), max(prev, val) + 1))
            prev, pending_range = val, False
    for a, b in _ORDINAL_RX.findall(text or ""):
        val = _as_int(a or b)
        if val is not None:
            cited.add(val)
    return {c for c in cited if 1 <= c <= max_n}


# ---------------------------------------------------------------------------
# Intermediate-conclusion probes (the derivation actually performed the elimination).
# ---------------------------------------------------------------------------
_STEP_KESTREL_FIRST_RX = re.compile(
    r"kestrel[^.]{0,50}(?:slot|position)\s*1\b|kestrel[^.]{0,40}\bfirst\b"
    r"|(?:slot|position)\s*1[^.]{0,40}kestrel|\bfirst\b[^.]{0,30}kestrel",
    re.IGNORECASE,
)
_STEP_BRAMLING_SECOND_RX = re.compile(
    r"bramling[^.]{0,50}(?:slot|position)\s*2\b|bramling[^.]{0,40}\bsecond\b"
    r"|(?:slot|position)\s*2[^.]{0,40}bramling|\bsecond\b[^.]{0,30}bramling",
    re.IGNORECASE,
)
_STEP_K3_ELIMINATED_RX = re.compile(
    r"tarnbeck[^.]{0,140}(?:nowhere|no\s+(?:legal|valid|available|remaining|free|possible)?\s*"
    r"(?:slot|position|place)|cannot\s+be\s+placed|can't\s+be\s+placed|no\s+place)"
    r"|(?:slot|position)\s*3[^.]{0,140}(?:impossib|ruled\s+out|eliminat|contradict|fails|"
    r"cannot|can't)"
    r"|(?:impossib|ruled\s+out|eliminat|contradict)\w*[^.]{0,140}(?:slot|position)\s*3",
    re.IGNORECASE,
)
_DERIVATION_STEPS = (
    ("kestrel_in_slot_1", _STEP_KESTREL_FIRST_RX),
    ("bramling_in_slot_2", _STEP_BRAMLING_SECOND_RX),
    ("kestrel_slot_3_eliminated", _STEP_K3_ELIMINATED_RX),
)

_HAS_CLUES_LINE_RX = re.compile(
    rf"{_CLUE_WORD}\s*(?:used|applied|needed|invoked|required)?\s*[:#=]\s*#?{_NUM_WORD}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "207",
        "test_name": "Reasoning: 6-clue ordering puzzle, necessity question (drum tests)",
        "difficulty_level": "7/10",
        "category": "Self-contained Deductive Reasoning",
        "level": "reasoning",
        "weight": "short",
        # Self-declared exemption from the web suite's grounding gate: this task has no page to
        # visit, so visit.count is structurally 0 and must never be scored.
        "grounding_required": False,
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {c['n']}. {c['text']}" for c in CLUES)
    return (
        "This is a self-contained reasoning problem. Do NOT search the web, open any page or "
        "use any outside knowledge: everything needed is written below.\n\n"
        "The Wenlock rope-works tests five cable drums — Kestrel, Tarnbeck, Bramling, Padgett and "
        "Sorrel — in five test slots numbered 1 to 5. Each drum is tested in exactly one slot "
        "and each slot holds exactly one drum. All six of the following are true:\n"
        f"{listing}\n\n"
        "These six facts do NOT pin down the full order: more than one arrangement of the five "
        "drums satisfies all of them. Answer only the question asked.\n\n"
        "QUESTION: Must Bramling be tested before Sorrel — that is, is Bramling's slot number "
        "lower than Sorrel's in EVERY arrangement consistent with facts 1-6?\n\n"
        "Report exactly three things, in this order:\n"
        "  (a) a line containing exactly 'ANSWER: YES' or 'ANSWER: NO';\n"
        "  (b) a line of the form 'FACTS USED: <numbers>' listing the number of every fact your "
        "derivation actually needs (cite only the facts that do real work);\n"
        "  (c) two to six sentences of derivation, naming each intermediate conclusion you "
        "reach on the way to the answer."
    )


def get_required_deliverables() -> List[str]:
    return [
        "A YES/NO verdict on whether Bramling must be tested before Sorrel",
        "The numbers of the facts the derivation actually uses",
        "A short derivation naming each intermediate conclusion",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Answers YES (Bramling precedes Sorrel in every consistent arrangement)",
        "Cites facts 1, 2, 4, 5 and 6 — the minimal set that forces the answer",
        "Does not rest the answer on fact 3, which is true but does no work",
        "States the intermediate conclusions (Kestrel in slot 1, Bramling in slot 2, and the "
        "elimination of Kestrel-in-slot-3 because Tarnbeck would have nowhere to go)",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """
    KEYSTONE predicate. Correct verdict AND a justification resting on the clues that actually
    force it. A coin-flip "YES" with no (or a wrong) justification scores 0: with no page-fetch
    grounding gate available to this category (see the module docstring), the citation
    requirement IS the anti-guessing gate on a 1-in-2 answer space.

    NOTE for anyone running ``scripts/validator_lint.py``: this predicate is deliberately
    worded to avoid the lint's grounding-heuristic keywords, so the [GATE] finding this task
    SHOULD produce is reported honestly instead of being silenced by an incidental docstring
    substring match. A [GATE] finding here is expected and correct — do not "fix" it.
    """
    text = extract_final_text(result)
    if _parse_answer(text) != ANSWER:
        return False
    return set(DRIVER_CLUES).issubset(_cited_clues(text))


def validate_keystone_answer(result: Dict[str, Any], observability: Dict[str, Any] = None) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): verdict YES *and* facts 1, 2, 4, 5, 6 cited as the justification."""
    text = extract_final_text(result)
    verdict = _parse_answer(text)
    cited = _cited_clues(text)
    missing = sorted(set(DRIVER_CLUES) - cited)
    passed = _keystone_ok(result, observability)
    if passed:
        reason = "ANSWER: YES justified by facts 1, 2, 4, 5, 6"
    elif verdict != ANSWER:
        reason = f"verdict {verdict!r} != required {ANSWER!r}"
    else:
        reason = f"verdict correct but justification omits driver fact(s) {missing}"
    return {"check": "keystone_answer", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": reason}


def validate_clue_coverage(result: Dict[str, Any], observability: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    UN-gated diagnostic: what fraction of the five load-bearing facts the report cites.

    Deliberately NOT short-circuited on the keystone — it measures how much of the elimination
    was actually carried out even when the final verdict is botched, the same role the breadth
    "coverage" diagnostic plays in the web suite (see test_052).
    """
    cited = _cited_clues(extract_final_text(result))
    hit = sorted(set(DRIVER_CLUES) & cited)
    return {"check": "clue_coverage", "passed": len(hit) == len(DRIVER_CLUES),
            "score": len(hit) / len(DRIVER_CLUES),
            "reason": f"{len(hit)}/{len(DRIVER_CLUES)} load-bearing facts cited ({hit or 'none'})"}


def validate_derivation_steps(result: Dict[str, Any], observability: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Gated secondary: which intermediate conclusions the derivation names. Short-circuits to 0
    when the keystone is absent, so a wrong answer can never bank partial credit for reciting
    plausible-looking steps (bimodal scores, never a constant-partial trap).

    Note what is NOT probed: Tarnbeck's, Padgett's and Sorrel's exact slots are genuinely open
    under the six facts, so an answer that says so is correct and is never penalized here.
    """
    if not _keystone_ok(result, observability):
        return {"check": "derivation_steps", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> derivation steps not credited"}
    text = extract_final_text(result)
    hits = [name for name, rx in _DERIVATION_STEPS if rx.search(text)]
    n = len(_DERIVATION_STEPS)
    return {"check": "derivation_steps", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} intermediate conclusions stated ({', '.join(hits) or 'none'})"}


def validate_answer_format(result: Dict[str, Any], observability: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    UN-gated protocol diagnostic: did the report emit the two machine-readable lines the mandate
    asks for (an explicit ANSWER verdict and a facts-used listing)? Instruction-following is a
    separate axis from reasoning, so this is scored separately and never gates anything.
    """
    text = extract_final_text(result)
    has_verdict = bool(_ANSWER_MARKER_RX.search(text))
    has_clues_line = bool(_HAS_CLUES_LINE_RX.search(text))
    score = 0.5 * has_verdict + 0.5 * has_clues_line
    return {"check": "answer_format", "passed": score == 1.0, "score": score,
            "reason": f"explicit verdict line: {has_verdict}; facts-used listing: {has_clues_line}"}


def get_validation_functions() -> List[callable]:
    return [validate_keystone_answer, validate_clue_coverage,
            validate_derivation_steps, validate_answer_format]


def get_llm_validation_function() -> callable:
    """No LLM judge: this category holds itself to the web suite's determinism bar."""
    return None
