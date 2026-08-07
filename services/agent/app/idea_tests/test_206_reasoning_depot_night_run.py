"""
Test 206: Self-contained reasoning — 5-rule deductive chain (parcel depot / night run)
Level: micro   Weight: short   Difficulty: 6/10   grounding_required: False

NO web access, no page visit, no entity lookup: the mandate contains every premise needed.
This is the "general reasoning" counterpart to the web-grounded suite — instead of a
``visit.count > 0`` grounding gate it is held honest by (a) SELF-CONTAINMENT (the five rules
below are the entire input) and (b) ANSWER-SPACE NOVELTY (a generic, invented depot scenario,
never a recognizable named logic puzzle whose *answer* could be memorized).

THE PUZZLE (propositional encoding, one parcel):
    D = destined for the Ashwold district     V = loaded onto the Larkhill van
    C = carries a cold-chain sticker          N = put on the night run
    H = flagged heavy                         M = missed the 18:00 cutoff

    rule 1:  D -> V
    rule 2:  V -> ~C
    rule 3:  M -> N          <- DECOY: consistent, true, and load-bearing for NOTHING
    rule 4:  N -> (C or H)
    rule 5:  H -> ~N
    given:   D            question: N?

FORCED ANSWER: **NO** (parcel 812 is NOT put on the night run), by a four-link chain:
    D and rule 1        => V              (modus ponens)
    V and rule 2        => ~C             (modus ponens)
    assume N; rule 4    => C or H; with ~C => H
    H and rule 5        => ~N             — contradiction, so ~N.       (modus tollens x2)

WHY IT DISCRIMINATES
  * It is NOT answerable from any single rule — verified exhaustively: no rule, no PAIR of
    rules and no TRIPLE of rules entails ~N; the unique minimal entailing set is exactly
    {1, 2, 4, 5}. A model that pattern-matches one salient rule cannot land the answer.
  * Rule 3 is a deliberate attractor for the two commonest failure modes: answering "YES"
    (a parcel *could* have missed the cutoff) or hedging to "cannot be determined" (we are
    never told whether it did). Neither is correct — the rules jointly force ~N, and hence
    ~M as a by-product.
  * Whether the parcel is flagged heavy (H) is genuinely OPEN under the rules, so the
    validators deliberately never punish an answer for saying so. Only the queried
    proposition is forced.

GROUND TRUTH PROVENANCE (this category's analog of "verified against live <source>")
  Solver A — ``reference_solve()`` below: exhaustive enumeration of all 2^6 truth assignments,
  keeping the models that satisfy every rule with D true, then checking N is False in all of
  them, that at least one such model exists (the rule set is not vacuous), and which rules are
  load-bearing (drop-one counterexample search).
  Solver B — an INDEPENDENTLY IMPLEMENTED CNF refutation solver in
  ``services/agent/tests/test_206_reasoning_depot_night_run_validators_test.py``: it proves
  entailment the other way round, by showing {rules} + D + N is UNSAT while {rules} + D + ~N is
  SAT. That test asserts A and B agree, that ANSWER/DRIVER_RULES below match what the solvers
  compute, and re-derives all of it on every offline test run — so the ground truth here is
  machine-checked in perpetuity rather than hand-asserted once.
  Both solvers run and agree: ANSWER = "no", DRIVER_RULES = (1, 2, 4, 5), decoy = (3,).
"""

from typing import Any, Dict, List, Optional, Sequence, Set
import itertools
import re

from agent.app.idea_test_utils import extract_final_text


# --- The puzzle: single source of truth for the statement, the solver and the validators ---
RULES: List[Dict[str, Any]] = [
    {"n": 1, "text": "Every parcel addressed to the Ashwold district is loaded onto the "
                     "Larkhill van.", "encoding": "D -> V"},
    {"n": 2, "text": "No parcel loaded onto the Larkhill van carries a cold-chain sticker.",
     "encoding": "V -> not C"},
    {"n": 3, "text": "Every parcel that misses the 18:00 cutoff is put on the night run.",
     "encoding": "M -> N"},
    {"n": 4, "text": "Every parcel put on the night run either carries a cold-chain sticker "
                     "or is flagged heavy (or both).", "encoding": "N -> (C or H)"},
    {"n": 5, "text": "No parcel flagged heavy is put on the night run.", "encoding": "H -> not N"},
]

ANSWER = "no"                       # keystone: parcel 812 is NOT put on the night run
DRIVER_RULES = (1, 2, 4, 5)         # the unique minimal set of rules that forces ANSWER
DECOY_RULES = (3,)                  # true, consistent, and needed for nothing


# ---------------------------------------------------------------------------
# Reference solver A (exhaustive model enumeration) — the ground truth generator.
# ---------------------------------------------------------------------------
_ATOMS = ("D", "V", "C", "N", "H", "M")
_RULE_PREDICATES = {
    1: lambda a: (not a["D"]) or a["V"],
    2: lambda a: (not a["V"]) or (not a["C"]),
    3: lambda a: (not a["M"]) or a["N"],
    4: lambda a: (not a["N"]) or a["C"] or a["H"],
    5: lambda a: (not a["H"]) or (not a["N"]),
}


def _models(active: Sequence[int]) -> List[Dict[str, bool]]:
    """All truth assignments with D true that satisfy every rule in ``active``."""
    out: List[Dict[str, bool]] = []
    for bits in itertools.product((False, True), repeat=len(_ATOMS)):
        a = dict(zip(_ATOMS, bits))
        if a["D"] and all(_RULE_PREDICATES[r](a) for r in active):
            out.append(a)
    return out


def reference_solve() -> Dict[str, Any]:
    """
    Brute-force the puzzle from the rule encodings — never hand-derived.
    :return: ``{"answer", "drivers", "n_models", "open_atoms", "minimal_entailing_sets"}``
             where ``answer`` is "no" iff N is False in EVERY model (i.e. the rules force it),
             and ``drivers`` are the rules whose removal admits a model with N true.
    """
    allr = [r["n"] for r in RULES]
    ms = _models(allr)
    forced_no = bool(ms) and all(not m["N"] for m in ms)
    drivers = tuple(r for r in allr if any(m["N"] for m in _models([x for x in allr if x != r])))
    entailing = [set(s) for k in range(len(allr) + 1) for s in itertools.combinations(allr, k)
                 if _models(list(s)) and all(not m["N"] for m in _models(list(s)))]
    minimal = [sorted(s) for s in entailing if not any(t < s for t in entailing)]
    return {
        "answer": "no" if forced_no else ("yes" if ms and all(m["N"] for m in ms) else "undetermined"),
        "drivers": drivers,
        "n_models": len(ms),
        "open_atoms": tuple(at for at in _ATOMS if len({m[at] for m in ms}) > 1),
        "minimal_entailing_sets": minimal,
    }


# ---------------------------------------------------------------------------
# Answer parsing.
# ---------------------------------------------------------------------------
# The mandate asks for a literal ``ANSWER: YES`` / ``ANSWER: NO`` line, so the primary parse is
# an explicit marker. Two deliberate properties:
#   * MULTIPLE CONFLICTING markers ("ANSWER: NO ... final answer: yes") parse to None, i.e. the
#     keystone fails — hedging both ways must never be scored as correct.
#   * ``\b`` after the verdict is what stops "the answer is not determined" from parsing as "no".
_ANSWER_MARKER_RX = re.compile(
    r"\b(?:final\s+|the\s+)?answers?\s*(?:is|are|:|=|-|—)+\s*\**\s*[\[(]?\s*"
    r"(yes|no|true|false)\b",
    re.IGNORECASE,
)
# Prose fallback, used ONLY when no explicit marker is present. Both branches are anchored on the
# subject token "812" so that merely restating rule 5 ("No parcel flagged heavy is put on the
# night run") can never be mistaken for the agent's own verdict.
_ASSERTS_NO_RX = re.compile(
    r"\b812\b[^.]{0,90}\b(?:not|never)\b[^.]{0,50}\bnight\s+run\b"
    r"|\b(?:not|never)\b[^.]{0,50}\bnight\s+run\b[^.]{0,70}\b812\b",
    re.IGNORECASE,
)
# The YES branch's gap is a TEMPERED match: it may not swallow a "not"/"never", so
# "812 is not put on the night run" asserts NO only, instead of matching both directions and
# collapsing to an unparseable None.
_ASSERTS_YES_RX = re.compile(
    r"\b812\b[^.]{0,90}\b(?:is|was|will\s+be|gets|goes)\b"
    r"(?:(?!\bnot\b|\bnever\b)[^.]){0,30}\bon\s+the\s+night\s+run\b",
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
    says_no = bool(_ASSERTS_NO_RX.search(text or ""))
    says_yes = bool(_ASSERTS_YES_RX.search(text or ""))
    if says_no != says_yes:
        return "no" if says_no else "yes"
    return None


# ---------------------------------------------------------------------------
# Citation parsing ("which rules did the derivation actually use").
# ---------------------------------------------------------------------------
_RULE_WORD = r"(?:rules?|premises?|clues?|statements?|constraints?|conditions?|facts?)"
_NUM_WORD = r"(?:\d{1,2}|one|two|three|four|five|six)"
_SEP = r"(?:\s*(?:,|;|&|\+|and|or|to|through|thru|-|–|—)\s*)"
_RULE_LIST_RX = re.compile(
    rf"{_RULE_WORD}\b\s*(?:used|applied|needed|invoked|required)?\s*[:#=–—-]?\s*"
    rf"(#?{_NUM_WORD}(?:{_SEP}#?{_NUM_WORD})*)",
    re.IGNORECASE,
)
_TOKEN_RX = re.compile(rf"({_NUM_WORD})|({_SEP})", re.IGNORECASE)
_ORDINAL_RX = re.compile(
    rf"(?:the\s+)?(first|second|third|fourth|fifth|sixth)\s+{_RULE_WORD}"
    rf"|{_RULE_WORD}\s+(first|second|third|fourth|fifth|sixth)\b",
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


def _cited_rules(text: str, max_n: int = len(RULES)) -> Set[int]:
    """
    Harvest the rule numbers a report actually cites. Accepts the mandated
    ``RULES USED: 1, 2, 4, 5`` line, inline prose ("by rules 1 and 2 ..."), ranges
    ("rules 1-5", "rules 1 through 5"), spelled-out numbers and ordinals ("the second rule").
    Anchoring every harvest on a rule-word keeps stray figures in the prose (slot numbers,
    "18:00", parcel ids) from being mistaken for citations.
    :param text: The agent's final deliverable.
    :param max_n: Highest legal rule number; anything outside 1..max_n is discarded.
    :return: Set of cited rule numbers.
    """
    cited: Set[int] = set()
    for listing in _RULE_LIST_RX.findall(text or ""):
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
# Intermediate-conclusion probes (the derivation actually traversed the chain).
# ---------------------------------------------------------------------------
_STEP_VAN_RX = re.compile(r"larkhill", re.IGNORECASE)
_STEP_NO_COLD_RX = re.compile(
    r"\b(?:no|not|never|without|cannot|can't|lacks?|excluded?)\b[^.]{0,70}cold[\s-]?chain"
    r"|cold[\s-]?chain[^.]{0,70}\b(?:is\s+not|isn't|cannot|can't|ruled\s+out|excluded|impossible)\b",
    re.IGNORECASE,
)
_STEP_HEAVY_RX = re.compile(
    r"heav\w*[^.]{0,120}(?:contradict|impossib|cannot|can't|not\s+allowed|excluded|ruled\s+out|"
    r"but\s+rule|violat)"
    r"|(?:contradict|impossib|ruled\s+out|violat)\w*[^.]{0,120}heav",
    re.IGNORECASE,
)
_DERIVATION_STEPS = (
    ("on_larkhill_van", _STEP_VAN_RX),
    ("no_cold_chain_sticker", _STEP_NO_COLD_RX),
    ("heavy_branch_contradiction", _STEP_HEAVY_RX),
)

_HAS_RULES_LINE_RX = re.compile(
    rf"{_RULE_WORD}\s*(?:used|applied|needed|invoked|required)?\s*[:#=]\s*#?{_NUM_WORD}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "206",
        "test_name": "Reasoning: 5-rule deductive chain (depot night run)",
        "difficulty_level": "6/10",
        "category": "Self-contained Deductive Reasoning",
        "level": "reasoning",
        "weight": "short",
        # Self-declared exemption from the web suite's grounding gate: this task has no page to
        # visit, so visit.count is structurally 0 and must never be scored.
        "grounding_required": False,
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {r['n']}. {r['text']}" for r in RULES)
    return (
        "This is a self-contained reasoning problem. Do NOT search the web, open any page or "
        "use any outside knowledge: everything needed is written below.\n\n"
        "The Halstead parcel depot sorts every parcel it handles under five standing rules. "
        "The rules hold WITHOUT EXCEPTION for every parcel:\n"
        f"{listing}\n\n"
        "Parcel 812 is addressed to the Ashwold district. Nothing else is known about parcel 812.\n\n"
        "QUESTION: Is parcel 812 put on the night run?\n\n"
        "Report exactly three things, in this order:\n"
        "  (a) a line containing exactly 'ANSWER: YES' or 'ANSWER: NO';\n"
        "  (b) a line of the form 'RULES USED: <numbers>' listing the number of every rule your "
        "derivation actually needs (cite only the rules that do real work);\n"
        "  (c) two to five sentences of derivation, naming each intermediate conclusion you "
        "reach on the way to the answer."
    )


def get_required_deliverables() -> List[str]:
    return [
        "A YES/NO verdict on whether parcel 812 is put on the night run",
        "The numbers of the rules the derivation actually uses",
        "A short derivation naming each intermediate conclusion",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Answers NO (the rules force it: parcel 812 is not put on the night run)",
        "Cites rules 1, 2, 4 and 5 — the minimal set that forces the answer",
        "Does not rest the answer on rule 3, which is consistent but does no work",
        "States the intermediate conclusions (on the Larkhill van; no cold-chain sticker; "
        "the night run would require a heavy flag, which rule 5 forbids)",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """
    KEYSTONE predicate. Correct verdict AND a justification resting on the rules that actually
    force it. A coin-flip "NO" with no (or a wrong) justification scores 0, because ANSWER alone
    is a 1-in-2 guess: this category has no page-fetch grounding gate to lean on (see the
    module docstring), so the citation requirement IS the anti-guessing gate.

    NOTE for anyone running ``scripts/validator_lint.py``: this predicate is deliberately
    worded to avoid the lint's grounding-heuristic keywords, so the [GATE] finding this task
    SHOULD produce is reported honestly instead of being silenced by an incidental docstring
    substring match. A [GATE] finding here is expected and correct — do not "fix" it.
    """
    text = extract_final_text(result)
    if _parse_answer(text) != ANSWER:
        return False
    return set(DRIVER_RULES).issubset(_cited_rules(text))


def validate_keystone_answer(result: Dict[str, Any], observability: Dict[str, Any] = None) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): verdict NO *and* rules 1, 2, 4, 5 cited as the justification."""
    text = extract_final_text(result)
    verdict = _parse_answer(text)
    cited = _cited_rules(text)
    missing = sorted(set(DRIVER_RULES) - cited)
    passed = _keystone_ok(result, observability)
    if passed:
        reason = "ANSWER: NO justified by rules 1, 2, 4, 5"
    elif verdict != ANSWER:
        reason = f"verdict {verdict!r} != required {ANSWER!r}"
    else:
        reason = f"verdict correct but justification omits driver rule(s) {missing}"
    return {"check": "keystone_answer", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": reason}


def validate_rule_coverage(result: Dict[str, Any], observability: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    UN-gated diagnostic: what fraction of the four load-bearing rules the report cites.

    Deliberately NOT short-circuited on the keystone — it measures how much of the deductive
    chain was actually traversed even when the final verdict is botched, the same role the
    breadth "coverage" diagnostic plays in the web suite (see test_052).
    """
    cited = _cited_rules(extract_final_text(result))
    hit = sorted(set(DRIVER_RULES) & cited)
    return {"check": "rule_coverage", "passed": len(hit) == len(DRIVER_RULES),
            "score": len(hit) / len(DRIVER_RULES),
            "reason": f"{len(hit)}/{len(DRIVER_RULES)} load-bearing rules cited ({hit or 'none'})"}


def validate_derivation_steps(result: Dict[str, Any], observability: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Gated secondary: which intermediate conclusions the derivation names. Short-circuits to 0
    when the keystone is absent, so a wrong answer can never bank partial credit for reciting
    plausible-looking steps (bimodal scores, never a constant-partial trap).

    Note what is NOT probed: whether the parcel is flagged heavy is genuinely undetermined by
    the rules, so an answer that says so is correct and is never penalized here.
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
    asks for (an explicit ANSWER verdict and a rules-used listing)? Instruction-following is a
    separate axis from reasoning, so this is scored separately and never gates anything.
    """
    text = extract_final_text(result)
    has_verdict = bool(_ANSWER_MARKER_RX.search(text))
    has_rules_line = bool(_HAS_RULES_LINE_RX.search(text))
    score = 0.5 * has_verdict + 0.5 * has_rules_line
    return {"check": "answer_format", "passed": score == 1.0, "score": score,
            "reason": f"explicit verdict line: {has_verdict}; rules-used listing: {has_rules_line}"}


def get_validation_functions() -> List[callable]:
    return [validate_keystone_answer, validate_rule_coverage,
            validate_derivation_steps, validate_answer_format]


def get_llm_validation_function() -> callable:
    """No LLM judge: this category holds itself to the web suite's determinism bar."""
    return None
