"""
Test 200: Self-contained reasoning — courier van load-out (0/1 knapsack over 9 parcels)
Level: reasoning   Weight: long   Difficulty: 7/10   grounding_required: False

SELF-CONTAINED: there is NO page to visit and NO fact to look up — the mandate carries the whole
input (nine parcels with a mass and a fee, plus a payload limit). This task therefore does NOT
try to satisfy the web suite's ``visit.count > 0`` grounding gate (``scripts/validator_lint.py``
will emit a ``[GATE]`` finding for the keystone here; that is EXPECTED and correct, see
``agent/tests/validator_lint_test.py::test_reasoning_suite_lints_clean_of_llm_judges``).
What replaces grounding as the validity bar is (a) self-containment and (b) ANSWER-SPACE NOVELTY:
the narrative is a deliberately generic courier-depot wrapper, never a recognizable named puzzle,
and every number below was drawn by a seeded RNG (see "Instance provenance"), so neither the setup
NOR the answer can be pattern-matched from a memorized puzzle — it has to be computed.

WHY IT DISCRIMINATES: the instance was accepted only if all three natural greedy heuristics are
strictly suboptimal, so a model that "reasons" by ranking rather than searching lands on a
specific wrong number:
  * fee-first  (take the biggest fees that still fit)      -> 474
  * lightest-first (cram in the most parcels)              -> 486
  * density-first (fee/kg, the textbook knapsack heuristic)-> 517   <- the strongest decoy
  * TRUE OPTIMUM (exhaustive search)                       -> 550
  The density-greedy load differs from the optimum by ONE swap (drop P5, add P6), i.e. the task is
  won by actually exploring the trade-off rather than by sorting once. The single highest-fee
  parcel (P8, 137 credits) is also a salient-number trap: it is too heavy (95 kg) to belong in the
  optimal load at all.

GROUND TRUTH — verified by TWO independently-implemented reference solvers embedded below, NOT
hand-derived:
  * ``_solve_bruteforce`` — bitmask enumeration over all 2^9 = 512 subsets, collecting every
    feasible (mass, fee) and every optimal mask.
  * ``_solve_dp``        — counting dynamic program over reachable (mass, fee) states with an
    independent recursive reconstruction of an optimal subset.
  Both agree; ``agent/tests/reasoning_200_van_load_validators_test.py`` RE-RUNS both on
  every test run and asserts they agree with each other AND with the hard-coded constants below,
  so the ground truth is machine-re-verifiable by any future reviewer.

  parcel   mass (kg)   fee (credits)   in optimal load?
    P1        68            35              -
    P2        47           135             YES
    P3        60           127             YES
    P4        93            67              -
    P5        27            63              -   (density-greedy takes this one; the optimum does not)
    P6        51            96             YES  (the swap the greedy heuristic misses)
    P7        26            75             YES
    P8        95           137              -   (highest fee on the board = the salient-number trap)
    P9        40           117             YES
  capacity = 240 kg          all-parcel totals: 507 kg / 852 credits

  KEYSTONE          = 550 credits  (= 135 + 127 + 96 + 75 + 117)
  optimal load      = {P2, P3, P6, P7, P9}, total mass 224 kg, unused capacity 16 kg
  UNIQUENESS        = exactly 1 of the 512 subsets attains 550 (so "the claimed subset sums to the
                      claimed total" is equivalent to "the claimed subset IS the optimal load")
  MARGIN            = the best feasible fee BELOW the optimum is 538, i.e. a 12-credit gap; no
                      near-miss arithmetic slip lands on 550 by accident, and 550 is not equal to
                      any single printed figure (no mass, no fee, not the capacity), so echoing the
                      table can never produce the keystone number.

Instance provenance (procedurally generated — reproduce with):
    rng = random.Random(5700)
    weights = rng.sample(range(17, 97), 9); fees = rng.sample(range(23, 141), 9); capacity = 240
  Seeds were scanned in order and the first one accepted that satisfies ALL of: 1.8*cap <= total
  mass <= 2.4*cap; a UNIQUE optimal subset; runner-up fee <= optimum - 8; optimal load of 4-6
  parcels; unused capacity in [15, 45] (>= 15 keeps every checked figure clear of the 1..9 digits
  in the parcel codes, so a code can never be misread as a figure); all three greedy heuristics
  strictly suboptimal; the highest-fee parcel EXCLUDED from the optimum; and the three checked
  figures (550 / 224 / 16) pairwise distinct and absent from the printed table.

Validators: one hard 0/1 keystone that requires BOTH the correct total fee AND a claimed parcel set
that actually sums to it within capacity (a "right number, fabricated subset" answer fails); one
gated secondary (the supporting figures) that short-circuits to 0 without the keystone; and two
UN-gated diagnostics (is the claimed load even feasible / how much fee did it actually capture)
that stay informative when the answer is wrong. No LLM judge — same determinism bar as the web
suite.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the instance (single source of truth for the statement, the validators and the solvers) ---
PARCELS: List[Dict[str, Any]] = [
    {"code": "P1", "num": 1, "mass": 68, "fee": 35},
    {"code": "P2", "num": 2, "mass": 47, "fee": 135},
    {"code": "P3", "num": 3, "mass": 60, "fee": 127},
    {"code": "P4", "num": 4, "mass": 93, "fee": 67},
    {"code": "P5", "num": 5, "mass": 27, "fee": 63},
    {"code": "P6", "num": 6, "mass": 51, "fee": 96},
    {"code": "P7", "num": 7, "mass": 26, "fee": 75},
    {"code": "P8", "num": 8, "mass": 95, "fee": 137},
    {"code": "P9", "num": 9, "mass": 40, "fee": 117},
]
CAPACITY_KG = 240

# Solver-verified ground truth (see the module docstring; re-derived in the unit tests).
OPTIMAL_FEE = 550                      # KEYSTONE
OPTIMAL_SET = frozenset({2, 3, 6, 7, 9})   # parcel numbers -> P2, P3, P6, P7, P9
OPTIMAL_MASS = 224
UNUSED_CAPACITY = CAPACITY_KG - OPTIMAL_MASS   # 16
RUNNER_UP_FEE = 538                    # best feasible fee strictly below the optimum (margin 12)
GREEDY_DENSITY_FEE = 517               # fee/kg-first heuristic  (the strongest decoy)
GREEDY_FEE_FIRST_FEE = 474             # biggest-fee-first heuristic
GREEDY_LIGHTEST_FEE = 486              # lightest-first heuristic


# ------------------------------------------------------------------ reference solver A (brute force)
def _solve_bruteforce(parcels: List[Dict[str, Any]], capacity: int) -> Dict[str, Any]:
    """REFERENCE SOLVER A — exhaustive bitmask enumeration over all 2^N subsets.

    Independent of :func:`_solve_dp` (different algorithm: explicit enumeration vs. a state DP).
    :param parcels: Item dicts with ``num``/``mass``/``fee``.
    :param capacity: Mass limit.
    :return: ``{"best_fee", "optimal_sets", "n_optimal", "runner_up", "optimal_mass"}``.
    """
    n = len(parcels)
    best_fee = -1
    optimal_sets: List[frozenset] = []
    feasible_fees: Set[int] = set()
    for mask in range(1 << n):
        mass = fee = 0
        for i in range(n):
            if mask >> i & 1:
                mass += parcels[i]["mass"]
                fee += parcels[i]["fee"]
        if mass > capacity:
            continue
        feasible_fees.add(fee)
        if fee > best_fee:
            best_fee = fee
            optimal_sets = [frozenset(parcels[i]["num"] for i in range(n) if mask >> i & 1)]
        elif fee == best_fee:
            optimal_sets.append(frozenset(parcels[i]["num"] for i in range(n) if mask >> i & 1))
    runner_up = max((f for f in feasible_fees if f < best_fee), default=None)
    by_num = {p["num"]: p for p in parcels}
    return {
        "best_fee": best_fee,
        "optimal_sets": optimal_sets,
        "n_optimal": len(optimal_sets),
        "runner_up": runner_up,
        "optimal_mass": sum(by_num[k]["mass"] for k in optimal_sets[0]) if optimal_sets else None,
    }


# ------------------------------------------------------------------- reference solver B (state DP)
def _solve_dp(parcels: List[Dict[str, Any]], capacity: int) -> Dict[str, Any]:
    """REFERENCE SOLVER B — counting DP over reachable ``(mass, fee)`` states, with an independent
    recursive reconstruction of an optimal subset. Deliberately a DIFFERENT algorithm from
    :func:`_solve_bruteforce` (incremental state expansion + counting, never materializing subsets),
    so agreement between the two is a real cross-check rather than a copy-paste tautology.

    :param parcels: Item dicts with ``num``/``mass``/``fee``.
    :param capacity: Mass limit.
    :return: ``{"best_fee", "n_optimal", "runner_up", "optimal_mass", "optimal_set"}``.
    """
    states: Dict[Tuple[int, int], int] = {(0, 0): 1}   # (mass, fee) -> number of subsets reaching it
    for p in parcels:
        nxt = dict(states)
        for (mass, fee), count in states.items():
            new_mass = mass + p["mass"]
            if new_mass > capacity:
                continue                      # infeasible states are never expanded
            key = (new_mass, fee + p["fee"])
            nxt[key] = nxt.get(key, 0) + count
        states = nxt
    best_fee = max(fee for _mass, fee in states)
    n_optimal = sum(c for (_m, fee), c in states.items() if fee == best_fee)
    runner_up = max((fee for _m, fee in states if fee < best_fee), default=None)
    optimal_mass = min(mass for mass, fee in states if fee == best_fee)

    def _reconstruct(idx: int, mass_left: int, fee_left: int) -> Optional[List[int]]:
        """Depth-first recovery of one subset with the target fee that fits in the mass left."""
        if fee_left == 0:
            return []
        if idx >= len(parcels):
            return None
        p = parcels[idx]
        if p["fee"] <= fee_left and p["mass"] <= mass_left:
            tail = _reconstruct(idx + 1, mass_left - p["mass"], fee_left - p["fee"])
            if tail is not None:
                return [p["num"]] + tail
        return _reconstruct(idx + 1, mass_left, fee_left)

    recovered = _reconstruct(0, capacity, best_fee)
    return {
        "best_fee": best_fee,
        "n_optimal": n_optimal,
        "runner_up": runner_up,
        "optimal_mass": optimal_mass,
        "optimal_set": frozenset(recovered) if recovered is not None else None,
    }


def _greedy_fee(parcels: List[Dict[str, Any]], capacity: int, key) -> int:
    """Fee captured by a single-sort greedy pass (the heuristics this instance is built to defeat)."""
    mass = fee = 0
    for p in sorted(parcels, key=key):
        if mass + p["mass"] <= capacity:
            mass += p["mass"]
            fee += p["fee"]
    return fee


# ------------------------------------------------------------------------------------- module API
def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "200",
        "test_name": "Reasoning: courier van load-out (0/1 knapsack, 9 parcels)",
        "difficulty_level": "7/10",
        "category": "Self-contained combinatorial optimization (0/1 knapsack)",
        "level": "reasoning",
        "weight": "long",
        # Self-declared: this task has no web lookup, so the web suite's visit.count grounding gate
        # does not (and must not) apply to it. See validator_lint_test.REASONING_SUITE_IDS.
        "grounding_required": False,
    }


def get_task_statement() -> str:
    rows = "\n".join(
        f"    {p['code']:<4}{p['mass']:>10}{p['fee']:>12}" for p in PARCELS
    )
    return (
        "This task is SELF-CONTAINED: every number you need is written below. There is nothing to "
        "look up and no page to open — do not search the web; just reason it through.\n\n"
        "A same-day courier depot has ONE van left for the evening run. Its cargo hold is empty and "
        f"its payload limit is {CAPACITY_KG} kg: the parcels loaded on this run must weigh at most "
        f"{CAPACITY_KG} kg in TOTAL. Nine parcels are waiting. Each parcel is indivisible (load it "
        "whole or leave it behind), each pays a fixed delivery fee if and only if it goes out on "
        "this run, and any parcel left behind simply waits for the next van (no penalty, no bonus).\n\n"
        "    parcel   mass (kg)   fee (credits)\n"
        f"{rows}\n\n"
        "Choose which parcels to load so that the TOTAL FEE of the loaded parcels is as LARGE as "
        f"possible, subject to their total mass being at most {CAPACITY_KG} kg. Exactly one set of "
        "parcels achieves that maximum. Beware: sorting the parcels by fee, by mass, or by "
        "fee-per-kilogram and greedily taking whatever still fits does NOT find it — you have to "
        "weigh the trade-offs.\n\n"
        "END YOUR REPLY with exactly these four lines, in this order and this shape:\n"
        "    SELECTED: <comma-separated parcel codes of the loaded parcels, e.g. P1, P4, P7 — "
        "codes only, nothing else on this line>\n"
        "    TOTAL FEE: <integer>\n"
        "    TOTAL MASS: <integer>\n"
        "    UNUSED CAPACITY: <integer>\n"
        "The TOTAL FEE is the primary answer: state it first in your final answer as well."
    )


def get_required_deliverables() -> List[str]:
    return [
        "TOTAL FEE of the best (fee-maximizing) load — the primary answer",
        "SELECTED: the parcel codes making up that load",
        "TOTAL MASS of the loaded parcels",
        "UNUSED CAPACITY left in the van",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reports the maximum achievable total fee (550) — not a greedy heuristic's total "
        "(517 / 486 / 474) and not any other feasible total",
        "Names a parcel set that actually fits in 240 kg and actually sums to the claimed fee "
        "(a correct number with a fabricated/inconsistent set scores 0)",
        "Reports the load's total mass (224 kg) and the unused capacity (16 kg)",
        "No web access is used or needed — the mandate is the entire input",
    ]


# ------------------------------------------------------------------------------ parsing utilities
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")
# Codes are matched permissively (any 1-2 digit P-code) rather than only the nine real ones, so an
# invented parcel ("P0", "P12") is CAUGHT as an unknown code instead of being silently dropped from
# a claimed load.
_CODE_RX = re.compile(r"\bP(\d{1,2})\b", re.I)
# A line that DECLARES the load. Anchored at line start (after bullet/heading markup) so a line
# beginning "Not selected:", "Parcels left behind:" etc. can never match and pollute the selection.
_MARKER_RX = re.compile(
    r"^[\s>*_#\-•\d.)]*"
    r"(?:final\s+)?(?:selected|selection|chosen|choice|load|loaded|manifest|packed|pack)"
    r"[^:\n]{0,30}:",
    re.I,
)
_NEGATION_RX = re.compile(
    r"\b(?:not|no|never|except|excluded?|excluding|omit|omitted|reject|rejected|left|leftover|"
    r"behind|remaining|remainder|unused|unloaded|skip|skipped|dropped)\b",
    re.I,
)
# The claimed total fee, in two passes: the exact block line the statement asks for, then a looser
# "...fee ... N" fallback. ``(?<![A-Za-z])`` keeps the digits of a parcel code ("P9") from ever
# being read as a figure.
_STRICT_FEE_RX = re.compile(
    r"^[\s>*_#\-]*(?:total\s*fee|fee\s*total)\b[^0-9\n]{0,12}(?<![A-Za-z])(\d[\d,]*)", re.I)
_LOOSE_FEE_RX = re.compile(r"\bfees?\b[^0-9\n]{0,28}(?<![A-Za-z])(\d[\d,]*)", re.I)

_BY_NUM = {p["num"]: p for p in PARCELS}


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text: ``deliverables[0]`` when the harness populated it, else the final text."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Everything the agent reported: the final deliverable plus every deliverable slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return "\n".join(parts)


def _int_values(text: str) -> List[int]:
    """Plain integers in ``text``: decimals dropped, grouping commas stripped, so "1,024" -> 1024
    and "55.0" is ignored. Digit runs are bounded by digits, so a trailing period is not swallowed."""
    vals: List[int] = []
    for tok in _NUM_TOKEN_RX.findall(text):
        if "." in tok:
            continue
        try:
            vals.append(int(tok.replace(",", "")))
        except ValueError:
            continue
    return vals


def _codes(segment: str) -> Set[int]:
    return {int(d) for d in _CODE_RX.findall(segment)}


def _parse_selection(result: Dict[str, Any]) -> Set[int]:
    """Recover the parcel set the agent CLAIMS to have loaded.

    Uses the LAST qualifying ``SELECTED:``-style line (the statement asks for the block at the end
    of the reply, so a later line supersedes any scratch work). Codes after a negation word on that
    line are dropped ("SELECTED: P2, P3 — P8 not loaded"). If the marker line itself carries no
    code, a bulleted layout is absorbed from the immediately following code-bearing lines. With no
    marker at all, falls back to every code mentioned in the primary answer text.
    """
    text = _all_text(result)
    lines = text.splitlines()
    found: Optional[Set[int]] = None
    for i, line in enumerate(lines):
        m = _MARKER_RX.match(line)
        if not m:
            continue
        if _NEGATION_RX.search(line[: m.end()]):
            continue                      # e.g. "Load NOT taken:" — not a selection declaration
        after = line[m.end():]
        neg = _NEGATION_RX.search(after)
        if neg:
            after = after[: neg.start()]
        codes = _codes(after)
        if not codes:                     # bulleted layout: "Selected parcels:" then one per line
            for nxt in lines[i + 1:]:
                if _NEGATION_RX.search(nxt) or _MARKER_RX.match(nxt):
                    break
                nxt_codes = _codes(nxt)
                if not nxt_codes:
                    break
                codes |= nxt_codes
        if codes:
            found = codes
    if found:
        return found
    return _codes(_primary_text(result))


def _selection_totals(selection: Set[int]) -> Tuple[int, int]:
    """(total mass, total fee) of a claimed selection; unknown codes contribute nothing."""
    mass = sum(_BY_NUM[c]["mass"] for c in selection if c in _BY_NUM)
    fee = sum(_BY_NUM[c]["fee"] for c in selection if c in _BY_NUM)
    return mass, fee


def _claimed_total_fee(result: Dict[str, Any]) -> Optional[int]:
    """The fee the agent CLAIMS its load earns.

    Prefers the LAST ``TOTAL FEE: N`` block line the statement asks for; falls back to the last
    loose "...fee ... N" mention. Lines that declare the selection ("SELECTED: P2, P3, ...") or talk
    about what was left behind are skipped, so neither a parcel code nor the left-behind fee can be
    mistaken for the claim. Returns ``None`` when the agent stated no fee line at all — the keystone
    then falls back to looking for the optimum in the primary answer text.
    """
    strict: Optional[int] = None
    loose: Optional[int] = None
    for line in _all_text(result).splitlines():
        if _MARKER_RX.match(line) or _NEGATION_RX.search(line):
            continue
        m = _STRICT_FEE_RX.match(line)
        if m:
            strict = int(m.group(1).replace(",", ""))
            continue
        m = _LOOSE_FEE_RX.search(line)
        if m:
            loose = int(m.group(1).replace(",", ""))
    return strict if strict is not None else loose


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1). BOTH must hold:

    1. the CLAIMED total fee equals the true optimum (550) — read from the ``TOTAL FEE:`` line when
       present, otherwise 550 must appear in the primary answer text; and
    2. the CLAIMED parcel set is REAL: it fits the 240 kg limit and its fees actually add up to
       550. Because exactly one subset attains 550 (uniqueness, solver-verified), this is equivalent
       to naming the true optimal load — and it is what stops a "right number, fabricated subset"
       answer (or a bare number with no set at all) from banking the keystone.

    There is deliberately NO grounding condition here — this task reads no page at all, so
    ``observability`` is accepted for signature parity and ignored. (``scripts/validator_lint.py``
    therefore reports a ``[GATE]`` finding for this task. That finding is correct and must NOT be
    silenced: it honestly says "this keystone can score without grounding", which is the defining
    property of the self-contained reasoning category.)
    """
    claimed = _claimed_total_fee(result)
    if claimed is not None:
        if claimed != OPTIMAL_FEE:
            return False
    elif OPTIMAL_FEE not in _int_values(_primary_text(result)):
        return False
    selection = _parse_selection(result)
    if not selection or not selection.issubset(set(_BY_NUM)):
        return False
    mass, fee = _selection_totals(selection)
    return mass <= CAPACITY_KG and fee == OPTIMAL_FEE


# ----------------------------------------------------------------------------------- validators
def validate_load_feasibility(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: did the agent commit to a parsable load that RESPECTS the constraint?

    Deliberately not gated on the keystone — a wrong-but-feasible answer (the greedy traps) still
    scores here, while an over-capacity or unparsable load scores 0. This is the axis that separates
    "searched badly" from "ignored the constraint / never committed to a set".
    """
    selection = _parse_selection(result)
    if not selection:
        return {"check": "load_feasibility", "passed": False, "score": 0.0,
                "reason": "No parsable parcel selection (expected a 'SELECTED: P..' line)"}
    unknown = sorted(c for c in selection if c not in _BY_NUM)
    mass, fee = _selection_totals(selection)
    ok = not unknown and mass <= CAPACITY_KG
    codes = ", ".join(f"P{c}" for c in sorted(selection))
    return {"check": "load_feasibility", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"Claimed load {{{codes}}} = {mass} kg / {fee} credits "
                       f"({'fits' if mass <= CAPACITY_KG else 'EXCEEDS'} the {CAPACITY_KG} kg limit)"
                       + (f"; unknown codes {unknown}" if unknown else ""))}


def validate_load_efficiency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated graduated diagnostic: what fraction of the attainable fee did the claimed load
    ACTUALLY capture (recomputed from the table, never from the agent's arithmetic)?

    An infeasible or missing load scores 0. This keeps a wrong answer informative — density-greedy
    lands at 517/550 = 0.94, fee-first at 0.86 — instead of collapsing every failure to a flat 0.
    """
    selection = _parse_selection(result)
    mass, fee = _selection_totals(selection)
    if not selection or mass > CAPACITY_KG or not selection.issubset(set(_BY_NUM)):
        return {"check": "load_efficiency", "passed": False, "score": 0.0,
                "reason": "No real, capacity-respecting load claimed -> 0 fee captured"}
    score = min(1.0, fee / OPTIMAL_FEE)
    return {"check": "load_efficiency", "passed": fee == OPTIMAL_FEE, "score": score,
            "reason": f"Claimed load captures {fee}/{OPTIMAL_FEE} credits ({score:.0%} of optimal)"}


def validate_keystone_optimal_fee(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the maximum achievable total fee, 550, TOGETHER with a parcel set that
    really fits and really sums to it. Exact match — the answer is an exact integer computed from
    given data, so no tolerance band is warranted; the nearest feasible alternative is 538."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_optimal_fee", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": (f"Optimal total fee {OPTIMAL_FEE} reported with a consistent, capacity-"
                       f"respecting load" if passed else
                       f"Optimal total fee ({OPTIMAL_FEE}) missing/incorrect, or the claimed parcel "
                       f"set does not fit {CAPACITY_KG} kg / does not sum to it")}


def validate_load_figures(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary (graduated, k/2): the two supporting figures — the load's total mass (224 kg)
    and the unused capacity (16 kg). Short-circuits to 0 when the keystone is absent, so a run that
    reports a greedy total cannot bank partial credit for its bookkeeping."""
    if not _keystone_ok(result, observability):
        return {"check": "load_figures", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> supporting figures not credited"}
    present = set(_int_values(_all_text(result)))
    wanted = [("total mass", OPTIMAL_MASS), ("unused capacity", UNUSED_CAPACITY)]
    hits = [name for name, val in wanted if val in present]
    return {"check": "load_figures", "passed": len(hits) == len(wanted), "score": len(hits) / len(wanted),
            "reason": f"{len(hits)}/{len(wanted)} supporting figures reported ({', '.join(hits) or 'none'})"}


def get_validation_functions() -> List[callable]:
    return [
        validate_load_feasibility,
        validate_load_efficiency,
        validate_keystone_optimal_fee,
        validate_load_figures,
    ]


def get_llm_validation_function() -> callable:
    # None -> no LLM judge. This category holds itself to the web suite's determinism bar: the
    # answer is an exact integer plus an exactly-checkable subset, so a judge would only add noise.
    return None
