"""
Test 201: Self-contained reasoning — warehouse outbound batch (subset-sum fill over 10 crates)
Level: reasoning   Weight: long   Difficulty: 7/10   grounding_required: False

SELF-CONTAINED: there is NO page to visit and NO fact to look up — the mandate carries the whole
input (ten crates with a mass, plus a trailer mass limit). This task therefore does NOT try to
satisfy the web suite's ``visit.count > 0`` grounding gate (``scripts/validator_lint.py`` will emit
a ``[GATE]`` finding for the keystone here; that is EXPECTED and correct, see
``services/agent/tests/validator_lint_test.py::test_reasoning_suite_lints_clean_of_llm_judges``).
What replaces grounding as the validity bar is (a) self-containment and (b) ANSWER-SPACE NOVELTY:
a deliberately generic warehouse-dispatch wrapper, never a recognizable named puzzle, over numbers
drawn by a seeded RNG (see "Instance provenance") — so neither the setup NOR the answer can be
pattern-matched from a memorized puzzle.

DELIBERATELY A DIFFERENT SHAPE FROM TASK 200 (which is a two-attribute mass/fee knapsack over 9
parcels): here there is ONE attribute only, so the objective is a pure bounded subset-sum — pack
the trailer as FULL as possible without exceeding the limit. Different item count (10 vs 9),
different capacity, different narrative, different reported figures, and a different failure mode:
with only one attribute the "sort by value density" heuristic that dominates 200 does not even
exist, and the natural shortcut becomes big-crates-first (or the assumption that the limit can be
hit exactly).

WHY IT DISCRIMINATES: the instance was accepted only if BOTH single-sort heuristics are strictly
suboptimal AND the limit is unreachable, so the two cheap shortcuts each land on a specific wrong
number:
  * heaviest-first  (fill with the biggest crates that still fit) -> 811
  * lightest-first  (fit the most crates)                         -> 800
  * "the answer is just the limit" (exact fill)                   -> 880, impossible: NO subset of
    these ten crates sums to 880, and none reaches 863..880 either
  * TRUE OPTIMUM (exhaustive search)                              -> 862, leaving 18 kg unused
  Getting 862 requires exploring combinations, not sorting once and not assuming a perfect fill.

GROUND TRUTH — verified by TWO independently-implemented reference solvers embedded below, NOT
hand-derived:
  * ``_solve_bruteforce`` — bitmask enumeration over all 2^10 = 1024 subsets, collecting every
    feasible total and every optimal mask.
  * ``_solve_dp``        — 1-D counting/reachability dynamic program over achievable totals
    0..capacity, with an independent backward reconstruction of an optimal subset.
  Both agree; ``services/agent/tests/reasoning_201_crate_batch_validators_test.py`` RE-RUNS both on
  every test run and asserts they agree with each other AND with the hard-coded constants below,
  so the ground truth is machine-re-verifiable by any future reviewer.

  crate    mass (kg)    in optimal batch?
    C1       200            -
    C2       186            -
    C3       212           YES
    C4       107           YES
    C5       108           YES
    C6       116           YES
    C7       213           YES
    C8        84            -
    C9       106           YES
    C10       93            -
  trailer limit = 880 kg        all-crate total = 1425 kg

  KEYSTONE     = 862 kg loaded  (= 212 + 107 + 108 + 116 + 213 + 106)
  optimal batch= {C3, C4, C5, C6, C7, C9}; unused capacity 18 kg; mass left behind 563 kg
  UNIQUENESS   = exactly 1 of the 1024 subsets attains 862 (so "the claimed crates sum to the
                 claimed total" is equivalent to "the claimed crates ARE the optimal batch")
  MARGIN       = the best feasible total BELOW the optimum is 850, a 12 kg gap; 863..880 are all
                 unreachable, so neither a near-miss sum nor the limit itself can pass as correct.
                 862 / 18 / 563 are pairwise distinct and none equals any printed crate mass or the
                 limit, so echoing the table can never produce a checked figure.

Instance provenance (procedurally generated — reproduce with):
    rng = random.Random(22028 * 1000 + 10 * 17 + 880)
    masses = rng.sample(range(37, 234), 10); capacity = 880
  Seeds were scanned in order and the first one accepted that satisfies ALL of: 1.6*cap <= total
  mass <= 2.2*cap; a UNIQUE optimal subset; runner-up total <= optimum - 10; optimal batch of 4-8
  crates; unused capacity in [15, 45] (>= 15 keeps every checked figure clear of the 1..10 digits in
  the crate codes, so a code can never be misread as a figure, and > 0 makes an exact fill
  impossible); both greedy heuristics strictly suboptimal; and the three checked figures
  (862 / 18 / 563) pairwise distinct and absent from the printed table.

Validators: one hard 0/1 keystone that requires BOTH the correct loaded total AND a claimed crate
set that actually sums to it within the limit (a "right number, fabricated subset" answer fails);
one gated secondary (the supporting figures) that short-circuits to 0 without the keystone; and two
UN-gated diagnostics (is the claimed batch even legal / how full did it actually get) that stay
informative when the answer is wrong. No LLM judge — same determinism bar as the web suite.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the instance (single source of truth for the statement, the validators and the solvers) ---
CRATES: List[Dict[str, Any]] = [
    {"code": "C1", "num": 1, "mass": 200},
    {"code": "C2", "num": 2, "mass": 186},
    {"code": "C3", "num": 3, "mass": 212},
    {"code": "C4", "num": 4, "mass": 107},
    {"code": "C5", "num": 5, "mass": 108},
    {"code": "C6", "num": 6, "mass": 116},
    {"code": "C7", "num": 7, "mass": 213},
    {"code": "C8", "num": 8, "mass": 84},
    {"code": "C9", "num": 9, "mass": 106},
    {"code": "C10", "num": 10, "mass": 93},
]
LIMIT_KG = 880

# Solver-verified ground truth (see the module docstring; re-derived in the unit tests).
OPTIMAL_LOAD = 862                          # KEYSTONE — max total mass that fits
OPTIMAL_SET = frozenset({3, 4, 5, 6, 7, 9})  # crate numbers -> C3, C4, C5, C6, C7, C9
UNUSED_CAPACITY = LIMIT_KG - OPTIMAL_LOAD    # 18
TOTAL_ALL_CRATES = 1425
MASS_LEFT_BEHIND = TOTAL_ALL_CRATES - OPTIMAL_LOAD   # 563
RUNNER_UP_LOAD = 850                        # best feasible total strictly below the optimum (gap 12)
GREEDY_HEAVIEST_LOAD = 811                  # heaviest-first heuristic
GREEDY_LIGHTEST_LOAD = 800                  # lightest-first heuristic


# ------------------------------------------------------------------ reference solver A (brute force)
def _solve_bruteforce(crates: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    """REFERENCE SOLVER A — exhaustive bitmask enumeration over all 2^N subsets.

    Independent of :func:`_solve_dp` (different algorithm: explicit enumeration vs. a 1-D DP).
    :param crates: Item dicts with ``num``/``mass``.
    :param limit: Mass limit.
    :return: ``{"best_load", "optimal_sets", "n_optimal", "runner_up", "reachable"}``.
    """
    n = len(crates)
    best_load = -1
    optimal_sets: List[frozenset] = []
    reachable: Set[int] = set()
    for mask in range(1 << n):
        total = 0
        for i in range(n):
            if mask >> i & 1:
                total += crates[i]["mass"]
        if total > limit:
            continue
        reachable.add(total)
        if total > best_load:
            best_load = total
            optimal_sets = [frozenset(crates[i]["num"] for i in range(n) if mask >> i & 1)]
        elif total == best_load:
            optimal_sets.append(frozenset(crates[i]["num"] for i in range(n) if mask >> i & 1))
    runner_up = max((t for t in reachable if t < best_load), default=None)
    return {"best_load": best_load, "optimal_sets": optimal_sets, "n_optimal": len(optimal_sets),
            "runner_up": runner_up, "reachable": reachable}


# --------------------------------------------------------------------- reference solver B (1-D DP)
def _solve_dp(crates: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    """REFERENCE SOLVER B — 1-D counting DP over achievable totals ``0..limit`` (the classic
    descending-index 0/1 update), plus a BACKWARD reconstruction that walks the per-prefix
    reachability snapshots. Deliberately a DIFFERENT algorithm from :func:`_solve_bruteforce` (no
    subset is ever materialized during the search), so agreement between the two is a real
    cross-check rather than a copy-paste tautology.

    :param crates: Item dicts with ``num``/``mass``.
    :param limit: Mass limit.
    :return: ``{"best_load", "n_optimal", "runner_up", "optimal_set"}``.
    """
    ways = [0] * (limit + 1)
    ways[0] = 1
    prefix_reachable: List[List[bool]] = []
    for c in crates:
        m = c["mass"]
        for s in range(limit, m - 1, -1):     # descending, so ways[s - m] is still the pre-update value
            if ways[s - m]:
                ways[s] += ways[s - m]
        prefix_reachable.append([w > 0 for w in ways])
    best_load = max(s for s in range(limit + 1) if ways[s])
    n_optimal = ways[best_load]
    runner_up = max((s for s in range(best_load) if ways[s]), default=None)

    # Backward reconstruction: item i is taken iff (total - mass_i) was reachable using items 0..i-1.
    base = [s == 0 for s in range(limit + 1)]
    chosen: List[int] = []
    remaining = best_load
    for i in range(len(crates) - 1, -1, -1):
        before = prefix_reachable[i - 1] if i > 0 else base
        m = crates[i]["mass"]
        if remaining - m >= 0 and before[remaining - m]:
            chosen.append(crates[i]["num"])
            remaining -= m
    return {"best_load": best_load, "n_optimal": n_optimal, "runner_up": runner_up,
            "optimal_set": frozenset(chosen) if remaining == 0 else None}


def _greedy_load(crates: List[Dict[str, Any]], limit: int, heaviest_first: bool) -> int:
    """Total mass a single-sort greedy pass achieves (the shortcuts this instance is built to defeat)."""
    total = 0
    for c in sorted(crates, key=lambda x: -x["mass"] if heaviest_first else x["mass"]):
        if total + c["mass"] <= limit:
            total += c["mass"]
    return total


# ------------------------------------------------------------------------------------- module API
def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "201",
        "test_name": "Reasoning: warehouse outbound batch (bounded subset-sum, 10 crates)",
        "difficulty_level": "7/10",
        "category": "Self-contained combinatorial optimization (bounded subset-sum)",
        "level": "reasoning",
        "weight": "long",
        # Self-declared: this task has no web lookup, so the web suite's visit.count grounding gate
        # does not (and must not) apply to it. See validator_lint_test.REASONING_SUITE_IDS.
        "grounding_required": False,
    }


def get_task_statement() -> str:
    rows = "\n".join(f"    {c['code']:<6}{c['mass']:>10}" for c in CRATES)
    return (
        "This task is SELF-CONTAINED: every number you need is written below. There is nothing to "
        "look up and no page to open — do not search the web; just reason it through.\n\n"
        "A warehouse is loading ONE outbound trailer tonight. The trailer is empty and its cargo "
        f"limit is {LIMIT_KG} kg: the crates put on board must weigh at most {LIMIT_KG} kg in "
        "TOTAL. Ten crates are staged on the dock. A crate is indivisible (load it whole or leave "
        "it), and the haulier bills strictly by the kilogram actually shipped, so the dispatcher "
        "wants this trailer to carry AS MUCH MASS AS POSSIBLE. Any crate left on the dock simply "
        "goes on a later trailer (no penalty, no bonus).\n\n"
        "    crate    mass (kg)\n"
        f"{rows}\n\n"
        f"Choose which crates to load so that their TOTAL MASS is as LARGE as possible while "
        f"staying at or below {LIMIT_KG} kg. Exactly one set of crates achieves that maximum. Two "
        "warnings: the limit itself is NOT achievable — no combination of these crates weighs "
        f"exactly {LIMIT_KG} kg — and simply taking the heaviest crates that still fit (or cramming "
        "in as many light ones as possible) does NOT find the best batch.\n\n"
        "END YOUR REPLY with exactly these four lines, in this order and this shape:\n"
        "    SELECTED: <comma-separated crate codes of the loaded crates, e.g. C1, C4, C7 — "
        "codes only, nothing else on this line>\n"
        "    TOTAL MASS LOADED: <integer>\n"
        "    UNUSED CAPACITY: <integer>\n"
        "    MASS LEFT BEHIND: <integer>\n"
        "The TOTAL MASS LOADED is the primary answer: state it first in your final answer as well."
    )


def get_required_deliverables() -> List[str]:
    return [
        "TOTAL MASS LOADED — the largest total mass that fits in the trailer (the primary answer)",
        "SELECTED: the crate codes making up that batch",
        "UNUSED CAPACITY left under the limit",
        "MASS LEFT BEHIND on the dock",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reports the maximum achievable loaded mass (862 kg) — not a greedy heuristic's total "
        "(811 / 800) and not the unachievable limit (880)",
        "Names a crate set that actually stays within 880 kg and actually sums to the claimed mass "
        "(a correct number with a fabricated/inconsistent set scores 0)",
        "Reports the unused capacity (18 kg) and the mass left behind (563 kg)",
        "No web access is used or needed — the mandate is the entire input",
    ]


# ------------------------------------------------------------------------------ parsing utilities
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")
_CODE_RX = re.compile(r"\bC(\d{1,2})\b", re.I)
# A line that DECLARES the batch. Anchored at line start (after bullet/heading markup) so a line
# beginning "Not selected:", "Crates left behind:" etc. can never match and pollute the selection.
_MARKER_RX = re.compile(
    r"^[\s>*_#\-•\d.)]*"
    r"(?:final\s+)?(?:selected|selection|chosen|choice|batch|load|loaded|manifest|packed|pack)"
    r"[^:\n]{0,30}:",
    re.I,
)
_NEGATION_RX = re.compile(
    r"\b(?:not|no|never|except|excluded?|excluding|omit|omitted|reject|rejected|left|leftover|"
    r"behind|remaining|remainder|unused|unloaded|dock|skip|skipped|dropped)\b",
    re.I,
)
# The claimed loaded total, in two passes: the exact block line the statement asks for, then a
# looser "...total/loaded ... N" fallback. ``(?<![A-Za-z])`` keeps the digits of a crate code
# ("C10") from ever being read as a figure.
_STRICT_LOAD_RX = re.compile(
    r"^[\s>*_#\-]*total\b[^:\n]{0,24}:[^0-9\n]{0,8}(?<![A-Za-z])(\d[\d,]*)", re.I)
_LOOSE_LOAD_RX = re.compile(
    r"\b(?:total|loaded|shipped|carried)\b[^0-9\n]{0,28}(?<![A-Za-z])(\d[\d,]*)", re.I)

_BY_NUM = {c["num"]: c for c in CRATES}


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
    """Plain integers in ``text``: decimals dropped, grouping commas stripped, so "1,425" -> 1425
    and "86.2" is ignored. Digit runs are bounded by digits, so a trailing period is not swallowed."""
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
    """Recover the crate set the agent CLAIMS to have loaded.

    Uses the LAST qualifying ``SELECTED:``-style line (the statement asks for the block at the end
    of the reply, so a later line supersedes any scratch work). Codes after a negation word on that
    line are dropped ("SELECTED: C3, C4 — C1 not loaded"). If the marker line itself carries no
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
            continue                      # e.g. "Load NOT taken:" — not a batch declaration
        after = line[m.end():]
        neg = _NEGATION_RX.search(after)
        if neg:
            after = after[: neg.start()]
        codes = _codes(after)
        if not codes:                     # bulleted layout: "Selected crates:" then one per line
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


def _selection_mass(selection: Set[int]) -> int:
    """Total mass of a claimed selection; unknown codes contribute nothing."""
    return sum(_BY_NUM[c]["mass"] for c in selection if c in _BY_NUM)


def _claimed_load(result: Dict[str, Any]) -> Optional[int]:
    """The mass the agent CLAIMS its batch carries.

    Prefers the LAST ``TOTAL MASS LOADED: N`` block line the statement asks for; falls back to the
    last loose "...total/loaded ... N" mention. Lines that declare the selection ("SELECTED: C3,
    C4, ...") or talk about what was left behind / unused are skipped, so neither a crate code nor
    the left-behind mass can be mistaken for the claim. Returns ``None`` when the agent stated no
    total line at all — the keystone then falls back to looking for the optimum in the primary
    answer text.
    """
    strict: Optional[int] = None
    loose: Optional[int] = None
    for line in _all_text(result).splitlines():
        if _MARKER_RX.match(line) or _NEGATION_RX.search(line):
            continue                      # skip "SELECTED: ..." and "... left behind: 563"
        m = _STRICT_LOAD_RX.match(line)
        if m:
            strict = int(m.group(1).replace(",", ""))
            continue
        m = _LOOSE_LOAD_RX.search(line)
        if m:
            loose = int(m.group(1).replace(",", ""))
    return strict if strict is not None else loose


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1). BOTH must hold:

    1. the CLAIMED loaded mass equals the true optimum (862) — read from the ``TOTAL MASS LOADED:``
       line when present, otherwise 862 must appear in the primary answer text; and
    2. the CLAIMED crate set is REAL: it stays within the 880 kg limit and its masses actually add
       up to 862. Because exactly one subset attains 862 (uniqueness, solver-verified), this is
       equivalent to naming the true optimal batch — and it is what stops a "right number,
       fabricated subset" answer (or a bare number with no crates at all) from banking the keystone.

    There is deliberately NO grounding condition here — this task reads no page at all, so
    ``observability`` is accepted for signature parity and ignored. (``scripts/validator_lint.py``
    therefore reports a ``[GATE]`` finding for this task. That finding is correct and must NOT be
    silenced: it honestly says "this keystone can score without grounding", which is the defining
    property of the self-contained reasoning category.)
    """
    claimed = _claimed_load(result)
    if claimed is not None:
        if claimed != OPTIMAL_LOAD:
            return False
    elif OPTIMAL_LOAD not in _int_values(_primary_text(result)):
        return False
    selection = _parse_selection(result)
    if not selection or not selection.issubset(set(_BY_NUM)):
        return False
    return _selection_mass(selection) == OPTIMAL_LOAD   # == 862 <= 880, so the limit is respected


# ----------------------------------------------------------------------------------- validators
def validate_batch_legality(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: did the agent commit to a parsable batch that RESPECTS the limit?

    Deliberately not gated on the keystone — a wrong-but-legal batch (the greedy traps) still scores
    here, while an over-limit or unparsable batch scores 0. This is the axis that separates
    "searched badly" from "ignored the constraint / never committed to a set".
    """
    selection = _parse_selection(result)
    if not selection:
        return {"check": "batch_legality", "passed": False, "score": 0.0,
                "reason": "No parsable crate selection (expected a 'SELECTED: C..' line)"}
    unknown = sorted(c for c in selection if c not in _BY_NUM)
    mass = _selection_mass(selection)
    ok = not unknown and mass <= LIMIT_KG
    codes = ", ".join(f"C{c}" for c in sorted(selection))
    return {"check": "batch_legality", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"Claimed batch {{{codes}}} = {mass} kg "
                       f"({'within' if mass <= LIMIT_KG else 'OVER'} the {LIMIT_KG} kg limit)"
                       + (f"; unknown codes {unknown}" if unknown else ""))}


def validate_fill_efficiency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated graduated diagnostic: how full did the claimed batch ACTUALLY get, as a fraction of
    the attainable maximum (recomputed from the table, never from the agent's arithmetic)?

    An over-limit or missing batch scores 0. This keeps a wrong answer informative — heaviest-first
    lands at 811/862 = 0.94, lightest-first at 0.93 — instead of collapsing every failure to a
    flat 0.
    """
    selection = _parse_selection(result)
    mass = _selection_mass(selection)
    if not selection or mass > LIMIT_KG or not selection.issubset(set(_BY_NUM)):
        return {"check": "fill_efficiency", "passed": False, "score": 0.0,
                "reason": "No real, limit-respecting batch claimed -> 0 kg shipped"}
    score = min(1.0, mass / OPTIMAL_LOAD)
    return {"check": "fill_efficiency", "passed": mass == OPTIMAL_LOAD, "score": score,
            "reason": f"Claimed batch ships {mass}/{OPTIMAL_LOAD} kg ({score:.0%} of optimal)"}


def validate_keystone_optimal_load(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the maximum loadable mass, 862 kg, TOGETHER with a crate set that really
    fits and really sums to it. Exact match — the answer is an exact integer computed from given
    data, so no tolerance band is warranted; the nearest feasible alternative is 850 and everything
    in 863..880 is unreachable."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_optimal_load", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": (f"Optimal loaded mass {OPTIMAL_LOAD} kg reported with a consistent, "
                       f"limit-respecting batch" if passed else
                       f"Optimal loaded mass ({OPTIMAL_LOAD} kg) missing/incorrect, or the claimed "
                       f"crate set exceeds {LIMIT_KG} kg / does not sum to it")}


def validate_batch_figures(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary (graduated, k/2): the two supporting figures — unused capacity (18 kg) and
    mass left behind (563 kg). Short-circuits to 0 when the keystone is absent, so a run that
    reports a greedy total cannot bank partial credit for its bookkeeping."""
    if not _keystone_ok(result, observability):
        return {"check": "batch_figures", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> supporting figures not credited"}
    present = set(_int_values(_all_text(result)))
    wanted = [("unused capacity", UNUSED_CAPACITY), ("mass left behind", MASS_LEFT_BEHIND)]
    hits = [name for name, val in wanted if val in present]
    return {"check": "batch_figures", "passed": len(hits) == len(wanted), "score": len(hits) / len(wanted),
            "reason": f"{len(hits)}/{len(wanted)} supporting figures reported ({', '.join(hits) or 'none'})"}


def get_validation_functions() -> List[callable]:
    return [
        validate_batch_legality,
        validate_fill_efficiency,
        validate_keystone_optimal_load,
        validate_batch_figures,
    ]


def get_llm_validation_function() -> callable:
    # None -> no LLM judge. This category holds itself to the web suite's determinism bar: the
    # answer is an exact integer plus an exactly-checkable subset, so a judge would only add noise.
    return None
