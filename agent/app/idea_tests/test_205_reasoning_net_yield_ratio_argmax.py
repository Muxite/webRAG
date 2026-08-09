"""
Test 205: Self-contained reasoning — SUBTRACT-then-DIVIDE derived-ratio ARGMAX over haulage runs.
Level: graph (reasoning shape, not a browsing shape)   Weight: short   Difficulty: 8/10

SELF-CONTAINED / NO WEB, like its sibling 204: the statement prints every raw number and there is
nothing to look up. The two properties that replace the web suite's grounding gate are
self-containment and answer-space novelty (procedurally generated figures, not a memorable
published puzzle — see "Provenance" below).

Shape: the same "ranking != raw value" trap as 204, but one step deeper — the derived quantity
takes a SUBTRACTION before the DIVISION, and the argmax runs the opposite direction (maximise,
not minimise), so the two instances share neither the arithmetic path nor a single number:

    cargo efficiency = (gross laden weight - tare weight) / fuel used     [kg of cargo per litre]

FOUR different shortcuts are baited, and every one of them lands on the wrong run:

  * the HEAVIEST gross load (Pellham, 38,092 kg) is only 4th of 6 — it is also the run that
    carried the most net cargo (26,838 kg), so both "biggest" readings fail together;
  * the LEAST fuel burned (Orrick, 225 L) is 3rd;
  * the LIGHTEST tare (Orrick again, 3,372 kg) is 3rd;
  * SKIPPING THE SUBTRACTION picks the wrong run outright: on gross-per-litre the order is
    Rothsay 94.6 > Marlowe 87.7, so a model that divides the GROSS weight by fuel answers
    Rothsay — the runner-up, not the winner. The tare column decides the argmax itself, not
    merely the reported value, which is what makes the subtraction load-bearing.

The winner (Marlowe) is 3rd of 6 by gross weight and 2nd-lowest by fuel: bland on every raw
axis, so no printed column points at it.

Grounding / anti-guess: ``grounding_required`` is False — there is no ``visit.count`` to gate on,
so a 1-in-6 name guess would otherwise bank the keystone at p=0.167. The replacement gate is that
the keystone also requires the correct DERIVED VALUE (71 kg per litre), a number printed nowhere
in the statement and obtainable only by subtracting the tare and then dividing.

Ground truth (derived by reference computation — NEVER by hand — and cross-checked by a SECOND,
independently implemented computation that performs NO division at all: it ranks the runs by
pairwise integer cross-multiplication, ``net_i * fuel_j`` vs ``net_j * fuel_i``. Both agree
exactly; ``test_205_reasoning_net_yield_ratio_argmax_validators_test.py`` re-derives the whole
table at test time with the cross-multiplication solver, so these literals can never drift):

  run         gross      tare       fuel     net cargo     kg/litre   rank   gross/litre
  Marlowe     20,090 kg   3,831 kg   229 L    16,259 kg       71        1        87.7
  Rothsay     26,952 kg   8,712 kg   285 L    18,240 kg       64        2        94.6  <- gross/L argmax
  Orrick      16,422 kg   3,372 kg   225 L    13,050 kg       58        3        73.0  <- least fuel, least tare
  Pellham     38,092 kg  11,254 kg   497 L    26,838 kg       54        4        76.6  <- heaviest gross + most cargo
  Northgate   19,699 kg   3,551 kg   367 L    16,148 kg       44        5        53.7
  Quarrow     27,164 kg   7,484 kg   492 L    19,680 kg       40        6        55.2

  Every division is exact (fuel divides net cargo without remainder for all six runs), so the
  ground truth is integer-clean and no rounding convention is in play.

  MARGIN: 71 vs the runner-up's 64 kg/L = +7 kg/L, i.e. +10.9%. The tightest adjacent gap
  anywhere in the ranking is +7.4% (54 -> 58), so the +/-0.5% acceptance bands the validators use
  are mutually disjoint by a factor of ~7: no rounding, and no single arithmetic slip, can flip
  the keystone or cross-credit one run's efficiency to another.

  GAP DELIVERABLE (the gated secondary): the winner's lead over the heaviest-gross run is
  71 - 54 = 17 kg/L — a second-order derived quantity that can only be produced by computing the
  decoy's efficiency too, so the trap cannot be answered by dismissing the decoy unexamined.

Provenance / procedural variation: the eighteen raw figures were emitted by a seeded constrained
generator (seed 20533; fuel 180-520 L, target efficiency 35-95 kg/L, net cargo kept in
8,000-28,000 kg, tare drawn at 14-60% of net, retained only when fuel divides net cargo exactly)
under the constraint set encoded in the import-time assertions below: the argmax is not the
heaviest-gross, most-cargo, least-fuel or lightest-tare run, AND not the gross-per-litre argmax;
the heaviest-gross run must rank >= 4th; keystone margin in [9%, 20%]; every adjacent efficiency
gap >= 6% relative; no efficiency or gap may collide with any printed raw figure (regex token
hygiene); and no printed raw figure may repeat. Sibling task 204 was generated from a different
seed with a different scenario and shares no number with this one.
"""

from typing import Dict, Any, List, Optional
import re
from agent.app.idea_test_utils import extract_final_text


# --------------------------------------------------------------------------------------------
# Fixtures: single source of truth for the statement, the validators and the invariants.
# ``net`` and ``eff`` are DERIVED (never printed in the statement) — producing them is the work.
# --------------------------------------------------------------------------------------------
RUNS: List[Dict[str, Any]] = [
    {"key": "marlowe",   "name": "Marlowe",   "gross": 20090, "tare": 3831,  "fuel": 229,
     "net": 16259, "eff": 71, "name_rx": r"\bmarlowe\b"},
    {"key": "northgate", "name": "Northgate", "gross": 19699, "tare": 3551,  "fuel": 367,
     "net": 16148, "eff": 44, "name_rx": r"\bnorthgate\b"},
    {"key": "orrick",    "name": "Orrick",    "gross": 16422, "tare": 3372,  "fuel": 225,
     "net": 13050, "eff": 58, "name_rx": r"\borrick\b"},
    {"key": "pellham",   "name": "Pellham",   "gross": 38092, "tare": 11254, "fuel": 497,
     "net": 26838, "eff": 54, "name_rx": r"\bpellham\b"},
    {"key": "quarrow",   "name": "Quarrow",   "gross": 27164, "tare": 7484,  "fuel": 492,
     "net": 19680, "eff": 40, "name_rx": r"\bquarrow\b"},
    {"key": "rothsay",   "name": "Rothsay",   "gross": 26952, "tare": 8712,  "fuel": 285,
     "net": 18240, "eff": 64, "name_rx": r"\brothsay\b"},
]

_BY_EFF = sorted(RUNS, key=lambda r: -r["eff"])
WINNER = _BY_EFF[0]        # Marlowe — most cargo per litre (keystone)
RUNNER_UP = _BY_EFF[1]     # Rothsay — second (and the run a no-subtraction model wrongly picks)
DECOY = max(RUNS, key=lambda r: r["gross"])       # Pellham — heaviest gross load
NO_SUBTRACTION_PICK = max(RUNS, key=lambda r: r["gross"] / r["fuel"])   # Rothsay
EFF_GAP = WINNER["eff"] - DECOY["eff"]            # 17 kg/L

# +/-0.5% acceptance tolerance on a reported number: every efficiency is an exact integer, so the
# tolerance only absorbs rendering ("71" / "71.0"), never blurs two runs together.
VALUE_TOL = 0.005

# ---------------------------------- import-time invariants ----------------------------------
for _r in RUNS:
    assert _r["gross"] - _r["tare"] == _r["net"], f"{_r['name']}: net cargo mismatch"
    assert _r["net"] % _r["fuel"] == 0, f"{_r['name']}: efficiency is not exact"
    assert _r["net"] // _r["fuel"] == _r["eff"], f"{_r['name']}: declared efficiency is wrong"
assert len({r["eff"] for r in RUNS}) == len(RUNS), "efficiencies must be distinct"
assert WINNER is max(RUNS, key=lambda r: r["eff"])
# --- the "ranking != raw value" traps: the winner tops NO raw column ---
assert WINNER is not max(RUNS, key=lambda r: r["gross"]), "winner must not be the heaviest gross"
assert WINNER is not max(RUNS, key=lambda r: r["net"]), "winner must not carry the most cargo"
assert WINNER is not min(RUNS, key=lambda r: r["fuel"]), "winner must not burn the least fuel"
assert WINNER is not min(RUNS, key=lambda r: r["tare"]), "winner must not have the lightest tare"
# --- and, decisively, SKIPPING THE SUBTRACTION must pick a different run ---
assert NO_SUBTRACTION_PICK is not WINNER, \
    "gross/fuel must NOT rank the winner first, else the tare column is decorative"
assert (NO_SUBTRACTION_PICK["gross"] / NO_SUBTRACTION_PICK["fuel"]) > \
       (WINNER["gross"] / WINNER["fuel"]) * 1.04, "the no-subtraction mistake must be decisive"
_RANK = {r["key"]: i for i, r in enumerate(_BY_EFF)}      # 0 = most efficient
assert _RANK[DECOY["key"]] >= 3, "the heaviest-gross run must be mid-or-worse on efficiency"
assert _RANK[min(RUNS, key=lambda r: r["fuel"])["key"]] >= 2, "least-fuel run must not be top-2"
assert _RANK[min(RUNS, key=lambda r: r["tare"])["key"]] >= 2, "lightest-tare run must not be top-2"
# --- keystone margin: wide enough that no rounding or single slip can flip the argmax ---
_MARGIN = (WINNER["eff"] - RUNNER_UP["eff"]) / RUNNER_UP["eff"]
assert 0.09 <= _MARGIN <= 0.20, f"keystone margin {_MARGIN:.3f} outside the designed band"
# --- acceptance bands must be mutually disjoint ---
_SORTED_EFF = [r["eff"] for r in sorted(RUNS, key=lambda r: r["eff"])]
for _i in range(len(_SORTED_EFF) - 1):
    _lo, _hi = _SORTED_EFF[_i], _SORTED_EFF[_i + 1]
    assert _hi * (1 - VALUE_TOL) > _lo * (1 + VALUE_TOL), f"bands overlap at {_lo}/{_hi} kg/L"
# --- token hygiene: no derived value may collide with a printed figure, none may repeat ---
_PRINTED = ({r["gross"] for r in RUNS} | {r["tare"] for r in RUNS} | {r["fuel"] for r in RUNS})
assert len(_PRINTED) == 3 * len(RUNS), "printed raw figures must all be distinct"
for _target in [r["eff"] for r in RUNS] + [EFF_GAP]:
    for _p in _PRINTED:
        assert abs(_p - _target) > _target * VALUE_TOL, f"printed {_p} collides with band of {_target}"
for _r in RUNS:
    assert abs(_r["eff"] - EFF_GAP) > EFF_GAP * VALUE_TOL, "efficiency gap collides with an efficiency"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "205",
        "test_name": "Reasoning: subtract-then-divide ratio argmax over haulage runs "
                     "(most cargo per litre)",
        "difficulty_level": "8/10",
        "category": "Self-contained reasoning: derived-quantity argmax",
        "level": "reasoning",
        "weight": "short",
        # Nothing to visit: every figure is in the statement. No grounding/visit gate applies.
        "grounding_required": False,
    }


def _table() -> str:
    rows = "\n".join(
        "    {:<10}  {:>12}   {:>11}   {:>9}".format(
            r["name"], f"{r['gross']:,} kg", f"{r['tare']:,} kg", f"{r['fuel']} L")
        for r in sorted(RUNS, key=lambda r: r["name"])
    )
    return ("    Run         Gross weight   Tare weight   Fuel used\n"
            "    ----------  ------------   -----------   ---------\n" + rows)


def get_task_statement() -> str:
    return (
        "This is a self-contained arithmetic-reasoning task. Every number you need is printed "
        "below. Do NOT search the web, do NOT open any page, and do NOT look anything up — there "
        "is nothing to find; the whole task is computing and comparing.\n\n"
        "A haulier is reviewing six completed delivery runs. For each run you are given the GROSS "
        "weight of the loaded vehicle, its TARE weight (the empty vehicle plus pallets — weight "
        "that was hauled but is not cargo), and the fuel the run consumed.\n\n"
        f"{_table()}\n\n"
        "For EACH run compute how much CARGO it moved per litre of fuel, where\n"
        "    cargo carried    = gross weight - tare weight\n"
        "    cargo per litre  = cargo carried / fuel used\n"
        "(each division comes out exact — a whole number of kilograms per litre). Then report:\n"
        "  (a) which run moved the MOST cargo per litre of fuel, and what that figure is, in "
        "kilograms per litre. State it explicitly as a sentence of the form \"<Run> moved the most "
        "cargo per litre at <N> kg per litre\".\n"
        "  (b) the cargo per litre for ALL SIX runs (one line each: run -> kg per litre).\n"
        "  (c) which run was SECOND for cargo per litre, and its figure.\n"
        "  (d) which run had the HEAVIEST gross weight, what ITS cargo per litre is, and how many "
        "kg per litre separate it from the best run.\n\n"
        "Note: the most cargo-efficient run is NOT necessarily the one with the heaviest gross "
        "weight, nor the one that burned the least fuel, nor the one with the lightest tare. "
        "Subtract the tare first, then divide — dividing the GROSS weight by fuel gives a "
        "different, wrong ranking."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The run that moved the MOST cargo per litre, stated explicitly (the primary answer)",
        "That run's cargo per litre in kg per litre",
        "The cargo per litre of all six runs",
        "The SECOND-best run and its cargo per litre",
        "The heaviest-gross run, its cargo per litre, and its gap to the best run",
    ]


def get_success_criteria() -> List[str]:
    return [
        f"Names {WINNER['name']} as the most cargo-efficient run (NOT {DECOY['name']}, which had "
        f"the heaviest gross weight and carried the most cargo, NOT "
        f"{min(RUNS, key=lambda r: r['fuel'])['name']}, which burned the least fuel, and NOT "
        f"{NO_SUBTRACTION_PICK['name']}, which wins only if the tare is left in)",
        f"Reports the winning figure of {WINNER['eff']} kg per litre",
        "Reports all six computed cargo-per-litre figures (tare subtracted before dividing)",
        f"Identifies the runner-up ({RUNNER_UP['name']}, {RUNNER_UP['eff']} kg/L)",
        f"Computes the heaviest-gross run's figure ({DECOY['eff']} kg/L) and its {EFF_GAP} kg/L "
        "deficit to the winner",
    ]


# --------------------------------------------------------------------------------------------
# Text helpers (same shape as 204 — each task module stays self-contained by house style)
# --------------------------------------------------------------------------------------------
def _primary_text(result: Dict[str, Any]) -> str:
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


_NUM_RX = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\d)")


def _numbers(text: str) -> List[float]:
    out: List[float] = []
    for whole, frac in _NUM_RX.findall(text):
        try:
            out.append(float(whole.replace(",", "") + ("." + frac if frac else "")))
        except ValueError:
            continue
    return out


def _has_value(text: str, *targets: float) -> bool:
    nums = _numbers(text)
    return any(abs(v - t) <= t * VALUE_TOL for t in targets for v in nums)


# --------------------------------------------------------------------------------------------
# Winner-assertion detection — see 204 for the rationale: a complete report necessarily contains
# all six figures, so the winner's number alone is too weak; the keystone also needs a superlative
# tied to the CARGO-PER-LITRE metric whose nearest run name is the winner (nearest-before
# preferred, because English puts the subject first).
# --------------------------------------------------------------------------------------------
_SUP = r"(?:highest|most\s+efficient|most|best|greatest|top|maximum|\bmax\b|leader|winner)"
_METRIC = (r"(?:kg\s*(?:/|per)\s*(?:l\b|lit(?:re|er)s?)|cargo\s*(?:/|per)\s*(?:l\b|lit(?:re|er)s?)|"
           r"per[-\s]?lit(?:re|er)|/\s*l\b|efficien\w*|payload\s+per|cargo[-\s]per)")
_ASSERT_RX = re.compile(_SUP + r"[^.;\n]{0,30}" + _METRIC + r"|" + _METRIC + r"[^.;\n]{0,30}" + _SUP,
                        re.IGNORECASE)
_NEG_RX = re.compile(r"\bnot\b|n't\b|\bunlike\b|\bexcept\b|\bthan\b|\bdespite\b", re.IGNORECASE)
_ORDINAL_RX = re.compile(r"(?:second|2nd|third|3rd|next|runner[-\s]?up)[-\s]*$", re.IGNORECASE)
_NAME_WINDOW_BEFORE = 60
_NAME_WINDOW_AFTER = 40


def _nearest_run(text: str, start: int, end: int) -> Optional[Dict[str, Any]]:
    """The run an assertion at ``[start, end)`` is about: an overlapping name, else the nearest
    name ending before it (within 60 chars), else the nearest starting after it (within 40)."""
    before: List[tuple] = []
    after: List[tuple] = []
    for r in RUNS:
        for m in re.finditer(r["name_rx"], text, re.IGNORECASE):
            if m.start() < end and m.end() > start:
                return r
            if m.end() <= start:
                before.append((start - m.end(), r))
            elif m.start() >= end:
                after.append((m.start() - end, r))
    if before:
        d, r = min(before, key=lambda t: t[0])
        if d <= _NAME_WINDOW_BEFORE:
            return r
    if after:
        d, r = min(after, key=lambda t: t[0])
        if d <= _NAME_WINDOW_AFTER:
            return r
    return None


def _asserts_winner(text: str) -> bool:
    for m in _ASSERT_RX.finditer(text):
        start, end = m.span()
        pre = text[max(0, start - 20):start]
        if _NEG_RX.search(pre + m.group(0)) or _ORDINAL_RX.search(pre):
            continue
        owner = _nearest_run(text, start, end)
        if owner is not None and owner["key"] == WINNER["key"]:
            return True
    return False


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1): the winner's DERIVED figure (71 kg/L — printed nowhere,
    obtainable only by subtracting the tare and then dividing) AND the winner asserted as the most
    cargo-efficient run (or a terse primary slot naming the winner and no rival). The value
    requirement is what replaces the web suite's page-read grounding gate: an unworked 1-in-6 name
    guess earns nothing, and a no-subtraction solver reports the wrong figure for the wrong run.
    (This function deliberately consults no observability signal -- there is nothing to browse
    here -- so ``scripts/validator_lint.py`` reporting a [GATE] finding for the keystone is CORRECT
    and must not be "fixed"; only an [LLM] finding would be a real violation.)"""
    if not _has_value(_all_text(result), float(WINNER["eff"])):
        return False
    primary = _primary_text(result)
    named = {r["key"] for r in RUNS if re.search(r["name_rx"], primary, re.IGNORECASE)}
    if named == {WINNER["key"]}:
        return True
    return _asserts_winner(primary)


# --------------------------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------------------------
def validate_keystone_highest_cargo_rate(result: Dict[str, Any],
                                         observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the argmax of the DERIVED (subtract-then-divide) cargo per litre is
    Marlowe at 71 kg/L — not the heaviest-gross run, not the least-fuel run, not the lightest-tare
    run, and not the run that wins when the tare is left in."""
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_highest_cargo_rate",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (f"{WINNER['name']} named as most cargo per litre with {WINNER['eff']} kg/L"
                   if passed else
                   f"most cargo per litre ({WINNER['name']}, {WINNER['eff']} kg/L) "
                   f"missing/incorrect — beware {DECOY['name']} (heaviest gross), "
                   f"{min(RUNS, key=lambda r: r['fuel'])['name']} (least fuel) and "
                   f"{NO_SUBTRACTION_PICK['name']} (wins only if the tare is not subtracted)"),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the SIX subtract-then-divide computations were
    actually carried out and reported (run named AND a number inside that run's +/-0.5% band; the
    bands are provably disjoint, so no cross-crediting).

    Deliberately NOT short-circuited on the keystone — it measures how much of the six-way work
    happened even when the final argmax is botched.
    """
    text = _all_text(result)
    hits = [r["name"] for r in RUNS
            if re.search(r["name_rx"], text, re.IGNORECASE) and _has_value(text, float(r["eff"]))]
    total = len(RUNS)
    return {"check": "coverage", "passed": len(hits) == total, "score": len(hits) / total,
            "reason": f"{len(hits)}/{total} cargo-per-litre figures computed "
                      f"({', '.join(hits) or 'none'})"}


def validate_runner_up(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the SECOND-best run (Rothsay, 64 kg/L) identified as such. Short-circuits
    to 0 without the keystone, so a botched argmax cannot bank ordering credit."""
    if not _keystone_ok(result, observability):
        return {"check": "runner_up", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> runner-up not credited"}
    text = _all_text(result)
    value_ok = _has_value(text, float(RUNNER_UP["eff"]))
    cue = re.compile(
        RUNNER_UP["name_rx"] + r"[^.;\n]{0,60}(?:second|2nd|runner[-\s]?up|next\s+(?:best|highest|most))"
        r"|(?:second|2nd|runner[-\s]?up|next\s+(?:best|highest|most))[^.;\n]{0,60}"
        + RUNNER_UP["name_rx"]
        + r"|(?:^|\n)\s*(?:2[.)]|#\s*2)\s*[^\n]{0,40}" + RUNNER_UP["name_rx"],
        re.IGNORECASE)
    named_ok = bool(cue.search(text))
    score = (0.5 if value_ok else 0.0) + (0.5 if named_ok else 0.0)
    return {"check": "runner_up", "passed": score == 1.0, "score": score,
            "reason": (f"runner-up {RUNNER_UP['name']} named={named_ok}, "
                       f"{RUNNER_UP['eff']} kg/L reported={value_ok}")}


def validate_decoy_gap(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the trap made load-bearing — the HEAVIEST-GROSS run's own figure
    (54 kg/L) and its 17 kg/L deficit to the winner. Both require the decoy to be computed rather
    than merely dismissed. Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "decoy_gap", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> heaviest-gross arithmetic not credited"}
    text = _all_text(result)
    decoy_named = bool(re.search(DECOY["name_rx"], text, re.IGNORECASE))
    decoy_eff_ok = decoy_named and _has_value(text, float(DECOY["eff"]))
    gap_ok = _has_value(text, float(EFF_GAP))
    score = (0.5 if decoy_eff_ok else 0.0) + (0.5 if gap_ok else 0.0)
    return {"check": "decoy_gap", "passed": score == 1.0, "score": score,
            "reason": (f"heaviest-gross run {DECOY['name']} at {DECOY['eff']} kg/L "
                       f"reported={decoy_eff_ok}, gap {EFF_GAP} kg/L reported={gap_ok}")}


def get_validation_functions() -> List[callable]:
    return [
        validate_keystone_highest_cargo_rate,
        validate_coverage,
        validate_runner_up,
        validate_decoy_gap,
    ]


def get_llm_validation_function() -> callable:
    # Deterministic-only: no LLM judge in the reasoning suite (validator_lint [LLM] severity).
    return None
