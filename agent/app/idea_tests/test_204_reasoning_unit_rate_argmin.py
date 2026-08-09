"""
Test 204: Self-contained reasoning — derived unit-rate ARGMIN over bulk purchase options.
Level: graph (reasoning shape, not a browsing shape)   Weight: short   Difficulty: 7/10

SELF-CONTAINED / NO WEB. This task belongs to the "general reasoning" category: unlike every
task in the web suite it requires NO page visit at all, because the statement itself supplies
every raw number needed. The two properties that replace the web-grounding gate are:

  (1) SELF-CONTAINMENT — all twelve raw figures (six kit prices, six kit sizes) are printed in
      ``get_task_statement()``. Nothing may be looked up; nothing CAN be looked up.
  (2) ANSWER-SPACE NOVELTY — the numbers are procedurally generated, not a memorable published
      puzzle, so a model cannot have memorized the answer. See "Provenance" below.

Shape: the "ranking != raw value" trap. Six suppliers sell the same single-use diagnostic
reagent in sealed kits. The agent must DIVIDE (kit price / tests per kit) six times and take the
ARGMIN of the six quotients. No printed figure is the answer and no printed ordering matches it:

  * the LARGEST kit (Eastvale, 1,200 tests) is only 5th of 6 on cost per test;
  * the MOST EXPENSIVE kit (Eastvale again, $4,188) is likewise 5th;
  * the CHEAPEST kit by sticker price (Corwell, $1,624) is DEAD LAST on cost per test;
  * the SMALLEST kit (Corwell, 400 tests) is dead last too;
  * the winner (Brightkiln) is 3rd-largest by kit size and 4th of 6 by kit price -- unremarkable
    on every raw axis, so eyeballing any single printed column lands on the wrong supplier.

Grounding / anti-guess: ``grounding_required`` is False, so there is no ``visit.count`` to gate
on -- a 1-in-6 name guess would otherwise bank the keystone at p=0.167. The replacement gate is
that the keystone requires the correct DERIVED VALUE (the winner's cost per test) alongside the
winner's name: that number appears nowhere in the statement and can only be produced by doing
the division, so an unworked guess scores 0.

Ground truth (derived by reference computation -- NEVER by hand -- and cross-checked by a
SECOND, independently implemented computation that does no division at all: it ranks the options
by pairwise cross-multiplication, ``price_i * tests_j`` vs ``price_j * tests_i``. Both agree
exactly; see ``test_204_reasoning_unit_rate_argmin_validators_test.py``, which re-derives the
whole table at test time with the cross-multiplication solver, so these literals can never drift):

  supplier     kit price   tests/kit   cost per test          rank
  Brightkiln   $1,953        900       217 c  ($2.17)          1   <- ARGMIN (keystone)
  Aldervane    $1,896        800       237 c  ($2.37)          2   <- runner-up
  Fennhold     $2,448        960       255 c  ($2.55)          3
  Dunmarsh     $1,914        600       319 c  ($3.19)          4
  Eastvale     $4,188      1,200       349 c  ($3.49)          5   <- biggest kit + priciest kit
  Corwell      $1,624        400       406 c  ($4.06)          6   <- cheapest sticker price

  MARGIN: 217 c vs the runner-up's 237 c = 20 c, i.e. the runner-up is +9.2% more expensive per
  test. Every adjacent pair in the ranking is separated by >= 18 c (>= 7.6% relative), so the
  +/-0.5% acceptance bands used by the validators are mutually disjoint by a factor of ~15 and no
  plausible rounding or single arithmetic slip can flip the keystone or cross-credit a rate.

  ANNUAL ARITHMETIC (the gated secondary): 14,400 tests is divisible by ALL SIX kit sizes
  (lcm(400, 600, 800, 900, 960, 1200) = 14,400), so "buy the year's tests from one supplier" is
  a whole number of kits for every supplier and the deliverable has no purchasing ambiguity:
      Brightkiln  16 kits x $1,953 = $31,248   (= 14,400 x $2.17)
      Eastvale    12 kits x $4,188 = $50,256   (= 14,400 x $3.49)
      saving of the argmin over the biggest kit = $19,008
  Both figures were produced by the same two independent computations.

Provenance / procedural variation: the twelve raw figures were emitted by a seeded constrained
generator (seed 20426; kit sizes drawn from a plausible catalogue, per-test rates drawn from
210-430 c, kept only when price*100/size is a whole dollar amount) under the constraint set
encoded in the import-time assertions below: argmin != biggest kit, != priciest kit, != cheapest
kit, != smallest kit; cheapest-sticker option must rank >= 5th; biggest kit must rank >= 4th;
keystone margin in [9%, 20%]; every adjacent rate gap >= 6 c; no derived rate may coincide with
any printed price or kit size (token-collision hygiene for the regex validators); and a single
whole-kit-clean annual volume must exist for all six suppliers. Sibling task 205 was generated
from a different seed with a different scenario and shares no number with this one.
"""

from typing import Dict, Any, List, Optional
import re
from agent.app.idea_test_utils import extract_final_text


# --------------------------------------------------------------------------------------------
# Fixtures: the single source of truth for the statement, the validators and the invariants.
# ``rate_c`` is the DERIVED quantity (cost per test in cents) and is never printed in the task
# statement -- producing it is the work.
# --------------------------------------------------------------------------------------------
OPTIONS: List[Dict[str, Any]] = [
    {"key": "aldervane",  "name": "Aldervane",  "price": 1896, "tests": 800,
     "rate_c": 237, "name_rx": r"\baldervane\b"},
    {"key": "brightkiln", "name": "Brightkiln", "price": 1953, "tests": 900,
     "rate_c": 217, "name_rx": r"\bbrightkiln\b"},
    {"key": "corwell",    "name": "Corwell",    "price": 1624, "tests": 400,
     "rate_c": 406, "name_rx": r"\bcorwell\b"},
    {"key": "dunmarsh",   "name": "Dunmarsh",   "price": 1914, "tests": 600,
     "rate_c": 319, "name_rx": r"\bdunmarsh\b"},
    {"key": "eastvale",   "name": "Eastvale",   "price": 4188, "tests": 1200,
     "rate_c": 349, "name_rx": r"\beastvale\b"},
    {"key": "fennhold",   "name": "Fennhold",   "price": 2448, "tests": 960,
     "rate_c": 255, "name_rx": r"\bfennhold\b"},
]

_BY_RATE = sorted(OPTIONS, key=lambda o: o["rate_c"])
WINNER = _BY_RATE[0]            # Brightkiln  -- lowest cost per test  (keystone)
RUNNER_UP = _BY_RATE[1]         # Aldervane   -- second lowest
DECOY = max(OPTIONS, key=lambda o: o["tests"])   # Eastvale -- biggest kit (and priciest)

ANNUAL_TESTS = 14400
ANNUAL_BEST_COST = ANNUAL_TESTS * WINNER["rate_c"] // 100      # 31_248 dollars
ANNUAL_DECOY_COST = ANNUAL_TESTS * DECOY["rate_c"] // 100      # 50_256 dollars
ANNUAL_SAVING = ANNUAL_DECOY_COST - ANNUAL_BEST_COST           # 19_008 dollars

# Acceptance tolerance for a reported numeric value (+/-0.5%). Deliberately tight: every rate in
# the table is an EXACT integer number of cents, so the tolerance exists only to absorb rendering
# ("$2.17" / "217 c" / "217.0"), never to blur two options together -- see the disjointness
# assertion below.
VALUE_TOL = 0.005

# ---------------------------------- import-time invariants ----------------------------------
# The design constraints are asserted, not commented, so the fixtures can never silently drift
# out of the shape the task is claiming to test.
for _o in OPTIONS:
    assert _o["price"] * 100 % _o["tests"] == 0, f"{_o['name']}: rate is not a whole cent"
    assert _o["price"] * 100 // _o["tests"] == _o["rate_c"], f"{_o['name']}: declared rate is wrong"
assert len({o["rate_c"] for o in OPTIONS}) == len(OPTIONS), "rates must be distinct"
assert WINNER is min(OPTIONS, key=lambda o: o["rate_c"])
# --- the "ranking != raw value" traps: the winner is not the argmax/argmin of ANY raw column ---
assert WINNER is not max(OPTIONS, key=lambda o: o["tests"]), "winner must not be the biggest kit"
assert WINNER is not min(OPTIONS, key=lambda o: o["tests"]), "winner must not be the smallest kit"
assert WINNER is not min(OPTIONS, key=lambda o: o["price"]), "winner must not be the cheapest kit"
assert WINNER is not max(OPTIONS, key=lambda o: o["price"]), "winner must not be the priciest kit"
_RANK = {o["key"]: i for i, o in enumerate(_BY_RATE)}          # 0 = best
assert _RANK[max(OPTIONS, key=lambda o: o["tests"])["key"]] >= 3, "biggest kit must be mid-or-worse"
assert _RANK[min(OPTIONS, key=lambda o: o["price"])["key"]] >= 4, "cheapest sticker must be near-worst"
assert _RANK[max(OPTIONS, key=lambda o: o["price"])["key"]] >= 2, "priciest kit must not be top-2"
# --- keystone margin: wide enough that no rounding or single slip can flip the argmin ---
_MARGIN = (RUNNER_UP["rate_c"] - WINNER["rate_c"]) / WINNER["rate_c"]
assert 0.09 <= _MARGIN <= 0.20, f"keystone margin {_MARGIN:.3f} outside the designed band"
# --- acceptance bands must be mutually disjoint (no cross-crediting one option's rate to another)
_SORTED_RATES = [o["rate_c"] for o in _BY_RATE]
for _i in range(len(_SORTED_RATES) - 1):
    _lo, _hi = _SORTED_RATES[_i], _SORTED_RATES[_i + 1]
    assert _hi * (1 - VALUE_TOL) > _lo * (1 + VALUE_TOL), f"bands overlap at {_lo}/{_hi} c"
# --- no derived rate may collide with a printed raw number (regex token hygiene) ---
_PRINTED = {o["price"] for o in OPTIONS} | {o["tests"] for o in OPTIONS}
for _o in OPTIONS:
    for _p in _PRINTED:
        assert abs(_p - _o["rate_c"]) > _o["rate_c"] * VALUE_TOL, \
            f"printed {_p} collides with the {_o['name']} rate band"
# --- the annual-volume deliverable must be whole-kit clean for EVERY supplier ---
for _o in OPTIONS:
    assert ANNUAL_TESTS % _o["tests"] == 0, f"{ANNUAL_TESTS} is not a whole number of {_o['name']} kits"
    assert ANNUAL_TESTS // _o["tests"] * _o["price"] == ANNUAL_TESTS * _o["rate_c"] // 100, \
        f"{_o['name']}: whole-kit cost disagrees with the per-test rate"
assert ANNUAL_SAVING == ANNUAL_TESTS * (DECOY["rate_c"] - WINNER["rate_c"]) // 100


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "204",
        "test_name": "Reasoning: unit-rate argmin over bulk purchase options (lowest cost per test)",
        "difficulty_level": "7/10",
        "category": "Self-contained reasoning: derived-quantity argmin",
        "level": "reasoning",
        "weight": "short",
        # No web lookup exists for this task: every fact is in the statement. The harness must NOT
        # apply a grounding/visit gate, and the validators below deliberately have none.
        "grounding_required": False,
    }


def _table() -> str:
    rows = "\n".join(
        "    {:<12}  {:>13}   {:>13}".format(o["name"], f"${o['price']:,}", f"{o['tests']:,}")
        for o in sorted(OPTIONS, key=lambda o: o["name"])
    )
    return ("    Supplier      Price per kit   Tests per kit\n"
            "    ------------  -------------   -------------\n" + rows)


def get_task_statement() -> str:
    return (
        "This is a self-contained arithmetic-reasoning task. Every number you need is printed "
        "below. Do NOT search the web, do NOT open any page, and do NOT look anything up — there "
        "is nothing to find; the whole task is computing and comparing.\n\n"
        "A laboratory is comparing six bulk purchase options for the same single-use diagnostic "
        "reagent. Each supplier sells it only in sealed kits: one kit contains a fixed number of "
        "tests and costs a fixed price. The kits are interchangeable in quality, there are no "
        "other discounts, shipping or fees, and no partial kits are sold.\n\n"
        f"{_table()}\n\n"
        "For EACH supplier compute the COST PER TEST = price per kit / tests per kit. Then "
        "report:\n"
        "  (a) which supplier has the LOWEST cost per test, and what that cost per test is "
        "(in cents, or equivalently in dollars to the cent). State it explicitly as a sentence of "
        "the form \"<Supplier> has the lowest cost per test at <N> cents per test\".\n"
        "  (b) the cost per test for ALL SIX suppliers (one line each: supplier -> cost per "
        "test).\n"
        "  (c) which supplier has the SECOND-lowest cost per test, and its cost per test.\n"
        f"  (d) the lab needs exactly {ANNUAL_TESTS:,} tests this year. {ANNUAL_TESTS:,} divides "
        "evenly by every kit size above, so any single supplier can supply it in whole kits. "
        f"Report the TOTAL cost of buying all {ANNUAL_TESTS:,} tests from the supplier with the "
        "lowest cost per test, and how many dollars that SAVES compared with buying all "
        f"{ANNUAL_TESTS:,} tests from the supplier with the LARGEST kit.\n\n"
        "Note: the lowest cost per test is NOT necessarily the supplier with the lowest kit "
        "price, nor the one with the largest kit, nor the one with the smallest kit. Do the six "
        "divisions before you answer."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The supplier with the LOWEST cost per test, stated explicitly (the primary answer)",
        "That supplier's cost per test (in cents or dollars to the cent)",
        "The cost per test for all six suppliers",
        "The SECOND-lowest supplier and its cost per test",
        f"Total cost of {ANNUAL_TESTS:,} tests at the best rate, and the saving versus the "
        "largest-kit supplier",
    ]


def get_success_criteria() -> List[str]:
    return [
        f"Names {WINNER['name']} as the lowest cost per test (NOT {DECOY['name']}, which has the "
        f"largest and the most expensive kit, and NOT "
        f"{min(OPTIONS, key=lambda o: o['price'])['name']}, which has the lowest kit price)",
        f"Reports the winning cost per test of {WINNER['rate_c']} cents (${WINNER['rate_c'] / 100:.2f})",
        "Reports all six computed cost-per-test values",
        f"Identifies the runner-up ({RUNNER_UP['name']}, {RUNNER_UP['rate_c']} cents)",
        f"Computes the {ANNUAL_TESTS:,}-test totals: ${ANNUAL_BEST_COST:,} at the best rate, "
        f"saving ${ANNUAL_SAVING:,} versus {DECOY['name']}",
    ]


# --------------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------------
def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text: ``deliverables[0]`` (the contract's primary slot) when present,
    else the final deliverable blob."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Everything the agent reported (final deliverable + every deliverable slot), so a figure
    placed in a table outside the primary slot still counts for coverage/value checks."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


# Numeric scanner: "217", "2.17", "$1,953", "14,400" -> floats. The lookbehind stops it matching
# the tail of a larger token; the trailing (?!\d) stops "21" matching inside "217" while still
# allowing a unit suffix to follow directly ("217c", "217¢").
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
    """True when some number in ``text`` lands within +/-VALUE_TOL of ANY target rendering."""
    nums = _numbers(text)
    return any(abs(v - t) <= t * VALUE_TOL for t in targets for v in nums)


def _rate_renderings(option: Dict[str, Any]) -> List[float]:
    """A cost per test may legitimately be written in cents (217) or dollars (2.17)."""
    return [float(option["rate_c"]), option["rate_c"] / 100.0]


# --------------------------------------------------------------------------------------------
# Winner-assertion detection
#
# The report necessarily contains ALL SIX rates (deliverable b), so "the winner's number appears
# somewhere" is far too weak on its own: a wrong-argmin answer with a correct table would bank
# the keystone. The keystone therefore also requires an ASSERTION -- a superlative tied to the
# per-test METRIC -- whose nearest supplier name is the winner. Nearest-BEFORE wins (English puts
# the subject first: "Brightkiln has the lowest cost per test"), falling back to nearest-after
# ("the lowest cost per test is Brightkiln"), so "Compared with Brightkiln, Corwell has the
# lowest cost per test" resolves to Corwell -- correctly denying credit.
# --------------------------------------------------------------------------------------------
_SUP = (r"(?:cheapest|lowest|least|lowest[-\s]cost|best|minimum|\bmin\b|"
        r"most\s+cost[-\s]?effective|greatest\s+value)")
_METRIC = (r"(?:per[-\s]?test|per[-\s]?unit|unit[-\s](?:cost|price|rate)|"
           r"cost[-\s]?per[-\s]?test|/\s*test)")
_ASSERT_RX = re.compile(_SUP + r"[^.;\n]{0,30}" + _METRIC + r"|" + _METRIC + r"[^.;\n]{0,30}" + _SUP,
                        re.IGNORECASE)
# Negation / ordinal guards, checked only in the 20 chars immediately BEFORE the assertion and
# inside it -- never after, so "...is the cheapest per test, second is Aldervane" is not vetoed by
# its own trailing clause.
_NEG_RX = re.compile(r"\bnot\b|n't\b|\bunlike\b|\bexcept\b|\bthan\b|\bavoid\b", re.IGNORECASE)
_ORDINAL_RX = re.compile(r"(?:second|2nd|third|3rd|next|runner[-\s]?up)[-\s]*$", re.IGNORECASE)
_NAME_WINDOW_BEFORE = 60
_NAME_WINDOW_AFTER = 40


def _nearest_option(text: str, start: int, end: int) -> Optional[Dict[str, Any]]:
    """The supplier an assertion at ``[start, end)`` is about: overlapping name, else the nearest
    name ending before it (within 60 chars), else the nearest starting after it (within 40)."""
    before: List[tuple] = []
    after: List[tuple] = []
    for o in OPTIONS:
        for m in re.finditer(o["name_rx"], text, re.IGNORECASE):
            if m.start() < end and m.end() > start:
                return o                              # the name sits inside the assertion
            if m.end() <= start:
                before.append((start - m.end(), o))
            elif m.start() >= end:
                after.append((m.start() - end, o))
    if before:
        d, o = min(before, key=lambda t: t[0])
        if d <= _NAME_WINDOW_BEFORE:
            return o
    if after:
        d, o = min(after, key=lambda t: t[0])
        if d <= _NAME_WINDOW_AFTER:
            return o
    return None


def _asserts_winner(text: str) -> bool:
    for m in _ASSERT_RX.finditer(text):
        start, end = m.span()
        pre = text[max(0, start - 20):start]
        if _NEG_RX.search(pre + m.group(0)) or _ORDINAL_RX.search(pre):
            continue
        owner = _nearest_option(text, start, end)
        if owner is not None and owner["key"] == WINNER["key"]:
            return True
    return False


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1).

    Requires BOTH:
      * the DERIVED value -- the winner's cost per test (217 c / $2.17), which is printed nowhere
        in the statement and can only come from doing the division. This is what replaces the web
        suite's page-read grounding gate: an unworked 1-in-6 name guess earns nothing.
        (This function deliberately consults no observability signal -- there is nothing to browse
        here -- so ``scripts/validator_lint.py`` reporting a [GATE] finding for the keystone is
        CORRECT and must not be "fixed"; only an [LLM] finding would be a real violation.)
      * the winner ASSERTED as the lowest cost per test (or a terse primary slot that names the
        winner and no rival), so a wrong argmin sitting next to a correct six-row table fails.
    """
    if not _has_value(_all_text(result), *_rate_renderings(WINNER)):
        return False
    primary = _primary_text(result)
    named = {o["key"] for o in OPTIONS if re.search(o["name_rx"], primary, re.IGNORECASE)}
    if named == {WINNER["key"]}:
        return True
    return _asserts_winner(primary)


# --------------------------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------------------------
def validate_keystone_lowest_unit_rate(result: Dict[str, Any],
                                       observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the argmin of the DERIVED cost per test is Brightkiln at 217 c —
    not the biggest kit, not the priciest kit, not the cheapest sticker price."""
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_lowest_unit_rate",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (f"{WINNER['name']} named as the lowest cost per test with "
                   f"{WINNER['rate_c']} c (${WINNER['rate_c'] / 100:.2f}) present" if passed
                   else f"lowest cost per test ({WINNER['name']}, {WINNER['rate_c']} c) "
                        f"missing/incorrect — beware {DECOY['name']} (biggest + priciest kit) "
                        f"and {min(OPTIONS, key=lambda o: o['price'])['name']} (cheapest kit)"),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the SIX divisions were actually carried out and
    reported (supplier named AND a number inside that supplier's +/-0.5% rate band, in cents or
    dollars; the bands are provably disjoint, so no cross-crediting).

    Deliberately NOT short-circuited on the keystone: it measures how much of the six-way work
    was done even when the final argmin is botched — the axis that separates a structured agent
    that computes all six rates from one that eyeballs a column and answers immediately.
    """
    text = _all_text(result)
    hits = [o["name"] for o in OPTIONS
            if re.search(o["name_rx"], text, re.IGNORECASE) and _has_value(text, *_rate_renderings(o))]
    n = len(OPTIONS)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} cost-per-test values computed ({', '.join(hits) or 'none'})"}


def validate_runner_up(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the SECOND-lowest supplier (Aldervane, 237 c) identified as such.
    Short-circuits to 0 without the keystone, so a botched argmin cannot bank ordering credit."""
    if not _keystone_ok(result, observability):
        return {"check": "runner_up", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> runner-up not credited"}
    text = _all_text(result)
    value_ok = _has_value(text, *_rate_renderings(RUNNER_UP))
    cue = re.compile(
        RUNNER_UP["name_rx"] + r"[^.;\n]{0,60}(?:second|2nd|runner[-\s]?up|next\s+(?:best|lowest|cheapest))"
        r"|(?:second|2nd|runner[-\s]?up|next\s+(?:best|lowest|cheapest))[^.;\n]{0,60}"
        + RUNNER_UP["name_rx"]
        + r"|(?:^|\n)\s*(?:2[.)]|#\s*2)\s*[^\n]{0,40}" + RUNNER_UP["name_rx"],
        re.IGNORECASE)
    named_ok = bool(cue.search(text))
    score = (0.5 if value_ok else 0.0) + (0.5 if named_ok else 0.0)
    return {"check": "runner_up", "passed": score == 1.0, "score": score,
            "reason": (f"runner-up {RUNNER_UP['name']} named={named_ok}, "
                       f"{RUNNER_UP['rate_c']} c reported={value_ok}")}


def validate_annual_arithmetic(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the two-step derived arithmetic on top of the argmin — the annual cost at
    the best rate ($31,248) and the saving over the largest-kit supplier ($19,008). Forces the
    decoy's rate to actually be computed rather than dismissed, and short-circuits to 0 without
    the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "annual_arithmetic", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> annual-cost arithmetic not credited"}
    text = _all_text(result)
    total_ok = _has_value(text, float(ANNUAL_BEST_COST))
    saving_ok = _has_value(text, float(ANNUAL_SAVING))
    score = (0.5 if total_ok else 0.0) + (0.5 if saving_ok else 0.0)
    return {"check": "annual_arithmetic", "passed": score == 1.0, "score": score,
            "reason": (f"{ANNUAL_TESTS:,}-test total ${ANNUAL_BEST_COST:,} reported={total_ok}, "
                       f"saving ${ANNUAL_SAVING:,} vs {DECOY['name']} reported={saving_ok}")}


def get_validation_functions() -> List[callable]:
    return [
        validate_keystone_lowest_unit_rate,
        validate_coverage,
        validate_runner_up,
        validate_annual_arithmetic,
    ]


def get_llm_validation_function() -> callable:
    # Deterministic-only: this category holds itself to the same zero-LLM-judge bar as the web
    # suite (validator_lint [LLM] severity).
    return None
