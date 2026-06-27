"""
Test 088: Tier 5 (graph) — argmax over a COMPUTED RATIO (mean discharge / total length),
          different domain from 079 (hydro-stations), 059 (football), 064 (lakes).
Level: graph   Weight: long   Difficulty: 9/10

The agent must determine which of FIVE rivers has the highest MEAN-DISCHARGE-TO-LENGTH ratio,
where that ratio is NOT printed anywhere as a single figure and must be COMPUTED by DIVIDING
two looked-up infobox numbers:

    ratio = mean discharge (m³/s)  /  total length (km)
            [ = average cubic-metres per second contributed per km of river course ]

The five rivers span four continents and include one DOUBLE-DECOY (Mekong, Southeast Asia),
which has BOTH the highest mean discharge (16,000 m³/s) AND the longest length (4,350 km) in
the set.  A "pick the biggest/most powerful river" shortcut lands on Mekong — the wrong answer.
The ratio winner is the Fly River (Papua New Guinea), an extremely obscure lowland river that
nobody would intuit as the winner: its moderate discharge (6,500 m³/s) over a very short course
(1,060 km) gives it the highest ratio in the group.

    Fly River             Papua New Guinea
    Mekong                Southeast Asia (China/Laos/Thailand/Cambodia/Vietnam)
    Fraser River          British Columbia, Canada
    Niger River           West Africa
    Salween River         China / Myanmar

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): shortcuts to the "biggest" river (Mekong, highest discharge +
    longest) or guesses a well-known name; cannot correctly divide all ten figures and compare
    five quotients without reading.
  * frontier sequential (ReAct): looks up all ten figures, divides and compares → decent.
  * graph_compiled: five parallel leaves each fetch ONE river's two figures from ONE page;
    aggregation owns every division AND the argmax, forced to WRITE OUT each division before
    concluding → cheap executor is rescued; a diverse-grounding reranker can re-derive.

The trap is engineered three ways:
  (1) the ratio is computed — the agent must DIVIDE discharge by length for each river and
      then compare the five quotients; no page states the cross-river ranking;
  (2) the ratio winner DIFFERS from BOTH raw rankings — the highest-discharge river AND the
      longest river are the SAME entity (Mekong), which is NOT the winner, so "pick the
      biggest" fails on BOTH axes simultaneously;
  (3) the winner (Fly River) has the LOWEST length of the five (1,060 km) and ranks 4th by
      discharge — it is the least obvious river in the group.

Ground truth (verified against live English Wikipedia 2026-06-27 — each river's infobox
'Discharge' and 'Length' rows; all figures are whole integers in their natural units):

  river          region                     discharge(m³/s)  length(km)  ratio
  Fly River      Papua New Guinea                     6,500       1,060   6.132  <- ARGMAX
  Mekong         Southeast Asia                      16,000       4,350   3.678  <- 2nd (decoy A+B)
  Fraser River   British Columbia, Canada             3,475       1,375   2.527
  Niger River    West Africa                          8,570       4,200   2.040
  Salween River  China / Myanmar                      6,600       3,289   2.007

    https://en.wikipedia.org/wiki/Fly_River
    https://en.wikipedia.org/wiki/Mekong
    https://en.wikipedia.org/wiki/Fraser_River
    https://en.wikipedia.org/wiki/Niger_River
    https://en.wikipedia.org/wiki/Salween_River

  RATIO ARGMAX = Fly River (6.132 m³/(s·km)). Runner-up = Mekong (3.678 m³/(s·km)), so
  the margin is +2.454 m³/(s·km) absolute (~+66.7% relative). Even a worst-case ±500 m³/s
  misread of Fly River's discharge (reading 6,000 instead of 6,500) gives ratio 5.660 →
  margin vs Mekong still ~53.8%; a ±100 km misread of Fly's length (reading 1,160 instead
  of 1,060) gives ratio 5.603 → margin still ~52.3%. The keystone cannot be flipped by any
  single plausible figure misread.

  DOUBLE-DECOY / WIDE-MARGIN checks (confirmed live, asserted at import time):
    (a) ratio-winner (Fly River) != highest-DISCHARGE river (Mekong, 16,000 m³/s) -> True
    (b) ratio-winner (Fly River) != longest         river (Mekong, 4,350 km)       -> True
    (c) winner ratio (6.132) exceeds 2nd place (Mekong, 3.678) by ~66.7% (>> 15%) -> True
    by discharge: Mekong 16,000 > Niger 8,570 > Salween 6,600 > Fly 6,500 > Fraser 3,475
                                                                             -> Mekong
    by length:    Mekong 4,350 > Niger 4,200 > Salween 3,289 > Fraser 1,375 > Fly 1,060
                                                                             -> Mekong
    by ratio:  Fly 6.132 > Mekong 3.678 > Fraser 2.527 > Niger 2.040 > Salween 2.007
                                                                         -> Fly River
    => the single most salient river (Mekong, largest by BOTH raw quantities) is the doubly-
       baited decoy; the winner (Fly River) ranks 4th by discharge and is the shortest.

  ANTI-PARAMETRIC: Mekong is the obvious guess — it is Southeast Asia's dominant river and
  one of the world's major rivers by volume and length. But its long course (4,350 km) dilutes
  its discharge advantage, giving it only a 3.678 ratio vs. the obscure Fly River's 6.132.
  Neither ratio value nor the discharge/length division is recallable; the agent must READ
  both figures and DIVIDE. The winner (Fly River) is known mainly as a biodiversity hotspot
  in Papua New Guinea — its high discharge-to-length ratio is not a commonly cited fact.

  KEYSTONE = Fly River (the argmax). Secondary (gated) = its ratio value ~6.132 m³/(s·km)
  (accepted ±3%). All ten figures are distinct across rivers, so the un-gated coverage
  diagnostic is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ---- verified fixtures (single source of truth for statement, validators, and plan) ----
# ``ratio`` shown for provenance only; nothing here is leaked into the task statement or plan.
ENTITIES: List[Dict[str, Any]] = [
    {
        "key": "fly",
        "name": "Fly River",
        "region": "Papua New Guinea",
        "discharge_m3s": 6_500,    # Wikipedia infobox: "6,500 m3/s" at Fly Delta
        "length_km":    1_060,     # Wikipedia infobox: "1,060 km (660 mi) Fly River"
        "ratio": 6_500 / 1_060,    # 6.132 m³/(s·km) <- WINNER (highest ratio)
        "winner": True,
        "name_rx":      r"\bfly\s+river\b",
        "discharge_rx": r"(?<!\d)6[,\s]?500(?!\d)",
        "length_rx":    r"(?<!\d)1[,\s]?060(?!\d)",
        "slug_rx":      r"wiki/fly_river",
    },
    {
        "key": "mekong",
        "name": "Mekong",
        "region": "Southeast Asia",
        "discharge_m3s": 16_000,   # Wikipedia infobox: "average 16,000 m3/s"
        "length_km":     4_350,    # Wikipedia infobox: "4,350 km (2,700 mi)"
        "ratio": 16_000 / 4_350,   # 3.678 m³/(s·km) <- runner-up AND highest discharge+length decoy
        "winner": False,
        "name_rx":      r"\bmekong\b",
        "discharge_rx": r"(?<!\d)16[,\s]?000(?!\d)",
        "length_rx":    r"(?<!\d)4[,\s]?350(?!\d)",
        "slug_rx":      r"wiki/mekong",
    },
    {
        "key": "fraser",
        "name": "Fraser River",
        "region": "British Columbia, Canada",
        "discharge_m3s": 3_475,    # Wikipedia infobox: "average 3,475 m3/s"
        "length_km":    1_375,     # Wikipedia infobox: "Length 1,375 km (854 mi)"
        "ratio": 3_475 / 1_375,    # 2.527 m³/(s·km)
        "winner": False,
        "name_rx":      r"\bfraser\b",
        "discharge_rx": r"(?<!\d)3[,\s]?475(?!\d)",
        "length_rx":    r"(?<!\d)1[,\s]?375(?!\d)",
        "slug_rx":      r"wiki/fraser_river",
    },
    {
        "key": "niger",
        "name": "Niger River",
        "region": "West Africa",
        "discharge_m3s": 8_570,    # Wikipedia infobox: "270.5 km3/a (8,570 m3/s)" at Niger Delta
        "length_km":    4_200,     # Wikipedia infobox: "Length 4,200 km (2,600 mi)"
        "ratio": 8_570 / 4_200,    # 2.040 m³/(s·km)
        "winner": False,
        "name_rx":      r"\bniger\b",
        "discharge_rx": r"(?<!\d)8[,\s]?570(?!\d)",
        "length_rx":    r"(?<!\d)4[,\s]?200(?!\d)",
        "slug_rx":      r"wiki/niger_river",
    },
    {
        "key": "salween",
        "name": "Salween River",
        "region": "China / Myanmar",
        "discharge_m3s": 6_600,    # Wikipedia infobox: "average 6,600 m3/s"
        "length_km":    3_289,     # Wikipedia infobox: "Length 3,289 km (2,044 mi)"
        "ratio": 6_600 / 3_289,    # 2.007 m³/(s·km)
        "winner": False,
        "name_rx":      r"\bsalween\b",
        "discharge_rx": r"(?<!\d)6[,\s]?600(?!\d)",
        "length_rx":    r"(?<!\d)3[,\s]?289(?!\d)",
        "slug_rx":      r"wiki/salween_river",
    },
]

WINNER = next(e for e in ENTITIES if e["winner"])          # Fly River — ratio argmax
WINNER_RATIO = WINNER["discharge_m3s"] / WINNER["length_km"]   # ≈ 6.132 m³/(s·km)
RATIO_TOL    = 0.03                                         # ±3% on the secondary value

# ---- DOUBLE-DECOY / WIDE-MARGIN invariants: asserted at import so fixtures can never
# ---- silently drift out of the design (winner ≠ either raw argmax; margin > 15%). ------
_BY_DISCHARGE = max(ENTITIES, key=lambda e: e["discharge_m3s"])
_BY_LENGTH    = max(ENTITIES, key=lambda e: e["length_km"])
_BY_RATIO     = sorted(ENTITIES, key=lambda e: e["discharge_m3s"] / e["length_km"],
                        reverse=True)
assert _BY_RATIO[0] is WINNER,        "ratio argmax must be the declared winner"
assert _BY_DISCHARGE is not WINNER,   "winner must NOT be the highest-discharge river (decoy A)"
assert _BY_LENGTH    is not WINNER,   "winner must NOT be the longest river (decoy B)"
_W_RATIO         = WINNER["discharge_m3s"] / WINNER["length_km"]
_RUNNER_UP_RATIO = _BY_RATIO[1]["discharge_m3s"] / _BY_RATIO[1]["length_km"]
assert (_W_RATIO - _RUNNER_UP_RATIO) / _RUNNER_UP_RATIO > 0.15, \
    "keystone margin must exceed 15% (currently ~66.7%)"

# ---- winner / others regexes used by the keystone gate ----------------------------------
_WINNER_RX = WINNER["name_rx"]    # r"\bfly\s+river\b"
_OTHERS    = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# Comparative / superlative triggers asserting a winner.
_SUP = (r"more|larger|greater|higher|bigger|largest|greatest|highest|biggest|most|maximum"
        r"|best|top|efficient|superior|leading|peak")

# Keystone winner detection: Fly River tied to a 'highest ratio' assertion, in either
# direction, with the proximity window forbidden from crossing into ANY other river's name.
#   dir 1  (subject -> superlative): "Fly River … has the highest ratio"
#   dir 2  (superlative -> subject): "the highest ratio belongs to the Fly River"
_FLY_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Ratio-value detection for the winner's ratio ~6.132 m³/(s·km) (±3% = 5.948–6.316):
#   Form A: decimal e.g. "6.13", "6.132", "6.1"   (numeric range check)
#   Form B: explicit fraction "6,500 / 1,060"
_RATIO_NUM  = re.compile(r"(?<!\d)(\d+\.\d{1,4})(?!\d)")
_RATIO_FRAC = re.compile(r"\b6[,\s]?500\s*/\s*1[,\s]?060\b")


# =====================================================================
#  CONTRACT FUNCTIONS
# =====================================================================

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "088",
        "test_name": (
            "Tier 5: Computed-ratio argmax, wide-margin "
            "(highest mean-discharge-to-length ratio, rivers)"
        ),
        "difficulty_level": "9/10",
        "category": "Quantitative reasoning + computed-ratio argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['region']})"
        for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not "
        "guess from memory). For EACH of the following five rivers, open the river's Wikipedia "
        "page and read TWO numbers from the infobox: (A) the river's mean (average) DISCHARGE "
        "in m³/s (cubic metres per second) and (B) its total LENGTH in km (kilometres):\n"
        f"{listing}\n\n"
        "Then COMPUTE, for each river, its DISCHARGE-TO-LENGTH ratio = mean discharge (m³/s) / "
        "total length (km). This quotient gives the average discharge contributed per kilometre "
        "of river course — no Wikipedia page prints this cross-river comparison, you must "
        "divide the two figures yourself. COMPARE the five ratios and determine which river has "
        "the HIGHEST discharge-to-length ratio. Note: the answer is the largest RATIO, which "
        "is NOT necessarily the river with the largest mean discharge, nor the one with the "
        "greatest total length.\n\n"
        "Report (a) which river has the highest discharge-to-length ratio (a single river name "
        "— the keystone), (b) that river's computed ratio value (in m³/(s·km)), (c) all five "
        "rivers' mean discharge (m³/s) and total length (km) (the ten figures you looked up), "
        "and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which river has the highest discharge-to-length ratio (the primary answer / keystone)",
        "That winning river's computed ratio value (m³/(s·km))",
        "All five rivers' mean discharge (m³/s) and total length (km) — the ten looked-up figures",
        "Source URL for each river's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per river, a five-way fan-out)",
        "Correctly names Fly River as the highest discharge-to-length ratio (NOT the Mekong, "
        "which is the largest by BOTH mean discharge AND total length)",
        "Reports the winner's ratio near 6.132 m³/(s·km) (within ±3%)",
        "Gathers all ten figures (discharge and length for each of the five rivers)",
        "Cites the five source pages",
    ]


# ---- internal helpers ---------------------------------------------------

def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text: prefer deliverables[0] when present; fall back to final_deliverable."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: final deliverable + every deliverable slot concatenated."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: deliverables[0] names Fly River as the highest-ratio river.

    Word-bounded, and NOT satisfied by merely listing Fly River among the five: when another
    river is also named, Fly River must be the one tied to a 'highest ratio' assertion
    (tempered so a rival named as the winner, or 'higher than Fly River', never counts).
    A terse primary answer naming only the winner (Fly River, with no rival in the slot)
    also passes.
    """
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True   # names only the winner, no rival present
    return bool(_FLY_WINS.search(text))


def _ratio_value_present(text: str) -> bool:
    """True when a number in the winner's ratio band appears.

    Accepts:
      Form A: decimal in [5.948, 6.316] e.g. "6.132", "6.13", "6.1"
      Form B: explicit fraction "6,500 / 1,060"
    """
    lo_r = WINNER_RATIO * (1.0 - RATIO_TOL)   # ~5.948
    hi_r = WINNER_RATIO * (1.0 + RATIO_TOL)   # ~6.316

    # Form A: decimal
    for raw in _RATIO_NUM.findall(text):
        try:
            v = float(raw)
            if lo_r <= v <= hi_r:
                return True
        except ValueError:
            pass

    # Form B: explicit fraction
    return bool(_RATIO_FRAC.search(text))


# =====================================================================
#  VALIDATION FUNCTIONS
# =====================================================================

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a five-way fan-out wants one page per river."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 4,
        "score": min(1.0, n / 5.0),
        "reason": f"{n} visit(s) (target ≥5: one page per river; ≥4 to pass)",
    }


def validate_keystone_argmax(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the highest discharge-to-length ratio is the Fly River —
    NOT the Mekong, which is the largest by BOTH mean discharge (16,000 m³/s) AND total
    length (4,350 km). A model that shortcuts to a raw quantity or guesses the most salient
    name (Mekong / Niger) mis-picks.
    """
    passed = _keystone_ok(result)
    return {
        "check": "keystone_argmax",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Fly River named as the highest discharge-to-length ratio"
            if passed else
            "Highest-ratio river (Fly River) missing/incorrect — beware: the Mekong is the "
            "largest by BOTH mean discharge (16,000 m³/s) AND total length (4,350 km), but "
            "its ratio (3.678 m³/(s·km)) is well below Fly River (6.132)"
        ),
    }


def validate_coverage(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the five rivers' figure-pairs were gathered.

    A river's inputs are credited only when BOTH its mean discharge AND its total length
    appear (alongside its name). All ten figures are distinct (no cross-crediting).
    Deliberately NOT gated on the keystone — it measures structured fan-out, not correctness.
    """
    text = _all_text(result)
    hits = [
        e["name"]
        for e in ENTITIES
        if (re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["discharge_rx"], text)
            and re.search(e["length_rx"], text))
    ]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} rivers' discharge+length gathered "
            f"({', '.join(hits) or 'none'})"
        ),
    }


def validate_winner_ratio(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: the winner's computed ratio (~6.132 m³/(s·km), within ±3%).
    Short-circuits to 0 when the keystone is absent, so a guessed winner cannot bank the
    value credit.
    """
    if not _keystone_ok(result):
        return {
            "check": "winner_ratio",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → ratio value not credited",
        }
    ok = _ratio_value_present(_all_text(result))
    lo_r = WINNER_RATIO * (1.0 - RATIO_TOL)
    hi_r = WINNER_RATIO * (1.0 + RATIO_TOL)
    return {
        "check": "winner_ratio",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"winner's ratio in {lo_r:.3f}–{hi_r:.3f} m³/(s·km) present"
            if ok else
            f"no ratio in {lo_r:.3f}–{hi_r:.3f} m³/(s·km) "
            f"(expected ~{WINNER_RATIO:.3f} m³/(s·km))"
        ),
    }


def validate_citation(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: cites the river pages. Short-circuits to 0 when keystone absent."""
    if not _keystone_ok(result):
        return {
            "check": "citation",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → source URLs not credited",
        }
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {
        "check": "citation",
        "passed": cited >= 3,
        "score": cited / n,
        "reason": f"{cited}/{n} river pages cited",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_argmax,
        validate_coverage,
        validate_winner_ratio,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None → the harness applies its default structured rubric judge, as in 064/079.
    return None


# =====================================================================
#  COMPILED PLAN  (offline-authored leak-free scaffold)
# =====================================================================

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out / aggregate scaffold for the ``graph_compiled`` variant.

    FIVE INDEPENDENT parallel leaves: for each of the five GIVEN rivers, one leaf opens that
    river's Wikipedia page and reads its two infobox figures — mean discharge (m³/s) and
    total length (km) — both on the SAME page, so one page-read per river. ALL arithmetic —
    every discharge/length division and the final argmax — lives only in the aggregation step,
    which is forced to WRITE OUT each per-river division explicitly before concluding, rescuing
    the cheap executor and allowing a diverse-grounding reranker to re-derive and catch slips.

    Encodes STRUCTURE ONLY: names the five GIVEN rivers and their regions, but leaks no
    discharge figure, no length figure, no ratio value, and not which river wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for {e['name']} ({e['region']}) and read, from the "
                "infobox, TWO figures: (1) the river's mean (average) DISCHARGE in m³/s "
                "(cubic metres per second) — use the figure labelled 'discharge' or 'average' "
                "at or near the river's mouth, and record the exact integer; (2) the river's "
                "total LENGTH in km (kilometres). Report ONLY those two numbers (clearly "
                "labelled which is discharge and which is length) and the source URL. Do not "
                "guess from memory, and do not divide or compute anything."
            ),
            "expect": "Mean discharge (m³/s) and total length (km), both labelled — source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five rivers, two figures: its mean discharge (m³/s) "
            "and its total length (km). For EACH of the five rivers, write out the division "
            "explicitly on its own line in the form '<river>: <discharge> m³/s / <length> km "
            "= <ratio> m³/(s·km)' — compute every one of the five divisions BEFORE drawing any "
            "conclusion. THEN, comparing those five computed ratios, state which SINGLE river "
            "has the HIGHEST discharge-to-length ratio — that river name is the keystone answer. "
            "This need NOT be the river with the largest mean discharge, nor the one with the "
            "greatest total length. "
            "Report (a) that river and its ratio value (m³/(s·km)), (b) all five rivers' mean "
            "discharge (m³/s) and total length (km), and (c) cite each river's source URL."
        ),
    }
