"""
Test 079: Tier 5 (graph) — argmax over a COMPUTED RATIO (annual generation / installed capacity),
          different domain from test 064 (lakes) and test 059 (football).
Level: graph   Weight: long   Difficulty: 9/10

The agent must determine which of FIVE large hydroelectric generating stations has the highest
ANNUAL-GENERATION-TO-INSTALLED-CAPACITY ratio, where that ratio is NOT printed anywhere as a
single figure and must be COMPUTED by DIVIDING two looked-up infobox numbers:

    ratio = annual energy generation (GWh)  /  installed capacity (MW)
            [ = equivalent full-load hours per year when multiplied by 1,000 ]

The five stations span four continents and include one obvious GIANT (Guri Dam, Venezuela),
which is by far the largest by BOTH generation and capacity — the double-decoy. The ratio winner
is Churchill Falls Generating Station (Labrador, Canada), which has a capacity of only 5,428 MW
(small vs. Guri's 10,235 MW) but an exceptionally consistent river flow, giving it the highest
capacity factor (~73.6%) of the group. A "pick the biggest/most powerful" shortcut lands on Guri,
the wrong answer; the correct winner is an obscure sub-Arctic station nobody would intuit.

    Churchill Falls Generating Station (Labrador, Canada)
    W.A.C. Bennett Dam               (British Columbia, Canada)
    Bratsk Hydroelectric Power Station (Siberia, Russia)
    Guri Dam                          (Bolívar State, Venezuela)
    Kariba Dam                        (Zambia/Zimbabwe border)

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): shortcuts to Guri (largest by both raw quantities) or guesses from
    memory; cannot correctly divide all ten figures and compare five quotients without reading.
  * frontier sequential (ReAct): looks up all ten figures, converts TWh→GWh where needed,
    divides and compares -> decent.
  * graph_compiled: five parallel leaves each fetch ONE station's two figures from ONE page;
    aggregation owns every division AND the argmax, forced to WRITE OUT each division before
    concluding -> cheap executor is rescued; a diverse-grounding reranker can re-derive.

The trap is engineered three ways:
  (1) the ratio is computed — the agent must DIVIDE generation by capacity for each station
      and then compare the five quotients; no page states the cross-station ranking;
  (2) the ratio winner DIFFERS from BOTH raw rankings — the largest-generation AND largest-
      capacity station is the SAME giant (Guri Dam), which is NOT the winner, so "pick the
      biggest" fails on both axes;
  (3) the winner (Churchill Falls) has the SECOND-LOWEST installed capacity (5,428 MW) of
      the five — it is the least obvious power-producing candidate.

Ground truth (verified against live English Wikipedia 2026-06-27 — each station's infobox
'Annual generation' and 'Installed capacity' rows; note that Bennett and Bratsk pages express
generation in TWh which must be converted to GWh by ×1000 before dividing):

  station                            region                  gen(GWh)  cap(MW)  ratio(GWh/MW)
  Churchill Falls Generating Station Labrador, Canada          35,000    5,428      6.448  <- ARGMAX
  W.A.C. Bennett Dam                 British Columbia, Canada  15,000    2,907      5.160  <- 2nd (15 TWh)
  Bratsk Hydroelectric Power Station Siberia, Russia           22,600    4,515      5.006  <- (22.6 TWh)
  Guri Dam                           Bolívar State, Venezuela  47,000   10,235      4.592  <- biggest GEN+CAP
  Kariba Dam                         Zambia/Zimbabwe border     6,400    2,010      3.184

    https://en.wikipedia.org/wiki/Churchill_Falls_Generating_Station
    https://en.wikipedia.org/wiki/W.A.C._Bennett_Dam
    https://en.wikipedia.org/wiki/Bratsk_Dam          (redirects to Bratsk Hydroelectric Power Station)
    https://en.wikipedia.org/wiki/Guri_Dam
    https://en.wikipedia.org/wiki/Kariba_Dam

  RATIO ARGMAX = Churchill Falls Generating Station (6.448 GWh/MW = ~6,448 equivalent full-load
  hours per year; Wikipedia corroborates this as a ~73.6% capacity factor, though that figure
  is NOT required — the agent must compute the ratio itself from the two page-only figures).
  Runner-up = W.A.C. Bennett Dam (5.160 GWh/MW), so the margin is +1.288 GWh/MW absolute
  (~+25.0% relative). A worst-case ±1,000 GWh misread of Churchill's generation (reading 34,000
  instead of 35,000) gives ratio 6.264 → margin vs Bennett still ~21%; a ±100 MW misread of
  Bennett's capacity (reading 3,007 instead of 2,907) gives Bennett ratio 4.988 → margin ~29%.
  The keystone cannot be flipped by any single plausible figure misread.

  DOUBLE-DECOY / WIDE-MARGIN checks (confirmed live, asserted at import time):
    (a) ratio-winner (Churchill Falls) != biggest-GENERATION station (Guri, 47,000 GWh)  -> True
    (b) ratio-winner (Churchill Falls) != biggest-CAPACITY   station (Guri, 10,235 MW)   -> True
    (c) winner ratio (6.448) exceeds 2nd place (Bennett, 5.160) by ~25.0% (>> 15%)       -> True
    by generation:  Guri 47,000 > Churchill 35,000 > Bratsk 22,600 > Bennett 15,000 > Kariba 6,400
                                                                                          -> Guri
    by capacity:    Guri 10,235 > Bratsk 4,515 > Churchill 5,428 > Bennett 2,907 > Kariba 2,010
                                                                                          -> Guri
    by ratio:  Churchill 6.448 > Bennett 5.160 > Bratsk 5.006 > Guri 4.592 > Kariba 3.184
                                                                                -> Churchill Falls
    => the single most salient station (Guri, largest by BOTH raw quantities) is the doubly-
       baited decoy; Churchill Falls, the winner, is the least-powerful station except for Kariba.

  ANTI-PARAMETRIC: Guri Dam (Venezuela's dominant power source) is the obvious guess — but its
  high generation is offset by its even larger capacity, giving it only a 52% capacity factor vs.
  Churchill's 74%. Neither capacity-factor figure nor the division result is recallable: the agent
  must READ both figures and DIVIDE. The winner (Churchill Falls) is known mainly for a 1969
  electricity-sales contract dispute with Hydro-Québec — its superior capacity factor is not a
  commonly cited fact even within Canadian energy circles.

  KEYSTONE = Churchill Falls Generating Station (the argmax). Secondary (gated) = its ratio
  value ~6.448 GWh/MW (accepted ±3%) or equivalently ~6,448 equivalent full-load hours/year.
  All ten figures are distinct across stations and units, so the un-gated coverage diagnostic
  is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ---- verified fixtures (single source of truth for statement, validators and plan) ----
# ``ratio`` is shown for provenance only; nothing here is leaked into the task statement or plan.
ENTITIES: List[Dict[str, Any]] = [
    {
        "key": "churchill_falls",
        "name": "Churchill Falls Generating Station",
        "region": "Labrador, Canada",
        "generation_gwh": 35_000,   # Wikipedia infobox: "35,000 GWh (130,000 TJ)"
        "capacity_mw": 5_428,       # Wikipedia infobox: "5,428 MW"
        "ratio": 35_000 / 5_428,    # 6.448 GWh/MW  <- WINNER (highest capacity factor)
        "winner": True,
        "name_rx": r"churchill\s+falls",
        "gen_rx":  r"(?<!\d)35[,\s]?000(?!\d)",                        # 35,000 GWh
        "cap_rx":  r"(?<!\d)5[,\s]?428(?!\d)",                         # 5,428 MW
        "slug_rx": r"wiki/churchill_falls",
    },
    {
        "key": "wac_bennett",
        "name": "W.A.C. Bennett Dam",
        "region": "British Columbia, Canada",
        "generation_gwh": 15_000,   # Wikipedia infobox: "15 TWh (54 PJ)" = 15,000 GWh
        "capacity_mw": 2_907,       # Wikipedia infobox: "2,907 MW"
        "ratio": 15_000 / 2_907,    # 5.160 GWh/MW  <- 2nd place (runner-up)
        "winner": False,
        "name_rx": r"w\.?a\.?c\.?\s*bennett|gordon\s+m\.?\s*shrum",
        "gen_rx":  r"(?<!\d)15[,\s]?000(?!\d)|(?<!\d)15\s*(?:T[Ww][Hh]?|terawatt)",
        "cap_rx":  r"(?<!\d)2[,\s]?907(?!\d)",
        "slug_rx": r"wiki/w\.a\.c\._bennett|wiki/gordon.*shrum",
    },
    {
        "key": "bratsk",
        "name": "Bratsk Hydroelectric Power Station",
        "region": "Siberia, Russia",
        "generation_gwh": 22_600,   # Wikipedia infobox: "22.6 TWh" = 22,600 GWh
        "capacity_mw": 4_515,       # Wikipedia infobox: "4,515 MW"
        "ratio": 22_600 / 4_515,    # 5.006 GWh/MW
        "winner": False,
        "name_rx": r"\bbratsk\b",
        "gen_rx":  r"(?<!\d)22[,\s]?600(?!\d)|(?<!\d)22\.6\s*(?:T[Ww][Hh]?|terawatt)",
        "cap_rx":  r"(?<!\d)4[,\s]?515(?!\d)",
        "slug_rx": r"wiki/bratsk",
    },
    {
        "key": "guri",
        "name": "Guri Dam",
        "region": "Bolívar State, Venezuela",
        "generation_gwh": 47_000,   # Wikipedia infobox: "47,000 GWh"
        "capacity_mw": 10_235,      # Wikipedia infobox: "10,235 MW"
        "ratio": 47_000 / 10_235,   # 4.592 GWh/MW  <- biggest by BOTH generation and capacity (decoy)
        "winner": False,
        "name_rx": r"\bguri\b",
        "gen_rx":  r"(?<!\d)47[,\s]?000(?!\d)",
        "cap_rx":  r"(?<!\d)10[,\s]?235(?!\d)",
        "slug_rx": r"wiki/guri",
    },
    {
        "key": "kariba",
        "name": "Kariba Dam",
        "region": "Zambia/Zimbabwe border",
        "generation_gwh": 6_400,    # Wikipedia infobox: "6,400 GWh (23,000 TJ)"
        "capacity_mw": 2_010,       # Wikipedia infobox: "2,010 MW" (combined north+south)
        "ratio": 6_400 / 2_010,     # 3.184 GWh/MW
        "winner": False,
        "name_rx": r"\bkariba\b",
        "gen_rx":  r"(?<!\d)6[,\s]?400(?!\d)",
        "cap_rx":  r"(?<!\d)2[,\s]?010(?!\d)",
        "slug_rx": r"wiki/kariba",
    },
]

WINNER = next(e for e in ENTITIES if e["winner"])         # Churchill Falls GS — ratio argmax
WINNER_RATIO_R = WINNER["generation_gwh"] / WINNER["capacity_mw"]   # GWh/MW ≈ 6.448
WINNER_RATIO_H = WINNER_RATIO_R * 1_000                  # equivalent full-load hours ≈ 6,448
RATIO_TOL = 0.03                                          # ±3% on the secondary value

# ---- DOUBLE-DECOY / WIDE-MARGIN invariants: asserted at import so the fixtures can never
# ---- silently drift out of the design (winner ≠ either raw argmax; margin > 15%). ----------
_BY_GEN   = max(ENTITIES, key=lambda e: e["generation_gwh"])
_BY_CAP   = max(ENTITIES, key=lambda e: e["capacity_mw"])
_BY_RATIO = sorted(ENTITIES, key=lambda e: e["generation_gwh"] / e["capacity_mw"], reverse=True)
assert _BY_RATIO[0] is WINNER,   "ratio argmax must be the declared winner"
assert _BY_GEN   is not WINNER,  "winner must NOT be the highest-generation station (decoy A)"
assert _BY_CAP   is not WINNER,  "winner must NOT be the highest-capacity station (decoy B)"
_W_RATIO          = WINNER["generation_gwh"] / WINNER["capacity_mw"]
_RUNNER_UP_RATIO  = _BY_RATIO[1]["generation_gwh"] / _BY_RATIO[1]["capacity_mw"]
assert (_W_RATIO - _RUNNER_UP_RATIO) / _RUNNER_UP_RATIO > 0.15, \
    "keystone margin must exceed 15% (currently ~25%)"

# ---- winner / others regexes used by the keystone gate ---------------------------------
_WINNER_RX = WINNER["name_rx"]   # r"churchill\s+falls"
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# Comparative / superlative triggers asserting a winner.
_SUP = (r"more|larger|greater|higher|bigger|largest|greatest|highest|biggest|most|maximum"
        r"|best|top|efficient|efficient(?:ly)?|superior|leading|peak")

# Keystone winner detection: Churchill Falls tied to a 'highest / most efficient ratio'
# assertion, in either direction, with the proximity window forbidden from crossing into ANY
# other station's name.  The pattern mirrors test_064's Issyk-Kul gate exactly:
#   dir 1  (subject -> superlative): "Churchill Falls … has the highest ratio"
#   dir 2  (superlative -> subject): "the highest ratio is Churchill Falls" (NOT "higher than …")
_CHURCHILL_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Ratio-value detection — two representations:
#   Form A: small decimal GWh/MW  e.g. "6.448", "6.45"  (range ~6.25 – 6.64)
#   Form B: full-load hours       e.g. "6448", "6,448"  (range ~6254 – 6642)
#   Form C: explicit fraction     e.g. "35,000 / 5,428"
_RATIO_NUM_SMALL = re.compile(r"(?<!\d)(6\.\d{2,})(?!\d)")          # matches 6.xx(x)
_RATIO_NUM_LARGE = re.compile(r"(?<!\d)(6[,\s]?\d{3})(?!\d)")       # matches 6,xxx or 6xxx
_RATIO_FRAC      = re.compile(r"\b35[,\s]?000\s*/\s*5[,\s]?428\b")


# =====================================================================
#  CONTRACT FUNCTIONS
# =====================================================================

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "079",
        "test_name": (
            "Tier 5: Computed-ratio argmax, wide-margin "
            "(highest annual-generation-to-installed-capacity, hydroelectric stations)"
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
        "guess from memory). For EACH of the following five hydroelectric generating stations, "
        "open the station's Wikipedia page and read TWO numbers from the infobox: (A) the "
        "station's annual energy generation (in GWh or TWh — note the unit carefully) and "
        "(B) its installed capacity (in MW):\n"
        f"{listing}\n\n"
        "IMPORTANT: some Wikipedia pages report annual generation in TWh (terawatt-hours) "
        "rather than GWh (gigawatt-hours). Convert any TWh figure to GWh by multiplying by "
        "1,000 before doing any arithmetic (1 TWh = 1,000 GWh).\n\n"
        "Then COMPUTE, for each station, its ANNUAL-GENERATION-TO-INSTALLED-CAPACITY ratio "
        "= annual generation (GWh) / installed capacity (MW). This quotient, in GWh/MW, "
        "equals the station's equivalent full-load hours per year when multiplied by 1,000 — "
        "no Wikipedia page prints this cross-station comparison, you must divide the two "
        "figures yourself. COMPARE the five ratios and determine which station has the "
        "HIGHEST ratio. Note: the answer is the largest RATIO, which is NOT necessarily the "
        "station with the largest annual generation, nor the one with the largest installed "
        "capacity.\n\n"
        "Report (a) which station has the highest annual-generation-to-capacity ratio (a "
        "single station name — the keystone), (b) that station's computed ratio value "
        "(in GWh/MW, or equivalently in equivalent full-load hours/year = ratio × 1,000), "
        "(c) all five stations' annual generation (GWh) and installed capacity (MW) (the ten "
        "figures you looked up), and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which station has the highest annual-generation-to-installed-capacity ratio "
        "(the primary answer / keystone)",
        "That winning station's computed ratio value (GWh/MW or equivalent full-load hours/year)",
        "All five stations' annual generation (GWh) and installed capacity (MW) — "
        "the ten looked-up figures",
        "Source URL for each station's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per station, a five-way fan-out)",
        "Correctly names Churchill Falls Generating Station as the highest "
        "generation-to-capacity ratio (NOT Guri Dam, which is the largest by BOTH "
        "annual generation and installed capacity)",
        "Reports the winner's ratio near 6.448 GWh/MW or ~6,448 full-load hours/year "
        "(within ±3%)",
        "Gathers all ten figures (generation and capacity for each of the five stations)",
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
    """KEYSTONE gate: deliverables[0] names Churchill Falls as the highest-ratio station.

    Word-bounded, and NOT satisfied by merely listing Churchill Falls among the five: when
    another station is also named, Churchill Falls must be the one tied to a 'highest / most
    efficient ratio' assertion (tempered so a rival named as the winner — or 'higher than
    Churchill Falls' — never counts). A terse primary answer naming only the winner (Churchill
    Falls, with no rival in the slot) also passes.
    """
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True   # names only the winner, no rival present
    return bool(_CHURCHILL_WINS.search(text))


def _ratio_value_present(text: str) -> bool:
    """True when a number in the winner's ratio band appears.

    Accepts:
      - ~6.448 GWh/MW (Form A: small decimal, ±3%)
      - ~6,448 full-load hours/year (Form B: 4-digit integer, ±3%)
      - the explicit fraction 35,000 / 5,428 (Form C)
    """
    lo_r = WINNER_RATIO_R * (1.0 - RATIO_TOL)   # ~6.255 GWh/MW
    hi_r = WINNER_RATIO_R * (1.0 + RATIO_TOL)   # ~6.642 GWh/MW
    lo_h, hi_h = lo_r * 1_000, hi_r * 1_000     # ~6,255 – 6,642 hours

    # Form A: small decimal  e.g. "6.45", "6.448"
    for raw in _RATIO_NUM_SMALL.findall(text):
        v = float(raw)
        if lo_r <= v <= hi_r:
            return True

    # Form B: full-load hours  e.g. "6448", "6,448", "6 448"
    for raw in _RATIO_NUM_LARGE.findall(text):
        v = float(raw.replace(",", "").replace(" ", ""))
        if lo_h <= v <= hi_h:
            return True

    # Form C: explicit fraction
    return bool(_RATIO_FRAC.search(text))


# =====================================================================
#  VALIDATION FUNCTIONS
# =====================================================================

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a five-way fan-out wants one page per station."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 4,
        "score": min(1.0, n / 5.0),
        "reason": f"{n} visit(s) (target ≥5: one page per station; ≥4 to pass)",
    }


def validate_keystone_argmax(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the highest generation-to-capacity ratio is Churchill Falls GS —
    NOT the largest station by generation nor by capacity (both Guri Dam). A model that
    shortcuts to a raw quantity or guesses the most salient name (Guri / Bratsk) mis-picks.
    """
    passed = _keystone_ok(result)
    return {
        "check": "keystone_argmax",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Churchill Falls Generating Station named as the highest generation-to-capacity ratio"
            if passed else
            "Highest-ratio station (Churchill Falls GS) missing/incorrect — beware: Guri Dam "
            "is the largest by BOTH annual generation (47,000 GWh) AND installed capacity "
            "(10,235 MW), but its ratio (4.59 GWh/MW) is well below Churchill Falls (6.45)"
        ),
    }


def validate_coverage(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the five stations' figure-pairs were gathered.

    A station's inputs are credited only when BOTH its annual generation AND its installed
    capacity appear (alongside its name). All ten figures are distinct (no cross-crediting).
    Deliberately NOT gated on the keystone — it measures structured fan-out, not correctness.
    Note: Bennett/Bratsk generation may appear in TWh (15 / 22.6) or GWh (15,000 / 22,600);
    both forms are accepted by gen_rx.
    """
    text = _all_text(result)
    hits = [
        e["name"]
        for e in ENTITIES
        if (re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["gen_rx"], text, re.IGNORECASE)
            and re.search(e["cap_rx"], text))
    ]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} stations' generation+capacity gathered "
            f"({', '.join(hits) or 'none'})"
        ),
    }


def validate_winner_ratio(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: the winner's computed ratio (~6.448 GWh/MW or ~6,448 full-load hours,
    within ±3%). Short-circuits to 0 when the keystone is absent, so a guessed winner cannot
    bank the value credit.
    """
    if not _keystone_ok(result):
        return {
            "check": "winner_ratio",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → ratio value not credited",
        }
    ok = _ratio_value_present(_all_text(result))
    lo_r = WINNER_RATIO_R * (1.0 - RATIO_TOL)
    hi_r = WINNER_RATIO_R * (1.0 + RATIO_TOL)
    return {
        "check": "winner_ratio",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"winner's ratio in {lo_r:.3f}–{hi_r:.3f} GWh/MW "
            f"(or {lo_r*1000:.0f}–{hi_r*1000:.0f} full-load hours/year) present"
            if ok else
            f"no ratio in {lo_r:.3f}–{hi_r:.3f} GWh/MW "
            f"(expected ~{WINNER_RATIO_R:.3f} GWh/MW or ~{WINNER_RATIO_H:.0f} hours/year)"
        ),
    }


def validate_citation(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: cites the station pages. Short-circuits to 0 when the keystone is
    absent."""
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
        "reason": f"{cited}/{n} station pages cited",
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
    # None → the harness applies its default structured rubric judge, as in 064.
    return None


# =====================================================================
#  COMPILED PLAN  (offline-authored leak-free scaffold)
# =====================================================================

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out / aggregate scaffold for the ``graph_compiled`` variant.

    FIVE INDEPENDENT parallel leaves: for each of the five GIVEN stations, one leaf opens
    that station's Wikipedia page and reads its two infobox figures — annual energy generation
    (GWh or TWh, noting the unit) and installed capacity (MW) — both on the SAME page, so one
    page-read per station. ALL arithmetic — every GWh conversion, every generation/capacity
    division, and the final argmax — lives only in the aggregation step, which is forced to
    WRITE OUT each per-station division explicitly before concluding, rescuing the cheap
    executor and allowing a diverse-grounding reranker to re-derive and catch any slip.

    Encodes STRUCTURE ONLY: names the five GIVEN stations and their regions, but leaks no
    generation figure, no capacity figure, no ratio value, and not which station wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for {e['name']} ({e['region']}) and read, from the "
                "infobox, TWO figures: (1) the station's annual energy GENERATION — note "
                "carefully whether the unit is GWh (gigawatt-hours) or TWh (terawatt-hours) "
                "and record both the number AND the unit exactly as shown; (2) the station's "
                "INSTALLED CAPACITY in MW (megawatts). Report ONLY those two figures (clearly "
                "labelled which is generation with its unit, and which is capacity) and the "
                "source URL. Do not convert units, do not divide, do not compute anything — "
                "just read and report."
            ),
            "expect": (
                "Annual generation (number + unit: GWh or TWh) and installed capacity (MW), "
                "both clearly labelled — source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five stations, two figures: its annual energy "
            "generation (in GWh or TWh as reported) and its installed capacity (in MW). "
            "FIRST, convert any TWh figure to GWh by multiplying by 1,000 "
            "(1 TWh = 1,000 GWh). "
            "THEN, for EACH of the five stations, write out the division explicitly on its own "
            "line in the form '<station>: <generation_GWh> GWh / <capacity_MW> MW = <ratio> "
            "GWh/MW' — compute every one of the five divisions BEFORE drawing any conclusion. "
            "THEN, comparing those five computed ratios, state which SINGLE station has the "
            "HIGHEST annual-generation-to-installed-capacity ratio — that station name is the "
            "keystone answer. This need NOT be the station with the largest annual generation, "
            "nor the one with the largest installed capacity. "
            "Report (a) that station and its ratio value (GWh/MW, or equivalently ×1,000 "
            "for equivalent full-load hours/year), (b) all five stations' annual generation "
            "(GWh after conversion) and installed capacity (MW), and (c) cite each station's "
            "source URL."
        ),
    }
