"""
Test 221: Tier 5 (graph) — argmax over a COMPUTED derived quantity: skyscraper average
floor-to-floor height (height / floor count).
Level: graph   Weight: long   Difficulty: 9/10

Sibling of 218/219/220 in the derivation-layer instrument cluster. Among FIVE supertall buildings
the agent must determine which has the HIGHEST average floor height, defined as:

    avg floor height = architectural height (m) / number of floors      [unit: m per floor]

No page states this cross-building comparison; each building's Wikipedia infobox gives only its
own architectural Height and Floor count, and the agent must divide.

Ground truth (verified against live English Wikipedia infoboxes, 2026-08-31 — "architectural
height" = height to the architectural top/tip as officially recognised, i.e. the same figure used
for world's-tallest-building rankings; floor counts are the above-ground figure the infobox
states as the building's primary floor count):

  building                       height(m)   floors   avg floor height (m/floor)
  One World Trade Center (US)      546.2        94         5.8106   <- ARGMAX (keystone)
  Burj Khalifa (UAE)                 828        154         5.3766   <- 2nd; TALLEST of the five
  Taipei 101 (Taiwan)               509.2        101        5.0416
  Shanghai Tower (China)             632        128         4.9375
  Willis Tower (US)                  442         110        4.0182
    https://en.wikipedia.org/wiki/One_World_Trade_Center
    https://en.wikipedia.org/wiki/Burj_Khalifa
    https://en.wikipedia.org/wiki/Taipei_101
    https://en.wikipedia.org/wiki/Shanghai_Tower
    https://en.wikipedia.org/wiki/Willis_Tower

  Notes on the figures: One World Trade Center's 546.2 m ("1,792 ft") is its height to the tip,
  including the spire, as CTBUH/Wikipedia record it; its floor count of 94 is the building's
  stated occupiable floor count (there are 5 further below-ground levels, not counted). Burj
  Khalifa's floor count is stated as "154 + 9 maintenance"; 154 is used as the primary figure.

  AVG-FLOOR-HEIGHT ARGMAX = One World Trade Center (5.8106 m/floor). Runner-up = Burj Khalifa
  (5.3766 m/floor), margin = +8.1% relative ((5.8106-5.3766)/5.3766). This margin is narrower than
  the sibling tasks' by design of the domain (both figures are EXACT integer/one-decimal official
  counts with no plausible rounding ambiguity), so the acceptance tolerance is tightened to +/-1%
  (a >8x separation): a misread of the floor count by 1 (94->95, or 154->155) still leaves One WTC
  ahead of Burj Khalifa by a comfortable margin (5.75 vs 5.34 = +7.7%), and the next-tightest
  adjacent gap in the five-way ranking (Taipei 101 vs Shanghai Tower, 5.0416 vs 4.9375 = 2.1%) is
  still more than double the tolerance band, so no single misread can cross-credit a rival.

  DOUBLE-DECOY (confirmed live, asserted at import time below):
    (a) ratio-winner (One WTC, 546.2 m) != TALLEST of the five (Burj Khalifa, 828 m)     -> True
    (b) ratio-winner (One WTC, 94 floors) != FEWEST FLOORS of the five (94 IS in fact the
        fewest -- expected, since minimising the denominator raises the ratio -- but One WTC is
        NOT the tallest, which is the salient "obvious guess" this task actually traps)
    by height: Burj Khalifa 828 > Shanghai 632 > One WTC 546.2 > Taipei 101 509.2 > Willis 442
    by floors: Burj Khalifa 154 > Shanghai 128 > Willis 110 > Taipei 101 101 > One WTC 94
    by ratio:  One WTC 5.81 > Burj Khalifa 5.38 > Taipei 101 5.04 > Shanghai 4.94 > Willis 4.02
  => "pick the world's tallest building" (Burj Khalifa, the single most salient guess in this
     domain) is wrong; the actual winner is only the THIRD-tallest of the five.

  ANTI-PARAMETRIC: individual building heights/floor counts are well-known trivia; the
  cross-building average-floor-height ranking is not a recallable statistic.

  KEYSTONE = the argmax BUILDING (One World Trade Center). Secondary (gated) value = its ratio
  (5.81 m/floor), accepted within +/- 1%. All ten raw figures are distinct and none collides with
  the derived value band (checked at import time).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


ENTITIES: List[Dict[str, Any]] = [
    {"key": "one_wtc", "name": "One World Trade Center", "height_m": 546.2, "floors": 94,
     "ratio": 546.2 / 94, "winner": True,
     "name_rx": r"one\s+world\s+trade\s+center|1\s+wtc\b",
     "h_rx": r"(?<!\d)546\.2(?!\d)", "f_rx": r"(?<!\d)94(?!\d)",
     "slug_rx": r"wiki/one_world_trade_center"},
    {"key": "burj_khalifa", "name": "Burj Khalifa", "height_m": 828, "floors": 154,
     "ratio": 828 / 154, "winner": False,
     "name_rx": r"burj\s+khalifa",
     "h_rx": r"(?<!\d)828(?!\d)", "f_rx": r"(?<!\d)154(?!\d)",
     "slug_rx": r"wiki/burj_khalifa"},
    {"key": "taipei_101", "name": "Taipei 101", "height_m": 509.2, "floors": 101,
     "ratio": 509.2 / 101, "winner": False,
     "name_rx": r"taipei\s*101",
     "h_rx": r"(?<!\d)509\.2(?!\d)",
     # "101" collides with the building's own name ("Taipei 101"), so the floor-count match
     # additionally requires a nearby "floor(s)" cue -- bare "Taipei 101" alone must not count.
     "f_rx": r"(?<!\d)101(?!\d)[^.\n]{0,20}?floors?|floors?[^.\n]{0,20}?(?<!\d)101(?!\d)",
     "slug_rx": r"wiki/taipei_101"},
    {"key": "shanghai_tower", "name": "Shanghai Tower", "height_m": 632, "floors": 128,
     "ratio": 632 / 128, "winner": False,
     "name_rx": r"shanghai\s+tower",
     "h_rx": r"(?<!\d)632(?!\d)", "f_rx": r"(?<!\d)128(?!\d)",
     "slug_rx": r"wiki/shanghai_tower"},
    {"key": "willis_tower", "name": "Willis Tower", "height_m": 442, "floors": 110,
     "ratio": 442 / 110, "winner": False,
     "name_rx": r"willis\s+tower",
     "h_rx": r"(?<!\d)442(?!\d)", "f_rx": r"(?<!\d)110(?!\d)",
     "slug_rx": r"wiki/willis_tower"},
]

WINNER = next(e for e in ENTITIES if e["winner"])          # One World Trade Center
WINNER_RATIO = WINNER["ratio"]                               # ~5.8106
RATIO_TOL = 0.01                                              # +/- 1% (tight domain, exact figures)

_BY_HEIGHT = max(ENTITIES, key=lambda e: e["height_m"])
_BY_RATIO = sorted(ENTITIES, key=lambda e: e["ratio"], reverse=True)
assert _BY_RATIO[0] is WINNER, "avg-floor-height argmax must be the declared winner"
assert _BY_HEIGHT is not WINNER, "winner must NOT be the tallest building (decoy: Burj Khalifa)"
_RUNNER_UP = _BY_RATIO[1]
assert _RUNNER_UP is _BY_HEIGHT, "the tallest building must be the closest rival (the salient decoy)"
_MARGIN = (WINNER["ratio"] - _RUNNER_UP["ratio"]) / _RUNNER_UP["ratio"]
assert 0.06 <= _MARGIN <= 0.15, f"keystone margin {_MARGIN:.4f} outside the designed band"
# every adjacent gap in the full ranking must clear a comfortable multiple of the tolerance
_SORTED = [e["ratio"] for e in _BY_RATIO]
for _i in range(len(_SORTED) - 1):
    _gap = (_SORTED[_i] - _SORTED[_i + 1]) / _SORTED[_i + 1]
    assert _gap > 2 * RATIO_TOL, f"adjacent ratio gap {_gap:.4f} too thin for tolerance {RATIO_TOL}"
# no printed raw number collides with any derived ratio band
_PRINTED = {e["height_m"] for e in ENTITIES} | {e["floors"] for e in ENTITIES}
for _e in ENTITIES:
    for _p in _PRINTED:
        assert abs(_p - _e["ratio"]) > _e["ratio"] * RATIO_TOL, \
            f"printed {_p} collides with the {_e['name']} ratio band"

_WINNER_RX = "(?:" + WINNER["name_rx"] + ")"
_OTHERS = "(?:" + "|".join(e["name_rx"] for e in ENTITIES if not e["winner"]) + ")"
_SUP = r"more|larger|greater|higher|taller\s+floors|highest|most|maximum|best|top"
_ONEWTC_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,90}" + _WINNER_RX,
    re.IGNORECASE,
)
_URL_RX = re.compile(r"https?://\S+")
_URL_TRAIL_PUNCT = ").,;:!?]}\"'"


def _strip_urls(text: str) -> str:
    def _repl(m: "re.Match") -> str:
        url = m.group(0)
        trail = ""
        while url and url[-1] in _URL_TRAIL_PUNCT:
            trail = url[-1] + trail
            url = url[:-1]
        return " " + trail
    return _URL_RX.sub(_repl, text)


_RATIO_NUM = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d+)?)(?!\d)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "221",
        "test_name": "Tier 5: Computed avg-floor-height argmax (tallest average storey)",
        "difficulty_level": "9/10",
        "category": "Quantitative reasoning + computed-ratio argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following five supertall buildings, open its Wikipedia "
        "page and read TWO numbers from the infobox: the building's architectural HEIGHT (in "
        "metres) and its FLOOR COUNT (number of floors):\n"
        f"{listing}\n\n"
        "Then COMPUTE, for each building, its AVERAGE FLOOR HEIGHT = height / floor count (no "
        "page prints this cross-building comparison, you must divide yourself). COMPARE the five "
        "values and determine which building has the HIGHEST average floor height. Note: this is "
        "NOT necessarily the world's tallest building in this set.\n\n"
        "Report (a) which building has the highest average floor height (the keystone), (b) that "
        "building's computed value (in metres per floor), (c) all five buildings' height and "
        "floor count (the ten figures you looked up), and (d) the exact source URL of every page "
        "you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which building has the highest average floor height (the primary answer / keystone)",
        "That winning building's computed average floor height (m per floor)",
        "All five buildings' height (m) and floor count",
        "Source URL for each building's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per building, a five-way fan-out)",
        f"Correctly names {WINNER['name']} as the highest average floor height (NOT Burj Khalifa, "
        "the tallest of the five)",
        f"Reports the winner's value near {WINNER_RATIO:.2f} m/floor (within +/- 1%)",
        "Gathers all ten figures (height and floor count for each of the five buildings)",
        "Cites the source pages",
    ]


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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _strip_urls(_primary_text(result))
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True
    return bool(_ONEWTC_WINS.search(text))


def _ratio_value_present(text: str) -> bool:
    lo, hi = WINNER_RATIO * (1.0 - RATIO_TOL), WINNER_RATIO * (1.0 + RATIO_TOL)
    for raw in _RATIO_NUM.findall(text):
        v = float(raw)
        if lo <= v <= hi:
            return True
    return False


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per building; >=4 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"{WINNER['name']} named as the highest average floor height" if passed
                      else f"Highest-avg-floor-height building ({WINNER['name']}) missing/incorrect "
                           "(beware: Burj Khalifa is the tallest of the five)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["h_rx"], text) and re.search(e["f_rx"], text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} buildings' height+floors gathered ({', '.join(hits) or 'none'})"}


def validate_winner_ratio(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "winner_ratio", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> ratio value not credited"}
    ok = _ratio_value_present(_all_text(result))
    return {"check": "winner_ratio", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"winner's ratio within +/-1% of {WINNER_RATIO:.2f} m/floor present" if ok
                       else f"no ratio near {WINNER_RATIO:.2f} m/floor found")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} building pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_argmax, validate_coverage,
            validate_winner_ratio, validate_citation]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for {e['name']} and read, from the infobox, TWO "
                "figures: the building's architectural HEIGHT (in metres) and its FLOOR COUNT "
                "(number of floors). Report ONLY those two numbers, clearly labelled, and the "
                "source URL. Do not guess from memory, and do not divide or compute anything."
            ),
            "expect": "HEIGHT (m) and FLOOR COUNT, both labelled -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five buildings, its height (m) and its floor count. "
            "For EACH building, write out the division explicitly on its own line in the form "
            "'<building>: <height> / <floors> = <avg floor height>' -- compute every one of the "
            "five divisions BEFORE drawing any conclusion. THEN, comparing those five computed "
            "values, state which SINGLE building has the HIGHEST average floor height -- that "
            "building's name is the keystone answer. This need NOT be the tallest building of the "
            "five. Report (a) that building and its average floor height, (b) all five buildings' "
            "height and floor count, and (c) cite each building's source URL."
        ),
    }
