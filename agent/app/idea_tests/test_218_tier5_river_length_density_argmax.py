"""
Test 218: Tier 5 (graph) — argmax over a COMPUTED derived quantity: river "channel-length
density" (length / drainage-basin area).
Level: graph   Weight: long   Difficulty: 9/10

Part of the derivation-layer instrument cluster (see sibling 219/220/221): the evidence graph
under test recomputes every derived number in Python from literal page-sourced spans rather than
trusting model prose. This task is the web-grounded argmax analogue of test_204's self-contained
unit-rate argmin — the winner is decided by a quantity that must be COMPUTED per candidate, never
read off a page.

Among FIVE major rivers the agent must determine which has the HIGHEST channel-length-density,
defined as:

    density = river length (in METRES) / drainage basin area (in km^2)      [unit: m per km^2]

No page prints this cross-river comparison; each river's Wikipedia infobox gives only its own
Length (km) and Basin size (km^2), and the agent must convert length to metres and divide.

Ground truth (verified against live English Wikipedia infoboxes, 2026-08-31 — the FIRST/primary
length and basin-size figures printed in each infobox; alternate/footnoted figures noted where
relevant):

  river          length(km)   basin(km^2)   density = length_m/basin (m/km^2)
  Mekong           4,350          795,000        5.4717   <- ARGMAX (keystone)
  Yangtze          6,300        1,808,500        3.4835   <- 2nd place
  Nile             7,088        2,927,843        2.4209   <- LONGEST river (decoy A)
  Mississippi      3,766        2,980,000        1.2638
  Amazon           6,575        6,925,674        0.9494   <- LARGEST basin (decoy B)
    https://en.wikipedia.org/wiki/Nile
    https://en.wikipedia.org/wiki/Amazon_River
    https://en.wikipedia.org/wiki/Yangtze
    https://en.wikipedia.org/wiki/Mississippi_River
    https://en.wikipedia.org/wiki/Mekong

  Sources note: Yangtze's infobox lists "6,300 km (3,900 mi)" as the primary length (a second,
  footnoted figure of 6,374 km including a further-upstream source is also present; 6,300 km is
  used here as the officially-cited figure). Mississippi's infobox lists the main-stem figures
  (3,766 km / 2,980,000 km^2), distinct from the "with Atchafalaya" alternate (3,220,000 km^2)
  which is not used.

  DENSITY ARGMAX = Mekong (5.4717 m/km^2). Runner-up = Yangtze (3.4835 m/km^2), margin =
  +57.1% relative ((5.4717-3.4835)/3.4835). No plausible single noisy extraction (a few tens of
  km off on length, or a few percent off on basin area) can close a 57% gap.

  DOUBLE-DECOY (confirmed live, asserted at import time below):
    (a) density-winner (Mekong) != longest river (Nile, 7,088 km)                    -> True
    (b) density-winner (Mekong) != largest-basin river (Amazon, 6,925,674 km^2)      -> True
    by length: Nile 7,088 > Amazon 6,575 > Yangtze 6,300 > Mekong 4,350 > Mississippi 3,766
    by basin:  Amazon 6.93M > Mississippi 2.98M > Nile 2.93M > Yangtze 1.81M > Mekong 0.795M
    by density: Mekong 5.47 > Yangtze 3.48 > Nile 2.42 > Mississippi 1.26 > Amazon 0.95
  => the single most salient "biggest river" guess (Nile, longest) and the "biggest basin" guess
     (Amazon) are BOTH wrong; the actual winner (Mekong) is the SHORTEST-basin AND 4th-of-5-longest
     river in the set — the least obvious candidate on either raw axis.

  ANTI-PARAMETRIC: river length and drainage-basin area are individually well-known trivia, but
  their CROSS-RIVER RATIO ranking is not a recallable fact — nobody has "Mekong has the highest
  length-to-basin-area ratio among major rivers" memorized. The cheap model must read the pages
  and divide.

  KEYSTONE = the argmax RIVER (Mekong). Secondary (gated) value = its density (~5.47 m/km^2),
  accepted within +/- 3%. All ten raw figures are distinct and none collides with the derived
  value band (checked at import time), so the un-gated coverage diagnostic is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


ENTITIES: List[Dict[str, Any]] = [
    {"key": "mekong", "name": "Mekong", "length_km": 4350, "basin_km2": 795000,
     "density": 4350 * 1000 / 795000, "winner": True,
     "name_rx": r"\bmekong\b",
     "len_rx": r"(?<!\d)4[,\s]?350(?!\d)", "basin_rx": r"(?<!\d)795[,\s]?000(?!\d)",
     "slug_rx": r"wiki/mekong"},
    {"key": "yangtze", "name": "Yangtze", "length_km": 6300, "basin_km2": 1808500,
     "density": 6300 * 1000 / 1808500, "winner": False,
     "name_rx": r"\byangtze\b",
     "len_rx": r"(?<!\d)6[,\s]?300(?!\d)", "basin_rx": r"(?<!\d)1[,\s]?808[,\s]?500(?!\d)",
     "slug_rx": r"wiki/yangtze"},
    {"key": "nile", "name": "Nile", "length_km": 7088, "basin_km2": 2927843,
     "density": 7088 * 1000 / 2927843, "winner": False,
     "name_rx": r"\bnile\b",
     "len_rx": r"(?<!\d)7[,\s]?088(?!\d)", "basin_rx": r"(?<!\d)2[,\s]?927[,\s]?843(?!\d)",
     "slug_rx": r"wiki/nile"},
    {"key": "mississippi", "name": "Mississippi", "length_km": 3766, "basin_km2": 2980000,
     "density": 3766 * 1000 / 2980000, "winner": False,
     "name_rx": r"\bmississippi\b",
     "len_rx": r"(?<!\d)3[,\s]?766(?!\d)", "basin_rx": r"(?<!\d)2[,\s]?980[,\s]?000(?!\d)",
     "slug_rx": r"wiki/mississippi_river"},
    {"key": "amazon", "name": "Amazon", "length_km": 6575, "basin_km2": 6925674,
     "density": 6575 * 1000 / 6925674, "winner": False,
     "name_rx": r"\bamazon\b",
     "len_rx": r"(?<!\d)6[,\s]?575(?!\d)", "basin_rx": r"(?<!\d)6[,\s]?925[,\s]?674(?!\d)",
     "slug_rx": r"wiki/amazon_river"},
]

WINNER = next(e for e in ENTITIES if e["winner"])          # Mekong
WINNER_DENSITY = WINNER["density"]                          # ~5.4717 m/km^2
DENSITY_TOL = 0.03                                           # +/- 3%

# ---- import-time invariants (double-decoy + margin + collision hygiene) ----
_BY_LEN = max(ENTITIES, key=lambda e: e["length_km"])
_BY_BASIN = max(ENTITIES, key=lambda e: e["basin_km2"])
_BY_DENSITY = sorted(ENTITIES, key=lambda e: e["density"], reverse=True)
assert _BY_DENSITY[0] is WINNER, "density argmax must be the declared winner"
assert _BY_LEN is not WINNER, "winner must NOT be the longest river (decoy A: Nile)"
assert _BY_BASIN is not WINNER, "winner must NOT be the largest-basin river (decoy B: Amazon)"
_RUNNER_UP = _BY_DENSITY[1]
_MARGIN = (WINNER["density"] - _RUNNER_UP["density"]) / _RUNNER_UP["density"]
assert _MARGIN > 0.15, f"keystone margin {_MARGIN:.3f} must exceed 15%"
_PRINTED_NUMS = {e["length_km"] for e in ENTITIES} | {e["basin_km2"] for e in ENTITIES}
for _e in ENTITIES:
    for _p in _PRINTED_NUMS:
        assert abs(_p - _e["density"]) > 1, "printed raw number collides with a density value"

_WINNER_RX = "(?:" + WINNER["name_rx"] + ")"
_OTHERS = "(?:" + "|".join(e["name_rx"] for e in ENTITIES if not e["winner"]) + ")"
_SUP = r"more|larger|greater|higher|bigger|largest|greatest|highest|densest|most|maximum|best|top"
_MEKONG_WINS = re.compile(
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


_DENSITY_NUM = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)(?!\d)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "218",
        "test_name": "Tier 5: Computed-density argmax (highest channel-length-to-basin-area ratio)",
        "difficulty_level": "9/10",
        "category": "Quantitative reasoning + computed-ratio argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following five rivers, open the river's Wikipedia page and "
        "read TWO numbers from the infobox: the river's LENGTH (in kilometres) and its DRAINAGE "
        "BASIN SIZE / basin area (in square kilometres, km^2):\n"
        f"{listing}\n\n"
        "Then COMPUTE, for each river, its CHANNEL-LENGTH DENSITY = length in METRES divided by "
        "basin area in km^2 (convert km to m first: multiply length by 1000). No page prints this "
        "cross-river comparison — you must divide yourself. COMPARE the five densities and "
        "determine which river has the HIGHEST channel-length density. Note: this is NOT "
        "necessarily the longest river, nor the one with the largest drainage basin.\n\n"
        "Report (a) which river has the highest channel-length density (the keystone), (b) that "
        "river's computed density value (in metres per km^2), (c) all five rivers' length and "
        "basin area (the ten figures you looked up), and (d) the exact source URL of every page "
        "you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which river has the highest channel-length density (the primary answer / keystone)",
        "That winning river's computed density value (metres per km^2)",
        "All five rivers' length (km) and basin area (km^2)",
        "Source URL for each river's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per river, a five-way fan-out)",
        f"Correctly names {WINNER['name']} as the highest channel-length density (NOT Nile, the "
        "longest river, and NOT Amazon, the largest-basin river)",
        f"Reports the winner's density near {WINNER_DENSITY:.2f} m/km^2 (within +/- 3%)",
        "Gathers all ten figures (length and basin area for each of the five rivers)",
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
    """KEYSTONE gate: grounded (visit.count > 0) AND the winner (Mekong) asserted as the
    highest-density river, not merely listed among the five."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _strip_urls(_primary_text(result))
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True
    return bool(_MEKONG_WINS.search(text))


def _density_value_present(text: str) -> bool:
    lo, hi = WINNER_DENSITY * (1.0 - DENSITY_TOL), WINNER_DENSITY * (1.0 + DENSITY_TOL)
    for raw in _DENSITY_NUM.findall(text):
        v = float(raw)
        if lo <= v <= hi:
            return True
    return False


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per river; >=4 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"{WINNER['name']} named as the highest channel-length density" if passed
                      else f"Highest-density river ({WINNER['name']}) missing/incorrect (beware: "
                           "Nile is the longest river, Amazon has the largest basin)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic — NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["len_rx"], text) and re.search(e["basin_rx"], text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} rivers' length+basin gathered ({', '.join(hits) or 'none'})"}


def validate_winner_density(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "winner_density", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> density value not credited"}
    ok = _density_value_present(_all_text(result))
    return {"check": "winner_density", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"winner's density within +/-3% of {WINNER_DENSITY:.2f} m/km^2 present" if ok
                       else f"no density near {WINNER_DENSITY:.2f} m/km^2 found")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} river pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_argmax, validate_coverage,
            validate_winner_density, validate_citation]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold. Five independent leaves (one page-read per
    river, both figures live on that one infobox), all arithmetic lives only in aggregation, which
    is forced to write out every division before concluding. Encodes STRUCTURE only: leaks no
    length, no basin area, no density, and not which river wins."""
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for the {e['name']} river and read, from the infobox, "
                "TWO figures: its LENGTH (in kilometres) and its DRAINAGE BASIN SIZE / basin area "
                "(in square kilometres, km^2). Report ONLY those two numbers, clearly labelled, "
                "and the source URL. Do not guess from memory, and do not divide or compute "
                "anything."
            ),
            "expect": "LENGTH (km) and BASIN AREA (km^2), both labelled -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five rivers, its length (km) and its drainage basin "
            "area (km^2). For EACH river, write out the division explicitly on its own line in "
            "the form '<river>: <length_in_metres> / <basin_area_km2> = <density>' (convert "
            "length from km to metres first by multiplying by 1000) -- compute every one of the "
            "five divisions BEFORE drawing any conclusion. THEN, comparing those five computed "
            "densities, state which SINGLE river has the HIGHEST channel-length density -- that "
            "river's name is the keystone answer. This need NOT be the longest river, nor the one "
            "with the largest drainage basin. Report (a) that river and its density value, (b) "
            "all five rivers' length and basin area, and (c) cite each river's source URL."
        ),
    }
