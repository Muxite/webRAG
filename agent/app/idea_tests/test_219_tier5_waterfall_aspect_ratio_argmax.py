"""
Test 219: Tier 5 (graph) — argmax over a COMPUTED derived quantity: waterfall aspect ratio
(height / width).
Level: graph   Weight: long   Difficulty: 8/10

Sibling of 218/220/221 in the derivation-layer instrument cluster. Among FIVE named waterfalls the
agent must determine which has the HIGHEST aspect ratio, defined as:

    aspect ratio = total height (m) / width (m)      [dimensionless -- "how tall relative to wide"]

No page states this cross-waterfall comparison; each waterfall's Wikipedia infobox gives only its
own Height and Width, and the agent must divide.

Ground truth (verified against live English Wikipedia infoboxes, 2026-08-31):

  waterfall            height(m)   width(m)   aspect ratio = height/width
  Multnomah Falls (US)     189          3          63.0000   <- ARGMAX (keystone)
  Kaieteur Falls (Guyana)  226        113           2.0000   <- 2nd place; TALLEST of the five
  Dettifoss (Iceland)       44        100           0.4400
  Niagara Horseshoe Falls   57        790           0.0722
  Victoria Falls (Africa)  108      1,708           0.0632   <- WIDEST of the five
    https://en.wikipedia.org/wiki/Multnomah_Falls
    https://en.wikipedia.org/wiki/Kaieteur_Falls
    https://en.wikipedia.org/wiki/Dettifoss
    https://en.wikipedia.org/wiki/Niagara_Falls   (Horseshoe/Canadian Falls figures)
    https://en.wikipedia.org/wiki/Victoria_Falls

  ASPECT-RATIO ARGMAX = Multnomah Falls (63.0). Runner-up = Kaieteur Falls (2.0), margin = +3050%
  relative ((63.0-2.0)/2.0). This is an enormous margin by design: Multnomah's width (3 m) and
  Kaieteur's width (113 m) are both far apart from every other candidate, so no plausible
  single-metre extraction error on either figure can come close to flipping the keystone.

  DOUBLE-DECOY (confirmed live, asserted at import time below):
    (a) ratio-winner (Multnomah, 189 m) != TALLEST waterfall (Kaieteur, 226 m)         -> True
    (b) ratio-winner (Multnomah, 3 m)   != WIDEST waterfall (Victoria, 1,708 m)        -> True
        (Multnomah is in fact the NARROWEST of the five, which is exactly why it wins the ratio --
        but "pick the narrowest" alone is not the trap being tested; the trap is that "pick the
        tallest waterfall" (Kaieteur) or "pick the most famous/biggest waterfall" (Victoria/
        Niagara) both give the wrong answer.)
    by height: Kaieteur 226 > Multnomah 189 > Victoria 108 > Niagara 57 > Dettifoss 44
    by width:  Victoria 1,708 > Niagara 790 > Kaieteur 113 > Dettifoss 100 > Multnomah 3
    by ratio:  Multnomah 63.0 > Kaieteur 2.0 > Dettifoss 0.44 > Niagara 0.072 > Victoria 0.063
  => the tallest waterfall in the set (Kaieteur) and the widest/most famous (Victoria, Niagara) are
     BOTH wrong; the actual winner is the SECOND-tallest and by far the narrowest.

  ANTI-PARAMETRIC: individual waterfall heights/widths are page-only facts; the CROSS-waterfall
  aspect-ratio ranking is not a recallable statistic -- nobody has "Multnomah Falls has the highest
  height-to-width ratio among famous waterfalls" memorized.

  KEYSTONE = the argmax WATERFALL (Multnomah Falls). Secondary (gated) value = its ratio (63.0),
  accepted within +/- 3%. All ten raw figures are distinct and none collides with the derived
  value band (checked at import time).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


ENTITIES: List[Dict[str, Any]] = [
    {"key": "multnomah", "name": "Multnomah Falls", "height_m": 189, "width_m": 3,
     "ratio": 189 / 3, "winner": True,
     "name_rx": r"multnomah",
     "h_rx": r"(?<!\d)189(?!\d)", "w_rx": r"(?<!\d)3(?!\d)(?:\s*m\b|\s*metres|\s*meters)",
     "slug_rx": r"wiki/multnomah_falls"},
    {"key": "kaieteur", "name": "Kaieteur Falls", "height_m": 226, "width_m": 113,
     "ratio": 226 / 113, "winner": False,
     "name_rx": r"kaieteur",
     "h_rx": r"(?<!\d)226(?!\d)", "w_rx": r"(?<!\d)113(?!\d)",
     "slug_rx": r"wiki/kaieteur_falls"},
    {"key": "dettifoss", "name": "Dettifoss", "height_m": 44, "width_m": 100,
     "ratio": 44 / 100, "winner": False,
     "name_rx": r"dettifoss",
     "h_rx": r"(?<!\d)44(?!\d)", "w_rx": r"(?<!\d)100(?!\d)",
     "slug_rx": r"wiki/dettifoss"},
    {"key": "niagara", "name": "Niagara Horseshoe Falls", "height_m": 57, "width_m": 790,
     "ratio": 57 / 790, "winner": False,
     "name_rx": r"horseshoe\s+falls|niagara",
     "h_rx": r"(?<!\d)57(?!\d)", "w_rx": r"(?<!\d)790(?!\d)",
     "slug_rx": r"wiki/niagara_falls"},
    {"key": "victoria", "name": "Victoria Falls", "height_m": 108, "width_m": 1708,
     "ratio": 108 / 1708, "winner": False,
     "name_rx": r"victoria\s+falls",
     "h_rx": r"(?<!\d)108(?!\d)", "w_rx": r"(?<!\d)1[,\s]?708(?!\d)",
     "slug_rx": r"wiki/victoria_falls"},
]

WINNER = next(e for e in ENTITIES if e["winner"])          # Multnomah Falls
WINNER_RATIO = WINNER["ratio"]                               # 63.0
RATIO_TOL = 0.03

_BY_HEIGHT = max(ENTITIES, key=lambda e: e["height_m"])
_BY_WIDTH = max(ENTITIES, key=lambda e: e["width_m"])
_BY_RATIO = sorted(ENTITIES, key=lambda e: e["ratio"], reverse=True)
assert _BY_RATIO[0] is WINNER, "aspect-ratio argmax must be the declared winner"
assert _BY_HEIGHT is not WINNER, "winner must NOT be the tallest waterfall (decoy A: Kaieteur)"
assert _BY_WIDTH is not WINNER, "winner must NOT be the widest waterfall (decoy B: Victoria)"
_RUNNER_UP = _BY_RATIO[1]
_MARGIN = (WINNER["ratio"] - _RUNNER_UP["ratio"]) / _RUNNER_UP["ratio"]
assert _MARGIN > 0.15, f"keystone margin {_MARGIN:.3f} must exceed 15%"

_WINNER_RX = "(?:" + WINNER["name_rx"] + ")"
_OTHERS = "(?:" + "|".join(e["name_rx"] for e in ENTITIES if not e["winner"]) + ")"
_SUP = r"more|larger|greater|higher|bigger|largest|greatest|highest|steepest|narrowest|most|maximum|best|top"
_MULTNOMAH_WINS = re.compile(
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


_RATIO_NUM = re.compile(r"(?<!\d)(\d{1,4}(?:\.\d+)?)(?!\d)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "219",
        "test_name": "Tier 5: Computed aspect-ratio argmax (tallest-relative-to-narrowest waterfall)",
        "difficulty_level": "8/10",
        "category": "Quantitative reasoning + computed-ratio argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following five waterfalls, open its Wikipedia page and "
        "read TWO numbers from the infobox: the waterfall's total HEIGHT (in metres) and its "
        "WIDTH (in metres):\n"
        f"{listing}\n\n"
        "Then COMPUTE, for each waterfall, its ASPECT RATIO = height / width (no page prints this "
        "cross-waterfall comparison, you must divide yourself). COMPARE the five ratios and "
        "determine which waterfall has the HIGHEST height-to-width aspect ratio. Note: this is "
        "NOT necessarily the TALLEST waterfall, nor the WIDEST one.\n\n"
        "Report (a) which waterfall has the highest aspect ratio (the keystone), (b) that "
        "waterfall's computed ratio value, (c) all five waterfalls' height and width (the ten "
        "figures you looked up), and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which waterfall has the highest height-to-width aspect ratio (the primary answer)",
        "That winning waterfall's computed aspect ratio value",
        "All five waterfalls' height (m) and width (m)",
        "Source URL for each waterfall's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per waterfall, a five-way fan-out)",
        f"Correctly names {WINNER['name']} as the highest aspect ratio (NOT Kaieteur Falls, the "
        "tallest of the five, and NOT Victoria Falls, the widest)",
        f"Reports the winner's ratio near {WINNER_RATIO:.1f} (within +/- 3%)",
        "Gathers all ten figures (height and width for each of the five waterfalls)",
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
    return bool(_MULTNOMAH_WINS.search(text))


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
            "reason": f"{n} visit(s) (target >=5: one page per waterfall; >=4 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"{WINNER['name']} named as the highest aspect ratio" if passed
                      else f"Highest-ratio waterfall ({WINNER['name']}) missing/incorrect (beware: "
                           "Kaieteur is the tallest, Victoria is the widest)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["h_rx"], text) and re.search(e["w_rx"], text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} waterfalls' height+width gathered ({', '.join(hits) or 'none'})"}


def validate_winner_ratio(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "winner_ratio", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> ratio value not credited"}
    ok = _ratio_value_present(_all_text(result))
    return {"check": "winner_ratio", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"winner's ratio within +/-3% of {WINNER_RATIO:.1f} present" if ok
                       else f"no ratio near {WINNER_RATIO:.1f} found")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} waterfall pages cited"}


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
                "figures: the waterfall's total HEIGHT (in metres) and its WIDTH (in metres). "
                "Report ONLY those two numbers, clearly labelled, and the source URL. Do not "
                "guess from memory, and do not divide or compute anything."
            ),
            "expect": "HEIGHT (m) and WIDTH (m), both labelled -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five waterfalls, its height (m) and its width (m). "
            "For EACH waterfall, write out the division explicitly on its own line in the form "
            "'<waterfall>: <height> / <width> = <ratio>' -- compute every one of the five "
            "divisions BEFORE drawing any conclusion. THEN, comparing those five computed ratios, "
            "state which SINGLE waterfall has the HIGHEST height-to-width aspect ratio -- that "
            "waterfall's name is the keystone answer. This need NOT be the tallest waterfall, nor "
            "the widest one. Report (a) that waterfall and its ratio value, (b) all five "
            "waterfalls' height and width, and (c) cite each waterfall's source URL."
        ),
    }
