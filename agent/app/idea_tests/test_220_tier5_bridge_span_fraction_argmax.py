"""
Test 220: Tier 5 (graph) — argmax over a COMPUTED derived quantity: bridge "span fraction"
(longest single span / total bridge length).
Level: graph   Weight: long   Difficulty: 8/10

Sibling of 218/219/221 in the derivation-layer instrument cluster. Among FIVE suspension bridges
the agent must determine which has the HIGHEST span fraction, defined as:

    span fraction = longest (main) span (m) / total bridge length (m)      [dimensionless, 0-1]

No page states this cross-bridge comparison; each bridge's Wikipedia infobox gives only its own
Total length and Longest span, and the agent must divide.

Ground truth (verified against live English Wikipedia infoboxes, 2026-08-31; all figures in
metres, from the infobox's metric conversion where the primary figure is imperial):

  bridge                  total length(m)   longest span(m)   span fraction
  Humber Bridge (UK)            2,220             1,410           0.6351   <- ARGMAX (keystone)
  Akashi Kaikyo (Japan)         3,911             1,991           0.5091   <- 2nd; longest SPAN
  Golden Gate (US)               2,737            1,280           0.4677
  Verrazzano-Narrows (US)        4,176            1,298           0.3108
  Mackinac Bridge (US)           8,038            1,158           0.1440   <- longest TOTAL LENGTH
    https://en.wikipedia.org/wiki/Humber_Bridge
    https://en.wikipedia.org/wiki/Akashi_Kaiky%C5%8D_Bridge
    https://en.wikipedia.org/wiki/Golden_Gate_Bridge
    https://en.wikipedia.org/wiki/Verrazzano-Narrows_Bridge
    https://en.wikipedia.org/wiki/Mackinac_Bridge

  SPAN-FRACTION ARGMAX = Humber Bridge (0.6351, i.e. 63.5% of the bridge is the single main span).
  Runner-up = Akashi Kaikyo (0.5091), margin = +24.8% relative ((0.6351-0.5091)/0.5091).

  DOUBLE-DECOY (confirmed live, asserted at import time below):
    (a) fraction-winner (Humber, span 1,410 m) != LONGEST SPAN of the five (Akashi, 1,991 m) -> True
    (b) fraction-winner (Humber, total 2,220 m) != LONGEST TOTAL LENGTH (Mackinac, 8,038 m) -> True
    by total length: Mackinac 8,038 > Verrazzano 4,176 > Akashi 3,911 > Golden Gate 2,737 > Humber 2,220
    by longest span: Akashi 1,991 > Verrazzano 1,298 > Golden Gate 1,280 > Humber 1,410 (note: NOT
                      monotonic with total length -- Humber's span exceeds Golden Gate's and
                      Verrazzano's despite Humber being the SHORTEST bridge overall) > Mackinac 1,158
    by fraction:  Humber 0.635 > Akashi 0.509 > Golden Gate 0.468 > Verrazzano 0.311 > Mackinac 0.144
  => "pick the bridge with the longest span" (Akashi) and "pick the longest bridge overall"
     (Mackinac) are BOTH wrong; the actual winner is the SHORTEST bridge of the five overall, yet
     its main span still eats nearly two-thirds of that shorter total length.

  ANTI-PARAMETRIC: individual bridge lengths/spans are page-only facts; the cross-bridge
  span-fraction ranking is not a recallable statistic.

  KEYSTONE = the argmax BRIDGE (Humber Bridge). Secondary (gated) value = its span fraction
  (0.6351, i.e. ~63.5%), accepted within +/- 3% (relative). All ten raw figures are distinct and
  none collides with the derived value band (checked at import time).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


ENTITIES: List[Dict[str, Any]] = [
    {"key": "humber", "name": "Humber Bridge", "total_m": 2220, "span_m": 1410,
     "fraction": 1410 / 2220, "winner": True,
     "name_rx": r"humber",
     "t_rx": r"(?<!\d)2[,\s]?220(?!\d)", "s_rx": r"(?<!\d)1[,\s]?410(?!\d)",
     "slug_rx": r"wiki/humber_bridge"},
    {"key": "akashi", "name": "Akashi Kaikyo Bridge", "total_m": 3911, "span_m": 1991,
     "fraction": 1991 / 3911, "winner": False,
     "name_rx": r"akashi",
     "t_rx": r"(?<!\d)3[,\s]?911(?!\d)", "s_rx": r"(?<!\d)1[,\s]?991(?!\d)",
     "slug_rx": r"wiki/akashi"},
    {"key": "golden_gate", "name": "Golden Gate Bridge", "total_m": 2737, "span_m": 1280,
     "fraction": 1280 / 2737, "winner": False,
     "name_rx": r"golden\s+gate",
     "t_rx": r"(?<!\d)2[,\s]?73[7-9](?!\d)", "s_rx": r"(?<!\d)1[,\s]?280(?!\d)",
     "slug_rx": r"wiki/golden_gate_bridge"},
    {"key": "verrazzano", "name": "Verrazzano-Narrows Bridge", "total_m": 4176, "span_m": 1298,
     "fraction": 1298 / 4176, "winner": False,
     "name_rx": r"verrazzano",
     "t_rx": r"(?<!\d)4[,\s]?176(?!\d)", "s_rx": r"(?<!\d)1[,\s]?298(?!\d)",
     "slug_rx": r"wiki/verrazzano"},
    {"key": "mackinac", "name": "Mackinac Bridge", "total_m": 8038, "span_m": 1158,
     "fraction": 1158 / 8038, "winner": False,
     "name_rx": r"mackinac",
     "t_rx": r"(?<!\d)8[,\s]?038(?!\d)", "s_rx": r"(?<!\d)1[,\s]?158(?!\d)",
     "slug_rx": r"wiki/mackinac_bridge"},
]

WINNER = next(e for e in ENTITIES if e["winner"])          # Humber Bridge
WINNER_FRACTION = WINNER["fraction"]                         # ~0.6351
FRACTION_TOL = 0.03

_BY_SPAN = max(ENTITIES, key=lambda e: e["span_m"])
_BY_TOTAL = max(ENTITIES, key=lambda e: e["total_m"])
_BY_FRACTION = sorted(ENTITIES, key=lambda e: e["fraction"], reverse=True)
assert _BY_FRACTION[0] is WINNER, "span-fraction argmax must be the declared winner"
assert _BY_SPAN is not WINNER, "winner must NOT have the longest single span (decoy A: Akashi)"
assert _BY_TOTAL is not WINNER, "winner must NOT be the longest bridge overall (decoy B: Mackinac)"
_RUNNER_UP = _BY_FRACTION[1]
_MARGIN = (WINNER["fraction"] - _RUNNER_UP["fraction"]) / _RUNNER_UP["fraction"]
assert _MARGIN > 0.15, f"keystone margin {_MARGIN:.3f} must exceed 15%"

_WINNER_RX = "(?:" + WINNER["name_rx"] + ")"
_OTHERS = "(?:" + "|".join(e["name_rx"] for e in ENTITIES if not e["winner"]) + ")"
_SUP = r"more|larger|greater|higher|bigger|largest|greatest|highest|most|maximum|best|top"
_HUMBER_WINS = re.compile(
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


# A reported fraction may be a decimal (0.635), a percentage (63.5, 63.5%) or the fraction 1410/2220.
_FRACTION_NUM = re.compile(r"(?<!\d)(\d(?:\.\d+)?)(?!\d)")
_PERCENT_NUM = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "220",
        "test_name": "Tier 5: Computed span-fraction argmax (largest share of a bridge that is one span)",
        "difficulty_level": "8/10",
        "category": "Quantitative reasoning + computed-ratio argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following five suspension bridges, open its Wikipedia page "
        "and read TWO numbers from the infobox: the bridge's TOTAL LENGTH (in metres) and its "
        "LONGEST (main) SPAN (in metres):\n"
        f"{listing}\n\n"
        "Then COMPUTE, for each bridge, its SPAN FRACTION = longest span / total length (no page "
        "prints this cross-bridge comparison, you must divide yourself). COMPARE the five "
        "fractions and determine which bridge has the HIGHEST span fraction (i.e. the largest "
        "share of its total length taken up by the single main span). Note: this is NOT "
        "necessarily the bridge with the longest single span, nor the longest bridge overall.\n\n"
        "Report (a) which bridge has the highest span fraction (the keystone), (b) that bridge's "
        "computed span fraction (as a decimal or percentage), (c) all five bridges' total length "
        "and longest span (the ten figures you looked up), and (d) the exact source URL of every "
        "page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which bridge has the highest span fraction (the primary answer / keystone)",
        "That winning bridge's computed span fraction",
        "All five bridges' total length (m) and longest span (m)",
        "Source URL for each bridge's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per bridge, a five-way fan-out)",
        f"Correctly names {WINNER['name']} as the highest span fraction (NOT Akashi Kaikyo, which "
        "has the longest single span, and NOT Mackinac Bridge, which is the longest bridge overall)",
        f"Reports the winner's span fraction near {WINNER_FRACTION * 100:.1f}% (within +/- 3%)",
        "Gathers all ten figures (total length and longest span for each of the five bridges)",
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
    return bool(_HUMBER_WINS.search(text))


def _fraction_value_present(text: str) -> bool:
    lo, hi = WINNER_FRACTION * (1.0 - FRACTION_TOL), WINNER_FRACTION * (1.0 + FRACTION_TOL)
    for raw in _FRACTION_NUM.findall(text):
        v = float(raw)
        if lo <= v <= hi:
            return True
    lo_pct, hi_pct = lo * 100, hi * 100
    for raw in _PERCENT_NUM.findall(text):
        v = float(raw)
        if lo_pct <= v <= hi_pct:
            return True
    return False


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per bridge; >=4 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"{WINNER['name']} named as the highest span fraction" if passed
                      else f"Highest span-fraction bridge ({WINNER['name']}) missing/incorrect "
                           "(beware: Akashi Kaikyo has the longest span, Mackinac is the longest "
                           "bridge overall)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["t_rx"], text) and re.search(e["s_rx"], text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} bridges' length+span gathered ({', '.join(hits) or 'none'})"}


def validate_winner_fraction(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "winner_fraction", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> span fraction not credited"}
    ok = _fraction_value_present(_all_text(result))
    return {"check": "winner_fraction", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"winner's span fraction within +/-3% of {WINNER_FRACTION:.3f} "
                       f"(~{WINNER_FRACTION*100:.1f}%) present" if ok
                       else f"no span fraction near {WINNER_FRACTION:.3f} found")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} bridge pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_argmax, validate_coverage,
            validate_winner_fraction, validate_citation]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for {e['name']} and read, from the infobox, TWO "
                "figures: the bridge's TOTAL LENGTH (in metres) and its LONGEST (main) SPAN (in "
                "metres). Report ONLY those two numbers, clearly labelled, and the source URL. Do "
                "not guess from memory, and do not divide or compute anything."
            ),
            "expect": "TOTAL LENGTH (m) and LONGEST SPAN (m), both labelled -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five bridges, its total length (m) and its longest "
            "span (m). For EACH bridge, write out the division explicitly on its own line in the "
            "form '<bridge>: <span> / <total length> = <fraction>' -- compute every one of the "
            "five divisions BEFORE drawing any conclusion. THEN, comparing those five computed "
            "fractions, state which SINGLE bridge has the HIGHEST span fraction -- that bridge's "
            "name is the keystone answer. This need NOT be the bridge with the longest single "
            "span, nor the longest bridge overall. Report (a) that bridge and its span fraction, "
            "(b) all five bridges' total length and longest span, and (c) cite each bridge's "
            "source URL."
        ),
    }
