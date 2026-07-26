"""
Test 064: Tier 5 (graph) — argmax over a COMPUTED RATIO (volume/area), WIDE-MARGIN hardening.
Level: graph   Weight: long   Difficulty: 9/10

A re-skin of the computed-ratio-argmax idea behind test 059 (goals-per-appearance), rebuilt to be
WIDE-MARGIN so it is NOT noisy. 059's flaw was a thin keystone margin (+16.6% over 2nd place), so a
single noisy extraction could nearly flip the ranking and the score carried high variance. 064 keeps
the same discriminating shape — a five-way fan-out, two page-only quantities per entity, ranked by a
ratio that neither page prints — but picks numbers so the winner clears 2nd place by ~495%.

Among FIVE lakes the agent must determine which has the highest VOLUME-TO-SURFACE-AREA ratio, where
that ratio is NOT printed as a single figure and must be COMPUTED by DIVIDING two looked-up infobox
numbers:

    ratio = water volume (km^3)  /  surface area (km^2)      ( = the lake's mean depth, in km;
                                                                x1000 to express it in metres )

The five lakes are deliberately a mix of one obvious GIANT and four lesser lakes. The ratio winner
is the SMALL, obscure member (Issyk-Kul) — the SMALLEST of the five by surface area and only third
by volume — so the comparison is NOT a memorized "fact": a parametric model that never opens the
pages cannot recall five obscure lake volumes/areas, and a "pick the biggest lake" shortcut lands on
the wrong one. Lake volumes and surface areas are obscure quantitative facts (unlike famous building
heights or film grosses), so this is parametric-leak resistant; and the figures are physically fixed
(natural lakes), so the ground truth does not drift the way census or box-office numbers do.

    Issyk-Kul (Kyrgyzstan)   Lake Victoria (East Africa)   Lake Ladoga (Russia)
    Lake Erie (North America)   Lake Winnipeg (Canada)

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): drops a lake, mis-divides, or shortcuts to the biggest raw quantity (the
    largest lake by volume AND by area is Lake Victoria) and names the WRONG lake; or guesses the
    most salient/famous name (Lake Victoria / Lake Erie) from memory.
  * frontier sequential (ReAct): looks up all ten figures, divides, compares -> decent.
  * graph_compiled: five parallel leaves each fetch ONE lake's two figures; the aggregation owns
    every division AND the argmax, and is forced to WRITE OUT each division before concluding -> the
    cheap executor is rescued and a diverse-grounding reranker can re-derive it.

The trap is engineered three ways:
  (1) the ratio is computed — the agent must DIVIDE volume by surface area for each lake and then
      compare the five quotients; no page states the cross-lake ranking;
  (2) the ratio winner DIFFERS from BOTH raw rankings — the largest-volume lake AND the largest-area
      lake are the SAME giant (Lake Victoria), which is NOT the winner, so "pick the biggest" fails
      on either axis;
  (3) the winner (Issyk-Kul) is the SMALLEST lake by surface area and only the THIRD largest by
      volume, so it is the least obvious candidate — nobody guesses the small lake.

Ground truth (verified against live English Wikipedia 2026-06-26 — each lake's infobox 'Volume' and
'Surface area' rows; volume in km^3, surface area in km^2; the computed ratio equals each lake's
stated 'Average depth', a cross-check used only during verification):

  lake             region          volume(km^3)  area(km^2)   ratio = vol/area (m)
  Issyk-Kul        Kyrgyzstan          1,736        6,236        278.4   <- ARGMAX (keystone)
  Lake Ladoga      Russia                837       17,891         46.8   <- 2nd place
  Lake Victoria    East Africa         2,424       59,947         40.4   <- biggest VOLUME + AREA (decoy)
  Lake Erie        North America         480       25,667         18.7
  Lake Winnipeg    Canada                294       24,514         12.0
    https://en.wikipedia.org/wiki/Issyk-Kul
    https://en.wikipedia.org/wiki/Lake_Victoria
    https://en.wikipedia.org/wiki/Lake_Ladoga
    https://en.wikipedia.org/wiki/Lake_Erie
    https://en.wikipedia.org/wiki/Lake_Winnipeg

  RATIO ARGMAX = Issyk-Kul (278.4 m). Runner-up = Lake Ladoga (46.8 m), so the margin is +231.6 m
  absolute (~+495% relative). This is the WIDE-MARGIN fix for 059: no plausible single noisy
  extraction can flip the keystone. Issyk-Kul's worst plausible slip (volume read 1,600 instead of
  1,736, or area read 6,500 instead of 6,236) still yields ~246-256 m, more than five times the
  runner-up's ~47 m. One — or several — noisy reads cannot change the keystone.

  WIDE-MARGIN / DOUBLE-DECOY checks (confirmed live, asserted at import time below):
    (a) ratio-winner (Issyk-Kul) != biggest-VOLUME lake (Lake Victoria, 2,424 km^3)      -> True
    (b) ratio-winner (Issyk-Kul) != biggest-AREA   lake (Lake Victoria, 59,947 km^2)     -> True
    (c) winner ratio (278.4 m) exceeds 2nd place (Ladoga, 46.8 m) by ~495% (>> 15%)       -> True
    by volume: Victoria 2,424 > Issyk-Kul 1,736 > Ladoga 837 > Erie 480 > Winnipeg 294   -> Victoria
    by area:   Victoria 59,947 > Erie 25,667 > Winnipeg 24,514 > Ladoga 17,891 > Issyk-Kul 6,236
                                                                                          -> Victoria
    by ratio:  Issyk-Kul 278.4 > Ladoga 46.8 > Victoria 40.4 > Erie 18.7 > Winnipeg 12.0 -> Issyk-Kul
    => the single most salient lake (Victoria, largest by BOTH raw quantities) is the doubly-baited
       decoy, and the obscure smallest-by-area lake (Issyk-Kul) wins the ratio by a commanding 495%.

  ANTI-PARAMETRIC: Lake Victoria (largest tropical lake, a household giant) and Lake Erie (a named
  Great Lake) are the salient guesses, but both are wrong; the keystone is Issyk-Kul, whose volume/
  area superiority is not a recallable fact — so the cheap model must READ the pages and DIVIDE
  rather than pattern-match a name from memory.

  KEYSTONE = the argmax LAKE (Issyk-Kul). Secondary (gated) value = its ratio ~278 m (mean depth),
  accepted within +/- 3% in metres OR the equivalent ~0.278 km. All ten figures are distinct, so the
  un-gated coverage diagnostic is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``ratio_m`` is shown for provenance only; nothing here is leaked into the task statement or plan.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "issyk_kul", "name": "Issyk-Kul", "region": "Kyrgyzstan",
     "volume": 1736, "area": 6236, "ratio_m": 278.4, "winner": True,
     "name_rx": r"issyk[\s\-]*k[uo]l",
     "vol_rx": r"(?<!\d)1[,\s]?736(?!\d)", "area_rx": r"(?<!\d)6[,\s]?236(?!\d)",
     "slug_rx": r"wiki/issyk[\s\-_]*kul"},
    {"key": "victoria", "name": "Lake Victoria", "region": "East Africa",
     "volume": 2424, "area": 59947, "ratio_m": 40.4, "winner": False,
     "name_rx": r"\bvictoria\b",
     "vol_rx": r"(?<!\d)2[,\s]?424(?!\d)", "area_rx": r"(?<!\d)59[,\s]?947(?!\d)",
     "slug_rx": r"wiki/lake_victoria"},
    {"key": "ladoga", "name": "Lake Ladoga", "region": "Russia",
     "volume": 837, "area": 17891, "ratio_m": 46.8, "winner": False,
     "name_rx": r"\bladoga\b",
     "vol_rx": r"(?<!\d)837(?!\d)", "area_rx": r"(?<!\d)17[,\s]?891(?!\d)",
     "slug_rx": r"wiki/lake_ladoga"},
    {"key": "erie", "name": "Lake Erie", "region": "North America",
     "volume": 480, "area": 25667, "ratio_m": 18.7, "winner": False,
     "name_rx": r"\berie\b",
     "vol_rx": r"(?<!\d)480(?!\d)", "area_rx": r"(?<!\d)25[,\s]?667(?!\d)",
     "slug_rx": r"wiki/lake_erie"},
    {"key": "winnipeg", "name": "Lake Winnipeg", "region": "Canada",
     "volume": 294, "area": 24514, "ratio_m": 12.0, "winner": False,
     "name_rx": r"\bwinnipeg\b",
     "vol_rx": r"(?<!\d)294(?!\d)", "area_rx": r"(?<!\d)24[,\s]?514(?!\d)",
     "slug_rx": r"wiki/lake_winnipeg"},
]

WINNER = next(e for e in ENTITIES if e["winner"])      # Issyk-Kul — the ratio argmax
WINNER_RATIO_M = 278.4                                  # 1,736 km^3 / 6,236 km^2 = 0.2784 km
RATIO_TOL = 0.03                                        # +/- 3% on the secondary value

# ---- WIDE-MARGIN / DOUBLE-DECOY invariants, asserted at import so the fixtures can never silently
# ---- drift out of the design (winner != either raw argmax; winner clears 2nd place by >15%). ----
_BY_VOL = max(ENTITIES, key=lambda e: e["volume"])
_BY_AREA = max(ENTITIES, key=lambda e: e["area"])
_BY_RATIO = sorted(ENTITIES, key=lambda e: e["volume"] / e["area"], reverse=True)
assert _BY_RATIO[0] is WINNER, "ratio argmax must be the declared winner"
assert _BY_VOL is not WINNER, "winner must NOT be the biggest-volume lake (decoy A)"
assert _BY_AREA is not WINNER, "winner must NOT be the biggest-area lake (decoy B)"
_W_RATIO = WINNER["volume"] / WINNER["area"]
_RUNNER_UP_RATIO = _BY_RATIO[1]["volume"] / _BY_RATIO[1]["area"]
assert (_W_RATIO - _RUNNER_UP_RATIO) / _RUNNER_UP_RATIO > 0.15, "keystone margin must exceed 15%"

# Winner / other-entity name regexes used by the keystone gate.
_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# Comparative / superlative triggers that assert a winner ("X has the highest / deepest ratio").
# Same battle-tested set as the football-ratio sibling (059), plus "deeper/deepest" because the
# ratio IS the lake's mean depth and is naturally asserted that way.
_SUP = (r"more|larger|greater|higher|bigger|largest|greatest|highest|biggest|most|maximum|best|top"
        r"|deeper|deepest")

# Keystone winner detection: the winner (Issyk-Kul) tied to a 'highest / deepest ratio' assertion,
# in either direction, with the proximity window forbidden from crossing into ANY other lake's name
# (so "Lake Victoria has the highest ratio; Issyk-Kul is second" can never satisfy it). The window is
# [^.;] — newline-tolerant (a header line then the answer below still matches) but bounded at
# sentence periods AND clause-separating semicolons, so a rival asserted as the winner in one clause
# cannot reach an Issyk-Kul mention in the next. The "than" guard blocks "deeper than Issyk-Kul"
# (Issyk-Kul the LOSER) from counting.
#   dir 1  (subject -> superlative):  "Issyk-Kul ... has the highest volume-to-area ratio"
#   dir 2  (superlative -> subject):  "the highest ratio is Issyk-Kul"  (NOT "deeper than Issyk-Kul")
_ISSYK_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Ratio-value detection: any integer/decimal up to three digits — "278", "278.4", "0.278", "0.28".
# Accepted when it lands in the winner's mean-depth band in METRES (~278) OR in KILOMETRES (~0.278).
_RATIO_NUM = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)(?!\d)")
# Or the exact fraction the agent might write out, "1,736 / 6,236".
_RATIO_FRAC = re.compile(r"\b1[,\s]?736\s*/\s*6[,\s]?236\b")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "064",
        "test_name": "Tier 5: Computed-ratio argmax, wide-margin (highest volume-to-surface-area)",
        "difficulty_level": "9/10",
        "category": "Quantitative reasoning + computed-ratio argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']} ({e['region']})" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following five lakes, open the lake's Wikipedia page and read "
        "TWO numbers from the infobox: the lake's water VOLUME (in cubic kilometres, km^3) and its "
        "SURFACE AREA (in square kilometres, km^2):\n"
        f"{listing}\n\n"
        "Then COMPUTE, for each lake, its VOLUME-TO-SURFACE-AREA ratio = volume / surface area "
        "(this quotient is the lake's mean depth; in km^3/km^2 it comes out in kilometres, so "
        "multiply by 1000 to read it in metres — no page prints this cross-lake comparison, you must "
        "divide the two figures yourself). COMPARE the five ratios and determine which lake has the "
        "HIGHEST volume-to-surface-area ratio. Note: the answer is the largest RATIO, which is NOT "
        "necessarily the lake with the largest volume, nor the one with the largest surface area.\n\n"
        "Report (a) which lake has the highest volume-to-surface-area ratio (a single lake name — "
        "the keystone), (b) that lake's computed ratio value (its mean depth, in metres), (c) all "
        "five lakes' volume and surface area (the ten figures you looked up), and (d) the exact "
        "source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which lake has the highest volume-to-surface-area ratio (the primary answer / keystone)",
        "That winning lake's computed ratio value (its mean depth, approximately, in metres)",
        "All five lakes' volume and surface area (the ten looked-up figures)",
        "Source URL for each lake's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per lake, a five-way fan-out)",
        "Correctly names Issyk-Kul as the highest volume-to-surface-area ratio (NOT Lake Victoria, "
        "which is the largest by BOTH volume and surface area)",
        "Reports the winner's ratio near 278 m (within +/- 3%, or the equivalent ~0.278 km)",
        "Gathers all ten figures (volume and surface area for each of the five lakes)",
        "Cites the source pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text. Prefer ``deliverables[0]`` (the contract's primary slot) when present;
    otherwise fall back to ``output.final_deliverable``."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: the final deliverable plus every deliverable slot, so coverage / value /
    citation checks can see figures the agent placed outside the primary answer slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names Issyk-Kul as the highest-ratio lake.

    Word-bounded, and NOT satisfied by merely listing Issyk-Kul among the five: when another lake is
    also named, Issyk-Kul must be the one tied to a 'highest / deepest ratio' assertion (tempered so
    a rival named as the winner — or 'deeper than Issyk-Kul' — never counts). A terse primary answer
    that names only the winner (Issyk-Kul, with no rival in the slot) also passes.

    GROUNDING: an ungrounded parametric-memory guess must not earn credit, so the winner is only
    credited when the agent actually visited at least one page (visit.count > 0).
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True  # names only the winner
    return bool(_ISSYK_WINS.search(text))


def _ratio_value_present(text: str) -> bool:
    """True when a number in the winner's ratio band appears: ~278 m (metres) OR ~0.278 km."""
    lo_m, hi_m = WINNER_RATIO_M * (1.0 - RATIO_TOL), WINNER_RATIO_M * (1.0 + RATIO_TOL)
    lo_km, hi_km = lo_m / 1000.0, hi_m / 1000.0
    for raw in _RATIO_NUM.findall(text):
        v = float(raw)
        if lo_m <= v <= hi_m or lo_km <= v <= hi_km:
            return True
    return bool(_RATIO_FRAC.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a five-way fan-out wants one page per lake."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per lake; >=4 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the highest volume-to-surface-area ratio is Issyk-Kul — NOT the largest
    lake by volume nor by surface area (both Lake Victoria). A model that shortcuts to a raw quantity
    or guesses the most salient name (Victoria / Erie) mis-picks."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Issyk-Kul named as the highest volume-to-surface-area ratio" if passed
                      else "Highest-ratio lake (Issyk-Kul) missing/incorrect (beware: Lake Victoria "
                           "is the largest by BOTH volume and surface area)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the FIVE ratios' inputs were gathered.

    A ratio's inputs are credited only when BOTH that lake's volume AND surface area appear
    (alongside its name); the ten figures are all distinct, so there is no cross-crediting.
    Deliberately NOT gated on the keystone — it measures whether the agent actually fanned out to all
    five lakes and collected both figures even when it botches the division or the argmax, the axis
    that separates a structured (multi-leaf) agent from a linear one that drops an entity.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["vol_rx"], text) and re.search(e["area_rx"], text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} lakes' volume+area gathered ({', '.join(hits) or 'none'})"}


def validate_winner_ratio(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winner's computed ratio (~278 m, within +/- 3%, or ~0.278 km).
    Short-circuits to 0 when the keystone is absent, so a wrong/guessed winner can't bank the
    value credit."""
    if not _keystone_ok(result, observability):
        return {"check": "winner_ratio", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> ratio value not credited"}
    ok = _ratio_value_present(_all_text(result))
    lo_m, hi_m = WINNER_RATIO_M * (1.0 - RATIO_TOL), WINNER_RATIO_M * (1.0 + RATIO_TOL)
    return {"check": "winner_ratio", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"winner's ratio within {lo_m:.1f}-{hi_m:.1f} m (or ~0.278 km) present" if ok
                       else f"no ratio in {lo_m:.1f}-{hi_m:.1f} m (expected ~{WINNER_RATIO_M} m)")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the lake pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} lake pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_argmax,
        validate_coverage,
        validate_winner_ratio,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), as in 059/060.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    FIVE INDEPENDENT parallel leaves: for each of the five GIVEN lakes, one leaf opens that lake's
    page and reads its two infobox figures — water volume (km^3) and surface area (km^2) — both of
    which sit on the SAME page, so one page-read per lake. ALL arithmetic — every volume/area
    division AND the final argmax — lives only in the aggregation step, which is forced to WRITE OUT
    each per-lake division explicitly before concluding, so the cheap executor never divides the
    wrong way or shortcuts to a raw maximum and a diverse-grounding reranker can re-derive and catch
    a slip. Encodes STRUCTURE only: it names the five GIVEN lakes and their regions, but leaks no
    volume figure, no area figure, no ratio, and not which lake wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for {e['name']} ({e['region']}) and read, from the "
                "infobox, TWO figures: the lake's water VOLUME (in cubic kilometres, km^3) and its "
                "SURFACE AREA (in square kilometres, km^2). Report ONLY those two numbers (clearly "
                "labelled which is volume and which is area) and the source URL. Do not guess from "
                "memory, and do not divide or compute anything."
            ),
            "expect": "VOLUME (km^3) and SURFACE AREA (km^2), both labelled -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five lakes, two figures: its water volume (km^3) and its "
            "surface area (km^2). For EACH of the five lakes, write out the division explicitly on "
            "its own line in the form '<lake>: <volume> / <area> = <ratio>' (in km^3/km^2 the "
            "quotient is in kilometres; multiply by 1000 for metres) — compute every one of the five "
            "divisions BEFORE drawing any conclusion. THEN, comparing those five computed ratios, "
            "state which SINGLE lake has the HIGHEST volume-to-surface-area ratio — that lake's name "
            "is the keystone answer. Show every division before concluding. This need NOT be the "
            "lake with the largest volume, nor the one with the largest surface area. Report (a) "
            "that lake and its ratio value (mean depth, in metres), (b) all five lakes' volume and "
            "surface area, and (c) cite each lake's source URL."
        ),
    }
