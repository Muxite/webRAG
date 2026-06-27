"""
Test 076: Tier 5 (graph) — TWO-CONSTRAINT NUMERIC AND-FILTER (obscure lakes).
Level: graph   Weight: long   Difficulty: 9/10

A hard set-intersection task built to punish the cheap-model failure mode of dropping a constraint
(or short-circuiting on a single prominent attribute) instead of holding TWO independent numeric
lookups in mind and intersecting them. Among SIX given obscure-to-moderate lakes the agent must
look up TWO page-only numeric attributes per lake and name the UNIQUE lake that satisfies BOTH
constraints simultaneously:

    (A) surface area OVER 5,000 km², AND
    (B) maximum depth UNDER 20 m.

The six lakes are solid-but-obscure (no Great Lakes, no Lake Victoria, no Lake Baikal) and are
engineered so that EACH single constraint is satisfied by SEVERAL of them — so dropping EITHER
constraint changes the answer set — yet EXACTLY ONE lake satisfies both. The keystone is the one
lake that is simultaneously large AND unusually shallow, a counter-intuitive combination that is
not recoverable from fame or a single-attribute shortcut.

    Lake Winnipegosis   Nettilling Lake   Reindeer Lake
    Lake Athabasca      Lake Okeechobee   Lake Khanka

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): collapses the two-way filter to one attribute or drops a constraint and
    mis-picks (a large-area shortcut → Athabasca, which is 124 m deep; a shallow-depth shortcut →
    Okeechobee, which is only 1,900 km²; or Khanka, which is only 4,070 km²); or defaults to a
    famous name from memory without reading either page.
  * frontier sequential (ReAct): looks up all twelve attribute values, applies both filters,
    returns the single survivor → decent.
  * graph_compiled: twelve parallel leaves each fetch ONE attribute for ONE lake; the aggregation
    owns the entire AND-filter and is forced to WRITE OUT each lake's constraint-by-constraint
    check before concluding → the cheap executor is rescued by structure and cannot short-circuit.

The trap is engineered two ways — EACH constraint is necessary (drop one → wrong, larger set):
  (1) drop AREA constraint  → {Winnipegosis, Okeechobee, Khanka} satisfy depth < 20 m  → 3 lakes
  (2) drop DEPTH constraint → {Winnipegosis, Nettilling, Reindeer, Athabasca} satisfy area > 5,000 → 4 lakes
  => the ONLY lake in both sets is Lake Winnipegosis. No single constraint isolates it; the
     full two-way conjunction is required.

Ground truth (verified against live English Wikipedia 2026-06-27 — each lake's infobox 'Surface
area' and 'Max. depth' rows, English-language article):

  lake                 area (km²)   max depth (m)   area>5000?  depth<20?  both?
  Lake Winnipegosis      5,370          12             YES         YES      YES  ← KEYSTONE
  Nettilling Lake        5,542         132             YES          no       no
  Reindeer Lake          6,650         219             YES          no       no
  Lake Athabasca         7,849         124             YES          no       no
  Lake Okeechobee        1,900           3.7            no          YES      no
  Lake Khanka            4,070          10.6            no          YES      no
    https://en.wikipedia.org/wiki/Lake_Winnipegosis
    https://en.wikipedia.org/wiki/Nettilling_Lake
    https://en.wikipedia.org/wiki/Reindeer_Lake
    https://en.wikipedia.org/wiki/Lake_Athabasca
    https://en.wikipedia.org/wiki/Lake_Okeechobee
    https://en.wikipedia.org/wiki/Lake_Khanka

  KEYSTONE = Lake Winnipegosis (the unique lake with area > 5,000 km² AND max depth < 20 m).

  EACH SINGLE CONSTRAINT IS SATISFIED BY MORE THAN ONE LAKE (so the conjunction is necessary):
    area > 5,000 km²  : Winnipegosis, Nettilling, Reindeer, Athabasca   (4 of 6)
    max depth < 20 m  : Winnipegosis, Okeechobee, Khanka                (3 of 6)
  No single filter leaves Winnipegosis alone — only both applied together do.

  ROBUSTNESS / MARGINS (no borderline value where a plausible misread flips membership):
    AREA threshold = 5,000 km²:
      Winnipegosis  5,370 km²:   370 km² above (6.9% margin) — clear positive
      Nettilling    5,542 km²:   542 km² above (9.8%)        — clear positive
      Reindeer      6,650 km²: 1,650 km² above (24.8%)       — clear positive
      Athabasca     7,849 km²: 2,849 km² above (36.3%)       — clear positive
      Okeechobee    1,900 km²: 3,100 km² below (62% of threshold) — unambiguous negative
      Khanka        4,070 km²:   930 km² below (18.6%)       — clear negative
    DEPTH threshold = 20 m:
      Winnipegosis   12 m:  8 m below threshold (40% below)  — clear positive (< 20 m)
      Okeechobee    3.7 m: 16.3 m below threshold (81.5%)    — clear positive
      Khanka       10.6 m:  9.4 m below threshold (47%)      — clear positive
      Nettilling   132 m: 112 m above threshold (5.6× threshold) — unambiguous negative
      Reindeer     219 m: 199 m above threshold (10× threshold)  — unambiguous negative
      Athabasca    124 m: 104 m above threshold (5.2×)           — unambiguous negative
    All six values straddle their respective thresholds with margins large enough that no realistic
    misread or source-rounding can flip membership.

  ANTI-PARAMETRIC:
    Lake Athabasca is a well-known Canadian lake (famous "fame decoy" for area), but its max depth
    of 124 m is far above the 20 m threshold — a model that picks "the famous large lake" is wrong.
    Lake Okeechobee is famous as Florida's largest lake and is genuinely very shallow (3.7 m), but
    its area of 1,900 km² falls far short of 5,000 km² — a "shallowest / most famous shallow lake"
    shortcut is wrong.
    Lake Khanka (Russia/China border) is a depth decoy: shallow (10.6 m) but too small (4,070 km²).
    Nettilling Lake and Reindeer Lake are obscure area decoys: large enough (> 5,000 km²) but deep.
    The winner, Lake Winnipegosis, is genuinely obscure: it lies in Manitoba, Canada, near but
    distinct from the more famous Lake Winnipeg, and its specific combination of area (5,370 km²)
    with max depth of only 12 m is not parametrically recallable — the agent must read the page.

  KEYSTONE = the unique satisfying ENTITY (Lake Winnipegosis). Secondary (gated) = correctly states
  both its attribute values (area ~5,370 km², max depth ~12 m). Coverage (un-gated) = how many of
  the six lakes had both attributes gathered; area values are pairwise distinct (1,900 / 4,070 /
  5,370 / 5,542 / 6,650 / 7,849), so coverage attribution is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# Per-entity regexes are case-insensitive where applied. ``area_rx`` and ``depth_rx`` match the
# specific infobox values (with small tolerance for formatting) so coverage attribution is tight.
# Constraint booleans (``area_over`` / ``depth_under``) are used to DERIVE the winner only;
# nothing numeric is leaked into the task statement or the compiled plan.
AREA_THRESHOLD = 5_000   # km²
DEPTH_THRESHOLD = 20     # metres

ENTITIES: List[Dict[str, Any]] = [
    {
        "key": "winnipegosis", "name": "Lake Winnipegosis",
        "area": 5_370, "depth": 12.0, "area_over": True, "depth_under": True,
        "name_rx":  r"\bwinnipegosis\b",
        "area_rx":  r"(?<!\d)5[,\s]?37[0-9](?!\d)",
        "depth_rx": r"(?<!\d)1[12](?!\d)",
        "slug_rx":  r"wiki/lake_winnipegosis|wiki/winnipegosis",
    },
    {
        "key": "nettilling", "name": "Nettilling Lake",
        "area": 5_542, "depth": 132.0, "area_over": True, "depth_under": False,
        "name_rx":  r"\bnettilling\b",
        "area_rx":  r"(?<!\d)5[,\s]?54[0-9](?!\d)",
        "depth_rx": r"(?<!\d)13[0-5](?!\d)",
        "slug_rx":  r"wiki/nettilling",
    },
    {
        "key": "reindeer", "name": "Reindeer Lake",
        "area": 6_650, "depth": 219.0, "area_over": True, "depth_under": False,
        "name_rx":  r"\breindeer\s+lake\b",
        "area_rx":  r"(?<!\d)6[,\s]?6[45][0-9](?!\d)",
        "depth_rx": r"(?<!\d)21[5-9](?!\d)|(?<!\d)22[0-4](?!\d)",
        "slug_rx":  r"wiki/reindeer_lake",
    },
    {
        "key": "athabasca", "name": "Lake Athabasca",
        "area": 7_849, "depth": 124.0, "area_over": True, "depth_under": False,
        "name_rx":  r"\bathabasca\b",
        "area_rx":  r"(?<!\d)7[,\s]?8[45][0-9](?!\d)",
        "depth_rx": r"(?<!\d)12[2-6](?!\d)",
        "slug_rx":  r"wiki/lake_athabasca",
    },
    {
        "key": "okeechobee", "name": "Lake Okeechobee",
        "area": 1_900, "depth": 3.7, "area_over": False, "depth_under": True,
        "name_rx":  r"\bokeechobee\b",
        "area_rx":  r"(?<!\d)1[,\s]?9[0-9][0-9](?!\d)",
        "depth_rx": r"(?<!\d)3\.7(?!\d)|(?<!\d)[34](?!\d)\s*m\b",
        "slug_rx":  r"wiki/lake_okeechobee",
    },
    {
        "key": "khanka", "name": "Lake Khanka",
        "area": 4_070, "depth": 10.6, "area_over": False, "depth_under": True,
        "name_rx":  r"\bkhanka\b",
        "area_rx":  r"(?<!\d)4[,\s]?07[0-9](?!\d)",
        "depth_rx": r"(?<!\d)10\.6(?!\d)|(?<!\d)10(?!\.[0-9])(?!\d)\s*m\b",
        "slug_rx":  r"wiki/lake_khanka|wiki/khanka_lake",
    },
]

# Derive the keystone purely from the two constraint booleans (self-check: exactly one lake
# satisfies area_over AND depth_under). If the fixtures ever drift, this raises at import.
_SATISFIERS = [e for e in ENTITIES if e["area_over"] and e["depth_under"]]
assert len(_SATISFIERS) == 1, (
    f"AND-filter must have exactly one survivor, got {[e['name'] for e in _SATISFIERS]}"
)
WINNER = _SATISFIERS[0]    # Lake Winnipegosis — the unique satisfier

_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if e is not WINNER)

# AND-filter "winner" assertion triggers: words that assert a lake is THE unique answer to the
# two-way intersection ("the only one", "satisfies both", "the answer is …", "qualifies").
_SAT = (
    r"only|sole|solely|unique|uniquely|satisfies both|meets both|both constraints|"
    r"fulfil|qualif|answer|sought|is the lake"
)

# Keystone winner detection: Winnipegosis tied to an "is the only / satisfies both / is the
# answer" assertion, in either direction, with the proximity window forbidden from crossing into
# ANY other lake's name. Pattern mirrors test_068.
#   dir 1 (subject → assertion): "Lake Winnipegosis … is the only one satisfying both"
#   dir 2 (assertion → subject): "the only lake meeting both constraints is Winnipegosis"
_WINNIPEGOSIS_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SAT + r")\b"
    + r"|\b(?:" + _SAT + r")\b(?:(?!" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Pattern that flags a non-winner lake tied to an "is the answer / satisfies both" assertion.
# Used by _keystone_enumeration_ok and _keystone_ok to veto a wrong explicit assertion.
_OTHER_WINS = re.compile(
    r"(?:" + _OTHERS + r")(?:(?!" + _WINNER_RX + r")[^.;]){0,90}\b(?:" + _SAT + r")\b"
    + r"|\b(?:" + _SAT + r")\b(?:(?!" + _WINNER_RX + r")[^.;]){0,55}(?:" + _OTHERS + r")",
    re.IGNORECASE,
)


def _keystone_enumeration_ok(text: str) -> bool:
    """Credits enumeration-style answers where Winnipegosis's row shows area > 5,000 km² AND
    max depth < 20 m (both constraints positive) but no explicit assertion word ('only', 'unique',
    'satisfies both', …) appears near its name.

    Handles aggregation outputs such as:
        Lake Winnipegosis: area=5,370 km² (>5,000? yes), max depth=12 m (<20 m? yes) — URL
        Nettilling Lake: area=5,542 km² (>5,000? yes), max depth=132 m (<20 m? no)
        …

    Guards:
      • Winnipegosis's row must contain its area value (≈5,370) AND depth value (≈12) AND
        >=2 'yes' tokens (one per constraint).
      • No other lake's SINGLE LINE may show >=2 'yes' tokens (guards against a hallucinated
        table where a decoy is also credited for both constraints).
      • Rejected outright if any non-winner lake is tied to an assertion word.

    Implementation note: the continuation-line blocker for the winner's row uses
    ``[^\n]*`` inside the lookahead so it catches lake names preceded by "Lake " (e.g.
    "Lake Athabasca:…") even when the distinctive word is not at the start of the line.
    For the decoy check, only the single matched line is examined (no continuation), which
    avoids cross-row spillover entirely.
    """
    # Distinctive substrings that identify any non-winner lake anywhere on a line.
    _other_distinctive = r"nettilling|reindeer\s+lake|athabasca|okeechobee|khanka"

    # 1. Find Winnipegosis's row (name + up to 3 continuation lines that do NOT contain any
    #    other lake's distinctive name anywhere on the line).
    wm = re.search(
        r"(?:" + _WINNER_RX + r")[^\n]*"
        r"(?:\n(?![^\n]*(?:" + _other_distinctive + r"))[^\n]*){0,3}",
        text, re.IGNORECASE,
    )
    if not wm:
        return False
    w_row = wm.group(0)

    # 2. Both attribute values must appear in the row.
    if not re.search(WINNER["area_rx"], w_row, re.IGNORECASE):
        return False
    if not re.search(WINNER["depth_rx"], w_row, re.IGNORECASE):
        return False

    # 3. At least 2 'yes' tokens (one for each constraint).
    if len(re.findall(r"\byes\b", w_row, re.IGNORECASE)) < 2:
        return False

    # 4. No other lake's single line may show >=2 'yes' tokens.  Using single-line match
    #    (no continuation) avoids cross-row spill when lake names are preceded by "Lake ".
    for e in ENTITIES:
        if e is WINNER:
            continue
        om = re.search(r"(?:" + e["name_rx"] + r")[^\n]*", text, re.IGNORECASE)
        if not om:
            continue
        if len(re.findall(r"\byes\b", om.group(0), re.IGNORECASE)) >= 2:
            return False  # hallucinated: a decoy shows both-positive

    # 5. Reject if any non-winner is explicitly asserted as "the answer / only / satisfies both".
    if _OTHER_WINS.search(text):
        return False

    return True


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "076",
        "test_name": "Tier 5: Two-constraint numeric AND-filter (obscure lakes)",
        "difficulty_level": "9/10",
        "category": "Multi-constraint set intersection (AND-filter, two numeric page-only attributes)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following six lakes, open the lake's English Wikipedia page "
        "and read TWO numeric attributes directly from the infobox: (A) the lake's SURFACE AREA in "
        "km², and (B) the lake's MAXIMUM DEPTH in metres:\n"
        f"{listing}\n\n"
        "Then determine the SINGLE lake that satisfies BOTH of the following constraints "
        "simultaneously:\n"
        "  (A) its surface area is OVER 5,000 km², AND\n"
        "  (B) its maximum depth is UNDER 20 m.\n\n"
        "Exactly one of the six lakes satisfies both. Note: several lakes satisfy each constraint "
        "on its own, so you must apply both constraints together — dropping either one gives a "
        "different, larger answer set. The winner is NOT necessarily the largest lake in area, nor "
        "the shallowest in depth.\n\n"
        "Report (a) which single lake satisfies both constraints (the keystone), (b) that lake's "
        "surface area in km² and maximum depth in metres, (c) for all six lakes both attribute "
        "values you gathered, and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which single lake has surface area over 5,000 km² AND maximum depth under 20 m "
        "(the primary answer / keystone)",
        "That winning lake's surface area (km²) and maximum depth (m)",
        "For all six lakes, both attribute values gathered (surface area in km² and max depth in m)",
        "Source URL for each lake's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 6 pages visited (one per lake, a six-way fan-out)",
        "Correctly names Lake Winnipegosis as the unique lake satisfying both constraints (NOT "
        "Lake Athabasca, which is large but 124 m deep; NOT Lake Okeechobee, which is very shallow "
        "but only 1,900 km²; NOT Lake Khanka, which is shallow but only 4,070 km²)",
        "States the winner's two attribute values: area ~5,370 km², max depth ~12 m",
        "Gathers both attributes (area and max depth) for each of the six lakes",
        "Cites the source pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text — prefer deliverables[0] when present, else final_deliverable."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text — final deliverable plus all deliverable slots."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: deliverables[0] names Lake Winnipegosis as the unique both-constraint
    satisfier.

    Three acceptance paths (any one suffices):
      1. Terse: only Winnipegosis named (no other lake present) — assumed to be the answer.
      2. Explicit: Winnipegosis tied to an 'is the only / satisfies both / is the answer'
         assertion within a window that never crosses another lake's name.
      3. Enumeration: Winnipegosis's row shows its area AND depth values AND >=2 'yes' tokens
         while no other lake's row shows the same (handles table-style aggregation output where
         the model writes per-lake checks without using assertion words).

    Rejected outright if Winnipegosis is absent, or if another lake is explicitly asserted as the
    answer, or if the enumeration table is hallucinated (a decoy also shows both-positive).
    """
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True  # path 1: only the winner is named
    if _WINNIPEGOSIS_WINS.search(text):
        return True  # path 2: explicit assertion near Winnipegosis
    return _keystone_enumeration_ok(text)  # path 3: enumeration table


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per lake."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 6.0),
        "reason": f"{n} visit(s) (target >=6: one page per lake; >=4 to pass)",
    }


def validate_keystone_filter(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the unique lake with area > 5,000 km² AND max depth < 20 m is Lake
    Winnipegosis. Models that drop a constraint mis-pick: chasing a large lake lands on Athabasca
    (7,849 km² but 124 m deep); chasing a shallow lake lands on Okeechobee (3.7 m deep but only
    1,900 km²) or Khanka (10.6 m but only 4,070 km²); Nettilling and Reindeer are large but deep."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_filter", "passed": passed, "score": 1.0 if passed else 0.0,
        "reason": ("Lake Winnipegosis named as the unique both-constraint satisfier" if passed
                   else "Unique satisfier (Lake Winnipegosis) missing/incorrect (beware: "
                        "Athabasca is large but 124 m deep; Okeechobee is shallow but only "
                        "1,900 km²; Khanka is shallow but only 4,070 km²; Nettilling and "
                        "Reindeer are large but deep)"),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: for how many of the SIX lakes were BOTH attributes gathered.

    A lake is credited only when its name AND its (pairwise-distinct) surface-area figure AND its
    max-depth figure all appear in the full reported text. The six area values (1,900 / 4,070 /
    5,370 / 5,542 / 6,650 / 7,849) are pairwise distinct, so area_rx attribution can never
    cross-credit between lakes. Deliberately NOT gated on the keystone — it measures whether the
    agent actually fanned out to all six lakes and pulled each one's two attributes even when it
    botches the intersection.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["area_rx"], text, re.IGNORECASE)
            and re.search(e["depth_rx"], text, re.IGNORECASE)]
    n = len(ENTITIES)
    return {
        "check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} lakes' two attributes gathered ({', '.join(hits) or 'none'})",
    }


def validate_winner_attributes(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winner's area (~5,370 km²) and max depth (~12 m) are both stated.
    Short-circuits to 0 when the keystone is absent, so a wrong or guessed winner cannot bank the
    attribute-value credit."""
    if not _keystone_ok(result):
        return {
            "check": "winner_attributes", "passed": False, "score": 0.0,
            "reason": "Keystone absent -> attribute values not credited",
        }
    text = _all_text(result)
    checks = {
        "area_~5370_km2": bool(re.search(WINNER["area_rx"], text, re.IGNORECASE)),
        "depth_~12_m":    bool(re.search(WINNER["depth_rx"], text, re.IGNORECASE)),
    }
    got = sum(checks.values())
    missing = [k for k, v in checks.items() if not v]
    return {
        "check": "winner_attributes", "passed": got == 2, "score": got / 2.0,
        "reason": ("both of Lake Winnipegosis's attribute values stated (area ~5,370 km², "
                   "max depth ~12 m)" if got == 2
                   else f"{got}/2 winner attributes stated; missing: {', '.join(missing)}"),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 3 of the 6 lake Wikipedia pages. Short-circuits to 0
    when the keystone is absent."""
    if not _keystone_ok(result):
        return {
            "check": "citation", "passed": False, "score": 0.0,
            "reason": "Keystone absent -> source URLs not credited",
        }
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {
        "check": "citation", "passed": cited >= 3, "score": cited / n,
        "reason": f"{cited}/{n} lake pages cited",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_filter,
        validate_coverage,
        validate_winner_attributes,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in test_068.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    TWELVE INDEPENDENT parallel leaves: for each of the six GIVEN lakes, one leaf reads its
    SURFACE AREA in km², and one reads its MAXIMUM DEPTH in metres — one atomic attribute per
    leaf. ALL filtering — applying both numeric constraints and intersecting them — lives only in
    the aggregation step, which is forced to WRITE OUT each lake's constraint-by-constraint check
    explicitly before concluding. Encodes STRUCTURE only: it names the six GIVEN lakes and the
    two GIVEN thresholds (area > 5,000 km² / depth < 20 m), but leaks no attribute value and
    never states which lake wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_area",
            "instruction": (
                f"Open the English Wikipedia page for {e['name']} and read its SURFACE AREA in "
                "km² directly from the infobox ('Surface area' or 'Area' row). Report ONLY that "
                "single number (in km²) and the source URL. Do not guess from memory."
            ),
            "expect": "SURFACE AREA in km² (a single number) — source URL",
            "depends_on": [],
        })
        leaves.append({
            "id": f"{e['key']}_depth",
            "instruction": (
                f"Open the English Wikipedia page for {e['name']} and read its MAXIMUM DEPTH in "
                "metres directly from the infobox ('Max. depth' or 'Maximum depth' row). Report "
                "ONLY that single number (in metres) and the source URL. Do not guess from memory."
            ),
            "expect": "MAXIMUM DEPTH in metres (a single number) — source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six lakes, two attributes: its surface area in km² and "
            "its maximum depth in metres. For EACH of the six lakes, write out its check "
            "explicitly on its own line in the form "
            "'<lake>: area=<val> km² (>5,000 km²? yes/no), max depth=<val> m (<20 m? yes/no)' "
            "— evaluate all six BEFORE drawing any conclusion. "
            "THEN identify the SINGLE lake for which BOTH checks are yes (area over 5,000 km² "
            "AND max depth under 20 m) — that lake's name is the keystone answer. "
            "Show every lake's two-way check before concluding. Exactly one lake satisfies both; "
            "it need not be the largest lake in area or the shallowest in depth. "
            "Report (a) that lake's name and both attribute values, (b) all six lakes' two "
            "attributes, and (c) the source URL for each lake's page."
        ),
    }
