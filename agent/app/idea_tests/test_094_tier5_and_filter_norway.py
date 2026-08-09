"""
Test 094: Tier 5 (graph) — TWO-CONSTRAINT NUMERIC AND-FILTER (Norwegian fjords).
Level: graph   Weight: long   Difficulty: 9/10

A hard set-intersection task built to punish the cheap-model failure mode of dropping a constraint
(or short-circuiting on a single prominent attribute) instead of holding TWO independent numeric
lookups in mind and intersecting them. Among SIX given Norwegian fjords the agent must look up TWO
page-only numeric attributes per fjord and name the UNIQUE fjord that satisfies BOTH constraints
simultaneously:

    (A) fjord LENGTH is OVER 110 km, AND
    (B) fjord MAXIMUM DEPTH is UNDER 750 m.

The six fjords span a wide range and are engineered so that EACH single constraint is satisfied by
SEVERAL of them — so dropping EITHER constraint changes the answer set — yet EXACTLY ONE fjord
satisfies both. The keystone is the one fjord that is simultaneously long AND comparatively shallow,
a counter-intuitive combination (Norway's famous fjords are both long AND very deep) that is not
recoverable from fame or a single-attribute shortcut.

    Sognefjord       Hardangerfjord   Trondheimsfjord
    Romsdalsfjord    Tysfjorden       Lysefjord

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): collapses the two-way filter to one attribute or drops a constraint and
    mis-picks (a long-fjord shortcut → Sognefjord at 205 km but 1,308 m deep, or Hardangerfjord at
    179 km but 860 m deep; a shallow-fjord shortcut → Lysefjord at 422 m but only 42 km long); or
    defaults to a famous name from memory (Sognefjord / Geirangerfjord) without reading any page.
  * frontier sequential (ReAct): looks up all twelve attribute values, applies both filters,
    returns the single survivor → decent.
  * graph_compiled: SIX parallel leaves each fetch BOTH attributes for ONE fjord and RESTATE that
    fjord's name in the answer (so the aggregation's numbered, id-stripped facts_block keeps every
    length/depth figure bound to its fjord); the aggregation owns the entire AND-filter and is
    forced to WRITE OUT each fjord's constraint-by-constraint check before concluding → the cheap
    executor is rescued by structure and cannot short-circuit OR mis-bind a scattered figure.

The trap is engineered two ways — EACH constraint is necessary (drop one → wrong, larger set):
  (1) drop LENGTH constraint → {Trondheimsfjord, Romsdalsfjord, Lysefjord} satisfy depth < 750 m → 3
  (2) drop DEPTH constraint  → {Sognefjord, Hardangerfjord, Trondheimsfjord} satisfy length > 110 km → 3
  => the ONLY fjord in both sets is the Trondheimsfjord. No single constraint isolates it; the
     full two-way conjunction is required.

Ground truth (verified against live English Wikipedia 2026-07-07 — each fjord's infobox 'Max.
length' and 'Max. depth' rows, or the article's measurement text where the infobox omits depth,
English-language article):

  fjord              length (km)   max depth (m)   len>110?  depth<750?  both?
  Trondheimsfjord       130            617           YES        YES       YES  ← KEYSTONE
  Sognefjord            205          1,308           YES         no        no
  Hardangerfjord        179            860           YES         no        no
  Romsdalsfjord          88            550            no        YES        no
  Tysfjorden             62            897            no         no        no
  Lysefjord              42            422            no        YES        no
    https://en.wikipedia.org/wiki/Trondheimsfjord
    https://en.wikipedia.org/wiki/Sognefjord
    https://en.wikipedia.org/wiki/Hardangerfjord
    https://en.wikipedia.org/wiki/Romsdalsfjord
    https://en.wikipedia.org/wiki/Tysfjorden
    https://en.wikipedia.org/wiki/Lysefjord

  KEYSTONE = Trondheimsfjord (the unique fjord with length > 110 km AND max depth < 750 m).

  EACH SINGLE CONSTRAINT IS SATISFIED BY MORE THAN ONE FJORD (so the conjunction is necessary):
    length > 110 km     : Sognefjord, Hardangerfjord, Trondheimsfjord      (3 of 6)
    max depth < 750 m   : Trondheimsfjord, Romsdalsfjord, Lysefjord        (3 of 6)
  No single filter leaves Trondheimsfjord alone — only both applied together do.

  ROBUSTNESS / MARGINS (no borderline value where a plausible misread flips membership):
    LENGTH threshold = 110 km:
      Trondheimsfjord 130 km:  20 km above (18.2% margin) — clear positive
      Hardangerfjord  179 km:  69 km above (62.7%)        — very clear positive
      Sognefjord      205 km:  95 km above (86.4%)        — very clear positive
      Romsdalsfjord    88 km:  22 km below (20.0% of threshold) — unambiguous negative
      Tysfjorden       62 km:  48 km below (43.6%)        — unambiguous negative
      Lysefjord        42 km:  68 km below (61.8%)        — unambiguous negative
    DEPTH threshold = 750 m:
      Trondheimsfjord 617 m:  133 m below threshold (17.7% below) — clear positive (< 750)
      Romsdalsfjord   550 m:  200 m below threshold (26.7%)       — clear positive
      Lysefjord       422 m:  328 m below threshold (43.7%)       — very clear positive
      Hardangerfjord  860 m:  110 m above threshold (14.7% above) — clear negative
      Tysfjorden      897 m:  147 m above threshold (19.6% above) — clear negative
      Sognefjord    1,308 m:  558 m above threshold (74.4% above) — very clear negative
    The nearest value on each side of each threshold is ≥110 m / ≥20 km clear, so no realistic
    misread or source-rounding can flip membership.

  ANTI-PARAMETRIC:
    GROUNDING: keystone credit additionally requires observability.visit.count > 0 — a correct
    keystone value reported with zero page visits (a parametric-memory guess) is rejected.

    Sognefjord is the longest of the six (205 km) AND the deepest (1,308 m) — a model that picks
    "the longest fjord" or the most famous one is wrong; its depth far exceeds the 750 m threshold.
    Hardangerfjord is Norway's second-longest fjord (179 km) but at 860 m is too deep.
    Lysefjord is the shallowest here (422 m) — a "shallowest fjord" shortcut lands on a fjord only
    42 km long, well short of 110 km. Tysfjorden is a deep-but-short decoy (897 m, 62 km).
    The winner, Trondheimsfjord, is genuinely counter-intuitive: at 130 km it is long yet its
    maximum depth is only 617 m — long-but-shallow cuts against the parametric intuition that
    Norway's long fjords are also its deepest. This combination is not parametrically recallable —
    the agent must read both values from the Wikipedia pages.

  KEYSTONE = the unique satisfying ENTITY (Trondheimsfjord). Secondary (gated) = correctly states
  both its attribute values (length ~130 km, max depth ~617 m). Coverage (un-gated) = how many of
  the six fjords had both attributes gathered; length values are pairwise distinct
  (42 / 62 / 88 / 130 / 179 / 205 km), so length_rx attribution is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- verified fixtures (single source of truth for statement, validators, and the plan) -----
# Per-entity regexes are case-insensitive where applied. ``length_rx`` and ``depth_rx`` match the
# specific infobox values (with small tolerance for formatting) so coverage attribution is tight.
# Constraint booleans (``length_over`` / ``depth_under``) are used to DERIVE the winner only;
# nothing numeric is leaked into the task statement or the compiled plan.
LENGTH_THRESHOLD = 110   # km
DEPTH_THRESHOLD  = 750   # m

ENTITIES: List[Dict[str, Any]] = [
    {
        "key": "trondheim", "name": "Trondheimsfjord",
        "length": 130, "depth": 617,
        "length_over": True, "depth_under": True,
        "name_rx":   r"\btrondheim",
        "length_rx": r"(?<!\d)130(?!\d)",
        "depth_rx":  r"(?<!\d)617(?!\d)",
        "slug_rx":   r"wiki/trondheim",
    },
    {
        "key": "sognefjord", "name": "Sognefjord",
        "length": 205, "depth": 1_308,
        "length_over": True, "depth_under": False,
        "name_rx":   r"\bsognefjord",
        "length_rx": r"(?<!\d)205(?!\d)",
        "depth_rx":  r"(?<!\d)1[,\s]?308(?!\d)",
        "slug_rx":   r"wiki/sognefjord",
    },
    {
        "key": "hardanger", "name": "Hardangerfjord",
        "length": 179, "depth": 860,
        "length_over": True, "depth_under": False,
        "name_rx":   r"\bhardanger",
        "length_rx": r"(?<!\d)179(?!\d)",
        "depth_rx":  r"(?<!\d)86[0-9](?!\d)",
        "slug_rx":   r"wiki/hardangerfjord",
    },
    {
        "key": "romsdal", "name": "Romsdalsfjord",
        "length": 88, "depth": 550,
        "length_over": False, "depth_under": True,
        "name_rx":   r"\bromsdal",
        "length_rx": r"(?<!\d)88(?!\d)",
        "depth_rx":  r"(?<!\d)550(?!\d)",
        "slug_rx":   r"wiki/romsdal",
    },
    {
        "key": "tysfjord", "name": "Tysfjorden",
        "length": 62, "depth": 897,
        "length_over": False, "depth_under": False,
        "name_rx":   r"\btysfjord",
        "length_rx": r"(?<!\d)62(?!\d)",
        "depth_rx":  r"(?<!\d)897(?!\d)",
        "slug_rx":   r"wiki/tysfjord",
    },
    {
        "key": "lysefjord", "name": "Lysefjord",
        "length": 42, "depth": 422,
        "length_over": False, "depth_under": True,
        "name_rx":   r"\blysefjord",
        "length_rx": r"(?<!\d)42(?!\d)",
        "depth_rx":  r"(?<!\d)422(?!\d)",
        "slug_rx":   r"wiki/lysefjord",
    },
]

# Derive the keystone purely from the two constraint booleans (self-check: exactly one fjord
# satisfies length_over AND depth_under). If the fixtures ever drift, this raises at import.
_SATISFIERS = [e for e in ENTITIES if e["length_over"] and e["depth_under"]]
assert len(_SATISFIERS) == 1, (
    f"AND-filter must have exactly one survivor, got {[e['name'] for e in _SATISFIERS]}"
)
WINNER = _SATISFIERS[0]    # Trondheimsfjord — the unique satisfier

_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if e is not WINNER)

# AND-filter "winner" assertion triggers: words that assert a fjord is THE unique answer to the
# two-way intersection ("the only one", "satisfies both", "the answer is …", "qualifies").
_SAT = (
    r"only|sole|solely|unique|uniquely|satisfies both|meets both|both constraints|"
    r"fulfil|qualif|answer|sought|is the fjord"
)

# Keystone winner detection: Trondheimsfjord tied to an "is the only / satisfies both / is the
# answer" assertion, in either direction, with the proximity window forbidden from crossing into
# ANY other fjord's name.
#   dir 1 (subject → assertion): "Trondheimsfjord … is the only one satisfying both"
#   dir 2 (assertion → subject): "the only fjord meeting both constraints is the Trondheimsfjord"
_WINNER_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SAT + r")\b"
    + r"|\b(?:" + _SAT + r")\b(?:(?!" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Pattern that flags a non-winner fjord tied to an "is the answer / satisfies both" assertion.
# Used by _keystone_enumeration_ok and _keystone_ok to veto a wrong explicit assertion.
_OTHER_WINS = re.compile(
    r"(?:" + _OTHERS + r")(?:(?!" + _WINNER_RX + r")[^.;]){0,90}\b(?:" + _SAT + r")\b"
    + r"|\b(?:" + _SAT + r")\b(?:(?!" + _WINNER_RX + r")[^.;]){0,55}(?:" + _OTHERS + r")",
    re.IGNORECASE,
)


def _keystone_enumeration_ok(text: str) -> bool:
    """Credits enumeration-style answers where Trondheimsfjord's row shows length > 110 km AND
    max depth < 750 m (both constraints positive) but no explicit assertion word ('only',
    'unique', 'satisfies both', …) appears near its name.

    Handles aggregation outputs such as:
        Trondheimsfjord: length=130 km (>110? yes), depth=617 m (<750? yes) — URL
        Sognefjord: length=205 km (>110? yes), depth=1,308 m (<750? no)
        …

    Guards:
      • Trondheimsfjord's row must contain its length value (130) AND depth value (617) AND
        >=2 'yes' tokens (one per constraint).
      • No other fjord's SINGLE LINE may show >=2 'yes' tokens (guards against a hallucinated
        table where a decoy is also credited for both constraints).
      • Rejected outright if any non-winner fjord is tied to an assertion word.
    """
    # Distinctive substrings that identify any non-winner fjord anywhere on a line.
    _other_distinctive = r"sognefjord|hardanger|romsdal|tysfjord|lysefjord"

    # 1. Find Trondheimsfjord's row (name + up to 3 continuation lines that do NOT contain any
    #    other fjord's distinctive name anywhere on the line).
    wm = re.search(
        r"(?:" + _WINNER_RX + r")[^\n]*"
        r"(?:\n(?![^\n]*(?:" + _other_distinctive + r"))[^\n]*){0,3}",
        text, re.IGNORECASE,
    )
    if not wm:
        return False
    w_row = wm.group(0)

    # 2. Both attribute values must appear in the row.
    if not re.search(WINNER["length_rx"], w_row, re.IGNORECASE):
        return False
    if not re.search(WINNER["depth_rx"], w_row, re.IGNORECASE):
        return False

    # 3. At least 2 'yes' tokens (one for each constraint).
    if len(re.findall(r"\byes\b", w_row, re.IGNORECASE)) < 2:
        return False

    # 4. No other fjord's single line may show >=2 'yes' tokens. Using single-line match
    #    (no continuation) avoids cross-row spill.
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
        "test_id": "094",
        "test_name": "Tier 5: Two-constraint numeric AND-filter (Norwegian fjords)",
        "difficulty_level": "9/10",
        "category": "Multi-constraint set intersection (AND-filter, two numeric page-only attributes)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following six Norwegian fjords, open the fjord's English "
        "Wikipedia page and read TWO numeric attributes directly from the page: (A) the fjord's "
        "LENGTH in km, and (B) the fjord's MAXIMUM DEPTH in metres (from the infobox 'Max. depth' "
        "row, or the article's measurement text where the infobox omits it):\n"
        f"{listing}\n\n"
        "Then determine the SINGLE fjord that satisfies BOTH of the following constraints "
        "simultaneously:\n"
        "  (A) its length is OVER 110 km, AND\n"
        "  (B) its maximum depth is UNDER 750 m.\n\n"
        "Exactly one of the six fjords satisfies both. Note: several fjords satisfy each constraint "
        "on its own, so you must apply both constraints together — dropping either one gives a "
        "different, larger answer set. The winner is NOT necessarily the longest fjord in this set, "
        "nor the shallowest one.\n\n"
        "Report (a) which single fjord satisfies both constraints (the keystone), (b) that fjord's "
        "length in km and maximum depth in metres, (c) for all six fjords both attribute values you "
        "gathered, and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which single fjord has length over 110 km AND maximum depth under 750 m "
        "(the primary answer / keystone)",
        "That winning fjord's length (km) and maximum depth (m)",
        "For all six fjords, both attribute values gathered (length in km and max depth in m)",
        "Source URL for each fjord's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 6 pages visited (one per fjord, a six-way fan-out)",
        "Correctly names the Trondheimsfjord as the unique fjord satisfying both constraints (NOT "
        "Sognefjord, which is longer at 205 km but 1,308 m deep; NOT Hardangerfjord, which is "
        "179 km but 860 m deep; NOT Lysefjord, which is shallow at 422 m but only 42 km long; "
        "NOT Tysfjorden at 897 m and only 62 km; NOT Romsdalsfjord at only 88 km)",
        "States the winner's two attribute values: length ~130 km, max depth ~617 m",
        "Gathers both attributes (length and max depth) for each of the six fjords",
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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names the Trondheimsfjord as the unique both-constraint
    satisfier, AND the agent actually visited at least one page (grounding requirement — a
    correct-but-ungrounded parametric-memory guess must not earn credit).

    Three acceptance paths (any one suffices):
      1. Terse: only Trondheimsfjord named (no other fjord present) — assumed to be the answer.
      2. Explicit: Trondheimsfjord tied to an 'is the only / satisfies both / is the answer'
         assertion within a window that never crosses another fjord's name.
      3. Enumeration: Trondheimsfjord's row shows its length AND depth values AND >=2 'yes' tokens
         while no other fjord's row shows the same (handles table-style aggregation output).

    Rejected outright if Trondheimsfjord is absent, or if another fjord is explicitly asserted as
    the answer, or if the enumeration table is hallucinated (a decoy also shows both-positive), or
    if the agent visited zero pages.
    """
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    if not grounded:
        return False
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True  # path 1: only the winner is named
    if _WINNER_WINS.search(text):
        return True  # path 2: explicit assertion near Trondheimsfjord
    return _keystone_enumeration_ok(text)  # path 3: enumeration table


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per fjord."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 6.0),
        "reason": f"{n} visit(s) (target >=6: one page per fjord; >=4 to pass)",
    }


def validate_keystone_filter(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the unique fjord with length > 110 km AND max depth < 750 m is the
    Trondheimsfjord. Models that drop a constraint mis-pick: chasing a long fjord lands on
    Sognefjord (205 km but 1,308 m deep) or Hardangerfjord (179 km but 860 m deep); chasing a
    shallow fjord lands on Lysefjord (422 m but only 42 km); Tysfjorden is short (62 km) and deep
    (897 m); Romsdalsfjord is shallow enough (550 m) but only 88 km long."""
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_filter", "passed": passed, "score": 1.0 if passed else 0.0,
        "reason": ("Trondheimsfjord named as the unique both-constraint satisfier" if passed
                   else "Unique satisfier (Trondheimsfjord) missing/incorrect (beware: "
                        "Sognefjord is longer but 1,308 m deep; Hardangerfjord is 179 km but "
                        "860 m deep; Lysefjord is shallow but only 42 km; Tysfjorden 897 m/62 km; "
                        "Romsdalsfjord only 88 km)"),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: for how many of the SIX fjords were BOTH attributes gathered.

    A fjord is credited only when its name AND its (pairwise-distinct) length figure AND its depth
    figure all appear in the full reported text. The six length values (42 / 62 / 88 / 130 / 179 /
    205 km) are pairwise distinct, so length_rx attribution can never cross-credit between fjords.
    Deliberately NOT gated on the keystone — it measures whether the agent actually fanned out to
    all six fjords and pulled each one's two attributes even when it botches the intersection.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["length_rx"], text, re.IGNORECASE)
            and re.search(e["depth_rx"], text, re.IGNORECASE)]
    n = len(ENTITIES)
    return {
        "check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} fjords' two attributes gathered ({', '.join(hits) or 'none'})",
    }


def validate_winner_attributes(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winner's length (~130 km) and max depth (~617 m) are both stated.
    Short-circuits to 0 when the keystone is absent, so a wrong or guessed winner cannot bank the
    attribute-value credit."""
    if not _keystone_ok(result, observability):
        return {
            "check": "winner_attributes", "passed": False, "score": 0.0,
            "reason": "Keystone absent -> attribute values not credited",
        }
    text = _all_text(result)
    checks = {
        "length_~130_km":  bool(re.search(WINNER["length_rx"], text, re.IGNORECASE)),
        "depth_~617_m":    bool(re.search(WINNER["depth_rx"],  text, re.IGNORECASE)),
    }
    got = sum(checks.values())
    missing = [k for k, v in checks.items() if not v]
    return {
        "check": "winner_attributes", "passed": got == 2, "score": got / 2.0,
        "reason": ("both of Trondheimsfjord's attribute values stated (length ~130 km, "
                   "max depth ~617 m)" if got == 2
                   else f"{got}/2 winner attributes stated; missing: {', '.join(missing)}"),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 3 of the 6 fjord Wikipedia pages. Short-circuits to 0
    when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {
            "check": "citation", "passed": False, "score": 0.0,
            "reason": "Keystone absent -> source URLs not credited",
        }
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {
        "check": "citation", "passed": cited >= 3, "score": cited / n,
        "reason": f"{cited}/{n} fjord pages cited",
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
    # None -> the harness applies its default structured rubric judge.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves — ONE per GIVEN fjord, and each leaf reads BOTH of that fjord's
    attributes (its LENGTH in km AND its MAXIMUM DEPTH in metres) in a single page read, then
    RESTATES the fjord's name alongside both figures in its answer. Consolidating both attributes
    into one per-fjord leaf (instead of twelve attribute-per-leaf leaves) and forcing each leaf to
    self-describe its fjord fixes the aggregation mis-bind: the executor's facts_block numbers facts
    and STRIPS leaf ids, so a bare "617" figure with no fjord name could be reattached to the wrong
    fjord when twelve scattered numbers are reassembled. With six self-describing per-fjord leaves,
    every length/depth pair stays bound to its fjord name through aggregation. ALL filtering —
    applying both numeric constraints and intersecting them — still lives only in the aggregation
    step, which is forced to WRITE OUT each fjord's constraint-by-constraint check explicitly before
    concluding. Encodes STRUCTURE only: it names the six GIVEN fjords and the two GIVEN thresholds
    (length > 110 km / max depth < 750 m), but leaks no attribute value and never states which fjord
    wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_attributes",
            "instruction": (
                f"Open the English Wikipedia page for {e['name']} and read TWO numeric attributes "
                "directly from the page: (A) its LENGTH in km (the 'Max. length' / 'Length' row), "
                "and (B) its MAXIMUM DEPTH in metres (the 'Max. depth' infobox row, or the "
                "article's measurement text where the infobox omits it). "
                f"Report your finding in the exact form '{e['name']}: length = <number> km; "
                "maximum depth = <number> m' followed by the source URL, so the fjord's name stays "
                "attached to both of its figures. Do not guess from memory."
            ),
            "expect": (
                f"'{e['name']}: length = <number> km; maximum depth = <number> m' — source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six fjords, two attributes: its length in km and its "
            "maximum depth in metres. For EACH of the six fjords, write out its check explicitly "
            "on its own line in the form "
            "'<fjord>: length=<val> km (>110 km? yes/no), depth=<val> m (<750 m? yes/no)' "
            "— evaluate all six BEFORE drawing any conclusion. "
            "THEN identify the SINGLE fjord for which BOTH checks are yes (length over 110 km AND "
            "maximum depth under 750 m) — that fjord's name is the keystone answer. "
            "Show every fjord's two-way check before concluding. Exactly one fjord satisfies both; "
            "it need not be the longest fjord or the shallowest one. "
            "Report (a) that fjord's name and both attribute values, (b) all six fjords' two "
            "attributes, and (c) the source URL for each fjord's page."
        ),
    }
