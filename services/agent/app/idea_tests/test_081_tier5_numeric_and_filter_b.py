"""
Test 081: Tier 5 (graph) — TWO-CONSTRAINT NUMERIC AND-FILTER (obscure rivers).
Level: graph   Weight: long   Difficulty: 9/10

A hard set-intersection task built to punish the cheap-model failure mode of dropping a constraint
(or short-circuiting on a single prominent attribute) instead of holding TWO independent numeric
lookups in mind and intersecting them. Among SIX given obscure-to-moderate rivers the agent must
look up TWO page-only numeric attributes per river and name the UNIQUE river that satisfies BOTH
constraints simultaneously:

    (A) river length OVER 800 km, AND
    (B) drainage basin area UNDER 35,000 km².

The six rivers are solid-but-obscure (no Amazon, no Nile, no Rhine) and are engineered so that
EACH single constraint is satisfied by SEVERAL of them — so dropping EITHER constraint changes the
answer set — yet EXACTLY ONE river satisfies both. The keystone is the one river that is
simultaneously long AND unusually compact in drainage area, a counter-intuitive combination that
is not recoverable from fame or a single-attribute shortcut.

    Prut River      Dniester River    Sava River
    Tisza River     Vardar River      Neretva River

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): collapses the two-way filter to one attribute or drops a constraint and
    mis-picks (a long-river shortcut → Dniester at 1,362 km but basin 68,627 km²; a small-basin
    shortcut → Neretva at 11,798 km² but only 225 km long; or Vardar at ~25,000 km² but only
    388 km); or defaults to a famous name from memory without reading any page.
  * frontier sequential (ReAct): looks up all twelve attribute values, applies both filters,
    returns the single survivor → decent.
  * graph_compiled: SIX parallel leaves each fetch BOTH attributes for ONE river and RESTATE that
    river's name in the answer (so the aggregation's numbered, id-stripped facts_block keeps every
    length/basin figure bound to its river); the aggregation owns the entire AND-filter and is
    forced to WRITE OUT each river's constraint-by-constraint check before concluding → the cheap
    executor is rescued by structure and cannot short-circuit OR mis-bind a scattered figure.

The trap is engineered two ways — EACH constraint is necessary (drop one → wrong, larger set):
  (1) drop LENGTH constraint  → {Prut, Vardar, Neretva} satisfy basin < 35,000 km²  → 3 rivers
  (2) drop BASIN constraint   → {Prut, Dniester, Sava, Tisza} satisfy length > 800 km → 4 rivers
  => the ONLY river in both sets is the Prut River. No single constraint isolates it; the
     full two-way conjunction is required.

Ground truth (verified against live English Wikipedia 2026-06-27 — each river's infobox
'Length' and 'Basin size' rows, English-language article):

  river              length (km)   basin (km²)   len>800?  basin<35k?  both?
  Prut River            953         27,540          YES        YES       YES  ← KEYSTONE
  Dniester River      1,362         68,627          YES         no        no
  Sava River            992         97,713          YES         no        no
  Tisza River           966        157,186          YES         no        no
  Vardar River          388        ~25,000           no        YES        no
  Neretva River         225         11,798           no        YES        no
    https://en.wikipedia.org/wiki/Prut
    https://en.wikipedia.org/wiki/Dniester
    https://en.wikipedia.org/wiki/Sava
    https://en.wikipedia.org/wiki/Tisza
    https://en.wikipedia.org/wiki/Vardar
    https://en.wikipedia.org/wiki/Neretva

  KEYSTONE = Prut River (the unique river with length > 800 km AND basin < 35,000 km²).

  EACH SINGLE CONSTRAINT IS SATISFIED BY MORE THAN ONE RIVER (so the conjunction is necessary):
    length > 800 km     : Prut, Dniester, Sava, Tisza        (4 of 6)
    basin < 35,000 km²  : Prut, Vardar, Neretva              (3 of 6)
  No single filter leaves Prut alone — only both applied together do.

  ROBUSTNESS / MARGINS (no borderline value where a plausible misread flips membership):
    LENGTH threshold = 800 km:
      Prut        953 km:   153 km above (19.1% margin) — clear positive
      Dniester  1,362 km:   562 km above (70.3%)         — very clear positive
      Sava        992 km:   192 km above (24.0%)          — clear positive
      Tisza       966 km:   166 km above (20.8%)          — clear positive
      Vardar      388 km:   412 km below (48.5% of threshold) — unambiguous negative
      Neretva     225 km:   575 km below (28.1% of threshold) — unambiguous negative
    BASIN threshold = 35,000 km²:
      Prut     27,540 km²:  7,460 km² below threshold (21.3% below) — clear positive (< 35,000)
      Vardar  ~25,000 km²: 10,000 km² below threshold (28.6%)       — clear positive
      Neretva  11,798 km²: 23,202 km² below threshold (66.3%)       — very clear positive
      Dniester 68,627 km²: 33,627 km² above threshold (96.1% above) — clear negative
      Sava     97,713 km²: 62,713 km² above threshold (179%)        — very clear negative
      Tisza   157,186 km²:122,186 km² above threshold (349%)        — extremely clear negative
    All six values straddle their respective thresholds with margins large enough that no realistic
    misread or source-rounding can flip membership.

  ANTI-PARAMETRIC:
    Dniester River is the longest of the six (1,362 km) — a model that picks "the longest river"
    is wrong; its drainage basin (68,627 km²) far exceeds the 35,000 km² threshold.
    Neretva River has the smallest basin (11,798 km²) — a "smallest basin" shortcut lands on a
    river only 225 km long, well short of 800 km.
    Vardar River also has a small basin (~25,000 km², Wikipedia infobox states "around 25,000")
    but is only 388 km long.
    Sava and Tisza are both > 800 km but have enormous basins (97,713 and 157,186 km²).
    The winner, Prut River, is genuinely counter-intuitive: at 953 km it is fairly long yet its
    drainage basin is only 27,540 km² — a compact corridor because the river runs roughly parallel
    to the Siret River, hemmed in on both sides. This combination is not parametrically
    recallable — the agent must read both values from the Wikipedia pages.

  KEYSTONE = the unique satisfying ENTITY (Prut River). Secondary (gated) = correctly states
  both its attribute values (length ~953 km, basin ~27,540 km²). Coverage (un-gated) = how many
  of the six rivers had both attributes gathered; length values are pairwise distinct
  (225 / 388 / 953 / 966 / 992 / 1,362 km), so length_rx attribution is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- verified fixtures (single source of truth for statement, validators, and the plan) -----
# Per-entity regexes are case-insensitive where applied. ``length_rx`` and ``basin_rx`` match
# the specific infobox values (with small tolerance for formatting) so coverage attribution is
# tight. Constraint booleans (``length_over`` / ``basin_under``) are used to DERIVE the winner
# only; nothing numeric is leaked into the task statement or the compiled plan.
LENGTH_THRESHOLD = 800    # km
BASIN_THRESHOLD  = 35_000  # km²

ENTITIES: List[Dict[str, Any]] = [
    {
        "key": "prut", "name": "Prut River",
        "length": 953, "basin": 27_540,
        "length_over": True, "basin_under": True,
        "name_rx":   r"\bprut\b",
        "length_rx": r"(?<!\d)953(?!\d)",
        "basin_rx":  r"(?<!\d)27[,\s]?54[0-9](?!\d)",
        "slug_rx":   r"wiki/prut",
    },
    {
        "key": "dniester", "name": "Dniester River",
        "length": 1_362, "basin": 68_627,
        "length_over": True, "basin_under": False,
        "name_rx":   r"\bdniester\b",
        "length_rx": r"(?<!\d)1[,\s]?36[0-9](?!\d)",
        "basin_rx":  r"(?<!\d)68[,\s]?6[0-9][0-9](?!\d)",
        "slug_rx":   r"wiki/dniester",
    },
    {
        "key": "sava", "name": "Sava River",
        "length": 992, "basin": 97_713,
        "length_over": True, "basin_under": False,
        "name_rx":   r"\bsava\b",
        "length_rx": r"(?<!\d)99[0-9](?!\d)",
        "basin_rx":  r"(?<!\d)97[,\s]?7[0-9][0-9](?!\d)",
        "slug_rx":   r"wiki/sava",
    },
    {
        "key": "tisza", "name": "Tisza River",
        "length": 966, "basin": 157_186,
        "length_over": True, "basin_under": False,
        "name_rx":   r"\btisza\b",
        "length_rx": r"(?<!\d)96[4-8](?!\d)",
        "basin_rx":  r"(?<!\d)15[5-9][,\s]?\d{3}(?!\d)",
        "slug_rx":   r"wiki/tisza",
    },
    {
        "key": "vardar", "name": "Vardar River",
        "length": 388, "basin": 25_000,
        "length_over": False, "basin_under": True,
        "name_rx":   r"\bvardar\b",
        "length_rx": r"(?<!\d)388(?!\d)",
        "basin_rx":  r"(?<!\d)25[,\s]?000(?!\d)",
        "slug_rx":   r"wiki/vardar",
    },
    {
        "key": "neretva", "name": "Neretva River",
        "length": 225, "basin": 11_798,
        "length_over": False, "basin_under": True,
        "name_rx":   r"\bneretva\b",
        "length_rx": r"(?<!\d)22[3-7](?!\d)",
        "basin_rx":  r"(?<!\d)11[,\s]?7[89][0-9](?!\d)",
        "slug_rx":   r"wiki/neretva",
    },
]

# Derive the keystone purely from the two constraint booleans (self-check: exactly one river
# satisfies length_over AND basin_under). If the fixtures ever drift, this raises at import.
_SATISFIERS = [e for e in ENTITIES if e["length_over"] and e["basin_under"]]
assert len(_SATISFIERS) == 1, (
    f"AND-filter must have exactly one survivor, got {[e['name'] for e in _SATISFIERS]}"
)
WINNER = _SATISFIERS[0]    # Prut River — the unique satisfier

_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if e is not WINNER)

# AND-filter "winner" assertion triggers: words that assert a river is THE unique answer to the
# two-way intersection ("the only one", "satisfies both", "the answer is …", "qualifies").
_SAT = (
    r"only|sole|solely|unique|uniquely|satisfies both|meets both|both constraints|"
    r"fulfil|qualif|answer|sought|is the river"
)

# Keystone winner detection: Prut tied to an "is the only / satisfies both / is the answer"
# assertion, in either direction, with the proximity window forbidden from crossing into ANY
# other river's name. Pattern mirrors test_076.
#   dir 1 (subject → assertion): "Prut River … is the only one satisfying both"
#   dir 2 (assertion → subject): "the only river meeting both constraints is the Prut"
_PRUT_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SAT + r")\b"
    + r"|\b(?:" + _SAT + r")\b(?:(?!" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Pattern that flags a non-winner river tied to an "is the answer / satisfies both" assertion.
# Used by _keystone_enumeration_ok and _keystone_ok to veto a wrong explicit assertion.
_OTHER_WINS = re.compile(
    r"(?:" + _OTHERS + r")(?:(?!" + _WINNER_RX + r")[^.;]){0,90}\b(?:" + _SAT + r")\b"
    + r"|\b(?:" + _SAT + r")\b(?:(?!" + _WINNER_RX + r")[^.;]){0,55}(?:" + _OTHERS + r")",
    re.IGNORECASE,
)


def _keystone_enumeration_ok(text: str) -> bool:
    """Credits enumeration-style answers where Prut River's row shows length > 800 km AND
    basin < 35,000 km² (both constraints positive) but no explicit assertion word ('only',
    'unique', 'satisfies both', …) appears near its name.

    Handles aggregation outputs such as:
        Prut River: length=953 km (>800? yes), basin=27,540 km² (<35,000? yes) — URL
        Dniester River: length=1,362 km (>800? yes), basin=68,627 km² (<35,000? no)
        …

    Guards:
      • Prut's row must contain its length value (≈953) AND basin value (≈27,540) AND
        >=2 'yes' tokens (one per constraint).
      • No other river's SINGLE LINE may show >=2 'yes' tokens (guards against a hallucinated
        table where a decoy is also credited for both constraints).
      • Rejected outright if any non-winner river is tied to an assertion word.

    Implementation note: the continuation-line blocker for the winner's row uses ``[^\n]*``
    inside the lookahead so it catches river names preceded by "River" even when the
    distinctive word is not at the start of the line. For the decoy check, only the single
    matched line is examined (no continuation), which avoids cross-row spillover entirely.
    """
    # Distinctive substrings that identify any non-winner river anywhere on a line.
    _other_distinctive = r"dniester|sava|tisza|vardar|neretva"

    # 1. Find Prut's row (name + up to 3 continuation lines that do NOT contain any other
    #    river's distinctive name anywhere on the line).
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
    if not re.search(WINNER["basin_rx"], w_row, re.IGNORECASE):
        return False

    # 3. At least 2 'yes' tokens (one for each constraint).
    if len(re.findall(r"\byes\b", w_row, re.IGNORECASE)) < 2:
        return False

    # 4. No other river's single line may show >=2 'yes' tokens. Using single-line match
    #    (no continuation) avoids cross-row spill when river names are preceded by "River".
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
        "test_id": "081",
        "test_name": "Tier 5: Two-constraint numeric AND-filter (obscure rivers)",
        "difficulty_level": "9/10",
        "category": "Multi-constraint set intersection (AND-filter, two numeric page-only attributes)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following six rivers, open the river's English Wikipedia "
        "page and read TWO numeric attributes directly from the infobox: (A) the river's LENGTH "
        "in km, and (B) the river's DRAINAGE BASIN AREA in km²:\n"
        f"{listing}\n\n"
        "Then determine the SINGLE river that satisfies BOTH of the following constraints "
        "simultaneously:\n"
        "  (A) its length is OVER 800 km, AND\n"
        "  (B) its drainage basin area is UNDER 35,000 km².\n\n"
        "Exactly one of the six rivers satisfies both. Note: several rivers satisfy each "
        "constraint on its own, so you must apply both constraints together — dropping either "
        "one gives a different, larger answer set. The winner is NOT necessarily the longest "
        "river in this set, nor the one with the smallest basin.\n\n"
        "Report (a) which single river satisfies both constraints (the keystone), (b) that "
        "river's length in km and drainage basin area in km², (c) for all six rivers both "
        "attribute values you gathered, and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which single river has length over 800 km AND drainage basin under 35,000 km² "
        "(the primary answer / keystone)",
        "That winning river's length (km) and drainage basin area (km²)",
        "For all six rivers, both attribute values gathered (length in km and basin in km²)",
        "Source URL for each river's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 6 pages visited (one per river, a six-way fan-out)",
        "Correctly names the Prut River as the unique river satisfying both constraints (NOT "
        "Dniester, which is longer at 1,362 km but has a basin of 68,627 km²; NOT Neretva, "
        "which has a small basin of 11,798 km² but is only 225 km long; NOT Vardar, which "
        "has a small basin but is only 388 km long; NOT Sava at 992 km with basin 97,713 km²; "
        "NOT Tisza at 966 km with basin 157,186 km²)",
        "States the winner's two attribute values: length ~953 km, basin ~27,540 km²",
        "Gathers both attributes (length and basin area) for each of the six rivers",
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
    """KEYSTONE gate: deliverables[0] names the Prut River as the unique both-constraint
    satisfier.

    Three acceptance paths (any one suffices):
      1. Terse: only Prut named (no other river present) — assumed to be the answer.
      2. Explicit: Prut tied to an 'is the only / satisfies both / is the answer' assertion
         within a window that never crosses another river's name.
      3. Enumeration: Prut's row shows its length AND basin values AND >=2 'yes' tokens while
         no other river's row shows the same (handles table-style aggregation output where the
         model writes per-river checks without using assertion words).

    Rejected outright if Prut is absent, or if another river is explicitly asserted as the
    answer, or if the enumeration table is hallucinated (a decoy also shows both-positive).
    """
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True  # path 1: only the winner is named
    if _PRUT_WINS.search(text):
        return True  # path 2: explicit assertion near Prut
    return _keystone_enumeration_ok(text)  # path 3: enumeration table


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per river."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 6.0),
        "reason": f"{n} visit(s) (target >=6: one page per river; >=4 to pass)",
    }


def validate_keystone_filter(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the unique river with length > 800 km AND basin < 35,000 km² is
    the Prut River. Models that drop a constraint mis-pick: chasing a long river lands on
    Dniester (1,362 km but basin 68,627 km²) or Sava (992 km but basin 97,713 km²); chasing
    a small basin lands on Neretva (11,798 km² but only 225 km) or Vardar (~25,000 km² but
    only 388 km); Tisza is long enough at 966 km but has an enormous basin (157,186 km²)."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_filter", "passed": passed, "score": 1.0 if passed else 0.0,
        "reason": ("Prut River named as the unique both-constraint satisfier" if passed
                   else "Unique satisfier (Prut River) missing/incorrect (beware: "
                        "Dniester is longer but basin is 68,627 km²; Neretva has a small "
                        "basin but is only 225 km long; Vardar has a small basin but is only "
                        "388 km; Sava 992 km has basin 97,713 km²; "
                        "Tisza 966 km has basin 157,186 km²)"),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: for how many of the SIX rivers were BOTH attributes gathered.

    A river is credited only when its name AND its (pairwise-distinct) length figure AND its
    basin figure all appear in the full reported text. The six length values (225 / 388 / 953 /
    966 / 992 / 1,362 km) are pairwise distinct, so length_rx attribution can never cross-credit
    between rivers. Deliberately NOT gated on the keystone — it measures whether the agent
    actually fanned out to all six rivers and pulled each one's two attributes even when it
    botches the intersection.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["length_rx"], text, re.IGNORECASE)
            and re.search(e["basin_rx"], text, re.IGNORECASE)]
    n = len(ENTITIES)
    return {
        "check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} rivers' two attributes gathered ({', '.join(hits) or 'none'})",
    }


def validate_winner_attributes(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winner's length (~953 km) and basin (~27,540 km²) are both stated.
    Short-circuits to 0 when the keystone is absent, so a wrong or guessed winner cannot bank
    the attribute-value credit."""
    if not _keystone_ok(result):
        return {
            "check": "winner_attributes", "passed": False, "score": 0.0,
            "reason": "Keystone absent -> attribute values not credited",
        }
    text = _all_text(result)
    checks = {
        "length_~953_km":   bool(re.search(WINNER["length_rx"], text, re.IGNORECASE)),
        "basin_~27540_km2": bool(re.search(WINNER["basin_rx"],  text, re.IGNORECASE)),
    }
    got = sum(checks.values())
    missing = [k for k, v in checks.items() if not v]
    return {
        "check": "winner_attributes", "passed": got == 2, "score": got / 2.0,
        "reason": ("both of Prut River's attribute values stated (length ~953 km, "
                   "basin ~27,540 km²)" if got == 2
                   else f"{got}/2 winner attributes stated; missing: {', '.join(missing)}"),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 3 of the 6 river Wikipedia pages. Short-circuits to 0
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
        "reason": f"{cited}/{n} river pages cited",
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
    # None -> the harness applies its default structured rubric judge, as in test_076.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves — ONE per GIVEN river, and each leaf reads BOTH of that
    river's attributes (its LENGTH in km AND its DRAINAGE BASIN AREA in km²) in a single page
    read, then RESTATES the river's name alongside both figures in its answer. Consolidating both
    attributes into one per-river leaf (instead of twelve attribute-per-leaf leaves) and forcing
    each leaf to self-describe its river fixes the aggregation mis-bind: the executor's facts_block
    numbers facts and STRIPS leaf ids, so a bare "953" figure with no river name could be reattached
    to the wrong river when twelve scattered numbers are reassembled. With six self-describing
    per-river leaves, every length/basin pair stays bound to its river name through aggregation.
    ALL filtering — applying both numeric constraints and intersecting them — still lives only in
    the aggregation step, which is forced to WRITE OUT each river's constraint-by-constraint check
    explicitly before concluding. Encodes STRUCTURE only: it names the six GIVEN rivers and the
    two GIVEN thresholds (length > 800 km / basin < 35,000 km²), but leaks no attribute value
    and never states which river wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_attributes",
            "instruction": (
                f"Open the English Wikipedia page for {e['name']} and read TWO numeric attributes "
                "directly from its infobox: (A) its LENGTH in km (the 'Length' row under Physical "
                "characteristics), and (B) its DRAINAGE BASIN AREA in km² (the 'Basin size', "
                "'Drainage basin', or 'Watershed area' row). "
                f"Report your finding in the exact form '{e['name']}: length = <number> km; "
                "drainage basin area = <number> km²' followed by the source URL, so the river's "
                "name stays attached to both of its figures. Do not guess from memory."
            ),
            "expect": (
                f"'{e['name']}: length = <number> km; drainage basin area = <number> km²' — "
                "source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six rivers, two attributes: its length in km and its "
            "drainage basin area in km². For EACH of the six rivers, write out its check "
            "explicitly on its own line in the form "
            "'<river>: length=<val> km (>800 km? yes/no), basin=<val> km² (<35,000 km²? yes/no)' "
            "— evaluate all six BEFORE drawing any conclusion. "
            "THEN identify the SINGLE river for which BOTH checks are yes (length over 800 km "
            "AND basin under 35,000 km²) — that river's name is the keystone answer. "
            "Show every river's two-way check before concluding. Exactly one river satisfies "
            "both; it need not be the longest river or the one with the smallest basin. "
            "Report (a) that river's name and both attribute values, (b) all six rivers' two "
            "attributes, and (c) the source URL for each river's page."
        ),
    }
