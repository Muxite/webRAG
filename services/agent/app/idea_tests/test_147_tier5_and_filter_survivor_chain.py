r"""
Test 147: Tier 5 (graph) — TWO-CONSTRAINT AND-FILTER -> SURVIVOR -> DISAMBIGUATED CHAIN TERMINUS.
Level: graph   Weight: long   Difficulty: 10/10

COMPOUND SHAPE (three axis TYPES stacked inside one visit budget). The suite's AND-filter tasks
(081, 094) stop at the survivor's own name; its branch-eliminate tasks (095, 116-121) start from a
categorical elimination, not a two-way numeric intersection. This task chains them:

    STAGE 1  AND-FILTER — six lakes, TWO page-only numeric attributes each; exactly ONE lake
             satisfies BOTH constraints (surface area OVER 250 km² AND maximum depth OVER 200 m).
             Each constraint on its own is satisfied by THREE of the six, so dropping either one
             leaves a different, larger answer set — the conjunction is load-bearing.
    STAGE 2  DISAMBIGUATION — the survivor's outflow is not a river but a NAMED SECTION of a major
             European river. That river has several differently-named sections, and one of the
             OTHER sections flows INTO the same lake. The agent must pick the section that flows
             OUT, not the one that flows in, and not the whole river.
    STAGE 3  CHAIN TERMINUS (the keystone) — open THAT SECTION's own article and read its length.

  Golden path = 7 visits (six lake pages + the outflow section's page), inside the 6-9 budget.

Ground truth (verified against live English Wikipedia via the MediaWiki API, 2026-08-06 — each
lake's infobox 'area' and 'max_depth' rows, and Lake Constance's infobox 'inflow'/'outflow' rows):

  lake                 area (km²)   max depth (m)   area>250?   depth>200?   both?
  -------------------  -----------  --------------  ----------  -----------  -----
  Lake Constance          536            251           YES         YES        YES  <- SURVIVOR
  Lake Balaton            600             12.2         YES          no         no
  Lake Neusiedl           315              1.8         YES          no         no
  Lake Maggiore           212.5          372            no         YES         no
  Lake Como               146            425            no         YES         no
  Lake Neuchâtel          218.3          152            no          no         no

  EACH SINGLE CONSTRAINT IS SATISFIED BY THREE LAKES (so the conjunction is necessary):
      area > 250 km²  : Constance, Balaton, Neusiedl
      max depth > 200 m: Constance, Maggiore, Como
  Intersection = Lake Constance alone.

  MARGINS (no borderline value; nearest miss on either side of either threshold >= 13%):
      area 250 km²  : Constance 536 (+114%), Balaton 600 (+140%), Neusiedl 315 (+26%) | misses:
                      Neuchâtel 218.3 (-13%), Maggiore 212.5 (-15%), Como 146 (-42%)
      depth 200 m   : Constance 251 (+25.5%), Maggiore 372 (+86%), Como 425 (+112%)   | misses:
                      Neuchâtel 152 (-24%), Balaton 12.2 (-94%), Neusiedl 1.8 (-99%)

  STAGE 2/3 chain (verified on the live pages):
      Lake Constance — infobox inflow = ALPINE RHINE, outflow = HIGH RHINE (two differently-named
      sections of the Rhine). The outflow section is the High Rhine (Hochrhein), Stein am Rhein to
      Basel.
      High Rhine — infobox length = 165 km  [KEYSTONE]  (basin 24,900 km²).

  DECOY VALUES the keystone must reject (all live on pages the agent will pass through):
      Rhine, whole river        1,230 km   (the obvious "length of the Rhine" grab)
      Alpine Rhine (the INFLOW)    93.5 km  (right lake, wrong section — the disambiguation trap)
      Lake Constance itself         63 km   (the lake's own length, printed next to area/depth)
  \b165\b collides with none of the twenty-odd figures reachable in this task (536 / 251 / 600 /
  12.2 / 315 / 1.8 / 212.5 / 372 / 146 / 425 / 218.3 / 152 / 63 / 395 / 48 / 93.5 / 1,230 / 24,900).

WHY IT DISCRIMINATES:
  * "largest lake" shortcut -> Balaton (600 km²) — but it is 12.2 m deep.
  * "deepest lake" shortcut -> Como (425 m) — but it is 146 km².
  * fame shortcut -> Como/Maggiore (the postcard lakes) — neither satisfies both.
  * a survivor found but the chain short-circuited -> the Rhine's 1,230 km or the Alpine Rhine's
    93.5 km, both firmly rejected by the keystone.

LEAK RESISTANCE: the keystone is the length of the HIGH RHINE — a sectional figure no consumer model
recalls parametrically (the memorable number for "the Rhine" is 1,230 km, which is a decoy here),
and it is unreachable without first electing Lake Constance from the two-way intersection and then
resolving which named section leaves the lake.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AREA_THRESHOLD = 250    # km²  — constraint (A): surface area OVER this
DEPTH_THRESHOLD = 200   # m    — constraint (B): maximum depth OVER this

# ── the AND-filter candidate set (single source of truth for statement, validators and the plan) ──
LAKES: List[Dict[str, Any]] = [
    {
        "key": "constance", "name": "Lake Constance",
        "desc": "Lake Constance (the Bodensee), on the Austria/Germany/Switzerland border",
        "area": 536, "depth": 251, "area_over": True, "depth_over": True,
        "name_rx": r"constance|bodensee", "area_rx": r"\b536\b", "depth_rx": r"\b251\b",
        "slug_rx": r"wiki/lake_constance",
    },
    {
        "key": "balaton", "name": "Lake Balaton",
        "desc": "Lake Balaton in Hungary",
        "area": 600, "depth": 12.2, "area_over": True, "depth_over": False,
        "name_rx": r"balaton", "area_rx": r"\b600\b", "depth_rx": r"\b12\.2\b",
        "slug_rx": r"wiki/lake_balaton",
    },
    {
        "key": "neusiedl", "name": "Lake Neusiedl",
        "desc": "Lake Neusiedl (Fertő) on the Austria/Hungary border",
        "area": 315, "depth": 1.8, "area_over": True, "depth_over": False,
        "name_rx": r"neusiedl|fert[őo]\b", "area_rx": r"\b315\b", "depth_rx": r"\b1\.8\b",
        "slug_rx": r"wiki/lake_neusiedl",
    },
    {
        "key": "maggiore", "name": "Lake Maggiore",
        "desc": "Lake Maggiore, on the Italy/Switzerland border",
        "area": 212.5, "depth": 372, "area_over": False, "depth_over": True,
        "name_rx": r"maggiore", "area_rx": r"\b212(?:\.5)?\b", "depth_rx": r"\b372\b",
        "slug_rx": r"wiki/lake_maggiore",
    },
    {
        "key": "como", "name": "Lake Como",
        "desc": "Lake Como in Lombardy, Italy",
        "area": 146, "depth": 425, "area_over": False, "depth_over": True,
        "name_rx": r"\bcomo\b", "area_rx": r"\b146\b", "depth_rx": r"\b425\b",
        "slug_rx": r"wiki/lake_como",
    },
    {
        "key": "neuchatel", "name": "Lake Neuchâtel",
        "desc": "Lake Neuchâtel in Switzerland",
        "area": 218.3, "depth": 152, "area_over": False, "depth_over": False,
        "name_rx": r"neuch[aâ]tel", "area_rx": r"\b218(?:\.3)?\b", "depth_rx": r"\b152\b",
        "slug_rx": r"wiki/lake_neuch",
    },
]

# Self-check: the AND-filter must have EXACTLY ONE survivor, derived from the two booleans.
_SATISFIERS = [e for e in LAKES if e["area_over"] and e["depth_over"]]
assert len(_SATISFIERS) == 1, f"AND-filter must have one survivor, got {[e['name'] for e in _SATISFIERS]}"
SURVIVOR = _SATISFIERS[0]      # Lake Constance

# ── STAGE 3 keystone: the OUTFLOW section's length (High Rhine, 165 km / 103 mi) ──
# Unit-tolerant (F26 discipline): accepts "165 km", "165 kilometres", "165\nkm" and the imperial
# "103 mi". A bare 165 without a length unit does NOT match, so an unrelated 165 cannot satisfy it.
KEYSTONE_RX = re.compile(r"\b165\s*(?:km|kilomet)|\b103\s*(?:mi\b|miles)", re.IGNORECASE)
SECTION_RX = re.compile(r"high\s*rhine|hochrhein", re.IGNORECASE)
# Decoys, for reporting only (never accepted): the whole Rhine (1,230 km) and the INFLOW section
# (Alpine Rhine, 93.5 km).
_DECOY_RX = re.compile(r"\b1[,.\s]?230\b|\b93\.5\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "147",
        "test_name": "Tier 5: AND-filter -> survivor -> disambiguated chain terminus (Alpine lakes -> outflow section length)",
        "difficulty_level": "10/10",
        "category": "Compound: two-constraint AND-filter + survivor disambiguation + chain terminus",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']} — {e['desc']}" for i, e in enumerate(LAKES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous one "
        "is resolved.\n\n"
        "STAGE 1 — two-constraint filter. Six European lakes:\n"
        f"{listing}\n"
        "Open EACH lake's page and read TWO numeric attributes from its infobox: (A) its SURFACE "
        f"AREA in km², and (B) its MAXIMUM DEPTH in metres. Then determine the SINGLE lake that "
        f"satisfies BOTH constraints at once:\n"
        f"  (A) surface area OVER {AREA_THRESHOLD} km², AND\n"
        f"  (B) maximum depth OVER {DEPTH_THRESHOLD} m.\n"
        "Exactly one of the six satisfies both. Several lakes satisfy each constraint on its own, "
        "so you must apply both together — the winner is NOT the largest lake in the list, and NOT "
        "the deepest one.\n\n"
        "STAGE 2 — disambiguate the surviving lake's OUTFLOW. On the surviving lake's page, read "
        "its OUTFLOW. It is not an independent river: it is one NAMED SECTION of a major European "
        "river, and a DIFFERENTLY-NAMED section of that same river flows INTO the lake. Identify "
        "the section that flows OUT of the lake — not the section that flows in, and not the whole "
        "river.\n\n"
        "STAGE 3 — read the keystone. Open THAT SECTION's own Wikipedia article and read ITS LENGTH "
        "in kilometres (the length of that section alone, not the length of the whole river).\n\n"
        "Report: (a) the LENGTH in km of the outflow section (this single figure is the keystone "
        "answer); (b) which lake survived the two-constraint filter and, for all six lakes, both "
        "attribute values you gathered; (c) the name of the outflow section; citing the exact "
        "Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The length in km of the surviving lake's outflow river-section (the leak-resistant keystone)",
        "Which lake satisfies BOTH constraints, plus all six lakes' area and max-depth values",
        "The name of the outflow section (not the inflow section, not the whole river)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (golden path 7: six lake pages + the outflow section's page)",
        "Gathers both attributes (surface area and max depth) for the six lakes",
        "Correctly elects Lake Constance as the unique lake with area > 250 km² AND max depth > 200 m "
        "(NOT Balaton, the largest at 600 km² but 12.2 m deep; NOT Como, the deepest at 425 m but "
        "only 146 km²)",
        "Correctly identifies the OUTFLOW section (the High Rhine), not the inflow (Alpine Rhine)",
        "Reports that section's length (165 km / 103 mi), NOT the whole Rhine's 1,230 km",
        "Cites the survivor's page and the outflow section's page",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text — prefer ``deliverables[0]`` (the keystone slot), else final_deliverable."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text — final deliverable plus every deliverable slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1): the outflow section's length (165 km / 103 mi) in the primary
    answer, AND the run is GROUNDED (visit.count > 0). Unreachable without electing the right
    survivor from the two-way intersection and then resolving the right river section: the whole
    Rhine (1,230 km) and the inflow section (93.5 km) are both rejected."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    return grounded and bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: golden path is 7 reads (six lakes + the outflow section)."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 7.0),
            "reason": f"{n} visit(s) (target 7: six lake pages + the outflow section's page; >=4 to pass)"}


def validate_keystone_section_length(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the length of the surviving lake's OUTFLOW SECTION — the High Rhine,
    165 km. A dropped constraint (Balaton/Como), a short-circuited chain (the Rhine's 1,230 km) or
    the wrong section (the inflow Alpine Rhine, 93.5 km) all fail."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_section_length", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Outflow section length 165 km (103 mi) present" if passed
                       else "Keystone outflow-section length (165 km, the High Rhine) missing/incorrect "
                            "— beware the whole Rhine (1,230 km) and the inflow section (93.5 km)")}


def validate_filter_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: for how many of the SIX lakes were BOTH filter
    attributes gathered (the lake's name AND its area figure AND its max-depth figure). This is the
    axis that separates a structured agent — which pulls twelve page-only numbers before
    intersecting — from a linear one that checks two lakes and guesses. Deliberately NOT gated on
    the keystone: breadth is credited even when the downstream chain is botched.

    The six area values (146 / 212.5 / 218.3 / 315 / 536 / 600) and the six depth values (1.8 /
    12.2 / 152 / 251 / 372 / 425) are pairwise distinct, so attribution is collision-free.
    Visit-capped (``min(hits, n_visits)``, the canonical F29 pattern) because both attributes are
    page-only: a 0-visit run that narrates them from memory banks nothing."""
    text = _all_text(result)
    hits = [e["name"] for e in LAKES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["area_rx"], text, re.IGNORECASE)
            and re.search(e["depth_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(LAKES)
    return {"check": "filter_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} lakes' two attributes gathered from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, "
                      f"{n_visits} visit(s))"}


def validate_survivor_and_section(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the two intermediate resolutions are spelled out — the AND-filter survivor
    (Lake Constance) and the OUTFLOW section (the High Rhine). Short-circuits to 0 without the
    keystone, so an incidental mention cannot bank partial credit."""
    if not _keystone_ok(result, observability):
        return {"check": "survivor_and_section", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/section resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(re.search(SURVIVOR["name_rx"], text, re.IGNORECASE))
    has_section = bool(SECTION_RX.search(text))
    hits = int(has_survivor) + int(has_section)
    return {"check": "survivor_and_section", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Lake Constance)={has_survivor}, outflow section(High Rhine)={has_section}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the pages the task had to read (the six lake pages plus the outflow
    section's page). Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [e["slug_rx"] for e in LAKES] + [r"wiki/high_rhine"]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 3, "score": min(1.0, cited / 4.0),
            "reason": f"{cited}/7 source page(s) cited (need >=3: e.g. survivor lake + the outflow section)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_section_length,
        validate_filter_coverage,
        validate_survivor_and_section,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored FILTER-THEN-CHAIN DAG scaffold for the ``graph_compiled`` variant.

    Three waves — a wide numeric-gathering wave, then a two-hop chain whose targets are unknown
    until the filter resolves (NOT a pure fan-out, NOT a pure chain):
      * WAVE 1 — six INDEPENDENT parallel leaves, one per GIVEN lake, each reading BOTH filter
        attributes (surface area, maximum depth) in a single page read and RESTATING the lake's
        name alongside both figures, so aggregation can never re-bind a stray number to the wrong
        lake (the fix 094 needed).
      * WAVE 2 — ONE dependent leaf that templates all six lakes' attribute pairs, applies the
        two-way intersection itself, and — on the surviving lake's page — resolves which NAMED
        SECTION of the river flows OUT of it (the disambiguation), explicitly warning off the
        section that flows IN.
      * WAVE 3 — ONE dependent leaf that templates the identified section and reads THAT section's
        own length.

    Leak-free: names the six GIVEN lakes and the two GIVEN thresholds, but no attribute value, not
    which lake survives, not the river, not either section's name, and not the keystone length."""
    lake_leaves = [
        {
            "id": f"lake_{e['key']}",
            "instruction": (
                f"Open the English Wikipedia page for {e['name']} — {e['desc']}. Read TWO numeric "
                "attributes directly from the infobox: (A) its SURFACE AREA in square kilometres, "
                "and (B) its MAXIMUM DEPTH in metres. Report your finding in the exact form "
                f"'{e['name']}: surface area = <number> km²; maximum depth = <number> m' followed "
                "by the exact source URL, so the lake's name stays attached to both figures. Do not "
                "report any other fact and do not guess from memory."
            ),
            "expect": f"'{e['name']}: surface area = <number> km²; maximum depth = <number> m' — source URL",
            "depends_on": [],
        }
        for e in LAKES
    ]
    filter_leaf = {
        "id": "survivor_outflow",
        "instruction": (
            "You are given the six candidate lakes with their surface area and maximum depth:\n"
            + "\n".join(f"  {e['name']} -> {{lake_{e['key']}}}" for e in LAKES)
            + "\nFIRST write out each lake's two-way check explicitly, one per line, in the form "
              f"'<lake>: area=<val> km² (>{AREA_THRESHOLD} km²? yes/no), depth=<val> m "
              f"(>{DEPTH_THRESHOLD} m? yes/no)'. THEN determine the SINGLE lake for which BOTH "
              "checks are yes. Open THAT surviving lake's Wikipedia page and read its OUTFLOW: it "
              "is a NAMED SECTION of a larger river, and a differently-named section of the same "
              "river flows INTO the lake. Report the surviving lake's name, the name of the section "
              "that flows OUT of it (NOT the one that flows in, NOT the whole river) and that "
              "section's exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "SURVIVING lake (both constraints) + the name of the river section flowing OUT of it — source URL",
        "depends_on": [f"lake_{e['key']}" for e in LAKES],
    }
    length_leaf = {
        "id": "section_length",
        "instruction": (
            "The previous step identified the river section that flows OUT of the surviving lake: "
            "{survivor_outflow}. Open THAT SECTION's own English Wikipedia article and read ITS "
            "LENGTH in kilometres from the infobox — the length of that section alone, NOT the "
            "length of the whole river it belongs to, and NOT the length of any other section. "
            "Report the section's name, its length in km, and the exact source URL. Do not guess "
            "from memory."
        ),
        "expect": "The outflow SECTION's own length in km — source URL",
        "depends_on": ["survivor_outflow"],
    }
    return {
        "leaves": lake_leaves + [filter_leaf, length_leaf],
        "aggregation": (
            "You now have (1) each of the six lakes' surface area and maximum depth, (2) which "
            "single lake satisfies BOTH constraints and the name of the river section flowing out "
            "of it, and (3) that section's own length. Write out all six lakes' two-way checks "
            "BEFORE concluding which one survives. Then report (a) the outflow section's LENGTH in "
            "km — this single figure is the keystone answer; (b) which lake satisfied both "
            "constraints and all six lakes' two attribute values; (c) the name of the outflow "
            "section; citing every source URL. Do not report the length of the whole river."
        ),
    }
