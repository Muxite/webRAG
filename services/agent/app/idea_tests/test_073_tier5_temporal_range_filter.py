"""
Test 073: Tier 5 (integration) — TEMPORAL RANGE FILTER (founding-year in-range subset).
Level: integration   Weight: long   Difficulty: 9/10

A temporal-reasoning and set-filtering task that exposes two cheap-model failure modes at once:
(1) hallucinating founding dates for obscure institutions it cannot recall from memory, and (2)
mis-binding a year to the wrong entity when years are retrieved ad hoc rather than restated before
the range filter is applied. The agent is given SIX solid-but-obscure institutions (a museum, a
zoo, a university, an academy, a theatre, and a history museum) from Kosovo, North Macedonia, and
Albania — none famous enough for their specific founding year to be recalled from parametric memory.
For each, the agent must look up the Wikipedia article and read the infobox "Established" / "Opened"
/ "Founded" year. It must then identify WHICH of the six were founded within the year range
[1940, 1963] (inclusive), and state how many.

The design engineers a clean 3-of-6 in-range split with all margins ≥ 6 years from each boundary:

  institution                              year   in [1940, 1963]?   min margin from boundary
  ───────────────────────────────────────────────────────────────────────────────────────────
  National Theatre of Kosovo               1946   YES (IN)           6 yrs from A, 17 from B
  Kosovo Museum                            1949   YES (IN)           9 yrs from A, 14 from B
  Luigj Gurakuqi University of Shkodra     1957   YES (IN)          17 yrs from A,  6 from B
  Skopje Zoo                               1926   NO  (below range) 14 yrs below A
  Academy of Sciences of Albania           1972   NO  (above range)  9 yrs above B
  National History Museum (Albania)        1981   NO  (above range) 18 yrs above B
  ───────────────────────────────────────────────────────────────────────────────────────────
  KEYSTONE = the exact in-range SET: {National Theatre of Kosovo, Kosovo Museum,
             Luigj Gurakuqi University of Shkodra}   COUNT = 3

Every year is separated from the range boundaries by at least 6 years: a ±5-year misread on any
single founding year cannot flip that entity's in-range/out-of-range membership. All six years are
distinct, so coverage is collision-free.

ANTI-PARAMETRIC: none of these six founding years is a commonly memorised fact:
  * National Theatre of Kosovo (1946): Kosovo-specific cultural institution, founded in Prizren
    during Yugoslav-administered Kosovo; its opening year is not in parametric training as a
    well-known fact.
  * Kosovo Museum (1949): Yugoslav-era provincial museum in Pristina; the 1949 founding year is
    highly unlikely to be recalled by a model without browsing.
  * Luigj Gurakuqi University of Shkodra (1957): Albanian communist-era public university named
    after the nationalist Luigj Gurakuqi; models may know the namesake person but not the
    institution's 1957 founding date.
  * Skopje Zoo (1926): interwar municipal zoo in Skopje; 1926 is not a recallable fact.
  * Academy of Sciences of Albania (1972): Albanian Hoxha-era institution; a model might guess
    "1960s–1970s" but cannot recall 1972 specifically.
  * National History Museum (Albania) (1981): opened under the Hoxha regime on 28 October 1981;
    its exact year is not commonly memorised.

GROUND TRUTH (verified against live English Wikipedia, 2026-06-27 — each infobox
"Established" / "Opened" / "Founded" field on each institution's own Wikipedia article):
  National Theatre of Kosovo     1946  en.wikipedia.org/wiki/National_Theatre_of_Kosovo
  Kosovo Museum                  1949  en.wikipedia.org/wiki/Kosovo_Museum
  Luigj Gurakuqi Univ. Shkodra   1957  en.wikipedia.org/wiki/Luigj_Gurakuqi_University_of_Shkod%C3%ABr
  Skopje Zoo                     1926  en.wikipedia.org/wiki/Skopje_Zoo
  Academy of Sciences Albania    1972  en.wikipedia.org/wiki/Academy_of_Sciences_of_Albania
  Nat. History Museum Albania    1981  en.wikipedia.org/wiki/National_History_Museum_(Albania)

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (parametric / graph): hallucinates one or more founding years → wrong in-range
    set. Typical errors: guessing Academy of Sciences founded ~1950 (a plausible communist-era
    date) drags an out-of-range entity into the range; guessing the National Theatre founded ~1975
    (post-range) pushes an in-range entity out. Even a single wrong year scrambles membership.
  * frontier sequential (ReAct): reads all six Wikipedia infoboxes, applies the range filter,
    names the correct three → decent.
  * graph_compiled: six independent parallel leaves each read ONE institution's founding year from
    its own article (one atomic fact per leaf, routing around the risk of hallucination); the
    aggregation is forced to RESTATE each entity → year explicitly before applying [1940, 1963],
    so the cheap executor cannot mis-bind a year to the wrong entity → the cheap executor is
    rescued from both hallucinated dates and year-entity mis-binding.

KEYSTONE = the exact in-range SUBSET (all three in-range names present in deliverables[0] and no
  out-of-range name present there). The keystone check is robust: it requires the correct complete
  set — a model that lists only two in-range entities, or includes an out-of-range entity, fails.
  Secondary (gated) = COUNT = 3 AND each in-range entity's founding year (1946, 1949, 1957).
  Coverage (ungated, collision-free) = all 6 founding years gathered.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified fixtures ─────────────────────────────────────────────────────────
# ``year`` = Wikipedia infobox "Established" / "Opened" / "Founded" value, live-verified.
# ``in_range`` = (RANGE_A <= year <= RANGE_B), derived here for use by validators and plan.
# Nothing — no year, no count, no membership — is leaked into the task statement or the plan.

RANGE_A: int = 1940   # range lower bound (inclusive)
RANGE_B: int = 1963   # range upper bound (inclusive)

ENTITIES: List[Dict[str, Any]] = [
    {
        "key": "theatre_kosovo",
        "name": "National Theatre of Kosovo",
        "year": 1946,
        "in_range": True,
        # Wikipedia slug: National_Theatre_of_Kosovo
        "slug_rx": r"national_theatre_of_kosovo",
        # matches "National Theatre of Kosovo", "Kosovo national theatre", "Kosovo Theatre", etc.
        "name_rx": (
            r"\b(?:national\s+)?(?:theatre|theater)\s+of\s+kosovo\b"
            r"|\bkosovo\s+(?:national\s+)?(?:theatre|theater)\b"
        ),
    },
    {
        "key": "kosovo_museum",
        "name": "Kosovo Museum",
        "year": 1949,
        "in_range": True,
        # Wikipedia slug: Kosovo_Museum
        "slug_rx": r"wiki/kosovo_museum",
        "name_rx": r"\bkosovo\s+museum\b",
    },
    {
        "key": "univ_shkodra",
        "name": "Luigj Gurakuqi University of Shkodra",
        "year": 1957,
        "in_range": True,
        # Wikipedia slug: Luigj_Gurakuqi_University_of_Shkodër (ë = %C3%AB in URL)
        "slug_rx": r"luigj_gurakuqi|university_of_shkod",
        # matches via distinctive personal name or city name prefix (handles ë / e / a variants)
        "name_rx": (
            r"\b(?:luigj|gurakuqi)\b"
            r"|\buniversity\s+of\s+shkod"       # prefix match: Shkodër / Shkodra / Shkoder
            r"|\bshkod[rë][ae]?\s+university\b"
        ),
    },
    {
        "key": "skopje_zoo",
        "name": "Skopje Zoo",
        "year": 1926,
        "in_range": False,
        # Wikipedia slug: Skopje_Zoo
        "slug_rx": r"wiki/skopje_zoo",
        "name_rx": r"\bskopje\s+zoo\b|\bzoo\s+(?:of\s+)?skopje\b",
    },
    {
        "key": "acad_albania",
        "name": "Academy of Sciences of Albania",
        "year": 1972,
        "in_range": False,
        # Wikipedia slug: Academy_of_Sciences_of_Albania
        "slug_rx": r"academy_of_sciences_of_albania",
        # requires Albania to be named (avoids matching generic "Academy of Sciences" phrases)
        "name_rx": (
            r"\bacademy\s+of\s+sciences\s+of\s+albania\b"
            r"|\balbanians?\s+academy\s+of\s+sciences\b"
        ),
    },
    {
        "key": "nat_hist_albania",
        "name": "National History Museum (Albania)",
        "year": 1981,
        "in_range": False,
        # Wikipedia slug: National_History_Museum_(Albania)
        "slug_rx": r"national_history_museum.*albania|national_history_museum_\(albania\)",
        # requires Albania nearby to avoid matching unrelated "national history museum" mentions
        "name_rx": (
            r"\bnational\s+history\s+museum\s*\(?albania\)?"
            r"|\bnational\s+history\s+museum\b[^.;]{0,60}\balban"
            r"|\balban[^.;]{0,60}\bnational\s+history\s+museum\b"
        ),
    },
]

IN_RANGE: List[Dict[str, Any]] = [e for e in ENTITIES if e["in_range"]]
OUT_RANGE: List[Dict[str, Any]] = [e for e in ENTITIES if not e["in_range"]]
IN_RANGE_COUNT: int = len(IN_RANGE)   # 3

# Numeric-token extractor: grabs each maximal digit run; drops decimals; strips grouping commas.
# Same pattern as test_070 — isolates plain integers, so years ("1946") and counts ("3") are
# cleanly extracted without decimal confusion.
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


# ── public API ────────────────────────────────────────────────────────────────

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "073",
        "test_name": "Tier 5: Temporal range filter (founding-year in-range subset identification)",
        "difficulty_level": "9/10",
        "category": "Temporal reasoning + set filtering (founding-year subset identification)",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find each institution's Wikipedia article and READ it "
        "(do not guess or recall from memory). The six institutions below are located in Kosovo, "
        "North Macedonia, and Albania:\n"
        f"{listing}\n\n"
        "For EACH of the six institutions, open its Wikipedia article and read, from the infobox, "
        "the year it was founded, established, or officially opened (whichever field is present). "
        "You MUST read the article — do not estimate or guess the year from memory.\n\n"
        f"Then identify WHICH of the six institutions were founded or established within the year "
        f"range {RANGE_A} to {RANGE_B} (inclusive), and state how many that is.\n\n"
        "Report (a) the names of the institutions whose founding year falls within "
        f"[{RANGE_A}, {RANGE_B}] — list only those names in your primary answer (the keystone); "
        "(b) the count of in-range institutions; "
        "(c) the founding year of each in-range institution; "
        "(d) the founding year of ALL SIX institutions; and "
        "(e) the source URL of every Wikipedia article you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        f"The names of the institutions founded within [{RANGE_A}, {RANGE_B}] "
        f"(list only the in-range institution names — the keystone answer)",
        f"The count of in-range institutions",
        "The founding year of each in-range institution",
        "The founding year of all six institutions",
        "Source URL for each institution's Wikipedia article",
    ]


def get_success_criteria() -> List[str]:
    in_names = ", ".join(e["name"] for e in IN_RANGE)
    in_years = "; ".join(f"{e['name']} → {e['year']}" for e in IN_RANGE)
    return [
        "At least 5 pages visited (target 6: one Wikipedia article per institution)",
        f"Correctly identifies the in-range set: {{{in_names}}} — the three institutions founded "
        f"within [{RANGE_A}, {RANGE_B}] — and places only those names in the primary answer slot",
        f"States the count = {IN_RANGE_COUNT}",
        f"Reports each in-range founding year: {in_years}",
        "Gathers all six founding years",
        "Cites each institution's source URL",
    ]


# ── internal helpers ──────────────────────────────────────────────────────────

def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text. Prefer ``deliverables[0]`` (the in-range subset / keystone slot)
    when present; otherwise fall back to ``output.final_deliverable``."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: the final deliverable plus every deliverable slot, so year / citation
    checks can see content placed outside the primary answer slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _int_values(text: str) -> List[int]:
    """Plain integer values present in ``text``: each numeric token with decimals dropped and
    grouping commas stripped (see ``_NUM_TOKEN_RX``)."""
    vals: List[int] = []
    for tok in _NUM_TOKEN_RX.findall(text):
        if "." in tok:
            continue   # skip decimals
        try:
            vals.append(int(tok.replace(",", "")))
        except ValueError:
            continue
    return vals


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: ``deliverables[0]`` names exactly the three in-range institutions and no
    out-of-range institution.

    Pass conditions (both required):
      (1) ALL three in-range entity name patterns are found in primary_text.
      (2) NONE of the three out-of-range entity name patterns are found in primary_text.

    The task instructs the agent to place only the in-range institution names in deliverables[0]
    (the keystone slot), so a correct terse listing of the three names passes cleanly, while any
    answer that omits an in-range entity or includes an out-of-range entity fails.
    """
    text = _primary_text(result)
    if not all(re.search(e["name_rx"], text, re.IGNORECASE) for e in IN_RANGE):
        return False   # at least one in-range entity missing from primary answer
    if any(re.search(e["name_rx"], text, re.IGNORECASE) for e in OUT_RANGE):
        return False   # an out-of-range entity appears in the keystone slot → wrong set
    return True


# ── validation functions ──────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per institution."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 5,
        "score": min(1.0, n / 6.0),
        "reason": f"{n} visit(s) (target >=6: one Wikipedia article per institution; >=5 to pass)",
    }


def validate_keystone_inrange_set(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact in-range SUBSET — the three institutions whose founding year
    falls within [1940, 1963] — named in deliverables[0], with no out-of-range entity present.

    A model that hallucinates a founding year for any institution may pull an out-of-range entity
    inside the set (e.g. 'Academy of Sciences of Albania founded ~1950') or push an in-range entity
    outside it (e.g. 'National Theatre founded ~1975'), failing the set-match. Even one wrong year
    scrambles membership.
    """
    passed = _keystone_ok(result)
    in_names = ", ".join(e["name"] for e in IN_RANGE)
    out_names = ", ".join(e["name"] for e in OUT_RANGE)
    return {
        "check": "keystone_inrange_set",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            f"In-range set {{{in_names}}} correctly identified in deliverables[0]" if passed
            else f"In-range set missing or incorrect in deliverables[0]. "
                 f"Correct set (3 institutions): {{{in_names}}}. "
                 f"Must NOT appear in deliverables[0]: {out_names}. "
                 f"Reading all six Wikipedia infoboxes and applying [{RANGE_A}, {RANGE_B}] is required."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: how many of the six founding years were gathered. All six years
    (1926, 1946, 1949, 1957, 1972, 1981) are DISTINCT, so coverage is collision-free — no year
    can be credited to more than one institution. Deliberately NOT gated on the keystone: it measures
    whether the agent fanned out to all six articles even when it botches the range filter, the axis
    that separates a structured multi-leaf agent from one that drops an entity."""
    present = set(_int_values(_all_text(result)))
    hits = [e["name"] for e in ENTITIES if e["year"] in present]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} founding years gathered "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_inrange_years(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all three in-range founding years (1946, 1949, 1957) explicitly reported.
    Short-circuits to 0 when the keystone is absent, so a wrong/guessed in-range set cannot bank
    credit for the per-institution founding years."""
    if not _keystone_ok(result):
        return {
            "check": "inrange_years",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → in-range founding years not credited",
        }
    present = set(_int_values(_all_text(result)))
    hits = [e for e in IN_RANGE if e["year"] in present]
    n = len(IN_RANGE)
    summary = ", ".join(f"{e['name']} → {e['year']}" for e in hits) if hits else "none"
    return {
        "check": "inrange_years",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} in-range founding years present ({summary})",
    }


def validate_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the count of in-range institutions (= 3) is stated. Short-circuits to 0
    when the keystone is absent."""
    if not _keystone_ok(result):
        return {
            "check": "count",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → count not credited",
        }
    text = _all_text(result)
    # Accept the digit "3" or the word "three"
    ok = (IN_RANGE_COUNT in _int_values(text)) or bool(
        re.search(r"\bthree\b", text, re.IGNORECASE)
    )
    return {
        "check": "count",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"Count = {IN_RANGE_COUNT} stated" if ok
            else f"Count = {IN_RANGE_COUNT} not found in answer"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: source URLs for the six institution pages cited. Short-circuits to 0 when
    the keystone is absent."""
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
        "passed": cited >= 4,
        "score": cited / n,
        "reason": f"{cited}/{n} institution pages cited (≥4 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_inrange_set,
        validate_coverage,
        validate_inrange_years,
        validate_count,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None → the harness applies its default structured rubric judge (gpt-5-mini), as in 055/059.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored six-way fan-out / filter / aggregate scaffold for the ``graph_compiled``
    variant.

    SIX INDEPENDENT parallel leaves: one per GIVEN institution, each fetching that institution's
    founding year from its OWN Wikipedia article — one atomic integer per leaf. The leaves carry
    NO year values, NO range bounds, and NO membership information; they encode structure only.

    The aggregation owns ALL filtering logic and is forced to execute in three explicit steps:
      (1) RESTATE each entity → year (one line per institution) before any filtering — this
          prevents the cheap executor from mis-binding a year retrieved for one institution to a
          different institution's name;
      (2) APPLY the range filter [{RANGE_A}, {RANGE_B}] explicitly for each institution, writing
          'IN RANGE' or 'OUT OF RANGE' for each one before drawing any conclusion;
      (3) COLLECT the in-range names, state the count, and report all required deliverables.

    Nothing about which institution is in-range, or any founding year, is embedded in this plan.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_year",
            "instruction": (
                f"Open the Wikipedia article for '{e['name']}' and read, from the infobox, "
                f"the year it was founded, established, or officially opened (use whichever "
                f"field — 'Founded', 'Established', or 'Opened' — is present in the infobox). "
                f"Report ONLY that single integer year and the source URL. "
                f"Do not guess or recall from memory."
            ),
            "expect": f"FOUNDING YEAR of {e['name']} (a single 4-digit integer) -- source URL",
            "depends_on": [],
        })

    return {
        "leaves": leaves,
        "aggregation": (
            f"You now have the Wikipedia-sourced founding year for each of the six institutions. "
            f"Complete the following three steps IN ORDER before reporting your final answer.\n\n"

            f"STEP 1 — RESTATE each institution's founding year explicitly, one line per "
            f"institution, in the form:\n"
            f"  <institution name>: founded <year>\n"
            f"List all six institutions before proceeding to step 2. Do not skip any.\n\n"

            f"STEP 2 — APPLY the range filter [{RANGE_A}, {RANGE_B}] (inclusive) to each "
            f"institution. For each one write:\n"
            f"  <institution name>: <year> — IN RANGE  (if {RANGE_A} ≤ <year> ≤ {RANGE_B})\n"
            f"  <institution name>: <year> — OUT OF RANGE  (otherwise)\n"
            f"Complete this for ALL SIX institutions before drawing any conclusion.\n\n"

            f"STEP 3 — COLLECT and REPORT:\n"
            f"  (a) The names of the in-range institutions only "
            f"(place these in deliverables[0] — the keystone answer);\n"
            f"  (b) the count of in-range institutions;\n"
            f"  (c) each in-range institution's founding year;\n"
            f"  (d) the founding year of all six institutions;\n"
            f"  (e) each institution's source URL."
        ),
    }
