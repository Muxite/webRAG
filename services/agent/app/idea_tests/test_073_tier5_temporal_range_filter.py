"""
Test 073: Tier 5 (integration) — TEMPORAL RANGE FILTER (count of founding-years in range).
Level: integration   Weight: long   Difficulty: 9/10

A temporal-reasoning and set-filtering task that exposes two cheap-model failure modes at once:
(1) hallucinating founding dates for obscure institutions it cannot recall from memory, and (2)
mis-binding a year to the wrong entity when years are retrieved ad hoc rather than restated before
the range filter is applied. The agent is given SIX solid-but-obscure institutions (a museum, a
zoo, a university, an academy, a theatre, and a history museum) from Kosovo, North Macedonia, and
Albania — none famous enough for their specific founding year to be recalled from parametric memory.
For each, the agent must look up the Wikipedia article and read the infobox "Established" / "Opened"
/ "Founded" year. It must then COUNT how many of the six were founded within the year range
[1940, 1963] (inclusive).

The keystone is a COUNT (an integer 0–6) rather than the exact named subset: the count-with-condition
keystone shape is the one that is reliably scorable for a cheap compiled-leaf executor (matching the
validated pattern of tests 072/078/082/087/089), whereas naming the EXACT in-range subset exceeds a
cheap executor's reliability. The in-range names remain an un-gated / gated diagnostic below, but the
0/1 gate is the computed count.

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
  KEYSTONE = the COUNT of in-range institutions = 3 (a single integer 0–6 in deliverables[0]).
  The three in-range institutions are {National Theatre of Kosovo, Kosovo Museum,
             Luigj Gurakuqi University of Shkodra}.

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

GROUND TRUTH (verified against live English Wikipedia, 2026-07-07 — each infobox
"Established" / "Opened" / "Founded" field on each institution's own Wikipedia article):
  National Theatre of Kosovo     1946  en.wikipedia.org/wiki/National_Theatre_of_Kosovo
  Kosovo Museum                  1949  en.wikipedia.org/wiki/Kosovo_Museum
  Luigj Gurakuqi Univ. Shkodra   1957  en.wikipedia.org/wiki/Luigj_Gurakuqi_University_of_Shkod%C3%ABr
  Skopje Zoo                     1926  en.wikipedia.org/wiki/Skopje_Zoo
  Academy of Sciences Albania    1972  en.wikipedia.org/wiki/Academy_of_Sciences_of_Albania
  Nat. History Museum Albania    1981  en.wikipedia.org/wiki/National_History_Museum_(Albania)

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (parametric / graph): hallucinates one or more founding years → wrong count.
    Typical errors: guessing Academy of Sciences founded ~1950 (a plausible communist-era date)
    drags an out-of-range entity into the range and pushes the count to 4; guessing the National
    Theatre founded ~1975 (post-range) pushes an in-range entity out and drops the count to 2.
    Even a single wrong year moves the count off 3.
  * frontier sequential (ReAct): reads all six Wikipedia infoboxes, applies the range filter,
    counts the correct three → decent.
  * graph_compiled: six independent parallel leaves each read ONE institution's founding year from
    its own article (one atomic fact per leaf, routing around the risk of hallucination); the
    aggregation is forced to RESTATE each entity → year explicitly before applying [1940, 1963],
    so the cheap executor cannot mis-bind a year to the wrong entity → the cheap executor is
    rescued from both hallucinated dates and year-entity mis-binding when computing the count.

KEYSTONE = the COUNT of in-range institutions = 3 (a single integer 0–6 in deliverables[0]). The
  count keystone is the reliably-scorable shape for a cheap compiled-leaf executor: a model that
  hallucinates a founding year, drops an entity, or mis-binds a year lands on a count ≠ 3 and fails.
  Secondary (gated) = the three in-range NAMES + each in-range founding year (1946, 1949, 1957).
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
        "test_name": "Tier 5: Temporal range filter (count of founding-years within date range)",
        "difficulty_level": "9/10",
        "category": "Temporal reasoning + count-with-condition (founding-years in date range)",
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
        f"Then COUNT how many of the six institutions were founded or established within the year "
        f"range {RANGE_A} to {RANGE_B} (inclusive).\n\n"
        "Report (a) the COUNT — how many of the six institutions have a founding year within "
        f"[{RANGE_A}, {RANGE_B}] (a single integer 0–6; this is the keystone answer and belongs "
        "in your primary answer); "
        "(b) the names of the in-range institutions; "
        "(c) the founding year of each in-range institution; "
        "(d) the founding year of ALL SIX institutions; and "
        "(e) the source URL of every Wikipedia article you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        f"The count of institutions founded within [{RANGE_A}, {RANGE_B}] "
        f"(the keystone integer, 0–6)",
        f"The names of the in-range institutions",
        "The founding year of each in-range institution",
        "The founding year of all six institutions",
        "Source URL for each institution's Wikipedia article",
    ]


def get_success_criteria() -> List[str]:
    in_names = ", ".join(e["name"] for e in IN_RANGE)
    in_years = "; ".join(f"{e['name']} → {e['year']}" for e in IN_RANGE)
    return [
        "At least 5 pages visited (target 6: one Wikipedia article per institution)",
        f"Correctly reports the count = {IN_RANGE_COUNT} (the number of institutions founded "
        f"within [{RANGE_A}, {RANGE_B}]) as the primary answer",
        f"Identifies the {IN_RANGE_COUNT} in-range institutions: {in_names}",
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


_LIST_MARKER_RX = re.compile(r"(?m)^\s*\(?(\d{1,2})[.)]\s+")


def _strip_list_markers(text: str) -> str:
    """Drop a leading enumeration marker ('1. ', '2) ', '(3) ') from the start of each line
    before counting asserted values -- but ONLY when the text actually looks like a numbered
    list (>=2 such markers present). A single leading digit-marker on an otherwise terse answer
    (e.g. '4.' or '4. Lakes exceed the threshold') is far more likely a genuine short answer than
    list enumeration, and must NOT be stripped -- confirmed via adversarial review that an
    earlier, unconditional version of this fix broke exactly that case."""
    if len(_LIST_MARKER_RX.findall(text)) < 2:
        return text
    return _LIST_MARKER_RX.sub("", text)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: ``deliverables[0]`` contains the integer ``IN_RANGE_COUNT`` (= 3).

    The primary answer slot must hold the computed count of institutions founded within
    [1940, 1963]. A model that hallucinates a founding year, drops an entity, or mis-binds a
    year lands on a count ≠ 3 (typically 2 or 4) and fails. Only the correct value 3 passes.

    Also requires GROUNDING: the value string alone is insufficient — the agent must have
    actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess would earn credit.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return IN_RANGE_COUNT in _int_values(_strip_list_markers(_primary_text(result)))


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


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact COUNT of institutions founded within [1940, 1963], reported
    as a single integer in deliverables[0].

    Correct answer = 3. Any other value (2, 4, 6, …) fails. A model that hallucinates a founding
    year for any institution may pull an out-of-range entity into the range (e.g. 'Academy of
    Sciences of Albania founded ~1950' → count 4) or push an in-range entity out (e.g. 'National
    Theatre founded ~1975' → count 2). Even one wrong year moves the count off 3. Only a model that
    reads all six infoboxes, applies the range, and counts correctly passes.
    """
    passed = _keystone_ok(result, observability)
    in_names = ", ".join(e["name"] for e in IN_RANGE)
    return {
        "check": "keystone_count",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            f"Count {IN_RANGE_COUNT} (institutions founded within [{RANGE_A}, {RANGE_B}]) present "
            f"in primary answer" if passed
            else f"Count {IN_RANGE_COUNT} absent or wrong in primary answer. The {IN_RANGE_COUNT} "
                 f"in-range institutions are: {in_names}. Beware: dragging an out-of-range entity "
                 f"in (e.g. via a hallucinated year) gives 4; dropping an in-range entity gives 2. "
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


def validate_inrange_names(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all three in-range institution NAMES present in the answer.

    Short-circuits to 0 when the keystone (count) is absent, so a run that reports the wrong count
    cannot bank partial credit for naming some of the in-range institutions. After the keystone
    gate, checks that all three in-range names appear anywhere in the reported text."""
    if not _keystone_ok(result, observability):
        return {
            "check": "inrange_names",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → in-range name list not credited",
        }
    text = _all_text(result)
    hits = [e["name"] for e in IN_RANGE if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(IN_RANGE)
    return {
        "check": "inrange_names",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} in-range institution names identified "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_inrange_years(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all three in-range founding years (1946, 1949, 1957) explicitly reported.
    Short-circuits to 0 when the keystone is absent, so a wrong count cannot bank credit for the
    per-institution founding years."""
    if not _keystone_ok(result, observability):
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


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: source URLs for the six institution pages cited. Short-circuits to 0 when
    the keystone is absent."""
    if not _keystone_ok(result, observability):
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
        validate_keystone_count,
        validate_coverage,
        validate_inrange_names,
        validate_inrange_years,
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
      (3) COUNT the IN-RANGE institutions (that integer is the keystone answer), then report the
          in-range names, years, and all required deliverables.

    Nothing about which institution is in-range, the count, or any founding year is embedded in
    this plan.
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

            f"STEP 3 — COUNT and REPORT:\n"
            f"  (a) COUNT how many institutions you marked IN RANGE and report that single integer "
            f"(0–6) as the primary keystone answer (place it in deliverables[0]);\n"
            f"  (b) the names of the in-range institutions;\n"
            f"  (c) each in-range institution's founding year;\n"
            f"  (d) the founding year of all six institutions;\n"
            f"  (e) each institution's source URL."
        ),
    }
