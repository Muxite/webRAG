"""
Test 078: Tier 5 (graph) — COUNT-WITH-CONDITION over page-only island areas.
Level: graph   Weight: long   Difficulty: 8/10

Among seven solid-but-obscure islands drawn from four different countries, the agent must open each
island's Wikipedia page, read its AREA in km² from the infobox, and report HOW MANY of the seven
have an area strictly greater than 8,000 km². The answer (a COUNT, integer 0–7) is COMPUTED from
seven independent page-only lookups and is not directly recallable.

WHY IT DISCRIMINATES (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap parametric (no tools): cannot know seven obscure island areas simultaneously; either
    guesses the wrong count, claims "all seven" (the naïve overcount), or refuses outright.
  * cheap native (graph): may open some pages but drops one or more islands, misattributes areas,
    or applies the threshold check inconsistently; count likely off by 1–2.
  * graph_compiled: seven parallel leaves each fetch ONE island's area from ONE infobox; the
    aggregation is forced to restate each entity→value mapping before comparing to the threshold
    and summing the boolean outcomes; the cheap executor is rescued from holding seven values
    simultaneously and a diverse-grounding reranker can catch any threshold misapplication.

Ground truth (verified against live English Wikipedia, 2026-06-27 — each island's infobox
'Area' field, read from the canonical article URL):

  island             Wikipedia slug             country          area (km²)   > 8000?
  ──────────────────────────────────────────────────────────────────────────────────────
  Flores             /wiki/Flores               Indonesia         14,250       YES
  Samar              /wiki/Samar                Philippines       13,428.8     YES
  Bangka Island      /wiki/Bangka_Island        Indonesia         11,831.02    YES
  Kodiak Island      /wiki/Kodiak_Island        Alaska, USA        9,311.2     YES
  Leyte              /wiki/Leyte                Philippines        7,367.6     NO
  Melville Island    /wiki/Melville_Island_(Australia) Australia  5,786        NO
  Chichagof Island   /wiki/Chichagof_Island     Alaska, USA        5,305.9     NO
  ──────────────────────────────────────────────────────────────────────────────────────
  KEYSTONE = count of islands with area > 8,000 km² = 4

THRESHOLD MARGINS (each area clears or misses 8,000 km² by ≥ 630 km² — no borderline value):
  PASS  Flores            14,250 − 8,000 = +6,250 km²  (largest above-threshold margin)
  PASS  Samar             13,428 − 8,000 = +5,428 km²
  PASS  Bangka Island     11,831 − 8,000 = +3,831 km²
  PASS  Kodiak Island      9,311 − 8,000 = +1,311 km²  (smallest above-threshold margin)
  FAIL  Leyte              7,367 − 8,000 =   −633 km²  (smallest below-threshold margin)
  FAIL  Melville Island    5,786 − 8,000 = −2,214 km²
  FAIL  Chichagof Island   5,305 − 8,000 = −2,695 km²  (largest below-threshold margin)

  Every area is ≥ 630 km² from the 8,000 km² threshold, so no plausible single-figure extraction
  error (a misread of a 4–5 digit infobox value) can flip any island's pass/fail status.

ANTI-PARAMETRIC: the seven islands span four countries and are solid but non-iconic:
  • Flores (Indonesia) is associated with Komodo dragon proximity and Homo floresiensis fossils,
    but its exact area (14,250 km²) is a page-only figure not commonly cited.
  • Samar (Philippines) is the third-largest Philippine island but rarely features in common
    knowledge; its area (13,428.8 km²) is not simultaneously recallable alongside the others.
  • Bangka Island (Indonesia) is a tin-producing island off Sumatra; its area (11,831 km²)
    is not commonly known.
  • Kodiak Island (Alaska, USA) is known for Kodiak bears, not for geography; its exact area
    (9,311.2 km²) is page-specific.
  • Leyte (Philippines) is known for the Battle of Leyte Gulf (WWII history), not for its area
    (7,367.6 km²).
  • Melville Island (Australia, Northern Territory) is a remote island north of Darwin; its
    area (5,786 km²) is entirely page-specific.
  • Chichagof Island (Alaska, USA) is a densely forested island in the Alexander Archipelago;
    its area (5,305.9 km²) is not commonly recalled.
  The COUNT (4) is not a published fact anywhere. A cheap parametric model has no reliable path
  to the correct count without opening all seven pages.

SELF-VERIFY (throwaway snippet — delete after):
    from services.agent.app.idea_tests.test_078_tier5_count_with_condition_b import (
        validate_keystone_count, ENTITIES, KEYSTONE_COUNT
    )
    # CORRECT result -> keystone PASSES
    correct = {"deliverables": [str(KEYSTONE_COUNT)], "output": {"final_deliverable": "4"}}
    assert validate_keystone_count(correct, {})["passed"], "CORRECT should pass"
    # DECOY off by one -> keystone FAILS
    decoy_3 = {"deliverables": ["3"], "output": {"final_deliverable": "3 islands"}}
    assert not validate_keystone_count(decoy_3, {})["passed"], "count=3 should fail"
    decoy_5 = {"deliverables": ["5"], "output": {"final_deliverable": "5 islands"}}
    assert not validate_keystone_count(decoy_5, {})["passed"], "count=5 should fail"
    # DECOY count=7 (naive 'all') -> keystone FAILS
    decoy_7 = {"deliverables": ["7"], "output": {"final_deliverable": "all 7"}}
    assert not validate_keystone_count(decoy_7, {})["passed"], "count=7 should fail"
    print("All self-verify assertions passed.")

KEYSTONE = the count 4 (exact integer in deliverables[0]).
Secondary (gated) = lists all four passing island names (Flores, Samar, Bangka Island, Kodiak Island).
Coverage (ungated) = how many of the seven (island, area-value) pairs gathered; the seven rounded
  areas (14250 / 13429 / 11831 / 9311 / 7368 / 5786 / 5306) are all distinct — coverage is
  collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified fixtures ────────────────────────────────────────────────────────────────────────────
# 'area' = area in km² from the Wikipedia infobox 'Area' field (live-verified 2026-06-27).
# 'passes' = (area > THRESHOLD).  Nothing below is leaked into the task statement or the plan.

THRESHOLD: int = 8_000  # km², strictly greater-than

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":     "bangka",
        "name":    "Bangka Island",
        "country": "Indonesia",
        "area":    11831,
        "passes":  True,
        # Wikipedia infobox shows 11,831.02 km²; regex matches 11,831 or nearby rounding.
        "name_rx": r"bangka",
        "area_rx": r"(?<!\d)11[,\s]?83[0-9](?!\d)",
        "slug_rx": r"wiki/bangka_island|wiki/bangka",
    },
    {
        "key":     "chichagof",
        "name":    "Chichagof Island",
        "country": "Alaska, USA",
        "area":    5306,
        "passes":  False,
        # Wikipedia infobox shows 5,305.9 km²; regex matches 5,305 or 5,306.
        "name_rx": r"chichagof",
        "area_rx": r"(?<!\d)5[,\s]?30[56](?!\d)",
        "slug_rx": r"wiki/chichagof_island",
    },
    {
        "key":     "flores",
        "name":    "Flores",
        "country": "Indonesia",
        "area":    14250,
        "passes":  True,
        # Wikipedia infobox shows 14,250 km² (exact integer).
        "name_rx": r"\bflores\b",
        "area_rx": r"(?<!\d)14[,\s]?250(?!\d)",
        "slug_rx": r"wiki/flores",
    },
    {
        "key":     "kodiak",
        "name":    "Kodiak Island",
        "country": "Alaska, USA",
        "area":    9311,
        "passes":  True,
        # Wikipedia infobox shows 9,311.2 km²; regex matches 9,311.
        "name_rx": r"kodiak",
        "area_rx": r"(?<!\d)9[,\s]?31[0-9](?!\d)",
        "slug_rx": r"wiki/kodiak_island",
    },
    {
        "key":     "leyte",
        "name":    "Leyte",
        "country": "Philippines",
        "area":    7368,
        "passes":  False,
        # Wikipedia infobox shows 7,367.6 km²; regex matches 7,367 or 7,368.
        "name_rx": r"\bleyte\b",
        "area_rx": r"(?<!\d)7[,\s]?36[678](?!\d)",
        "slug_rx": r"wiki/leyte",
    },
    {
        "key":     "melville",
        "name":    "Melville Island",
        "country": "Australia",
        "area":    5786,
        "passes":  False,
        # Wikipedia infobox shows 5,786 km² (exact integer).
        "name_rx": r"melville",
        "area_rx": r"(?<!\d)5[,\s]?786(?!\d)",
        "slug_rx": r"wiki/melville_island",
    },
    {
        "key":     "samar",
        "name":    "Samar",
        "country": "Philippines",
        "area":    13429,
        "passes":  True,
        # Wikipedia infobox shows 13,428.8 km²; regex matches 13,428 or 13,429.
        "name_rx": r"\bsamar\b",
        "area_rx": r"(?<!\d)13[,\s]?42[89](?!\d)",
        "slug_rx": r"wiki/samar",
    },
]

PASSING: List[Dict[str, Any]] = [e for e in ENTITIES if e["passes"]]    # 4 islands
FAILING: List[Dict[str, Any]] = [e for e in ENTITIES if not e["passes"]] # 3 islands
KEYSTONE_COUNT: int = len(PASSING)   # 4

# Numeric-token extractor: pulls maximal digit-runs (with internal grouping commas).
# _int_values then keeps only PLAIN integers (decimal-point tokens are dropped).
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


# ── metadata ─────────────────────────────────────────────────────────────────────────────────────

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "078",
        "test_name":        "Tier 5: Count-with-condition — islands with area > 8,000 km² (7 obscure islands)",
        "difficulty_level": "8/10",
        "category":         "Count-with-condition (page-only island areas, 7-way fan-out)",
        "level":            "graph",
        "weight":           "long",
    }


# ── task statement ────────────────────────────────────────────────────────────────────────────────

def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['country']})"
        for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not "
        "guess from memory). For EACH of the following seven islands, open the island's Wikipedia "
        "page and read ONE figure directly from the infobox: its AREA in km² "
        "(the 'Area' field — the total land area of the island; NOT its population, NOT its "
        "elevation, NOT its coastline length):\n"
        f"{listing}\n\n"
        f"Then COUNT how many of the seven islands have an area STRICTLY GREATER THAN "
        f"{THRESHOLD:,} km² and report:\n"
        f"  (a) the count — how many of the seven islands have area > {THRESHOLD:,} km² "
        "(a single integer 0–7; this is the keystone answer and belongs in your primary answer),\n"
        "  (b) which specific islands exceed that area and which do not (two named lists),\n"
        "  (c) each island's area in km² as read from its infobox, and\n"
        "  (d) the exact source URL for every Wikipedia page you read.\n\n"
        "IMPORTANT: do not guess areas from memory or from an island's general fame. "
        "You MUST open each island's Wikipedia page and read the infobox 'Area' figure."
    )


# ── deliverables / success criteria ──────────────────────────────────────────────────────────────

def get_required_deliverables() -> List[str]:
    return [
        f"Count of islands with area > {THRESHOLD:,} km² (the keystone integer, 0–7)",
        f"Which islands exceed {THRESHOLD:,} km² and which do not (passing and failing island names)",
        "Each island's area in km² (seven individual infobox values)",
        "Source URL for each island's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    passing_names = ", ".join(e["name"] for e in PASSING)
    failing_names = ", ".join(e["name"] for e in FAILING)
    return [
        "At least 6 pages visited (target 7, one per island)",
        f"Correctly reports the count as {KEYSTONE_COUNT} "
        f"(the number of islands with area > {THRESHOLD:,} km²)",
        f"Identifies the {len(PASSING)} passing islands: {passing_names}",
        f"Identifies the {len(FAILING)} failing islands: {failing_names}",
        "Reports all seven islands' areas from their infoboxes",
        "Cites a Wikipedia source URL for each island",
    ]


# ── shared text helpers ───────────────────────────────────────────────────────────────────────────

def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text: deliverables[0] if populated, else final_deliverable."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: final_deliverable plus every deliverable slot concatenated."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _int_values(text: str) -> List[int]:
    """Plain integer values present in ``text`` (decimal tokens dropped; grouping commas stripped)."""
    vals: List[int] = []
    for tok in _NUM_TOKEN_RX.findall(text):
        if "." in tok:
            continue  # drop decimals like "7367.6"
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
    """KEYSTONE gate: deliverables[0] contains the integer KEYSTONE_COUNT (= 4).

    Checks that the primary answer slot holds exactly the correct count of islands exceeding
    the threshold. Count=7 ('all of them'), count=3 (one island dropped), and count=5 (one extra
    island wrongly included) all fail — only the computed value 4 passes.

    GROUNDING: an ungrounded parametric-memory guess must not earn credit, so the count is only
    credited when the agent actually visited at least one page (visit.count > 0).
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return KEYSTONE_COUNT in _int_values(_strip_list_markers(_primary_text(result)))


# ── validation functions ──────────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a seven-way fan-out wants one page per island."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check":  "visit_count",
        "passed": n >= 6,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": f"{n} visit(s) (target >= {len(ENTITIES)}: one page per island; >= 6 to pass)",
    }


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact count of islands with area > 8,000 km², reported as a
    single integer in deliverables[0].

    Correct answer = 4.  Any other value (3, 5, 7, …) fails.  A model that guesses 'all 7'
    scores 0; a model that drops an island and reports 3 also scores 0.  Only a model that opens
    all seven pages, reads each area, applies the threshold, and counts correctly passes.
    """
    passed = _keystone_ok(result, observability)
    passing_names = ", ".join(e["name"] for e in PASSING)
    return {
        "check":  "keystone_count",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"Count {KEYSTONE_COUNT} (islands with area > {THRESHOLD:,} km²) present in primary answer"
            if passed else
            f"Count {KEYSTONE_COUNT} absent or wrong in primary answer. "
            f"The {len(PASSING)} islands exceeding {THRESHOLD:,} km² are: {passing_names}. "
            f"Beware: a naive 'all seven pass' gives count=7 (wrong); dropping an island gives "
            f"count=3 (wrong); including Leyte (7,368 km²) as a passer gives count=5 (wrong)."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the seven (island, area) pairs gathered.

    An island is credited only when BOTH its name AND its infobox area appear together in the
    response. The seven rounded areas (14250, 13429, 11831, 9311, 7368, 5786, 5306) are all
    distinct, so there is no cross-crediting. Deliberately NOT gated on the keystone — this axis
    measures whether the agent actually fanned out to all seven pages even if it miscounts or
    misapplies the threshold.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE)
        and re.search(e["area_rx"], text)
    ]
    n = len(ENTITIES)
    return {
        "check":  "coverage",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} (island, area) pairs gathered "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_passing_islands(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all four passing island names present in the answer.

    Short-circuits to 0 when the keystone is absent, so a run that reports the wrong count
    cannot bank partial credit for naming some of the correct islands. After the keystone gate,
    checks that Flores, Samar, Bangka Island, and Kodiak Island all appear in the text.
    """
    if not _keystone_ok(result, observability):
        return {
            "check":  "passing_islands",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> passing-island list not credited",
        }
    text = _all_text(result)
    hits = [e["name"] for e in PASSING if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(PASSING)
    return {
        "check":  "passing_islands",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} passing island names identified "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 5 of the 7 island Wikipedia pages.

    Short-circuits to 0 when the keystone is absent.
    """
    if not _keystone_ok(result, observability):
        return {
            "check":  "citation",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> source URL credit withheld",
        }
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {
        "check":  "citation",
        "passed": cited >= 5,
        "score":  cited / n,
        "reason": f"{cited}/{n} island pages cited (>= 5 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_count,
        validate_coverage,
        validate_passing_islands,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 062/070/072.
    return None


# ── compiled plan ─────────────────────────────────────────────────────────────────────────────────

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SEVEN INDEPENDENT parallel leaves: one per island, each fetching that island's area in km²
    from its Wikipedia infobox. ALL comparison and counting logic lives ONLY in the aggregation
    step, which is forced to restate each entity → its area value explicitly BEFORE applying the
    threshold or counting. This structural requirement:
      (a) prevents the cheap executor from hallucinating a count without visiting all pages;
      (b) ensures the threshold comparison is applied entity-by-entity, not globally guessed;
      (c) lets a diverse-grounding reranker catch any area misattribution.

    Encodes STRUCTURE only: names the seven given islands and their countries, but leaks no area
    value and never states the count or which islands pass.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_area",
            "instruction": (
                f"Open the Wikipedia article for {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its AREA in km² — the 'Area' field (the total land "
                "area of the island in square kilometres). Report ONLY that single area figure "
                "(in km²) and the exact source URL. "
                "Do NOT report population, elevation, or coastline length. Do not guess from memory."
            ),
            "expect": (
                f"AREA of {e['name']} in km² (a single number) -- source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        # Deterministic composition: the executor applies the threshold and counts in Python over
        # the seven gathered areas, rendering each island's check and both named lists itself (zero
        # extra LLM calls) — free-text counting/enumeration is the confirmed failure mode here even
        # when every area is correct. Encodes the GIVEN threshold only; no area value and no count.
        # If any leaf fails to resolve, the composer returns nothing and the recipe below runs
        # unchanged.
        "agg_mode": "computed",
        "composition": {
            "op": "count_threshold",
            "answer_noun": "island",
            "value_label": "area",
            "unit": "km²",
            "comparator": ">",
            "threshold": THRESHOLD,
            "items": [{"leaf": f"{e['key']}_area", "label": e["name"], "type": "number"}
                      for e in ENTITIES],
        },
        "aggregation": (
            "You now have, for each of the seven islands, its area in km² and a source URL. "
            "Before applying any threshold or counting, RESTATE each island's area explicitly "
            "in this format:\n"
            "  Bangka Island → [area] km²\n"
            "  Chichagof Island → [area] km²\n"
            "  Flores → [area] km²\n"
            "  Kodiak Island → [area] km²\n"
            "  Leyte → [area] km²\n"
            "  Melville Island → [area] km²\n"
            "  Samar → [area] km²\n"
            "(substituting the number you retrieved for each '[area]' placeholder).\n\n"
            "Then, FOR EACH ISLAND IN TURN, state whether its area is strictly greater than "
            "8,000 km². Finally, COUNT the number of islands whose area exceeds 8,000 km² — "
            "that integer (0–7) is the keystone answer. Report:\n"
            "  (a) the count (a single integer, 0–7) as the primary keystone answer,\n"
            "  (b) which islands exceed 8,000 km² and which do not (two named lists),\n"
            "  (c) the full list of seven areas as you restated them above, and\n"
            "  (d) the source URL for each island."
        ),
    }
