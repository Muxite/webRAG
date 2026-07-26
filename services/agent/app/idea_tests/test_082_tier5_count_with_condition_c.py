"""
Test 082: Tier 5 (graph) — COUNT-WITH-CONDITION over page-only river lengths.
Level: graph   Weight: long   Difficulty: 8/10

Among seven solid-but-obscure rivers drawn from four different continents, the agent must open
each river's Wikipedia page, read its LENGTH in km from the infobox, and report HOW MANY of
the seven have a length strictly greater than 1,000 km. The answer (a COUNT, integer 0–7) is
COMPUTED from seven independent page-only lookups and is not directly recallable.

WHY IT DISCRIMINATES (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap parametric (no tools): cannot know seven obscure river lengths simultaneously; either
    guesses the wrong count, claims "all seven" (the naïve overcount), or refuses outright.
  * cheap native (graph): may open some pages but drops one or more rivers, misattributes lengths,
    or applies the threshold check inconsistently; count likely off by 1–2.
  * graph_compiled: seven parallel leaves each fetch ONE river's length from ONE infobox; the
    aggregation is forced to restate each entity→value mapping before comparing to the threshold
    and summing the boolean outcomes; the cheap executor is rescued from holding seven values
    simultaneously and a diverse-grounding reranker can catch any threshold misapplication.

Ground truth (verified against live English Wikipedia, 2026-06-27 — each river's infobox
'Length' field, read from the canonical article URL):

  river                 Wikipedia slug                            country/region               length km   > 1000?
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Sepik River           /wiki/Sepik_River                         Papua New Guinea                1,126      YES
  Atbara River          /wiki/Atbarah_River                       Ethiopia / Sudan                  805      NO
  Benue River           /wiki/Benue_River                         Nigeria / Cameroon              1,440      YES
  Fitzroy River         /wiki/Fitzroy_River_(Western_Australia)   Western Australia, Australia      733      NO
  Madre de Dios River   /wiki/Madre_de_Dios_River                 Peru / Bolivia                  1,347      YES
  Tugela River          /wiki/Tugela_River                        KwaZulu-Natal, South Africa       560      NO
  Guaporé River         /wiki/Guaporé_River                       Brazil / Bolivia                1,260      YES
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  KEYSTONE = count of rivers with length > 1,000 km = 4

THRESHOLD MARGINS (each length clears or misses 1,000 km by ≥ 126 km — no borderline value):
  PASS  Benue River           1,440 − 1,000 = +440 km   (largest above-threshold margin)
  PASS  Madre de Dios River   1,347 − 1,000 = +347 km
  PASS  Guaporé River         1,260 − 1,000 = +260 km
  PASS  Sepik River           1,126 − 1,000 = +126 km   (smallest above-threshold margin)
  FAIL  Atbara River            805 − 1,000 = −195 km   (smallest below-threshold margin)
  FAIL  Fitzroy River           733 − 1,000 = −267 km
  FAIL  Tugela River            560 − 1,000 = −440 km   (largest below-threshold margin)

  Every length is ≥ 126 km from the 1,000 km threshold, so no plausible single-figure extraction
  error (a misread of a 3–4 digit infobox value) can flip any river's pass/fail status.

ANTI-PARAMETRIC: the seven rivers span four continents and are solid but non-iconic:
  • Sepik River (Papua New Guinea, 1,126 km) is renowned in anthropology for the distinctive
    wood-carving cultures along its banks; its exact length is not a commonly cited statistic.
  • Atbara River (Ethiopia/Sudan, 805 km) is the last major right-bank tributary of the Nile,
    draining the Ethiopian highlands into Sudan — obscure outside regional geography.
  • Benue River (Nigeria/Cameroon, 1,440 km) is the principal tributary of the Niger River;
    the Niger itself is well-known but the Benue's exact length is not commonly recalled.
  • Fitzroy River (Western Australia, 733 km) drains the Kimberley plateau into the Indian Ocean
    near Derby; virtually unknown outside Australia.
  • Madre de Dios River (Peru/Bolivia, 1,347 km) is a major Amazon tributary; its exact length
    is page-specific and not simultaneously recalled alongside the other six.
  • Tugela River (South Africa, 560 km) is best known for Tugela Falls (the world's second-
    tallest waterfall); its total river length is far less familiar than the waterfall's fame.
  • Guaporé River (Brazil/Bolivia, 1,260 km), known in Bolivia as the Iténez River, is a remote
    border river — its length is entirely page-specific and rarely cited.
  The COUNT (4) is not a published fact anywhere. A cheap parametric model has no reliable path
  to the correct count without opening all seven pages.

SELF-VERIFY (throwaway snippet — delete after):
    from services.agent.app.idea_tests.test_082_tier5_count_with_condition_c import (
        validate_keystone_count, ENTITIES, KEYSTONE_COUNT
    )
    # CORRECT result -> keystone PASSES
    correct = {"deliverables": [str(KEYSTONE_COUNT)], "output": {"final_deliverable": "4"}}
    assert validate_keystone_count(correct, {})["passed"], "CORRECT should pass"
    # DECOY off by one (too low) -> keystone FAILS
    decoy_3 = {"deliverables": ["3"], "output": {"final_deliverable": "3 rivers"}}
    assert not validate_keystone_count(decoy_3, {})["passed"], "count=3 should fail"
    # DECOY off by one (too high) -> keystone FAILS
    decoy_5 = {"deliverables": ["5"], "output": {"final_deliverable": "5 rivers"}}
    assert not validate_keystone_count(decoy_5, {})["passed"], "count=5 should fail"
    # DECOY count=7 (naive 'all') -> keystone FAILS
    decoy_7 = {"deliverables": ["7"], "output": {"final_deliverable": "all 7"}}
    assert not validate_keystone_count(decoy_7, {})["passed"], "count=7 should fail"
    print("All self-verify assertions passed.")

KEYSTONE = the count 4 (exact integer in deliverables[0]).
Secondary (gated) = lists all four passing river names (Benue, Madre de Dios, Guaporé, Sepik).
Coverage (ungated) = how many of the seven (river, length-value) pairs gathered; the seven
  lengths (1440 / 1347 / 1260 / 1126 / 805 / 733 / 560) are all distinct — coverage is
  collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified fixtures ────────────────────────────────────────────────────────────────────────────
# 'length_km' = river length in km from the Wikipedia infobox 'Length' field (live-verified
# 2026-06-27).  'passes' = (length_km > THRESHOLD).  Nothing below is leaked into the task
# statement or the plan.

THRESHOLD: int = 1_000  # km, strictly greater-than

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":       "sepik",
        "name":      "Sepik River",
        "country":   "Papua New Guinea",
        "length_km": 1126,
        "passes":    True,
        # Wikipedia infobox shows 1,126 km (700 mi).
        "name_rx":   r"sepik",
        "len_rx":    r"(?<!\d)1[,\s]?126(?!\d)",
        "slug_rx":   r"wiki/sepik_river",
    },
    {
        "key":       "atbara",
        "name":      "Atbara River",
        "country":   "Ethiopia / Sudan",
        "length_km": 805,
        "passes":    False,
        # Wikipedia article is titled "Atbarah River"; infobox shows 805 km (500 mi).
        # Both the common spelling ("Atbara") and the article title ("Atbarah") are handled.
        "name_rx":   r"atbar[ah]",
        "len_rx":    r"(?<!\d)805(?!\d)",
        "slug_rx":   r"wiki/atbar[ah]",
    },
    {
        "key":       "benue",
        "name":      "Benue River",
        "country":   "Nigeria / Cameroon",
        "length_km": 1440,
        "passes":    True,
        # Wikipedia infobox shows 1,440 km (890 mi).
        "name_rx":   r"benue",
        "len_rx":    r"(?<!\d)1[,\s]?440(?!\d)",
        "slug_rx":   r"wiki/benue_river",
    },
    {
        "key":       "fitzroy",
        "name":      "Fitzroy River",
        "country":   "Western Australia, Australia",
        "length_km": 733,
        "passes":    False,
        # Wikipedia article is "Fitzroy River (Western Australia)"; infobox shows 733 km (455 mi).
        "name_rx":   r"fitzroy",
        "len_rx":    r"(?<!\d)733(?!\d)",
        "slug_rx":   r"wiki/fitzroy_river",
    },
    {
        "key":       "madre_de_dios",
        "name":      "Madre de Dios River",
        "country":   "Peru / Bolivia",
        "length_km": 1347,
        "passes":    True,
        # Wikipedia infobox shows 1,347 km (837 mi).
        "name_rx":   r"madre\s+de\s+dios",
        "len_rx":    r"(?<!\d)1[,\s]?347(?!\d)",
        "slug_rx":   r"wiki/madre_de_dios",
    },
    {
        "key":       "tugela",
        "name":      "Tugela River",
        "country":   "KwaZulu-Natal, South Africa",
        "length_km": 560,
        "passes":    False,
        # Wikipedia infobox shows 560 km (350 mi).
        "name_rx":   r"tugela",
        "len_rx":    r"(?<!\d)560(?!\d)",
        "slug_rx":   r"wiki/tugela",
    },
    {
        "key":       "guapore",
        "name":      "Guaporé River",
        "country":   "Brazil / Bolivia",
        "length_km": 1260,
        "passes":    True,
        # Wikipedia article is "Guaporé River" (accented); infobox shows 1,260 km (780 mi).
        # slug_rx prefix matches both encoded (%C3%A9) and plain ASCII URL forms.
        "name_rx":   r"guapor",
        "len_rx":    r"(?<!\d)1[,\s]?260(?!\d)",
        "slug_rx":   r"wiki/guapor",
    },
]

PASSING: List[Dict[str, Any]] = [e for e in ENTITIES if e["passes"]]    # 4 rivers
FAILING: List[Dict[str, Any]] = [e for e in ENTITIES if not e["passes"]] # 3 rivers
KEYSTONE_COUNT: int = len(PASSING)   # 4

# Numeric-token extractor: pulls maximal digit-runs (with internal grouping commas).
# _int_values then keeps only PLAIN integers (decimal-point tokens are dropped).
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


# ── metadata ─────────────────────────────────────────────────────────────────────────────────────

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "082",
        "test_name":        "Tier 5: Count-with-condition — rivers with length > 1,000 km (7 obscure rivers)",
        "difficulty_level": "8/10",
        "category":         "Count-with-condition (page-only river lengths, 7-way fan-out)",
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
        "guess from memory). For EACH of the following seven rivers, open the river's Wikipedia "
        "page and read ONE figure directly from the infobox: its LENGTH in km "
        "(the 'Length' field — the total length of the river from source to mouth; NOT its "
        "drainage basin area, NOT its average discharge, NOT its elevation):\n"
        f"{listing}\n\n"
        f"Then COUNT how many of the seven rivers have a length STRICTLY GREATER THAN "
        f"{THRESHOLD:,} km and report:\n"
        f"  (a) the count — how many of the seven rivers have length > {THRESHOLD:,} km "
        "(a single integer 0–7; this is the keystone answer and belongs in your primary answer),\n"
        "  (b) which specific rivers exceed that length and which do not (two named lists),\n"
        "  (c) each river's length in km as read from its infobox, and\n"
        "  (d) the exact source URL for every Wikipedia page you read.\n\n"
        "IMPORTANT: do not guess lengths from memory or from a river's general fame. "
        "You MUST open each river's Wikipedia page and read the infobox 'Length' figure."
    )


# ── deliverables / success criteria ──────────────────────────────────────────────────────────────

def get_required_deliverables() -> List[str]:
    return [
        f"Count of rivers with length > {THRESHOLD:,} km (the keystone integer, 0–7)",
        f"Which rivers exceed {THRESHOLD:,} km and which do not (passing and failing river names)",
        "Each river's length in km (seven individual infobox values)",
        "Source URL for each river's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    passing_names = ", ".join(e["name"] for e in PASSING)
    failing_names = ", ".join(e["name"] for e in FAILING)
    return [
        "At least 6 pages visited (target 7, one per river)",
        f"Correctly reports the count as {KEYSTONE_COUNT} "
        f"(the number of rivers with length > {THRESHOLD:,} km)",
        f"Identifies the {len(PASSING)} passing rivers: {passing_names}",
        f"Identifies the {len(FAILING)} failing rivers: {failing_names}",
        "Reports all seven rivers' lengths from their infoboxes",
        "Cites a Wikipedia source URL for each river",
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
            continue  # drop decimals like "1347.5"
        try:
            vals.append(int(tok.replace(",", "")))
        except ValueError:
            continue
    return vals


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] contains the integer KEYSTONE_COUNT (= 4).

    Checks that the primary answer slot holds exactly the correct count of rivers exceeding
    the threshold. Count=7 ('all of them'), count=3 (one river dropped), and count=5 (one extra
    river wrongly included) all fail — only the computed value 4 passes.

    GROUNDING: an ungrounded parametric-memory guess must not earn credit, so the count is only
    credited when the agent actually visited at least one page (visit.count > 0).
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return KEYSTONE_COUNT in _int_values(_primary_text(result))


# ── validation functions ──────────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a seven-way fan-out wants one page per river."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check":  "visit_count",
        "passed": n >= 6,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": f"{n} visit(s) (target >= {len(ENTITIES)}: one page per river; >= 6 to pass)",
    }


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact count of rivers with length > 1,000 km, reported as a
    single integer in deliverables[0].

    Correct answer = 4.  Any other value (3, 5, 7, …) fails.  A model that guesses 'all 7'
    scores 0; a model that drops a river and reports 3 also scores 0.  Only a model that opens
    all seven pages, reads each length, applies the threshold, and counts correctly passes.
    """
    passed = _keystone_ok(result, observability)
    passing_names = ", ".join(e["name"] for e in PASSING)
    return {
        "check":  "keystone_count",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"Count {KEYSTONE_COUNT} (rivers with length > {THRESHOLD:,} km) present in "
            f"primary answer"
            if passed else
            f"Count {KEYSTONE_COUNT} absent or wrong in primary answer. "
            f"The {len(PASSING)} rivers exceeding {THRESHOLD:,} km are: {passing_names}. "
            f"Beware: a naive 'all seven pass' gives count=7 (wrong); dropping a river gives "
            f"count=3 (wrong); including Atbara River (805 km) as a passer gives "
            f"count=5 (wrong)."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the seven (river, length) pairs gathered.

    A river is credited only when BOTH its name AND its infobox length appear together in the
    response. The seven lengths (1440, 1347, 1260, 1126, 805, 733, 560) are all distinct, so
    there is no cross-crediting. Deliberately NOT gated on the keystone — this axis measures
    whether the agent actually fanned out to all seven pages even if it miscounts or misapplies
    the threshold.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE)
        and re.search(e["len_rx"], text)
    ]
    n = len(ENTITIES)
    return {
        "check":  "coverage",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} (river, length) pairs gathered "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_passing_rivers(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all four passing river names present in the answer.

    Short-circuits to 0 when the keystone is absent, so a run that reports the wrong count
    cannot bank partial credit for naming some of the correct rivers. After the keystone gate,
    checks that Benue River, Madre de Dios River, Guaporé River, and Sepik River all appear.
    """
    if not _keystone_ok(result, observability):
        return {
            "check":  "passing_rivers",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> passing-river list not credited",
        }
    text = _all_text(result)
    hits = [e["name"] for e in PASSING if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(PASSING)
    return {
        "check":  "passing_rivers",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} passing river names identified "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 5 of the 7 river Wikipedia pages.

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
        "reason": f"{cited}/{n} river pages cited (>= 5 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_count,
        validate_coverage,
        validate_passing_rivers,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 062/070/072/078.
    return None


# ── compiled plan ─────────────────────────────────────────────────────────────────────────────────

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SEVEN INDEPENDENT parallel leaves: one per river, each fetching that river's length in km
    from its Wikipedia infobox. ALL comparison and counting logic lives ONLY in the aggregation
    step, which is forced to restate each entity → its length value explicitly BEFORE applying
    the threshold or counting. This structural requirement:
      (a) prevents the cheap executor from hallucinating a count without visiting all pages;
      (b) ensures the threshold comparison is applied entity-by-entity, not globally guessed;
      (c) lets a diverse-grounding reranker catch any length misattribution.

    Encodes STRUCTURE only: names the seven given rivers and their regions, but leaks no length
    value and never states the count or which rivers pass.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_length",
            "instruction": (
                f"Open the Wikipedia article for {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its LENGTH in km — the 'Length' field (the total "
                "length of the river from source to mouth in kilometres). Report ONLY that "
                "single length figure (in km) and the exact source URL. "
                "Do NOT report drainage basin area, discharge rate, or elevation. "
                "Do not guess from memory."
            ),
            "expect": (
                f"LENGTH of {e['name']} in km (a single number) -- source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the seven rivers, its length in km and a source URL. "
            "Before applying any threshold or counting, RESTATE each river's length explicitly "
            "in this format:\n"
            "  Sepik River → [length] km\n"
            "  Atbara River → [length] km\n"
            "  Benue River → [length] km\n"
            "  Fitzroy River → [length] km\n"
            "  Madre de Dios River → [length] km\n"
            "  Tugela River → [length] km\n"
            "  Guaporé River → [length] km\n"
            "(substituting the number you retrieved for each '[length]' placeholder).\n\n"
            "Then, FOR EACH RIVER IN TURN, state whether its length is strictly greater than "
            "1,000 km. Finally, COUNT the number of rivers whose length exceeds 1,000 km — "
            "that integer (0–7) is the keystone answer. Report:\n"
            "  (a) the count (a single integer, 0–7) as the primary keystone answer,\n"
            "  (b) which rivers exceed 1,000 km and which do not (two named lists),\n"
            "  (c) the full list of seven lengths as you restated them above, and\n"
            "  (d) the source URL for each river."
        ),
    }
