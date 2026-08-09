"""
Test 089: Tier 5 (graph) — COUNT-WITH-CONDITION over page-only dam heights.
Level: graph   Weight: long   Difficulty: 8/10

Among seven solid-but-obscure dams drawn from six different countries, the agent must open each
dam's Wikipedia page, read its HEIGHT in metres from the infobox, and report HOW MANY of the
seven have a height strictly greater than 200 metres. The answer (a COUNT, integer 0–7) is
COMPUTED from seven independent page-only lookups and is not directly recallable.

WHY IT DISCRIMINATES (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap parametric (no tools): cannot know seven obscure dam heights simultaneously; either
    guesses the wrong count, claims "all seven" (the naïve overcount), or refuses outright.
  * cheap native (graph): may open some pages but drops one or more dams, misattributes heights,
    or applies the threshold check inconsistently; count likely off by 1–2.
  * graph_compiled: seven parallel leaves each fetch ONE dam's height from ONE infobox; the
    aggregation is forced to restate each entity→value mapping before comparing to the threshold
    and summing the boolean outcomes; the cheap executor is rescued from holding seven values
    simultaneously and a diverse-grounding reranker can catch any threshold misapplication.

Ground truth (verified against live English Wikipedia, 2026-06-27 — each dam's infobox
'Height' field, read from the canonical article URL):

  dam                          Wikipedia title              country              height  > 200 m?
  ──────────────────────────────────────────────────────────────────────────────────────────────
  Laxiwa Dam                   Laxiwa Dam                   China (Qinghai)       250 m    YES
  Deriner Dam                  Deriner Dam                  Turkey (Artvin)       249 m    YES
  Sayano-Shushenskaya Dam      Sayano-Shushenskaya Dam      Russia (Khakassia)    242 m    YES
  Mratinje Dam                 Mratinje Dam                 Montenegro            220 m    YES
  Revelstoke Dam               Revelstoke Dam               Canada (BC)           175 m    NO
  Karakaya Dam                 Karakaya Dam                 Turkey (Euphrates)    158 m    NO
  Srinagarind Dam              Srinagarind Dam              Thailand              140 m    NO
  ──────────────────────────────────────────────────────────────────────────────────────────────
  KEYSTONE = count of dams with height > 200 m = 4

THRESHOLD MARGINS (each height clears or misses 200 m by ≥ 20 m — no borderline value):
  PASS  Laxiwa Dam                  250 − 200 = +50 m    (largest above-threshold margin)
  PASS  Deriner Dam                 249 − 200 = +49 m
  PASS  Sayano-Shushenskaya Dam     242 − 200 = +42 m
  PASS  Mratinje Dam                220 − 200 = +20 m    (smallest above-threshold margin)
  FAIL  Revelstoke Dam              175 − 200 = −25 m    (smallest below-threshold margin)
  FAIL  Karakaya Dam                158 − 200 = −42 m
  FAIL  Srinagarind Dam             140 − 200 = −60 m    (largest below-threshold margin)

  Every height is ≥ 20 m from the 200 m threshold, so no plausible single-figure extraction
  error (±10–15 m misread of a 3-digit infobox value) can flip any dam's pass/fail status.

ANTI-PARAMETRIC: the seven dams span six countries and are solid but non-iconic:
  • Laxiwa Dam on the Yellow River in Qinghai, China (250 m) — a double-curvature arch dam
    completed in 2010; rarely cited in global rankings and its precise height is not recalled.
  • Deriner Dam on the Coruh River in Artvin, Turkey (249 m) — one of Turkey's tallest dams
    yet largely unknown outside specialist infrastructure contexts.
  • Sayano-Shushenskaya Dam (Russia, 242 m) is known primarily for its 2009 accident; the
    precise height figure is not a parametrically recallable fact.
  • Mratinje Dam on the Piva River in Montenegro (220 m) is very obscure outside the region.
  • Revelstoke Dam (Canada, 175 m), Karakaya Dam (Turkey, 158 m), and Srinagarind Dam
    (Thailand, 140 m) are all sufficiently obscure that their heights are not simultaneously
    recallable without opening each dam's Wikipedia page.
  The COUNT (4) is not a published fact anywhere. A cheap parametric model has no reliable path
  to the correct count without opening all seven pages.

SELF-VERIFY (throwaway snippet — delete after):
    from agent.app.idea_tests.test_089_tier5_count_with_condition_e import (
        validate_keystone_count, ENTITIES, KEYSTONE_COUNT
    )
    # CORRECT result -> keystone PASSES
    correct = {"deliverables": [str(KEYSTONE_COUNT)], "output": {"final_deliverable": "4"}}
    assert validate_keystone_count(correct, {})["passed"], "CORRECT should pass"
    # DECOY off by one -> keystone FAILS
    decoy_3 = {"deliverables": ["3"], "output": {"final_deliverable": "3 dams"}}
    assert not validate_keystone_count(decoy_3, {})["passed"], "count=3 should fail"
    decoy_5 = {"deliverables": ["5"], "output": {"final_deliverable": "5 dams"}}
    assert not validate_keystone_count(decoy_5, {})["passed"], "count=5 should fail"
    # DECOY count=7 (naive 'all') -> keystone FAILS
    decoy_7 = {"deliverables": ["7"], "output": {"final_deliverable": "all 7"}}
    assert not validate_keystone_count(decoy_7, {})["passed"], "count=7 should fail"
    print("All self-verify assertions passed.")

KEYSTONE = the count 4 (exact integer in deliverables[0]).
Secondary (gated) = lists all four passing dam names (Laxiwa, Deriner, Sayano-Shushenskaya,
  Mratinje).
Coverage (ungated) = how many of the seven (dam, height-value) pairs gathered; the seven
  heights (250, 249, 242, 220, 175, 158, 140) are all distinct — coverage is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified fixtures ────────────────────────────────────────────────────────────────────────────
# 'height' = height in metres from the Wikipedia infobox 'Height' field (live-verified).
# 'passes' = (height > THRESHOLD).  Nothing below is leaked into the task statement or the plan.

THRESHOLD: int = 200  # metres, strictly greater-than

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":       "laxiwa",
        "name":      "Laxiwa Dam",
        "country":   "China (Qinghai Province)",
        "height":    250,
        "passes":    True,
        "name_rx":   r"laxiwa",
        "height_rx": r"(?<!\d)250(?!\d)",
        "slug_rx":   r"wiki/laxiwa_dam",
    },
    {
        "key":       "deriner",
        "name":      "Deriner Dam",
        "country":   "Turkey (Artvin Province)",
        "height":    249,
        "passes":    True,
        "name_rx":   r"deriner",
        "height_rx": r"(?<!\d)249(?!\d)",
        "slug_rx":   r"wiki/deriner_dam",
    },
    {
        "key":       "sayano",
        # Wikipedia article title is "Sayano-Shushenskaya Dam" (en-dash in URL, hyphen in title).
        "name":      "Sayano-Shushenskaya Dam",
        "country":   "Russia (Khakassia)",
        "height":    242,
        "passes":    True,
        "name_rx":   r"sayano|shushenskaya",
        "height_rx": r"(?<!\d)242(?!\d)",
        "slug_rx":   r"wiki/sayano",
    },
    {
        "key":       "mratinje",
        "name":      "Mratinje Dam",
        "country":   "Montenegro",
        "height":    220,
        "passes":    True,
        "name_rx":   r"mratinje",
        "height_rx": r"(?<!\d)220(?!\d)",
        "slug_rx":   r"wiki/mratinje_dam",
    },
    {
        "key":       "revelstoke",
        "name":      "Revelstoke Dam",
        "country":   "Canada (British Columbia)",
        "height":    175,
        "passes":    False,
        "name_rx":   r"revelstoke",
        "height_rx": r"(?<!\d)175(?!\d)",
        "slug_rx":   r"wiki/revelstoke_dam",
    },
    {
        "key":       "karakaya",
        "name":      "Karakaya Dam",
        "country":   "Turkey (Euphrates River)",
        "height":    158,
        "passes":    False,
        "name_rx":   r"karakaya",
        "height_rx": r"(?<!\d)158(?!\d)",
        "slug_rx":   r"wiki/karakaya_dam",
    },
    {
        "key":       "srinagarind",
        "name":      "Srinagarind Dam",
        "country":   "Thailand",
        "height":    140,
        "passes":    False,
        "name_rx":   r"srinagarind",
        "height_rx": r"(?<!\d)140(?!\d)",
        "slug_rx":   r"wiki/srinagarind_dam",
    },
]

PASSING: List[Dict[str, Any]] = [e for e in ENTITIES if e["passes"]]     # 4 dams
FAILING: List[Dict[str, Any]] = [e for e in ENTITIES if not e["passes"]]  # 3 dams
KEYSTONE_COUNT: int = len(PASSING)   # 4

# Numeric-token extractor: pulls maximal digit-runs (with internal grouping commas).
# _int_values then keeps only PLAIN integers (decimal-point tokens are dropped).
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


# ── metadata ─────────────────────────────────────────────────────────────────────────────────────

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "089",
        "test_name":        "Tier 5: Count-with-condition — dams with height > 200 m (7 obscure dams)",
        "difficulty_level": "8/10",
        "category":         "Count-with-condition (page-only dam heights, 7-way fan-out)",
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
        "guess from memory). For EACH of the following seven dams, open the dam's Wikipedia "
        "page and read ONE figure directly from the infobox: its HEIGHT in metres "
        "(the 'Height' field — the height of the dam structure from foundation to crest; "
        "NOT its reservoir capacity, NOT its length or width):\n"
        f"{listing}\n\n"
        f"Then COUNT how many of the seven dams have a height STRICTLY GREATER THAN "
        f"{THRESHOLD} metres and report:\n"
        f"  (a) the count — how many of the seven dams have height > {THRESHOLD} m "
        "(a single integer 0–7; this is the keystone answer and belongs in your primary answer),\n"
        "  (b) which specific dams exceed that height and which do not (two named lists),\n"
        "  (c) each dam's height in metres as read from its infobox, and\n"
        "  (d) the exact source URL for every Wikipedia page you read.\n\n"
        "IMPORTANT: do not guess heights from memory or from a dam's general fame. "
        "You MUST open each dam's Wikipedia page and read the infobox 'Height' figure."
    )


# ── deliverables / success criteria ──────────────────────────────────────────────────────────────

def get_required_deliverables() -> List[str]:
    return [
        f"Count of dams with height > {THRESHOLD} m (the keystone integer, 0–7)",
        f"Which dams exceed {THRESHOLD} m and which do not (passing and failing dam names)",
        "Each dam's height in metres (seven individual infobox values)",
        "Source URL for each dam's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    passing_names = ", ".join(e["name"] for e in PASSING)
    failing_names = ", ".join(e["name"] for e in FAILING)
    return [
        "At least 6 pages visited (target 7, one per dam)",
        f"Correctly reports the count as {KEYSTONE_COUNT} "
        f"(the number of dams with height > {THRESHOLD} m)",
        f"Identifies the {len(PASSING)} passing dams: {passing_names}",
        f"Identifies the {len(FAILING)} failing dams: {failing_names}",
        "Reports all seven dams' heights from their infoboxes",
        "Cites a Wikipedia source URL for each dam",
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
            continue  # drop decimals like "175.5"
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


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: deliverables[0] contains the integer KEYSTONE_COUNT (= 4).

    Checks that the primary answer slot holds exactly the correct count of dams exceeding
    the threshold. Count=7 ('all of them'), count=3 (one dam dropped), and count=5 (one extra
    dam wrongly included) all fail — only the computed value 4 passes.
    """
    return KEYSTONE_COUNT in _int_values(_strip_list_markers(_primary_text(result)))


# ── validation functions ──────────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a seven-way fan-out wants one page per dam."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check":  "visit_count",
        "passed": n >= 6,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": f"{n} visit(s) (target >= {len(ENTITIES)}: one page per dam; >= 6 to pass)",
    }


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact count of dams with height > 200 m, reported as a
    single integer in deliverables[0].

    Correct answer = 4.  Any other value (3, 5, 7, …) fails.  A model that guesses 'all 7'
    scores 0; a model that drops a dam and reports 3 also scores 0.  Only a model that opens
    all seven pages, reads each height, applies the threshold, and counts correctly passes.
    """
    passed = _keystone_ok(result)
    passing_names = ", ".join(e["name"] for e in PASSING)
    return {
        "check":  "keystone_count",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"Count {KEYSTONE_COUNT} (dams with height > {THRESHOLD} m) present in primary answer"
            if passed else
            f"Count {KEYSTONE_COUNT} absent or wrong in primary answer. "
            f"The {len(PASSING)} dams exceeding {THRESHOLD} m are: {passing_names}. "
            f"Beware: a naive 'all seven pass' gives count=7 (wrong); dropping one dam gives "
            f"count=3 (wrong); including Revelstoke Dam (175 m) as a passer gives count=5 (wrong)."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the seven (dam, height) pairs gathered.

    A dam is credited only when BOTH its name AND its infobox height appear together in the
    response. The seven height values (250, 249, 242, 220, 175, 158, 140) are all distinct,
    so there is no cross-crediting. Deliberately NOT gated on the keystone — this axis measures
    whether the agent actually fanned out to all seven pages even if it miscounts or misapplies
    the threshold.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE)
        and re.search(e["height_rx"], text)
    ]
    n = len(ENTITIES)
    return {
        "check":  "coverage",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} (dam, height) pairs gathered "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_passing_dams(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all four passing dam names present in the answer.

    Short-circuits to 0 when the keystone is absent, so a run that reports the wrong count
    cannot bank partial credit for naming some of the correct dams. After the keystone gate,
    checks that Laxiwa Dam, Deriner Dam, Sayano-Shushenskaya Dam, and Mratinje Dam all appear
    in the text.
    """
    if not _keystone_ok(result):
        return {
            "check":  "passing_dams",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> passing-dam list not credited",
        }
    text = _all_text(result)
    hits = [e["name"] for e in PASSING if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(PASSING)
    return {
        "check":  "passing_dams",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} passing dam names identified "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 5 of the 7 dam Wikipedia pages.

    Short-circuits to 0 when the keystone is absent.
    """
    if not _keystone_ok(result):
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
        "reason": f"{cited}/{n} dam pages cited (>= 5 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_count,
        validate_coverage,
        validate_passing_dams,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 062/070.
    return None


# ── compiled plan ─────────────────────────────────────────────────────────────────────────────────

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SEVEN INDEPENDENT parallel leaves: one per dam, each fetching that dam's height in metres
    from its Wikipedia infobox. ALL comparison and counting logic lives ONLY in the aggregation
    step, which is forced to restate each entity → its height value explicitly BEFORE applying
    the threshold or counting. This structural requirement:
      (a) prevents the cheap executor from hallucinating a count without visiting all pages;
      (b) ensures the threshold comparison is applied entity-by-entity, not globally guessed;
      (c) lets a diverse-grounding reranker catch any height misattribution.

    Encodes STRUCTURE only: names the seven given dams and their countries, but leaks no height
    value and never states the count or which dams pass.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_height",
            "instruction": (
                f"Open the Wikipedia article for {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its HEIGHT in metres — the 'Height' field "
                "(the height of the dam structure from foundation to crest). Report ONLY "
                "that single height figure (a whole number in metres) and the exact source URL. "
                "Do NOT report reservoir capacity, dam length, or crest elevation. "
                "Do not guess from memory."
            ),
            "expect": (
                f"HEIGHT of {e['name']} in metres (a single integer) -- source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the seven dams, its height in metres and a source URL. "
            "Before applying any threshold or counting, RESTATE each dam's height explicitly "
            "in this format:\n"
            "  Laxiwa Dam → [height] m\n"
            "  Deriner Dam → [height] m\n"
            "  Sayano-Shushenskaya Dam → [height] m\n"
            "  Mratinje Dam → [height] m\n"
            "  Revelstoke Dam → [height] m\n"
            "  Karakaya Dam → [height] m\n"
            "  Srinagarind Dam → [height] m\n"
            "(substituting the integer you retrieved for each '[height]' placeholder).\n\n"
            "Then, FOR EACH DAM IN TURN, state whether its height is strictly greater than "
            "200 metres. Finally, COUNT the number of dams whose height exceeds 200 m — "
            "that integer (0–7) is the keystone answer. Report:\n"
            "  (a) the count (a single integer, 0–7) as the primary keystone answer,\n"
            "  (b) which dams exceed 200 m and which do not (two named lists),\n"
            "  (c) the full list of seven heights as you restated them above, and\n"
            "  (d) the source URL for each dam."
        ),
    }
