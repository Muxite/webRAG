"""
Test 087: Tier 5 (graph) — COUNT-WITH-CONDITION over page-only road tunnel lengths.
Level: graph   Weight: long   Difficulty: 8/10

Among seven solid-but-obscure road tunnels drawn from five different countries spanning two
continents, the agent must open each tunnel's Wikipedia page, read its total LENGTH in km, and
report HOW MANY of the seven have a length strictly greater than 10 km. The answer (a COUNT,
integer 0–7) is COMPUTED from seven independent page-only lookups and is not directly recallable.

WHY IT DISCRIMINATES (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap parametric (no tools): cannot simultaneously know seven obscure road tunnel lengths;
    either guesses the wrong count, claims "all seven" (the naïve overcount), or refuses.
  * cheap native (graph): may open some pages but drops one or more tunnels, misattributes
    lengths, or applies the threshold check inconsistently; count likely off by 1–2.
  * graph_compiled: seven parallel leaves each fetch ONE tunnel's length from ONE Wikipedia page;
    the aggregation is forced to restate each entity→value mapping before comparing to the
    threshold and summing the boolean outcomes; the cheap executor is rescued from holding seven
    values simultaneously and a diverse-grounding reranker can catch any threshold misapplication.

Ground truth (verified against live English Wikipedia, 2026-06-27 — each tunnel's Wikipedia
article infobox 'Length' field, read from the canonical article URL):

  tunnel                  Wikipedia slug                   country/region              length km   > 10 km?
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Zhongnanshan Tunnel     /wiki/Zhongnanshan_Tunnel        Shaanxi, China               18.040      YES
  Ryfylke Tunnel          /wiki/Ryfylke_Tunnel             Rogaland, Norway             14.400      YES
  Arlberg Road Tunnel     /wiki/Arlberg_Road_Tunnel        Vorarlberg, Austria          13.972      YES
  Gudvanga Tunnel         /wiki/Gudvanga_Tunnel            Aurland, Norway              11.428      YES
  Tauern Road Tunnel      /wiki/Tauern_Road_Tunnel         Salzburg, Austria             6.546      NO
  Hvalfjörður Tunnel      /wiki/Hvalfjörður_Tunnel         Iceland                       5.770      NO
  Hải Vân Tunnel          /wiki/Hải_Vân_Tunnel             Da Nang / Huế, Vietnam        6.280      NO
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  KEYSTONE = count of tunnels with length > 10 km = 4

THRESHOLD MARGINS (each length clears or misses 10 km by ≥ 1.428 km — no borderline value):
  PASS  Zhongnanshan Tunnel    18.040 − 10 = +8.040 km   (largest above-threshold margin)
  PASS  Ryfylke Tunnel         14.400 − 10 = +4.400 km
  PASS  Arlberg Road Tunnel    13.972 − 10 = +3.972 km
  PASS  Gudvanga Tunnel        11.428 − 10 = +1.428 km   (smallest above-threshold margin)
  FAIL  Tauern Road Tunnel      10 − 6.546 = −3.454 km   (smallest below-threshold margin)
  FAIL  Hải Vân Tunnel          10 − 6.280 = −3.720 km
  FAIL  Hvalfjörður Tunnel      10 − 5.770 = −4.230 km   (largest below-threshold margin)

  Every length is ≥ 1.428 km from the 10 km threshold, so no plausible single-figure extraction
  error (a misread of a 4–5 digit metre value or a decimal km value) can flip any tunnel's
  pass/fail status.

ANTI-PARAMETRIC: the seven tunnels span five countries and are solid but non-iconic:
  • Zhongnanshan Tunnel (China, 18.040 km) was one of the world's longest road tunnels when
    completed in 2007, boring beneath the Zhongnan Mountains in Shaanxi; its precise length
    is a technical specification rarely cited outside Chinese infrastructure reporting.
  • Ryfylke Tunnel (Norway, 14.400 km) is a subsea road tunnel opened in 2019 beneath Ryfylkefjord,
    linking Stavanger to Tau; its depth record draws more attention than its exact length.
  • Arlberg Road Tunnel (Austria, 13.972 km) carries the Arlberg Expressway through the Arlberg
    massif in Vorarlberg; a major Alpine route, but its exact length is not commonly cited.
  • Gudvanga Tunnel (Norway, 11.428 km) connects Gudvangen and Flåm along the E16 in Aurland;
    one of several long Norwegian fjord tunnels, its specific length is not distinguished from peers.
  • Tauern Road Tunnel (Austria, 6.546 km) carries the Tauern Autobahn through the Hohe Tauern
    in Salzburg state; regionally significant but very obscure globally.
  • Hvalfjörður Tunnel (Iceland, 5.770 km) is a subsea tunnel beneath Hvalfjörður fjord north of
    Reykjavík, opened in 1998; unknown outside Iceland.
  • Hải Vân Tunnel (Vietnam, 6.280 km) passes through the Hải Vân Pass on Route 1 between Da Nang
    and Huế, opened in 2005; the longest road tunnel in Southeast Asia at the time, but its exact
    length is not recalled alongside the other six.
  The COUNT (4) is not a published fact anywhere. A cheap parametric model has no reliable path
  to the correct count without opening all seven pages.

SELF-VERIFY (throwaway snippet — delete after):
    from services.agent.app.idea_tests.test_087_tier5_count_with_condition_d import (
        validate_keystone_count, ENTITIES, KEYSTONE_COUNT
    )
    # CORRECT result -> keystone PASSES
    correct = {"deliverables": [str(KEYSTONE_COUNT)], "output": {"final_deliverable": "4"}}
    assert validate_keystone_count(correct, {})["passed"], "CORRECT should pass"
    # DECOY off by one (too low) -> keystone FAILS
    decoy_3 = {"deliverables": ["3"], "output": {"final_deliverable": "3 tunnels"}}
    assert not validate_keystone_count(decoy_3, {})["passed"], "count=3 should fail"
    # DECOY off by one (too high) -> keystone FAILS
    decoy_5 = {"deliverables": ["5"], "output": {"final_deliverable": "5 tunnels"}}
    assert not validate_keystone_count(decoy_5, {})["passed"], "count=5 should fail"
    # DECOY count=7 (naive 'all') -> keystone FAILS
    decoy_7 = {"deliverables": ["7"], "output": {"final_deliverable": "all 7"}}
    assert not validate_keystone_count(decoy_7, {})["passed"], "count=7 should fail"
    print("All self-verify assertions passed.")

KEYSTONE = the count 4 (exact integer in deliverables[0]).
Secondary (gated) = lists all four passing tunnel names (Zhongnanshan, Ryfylke, Arlberg Road,
  Gudvanga).
Coverage (ungated) = how many of the seven (tunnel, length-value) pairs gathered; the seven
  lengths (18.040 / 14.400 / 13.972 / 11.428 / 6.546 / 5.770 / 6.280) are all distinct —
  coverage is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified fixtures ────────────────────────────────────────────────────────────────────────────
# 'length_km' = tunnel total length in km from the Wikipedia article infobox 'Length' field
# (live-verified 2026-06-27).  Wikipedia may express the value in metres — the km equivalent is
# recorded here.  'passes' = (length_km > THRESHOLD).  Nothing below is leaked into the task
# statement or the plan.

THRESHOLD: int = 10  # km, strictly greater-than

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":       "zhongnanshan",
        "name":      "Zhongnanshan Tunnel",
        "country":   "Shaanxi, China",
        "length_km": 18.040,
        "passes":    True,
        # Wikipedia infobox (technical specs table) shows 18.040 km (11.210 mi).
        "name_rx":   r"zhongnanshan",
        "len_rx":    r"(?<!\d)18[.,\s]?04\d?(?!\d)",
        "slug_rx":   r"wiki/zhongnanshan",
    },
    {
        "key":       "ryfylke",
        "name":      "Ryfylke Tunnel",
        "country":   "Rogaland, Norway",
        "length_km": 14.400,
        "passes":    True,
        # Wikipedia infobox 'Length' field: 14.4 km (8.9 mi). Also confirmed in article text.
        "name_rx":   r"ryfylke",
        "len_rx":    r"(?<!\d)14[.,\s]?4\d*(?!\d)",
        "slug_rx":   r"wiki/ryfylke",
    },
    {
        "key":       "arlberg",
        "name":      "Arlberg Road Tunnel",
        "country":   "Vorarlberg, Austria",
        "length_km": 13.972,
        "passes":    True,
        # Wikipedia infobox shows 13.972 km (8.68 mi) as primary measurement;
        # a second figure (15.537 km) includes approach galleries — the infobox primary is used.
        "name_rx":   r"arlberg",
        "len_rx":    r"(?<!\d)13[.,\s]?972(?!\d)",
        "slug_rx":   r"wiki/arlberg",
    },
    {
        "key":       "gudvanga",
        "name":      "Gudvanga Tunnel",
        "country":   "Aurland, Norway",
        "length_km": 11.428,
        "passes":    True,
        # Wikipedia infobox (technical specifications) shows 11,428 m (7.1 mi).
        # Equivalent: 11.428 km. Norway's third-longest road tunnel.
        "name_rx":   r"gudvanga",
        "len_rx":    r"(?<!\d)11[.,\s]?42\d?(?!\d)",
        "slug_rx":   r"wiki/gudvanga",
    },
    {
        "key":       "tauern",
        "name":      "Tauern Road Tunnel",
        "country":   "Salzburg, Austria",
        "length_km": 6.546,
        "passes":    False,
        # Wikipedia infobox shows 6,546 m (21,476 ft). Equivalent: 6.546 km.
        "name_rx":   r"tauern",
        "len_rx":    r"(?<!\d)6[.,\s]?54\d?(?!\d)",
        "slug_rx":   r"wiki/tauern",
    },
    {
        "key":       "hvalfjordur",
        "name":      "Hvalfjörður Tunnel",
        "country":   "Iceland",
        "length_km": 5.770,
        "passes":    False,
        # Wikipedia infobox (technical specifications) shows 5,770 m (18,930 ft).
        # Equivalent: 5.770 km. Article slug contains the accented Icelandic character.
        # Both the accented and URL-encoded slug forms are matched.
        "name_rx":   r"hvalf",
        "len_rx":    r"(?<!\d)5[.,\s]?77\d?(?!\d)",
        "slug_rx":   r"wiki/hvalf",
    },
    {
        "key":       "haivan",
        "name":      "Hải Vân Tunnel",
        "country":   "Da Nang / Huế, Vietnam",
        "length_km": 6.280,
        "passes":    False,
        # Wikipedia infobox (technical specifications) shows 6,280 m (3.90 mi).
        # Equivalent: 6.280 km. Article title uses Vietnamese diacritics; the slug_rx covers
        # both the literal Unicode form and the URL-percent-encoded form.
        "name_rx":   r"h.i.v.n",          # matches "Hải Vân", "Hai Van", etc.
        "len_rx":    r"(?<!\d)6[.,\s]?28\d?(?!\d)",
        "slug_rx":   r"wiki/h%e1|wiki/h.i.v.n",  # URL-encoded or Unicode/ASCII form
    },
]

PASSING: List[Dict[str, Any]] = [e for e in ENTITIES if e["passes"]]    # 4 tunnels
FAILING: List[Dict[str, Any]] = [e for e in ENTITIES if not e["passes"]] # 3 tunnels
KEYSTONE_COUNT: int = len(PASSING)   # 4

# Numeric-token extractor: pulls maximal digit-runs (with internal grouping commas).
# _int_values then keeps only PLAIN integers (decimal-point tokens are dropped).
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


# ── metadata ─────────────────────────────────────────────────────────────────────────────────────

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "087",
        "test_name":        "Tier 5: Count-with-condition — road tunnels with length > 10 km (7 obscure tunnels)",
        "difficulty_level": "8/10",
        "category":         "Count-with-condition (page-only road tunnel lengths, 7-way fan-out)",
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
        "guess from memory). For EACH of the following seven road tunnels, open the tunnel's "
        "Wikipedia page and read ONE figure directly from the infobox: its total LENGTH in km "
        "(the total length of the tunnel from portal to portal; NOT the highway or motorway "
        "length, NOT approach roads, NOT galleries — the core tunnel bore length only). "
        "Wikipedia may express the figure in metres; convert to km if so:\n"
        f"{listing}\n\n"
        f"Then COUNT how many of the seven tunnels have a length STRICTLY GREATER THAN "
        f"{THRESHOLD} km and report:\n"
        f"  (a) the count — how many of the seven tunnels have length > {THRESHOLD} km "
        "(a single integer 0–7; this is the keystone answer and belongs in your primary answer),\n"
        "  (b) which specific tunnels exceed that length and which do not (two named lists),\n"
        "  (c) each tunnel's length in km as read from its Wikipedia page, and\n"
        "  (d) the exact source URL for every Wikipedia page you read.\n\n"
        "IMPORTANT: do not guess lengths from memory or from a tunnel's general fame. "
        "You MUST open each tunnel's Wikipedia page and read the infobox length figure."
    )


# ── deliverables / success criteria ──────────────────────────────────────────────────────────────

def get_required_deliverables() -> List[str]:
    return [
        f"Count of tunnels with length > {THRESHOLD} km (the keystone integer, 0–7)",
        f"Which tunnels exceed {THRESHOLD} km and which do not (passing and failing tunnel names)",
        "Each tunnel's length in km (seven individual infobox values)",
        "Source URL for each tunnel's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    passing_names = ", ".join(e["name"] for e in PASSING)
    failing_names = ", ".join(e["name"] for e in FAILING)
    return [
        "At least 6 pages visited (target 7, one per tunnel)",
        f"Correctly reports the count as {KEYSTONE_COUNT} "
        f"(the number of tunnels with length > {THRESHOLD} km)",
        f"Identifies the {len(PASSING)} passing tunnels: {passing_names}",
        f"Identifies the {len(FAILING)} failing tunnels: {failing_names}",
        "Reports all seven tunnels' lengths from their Wikipedia pages",
        "Cites a Wikipedia source URL for each tunnel",
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
            continue  # drop decimals like "13.972"
        try:
            vals.append(int(tok.replace(",", "")))
        except ValueError:
            continue
    return vals


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: deliverables[0] contains the integer KEYSTONE_COUNT (= 4).

    Checks that the primary answer slot holds exactly the correct count of tunnels exceeding
    the threshold. Count=7 ('all of them'), count=3 (one tunnel dropped), and count=5 (one
    failing tunnel wrongly counted) all fail — only the computed value 4 passes.
    """
    return KEYSTONE_COUNT in _int_values(_primary_text(result))


# ── validation functions ──────────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a seven-way fan-out wants one page per tunnel."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check":  "visit_count",
        "passed": n >= 6,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": f"{n} visit(s) (target >= {len(ENTITIES)}: one page per tunnel; >= 6 to pass)",
    }


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact count of tunnels with length > 10 km, reported as a
    single integer in deliverables[0].

    Correct answer = 4.  Any other value (3, 5, 7, …) fails.  A model that guesses 'all 7'
    scores 0; a model that drops Gudvanga Tunnel (11.428 km, the closest to threshold) and
    reports 3 also scores 0.  Only a model that opens all seven pages, reads each length,
    converts metres to km where necessary, applies the threshold, and counts correctly passes.
    """
    passed = _keystone_ok(result)
    passing_names = ", ".join(e["name"] for e in PASSING)
    return {
        "check":  "keystone_count",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"Count {KEYSTONE_COUNT} (tunnels with length > {THRESHOLD} km) present in "
            f"primary answer"
            if passed else
            f"Count {KEYSTONE_COUNT} absent or wrong in primary answer. "
            f"The {len(PASSING)} tunnels exceeding {THRESHOLD} km are: {passing_names}. "
            f"Beware: guessing 'all seven pass' gives count=7 (wrong); dropping Gudvanga "
            f"(11.428 km, the nearest above-threshold tunnel) gives count=3 (wrong); "
            f"miscounting any of the three failing tunnels (Tauern 6.546 km, Hải Vân 6.280 km, "
            f"Hvalfjörður 5.770 km) as passing gives count=5 (wrong)."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the seven (tunnel, length) pairs gathered.

    A tunnel is credited only when BOTH its name AND its length value appear together in the
    response. The seven lengths (18.040 / 14.400 / 13.972 / 11.428 / 6.546 / 5.770 / 6.280 km)
    are all distinct, so there is no cross-crediting. Deliberately NOT gated on the keystone —
    this axis measures whether the agent fanned out to all seven pages even if it miscounts or
    misapplies the threshold.
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
            f"{len(hits)}/{n} (tunnel, length) pairs gathered "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_passing_tunnels(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all four passing tunnel names present in the answer.

    Short-circuits to 0 when the keystone is absent, so a run that reports the wrong count
    cannot bank partial credit for naming some of the correct tunnels. After the keystone gate,
    checks that Zhongnanshan Tunnel, Ryfylke Tunnel, Arlberg Road Tunnel, and Gudvanga Tunnel
    all appear.
    """
    if not _keystone_ok(result):
        return {
            "check":  "passing_tunnels",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> passing-tunnel list not credited",
        }
    text = _all_text(result)
    hits = [e["name"] for e in PASSING if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(PASSING)
    return {
        "check":  "passing_tunnels",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} passing tunnel names identified "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 5 of the 7 tunnel Wikipedia pages.

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
        "reason": f"{cited}/{n} tunnel pages cited (>= 5 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_count,
        validate_coverage,
        validate_passing_tunnels,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 062/070/072/078/082.
    return None


# ── compiled plan ─────────────────────────────────────────────────────────────────────────────────

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SEVEN INDEPENDENT parallel leaves: one per tunnel, each fetching that tunnel's total length
    in km from its Wikipedia page. ALL comparison and counting logic lives ONLY in the
    aggregation step, which is forced to restate each entity → its length value explicitly
    BEFORE applying the threshold or counting. This structural requirement:
      (a) prevents the cheap executor from hallucinating a count without visiting all pages;
      (b) ensures the threshold comparison is applied tunnel-by-tunnel, not globally guessed;
      (c) lets a diverse-grounding reranker catch any length misattribution.

    Encodes STRUCTURE only: names the seven given tunnels and their regions, but leaks no
    length value and never states the count or which tunnels pass.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_length",
            "instruction": (
                f"Open the Wikipedia article for {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its total LENGTH — the tunnel bore length from "
                "portal to portal (NOT highway length, NOT approach roads, NOT galleries). "
                "Wikipedia may express this in metres; if so, convert to km. "
                "Report ONLY that single length figure in km and the exact source URL. "
                "Do not guess from memory."
            ),
            "expect": (
                f"Total length of {e['name']} in km (a single number) -- source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the seven tunnels, its total length in km and a source URL. "
            "Before applying any threshold or counting, RESTATE each tunnel's length explicitly "
            "in this format:\n"
            "  Zhongnanshan Tunnel → [length] km\n"
            "  Ryfylke Tunnel → [length] km\n"
            "  Arlberg Road Tunnel → [length] km\n"
            "  Gudvanga Tunnel → [length] km\n"
            "  Tauern Road Tunnel → [length] km\n"
            "  Hvalfjörður Tunnel → [length] km\n"
            "  Hải Vân Tunnel → [length] km\n"
            "(substituting the number you retrieved for each '[length]' placeholder).\n\n"
            "Then, FOR EACH TUNNEL IN TURN, state whether its length is strictly greater than "
            "10 km. Finally, COUNT the number of tunnels whose length exceeds 10 km — "
            "that integer (0–7) is the keystone answer. Report:\n"
            "  (a) the count (a single integer, 0–7) as the primary keystone answer,\n"
            "  (b) which tunnels exceed 10 km and which do not (two named lists),\n"
            "  (c) the full list of seven lengths as you restated them above, and\n"
            "  (d) the source URL for each tunnel."
        ),
    }
