"""
Test 072: Tier 5 (graph) — COUNT-WITH-CONDITION over page-only lake maximum depths.
Level: graph   Weight: long   Difficulty: 8/10

Among seven solid-but-obscure lakes drawn from five different countries, the agent must open each
lake's Wikipedia page, read its MAXIMUM DEPTH in metres from the infobox, and report HOW MANY of
the seven have a maximum depth strictly greater than 480 metres. The answer (a COUNT, integer
0–7) is COMPUTED from seven independent page-only lookups and is not directly recallable.

WHY IT DISCRIMINATES (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap parametric (no tools): cannot know seven obscure lake depths simultaneously; either
    guesses the wrong count, claims "all seven" (the naïve overcount), or refuses outright.
  * cheap native (graph): may open some pages but drops one or more lakes, misattributes depths,
    or applies the threshold check inconsistently; count likely off by 1–2.
  * graph_compiled: seven parallel leaves each fetch ONE lake's depth from ONE infobox; the
    aggregation is forced to restate each entity→value mapping before comparing to the threshold
    and summing the boolean outcomes; the cheap executor is rescued from holding seven values
    simultaneously and a diverse-grounding reranker can catch any threshold misapplication.

Ground truth (verified against live English Wikipedia, 2026-06-27 — each lake's infobox
'Max. depth' field, read from the canonical article URL):

  lake                Wikipedia title       country                  max depth  > 480 m?
  ─────────────────────────────────────────────────────────────────────────────────────────
  Lake Matano         Lake Matano           Indonesia (S. Sulawesi)    590 m      YES
  Hornindalsvatnet    Hornindalsvatnet      Norway                     514 m      YES
  Quesnel Lake        Quesnel Lake          Canada (British Columbia)  511 m      YES
  Sarez Lake          Sarez Lake            Tajikistan                 505 m      YES
  Tinnsjå             Tinnsjå               Norway                     460 m      NO
  Lake Manapouri      Lake Manapouri        New Zealand                444 m      NO
  Lake Ohrid          Lake Ohrid            North Macedonia / Albania  288 m      NO
  ─────────────────────────────────────────────────────────────────────────────────────────
  KEYSTONE = count of lakes with max depth > 480 m = 4

THRESHOLD MARGINS (each depth clears or misses 480 m by ≥ 20 m — no borderline value):
  PASS  Lake Matano       590 − 480 = +110 m    (largest above-threshold margin)
  PASS  Hornindalsvatnet  514 − 480 =  +34 m
  PASS  Quesnel Lake      511 − 480 =  +31 m
  PASS  Sarez Lake        505 − 480 =  +25 m
  FAIL  Tinnsjå           460 − 480 =  −20 m    (smallest below-threshold margin)
  FAIL  Lake Manapouri    444 − 480 =  −36 m
  FAIL  Lake Ohrid        288 − 480 = −192 m    (largest below-threshold margin)

  Every depth is ≥ 20 m from the 480 m threshold, so no plausible single-figure extraction
  error (±10–15 m misread of a 3-digit infobox value) can flip any lake's pass/fail status.

ANTI-PARAMETRIC: the seven lakes span five countries and are solid but non-iconic:
  • Lake Matano is the 11th deepest lake in the world but sits on the remote Indonesian island
    of Sulawesi — its depth (590 m) is not commonly recalled.
  • Hornindalsvatnet's fame is that it is the deepest lake in Western Europe, but the exact
    figure (514 m) is page-specific, and knowing the superlative does not reveal whether it
    falls above or below 480 m without looking up the number.
  • Quesnel Lake (British Columbia, 511 m), Sarez Lake (Tajikistan, 505 m), Tinnsjå (Norway,
    460 m), Lake Manapouri (New Zealand, 444 m), and Lake Ohrid (N. Macedonia/Albania, 288 m)
    are all obscure enough that their depths are not simultaneously recallable.
  The COUNT (4) is not a published fact anywhere. A cheap parametric model has no reliable path
  to the correct count without opening all seven pages.

SELF-VERIFY (throwaway snippet — delete after):
    from services.agent.app.idea_tests.test_072_tier5_count_with_condition import (
        validate_keystone_count, ENTITIES, KEYSTONE_COUNT
    )
    # CORRECT result -> keystone PASSES
    correct = {"deliverables": [str(KEYSTONE_COUNT)], "output": {"final_deliverable": "4"}}
    assert validate_keystone_count(correct, {})["passed"], "CORRECT should pass"
    # DECOY off by one -> keystone FAILS
    decoy_3 = {"deliverables": ["3"], "output": {"final_deliverable": "3 lakes"}}
    assert not validate_keystone_count(decoy_3, {})["passed"], "count=3 should fail"
    decoy_5 = {"deliverables": ["5"], "output": {"final_deliverable": "5 lakes"}}
    assert not validate_keystone_count(decoy_5, {})["passed"], "count=5 should fail"
    # DECOY count=7 (naive 'all') -> keystone FAILS
    decoy_7 = {"deliverables": ["7"], "output": {"final_deliverable": "all 7"}}
    assert not validate_keystone_count(decoy_7, {})["passed"], "count=7 should fail"
    print("All self-verify assertions passed.")

KEYSTONE = the count 4 (exact integer in deliverables[0]).
Secondary (gated) = lists all four passing lake names (Matano, Hornindalsvatnet, Quesnel, Sarez).
Coverage (ungated) = how many of the seven (lake, depth-value) pairs gathered; the seven
  depths (590, 514, 511, 505, 460, 444, 288) are all distinct — coverage is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified fixtures ────────────────────────────────────────────────────────────────────────────
# 'depth' = max depth in metres from the Wikipedia infobox 'Max. depth' field (live-verified).
# 'passes' = (depth > THRESHOLD).  Nothing below is leaked into the task statement or the plan.

THRESHOLD: int = 480  # metres, strictly greater-than

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":      "matano",
        "name":     "Lake Matano",
        "country":  "Indonesia (South Sulawesi)",
        "depth":    590,
        "passes":   True,
        "name_rx":  r"matano",
        "depth_rx": r"(?<!\d)590(?!\d)",
        "slug_rx":  r"wiki/lake_matano",
    },
    {
        "key":      "hornindalsvatnet",
        "name":     "Hornindalsvatnet",
        "country":  "Norway",
        "depth":    514,
        "passes":   True,
        "name_rx":  r"hornindals",
        "depth_rx": r"(?<!\d)514(?!\d)",
        "slug_rx":  r"wiki/hornindalsvatnet",
    },
    {
        "key":      "quesnel",
        "name":     "Quesnel Lake",
        "country":  "Canada (British Columbia)",
        "depth":    511,
        "passes":   True,
        "name_rx":  r"quesnel",
        "depth_rx": r"(?<!\d)511(?!\d)",
        "slug_rx":  r"wiki/quesnel_lake",
    },
    {
        "key":      "sarez",
        # Wikipedia article title is "Sarez Lake" (Lake Sarez redirects there).
        "name":     "Sarez Lake",
        "country":  "Tajikistan",
        "depth":    505,
        "passes":   True,
        "name_rx":  r"sarez",
        "depth_rx": r"(?<!\d)505(?!\d)",
        "slug_rx":  r"wiki/sarez_lake|wiki/lake_sarez",
    },
    {
        "key":      "tinnsja",
        # Wikipedia article title is "Tinnsjå" (å, not ø) — both spellings are findable.
        "name":     "Tinnsjå",
        "country":  "Norway",
        "depth":    460,
        "passes":   False,
        "name_rx":  r"tinnsj",          # matches Tinnsjå, Tinnsjø, Tinnsjaa, etc.
        "depth_rx": r"(?<!\d)460(?!\d)",
        "slug_rx":  r"wiki/tinnsj",
    },
    {
        "key":      "manapouri",
        "name":     "Lake Manapouri",
        "country":  "New Zealand",
        "depth":    444,
        "passes":   False,
        "name_rx":  r"manapouri",
        "depth_rx": r"(?<!\d)444(?!\d)",
        "slug_rx":  r"wiki/lake_manapouri",
    },
    {
        "key":      "ohrid",
        "name":     "Lake Ohrid",
        "country":  "North Macedonia / Albania",
        "depth":    288,
        "passes":   False,
        "name_rx":  r"ohrid",
        "depth_rx": r"(?<!\d)288(?!\d)",
        "slug_rx":  r"wiki/lake_ohrid",
    },
]

PASSING: List[Dict[str, Any]] = [e for e in ENTITIES if e["passes"]]   # 4 lakes
FAILING: List[Dict[str, Any]] = [e for e in ENTITIES if not e["passes"]]  # 3 lakes
KEYSTONE_COUNT: int = len(PASSING)   # 4

# Numeric-token extractor: pulls maximal digit-runs (with internal grouping commas).
# _int_values then keeps only PLAIN integers (decimal-point tokens are dropped).
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


# ── metadata ─────────────────────────────────────────────────────────────────────────────────────

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "072",
        "test_name":        "Tier 5: Count-with-condition — lakes with max depth > 480 m (7 obscure lakes)",
        "difficulty_level": "8/10",
        "category":         "Count-with-condition (page-only lake depths, 7-way fan-out)",
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
        "guess from memory). For EACH of the following seven lakes, open the lake's Wikipedia "
        "page and read ONE figure directly from the infobox: its MAXIMUM DEPTH in metres "
        "(the 'Max. depth' field — the deepest measured point of the lake; NOT its surface "
        "elevation, NOT its average depth):\n"
        f"{listing}\n\n"
        f"Then COUNT how many of the seven lakes have a maximum depth STRICTLY GREATER THAN "
        f"{THRESHOLD} metres and report:\n"
        f"  (a) the count — how many of the seven lakes have max depth > {THRESHOLD} m "
        "(a single integer 0–7; this is the keystone answer and belongs in your primary answer),\n"
        "  (b) which specific lakes exceed that depth and which do not (two named lists),\n"
        "  (c) each lake's maximum depth in metres as read from its infobox, and\n"
        "  (d) the exact source URL for every Wikipedia page you read.\n\n"
        "IMPORTANT: do not guess depths from memory or from a lake's general fame. "
        "You MUST open each lake's Wikipedia page and read the infobox 'Max. depth' figure."
    )


# ── deliverables / success criteria ──────────────────────────────────────────────────────────────

def get_required_deliverables() -> List[str]:
    return [
        f"Count of lakes with maximum depth > {THRESHOLD} m (the keystone integer, 0–7)",
        f"Which lakes exceed {THRESHOLD} m and which do not (passing and failing lake names)",
        "Each lake's maximum depth in metres (seven individual infobox values)",
        "Source URL for each lake's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    passing_names = ", ".join(e["name"] for e in PASSING)
    failing_names = ", ".join(e["name"] for e in FAILING)
    return [
        "At least 6 pages visited (target 7, one per lake)",
        f"Correctly reports the count as {KEYSTONE_COUNT} "
        f"(the number of lakes with max depth > {THRESHOLD} m)",
        f"Identifies the {len(PASSING)} passing lakes: {passing_names}",
        f"Identifies the {len(FAILING)} failing lakes: {failing_names}",
        "Reports all seven lakes' maximum depths from their infoboxes",
        "Cites a Wikipedia source URL for each lake",
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
            continue  # drop decimals like "444.5"
        try:
            vals.append(int(tok.replace(",", "")))
        except ValueError:
            continue
    return vals


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] contains the integer KEYSTONE_COUNT (= 4).

    Checks that the primary answer slot holds exactly the correct count of lakes exceeding
    the threshold. Count=7 ('all of them'), count=3 (one lake dropped), and count=5 (one extra
    lake wrongly included) all fail — only the computed value 4 passes.

    Also requires GROUNDING: the value string alone is insufficient — the agent must have
    actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess would earn credit.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return KEYSTONE_COUNT in _int_values(_primary_text(result))


# ── validation functions ──────────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a seven-way fan-out wants one page per lake."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check":  "visit_count",
        "passed": n >= 6,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": f"{n} visit(s) (target >= {len(ENTITIES)}: one page per lake; >= 6 to pass)",
    }


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact count of lakes with max depth > 480 m, reported as a
    single integer in deliverables[0].

    Correct answer = 4.  Any other value (3, 5, 7, …) fails.  A model that guesses 'all 7'
    scores 0; a model that drops a lake and reports 3 also scores 0.  Only a model that opens
    all seven pages, reads each depth, applies the threshold, and counts correctly passes.
    """
    passed = _keystone_ok(result, observability)
    passing_names = ", ".join(e["name"] for e in PASSING)
    return {
        "check":  "keystone_count",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"Count {KEYSTONE_COUNT} (lakes with max depth > {THRESHOLD} m) present in primary answer"
            if passed else
            f"Count {KEYSTONE_COUNT} absent or wrong in primary answer. "
            f"The {len(PASSING)} lakes exceeding {THRESHOLD} m are: {passing_names}. "
            f"Beware: a naive 'all seven pass' gives count=7 (wrong); dropping one lake gives "
            f"count=3 (wrong); including Tinnsjå (460 m) as a passer gives count=5 (wrong)."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the seven (lake, depth) pairs gathered.

    A lake is credited only when BOTH its name AND its infobox depth appear together in the
    response. The seven depth values (590, 514, 511, 505, 460, 444, 288) are all distinct,
    so there is no cross-crediting. Deliberately NOT gated on the keystone — this axis measures
    whether the agent actually fanned out to all seven pages even if it miscounts or misapplies
    the threshold.

    Credit is CAPPED BY visit count (``min(hits, n_visits)``) so a 0-visit parametric-memory
    answer that merely recalls the figures cannot bank partial credit here without ever browsing.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE)
        and re.search(e["depth_rx"], text)
    ]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(ENTITIES)
    return {
        "check":  "coverage",
        "passed": credited == n,
        "score":  credited / n,
        "reason": (
            f"{credited}/{n} (lake, depth) pairs gathered "
            f"({', '.join(hits[:credited]) if credited else 'none'}; "
            f"{len(hits)} text-matched, {n_visits} visit(s))"
        ),
    }


def validate_passing_lakes(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all four passing lake names present in the answer.

    Short-circuits to 0 when the keystone is absent, so a run that reports the wrong count
    cannot bank partial credit for naming some of the correct lakes. After the keystone gate,
    checks that Matano, Hornindalsvatnet, Quesnel Lake, and Sarez Lake all appear in the text.
    """
    if not _keystone_ok(result, observability):
        return {
            "check":  "passing_lakes",
            "passed": False,
            "score":  0.0,
            "reason": "Keystone absent -> passing-lake list not credited",
        }
    text = _all_text(result)
    hits = [e["name"] for e in PASSING if re.search(e["name_rx"], text, re.IGNORECASE)]
    n = len(PASSING)
    return {
        "check":  "passing_lakes",
        "passed": len(hits) == n,
        "score":  len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} passing lake names identified "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least 5 of the 7 lake Wikipedia pages.

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
        "reason": f"{cited}/{n} lake pages cited (>= 5 to pass)",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_count,
        validate_coverage,
        validate_passing_lakes,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 062/070.
    return None


# ── compiled plan ─────────────────────────────────────────────────────────────────────────────────

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SEVEN INDEPENDENT parallel leaves: one per lake, each fetching that lake's maximum depth
    in metres from its Wikipedia infobox. ALL comparison and counting logic lives ONLY in the
    aggregation step, which is forced to restate each entity → its depth value explicitly
    BEFORE applying the threshold or counting. This structural requirement:
      (a) prevents the cheap executor from hallucinating a count without visiting all pages;
      (b) ensures the threshold comparison is applied entity-by-entity, not globally guessed;
      (c) lets a diverse-grounding reranker catch any depth misattribution.

    Encodes STRUCTURE only: names the seven given lakes and their countries, but leaks no depth
    value and never states the count or which lakes pass.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_depth",
            "instruction": (
                f"Open the Wikipedia article for {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its MAXIMUM DEPTH in metres — the 'Max. depth' "
                "field (the deepest point of the lake). Report ONLY that single depth figure "
                "(a whole number in metres) and the exact source URL. "
                "Do NOT report surface elevation or average depth. Do not guess from memory."
            ),
            "expect": (
                f"MAXIMUM DEPTH of {e['name']} in metres (a single integer) -- source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        # Deterministic composition: the executor applies the threshold and counts in Python over
        # the seven gathered depths, rendering each lake's check and both named lists itself (zero
        # extra LLM calls) — free-text counting is the confirmed failure mode here even when every
        # depth is correct. Encodes the GIVEN threshold only; no depth value and no count. If any
        # leaf fails to resolve, the composer returns nothing and the recipe below runs unchanged.
        "agg_mode": "computed",
        "composition": {
            "op": "count_threshold",
            "answer_noun": "lake",
            "value_label": "maximum depth",
            "unit": "m",
            "comparator": ">",
            "threshold": THRESHOLD,
            "items": [{"leaf": f"{e['key']}_depth", "label": e["name"], "type": "number"}
                      for e in ENTITIES],
        },
        "aggregation": (
            "You now have, for each of the seven lakes, its maximum depth in metres and a "
            "source URL. Before applying any threshold or counting, RESTATE each lake's depth "
            "explicitly in this format:\n"
            "  Lake Matano → [depth] m\n"
            "  Hornindalsvatnet → [depth] m\n"
            "  Quesnel Lake → [depth] m\n"
            "  Sarez Lake → [depth] m\n"
            "  Tinnsjå → [depth] m\n"
            "  Lake Manapouri → [depth] m\n"
            "  Lake Ohrid → [depth] m\n"
            "(substituting the integer you retrieved for each '[depth]' placeholder).\n\n"
            "Then, FOR EACH LAKE IN TURN, state whether its depth is strictly greater than "
            "480 metres. Finally, COUNT the number of lakes whose depth exceeds 480 m — "
            "that integer (0–7) is the keystone answer. Report:\n"
            "  (a) the count (a single integer, 0–7) as the primary keystone answer,\n"
            "  (b) which lakes exceed 480 m and which do not (two named lists),\n"
            "  (c) the full list of seven depths as you restated them above, and\n"
            "  (d) the source URL for each lake."
        ),
    }
