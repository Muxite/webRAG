"""
Test 090: Tier 5 (graph) — COUNT-WITH-CONDITION over page-only Icelandic road tunnel lengths.
Level: graph   Weight: long   Difficulty: 8/10

Among six solid-but-obscure Icelandic road tunnels, the agent must open each tunnel's Wikipedia
page, read its total LENGTH in metres from the infobox, and report HOW MANY of the six have a
length strictly greater than 4,500 metres. The answer (a COUNT, integer 0–6) is COMPUTED from six
independent page-only lookups and is not directly recallable.

WHY IT DISCRIMINATES (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap parametric (no tools): cannot simultaneously know six obscure Icelandic tunnel lengths;
    either guesses the wrong count, claims "all six" (the naïve overcount), or refuses.
  * cheap native (graph): may open some pages but drops one or more tunnels, misattributes
    lengths, or applies the threshold check inconsistently; count likely off by 1–2.
  * graph_compiled: six parallel leaves each fetch ONE tunnel's length from ONE Wikipedia page;
    the aggregation is forced to restate each entity→value mapping before comparing to the
    threshold and summing the boolean outcomes; the cheap executor is rescued from holding six
    values simultaneously and a diverse-grounding reranker can catch any threshold misapplication.

Ground truth (verified against live English Wikipedia, 2026-07-07 — each tunnel's Wikipedia
article infobox 'Length' field, read from the canonical article URL; Wikipedia expresses several
values in km, the metre equivalent is recorded here):

  tunnel                    Wikipedia slug                              length (infobox)   > 4,500 m?
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
  Vaðlaheiðargöng           /wiki/Va%C3%B0lahei%C3%B0arg%C3%B6ng        7.4 km = 7,400 m     YES
  Hvalfjörður Tunnel        /wiki/Hvalfj%C3%B6r%C3%B0ur_Tunnel          5,770 m              YES
  Dýrafjarðargöng           /wiki/D%C3%BDrafjar%C3%B0arg%C3%B6ng        5,600 m              YES
  Bolungarvíkurgöng         /wiki/Bolungarv%C3%ADkurg%C3%B6ng           5.426 km = 5,426 m   YES
  Múlagöng                  /wiki/M%C3%BAlag%C3%B6ng                    3.4 km = 3,400 m     NO
  Almannaskarðsgöng         /wiki/Almannaskar%C3%B0sg%C3%B6ng           1.3 km = 1,300 m     NO
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
  KEYSTONE = count of tunnels with length > 4,500 m = 4

THRESHOLD MARGINS (each length clears or misses 4,500 m by ≥ 926 m — no borderline value):
  PASS  Vaðlaheiðargöng      7,400 − 4,500 = +2,900 m   (largest above-threshold margin)
  PASS  Hvalfjörður Tunnel   5,770 − 4,500 = +1,270 m
  PASS  Dýrafjarðargöng      5,600 − 4,500 = +1,100 m
  PASS  Bolungarvíkurgöng    5,426 − 4,500 =   +926 m   (smallest above-threshold margin)
  FAIL  Múlagöng             4,500 − 3,400 = −1,100 m   (smallest below-threshold margin)
  FAIL  Almannaskarðsgöng    4,500 − 1,300 = −3,200 m   (largest below-threshold margin)

  Every length is ≥ 926 m from the 4,500 m threshold, so no plausible single-figure extraction
  error (a misread of a 4-digit metre value or a decimal km value) can flip any tunnel's
  pass/fail status. The nearest failing tunnel (Múlagöng, 3,400 m) sits a full 1,100 m below the
  threshold, so an extractor cannot accidentally promote it.

THRESHOLD/COUNT COLLISION AVOIDED: the threshold is 4,500 m (4.5 km). As an integer token it reads
  4500 (never a bare 4), and the km form 4.5 is a decimal token that the keystone extractor drops.
  The keystone count (4) therefore never collides with the threshold value in the primary answer.

ANTI-PARAMETRIC: the six tunnels are all Icelandic and uniformly obscure:
  • Vaðlaheiðargöng (7,400 m) is a tolled tunnel through Vaðlaheiði east of Akureyri, opened 2018.
  • Hvalfjörður Tunnel (5,770 m) is a subsea tunnel beneath Hvalfjörður north of Reykjavík (1998).
  • Dýrafjarðargöng (5,600 m) links Arnarfjörður and Dýrafjörður in the Westfjords, opened 2020.
  • Bolungarvíkurgöng (5,426 m) connects Bolungarvík and Ísafjörður in the Westfjords (2010).
  • Múlagöng (3,400 m) pierces Ólafsfjarðarmúli near Ólafsfjörður in North Iceland (1991).
  • Almannaskarðsgöng (1,300 m) is a short tunnel near Höfn in Southeast Iceland (2005).
  Their exact lengths are page-specific technical specifications not simultaneously recallable, and
  the COUNT (4) is not a published fact anywhere. A cheap parametric model has no reliable path to
  the correct count without opening all six pages.

SELF-VERIFY (throwaway snippet — delete after):
    from services.agent.app.idea_tests.test_090_tier5_tunnel_count_iceland import (
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
    # DECOY count=6 (naive 'all') -> keystone FAILS
    decoy_6 = {"deliverables": ["6"], "output": {"final_deliverable": "all 6"}}
    assert not validate_keystone_count(decoy_6, {})["passed"], "count=6 should fail"
    print("All self-verify assertions passed.")

KEYSTONE = the count 4 (exact integer in deliverables[0]).
Secondary (gated) = lists all four passing tunnel names (Vaðlaheiðargöng, Hvalfjörður Tunnel,
  Dýrafjarðargöng, Bolungarvíkurgöng).
Coverage (ungated) = how many of the six (tunnel, length-value) pairs gathered; the six lengths
  (7,400 / 5,770 / 5,600 / 5,426 / 3,400 / 1,300 m) are all distinct — coverage is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified fixtures ────────────────────────────────────────────────────────────────────────────
# 'length_m' = tunnel total length in metres from the Wikipedia article infobox 'Length' field
# (live-verified 2026-07-07).  Wikipedia may express the value in km — the metre equivalent is
# recorded here.  'passes' = (length_m > THRESHOLD).  Nothing below is leaked into the task
# statement or the plan.

THRESHOLD: int = 4500  # metres, strictly greater-than (4.5 km)

ENTITIES: List[Dict[str, Any]] = [
    {
        "key":      "vadlaheidi",
        "name":     "Vaðlaheiðargöng",
        "region":   "North Iceland (near Akureyri)",
        "length_m": 7400,
        "passes":   True,
        # Wikipedia infobox 'Length' field shows 7.4 km. Equivalent: 7,400 m.
        "name_rx":  r"va.lahei",           # matches Vaðlaheiði / Vadlaheidi / Vaðlaheiðargöng
        "len_rx":   r"(?<!\d)7[.,\s]?4(?:00)?(?!\d)",   # 7.4 / 7,400 / 7400
        "slug_rx":  r"wiki/va.lahei|wiki/va%c3",
    },
    {
        "key":      "hvalfjordur",
        "name":     "Hvalfjörður Tunnel",
        "region":   "Southwest Iceland (near Reykjavík)",
        "length_m": 5770,
        "passes":   True,
        # Wikipedia infobox shows 5,770 m (18,930 ft). Article title uses accented characters;
        # both the accented slug and the '_Tunnel' English slug are matched.
        "name_rx":  r"hvalf",
        "len_rx":   r"(?<!\d)5[.,\s]?77\d?(?!\d)",       # 5.77 / 5,770 / 5770
        "slug_rx":  r"wiki/hvalf",
    },
    {
        "key":      "dyrafjordur",
        "name":     "Dýrafjarðargöng",
        "region":   "Westfjords",
        "length_m": 5600,
        "passes":   True,
        # Wikipedia infobox shows 5,600 metres. Article text: 5.6 km.
        "name_rx":  r"d.rafjar",           # matches Dýrafjar / Dyrafjar / Dýrafjarðargöng
        "len_rx":   r"(?<!\d)5[.,\s]?6(?:00)?(?!\d)",    # 5.6 / 5,600 / 5600
        "slug_rx":  r"wiki/d.rafjar|wiki/d%c3%bdrafjar",
    },
    {
        "key":      "bolungarvik",
        "name":     "Bolungarvíkurgöng",
        "region":   "Westfjords",
        "length_m": 5426,
        "passes":   True,
        # Wikipedia infobox shows 5.426 km. Article text: 5,426 m (17,802 ft).
        "name_rx":  r"bolungarv",
        "len_rx":   r"(?<!\d)5[.,\s]?426(?!\d)",         # 5.426 / 5,426 / 5426
        "slug_rx":  r"wiki/bolungarv",
    },
    {
        "key":      "mulagong",
        "name":     "Múlagöng",
        "region":   "North Iceland (near Ólafsfjörður)",
        "length_m": 3400,
        "passes":   False,
        # Wikipedia infobox shows 3.4 km. Article text: 3,400 m (11,155 ft).
        "name_rx":  r"m.lag",              # matches Múlag / Mulag / Múlagöng
        "len_rx":   r"(?<!\d)3[.,\s]?4(?:00)?(?!\d)",    # 3.4 / 3,400 / 3400
        "slug_rx":  r"wiki/m.lag|wiki/m%c3%balag",
    },
    {
        "key":      "almannaskard",
        "name":     "Almannaskarðsgöng",
        "region":   "Southeast Iceland (near Höfn)",
        "length_m": 1300,
        "passes":   False,
        # Wikipedia infobox shows 1.3 km; article text notes 1,312 m (a sign rounds to 1,300 m).
        "name_rx":  r"almannaskar",
        "len_rx":   r"(?<!\d)1[.,\s]?3(?:00|12)?(?!\d)", # 1.3 / 1,300 / 1300 / 1,312 / 1312
        "slug_rx":  r"wiki/almannaskar",
    },
]

PASSING: List[Dict[str, Any]] = [e for e in ENTITIES if e["passes"]]     # 4 tunnels
FAILING: List[Dict[str, Any]] = [e for e in ENTITIES if not e["passes"]]  # 2 tunnels
KEYSTONE_COUNT: int = len(PASSING)   # 4

# Numeric-token extractor: pulls maximal digit-runs (with internal grouping commas).
# _int_values then keeps only PLAIN integers (decimal-point tokens are dropped).
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


# ── metadata ─────────────────────────────────────────────────────────────────────────────────────

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "090",
        "test_name":        "Tier 5: Count-with-condition — Icelandic tunnels with length > 4,500 m (6 obscure tunnels)",
        "difficulty_level": "8/10",
        "category":         "Count-with-condition (page-only Icelandic tunnel lengths, 6-way fan-out)",
        "level":            "graph",
        "weight":           "long",
    }


# ── task statement ────────────────────────────────────────────────────────────────────────────────

def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['region']})"
        for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not "
        "guess from memory). For EACH of the following six Icelandic road tunnels, open the "
        "tunnel's Wikipedia page and read ONE figure directly from the infobox: its total "
        "LENGTH in metres (the total length of the tunnel bore from portal to portal). "
        "Wikipedia may express the figure in km; convert to metres if so (1 km = 1,000 m):\n"
        f"{listing}\n\n"
        f"Then COUNT how many of the six tunnels have a length STRICTLY GREATER THAN "
        f"{THRESHOLD:,} metres (4.5 km) and report:\n"
        f"  (a) the count — how many of the six tunnels have length > {THRESHOLD:,} m "
        "(a single integer 0–6; this is the keystone answer and belongs in your primary answer),\n"
        "  (b) which specific tunnels exceed that length and which do not (two named lists),\n"
        "  (c) each tunnel's length in metres as read from its Wikipedia page, and\n"
        "  (d) the exact source URL for every Wikipedia page you read.\n\n"
        "IMPORTANT: do not guess lengths from memory or from a tunnel's general fame. "
        "You MUST open each tunnel's Wikipedia page and read the infobox length figure."
    )


# ── deliverables / success criteria ──────────────────────────────────────────────────────────────

def get_required_deliverables() -> List[str]:
    return [
        f"Count of tunnels with length > {THRESHOLD:,} m (the keystone integer, 0–6)",
        f"Which tunnels exceed {THRESHOLD:,} m and which do not (passing and failing tunnel names)",
        "Each tunnel's length in metres (six individual infobox values)",
        "Source URL for each tunnel's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    passing_names = ", ".join(e["name"] for e in PASSING)
    failing_names = ", ".join(e["name"] for e in FAILING)
    return [
        "At least 5 pages visited (target 6, one per tunnel)",
        f"Correctly reports the count as {KEYSTONE_COUNT} "
        f"(the number of tunnels with length > {THRESHOLD:,} m)",
        f"Identifies the {len(PASSING)} passing tunnels: {passing_names}",
        f"Identifies the {len(FAILING)} failing tunnels: {failing_names}",
        "Reports all six tunnels' lengths from their Wikipedia pages",
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
            continue  # drop decimals like "5.426"
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

    Checks that the primary answer slot holds exactly the correct count of tunnels exceeding the
    threshold. Count=6 ('all of them'), count=3 (one tunnel dropped), and count=5 (one failing
    tunnel wrongly counted) all fail — only the computed value 4 passes. The threshold token
    4,500 reads as 4500 (never a bare 4) and its km form 4.5 is dropped as a decimal, so the
    threshold cannot masquerade as the count.

    Also requires GROUNDING: the value string alone is insufficient — the agent must have
    actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess would earn credit.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return KEYSTONE_COUNT in _int_values(_strip_list_markers(_primary_text(result)))


# ── validation functions ──────────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per tunnel."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check":  "visit_count",
        "passed": n >= 5,
        "score":  min(1.0, n / len(ENTITIES)),
        "reason": f"{n} visit(s) (target >= {len(ENTITIES)}: one page per tunnel; >= 5 to pass)",
    }


def validate_keystone_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the exact count of tunnels with length > 4,500 m, reported as a
    single integer in deliverables[0].

    Correct answer = 4.  Any other value (3, 5, 6, …) fails.  A model that guesses 'all 6' scores
    0; a model that drops Bolungarvíkurgöng (5,426 m, the closest to threshold) and reports 3 also
    scores 0.  Only a model that opens all six pages, reads each length, converts km to metres
    where necessary, applies the threshold, and counts correctly passes.
    """
    passed = _keystone_ok(result, observability)
    passing_names = ", ".join(e["name"] for e in PASSING)
    return {
        "check":  "keystone_count",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"Count {KEYSTONE_COUNT} (tunnels with length > {THRESHOLD:,} m) present in "
            f"primary answer"
            if passed else
            f"Count {KEYSTONE_COUNT} absent or wrong in primary answer. "
            f"The {len(PASSING)} tunnels exceeding {THRESHOLD:,} m are: {passing_names}. "
            f"Beware: guessing 'all six pass' gives count=6 (wrong); dropping Bolungarvíkurgöng "
            f"(5,426 m, the nearest above-threshold tunnel) gives count=3 (wrong); miscounting "
            f"Múlagöng (3,400 m) as passing gives count=5 (wrong)."
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the six (tunnel, length) pairs gathered.

    A tunnel is credited only when BOTH its name AND its length value appear together in the
    response. The six lengths (7,400 / 5,770 / 5,600 / 5,426 / 3,400 / 1,300 m) are all distinct,
    so there is no cross-crediting. Deliberately NOT gated on the keystone — this axis measures
    whether the agent fanned out to all six pages even if it miscounts or misapplies the threshold.

    Credit is CAPPED BY visit count (``min(hits, n_visits)``) so a 0-visit parametric-memory
    answer that merely recalls the figures cannot bank partial credit here without ever browsing.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE)
        and re.search(e["len_rx"], text)
    ]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(ENTITIES)
    return {
        "check":  "coverage",
        "passed": credited == n,
        "score":  credited / n,
        "reason": (
            f"{credited}/{n} (tunnel, length) pairs gathered "
            f"({', '.join(hits[:credited]) if credited else 'none'}; "
            f"{len(hits)} text-matched, {n_visits} visit(s))"
        ),
    }


def validate_passing_tunnels(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: all four passing tunnel names present in the answer.

    Short-circuits to 0 when the keystone is absent, so a run that reports the wrong count cannot
    bank partial credit for naming some of the correct tunnels. After the keystone gate, checks
    that Vaðlaheiðargöng, Hvalfjörður Tunnel, Dýrafjarðargöng, and Bolungarvíkurgöng all appear.
    """
    if not _keystone_ok(result, observability):
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
    """GATED secondary: cites at least 4 of the 6 tunnel Wikipedia pages.

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
        "passed": cited >= 4,
        "score":  cited / n,
        "reason": f"{cited}/{n} tunnel pages cited (>= 4 to pass)",
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
    # None -> the harness applies its default structured rubric judge, as in 062/070/072/087.
    return None


# ── compiled plan ─────────────────────────────────────────────────────────────────────────────────

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves: one per tunnel, each fetching that tunnel's total length in
    metres from its Wikipedia page. ALL comparison and counting logic lives ONLY in the
    aggregation step, which is forced to restate each entity → its length value explicitly BEFORE
    applying the threshold or counting. This structural requirement:
      (a) prevents the cheap executor from hallucinating a count without visiting all pages;
      (b) ensures the threshold comparison is applied tunnel-by-tunnel, not globally guessed;
      (c) lets a diverse-grounding reranker catch any length misattribution.

    Each leaf's ``expect`` restates its own tunnel name so the aggregator (whose facts_block strips
    leaf IDs) can bind each length back to the right entity.

    Encodes STRUCTURE only: names the six given tunnels and their regions and the given threshold,
    but leaks no length value and never states the count or which tunnels pass.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_length",
            "instruction": (
                f"Open the Wikipedia article for the Icelandic road tunnel {e['name']} "
                f"({e['region']}) and read, directly from the infobox, its total LENGTH — the "
                "tunnel bore length from portal to portal (NOT highway length, NOT approach roads). "
                "Wikipedia may express this in km; if so, convert to metres (1 km = 1,000 m). "
                f"Report your answer as '{e['name']}: length = <number> m' followed by the exact "
                "source URL. Do not guess from memory."
            ),
            "expect": (
                f"{e['name']}: total length in metres (a single number) -- source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six Icelandic tunnels, its total length in metres and a "
            "source URL. Before applying any threshold or counting, RESTATE each tunnel's length "
            "explicitly in this format:\n"
            "  Vaðlaheiðargöng → [length] m\n"
            "  Hvalfjörður Tunnel → [length] m\n"
            "  Dýrafjarðargöng → [length] m\n"
            "  Bolungarvíkurgöng → [length] m\n"
            "  Múlagöng → [length] m\n"
            "  Almannaskarðsgöng → [length] m\n"
            "(substituting the number of metres you retrieved for each '[length]' placeholder).\n\n"
            "Then, FOR EACH TUNNEL IN TURN, state whether its length is strictly greater than "
            "4,500 metres. Finally, COUNT the number of tunnels whose length exceeds 4,500 m — "
            "that integer (0–6) is the keystone answer. Report:\n"
            "  (a) the count (a single integer, 0–6) as the primary keystone answer,\n"
            "  (b) which tunnels exceed 4,500 m and which do not (two named lists),\n"
            "  (c) the full list of six lengths as you restated them above, and\n"
            "  (d) the source URL for each tunnel."
        ),
    }
