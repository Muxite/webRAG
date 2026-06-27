"""
Test 070: Tier 5 (graph) — BOUNDED SUBSET-SUM with an AGGREGATE DISTRACTOR.
Level: graph   Weight: long   Difficulty: 9/10

A quantitative-reasoning task engineered around the cheap-model shortcut of "grab the big salient
number printed on the page". The agent must SUM a per-item integer across a DEFINED SUBSET of items
while a MISLEADING larger aggregate is printed prominently nearby. Concretely: sum the episode
counts of the FIRST FOUR seasons of the TV series *Chuck* — while the main series page's infobox
prints a single 'No. of episodes' figure that counts ALL FIVE seasons (the whole-series total).
That whole-series total is the DISTRACTOR: it is larger than the requested partial sum because it
also folds in the fifth season, so a naive read that grabs the infobox number answers the wrong
question.

*Chuck* (NBC, 2007-2012) is deliberately SOLID-BUT-NOT-ICONIC (a cult action-comedy, not a
Friends / Lost / Breaking Bad household-name), it is FINISHED (its episode counts are closed,
stable quantities that will never change), and crucially the requested partial sum is NOT a
recallable fact: nobody memorizes "the first four seasons of Chuck total 78 episodes", whereas the
whole-series total (91) is the comparatively memorable aggregate — which is exactly the decoy.

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): grabs the salient infobox total (91, the distractor), or reads the
    season table but drops a season / mis-adds -> wrong number.
  * frontier sequential (ReAct): reads the per-season figures, sums the first four -> decent.
  * graph_compiled: four parallel leaves each fetch ONE season's episode count from that season's
    own article (routing AROUND the whole-series total that lives on the main page); the
    aggregation owns the summation and is forced to WRITE OUT the addition (a+b+c+d=T) before
    concluding -> the cheap executor is rescued and a diverse-grounding reranker can re-derive it.

Ground truth (verified against live English Wikipedia, 2026-06-27 — each season's infobox
'No. of episodes' field on its own per-season article; the whole-series total from the main
series page infobox):

  item (subset)        episodes   source slug
  Chuck season 1          13      en.wikipedia.org/wiki/Chuck_(season_1)
  Chuck season 2          22      en.wikipedia.org/wiki/Chuck_(season_2)
  Chuck season 3          19      en.wikipedia.org/wiki/Chuck_(season_3)
  Chuck season 4          24      en.wikipedia.org/wiki/Chuck_(season_4)
  --------------------------------------------------------------------------
  KEYSTONE = first-four-season SUM = 13 + 22 + 19 + 24 = 78   <- the answer
  (excluded) Chuck season 5         13      en.wikipedia.org/wiki/Chuck_(season_5)
  DISTRACTOR = whole-series infobox 'No. of episodes' = 91     en.wikipedia.org/wiki/Chuck_(TV_series)
                                                              (= 78 + 13, all five seasons)

  KEYSTONE TOTAL = 78.  DISTRACTOR = 91.  They DIFFER by a clear margin of 13 (the fifth season's
  count), i.e. the printed aggregate is ~16.7% larger than the requested partial sum. The four
  subset integers (13, 22, 19, 24) are DISTINCT among themselves, so the coverage diagnostic is
  collision-free.

ROBUSTNESS / tolerance band:
  The validator accepts the keystone in the TIGHT band [77, 79] = 78 +/- 1 — the exact partial sum,
  forgiving a single +/-1 misread on one season. That band:
    * ACCEPTS the true total 78 (and a one-off +/-1 slip -> 77 or 79);
    * REJECTS the DISTRACTOR 91 (whole-series total) by a margin of 12 from the band edge;
    * REJECTS every DROP-ONE partial sum (78-13=65, 78-22=56, 78-19=59, 78-24=54) — each at least
      12 below the band, so omitting any single season is detectable;
    * REJECTS the first-three sum (13+22+19=54) and the all-five sum (91).
  Because every season count is >= 13, dropping a season moves the total by >= 13 (far outside the
  +/-1 band) and a +/-2 misread on a single season (-> 76 or 80) is likewise detected; only a
  genuine first-four sum (modulo a single +/-1 slip) lands in band. The only other 4-of-5 subset
  that also sums to 78 is "seasons 2-5" (22+19+24+13), a degenerate coincidence because season 1
  and season 5 share the count 13 — but that wrong subset still lands on the correct number, so it
  is not a discrimination leak; the genuine decoys (the printed aggregate and any dropped season)
  are all firmly rejected.

ANTI-PARAMETRIC: the whole-series total (91) is the comparatively memorable / guessable number and
is the DECOY; the keystone (78) is a computed partial sum that is not a recallable fact, so a model
that pattern-matches a salient quantity from memory mis-answers and a correct answer requires
actually reading the four per-season figures and adding them.

  KEYSTONE = the computed first-four-season SUM (78, exact-match within +/- 1). Secondary (gated) =
  the four per-season episode counts. Coverage (ungated) = how many of the four counts were
  gathered. The four subset integers are distinct, so coverage is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# Each item is ONE season of *Chuck*; ``eps`` is that season's infobox 'No. of episodes' figure.
# Nothing here (no per-season figure, no total) is leaked into the task statement or the plan.
SEASONS: List[Dict[str, Any]] = [
    {"key": "s1", "label": "Season 1", "num": 1, "eps": 13, "slug_rx": r"chuck_\(season_1\)"},
    {"key": "s2", "label": "Season 2", "num": 2, "eps": 22, "slug_rx": r"chuck_\(season_2\)"},
    {"key": "s3", "label": "Season 3", "num": 3, "eps": 19, "slug_rx": r"chuck_\(season_3\)"},
    {"key": "s4", "label": "Season 4", "num": 4, "eps": 24, "slug_rx": r"chuck_\(season_4\)"},
]

KEYSTONE_TOTAL = 78           # 13 + 22 + 19 + 24 — the first-four-season subset sum
DISTRACTOR_TOTAL = 91         # whole-series infobox 'No. of episodes' (all five seasons; = 78 + 13)
EXCLUDED_SEASON_EPS = 13      # Chuck season 5 — folded into the distractor but NOT the keystone
KEYSTONE_BAND = {77, 78, 79}  # 78 +/- 1: accepts the exact sum (and a single +/-1 misread)

# Numeric-token extractor: grabs each maximal digit run together with any internal decimal/grouping
# separators, but bounded by digits on both ends (so a trailing sentence period in "78." is NOT
# swallowed and the integer 78 is still recovered). ``_int_values`` then keeps only PLAIN integers:
# tokens with a decimal point ("78.5") are dropped, grouping commas are stripped ("24,000" -> 24000),
# so years ("2019"), decimals and grouped counts can never masquerade as the keystone or a figure.
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "070",
        "test_name": "Tier 5: Bounded subset-sum with aggregate distractor (first-four-season episodes)",
        "difficulty_level": "9/10",
        "category": "Quantitative reasoning + bounded subset-sum (aggregate distractor)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {s['num']}. Chuck {s['label']}" for s in SEASONS)
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). The American television series 'Chuck' (NBC, 2007-2012) ran for FIVE "
        "seasons. This task concerns ONLY its FIRST FOUR seasons:\n"
        f"{listing}\n\n"
        "For EACH of seasons 1 through 4, open that season's own Wikipedia article (e.g. "
        "'Chuck (season 1)') and read, from the infobox 'No. of episodes' field, the number of "
        "episodes in THAT season alone. Then ADD UP the episode counts of those four seasons and "
        "report the SUM — the combined number of episodes across the first four seasons only.\n\n"
        "IMPORTANT (read carefully): the main 'Chuck (TV series)' page prints, in its infobox, a "
        "single 'No. of episodes' figure that counts ALL FIVE seasons (the whole-series total). "
        "That whole-series total is NOT the answer — it is larger than the figure you need because "
        "it also includes the fifth season. Do not report it; you must add up ONLY the first four "
        "seasons yourself.\n\n"
        "Report (a) the sum of the episode counts of the first four seasons (a single integer — the "
        "keystone), (b) the four individual per-season episode counts you looked up, and (c) the "
        "exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Sum of the episode counts of the first FOUR seasons of Chuck (the computed keystone total)",
        "The four individual per-season episode counts (the looked-up figures)",
        "Source URL for each season's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (one per season article; target 4 — a four-way fan-out)",
        "Correctly reports the first-four-season sum (78 = 13 + 22 + 19 + 24), NOT the whole-series "
        "total (91), and NOT a partial sum that drops a season",
        "Gathers all four per-season episode counts (13, 22, 19, 24)",
        "Cites the season source pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text. Prefer ``deliverables[0]`` (the contract's primary slot) when present;
    otherwise fall back to ``output.final_deliverable``."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: the final deliverable plus every deliverable slot, so coverage / figure /
    citation checks can see numbers the agent placed outside the primary answer slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _int_values(text: str) -> List[int]:
    """Plain integer values present in ``text``: each numeric token, with decimals dropped and
    grouping commas stripped (see ``_NUM_TOKEN_RX``)."""
    vals: List[int] = []
    for tok in _NUM_TOKEN_RX.findall(text):
        if "." in tok:
            continue  # a decimal like "78.5" is not a plain integer
        try:
            vals.append(int(tok.replace(",", "")))
        except ValueError:
            continue
    return vals


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: deliverables[0] reports the first-four-season SUM inside the tight band
    [77, 79] (= 78 +/- 1). The band ACCEPTS the exact total (and a single +/-1 misread) but REJECTS
    the printed whole-series aggregate (91, the distractor), every drop-one partial sum (<= 65) and
    the all-five sum (91)."""
    return any(n in KEYSTONE_BAND for n in _int_values(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a four-way fan-out wants one page per season."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: one article per season; >=3 to pass)"}


def validate_keystone_subset_sum(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the COMPUTED first-four-season subset sum, 78 = 13 + 22 + 19 + 24,
    reported in deliverables[0] within [77, 79]. A model that grabs the salient whole-series total
    (91, the aggregate distractor), drops a season, or mis-adds lands outside the band and fails."""
    passed = _keystone_ok(result)
    return {"check": "keystone_subset_sum", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("First-four-season sum (78 = 13+22+19+24) present in band [77,79]" if passed
                       else "First-four-season sum (78) missing/incorrect (beware the whole-series "
                            "total 91 — the aggregate distractor — and any dropped-season partial sum)")}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the FOUR per-season episode counts were
    gathered. The four subset integers (13, 22, 19, 24) are DISTINCT, so there is no cross-crediting
    (collision-free). Deliberately NOT gated on the keystone — it measures whether the agent fanned
    out to all four seasons and collected each figure even when it botches the addition or grabs the
    aggregate, the axis that separates a structured (multi-leaf) agent from a linear one that drops
    an item."""
    present = set(_int_values(_all_text(result)))
    hits = [s["label"] for s in SEASONS if s["eps"] in present]
    n = len(SEASONS)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} per-season episode counts gathered ({', '.join(hits) or 'none'})"}


def validate_perseason_figures(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: are ALL four per-season counts (13, 22, 19, 24) reported? Short-circuits to
    0 when the keystone is absent, so a run that grabs the distractor or mis-adds cannot bank credit
    for the per-item figures."""
    if not _keystone_ok(result):
        return {"check": "perseason_figures", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> per-season figures not credited"}
    present = set(_int_values(_all_text(result)))
    hits = [s["label"] for s in SEASONS if s["eps"] in present]
    n = len(SEASONS)
    return {"check": "perseason_figures", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} per-season figures reported ({', '.join(hits) or 'none'})"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the season pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for s in SEASONS if re.search(s["slug_rx"], text))
    n = len(SEASONS)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} season pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_subset_sum,
        validate_coverage,
        validate_perseason_figures,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), as in 055/059.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    FOUR INDEPENDENT parallel leaves: one per season in the subset, each fetching that single
    season's episode count from that season's OWN article — one atomic integer per leaf, which
    structurally routes AROUND the whole-series total that lives on the main series page (the
    aggregate distractor). ALL arithmetic — the four-term addition — lives only in the aggregation
    step, which is forced to WRITE OUT the addition explicitly (a + b + c + d = T) before concluding,
    so the cheap executor never grabs the salient aggregate or drops a term and a diverse-grounding
    reranker can re-derive and catch a slip. Encodes STRUCTURE only: it names the four GIVEN seasons
    but leaks no per-season figure, no whole-series total, and not the keystone sum.
    """
    leaves: List[Dict[str, Any]] = []
    for s in SEASONS:
        leaves.append({
            "id": f"{s['key']}_episodes",
            "instruction": (
                f"Open the Wikipedia article for the season of the TV series Chuck titled "
                f"'Chuck ({s['label'].lower()})' and read, from the infobox 'No. of episodes' "
                f"field, the number of episodes in THAT season alone. Report ONLY that single "
                f"integer and the source URL. Do NOT report the whole-series episode total, and do "
                f"not guess from memory."
            ),
            "expect": f"EPISODE COUNT for Chuck {s['label']} (a single integer) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of Chuck seasons 1, 2, 3 and 4, that single season's episode "
            "count (one integer per season). Write out the addition EXPLICITLY on one line in the "
            "form 'Season 1 + Season 2 + Season 3 + Season 4 = TOTAL', substituting the four "
            "integers you gathered, and compute the sum — do this BEFORE drawing any conclusion. "
            "Report that sum as the keystone answer: the combined episode count of the FIRST FOUR "
            "seasons only (a partial total). Do NOT report the whole-series episode count that "
            "spans all five seasons — report only the sum of these four. Also list the four "
            "per-season counts you used, and cite each season article's source URL."
        ),
    }
