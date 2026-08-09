"""
Test 083: Tier 5 (graph) — BOUNDED SUBSET-SUM with an AGGREGATE DISTRACTOR.
Level: graph   Weight: long   Difficulty: 9/10

A quantitative-reasoning task engineered around the cheap-model shortcut of "grab the big salient
number printed on the page". The agent must SUM the Althing seat counts of a DEFINED SUBSET of
parties from an obscure legislature election while the TOTAL CHAMBER SIZE (63) is printed
prominently on every page the agent reads — both on the main Althing article's infobox and as the
denominator in each party article's "N / 63" seat display. Concretely: sum the Althing seats of
the FOUR parties that won the most seats in the November 2024 Icelandic parliamentary election —
while every party's Wikipedia article and the Althing article itself prints "63" as the total
chamber size. That 63 is the DISTRACTOR: it spans ALL six parties (not just the four), so any
naive read that grabs the salient total answers the wrong question.

The four parties (Social Democratic Alliance, Independence Party, Viðreisn, People's Party of
Iceland) are deliberately SOLID-BUT-NOT-ICONIC (parties of a small Nordic parliament unknown to
most non-Icelandic speakers), and FINISHED (the November 2024 election results are closed, stable
facts unchanged until the next election). Crucially the requested partial sum is NOT a recallable
fact: nobody memorises "the four largest Icelandic parties in 2024 jointly hold 50 Althing seats",
whereas the total chamber size (63) is the comparatively salient and memorable aggregate — the decoy.

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): grabs the omnipresent 63 from any page it reads (denominator on party
    pages OR infobox total on the Althing page), or sums all six parties (= 63, same distractor),
    or reads a party page and confuses numerator with denominator -> wrong number.
  * frontier sequential (ReAct): reads each party's article, correctly extracts the numerator for
    each, and sums the four -> decent.
  * graph_compiled: four parallel leaves each fetch ONE party's seat count from THAT party's OWN
    article; each leaf is explicitly told to report the numerator (N), not the /63 denominator;
    the aggregation owns the summation and is forced to WRITE OUT the addition (a+b+c+d=T) before
    concluding -> the cheap executor is rescued and a diverse-grounding reranker can re-derive it.

Ground truth (verified against live English Wikipedia, 2026-06-27 — each party's infobox
'Seats in Parliament' field on its own party article; the total from the Althing article):

  item (subset)               seats   source slug
  Social Democratic Alliance    15    en.wikipedia.org/wiki/Social_Democratic_Alliance_(Iceland)
  Independence Party            14    en.wikipedia.org/wiki/Independence_Party_(Iceland)
  Viðreisn                      11    en.wikipedia.org/wiki/Vi%C3%B0reisn
  People's Party                10    en.wikipedia.org/wiki/People%27s_Party_(Iceland)
  --------------------------------------------------------------------------
  KEYSTONE = four-party SUM = 15 + 14 + 11 + 10 = 50   <- the answer
  (excluded) Centre Party        8    en.wikipedia.org/wiki/Centre_Party_(Iceland)
  (excluded) Progressive Party   5    en.wikipedia.org/wiki/Progressive_Party_(Iceland)
  DISTRACTOR = total Althing seats = 63   en.wikipedia.org/wiki/Althing
                                         (also appears as "/63" on every party article)

  KEYSTONE TOTAL = 50.  DISTRACTOR = 63.  They DIFFER by a clear margin of 13: the printed
  aggregate is ~26% larger than the requested partial sum.  The four subset integers (15, 14, 11,
  10) are DISTINCT among themselves, so the coverage diagnostic is collision-free.

ROBUSTNESS / tolerance band:
  The validator accepts the keystone in the TIGHT band [49, 51] = 50 +/- 1 — the exact partial sum,
  forgiving a single +/-1 misread on one party. That band:
    * ACCEPTS the true total 50 (and a one-off +/-1 slip -> 49 or 51);
    * REJECTS the DISTRACTOR 63 (total chamber size) — it is 12 above the upper band edge;
    * REJECTS every DROP-ONE partial sum:
        Drop SDA (15)         -> 14+11+10 = 35, margin  14 below the lower band edge (49);
        Drop Independence (14)-> 15+11+10 = 36, margin  13 below the lower band edge;
        Drop Viðreisn (11)    -> 15+14+10 = 39, margin  10 below the lower band edge;
        Drop People's (10)    -> 15+14+11 = 40, margin   9 below the lower band edge.
      Every drop-one sum is at least 9 below the lower band edge, so omitting any single party is
      unambiguously detectable even with a further +/-1 misread on another party;
    * REJECTS the all-six sum (63, identical to the distractor);
    * REJECTS any three-party combination (max = 15+14+11 = 40, still 9 below lower edge).
  Because every party's seat count is >= 10, dropping any one party moves the total by >= 10,
  which exceeds the +/-1 tolerance; a +/-2 misread on a single party (-> 48 or 52) is likewise
  detected.

ANTI-PARAMETRIC: the chamber total (63) is the comparatively memorable / guessable number and is
the DECOY — it appears on EVERY page the agent is likely to read. The keystone (50) is a computed
partial sum that is not a recallable fact, so a model that pattern-matches a salient quantity from
memory mis-answers and a correct answer requires actually reading the four per-party figures and
adding them.

  KEYSTONE = the computed four-party seat SUM (50, exact-match within +/- 1). Secondary (gated) =
  the four per-party seat counts. Coverage (ungated) = items gathered. get_llm_validation_function
  -> None. Leak-free get_compiled_plan routes AROUND the /63 denominator distractor.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# Each item is ONE party in the four-party subset; ``seats`` is that party's infobox seat count.
# Nothing here (no per-party figure, no total) is leaked into the task statement or the plan.
PARTIES: List[Dict[str, Any]] = [
    {
        "key": "sda",
        "label": "Social Democratic Alliance",
        "seats": 15,
        "slug_rx": r"social_democratic_alliance",
    },
    {
        "key": "ip",
        "label": "Independence Party",
        "seats": 14,
        "slug_rx": r"independence_party_\(iceland\)",
    },
    {
        "key": "vid",
        "label": "Vi\xf0reisn",                        # Viðreisn — ð is U+00F0
        "seats": 11,
        "slug_rx": "vi(?:%c3%b0|\xf0)reisn",           # matches Vi%C3%B0reisn (URL-encoded) or Viðreisn
    },
    {
        "key": "pp",
        "label": "People's Party",
        "seats": 10,
        "slug_rx": r"people(?:%27|')s_party_\(iceland\)",  # matches People%27s_Party_ or People's_Party_
    },
]

KEYSTONE_TOTAL = 50            # 15 + 14 + 11 + 10 — the four-party seat subset sum
DISTRACTOR_TOTAL = 63          # total Althing seats — printed on Althing article AND as "/63"
                               # denominator on every individual party article
EXCLUDED_SEATS = [8, 5]        # Centre Party (8) and Progressive Party (5) — NOT in the subset
KEYSTONE_BAND = {49, 50, 51}   # 50 +/- 1: accepts the exact sum (and a single +/-1 misread)

# Numeric-token extractor: grabs each maximal digit run together with any internal decimal/grouping
# separators, bounded by digits on both ends (so a trailing sentence period in "50." is NOT
# swallowed and the integer 50 is still recovered). ``_int_values`` then keeps only PLAIN integers:
# tokens with a decimal point ("50.5") are dropped, grouping commas are stripped ("1,050" -> 1050),
# so years ("2024"), decimals and grouped counts can never masquerade as the keystone or a figure.
_NUM_TOKEN_RX = re.compile(r"\d[\d.,]*\d|\d")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "083",
        "test_name": (
            "Tier 5: Bounded subset-sum with aggregate distractor "
            "(four-party seats in the 2024 Icelandic parliament)"
        ),
        "difficulty_level": "9/10",
        "category": "Quantitative reasoning + bounded subset-sum (aggregate distractor)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i + 1}. {p['label']}" for i, p in enumerate(PARTIES))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). Iceland's parliament, the Althing, has 63 seats in total. In the general "
        "election of 30 November 2024, six parties won seats. This task concerns ONLY the four "
        "parties that won the most seats in that election:\n"
        f"{listing}\n\n"
        "For EACH of these four parties, open that party's own English Wikipedia article (e.g. "
        "'Social Democratic Alliance (Iceland)') and read, from the infobox, the number of seats "
        "that party currently holds in the Althing. Then ADD UP the seat counts of those four "
        "parties and report the SUM — the combined number of Althing seats held by these four "
        "parties.\n\n"
        "IMPORTANT (read carefully): every party's Wikipedia article, and the main 'Althing' "
        "article, displays the total chamber size — 63 seats. That 63 is NOT the answer — it "
        "counts ALL six parties (not just the four), so it is larger than the figure you need. "
        "In particular, each party article's infobox shows the party's seats in the format "
        "'N / 63' — take ONLY the N (the party's own seats), not the 63 (the total). Do not "
        "report the total chamber size; you must add up ONLY the four specified parties' seat "
        "counts yourself.\n\n"
        "Report (a) the sum of the seat counts of the four parties (a single integer — the "
        "keystone), (b) the four individual per-party seat counts you looked up, and (c) the "
        "exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Sum of the Althing seat counts of the four specified parties (the computed keystone total)",
        "The four individual per-party seat counts (the looked-up figures)",
        "Source URL for each party's Wikipedia article",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (one per party article; target 4 — a four-way fan-out)",
        "Correctly reports the four-party seat sum (50 = 15+14+11+10), NOT the total Althing size "
        "(63, the aggregate distractor), and NOT a partial sum that drops a party",
        "Gathers all four per-party seat counts (15, 14, 11, 10)",
        "Cites the party source pages",
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
            continue  # a decimal like "50.5" is not a plain integer
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
    """KEYSTONE gate: deliverables[0] reports the four-party seat SUM inside the tight band
    [49, 51] (= 50 +/- 1). The band ACCEPTS the exact total (and a single +/-1 misread) but
    REJECTS the printed total chamber size (63, the distractor), every drop-one partial sum
    (35, 36, 39, 40 — all at least 9 below the lower band edge) and the all-six sum (63)."""
    return any(n in KEYSTONE_BAND for n in _int_values(_strip_list_markers(_primary_text(result))))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a four-way fan-out wants one page per party."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 3,
        "score": min(1.0, n / 4.0),
        "reason": f"{n} visit(s) (target >=4: one article per party; >=3 to pass)",
    }


def validate_keystone_subset_sum(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the COMPUTED four-party seat subset sum, 50 = 15+14+11+10, reported in
    deliverables[0] within [49, 51]. A model that grabs the salient total chamber size (63, the
    aggregate distractor), drops a party, or mis-adds lands outside the band and fails."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_subset_sum",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Four-party seat sum (50 = 15+14+11+10) present in band [49,51]" if passed else
            "Four-party seat sum (50) missing/incorrect (beware the total Althing size 63 — the "
            "aggregate distractor — and any dropped-party partial sum)"
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the FOUR per-party seat counts were
    gathered. The four subset integers (15, 14, 11, 10) are DISTINCT, so there is no
    cross-crediting (collision-free). Deliberately NOT gated on the keystone — it measures
    whether the agent fanned out to all four parties and collected each figure even when it botches
    the addition or grabs the aggregate."""
    present = set(_int_values(_all_text(result)))
    hits = [p["label"] for p in PARTIES if p["seats"] in present]
    n = len(PARTIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} per-party seat counts gathered ({', '.join(hits) or 'none'})",
    }


def validate_perparty_figures(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: are ALL four per-party seat counts (15, 14, 11, 10) reported? Short-circuits
    to 0 when the keystone is absent, so a run that grabs the distractor or mis-adds cannot bank
    credit for the per-item figures."""
    if not _keystone_ok(result):
        return {
            "check": "perparty_figures",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> per-party figures not credited",
        }
    present = set(_int_values(_all_text(result)))
    hits = [p["label"] for p in PARTIES if p["seats"] in present]
    n = len(PARTIES)
    return {
        "check": "perparty_figures",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} per-party figures reported ({', '.join(hits) or 'none'})",
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the party pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {
            "check": "citation",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> source URLs not credited",
        }
    text = _all_text(result).lower()
    cited = sum(1 for p in PARTIES if re.search(p["slug_rx"], text))
    n = len(PARTIES)
    return {
        "check": "citation",
        "passed": cited >= 3,
        "score": cited / n,
        "reason": f"{cited}/{n} party pages cited",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_subset_sum,
        validate_coverage,
        validate_perparty_figures,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 055/059/070.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    FOUR INDEPENDENT parallel leaves: one per party in the subset, each fetching that single
    party's seat count from that party's OWN Wikipedia article — one atomic integer per leaf.
    This structurally routes AROUND the omnipresent 63 denominator (which appears as "N / 63" on
    every party page and as the total in the Althing article's infobox). Each leaf is explicitly
    told to report ONLY the numerator N, not the /63. ALL arithmetic — the four-term addition —
    lives only in the aggregation step, which is forced to WRITE OUT the addition explicitly
    (a + b + c + d = T) before concluding, so the cheap executor never grabs the salient aggregate
    and a diverse-grounding reranker can re-derive and catch a slip. Encodes STRUCTURE only: it
    names the four parties but leaks no per-party seat count and not the keystone sum.
    """
    leaves: List[Dict[str, Any]] = []
    for p in PARTIES:
        leaves.append({
            "id": f"{p['key']}_seats",
            "instruction": (
                f"Open the English Wikipedia article for the Icelandic political party "
                f"'{p['label']}' (search for its Wikipedia page if needed) and read, from the "
                f"infobox, the number of seats it currently holds in the Althing (Iceland's "
                f"parliament). The infobox displays the party's seats in the format 'N / 63' — "
                f"report ONLY the N (the party's own seat count), not the 63 (the total Althing "
                f"chamber size, which is NOT what you want here). Report that single integer and "
                f"the source URL. Do NOT report the total Althing seat count, and do not guess "
                f"from memory."
            ),
            "expect": (
                f"SEAT COUNT for {p['label']} in the Althing (a single integer) — source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the four specified Icelandic parties (Social Democratic "
            "Alliance, Independence Party, Vi\xf0reisn, People's Party), that party's seat count "
            "in the Althing. Write out the addition EXPLICITLY on one line in the form "
            "'Social Democratic Alliance + Independence Party + Vi\xf0reisn + People’s Party "
            "= TOTAL', substituting the four integers you gathered, and compute the sum — do this "
            "BEFORE drawing any conclusion. Report that sum as the keystone answer: the combined "
            "seat count of ONLY these four parties. Do NOT report the total Althing chamber size "
            "(63 seats, spanning all six parties) — report only the sum of these four. Also list "
            "the four per-party seat counts you used, and cite each party article's source URL."
        ),
    }
