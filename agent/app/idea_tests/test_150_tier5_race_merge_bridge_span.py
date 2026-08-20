r"""
Test 150: Tier 5 (graph) — RACE-AND-MERGE, THREE REDUNDANT ROUTES TO ONE VALUE (A).
Level: graph   Weight: medium   Difficulty: 6/10

The shape this task exists for is deliberately NOT a chain and NOT an AND-join: there is ONE
target value, and THREE independent, non-overlapping lookup paths that each state it in full.
Any ONE of the three suffices for a correct answer — so the siblings are REDUNDANT, not
complementary. That is what makes it a clean probe for the ``race-and-merge`` mechanism
(author k candidate approaches as independent siblings after the same fact, run them
concurrently, and take the first that reaches DONE with a verified datum):

  * a mechanism that races correctly finishes as soon as the FASTEST route resolves, and still
    answers correctly when a slower/heavier route stalls or returns nothing;
  * a linear agent walks exactly one route and has no fallback if that route is the slow one;
  * an AND-style planner that insists all three resolve before merging pays 3x for nothing.

Contrast with tests 146/149 (per-branch chains merged by argmax) and 147/148 (AND-filters):
there, every branch is REQUIRED — dropping one destroys the answer. Here, dropping two is
harmless. The three routes also have DIFFERENT failure modes, which is the substantive reason
redundancy pays off here rather than being mere duplication:

  ROUTE 1 (subject page, fast, decoy-rich) — the bridge's own English Wikipedia article. Its
      infobox prints BOTH the total length and the main span, one line apart; the total length
      is the trap an inattentive read grabs.
  ROUTE 2 (ranked table, slow, decoy-free) — English Wikipedia's ranked list of the longest
      suspension bridge spans. The row must be located in a ~100-row table (expensive to scan),
      but the table's only length column IS the main span, so route 2 cannot be fooled by the
      total-length decoy at all.
  ROUTE 3 (foreign-language subject page, cross-edition) — the Norwegian Wikipedia article
      'Hardangerbrua'. Independent of the English edition's wording; states 'hovedspenn'
      (main span) explicitly, and prints the figure in Norwegian digit-group style ("1 310").

Ground truth (verified LIVE against all three routes, 2026-08-20):
  target value = MAIN SPAN of the Hardanger Bridge = 1,310 m
    ROUTE 1  en.wikipedia.org/wiki/Hardanger_Bridge          infobox "Longest span 1,310 metres (4,300 ft)"
             (same infobox, DECOY: "Total length 1,380 metres (4,530 ft)"; also height 201.5 m,
              clearance below 55 m — three other infobox figures in the same block)
    ROUTE 2  en.wikipedia.org/wiki/List_of_longest_suspension_bridge_spans   row: rank 21,
             "1,310 m (4,298 ft)", 2013, Norway
    ROUTE 3  no.wikipedia.org/wiki/Hardangerbrua              "hovedspenn 1 310 meter"
             (same page's total length: "1 380 meter")
  ALL THREE AGREE EXACTLY on 1,310 m. The nearest competing figure printed anywhere on those
  pages is the 1,380 m total length — 70 m / 5.3% away, and separated by a *semantic* label
  (span vs. total length) rather than by measurement noise, so the keystone cannot be flipped by
  a rounding difference; only by reading the wrong LABEL, which is exactly the error route 2 and
  route 3's explicit 'hovedspenn' wording are there to catch.

  Route-2 stability note: the ranked list's RANK for this bridge moves as new bridges open (it
  was 21st on 2026-08-20). Nothing in this task keys on the rank — only on the span in the row —
  so route 2 stays valid as the table grows.

Leak surface: the span of a mid-rank Norwegian suspension bridge is weak in parametric memory
(the fame decoys of this genre — Akashi Kaikyo, Golden Gate, 1915 Canakkale — are elsewhere in
the same table), and the keystone additionally requires GROUNDING (visit.count > 0), so a
0-visit narration banks nothing on any check.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── the three redundant routes (single source of truth for statement, validators and the plan) ──
# Each route independently yields the SAME target value; ``slug`` is how a cited route is detected.
ROUTES: List[Dict[str, str]] = [
    {
        "key": "subject_en",
        "name": "the bridge's own English Wikipedia article",
        "how": ("open the English Wikipedia article for the Hardanger Bridge and read the main-span "
                "figure out of its infobox"),
        "slug": r"en\.wikipedia\.org/wiki/hardanger_bridge|/wiki/hardanger_bridge",
        "trait": "fastest route, but its infobox also prints the bridge's TOTAL length",
    },
    {
        "key": "ranked_list",
        "name": "English Wikipedia's ranked list of the longest suspension bridge spans",
        "how": ("open English Wikipedia's ranked list of the world's longest suspension bridge spans "
                "and find this bridge's row in the table"),
        "slug": r"/wiki/list_of_longest_suspension_bridge_spans",
        "trait": "slow to scan (~100 rows), but its length column is the main span and nothing else",
    },
    {
        "key": "subject_no",
        "name": "the Norwegian Wikipedia article 'Hardangerbrua'",
        "how": ("open the Norwegian-language Wikipedia article titled 'Hardangerbrua' and read the "
                "value it labels 'hovedspenn' (main span)"),
        "slug": r"no\.wikipedia\.org/wiki/hardangerbrua|/wiki/hardangerbrua",
        "trait": "a different language edition, so it is independent of the English pages' wording",
    },
]

SUBJECT = "Hardanger Bridge"
TARGET_VALUE_M = 1310
DECOY_TOTAL_LENGTH_M = 1380

# The subject must be named for a bare number to mean anything.
_SUBJECT_RX = r"hardanger"

# Span vocabulary, English and Norwegian (route 3 prints 'hovedspenn').
_SPAN_WORDS = r"main\s+span|longest\s+span|central\s+span|hovedspenn|main-span|span"

# 1,310 written any of the ways the three routes print it: "1,310" (en infobox / table),
# "1 310" (no.wikipedia digit grouping), "1310" (bare). The leading guard blocks a longer number
# ending in 1310 and the trailing \b keeps "13100" out.
_VAL = r"(?<![\d.,])1[,\s.]?310\b"

# KEYSTONE: the value must be BOUND to span vocabulary, with NO other number between the label and
# the figure — that lookahead is what stops "Main span: 1,380 m (total 1,310 m)" from scoring. The
# proximity windows use [^.] (newline-tolerant: a report may break the line after "Main span:").
KEYSTONE_RX = re.compile(
    r"(?:" + _SPAN_WORDS + r")(?:(?!\d)[^.]){0,40}" + _VAL
    + r"|" + _VAL + r"(?:(?!\d)[^.]){0,40}(?:" + _SPAN_WORDS + r")",
    re.IGNORECASE,
)

# VETO: the total-length decoy asserted AS the span (the failure route 1 invites).
_SPAN_IS_DECOY_RX = re.compile(
    r"(?:main\s+span|longest\s+span|hovedspenn)(?:(?!\d)[^.]){0,40}(?<![\d.,])1[,\s.]?380\b",
    re.IGNORECASE,
)

# Corroboration language — an answer that actually merged two racing routes says so.
_AGREEMENT_RX = re.compile(
    r"corroborat|cross-?check|independently\s+confirm|both\s+(?:routes|sources|pages)|"
    r"all\s+three|agree|agreement|match(?:es|ed|ing)?|consistent|same\s+(?:value|figure)|"
    r"confirmed\s+by",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "150",
        "test_name": "Tier 5: Race-and-merge A — one value, three redundant lookup routes (bridge main span)",
        "difficulty_level": "6/10",
        "category": "Race-and-merge: redundant independent routes to one value",
        "level": "graph",
        "weight": "medium",
    }


def get_task_statement() -> str:
    listing = "\n".join(
        f"  ROUTE {i} — {r['name']}: {r['how']} ({r['trait']})."
        for i, r in enumerate(ROUTES, 1)
    )
    return (
        "You are given NO URLs — navigate the web yourself and READ the pages (do not guess from "
        f"memory). ONE value is wanted: the length of the MAIN SPAN, in metres, of the {SUBJECT} "
        "in Norway (the suspension bridge across the Hardangerfjord between Vallavik and Bu, "
        "opened in 2013).\n\n"
        "Three INDEPENDENT routes each state that value in full. They do not depend on one "
        "another, and ANY ONE OF THEM IS SUFFICIENT — you do not need to combine them, and there "
        "is nothing to add up or compare:\n"
        f"{listing}\n\n"
        "Pursue the routes CONCURRENTLY rather than one after another, and answer as soon as a "
        "route delivers the value; if a route stalls, returns nothing, or cannot be reached, drop "
        "it and answer from another. Where two or more routes do return a value, check that they "
        "agree and say so.\n\n"
        "Careful: the bridge's total length is NOT its main span, and the two figures sit one line "
        "apart in the same infobox. Report the MAIN SPAN.\n\n"
        "Report (a) the main span in metres, stated as the main span (this is the keystone "
        "answer); (b) which routes you actually reached and what each of them returned; and "
        "(c) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The main span of the bridge in metres, labelled as the main span (the keystone)",
        "Which of the three routes were reached, and the value each returned",
        "Source URL for every route/page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least one page visited and the value read from a page, not from memory",
        "Reports the MAIN SPAN (1,310 m) and not the total length (1,380 m)",
        "Reaches more than one independent route (the breadth diagnostic; a race launches all "
        "of them, a linear walk reaches one)",
        "States that the routes reached agree on the value",
        "Cites the source URL of each route used",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text — prefer ``deliverables[0]`` (the keystone slot), else final_deliverable."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text — final deliverable plus every deliverable slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (hard 0/1): the primary answer names the subject and binds the 1,310 m figure
    to span vocabulary, the total-length decoy is not asserted as the span, and the run is
    GROUNDED (visit.count > 0) so a parametric guess banks nothing."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    if not grounded:
        return False
    text = _primary_text(result)
    if not re.search(_SUBJECT_RX, text, re.IGNORECASE):
        return False
    if _SPAN_IS_DECOY_RX.search(text):
        return False
    return bool(KEYSTONE_RX.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric. A race launches all three routes (3+ reads); one read is enough to
    answer, so the bar to PASS is low while the score still rewards actually racing."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 1, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (>=1 to pass; a race reaches all 3 routes)"}


def validate_keystone_main_span(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the main span of the Hardanger Bridge, 1,310 m, reported AS the main
    span. The decoy is the 1,380 m total length printed one line above it in the same infobox."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_main_span", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Main span = 1,310 m reported and bound to the bridge" if passed
                       else "Main-span answer (1,310 m for the Hardanger Bridge) missing/incorrect "
                            "— beware the 1,380 m TOTAL length")}


def validate_route_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the THREE redundant routes were actually
    reached (route URL cited).

    This is the axis the race mechanism is supposed to move and the one a linear agent cannot: a
    linear walk reaches exactly one route, a racer reaches all three and can fall back when one
    fails. Deliberately NOT gated on the keystone — breadth is retained even when the final value
    is botched (e.g. the total-length decoy is reported). Visit-capped (``min(hits, n_visits)``,
    the canonical F29 pattern) so URLs recited without reading cannot bank breadth.
    """
    text = _all_text(result).lower()
    hits = [r["key"] for r in ROUTES if re.search(r["slug"], text)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(ROUTES)
    return {"check": "route_coverage", "passed": credited >= 2, "score": credited / n,
            "reason": f"{credited}/{n} independent routes reached ({', '.join(hits[:credited]) or 'none'}; "
                      f"{len(hits)} cited, {n_visits} visit(s); >=2 to pass)"}


def validate_route_agreement(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the merge step is spelled out — at least two independent routes were
    reported and the answer states that they agree. Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "route_agreement", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> route agreement not credited"}
    text = _all_text(result)
    n_routes = sum(1 for r in ROUTES if re.search(r["slug"], text.lower()))
    says_agree = bool(_AGREEMENT_RX.search(text))
    if n_routes >= 2 and says_agree:
        return {"check": "route_agreement", "passed": True, "score": 1.0,
                "reason": f"{n_routes} routes reported and stated to agree"}
    if n_routes >= 2:
        return {"check": "route_agreement", "passed": False, "score": 0.5,
                "reason": f"{n_routes} routes reported but no explicit agreement/corroboration statement"}
    return {"check": "route_agreement", "passed": False, "score": 0.0,
            "reason": f"only {n_routes} route(s) reported — nothing was merged"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: source URLs for the routes used. Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for r in ROUTES if re.search(r["slug"], text))
    n = len(ROUTES)
    return {"check": "citations", "passed": cited >= 1, "score": cited / n,
            "reason": f"{cited}/{n} route URL(s) cited (>=1 to pass)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_main_span,
        validate_route_coverage,
        validate_route_agreement,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored RACE-AND-MERGE scaffold for the ``graph_compiled`` variant.

    Structurally a single wave of three independent leaves (no edges) — the same topology as a
    fan-out plan, with the opposite SEMANTICS: the three leaves resolve the SAME quantity by
    different means, so the merge is a first-past-the-post pick with corroboration, never a
    combination. Each leaf therefore carries an explicit mechanical DONE/NOT-FOUND contract so the
    merge can tell 'this route resolved' from 'this route failed' without re-reading any page, and
    is forbidden from consulting the other routes' pages so the three stay genuinely independent.

    Leak-free: the leaves name only GIVENS (the bridge, the three pages, the attribute wanted).
    No span figure, no total-length figure, and no statement of which route wins the race.
    """
    leaves = [
        {
            # id keyed on the ROUTE (a given), never on the value being sought.
            "id": f"route_{r['key']}",
            "instruction": (
                f"Independently of any other lookup, {r['how']} for the {SUBJECT} in Norway (the "
                "suspension bridge over the Hardangerfjord, opened in 2013). Read the length of the "
                "MAIN SPAN in metres — the span between the two towers, which is NOT the bridge's "
                "total length; if the page prints both, take the one labelled as the span. Use ONLY "
                "this route's page: do not open the other routes' pages, and do not guess from "
                "memory. Report in the form 'ROUTE RESULT: main span = <number> m' followed by the "
                "exact source URL. If this route's page does not state the main span, or you cannot "
                "reach it, report exactly 'ROUTE RESULT: NOT FOUND' and stop — do not substitute "
                "another page."
            ),
            "expect": "'ROUTE RESULT: main span = <number> m' + exact source URL, or 'ROUTE RESULT: NOT FOUND'",
            "depends_on": [],
        }
        for r in ROUTES
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "The three results above are INDEPENDENT lookups of the SAME single quantity — the "
            "bridge's main span — by three different routes. They are redundant, not "
            "complementary: do NOT add, average or otherwise combine them, and do not treat a "
            "missing route as a missing part of the answer. MERGE them as follows: ignore every "
            "route that reported NOT FOUND; if the remaining routes all report the same figure, "
            "that figure is the answer; if they disagree, prefer the figure that the majority of "
            "routes report, and if there is no majority prefer the route that read the value from "
            "a field explicitly labelled as the span. Then report (a) the main span in metres, "
            "stated as the main span — this is the keystone answer; (b) one line per route saying "
            "which routes returned a value, what each returned, and whether they agree; and (c) "
            "the exact source URL of every route that returned a value."
        ),
    }
