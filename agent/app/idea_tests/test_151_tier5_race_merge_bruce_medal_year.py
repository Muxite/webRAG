r"""
Test 151: Tier 5 (graph) — RACE-AND-MERGE, THREE REDUNDANT ROUTES TO ONE VALUE (B).
Level: graph   Weight: medium   Difficulty: 7/10

The cross-domain replication partner of test 150 (same shape, unrelated subject matter and a
route set that leaves Wikipedia entirely), so the pair can be used as a within-shape replication
— or as a seed/held-out split — rather than a single lucky fixture.

Shape (identical to 150, deliberately): ONE target value, THREE independent non-overlapping
lookup paths, each of which states the value in full. The siblings are REDUNDANT — any one of
them answers the task — which is what makes the task a probe for ``race-and-merge`` (author k
candidate approaches as concurrent siblings after the SAME fact; the first to reach DONE with a
verified datum wins) rather than for the AND-join / chain-merge shapes that tests 146-149
already cover. Nothing here is added, compared or filtered.

  ROUTE 1 (subject page, fast, decoy-rich) — the laureate's own English Wikipedia biography. The
      award year sits in an honours list among several other years on the same page (see the
      decoy note below), so this route is quick but easy to mis-read.
  ROUTE 2 (award page, table lookup) — English Wikipedia's article on the medal itself, whose
      recipient table has one row per year since 1898. The row must be found in a ~120-row list,
      but the table's year column cannot be confused with anything biographical.
  ROUTE 3 (specialist reference site, off-Wikipedia) — the Bruce Medalists reference pages hosted
      by Sonoma State University's Department of Physics and Astronomy, the medal's standing
      biographical reference, which publishes the medalists sorted by award date. Entirely
      independent of Wikipedia: a different publisher, a different editorial chain.

Ground truth (verified LIVE on all three routes, 2026-08-20):
  target value = the YEAR Martin Harwit received the Catherine Wolfe Bruce Gold Medal = 2007
    ROUTE 1  en.wikipedia.org/wiki/Martin_Harwit                         honours: "Bruce Medal (2007)"
    ROUTE 2  en.wikipedia.org/wiki/Catherine_Wolfe_Bruce_Gold_Medal      recipients: "2007  Martin Harwit"
             (the title 'Bruce Medal' redirects here; both slugs are accepted as a citation)
    ROUTE 3  phys-astro.sonoma.edu (Bruce Medalists sorted by award date, node/71)
             "2007 Martin Otto Harwit"
  ALL THREE AGREE EXACTLY on 2007. Note route 3 prints the middle name ("Martin Otto Harwit")
  while both Wikipedia routes print "Martin Harwit" — the validators key on the surname, so a
  report that copies either rendering is credited.

  DECOY YEARS, all printed on route 1's own page (this is why the keystone demands the year be
  bound to the laureate/medal with no other year in between, rather than merely present):
      1931 — his year of birth;
      1987 — elected a Fellow of the American Physical Society;
      1995 — resigned as director of the Smithsonian National Air and Space Museum over the
             Enola Gay exhibition, the single most-reported fact about him and by far the most
             likely parametric-memory anchor;
      12143 — the number of the asteroid named after him (a five-digit near-miss for a naive
             four-digit year regex; guarded against explicitly).
  The adjacent medal years (2006 Frank J. Low, 2008 Sidney van den Bergh) are the decoys of the
  route-2 table scan: an off-by-one row read lands on a real recipient with a real year.

  Margin: the answer is an exact identifier-like datum (an award year), not a measurement, so
  there is no rounding noise to flip — the three routes agree to the digit. The nearest wrong
  answers are one table row away, which is what the row-binding proximity guard is for.

Leak surface: the year a specific astronomer received a specialist society medal is close to
unrecoverable from parametric memory (it appears on exactly these pages), which is the point of
choosing an identifier-shaped fact over a famous measurement. The keystone still requires
GROUNDING (visit.count > 0).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── the three redundant routes (single source of truth for statement, validators and the plan) ──
ROUTES: List[Dict[str, str]] = [
    {
        "key": "laureate_page",
        "name": "the laureate's own English Wikipedia biography",
        "how": ("open the English Wikipedia biography of the astronomer Martin Harwit and read the "
                "year of this award from his list of honours"),
        "slug": r"/wiki/martin_harwit",
        "trait": "fastest route, but his page carries several other years that are not the award year",
    },
    {
        "key": "medal_page",
        "name": "English Wikipedia's article on the medal itself",
        "how": ("open English Wikipedia's article on the Catherine Wolfe Bruce Gold Medal and find "
                "this recipient's row in its table of recipients"),
        "slug": r"/wiki/catherine_wolfe_bruce_gold_medal|/wiki/bruce_medal",
        "trait": "one row per year since 1898, so the row must be located carefully",
    },
    {
        "key": "sonoma_reference",
        "name": ("the Bruce Medalists reference pages published by Sonoma State University's "
                 "Department of Physics and Astronomy"),
        "how": ("open the Bruce Medalists reference pages published by Sonoma State University's "
                "Department of Physics and Astronomy, which list the medalists sorted by award "
                "date, and find this recipient"),
        "slug": r"phys-astro\.sonoma\.edu",
        "trait": "off Wikipedia entirely — a different publisher and editorial chain",
    },
]

SUBJECT = "Martin Harwit"
AWARD = "Catherine Wolfe Bruce Gold Medal"
TARGET_YEAR = "2007"

_SUBJECT_RX = r"harwit"
_AWARD_RX = r"bruce"

# KEYSTONE: the year must be BOUND to the laureate (or to the medal) with NO other year between the
# name and the figure — that is what rejects "Harwit (born 1931) ... 2007" style near-misses and the
# 1995 Enola Gay anchor. ``(?<!\d)`` before the year blocks the asteroid number 12143 and any other
# longer digit run; the windows use [^.] so a line break between the label and the year still matches.
_YEAR = r"(?<![\d,.])" + TARGET_YEAR + r"\b"
_NOT_A_YEAR = r"(?!\b(?:1[6-9]\d\d|20[0-4]\d)\b)"
KEYSTONE_RX = re.compile(
    r"(?:" + _SUBJECT_RX + r"|" + _AWARD_RX + r")(?:" + _NOT_A_YEAR + r"[^.]){0,90}" + _YEAR
    + r"|" + _YEAR + r"(?:" + _NOT_A_YEAR + r"[^.]){0,90}(?:" + _SUBJECT_RX + r"|" + _AWARD_RX + r")",
    re.IGNORECASE,
)

# VETO: a decoy year asserted AS this medal's year — i.e. a decoy is the FIRST year following the
# medal's name (the ``_NOT_A_YEAR`` lookahead makes the scan stop at whatever year comes first, so
# "Bruce Medal (2007) ... 2006 Frank Low" is safe). Deliberately forward-only: a backward window
# would false-fire on the honours-list layout "APS Fellow 1987; Bruce Medal 2007", where an earlier
# unrelated year sits just before the medal's name.
_WRONG_AWARD_YEAR_RX = re.compile(
    _AWARD_RX + r"(?:" + _NOT_A_YEAR + r"[^.]){0,60}\b(?:1931|1987|1995|2005|2006|2008|2009)\b",
    re.IGNORECASE,
)

_AGREEMENT_RX = re.compile(
    r"corroborat|cross-?check|independently\s+confirm|both\s+(?:routes|sources|pages)|"
    r"all\s+three|agree|agreement|match(?:es|ed|ing)?|consistent|same\s+(?:value|year)|"
    r"confirmed\s+by",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "151",
        "test_name": "Tier 5: Race-and-merge B — one value, three redundant lookup routes (medal award year)",
        "difficulty_level": "7/10",
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
        f"memory). ONE value is wanted: the YEAR in which the American astronomer {SUBJECT} "
        f"received the {AWARD}, the lifetime-achievement medal awarded by the Astronomical Society "
        "of the Pacific.\n\n"
        "Three INDEPENDENT routes each state that year in full. They do not depend on one another, "
        "and ANY ONE OF THEM IS SUFFICIENT — there is nothing to combine, compare or compute:\n"
        f"{listing}\n\n"
        "Pursue the routes CONCURRENTLY rather than one after another, and answer as soon as a "
        "route delivers the year; if a route stalls, returns nothing, or cannot be reached, drop it "
        "and answer from another. Where two or more routes do return a year, check that they agree "
        "and say so.\n\n"
        "Careful: several OTHER years appear next to this person's name — his year of birth, the "
        "year he was elected a Fellow of the American Physical Society, and the year he resigned "
        "from the Smithsonian National Air and Space Museum, which is the fact he is most often "
        "written about. Report the year of THIS MEDAL, and beware of reading the row above or "
        "below his in a table of recipients.\n\n"
        "Report (a) the award year, stated as the year he received this medal (this is the "
        "keystone answer); (b) which routes you actually reached and what each of them returned; "
        "and (c) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The year the laureate received the medal, stated as the award year (the keystone)",
        "Which of the three routes were reached, and the year each returned",
        "Source URL for every route/page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least one page visited and the year read from a page, not from memory",
        "Reports the medal year (2007), not his birth year, his APS fellowship year, the 1995 "
        "Smithsonian resignation, or an adjacent recipient's year",
        "Reaches more than one independent route (the breadth diagnostic; a race launches all "
        "three, a linear walk reaches one)",
        "States that the routes reached agree on the year",
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
    """KEYSTONE gate (hard 0/1): the primary answer binds 2007 to the laureate/medal with no other
    year in between, does not assert a decoy year as the award year, and the run is GROUNDED
    (visit.count > 0) so an ungrounded guess banks nothing."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    if not grounded:
        return False
    text = _primary_text(result)
    if not re.search(_SUBJECT_RX, text, re.IGNORECASE):
        return False
    if not KEYSTONE_RX.search(text):
        return False
    return not _WRONG_AWARD_YEAR_RX.search(text)


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric. One read answers the task; a race reaches all three routes."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 1, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (>=1 to pass; a race reaches all 3 routes)"}


def validate_keystone_award_year(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the medal was awarded to this laureate in 2007. The decoys are the
    other years printed on his own page (1931 birth, 1987 APS, 1995 Smithsonian) and the adjacent
    rows of the recipient table (2006, 2008)."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_award_year", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Award year 2007 reported and bound to the laureate/medal" if passed
                       else "Award-year answer (2007) missing/incorrect — beware 1995 (Smithsonian), "
                            "1931 (birth), 1987 (APS) and the adjacent recipient rows")}


def validate_route_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the THREE redundant routes were actually
    reached (route URL cited).

    The axis the race mechanism moves and a linear agent cannot: a linear walk reaches exactly one
    route and has no fallback, a racer reaches all three. NOT gated on the keystone, so breadth is
    retained when the wrong year is reported. Visit-capped (``min(hits, n_visits)``, the canonical
    F29 pattern) so recited URLs cannot bank breadth without reads.
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
    """GATED secondary: the merge step is spelled out — at least two independent routes reported
    and stated to agree. Short-circuits to 0 without the keystone."""
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
        validate_keystone_award_year,
        validate_route_coverage,
        validate_route_agreement,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored RACE-AND-MERGE scaffold for the ``graph_compiled`` variant.

    One wave of three independent leaves (no edges) whose SEMANTICS are redundancy, not fan-out:
    every leaf resolves the SAME quantity by a different route, and the aggregation is a
    first-past-the-post pick with corroboration. Each leaf carries the same mechanical
    DONE/NOT-FOUND contract so the merge can distinguish 'this route resolved' from 'this route
    failed' without reopening a page, and is confined to its own route so the three stay
    genuinely independent.

    Leak-free: the leaves name only GIVENS (the laureate, the medal, the three publications).
    No year appears anywhere in the plan, and nothing says which route wins.
    """
    leaves = [
        {
            # id keyed on the ROUTE (a given), never on the value being sought.
            "id": f"route_{r['key']}",
            "instruction": (
                f"Independently of any other lookup, {r['how']}. The award is the {AWARD}, the "
                "lifetime-achievement medal of the Astronomical Society of the Pacific; the "
                f"recipient is the astronomer {SUBJECT} (some sources print his middle name). Read "
                "the YEAR IN WHICH HE RECEIVED THIS MEDAL — not his year of birth, not the year of "
                "any other honour, and not the year of a neighbouring recipient in any list. Use "
                "ONLY this route's pages: do not open the other routes' pages, and do not guess "
                "from memory. Report in the form 'ROUTE RESULT: medal awarded in <year>' followed "
                "by the exact source URL. If this route does not state the year, or you cannot "
                "reach it, report exactly 'ROUTE RESULT: NOT FOUND' and stop — do not substitute "
                "another source."
            ),
            "expect": "'ROUTE RESULT: medal awarded in <year>' + exact source URL, or 'ROUTE RESULT: NOT FOUND'",
            "depends_on": [],
        }
        for r in ROUTES
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "The three results above are INDEPENDENT lookups of the SAME single quantity — the "
            "year this laureate received this medal — by three different routes. They are "
            "redundant, not complementary: do NOT combine them, and do not treat a missing route "
            "as a missing part of the answer. MERGE them as follows: ignore every route that "
            "reported NOT FOUND; if the remaining routes all report the same year, that year is "
            "the answer; if they disagree, prefer the year reported by the majority of routes, and "
            "if there is no majority prefer the route whose page ties the year directly to this "
            "medal rather than to the person's other honours. Then report (a) the award year, "
            "stated as the year he received this medal — this is the keystone answer; (b) one line "
            "per route saying which routes returned a year, what each returned, and whether they "
            "agree; and (c) the exact source URL of every route that returned a year."
        ),
    }
