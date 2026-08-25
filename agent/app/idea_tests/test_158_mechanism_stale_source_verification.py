"""
Test 158: Mechanism suite — stale-vs-current source conflict requiring an explicit VERIFY hop
Level: integration   Weight: long   Difficulty: 8/10

Mechanism under test (DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md §8.3, item 4: "source conflict
requiring VERIFY"): the agent is HANDED one confident-looking, currently-live reference page and
must not stop there. The page publishes a summit elevation that *was* the official figure and is
still printed today; the figure was superseded by a later resurvey. Solving the task requires an
explicit second-source verification hop plus a temporal verdict (CURRENT vs SUPERSEDED) on the
given page's own number.

How this differs from the existing contradiction tasks (deliberately NOT a clone):
  * test_001 (conflicting information): gather three source *types* for one statistic and argue
    which is most plausible. No temporal supersession, no given source to audit, and its
    validators are generic keyword presence with no keystone.
  * test_056 (Everest 8,848 vs 8,848.86) and test_066 (Great Wall 8,850 km vs 21,196.18 km):
    "popular parametric memory is wrong, the authoritative page is right". The agent is given NO
    source; the conflict is memory-vs-page and the keystone is a single correct value.
  * test_158 (this task): the conflict is PAGE-vs-PAGE and TEMPORAL. A specific live URL is
    supplied, the figure on it is internally consistent and confidently presented, and the
    keystone can only be earned by (a) carrying the current figure AND (b) explicitly labelling
    the supplied page's figure as superseded/outdated. Reporting the current value alone — the
    shape that satisfies 056/066 — scores 0 here, because it does not prove the verification hop
    happened. There are THREE live-circulating generations of the same number, so "pick the
    biggest / pick the smallest" heuristics do not work either.

Ground truth (each figure verified against the live source on 2026-08-25):
  * SUPPLIED (stale, still live): https://www.mountain-forecast.com/peaks/Aoraki-Mount-Cook
    prints "Elevation: 3754" (metres) for Aoraki / Mount Cook — the figure that was official
    from 1991 until 2013/2014. Fetched 2026-08-25: the string "3754" is what the page serves.
  * CURRENT / KEYSTONE: 3,724 m (12,218 ft) — https://en.wikipedia.org/wiki/Aoraki_/_Mount_Cook
    infobox and body: "Two decades of erosion of the ice cap exposed after this collapse reduced
    the height by another 30 m (98 ft) to 3,724 m (12,218 ft)."
  * PRIMARY resolution source: https://www.otago.ac.nz/surveying/potree/pub/mrc/projects/aoraki
    (University of Otago Mountain Research Centre, "Aoraki 2013") — states the photogrammetric
    model gives 3724 m, that "3754 m" was the official height established in December 1991 after
    the major rock avalanche, and dates the GPS ascent to 23 November 2013 (supervised by
    Dr Pascal Sirguey). Verified live 2026-08-25.
  * PRE-1991 generation: 3,764 m (12,349 ft), per the Wikipedia article (Roberts 1881 /
    Brodrick 1889 measurements), lost ~10 m in the 14 December 1991 rock avalanche.

Margins: the three generations are 3,764 / 3,754 / 3,724 m — 10 m and 30 m apart, distinct
integers with no rounding overlap, so a noisy extraction cannot silently flip the keystone. The
Otago page also carries a decoy, 3719 m (the GPS reading taken on the ice cap itself, below the
photogrammetric summit), which does not collide with any of the three.

Scoring is bimodal by construction: the keystone is a hard 0/1 (current value + explicit stale
verdict on the supplied page's value + grounding), the coverage diagnostic is UN-gated (it
measures how much of the verification dossier was actually gathered even when the verdict is
botched — the axis that separates a structured agent from a linear one), and both secondaries
(conflict resolution, citations) short-circuit to 0 when the keystone is absent.
"""

from typing import Dict, Any, List
import re
from urllib.parse import urlparse

from agent.app.idea_test_utils import extract_final_text


STALE_PAGE_URL = "https://www.mountain-forecast.com/peaks/Aoraki-Mount-Cook"


def _num(tail: str) -> str:
    """Regex source for a 4-digit metre figure ``3<tail>``, tolerant of ``3,724``/``3 724``/
    ``3724`` and of a directly-appended unit (``3724m``), while refusing to match inside a longer
    number (``13,724``)."""
    return r"(?<![\d.,])3[,\s]?" + tail + r"(?!\d)"


_CURRENT_M = re.compile(_num("724"))          # current accepted elevation, metres
_SUPERSEDED_M = re.compile(_num("754"))       # the supplied page's figure (official 1991-2013)
_PRE1991_M = re.compile(_num("764"))          # the pre-avalanche generation
_YEAR_RESURVEY = re.compile(r"\b2013\b")
_YEAR_AVALANCHE = re.compile(r"\b1991\b")
_INSTITUTION = re.compile(r"\botago\b", re.IGNORECASE)

_CURRENT_WORDS = r"current|currently|accepted|official|correct|revised|latest|today|now|up[-\s]to[-\s]date"
_STALE_WORDS = r"supersed\w*|outdated|out[-\s]of[-\s]date|obsolete|no\s+longer|former\w*|previous\w*|prior|stale|old(?:er)?|incorrect|not\s+current"

# Proximity windows use [^.] (newline-tolerant: a report may break the line between the label and
# the number) but stop at a sentence boundary so a claim cannot borrow a word from a neighbour.
_CURRENT_NEAR_3724 = re.compile(
    r"(?:" + _CURRENT_WORDS + r")[^.]{0,60}" + _num("724")
    + r"|" + _num("724") + r"[^.]{0,60}(?:" + _CURRENT_WORDS + r")",
    re.IGNORECASE,
)
_STALE_NEAR_3754 = re.compile(
    r"(?:" + _STALE_WORDS + r")[^.]{0,60}" + _num("754")
    + r"|" + _num("754") + r"[^.]{0,60}(?:" + _STALE_WORDS + r")",
    re.IGNORECASE,
)
# Narrow anti-swap guards: a stale noun phrase bound DIRECTLY to the current value (or a current
# noun phrase bound directly to the superseded value) means the agent inverted the two roles, a
# failure the two proximity checks above can otherwise both satisfy in one ambiguous sentence
# ("Current elevation: 3,754 m (superseded value: 3,724 m)").
_SWAPPED = re.compile(
    r"(?:superseded|outdated|previous|former|old|prior)\s+(?:value|figure|height|elevation)\W{0,10}" + _num("724")
    + r"|(?:current|accepted|correct|official)\s+(?:value|figure|height|elevation)\W{0,10}" + _num("754"),
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "158",
        "test_name": "Mechanism: stale-vs-current source conflict requiring verification",
        "difficulty_level": "8/10",
        "category": "Source Conflict & Verification",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "A widely used mountaineering reference page publishes a summit elevation for Aoraki / "
        "Mount Cook, New Zealand's highest peak:\n"
        f"  {STALE_PAGE_URL}\n\n"
        "The figure on that page looks complete and confident. Do NOT take it at face value and "
        "do NOT answer from memory. Open that page and record the elevation it publishes "
        "verbatim, then VERIFY that figure against at least one independent authoritative source "
        "(the mountain's encyclopedia article and/or the primary report of the most recent "
        "scientific resurvey of the summit).\n\n"
        "Report:\n"
        "  (a) the elevation figure the supplied page publishes, and an explicit verdict on "
        "whether that figure is CURRENT or SUPERSEDED/out of date;\n"
        "  (b) the currently accepted summit elevation, in metres;\n"
        "  (c) any earlier elevation figure that these superseded, and the event and year that "
        "changed it;\n"
        "  (d) the resurvey that established the current figure — the year and the institution "
        "that carried it out;\n"
        "  (e) a source URL for every figure you report, and a one-line statement of which source "
        "you treat as authoritative and why.\n\n"
        "If the sources disagree, say so explicitly and resolve the disagreement by recency and "
        "source authority — do not silently pick one."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The elevation published by the supplied page, quoted verbatim",
        "An explicit CURRENT-or-SUPERSEDED verdict on that published figure",
        "The currently accepted summit elevation in metres",
        "The earlier (pre-avalanche) elevation and the event/year that changed it",
        "The resurvey year and the institution that carried it out",
        "A source URL for every figure, plus the authority/recency rationale",
    ]


def get_success_criteria() -> List[str]:
    return [
        "The supplied page is actually opened (at least 2 pages visited overall)",
        "Reports the current accepted elevation (3,724 m) as current",
        "Explicitly labels the supplied page's figure (3,754 m) as superseded/out of date",
        "Surfaces the earlier 3,764 m generation, the 1991 avalanche and the 2013 resurvey",
        "Cites the supplied page AND at least one independent authoritative source",
        "States the disagreement explicitly and resolves it by recency/authority",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE predicate. Requires all of:
      * GROUNDING — at least one page actually visited, so a parametric-memory answer that
        happens to name both numbers cannot earn credit;
      * the current elevation (3,724 m) asserted AS the current/accepted figure;
      * the supplied page's figure (3,754 m) explicitly labelled superseded/out of date;
      * the two roles not inverted (``_SWAPPED``).
    Reporting only one of the two figures — the "took the first confident source at face value"
    failure and the "answered the value without the verification hop" failure alike — fails.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = extract_final_text(result)
    if _SWAPPED.search(text):
        return False
    return bool(
        _CURRENT_M.search(text)
        and _SUPERSEDED_M.search(text)
        and _CURRENT_NEAR_3724.search(text)
        and _STALE_NEAR_3754.search(text)
    )


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Verification needs at least two page reads: the supplied page plus an independent check."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 2,
        "score": min(1.0, n / 3.0),
        "reason": f"{n} visit(s) (target >=3 — supplied page + independent verification; >=2 to pass)",
    }


def validate_keystone_stale_verdict(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): current elevation 3,724 m carried AS current AND the supplied page's
    3,754 m explicitly flagged as superseded, grounded in at least one real visit."""
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_stale_verdict",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Current 3,724 m reported as current AND the supplied page's 3,754 m flagged as superseded"
            if passed
            else "Missing the verified pair: current 3,724 m as current + supplied 3,754 m as superseded (or ungrounded/roles inverted)"
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated, by design): how much of the verification dossier was
    actually gathered — the current figure, the superseded figure, the pre-avalanche figure, the
    avalanche year, the resurvey year and the surveying institution. Deliberately NOT
    short-circuited on the keystone: it measures gathering, which separates a structured agent
    from a linear one even when the final verdict is botched."""
    text = extract_final_text(result)
    items = [
        ("current_3724", _CURRENT_M),
        ("superseded_3754", _SUPERSEDED_M),
        ("pre1991_3764", _PRE1991_M),
        ("avalanche_year_1991", _YEAR_AVALANCHE),
        ("resurvey_year_2013", _YEAR_RESURVEY),
        ("surveying_institution", _INSTITUTION),
    ]
    hits = [name for name, rx in items if rx.search(text)]
    n = len(items)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} dossier elements present ({', '.join(hits) or 'none'})",
    }


def validate_conflict_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary — the conflict is named and resolved by recency/authority. Short-circuits to 0
    when the keystone is absent (no keystone -> the 'resolution' is unearned prose)."""
    if not _keystone_ok(result, observability):
        return {
            "check": "conflict_resolution",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> conflict-resolution prose not credited",
        }
    text = extract_final_text(result)
    named = bool(re.search(r"disagree\w*|discrepan\w*|conflict\w*|contradict\w*|inconsisten\w*|does not match|differ\w*", text, re.IGNORECASE))
    verdict = bool(re.search(r"supersed\w*|out[-\s]of[-\s]date|outdated|no longer (?:current|official|accurate)|stale", text, re.IGNORECASE))
    rationale = bool(re.search(r"more recent|most recent|recency|newer|later survey|resurvey|re-survey|authoritative|primary source|peer[-\s]review|GPS|photogrammetr\w*", text, re.IGNORECASE))
    hits = sum([named, verdict, rationale])
    return {
        "check": "conflict_resolution",
        "passed": hits >= 2,
        "score": hits / 3.0,
        "reason": f"conflict named={named}, stale verdict={verdict}, recency/authority rationale={rationale}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary — both sides of the conflict are cited. Gated on the keystone."""
    if not _keystone_ok(result, observability):
        return {
            "check": "citations",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> source URLs not credited",
        }
    text = extract_final_text(result)
    lowered = text.lower()
    cited_supplied = bool(re.search(r"mountain-forecast\.com", lowered))
    cited_authoritative = bool(re.search(r"wikipedia\.org/wiki/aoraki|otago\.ac\.nz|doc\.govt\.nz|linz\.govt\.nz|nzgeographic", lowered))
    hosts = {
        (urlparse(u).netloc or "").lower().removeprefix("www.")
        for u in re.findall(r"https?://[^\s)\]\"'>]+", text)
    }
    hosts.discard("")
    two_hosts = len(hosts) >= 2
    hits = sum([cited_supplied, cited_authoritative, two_hosts])
    return {
        "check": "citations",
        "passed": cited_authoritative and two_hosts,
        "score": hits / 3.0,
        "reason": f"supplied page cited={cited_supplied}, authoritative source cited={cited_authoritative}, distinct hosts={len(hosts)}",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_stale_verdict,
        validate_coverage,
        validate_conflict_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored verification scaffold for the ``graph_compiled`` arm.

    Structure only: read the supplied page, read an independent encyclopedia page, read the
    primary resurvey report, and — chained on the first leaf via ``{published_figure}`` — ask
    whether the number the supplied page actually served is still current. No elevation, year,
    institution or verdict appears anywhere in the plan; the runtime model still does every read,
    extraction and the stale-vs-current judgement.
    """
    return {
        "leaves": [
            {
                "id": "published_figure",
                "instruction": (
                    f"Open the page {STALE_PAGE_URL} and record VERBATIM the summit elevation "
                    "figure it publishes for Aoraki / Mount Cook, together with its units."
                ),
                "expect": "the elevation exactly as printed on that page + the page URL",
                "depends_on": [],
            },
            {
                "id": "encyclopedia_figure",
                "instruction": (
                    "Open the English Wikipedia article for the New Zealand mountain "
                    "Aoraki / Mount Cook and read from the page: the summit elevation in the "
                    "infobox, any earlier elevation figures the article says it replaced, and the "
                    "event and year that changed them."
                ),
                "expect": "current elevation + earlier elevation(s) + the event/year that changed them + the article URL",
                "depends_on": [],
            },
            {
                "id": "resurvey_report",
                "instruction": (
                    "Find and open the primary report of the most recent scientific resurvey of "
                    "the summit height of Aoraki / Mount Cook (a university or geodetic-survey "
                    "publication, or a news report of it). Read from the page: the newly measured "
                    "height, the height it replaced, the date of the survey and the institution "
                    "that carried it out."
                ),
                "expect": "new height + replaced height + survey date + institution + source URL",
                "depends_on": [],
            },
            {
                "id": "cross_check",
                "instruction": (
                    "Search for the elevation figure {published_figure} for Aoraki / Mount Cook on "
                    "sources OTHER than the page it came from, and determine whether that exact "
                    "figure is still published as the current official summit height or has been "
                    "replaced by a later measurement."
                ),
                "expect": "still-current or replaced, with the replacing figure if any, plus the URL checked",
                "depends_on": ["published_figure"],
            },
        ],
        "aggregation": (
            "Compare the elevation published by the supplied page against the encyclopedia figure "
            "and the resurvey report. If they disagree, say so explicitly and resolve the "
            "disagreement by recency and source authority rather than silently picking one. "
            "Report: (a) the figure the supplied page publishes and an explicit verdict on whether "
            "it is CURRENT or SUPERSEDED/out of date; (b) the currently accepted summit elevation "
            "in metres, stated as the current figure; (c) any earlier elevation these superseded "
            "and the event and year that changed it; (d) the resurvey year and the institution "
            "that carried it out; (e) a source URL for every figure, and which source you treat as "
            "authoritative and why."
        ),
    }
