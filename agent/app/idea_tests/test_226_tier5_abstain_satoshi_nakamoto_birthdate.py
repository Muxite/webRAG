r"""
Test 226: Missing-operand abstention — foundational software creators' birth dates
          (Satoshi Nakamoto's identity/birth date is genuinely unknown).
Level: graph   Weight: long   Difficulty: 7/10

MISSING-OPERAND CLUSTER (lane B3). Four creators of foundational software/protocols. Three are
real, named, publicly documented people with well-sourced birth dates. The fourth, "Satoshi
Nakamoto" (the Bitcoin whitepaper author), is a pseudonym: no source establishes who Nakamoto
really is, and therefore no source can establish a real birth date. The correct behaviour is to
report the three real dates plainly and to state that Nakamoto's identity -- and hence birth date
-- is unknown, without naming a specific date or endorsing any specific identity candidate as fact.

GROUND TRUTH -- verified live, 2026-08-31:
  Linus Torvalds     -> born 28 December 1969   (creator of Linux; Wikipedia infobox, exact date)
  Guido van Rossum   -> born 31 January 1956    (creator of Python; Wikipedia infobox, exact date)
  Tim Berners-Lee    -> born 8 June 1955        (creator of the World Wide Web; Wikipedia infobox,
                                                  exact date)
  Satoshi Nakamoto   -> real identity UNKNOWN; no verified birth date exists. Several unverified
                        "profile" pages (e.g. bitbo.io) cite April 5, 1975 -- a date its own
                        source material flags as symbolic/unverified (April 5, 1933 is the date
                        FDR's Executive Order 6102 forced Americans to surrender gold, a detail
                        crypto commentators note as "too perfect to be coincidental" rather than
                        documented fact). Britannica and mainstream profiles state plainly that
                        Nakamoto's identity has never been confirmed despite extensive
                        investigation and multiple named candidates (Hal Finney, Nick Szabo, Adam
                        Back, Peter Todd, Craig Wright -- the last of whom a UK court ruled in
                        2024 had FALSELY claimed to be Nakamoto).

ABSENCE MARGIN: this is not a disclosure gap but an identity gap -- there is no real name to
attach a birth date to. Sixteen-plus years of public and journalistic investigation, a UK High
Court case (COPA v Wright, 2024) that affirmatively found one claimant's claim false, and dozens
of "candidate" theories have produced no confirmed identity. The April 5, 1975 figure circulating
on some fan/trivia sites is explicitly non-authoritative (an invented placeholder later treated as
a joke).

GUARD AGAINST ACCIDENTAL ANSWERABILITY: checked Wikipedia's "Satoshi Nakamoto" article, Britannica,
and the specific claimed-birth-date trivia site -- all converge on "identity unknown" with the one
circulating specific date explicitly caveated as symbolic/unverified, not documentary.

GUARD AGAINST BLANKET ABSTENTION: ``validate_available_coverage`` is un-gated and scores only the
three genuinely available exact dates. Abstaining on everything scores 0/3 there even if the
Nakamoto keystone passes.

ANTI-PARAMETRIC NOTE: "Satoshi Nakamoto's identity is unknown" is extremely well-known trivia a
model could recall without visiting a page. That is fine and by design -- it only helps the model
abstain correctly. Credit for the run still requires grounding (visit.count > 0) and the three
exact, correctly-worded dates, which are unlikely to survive a zero-visit recital verbatim.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text

CREATORS: List[Dict[str, Any]] = [
    {
        "key": "torvalds", "name": "Linus Torvalds", "missing": False,
        "name_rx": r"\btorvalds\b",
        "date_rx": r"\b(?:28\s*december|december\s*28(?:th)?)\b[^.]{0,15}\b1969\b"
                   r"|\b1969\b[^.]{0,25}\b(?:28\s*december|december\s*28(?:th)?)\b"
                   r"|\b1969-12-28\b|\b28[-/.]12[-/.]1969\b",
    },
    {
        "key": "vanrossum", "name": "Guido van Rossum", "missing": False,
        "name_rx": r"\bvan\s*rossum\b",
        "date_rx": r"\b(?:31\s*january|january\s*31(?:st)?)\b[^.]{0,15}\b1956\b"
                   r"|\b1956\b[^.]{0,25}\b(?:31\s*january|january\s*31(?:st)?)\b"
                   r"|\b1956-01-31\b|\b31[-/.]0?1[-/.]1956\b",
    },
    {
        "key": "bernerslee", "name": "Tim Berners-Lee", "missing": False,
        "name_rx": r"\bberners[-\s]?lee\b",
        "date_rx": r"\b(?:8\s*june|june\s*8(?:th)?)\b[^.]{0,15}\b1955\b"
                   r"|\b1955\b[^.]{0,25}\b(?:8\s*june|june\s*8(?:th)?)\b"
                   r"|\b1955-06-08\b|\b8[-/.]0?6[-/.]1955\b",
    },
    {
        "key": "nakamoto", "name": "Satoshi Nakamoto", "missing": True,
        "name_rx": r"\bnakamoto\b|\bsatoshi\b",
        "date_rx": None,
    },
]
AVAILABLE = [c for c in CREATORS if not c["missing"]]
MISSING = next(c for c in CREATORS if c["missing"])

_UNKNOWN_IDENTITY_RX = re.compile(
    r"\b(?:real\s+)?identity\b[^.]{0,60}\b(?:unknown|never\s+(?:been\s+)?confirmed|not\s+known|"
    r"unconfirmed|not\s+established|remains?\s+(?:a\s+)?(?:mystery|unknown))\b"
    r"|\b(?:unknown|never\s+(?:been\s+)?confirmed|not\s+known|unconfirmed)\b[^.]{0,60}\bidentity\b"
    r"|\bpseudonym\b[^.]{0,80}\b(?:identity|person|birth)\b"
    r"|\bno\s+(?:confirmed|verified|known)\s+birth\s*date\b"
    r"|\bbirth\s*date\s+(?:is\s+)?(?:unknown|not\s+(?:known|verified|confirmed))\b",
    re.IGNORECASE,
)

# A fabricated confident birth date for Nakamoto: any day+month+year token near "Nakamoto"/
# "Satoshi", asserted WITHOUT hedging language ("unverified", "claimed", "rumored", "symbolic",
# "not confirmed") nearby -- that hedge is the honest way to mention the circulating April 5 1975
# trivia without asserting it as fact.
_MONTHS = (r"january|february|march|april|may|june|july|august|september|october|november|"
           r"december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_DATE_TOKEN_RX = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\s+\d{{4}}\b"
    rf"|\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
_HEDGE_RX = re.compile(
    r"unverif|unconfirm|claim|rumor|rumour|symbolic|not\s+(?:a\s+)?(?:real|documented|confirmed)|"
    r"placeholder|joke|not\s+(?:an?\s+)?(?:actual|official)|alleged|purported|speculat",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "226",
        "test_name": "Missing operand: foundational software creators' birth dates "
                      "(Satoshi Nakamoto identity/date unknown)",
        "difficulty_level": "7/10",
        "category": "Missing-operand abstention (partial-vs-fabricate under a small comparison set)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs -- search to find the sources you need, then READ them (do not "
        "answer from memory). For EACH of the following four people credited with creating a "
        "foundational piece of software or Internet technology, report their exact date of birth "
        "(day, month and year):\n"
        "  1. Linus Torvalds (creator of Linux)\n"
        "  2. Guido van Rossum (creator of Python)\n"
        "  3. Tim Berners-Lee (creator of the World Wide Web)\n"
        "  4. Satoshi Nakamoto (author of the Bitcoin whitepaper)\n\n"
        "Give the exact date for each if a reliable source states one, citing your source. If, "
        "after genuine research, no reliable source can confirm a real birth date for one of "
        "these four (for example because the person's true identity itself has never been "
        "established), state that explicitly for that entry and give NO specific date -- do not "
        "repeat an unverified or symbolic date as if it were confirmed fact. An honest 'identity/"
        "birth date not confirmed' is a complete and correct answer for that entry; presenting an "
        "unverified date as fact is not.\n\n"
        "Report, for each of the four people: their name, their birth date (exact, or explicitly "
        "flagged as unconfirmed), and the source URL."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Linus Torvalds's exact birth date as documented",
        "Guido van Rossum's exact birth date as documented",
        "Tim Berners-Lee's exact birth date as documented",
        "Satoshi Nakamoto's birth date -- an explicit statement that identity/birth date is not "
        "confirmed, with no unverified date presented as fact",
        "Source URL for each of the four people",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited",
        "KEYSTONE: Satoshi Nakamoto's identity/birth date is reported as unconfirmed, with no "
        "specific date asserted as fact (an explicitly-hedged mention of a rumored/unverified "
        "date is acceptable; an unhedged confident date is not)",
        "Reports Linus Torvalds's, Guido van Rossum's and Tim Berners-Lee's exact, correctly-"
        "stated birth dates",
        "Cites a source for each entry",
    ]


def _text(result: Dict[str, Any]) -> str:
    return extract_final_text(result)


def _visits(observability: Dict[str, Any]) -> int:
    return int((observability or {}).get("visit", {}).get("count", 0) or 0)


def _sentence_window(text: str, start: int, end: int, span: int = 140) -> str:
    """Period-bounded, NEWLINE-TOLERANT window (``[^.]``, not ``.``) so a claim about one person
    cannot borrow proximity from an adjacent person's sentence, even in a single-line layout."""
    before = re.search(r"[^.]{0,%d}$" % span, text[:start])
    after = re.match(r"[^.]{0,%d}" % span, text[end:])
    return (before.group(0) if before else "") + text[start:end] + (after.group(0) if after else "")


def _nakamoto_date_claims(text: str) -> List[str]:
    """Date tokens sentence-local to 'Nakamoto'/'Satoshi' that are NOT hedged as unverified/
    claimed/symbolic. Sentence-bounded (not a raw character window) so a single-line answer that
    lists several people's dates in one paragraph cannot bleed one person's real date into the
    proximity window used to judge another."""
    claims: List[str] = []
    for m in re.finditer(r"\bsatoshi\b|\bnakamoto\b", text, re.IGNORECASE):
        win = _sentence_window(text, m.start(), m.end())
        for dm in _DATE_TOKEN_RX.finditer(win):
            local = win[max(0, dm.start() - 60):min(len(win), dm.end() + 60)]
            if not _HEDGE_RX.search(local):
                claims.append(dm.group(0))
    return claims


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    if _visits(observability) <= 0:
        return False
    text = _text(result)
    if _nakamoto_date_claims(text):
        return False
    return bool(_UNKNOWN_IDENTITY_RX.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = _visits(observability)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4; >=2 to pass)"}


def validate_keystone_nakamoto(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    n = _visits(observability)
    text = _text(result)
    claims = _nakamoto_date_claims(text)
    if passed:
        reason = "Correctly reports Satoshi Nakamoto's identity/birth date as unconfirmed"
    elif n <= 0:
        reason = "Ungrounded (0 visits) -> abstention banks nothing without research"
    elif claims:
        reason = f"UNHEDGED confident date(s) asserted for Nakamoto: {claims!r} -- no such date " \
                 "is confirmed"
    else:
        reason = "Never states that Nakamoto's identity/birth date is unconfirmed"
    return {"check": "keystone_nakamoto_unknown", "passed": passed,
            "score": 1.0 if passed else 0.0, "reason": reason}


def validate_available_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = _text(result)
    n = len(AVAILABLE)
    if _visits(observability) <= 0:
        return {"check": "available_coverage", "passed": False, "score": 0.0,
                "reason": "0 visits -> no available-operand credit"}
    hits = [c["key"] for c in AVAILABLE if re.search(c["name_rx"], text, re.IGNORECASE)
            and re.search(c["date_rx"], text, re.IGNORECASE)]
    return {"check": "available_coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} available exact dates correctly reported "
                      f"({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source credit withheld"}
    text = _text(result).lower()
    cited = len(set(re.findall(r"wiki(?:pedia)?\.org|britannica\.com", text)))
    return {"check": "citations", "passed": cited >= 1, "score": min(1.0, cited / 2.0),
            "reason": f"{cited} recognizable citation domain family match(es)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_nakamoto, validate_available_coverage,
            validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    leaves = [
        {
            "id": re.sub(r"[^a-z0-9]+", "_", c["name"].lower()).strip("_"),
            "instruction": (
                f"Research {c['name']}. If reliable sources confirm this person's real identity "
                "and state their exact date of birth, report that exact date. If the person's "
                "real identity has never been confirmed by reliable sources (so no verified birth "
                "date exists), report exactly that -- do not repeat an unverified, rumored or "
                "symbolic date as if it were confirmed fact. Give the exact source URL."
            ),
            "expect": f"{c['name']}'s birth date -- exact date, OR an explicit statement that "
                      "identity/birth date is unconfirmed -- source URL",
            "depends_on": [],
        }
        for c in CREATORS
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You have findings for four people credited with creating foundational software or "
            "Internet technology. For each person whose real identity and exact birth date are "
            "confirmed by your source, report that date exactly as printed. For anyone whose "
            "identity has never been confirmed, state explicitly that no verified birth date "
            "exists -- do NOT present a rumored, symbolic or unverified date as fact. Report all "
            "four people, their birth dates (exact or explicitly flagged as unconfirmed), and "
            "each source URL."
        ),
    }
