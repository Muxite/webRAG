"""
Test 161: Tier 5 (graph) — SEQUENTIAL-PREFIX then FAN-OUT ("seqfan"): rail network line roster.
Level: graph   Weight: long   Difficulty: 9/10   Category: sequential_prefix_fanout_merge

The suite's existing breadth tasks (152-158) are sets of independent ONE-HOP page reads whose
members are handed to the agent in the prompt. This task is the missing shape: the candidate set
is UNKNOWABLE from the mandate and must be *discovered* by a dependent chain first, and each
discovered member then needs its own multi-step subtask.

    PHASE 1 - SEQUENTIAL PREFIX (3 dependent hops, width 1, nothing parallelisable)
      HOP 1  Emerson Park railway station (the only given)      -> the named LINE serving it
      HOP 2  that line's page                                   -> the NETWORK/brand it belongs to
      HOP 3  that network's page                                -> the ROSTER: every named line of
                                                                   the network + its termini
    PHASE 2 - FAN-OUT (6 mutually independent branches, one per rostered line)
      per line: open the line page, then open EACH of its terminus stations' pages and read the
      year that station FIRST opened to passengers; keep the line's most recent such year.
      (2-5 page reads per branch; no branch consumes any other branch's output.)
    PHASE 3 - MERGE
      argmax across the six lines: which line has the most recently opened terminus station.

Ground truth (verified against live English Wikipedia, 2026-08-29):

  Prefix
    https://en.wikipedia.org/wiki/Emerson_Park_railway_station
      "It is the only intermediate station on the single-track Liberty line" ... "London
      Overground station on Butts Green Road"                     -> HOP 1 = Liberty line
    https://en.wikipedia.org/wiki/Liberty_line
      infobox Termini "Romford" / "Upminster"; London Overground  -> HOP 2 = London Overground
    https://en.wikipedia.org/wiki/London_Overground
      the article's line table lists all SIX branded lines with their routes:
        Lioness     "Watford Junction to Euston"
        Mildmay     "Stratford to Richmond/Clapham Junction"
        Windrush    "Highbury & Islington to Clapham Junction/New Cross/Crystal Palace/West Croydon"
                    (line infobox also lists Dalston Junction and Battersea Park as termini)
        Weaver      "Liverpool Street to Cheshunt/Enfield Town/Chingford"
        Suffragette "Gospel Oak to Barking Riverside"
        Liberty     "Romford to Upminster"                        -> HOP 3 = the candidate set

  Fan-out (each terminus' FIRST opening to passengers, read off the station's own infobox)
    LIONESS      Euston 1837 | Watford Junction "20 July 1837 First station, named Watford,
                 opened" / relocated-and-renamed 5 May 1858            -> newest 1858 (or 1837)
    MILDMAY      Stratford "20 June 1839 Opened by ECR" | Richmond "27 July 1846 Opened as
                 Terminus (R&WER)" | Clapham Junction "2 March 1863"   -> newest 1863
    WINDRUSH     West Croydon 1839 | New Cross 1849 | Highbury & Islington "26 September 1850" |
                 Crystal Palace 1854 | Clapham Junction 1863 | Dalston Junction "1 November 1865"
                 (closed 1986, reopened 27 April 2010) | Battersea Park "1 May 1867 Opened"
                                                                       -> newest 1867
    WEAVER       Cheshunt 1846 | Enfield Town 1849 | Chingford "17 November 1873 Opened" |
                 London Liverpool Street "2 October 1874 Opened"       -> newest 1874
    SUFFRAGETTE  Gospel Oak "2 January 1860" | Barking Riverside "18 July 2022"  -> newest 2022
    LIBERTY      Romford 1839 | Upminster "Opened: 1 May 1885"          -> newest 1885

  KEYSTONE (argmax): the SUFFRAGETTE line, terminus BARKING RIVERSIDE, opened 2022.
      https://en.wikipedia.org/wiki/Barking_Riverside_railway_station -> "18 July 2022"; the page
      calls it "the eastern terminus of the Suffragette line of the London Overground".

  Margin: 2022 vs the runner-up 1885 (Liberty / Upminster) = 137 years. Both are exact,
  undisputed calendar dates - no rounding, no unit conversion, no source disagreement. Even under
  the most adversarial misreading available on these pages (counting Dalston Junction's 2010
  RE-opening as Windrush's value) the keystone still wins by 12 years, so no single noisy
  extraction can flip the argmax.

  Leak resistance: the six branded line names date from the 2024 renaming and appear nowhere in
  the mandate; Barking Riverside is a 2022 station on a page-only infobox date. A model that
  answers from parametric memory of "London Overground" cannot produce the roster, and the
  mandate never names the network at all.

Solvable linearly: a sequential ReAct agent can walk the prefix and then the six branches in any
order and reach the same answer - it simply pays more turns holding six intermediates. Nothing in
the shape rewards a particular engine.
"""

from typing import Dict, Any, List
import re

from agent.app.idea_test_utils import extract_final_text


START_STATION = "Emerson Park railway station"

# One row per DISCOVERED line. ``years`` holds every defensible reading of that line's most
# recently first-opened terminus (see the docstring's margin analysis); ``slug`` accepts the
# branded title and the historical article title it redirects between.
LINES: List[Dict[str, Any]] = [
    {"line": "Lioness", "name_rx": r"\blioness\b", "terminus": "Watford Junction",
     "years": ["1858", "1837"], "slug": r"wiki/(lioness_line|watford_dc_line)"},
    {"line": "Mildmay", "name_rx": r"\bmildmay\b", "terminus": "Clapham Junction",
     "years": ["1863"], "slug": r"wiki/(mildmay_line|north_london_line)"},
    {"line": "Windrush", "name_rx": r"\bwindrush\b", "terminus": "Battersea Park",
     "years": ["1867", "1865", "2010"], "slug": r"wiki/(windrush_line|east_london_line)"},
    {"line": "Weaver", "name_rx": r"\bweaver\b", "terminus": "London Liverpool Street",
     "years": ["1874"], "slug": r"wiki/(weaver_line|lea_valley_lines)"},
    {"line": "Suffragette", "name_rx": r"\bsuffragette\b", "terminus": "Barking Riverside",
     "years": ["2022"], "slug": r"wiki/(suffragette_line|gospel_oak_to_barking_line)"},
    {"line": "Liberty", "name_rx": r"\bliberty\b", "terminus": "Upminster",
     "years": ["1885"], "slug": r"wiki/(liberty_line|romford_to_upminster_line)"},
]

N_LINES = len(LINES)
EVIDENCE_FLOOR = 5

_SUPERLATIVE = r"most\s+recently\s+opened|most\s+recent|newest|latest|youngest"
_KEYSTONE_NEAR = re.compile(
    rf"({_SUPERLATIVE})[^.]{{0,90}}(suffragette|barking\s+riverside)"
    rf"|(suffragette|barking\s+riverside)[^.]{{0,110}}({_SUPERLATIVE})",
    re.IGNORECASE,
)
_KEYSTONE_STATION = re.compile(r"barking\s+riverside", re.IGNORECASE)
_KEYSTONE_YEAR = re.compile(r"\b2022\b")
# A stated conclusion naming a RIVAL line as the network-wide winner contradicts the keystone even
# if a correct per-line row happens to sit in the same report.
_RIVAL_VERDICT = re.compile(
    r"(answer|winner|wins|therefore|conclusion|conclude)"
    r"(?:(?!suffragette|barking)[^.]){0,50}?"
    r"\b(lioness|mildmay|windrush|weaver|liberty)\s+line\b",
    re.IGNORECASE,
)
_NETWORK = re.compile(r"london\s+overground", re.IGNORECASE)
_HOP1 = re.compile(r"\bliberty\s+line\b|emerson\s+park[^.]{0,80}\bliberty\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    """Harness metadata for the task registry.

    Returns:
        dict: test_id, test_name, difficulty_level, category, level and weight.
    """
    return {
        "test_id": "161",
        "test_name": "Tier 5: sequential prefix -> 6-way fan-out -> argmax (newest line terminus)",
        "difficulty_level": "9/10",
        "category": "Sequential Prefix + Fan-out & Aggregation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    """The mandate. Names ONE entity; the candidate set must be discovered.

    Returns:
        str: the task text handed to the agent (no URLs, no line names, no answer).
    """
    return (
        "You are given NO URLs and exactly one starting point: "
        f"{START_STATION}, in east London.\n\n"
        "1. From that station's own page, identify the named railway LINE that serves it.\n"
        "2. From that line's page, identify the wider NETWORK (the operating brand) the line "
        "belongs to.\n"
        "3. From that network's page, read off its full roster: EVERY named line of the network "
        "together with the terminus station(s) at each end of each line. Take the roster from the "
        "page, not from memory.\n"
        "4. Then, for EACH line on that roster independently: open the page of every one of its "
        "terminus stations and read the year that station FIRST opened to passengers (its "
        "original opening - ignore later closures, re-openings, rebuildings and added platforms). "
        "For each line keep the terminus whose first opening year is the MOST RECENT.\n"
        "5. Finally, AGGREGATE across the whole roster and report:\n"
        "   (a) which line has the most recently opened terminus station, naming that station and "
        "its opening year;\n"
        "   (b) the full table line -> its most recently opened terminus -> that terminus' "
        "opening year, for every line on the roster;\n"
        "   (c) the source URL of every line page and station page you read."
    )


def get_required_deliverables() -> List[str]:
    """Returns:
        list[str]: the deliverables the answer must contain.
    """
    return [
        "The line whose terminus opened most recently, plus that station and its opening year",
        "The identity of the network and of the line serving the starting station",
        f"One row per rostered line: line -> newest terminus -> opening year ({N_LINES} rows)",
        "Source URLs for the line pages and the terminus station pages read",
    ]


def get_success_criteria() -> List[str]:
    """Returns:
        list[str]: human-readable pass conditions mirroring the validators.
    """
    return [
        "Resolves the 3-hop prefix: starting station -> its line -> the network -> the roster",
        "At least 6 pages visited (prefix chain plus the fan-out branches)",
        "Names the correct line and terminus as the most recently opened (Suffragette line, "
        "Barking Riverside, 2022)",
        f"Reports a newest-terminus year for at least {EVIDENCE_FLOOR} of the {N_LINES} lines",
        "Cites the line and station pages actually read",
    ]


def _coverage_hits(result: Dict[str, Any], observability: Dict[str, Any]) -> List[str]:
    """Lines for which a name AND an accepted newest-terminus year are both present.

    Args:
        result: the harness result dict.
        observability: the harness observability dict (used only for the visit cap).

    Returns:
        list[str]: names of the resolved lines, capped at the number of pages actually visited so
            that a zero-visit recall answer banks no breadth.
    """
    text = extract_final_text(result)
    hits = [
        entry["line"] for entry in LINES
        if re.search(entry["name_rx"], text, re.IGNORECASE)
        and any(re.search(r"\b" + y + r"\b", text) for y in entry["years"])
    ]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return hits[:n_visits]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Whether the argmax answer is present, grounded and evidenced.

    Requires (a) at least one real page visit, (b) the argmax claim naming the Suffragette line /
    Barking Riverside with 2022, (c) no stated verdict naming a rival line as the winner, and
    (d) an evidence floor of resolved branches - a lucky guess backed by two rows is not a solve.

    Args:
        result: the harness result dict.
        observability: the harness observability dict.

    Returns:
        bool: True when the keystone is earned.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = extract_final_text(result)
    if not (_KEYSTONE_NEAR.search(text) and _KEYSTONE_STATION.search(text)
            and _KEYSTONE_YEAR.search(text)):
        return False
    if _RIVAL_VERDICT.search(text):
        return False
    return len(_coverage_hits(result, observability)) >= EVIDENCE_FLOOR


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Grounding/effort gate: the prefix plus six branches cannot be done from memory.

    Args:
        result: the harness result dict (unused).
        observability: the harness observability dict.

    Returns:
        dict: check/passed/score/reason.
    """
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 6, "score": min(1.0, n / 9.0),
            "reason": f"{n} visit(s) (target >=9 for a 3-hop prefix + {N_LINES} branches; "
                      ">=6 to pass)"}


def validate_keystone_newest_terminus(result: Dict[str, Any],
                                      observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the Suffragette line / Barking Riverside / 2022 argmax.

    Args:
        result: the harness result dict.
        observability: the harness observability dict.

    Returns:
        dict: check/passed/score/reason.
    """
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_newest_terminus", "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": "Most recently opened terminus = Barking Riverside (2022), Suffragette line"
                      if passed else
                      "Argmax answer (Suffragette line / Barking Riverside / 2022) missing, "
                      "contradicted, ungrounded or below the evidence floor"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the discovered lines were actually resolved.

    Not short-circuited on the keystone: it reports how much of the fan-out was genuinely
    gathered even when the final argmax is botched.

    Args:
        result: the harness result dict.
        observability: the harness observability dict.

    Returns:
        dict: check/passed/score/reason, score = resolved/N_LINES.
    """
    hits = _coverage_hits(result, observability)
    return {"check": "line_coverage", "passed": len(hits) == N_LINES, "score": len(hits) / N_LINES,
            "reason": f"{len(hits)}/{N_LINES} lines resolved with a newest-terminus year "
                      f"({', '.join(hits) or 'none'})"}


def validate_prefix_chain(result: Dict[str, Any],
                          observability: Dict[str, Any]) -> Dict[str, Any]:
    """Intermediate check (GATED on the keystone): was the 3-hop discovery prefix reported?

    Args:
        result: the harness result dict.
        observability: the harness observability dict.

    Returns:
        dict: check/passed/score/reason, score = resolved hops / 3.
    """
    if not _keystone_ok(result, observability):
        return {"check": "prefix_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> prefix hops not credited"}
    text = extract_final_text(result)
    hops = {
        "line": bool(_HOP1.search(text)),
        "network": bool(_NETWORK.search(text)),
        "roster": sum(bool(re.search(e["name_rx"], text, re.IGNORECASE)) for e in LINES) >= 5,
    }
    done = sum(hops.values())
    return {"check": "prefix_chain", "passed": done == 3, "score": done / 3.0,
            "reason": f"{done}/3 prefix hops reported "
                      f"({', '.join(k for k, v in hops.items() if v) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Citation check (GATED on the keystone): line pages and the winning station page.

    Args:
        result: the harness result dict.
        observability: the harness observability dict.

    Returns:
        dict: check/passed/score/reason, score = cited slugs / (N_LINES + 1).
    """
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    slugs = [e["slug"] for e in LINES] + [r"wiki/barking_riverside"]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 4, "score": cited / len(slugs),
            "reason": f"{cited}/{len(slugs)} line/station pages cited"}


def get_validation_functions() -> List[callable]:
    """Returns:
        list[callable]: the deterministic validators, keystone first among the gated set.
    """
    return [validate_visits, validate_keystone_newest_terminus, validate_coverage,
            validate_prefix_chain, validate_citations]


def get_llm_validation_function() -> callable:
    """Returns:
        None: this task is fully deterministic.
    """
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored seqfan scaffold: a width-1 chain of 3 leaves, then a wave of 6.

    The plan encodes only STRUCTURE - a discovery chain templated on each predecessor
    ({line}, {network}, {roster}) followed by six ordinal branches over the roster the agent
    itself produced. It names only the GIVEN starting station and leaks no line name, no network,
    no terminus, no year and no argmax.

    Returns:
        dict: schema-v2 plan with ``leaves`` (id/instruction/expect/depends_on) and
            ``aggregation``.
    """
    chain = [
        {
            "id": "line",
            "instruction": (
                f"Open the Wikipedia page for {START_STATION} (east London). Read WHICH named "
                "railway line serves this station, exactly as the page names it. Report that line "
                "and the exact source URL. Do not guess from memory; report no other fact."
            ),
            "expect": "The named line serving the starting station — source URL",
            "depends_on": [],
        },
        {
            "id": "network",
            "instruction": (
                "Open the Wikipedia page of the line identified in the previous step ({line}). "
                "Read WHICH wider railway network or operating brand that line is part of. Report "
                "the network and the exact source URL. Do not guess from memory."
            ),
            "expect": "The network the line belongs to — source URL",
            "depends_on": ["line"],
        },
        {
            "id": "roster",
            "instruction": (
                "Open the Wikipedia page of the network identified in the previous step "
                "({network}). It contains a table of the network's named lines with the route of "
                "each. Report EVERY line on that roster, in the page's own order, with the "
                "terminus station(s) at each end of each line, plus the source URL. Copy the "
                "roster from the page; do not add or drop lines from memory."
            ),
            "expect": "The complete roster: each line and its terminus stations — source URL",
            "depends_on": ["network"],
        },
    ]
    branches = [
        {
            "id": f"branch_{i}",
            "instruction": (
                f"From the roster produced in the previous step ({{roster}}), take line number {i} "
                "in that roster's order. Open that line's own page to confirm its terminus "
                "stations, then open the page of EACH of those terminus stations and read the year "
                "that station FIRST opened to passengers (its original opening; ignore later "
                "closures, re-openings, rebuildings and added platforms). Report the line's name, "
                "the terminus whose first opening year is the most recent, that year, and the "
                "source URL of every page you read. Work on this line only."
            ),
            "expect": ("Line name — its most recently first-opened terminus — that opening year "
                       "— source URLs"),
            "depends_on": ["roster"],
        }
        for i in range(1, N_LINES + 1)
    ]
    return {
        "leaves": chain + branches,
        "aggregation": (
            "You now have, for every line on the network's roster, the terminus station that "
            "opened most recently and its opening year. AGGREGATE across the whole roster: take "
            "the MAXIMUM of those years. Report (a) which line has the most recently opened "
            "terminus station, naming that station and its opening year and stating explicitly "
            "that it is the most recent on the network; (b) the full table line -> newest "
            "terminus -> opening year for every rostered line; and (c) the line serving the "
            "starting station and the network, citing every source URL. Do not drop a line from "
            "the table, and do not substitute a line remembered from training data."
        ),
    }
