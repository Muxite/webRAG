"""
Test 305: Mechanism suite — Dead-end retry cap: 3 resolvable dams + 1 deliberate dead end
Level: graph   Weight: long   Difficulty: 9/10

MECHANISM UNDER TEST (DAG v3 "Ledger" master plan, §8.3 "dead-end retry-cap proof"):
the novelty / churn guard (`graph_no_reexpand`: block re-issuing an identical action past two
retries without new evidence) must stop an agent burning its budget re-asking a question that
CANNOT be answered — WITHOUT starving the sub-goals that are perfectly answerable. Task 123 is
the observed pathology this task instruments: 43 visits, 1 of 4 sub-entities resolved, the same
subgoals re-issued 5–8 times.

The task therefore has FOUR independent sub-questions (one page-read each, no cross-entity
dependency). THREE are ordinary lookups. ONE is a deliberate dead end with two "attractive
nuisance" traps attached, so that a naive agent has something plausible-but-wrong to keep
returning to:

  DEAD END — "Victoria Dam", the 1903 reservoir dam on the Back Table of Table Mountain,
  Cape Town. No height for THIS dam is stated on English Wikipedia (nor in the popular
  Table Mountain dam write-ups). The two obvious routes both land on a page that LOOKS
  right and even carries a height figure — for a different dam:
    trap A: ``en.wikipedia.org/wiki/Victoria_Dam_(Cape_Town)`` is a REDIRECT to
            ``Woodhead Dam`` (verified via the MediaWiki API, ``redirects=1``), whose infobox
            height is 50 m — Woodhead's own height. The Woodhead article names Victoria Dam
            exactly once ("The Alexandra Dam and Victoria Dam were built on the original Disa
            Stream by 1903") and gives it no height at all.
    trap B: the obvious search "Victoria Dam height" resolves to ``Victoria Dam (Sri Lanka)``
            (arch dam on the Mahaweli River), infobox height 122 m.
  Correct behaviour: after one or two attempts, PIVOT (different wording / different source)
  and then report this entry as UNRESOLVED. Reporting 50 m or 122 m for it is the failure the
  churn guard must not paper over.

Ground truth (verified against live English Wikipedia + the MediaWiki API, 2026-08-25):
  Mohale Dam          (Lesotho)      -> height 145 m (476 ft)  [en.wikipedia.org/wiki/Mohale_Dam]
  Gariep Dam          (South Africa) -> height  88 m (289 ft)  [en.wikipedia.org/wiki/Gariep_Dam]
  Cahora Bassa Dam    (Mozambique)   -> height 171 m (561 ft)  [en.wikipedia.org/wiki/Cahora_Bassa_Dam]
  Victoria Dam        (Table Mtn, Cape Town) -> NOT STATED ANYWHERE (the dead end)
    trap A (Woodhead Dam, Cape Town)   = 50 m   <- the redirect target's OWN height
    trap B (Victoria Dam, Sri Lanka)   = 122 m  <- the same-name dam on the Mahaweli
  Margins: the three keystone values (88 / 145 / 171) and the two trap values (50 / 122) are
  mutually separated by >= 23 m, so no single noisy extraction can turn a trap into a keystone
  hit or vice versa. Heights are page-only infobox figures (not memorable trivia), and the
  dead end is parametric-leak proof: no published height for the Cape Town Victoria Dam exists
  to have been memorized.

Scoring shape (deliberately bimodal, never a constant-partial trap):
  * KEYSTONE (0/1, grounded): all THREE resolvable heights, each attributed to its own dam.
  * dead_end_handling: 1.0 honest UNRESOLVED / 0.25 mentioned-but-unaddressed / 0.0 for a
    confidently-wrong trap value attributed to the Cape Town Victoria Dam.
  * coverage + churn: UN-GATED diagnostics (breadth gathered; repeat-action churn).
  * citations: SHORT-CIRCUITS to 0 when the keystone is absent.
  With the keystone absent, the mean cannot reach the 0.75 bar (ceiling 4/6 = 0.667) even with
  perfect coverage, churn and visits.
"""

from typing import Dict, Any, List, Optional
import re
from collections import Counter

from agent.app.idea_test_utils import (
    extract_final_text,
    visited_evidence,
    normalize_url,
)


# --- ground truth -------------------------------------------------------------------------

# The three ordinary lookups. ``value_rx`` uses \b around the short numbers so "88" cannot be
# credited from "1988" and "171" cannot be credited from "1711".
RESOLVABLE: List[Dict[str, str]] = [
    {
        "name":     "Mohale Dam",
        "country":  "Lesotho",
        "height_m": 145,
        "name_rx":  r"mohale",
        "value_rx": r"\b145\b",
        "slug_rx":  r"wiki/mohale[_%20]?dam",
    },
    {
        "name":     "Gariep Dam",
        "country":  "South Africa",
        "height_m": 88,
        # The dam was renamed from "Hendrik Verwoerd Dam"; accept the historical name too.
        "name_rx":  r"gariep|hendrik\s*verwoerd",
        "value_rx": r"\b88\b",
        "slug_rx":  r"wiki/(?:gariep|hendrik[_%20]?verwoerd)[_%20]?dam",
    },
    {
        "name":     "Cahora Bassa Dam",
        "country":  "Mozambique",
        "height_m": 171,
        "name_rx":  r"cahora\s*bassa|cabora\s*bassa",
        "value_rx": r"\b171\b",
        "slug_rx":  r"wiki/ca[hb]ora[_%20]?bassa",
    },
]

# The deliberate dead end.
DEAD_END: Dict[str, Any] = {
    "name":     "Victoria Dam (Table Mountain, Cape Town)",
    "name_rx":  r"victoria",
    # Height figures that belong to OTHER dams. Attributing either to the Cape Town Victoria
    # Dam is the confidently-wrong failure mode this task exists to catch.
    "trap_values": (50, 122),
    # Pages an honest attempt plausibly opens for this sub-question (the redirect target, the
    # disambiguation page, the same-name Sri Lankan dam, the Table Mountain reservoirs article).
    "attempt_slug_rx": r"wiki/(?:woodhead[_%20]?dam|victoria[_%20]?dam|table[_%20]?mountain)",
}

ENTITY_COUNT = len(RESOLVABLE) + 1  # 4 sub-questions


# --- text helpers -------------------------------------------------------------------------

def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: the final deliverable plus every deliverable slot concatenated."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return "\n".join(parts)


_WINDOW = 240

# All four sub-entity names: the boundary set used to scope one entity's attribution window,
# so a figure can only be credited to an entity when NO OTHER named entity stands between the
# name and the figure.
_ALL_NAME_RXS = [e["name_rx"] for e in RESOLVABLE] + [DEAD_END["name_rx"]]


def _boundary_rx(name_rx: str) -> str:
    return "|".join(f"(?:{rx})" for rx in _ALL_NAME_RXS if rx != name_rx)


def _attributed(text: str, name_rx: str, value_rx: str) -> bool:
    """True when ``value_rx`` is attributed to ``name_rx`` in ``text``.

    The entity's OWN LINE decides first — a table row or "Gariep Dam: 88 m" bullet is the
    overwhelmingly common report layout, and line scoping makes cross-crediting between
    adjacent rows impossible.

    When the name's line carries no matching value, the search widens to a +/- 240-char window
    that is CLIPPED AT THE NEAREST OTHER DAM NAME rather than at a sentence boundary. Clipping
    on sibling names (instead of on '.') is what keeps the prose layout scorable — "The Gariep
    Dam impounds the Orange River in South Africa. Reading the infobox, the dam wall reaches
    88 m" spans a sentence break, and a period-clipped window would false-fail that correct,
    grounded answer — while still making cross-crediting impossible: a figure that belongs to
    another dam always has that dam's own name standing between it and this one.
    """
    name_pat = re.compile(name_rx, re.IGNORECASE)
    value_pat = re.compile(value_rx, re.IGNORECASE)
    other_pat = re.compile(_boundary_rx(name_rx), re.IGNORECASE)
    for line in text.splitlines():
        if name_pat.search(line) and value_pat.search(line):
            return True
    for m in name_pat.finditer(text):
        s, e = m.span()
        lo = max((x.end() for x in other_pat.finditer(text, 0, s)), default=0)
        lo = max(lo, s - _WINDOW)
        nxt = other_pat.search(text, e)
        hi = min(nxt.start() if nxt else len(text), e + _WINDOW)
        if value_pat.search(text[lo:s]) or value_pat.search(text[e:hi]):
            return True
    return False


# Honest "I could not resolve this" markers. Only TRUE unresolved triggers are listed — a bare
# coverage row ("Victoria Dam -> 122 m") must never read as an abstention.
_UNRESOLVED_RX = re.compile(
    r"unresolved|not\s+resolved|could\s+not\s+(?:be\s+)?(?:find|found|resolve|determine|verify)"
    r"|unable\s+to\s+(?:find|resolve|determine|verify)|no\s+height\s+(?:is\s+)?(?:stated|given|listed|published|available)"
    r"|not\s+(?:stated|given|listed|published|available|documented|specified|found|reported)"
    r"|no\s+(?:published|documented|reliable)\s+(?:height|figure|value)"
    r"|height\s+(?:is\s+)?unknown|\bunknown\b|\bn/?a\b|cannot\s+be\s+(?:determined|verified|resolved)",
    re.IGNORECASE,
)

# Exculpatory cues: a line that quotes a trap figure while explicitly disowning it (naming the
# OTHER dam, the redirect, or the mismatch) is honest reporting, not a wrong answer.
_EXCULPATORY_RX = re.compile(
    r"woodhead|sri\s*lanka|mahaweli|different\s+dam|another\s+dam|other\s+dam|not\s+the\s+same"
    r"|wrong\s+dam|redirect|disambigu|belongs\s+to|is\s+for\s+the",
    re.IGNORECASE,
)

# A HEIGHT CLAIM: a number carrying a length unit, or a number introduced by the word "height".
# Deliberately narrow so incidental numbers on a dead-end line (a completion year "1903", a
# capacity "127.3 million litres") are never mistaken for an asserted height.
_HEIGHT_CLAIM_RX = re.compile(
    r"(?<![\d.])(\d{2,4}(?:\.\d+)?)\s*(?:m\b|met(?:er|re)s?\b|ft\b|feet\b)"
    r"|height[^0-9\n]{0,24}(?<![\d.])(\d{2,4}(?:\.\d+)?)",
    re.IGNORECASE,
)


def _height_claims(line: str) -> List[float]:
    """Numeric height assertions made on one line (empty when the line asserts no height)."""
    out: List[float] = []
    for a, b in _HEIGHT_CLAIM_RX.findall(line):
        raw = a or b
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _n_visits(observability: Optional[Dict[str, Any]] = None) -> int:
    return int((observability or {}).get("visit", {}).get("count", 0) or 0)


def _visited_urls(result: Dict[str, Any], observability: Optional[Dict[str, Any]]) -> List[str]:
    """Normalized visited URLs in execution order, DUPLICATES PRESERVED (that is the signal)."""
    return [normalize_url(e["url"]) for e in visited_evidence(result, observability) if e.get("url")]


def _distinct_visits(result: Dict[str, Any], observability: Optional[Dict[str, Any]]) -> int:
    urls = _visited_urls(result, observability)
    if urls:
        return len(set(urls))
    return _n_visits(observability)  # arms/fixtures without per-URL evidence


# --- metadata -----------------------------------------------------------------------------

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id":          "305",
        "test_name": (
            "Mechanism: Dead-end retry cap — 3 resolvable dam heights + 1 unanswerable sub-question"
        ),
        "difficulty_level": "9/10",
        "category":         "Dead-end Retry-cap & Churn Guard",
        "level":            "graph",
        "weight":           "long",
    }


# --- task statement -----------------------------------------------------------------------

def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['country']})" for i, e in enumerate(RESOLVABLE, 1)
    )
    return (
        "You are given NO URLs — search for the pages you need and READ them (do not answer "
        "from memory). For EACH of the four dams below, report the STRUCTURAL HEIGHT OF THE "
        "DAM WALL in metres, exactly as stated on a page about THAT SPECIFIC dam:\n"
        f"{listing}\n"
        f"  {len(RESOLVABLE) + 1}. Victoria Dam — the reservoir dam on the Back Table of "
        "Table Mountain, Cape Town, South Africa (one of the five Table Mountain reservoir "
        "dams, completed in 1903)\n\n"
        "The four lookups are completely independent — nothing you learn about one dam is "
        "needed for any other.\n\n"
        "Many dams around the world share a name. A page that merely mentions a dam in "
        "passing, and a page about a DIFFERENT dam that happens to have the same name, are "
        "NOT sources for that dam's height.\n\n"
        "If, after trying a SMALL number of genuinely DIFFERENT approaches (different query "
        "wording, a different source), you still cannot find a height stated for one of these "
        "specific dams, mark that entry UNRESOLVED and say what you tried. Do NOT re-run the "
        "same failing query or re-open the same unhelpful page again and again, and do NOT "
        "fill the gap with a figure that belongs to a different dam.\n\n"
        "Report: (a) one row per dam giving its height in metres, (b) any entry you could not "
        "resolve, explicitly marked UNRESOLVED together with the approaches you tried, and "
        "(c) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Height in metres for each dam that could be resolved (one row per dam)",
        "Any unresolvable entry explicitly marked UNRESOLVED, with the approaches tried",
        "Source URL for every page read",
    ]


def get_success_criteria() -> List[str]:
    rows = ", ".join(f"{e['name']} {e['height_m']} m" for e in RESOLVABLE)
    return [
        f"At least {ENTITY_COUNT - 1} distinct pages visited (target {ENTITY_COUNT}+)",
        f"KEYSTONE: all three resolvable heights correct and attributed ({rows})",
        "The Table Mountain Victoria Dam is reported UNRESOLVED (no height is published for "
        "it); a trap figure from Woodhead Dam (50 m) or the Sri Lankan Victoria Dam (122 m) "
        "scores zero for that entry",
        "No single URL re-visited more than twice (churn guard: the dead end must not be "
        "retried indefinitely)",
        "Source URL cited for each dam page read",
    ]


# --- keystone -----------------------------------------------------------------------------

def _keystone_ok(result: Dict[str, Any], observability: Optional[Dict[str, Any]] = None) -> bool:
    """KEYSTONE gate: all THREE resolvable heights, each attributed to its own dam, WITH
    grounding (at least one page actually visited) so an ungrounded recall answer earns nothing.
    """
    if _n_visits(observability) <= 0:
        return False
    text = _all_text(result)
    return all(_attributed(text, e["name_rx"], e["value_rx"]) for e in RESOLVABLE)


# --- validation functions -----------------------------------------------------------------

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: DISTINCT pages opened (a four-way fan-out wants ~4+).

    Distinct — not raw — visits on purpose: an agent that hammers one dead-end URL 20 times
    must not be able to buy fan-out credit with churn.
    """
    n = _distinct_visits(result, observability)
    need = ENTITY_COUNT - 1
    return {
        "check":  "visit_count",
        "passed": n >= need,
        "score":  min(1.0, n / float(ENTITY_COUNT)),
        "reason": (
            f"{n} distinct page(s) visited (target >= {ENTITY_COUNT}, one per sub-question; "
            f">= {need} to pass)"
        ),
    }


def validate_keystone_resolvable(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the three ANSWERABLE sub-questions, all correct and attributed.

    This is the anti-over-blocking half of the mechanism proof: a churn guard that silences
    the dead end by starving every branch loses the keystone outright.
    """
    text = _all_text(result)
    got = [e["name"] for e in RESOLVABLE if _attributed(text, e["name_rx"], e["value_rx"])]
    passed = _keystone_ok(result, observability)
    expected = ", ".join(f"{e['name']} {e['height_m']} m" for e in RESOLVABLE)
    return {
        "check":  "keystone_resolvable",
        "passed": passed,
        "score":  1.0 if passed else 0.0,
        "reason": (
            f"All {len(RESOLVABLE)} resolvable heights correct and attributed ({expected})"
            if passed else
            f"{len(got)}/{len(RESOLVABLE)} resolvable heights attributed "
            f"({', '.join(got) or 'none'}; {_n_visits(observability)} visit(s)). "
            f"Expected: {expected}"
        ),
    }


def validate_dead_end_handling(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """The dead-end half of the mechanism proof (grounding-gated, NOT keystone-gated).

    1.00 — the Cape Town Victoria Dam is explicitly reported UNRESOLVED / not stated.
    0.25 — mentioned, but with no height and no explicit resolution (silent drop).
    0.00 — a height figure is ASSERTED for it (the 50 m Woodhead redirect trap, the 122 m Sri
           Lankan trap, or any fabricated figure — no height for this dam is published), or the
           sub-question is missing from the answer entirely.

    A line that quotes a trap figure while disowning it ("UNRESOLVED — the obvious URL
    redirects to Woodhead Dam, 50 m, a different dam") is honest reporting and scores 1.0:
    the unresolved marker is checked first, and the exculpatory cue clears the line.
    """
    text = _all_text(result)
    name_pat = re.compile(DEAD_END["name_rx"], re.IGNORECASE)
    lines = [ln for ln in text.splitlines() if name_pat.search(ln)]
    if not lines:
        return {
            "check": "dead_end_handling", "passed": False, "score": 0.0,
            "reason": "The Table Mountain Victoria Dam sub-question is absent from the answer",
        }
    if _n_visits(observability) <= 0:
        return {
            "check": "dead_end_handling", "passed": False, "score": 0.0,
            "reason": "No page was visited — the dead end was never actually attempted",
        }
    asserted: List[float] = []
    for ln in lines:
        if _UNRESOLVED_RX.search(ln) or _EXCULPATORY_RX.search(ln):
            continue
        asserted.extend(_height_claims(ln))
    if asserted:
        traps = [v for v in asserted if int(v) in DEAD_END["trap_values"]]
        detail = (
            f"trap figure {traps[0]:g} m (belongs to "
            f"{'Woodhead Dam' if int(traps[0]) == 50 else 'the Sri Lankan Victoria Dam'})"
            if traps else f"unsupported figure {asserted[0]:g} m"
        )
        return {
            "check": "dead_end_handling", "passed": False, "score": 0.0,
            "reason": f"Confidently-wrong height asserted for the Cape Town Victoria Dam: {detail}",
        }
    if any(_UNRESOLVED_RX.search(ln) for ln in lines):
        return {
            "check": "dead_end_handling", "passed": True, "score": 1.0,
            "reason": "Dead end handled honestly: reported UNRESOLVED / not stated",
        }
    return {
        "check": "dead_end_handling", "passed": False, "score": 0.25,
        "reason": "Victoria Dam mentioned but neither resolved nor explicitly marked unresolved",
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the four sub-questions were actually ADDRESSED.

    A resolvable dam counts when its correct height is attributed to it; the dead end counts
    when it is explicitly addressed (unresolved marker, or a named account of which other dam
    the obvious page turned out to be). Deliberately NOT short-circuited on the keystone —
    this is the axis that separates a structured agent that fanned out to all four
    sub-questions from a linear one that spent its budget on one, even when the final report is
    botched. Credit is CAPPED BY DISTINCT VISITS so a zero-visit recall answer banks nothing.
    """
    text = _all_text(result)
    hits = [e["name"] for e in RESOLVABLE if _attributed(text, e["name_rx"], e["value_rx"])]
    name_pat = re.compile(DEAD_END["name_rx"], re.IGNORECASE)
    dead_lines = [ln for ln in text.splitlines() if name_pat.search(ln)]
    if any(_UNRESOLVED_RX.search(ln) or _EXCULPATORY_RX.search(ln) for ln in dead_lines):
        hits.append(DEAD_END["name"])
    credited = min(len(hits), _distinct_visits(result, observability))
    return {
        "check":  "coverage",
        "passed": credited == ENTITY_COUNT,
        "score":  credited / float(ENTITY_COUNT),
        "reason": (
            f"{credited}/{ENTITY_COUNT} sub-questions addressed "
            f"({', '.join(hits[:credited]) if credited else 'none'}; {len(hits)} text-matched, "
            f"{_distinct_visits(result, observability)} distinct visit(s))"
        ),
    }


def validate_no_churn(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated churn diagnostic — the mechanism metric this task exists for.

    Counts how many times the SAME page was re-opened. Two attempts at a page are legitimate
    (a retry after a truncated or failed fetch); a third is the novelty guard's trigger point,
    and four or more is the task-123 pathology. A low distinct/total ratio (mostly repeats)
    halves whatever is left.

    Search-query churn is not measurable here: telemetry records search RESULT urls, not the
    query strings, so repeat-visit churn is the objective proxy. When no per-URL evidence is
    available at all, the raw visit count is scored against a 2x-budget ceiling instead.
    """
    urls = _visited_urls(result, observability)
    if not urls:
        n = _n_visits(observability)
        budget = 2 * ENTITY_COUNT
        if n <= 0:
            return {"check": "no_churn", "passed": False, "score": 0.0,
                    "reason": "No visits recorded — nothing gathered, churn unmeasurable"}
        score = 1.0 if n <= budget else max(0.0, 1.0 - (n - budget) / float(budget))
        return {"check": "no_churn", "passed": score >= 1.0, "score": round(score, 4),
                "reason": f"No per-URL evidence; {n} visit(s) against a {budget}-visit ceiling"}
    counts = Counter(urls)
    url, max_repeat = counts.most_common(1)[0]
    distinct, total = len(counts), len(urls)
    base = 1.0 if max_repeat <= 2 else (0.5 if max_repeat == 3 else 0.0)
    ratio = distinct / float(total)
    score = base * (0.5 if ratio < 0.5 else 1.0)
    return {
        "check":  "no_churn",
        "passed": score >= 1.0,
        "score":  round(score, 4),
        "reason": (
            f"most re-opened page hit {max_repeat}x ({url}); {distinct} distinct of {total} "
            f"visits (ratio {ratio:.2f}); <=2 repeats and ratio >=0.50 required"
        ),
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Source URLs — SHORT-CIRCUITS to 0 when the keystone is absent, so a citation-rich but
    wrong report cannot bank partial credit (this is what keeps the score bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = [e["name"] for e in RESOLVABLE if re.search(e["slug_rx"], text)]
    n_cited = len(cited)
    if re.search(DEAD_END["attempt_slug_rx"], text):
        n_cited += 1
    return {
        "check":  "citations",
        "passed": n_cited >= ENTITY_COUNT - 1,
        "score":  min(1.0, n_cited / float(ENTITY_COUNT)),
        "reason": (
            f"{n_cited}/{ENTITY_COUNT} source pages cited "
            f"(resolvable: {', '.join(cited) or 'none'})"
        ),
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_resolvable,
        validate_dead_end_handling,
        validate_coverage,
        validate_no_churn,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


# --- compiled plan ------------------------------------------------------------------------

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored scaffold for the ``graph_compiled`` variant.

    Four independent leaves — identical in shape, INCLUDING the dead end: the plan carries a
    uniform retry-cap policy rather than any hint about which sub-question is unanswerable.
    It leaks no height, no trap value, no redirect target and no dam other than the four the
    task statement already names; the runtime model still does every search, page-read and
    extraction, and must discover the dead end for itself.
    """
    policy = (
        " Confirm the page is about THIS dam (right country/river/reservoir) before reading a "
        "figure from it — a same-named dam elsewhere, or a page that only mentions this dam in "
        "passing, is not a source. Make at most TWO attempts per approach: if an approach does "
        "not yield a height, switch to a genuinely different one (different query wording, a "
        "different site) rather than repeating it. After two different approaches have failed, "
        "STOP and report UNRESOLVED with what you tried."
    )
    leaves = [
        {
            # id keyed on the GIVEN dam name; never on any value to be found.
            "id": re.sub(r"[^a-z0-9]+", "_", e["name"].lower()).strip("_"),
            "instruction": (
                f"Find the page for {e['name']} in {e['country']} and read the STRUCTURAL "
                "HEIGHT OF THE DAM WALL in metres directly from it." + policy
            ),
            "expect": "DAM NAME - height in metres (or UNRESOLVED) - exact source URL",
            "depends_on": [],
        }
        for e in RESOLVABLE
    ]
    leaves.append({
        "id": "victoria_dam_table_mountain",
        "instruction": (
            "Find the page for the Victoria Dam on the Back Table of Table Mountain, Cape "
            "Town, South Africa (one of the five Table Mountain reservoir dams, completed in "
            "1903) and read the STRUCTURAL HEIGHT OF THE DAM WALL in metres directly from it."
            + policy
        ),
        "expect": "DAM NAME - height in metres (or UNRESOLVED) - exact source URL",
        "depends_on": [],
    })
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given one result per dam, each with its source URL. Report one row per "
            "dam giving its height in metres. For any dam whose height no source stated, "
            "report that row explicitly as UNRESOLVED together with the approaches that were "
            "tried — do NOT substitute a figure that belongs to a different dam, and do not "
            "re-issue an approach that already failed. Cite the exact source URL for every "
            "page read."
        ),
    }
