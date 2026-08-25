"""
Test 302: Mechanism suite — DUPLICATED / SYNDICATED URLs (illusory corroboration)
Level: integration   Weight: long   Difficulty: 9/10

Mechanism under test (DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md §8.3, item 2): the agent is
handed THREE distinct domains that all republish the SAME underlying English-Wikipedia
article on Crater Lake, and is asked how many INDEPENDENT sources actually support a depth
claim. A weak/linear agent reads three URLs, sees the same number three times and reports
"three sources confirm it" — that is the defect this task is built to catch. The correct
answer is that the three URLs are ONE source republished, and that the genuinely independent
authority (the U.S. National Park Service's own page) prints a DIFFERENT figure.

This is exactly the shape the evidence-ledger / deficit injector is supposed to handle:
source *provenance* has to be tracked per-evidence, and "three copies" must not retire the
requirement "corroborate this number from an independent source".

Ground truth (verified live, 2026-08-25 — every figure below re-fetched from the page named):

  | Page                                              | Depth figure printed        | Provenance                                     |
  |---------------------------------------------------|-----------------------------|------------------------------------------------|
  | https://en.wikipedia.org/wiki/Crater_Lake          | "1,949 feet (594 m)"        | ORIGIN of the family; cites Bacon/Gardner/Mayer |
  |                                                    |                             | et al., GSA Bulletin, June 2002 (2000 sonar)   |
  | https://alchetron.com/Crater-Lake                  | "1,949 feet (594 m)"        | footer credits "Crater Lake Wikipedia", CC BY-SA|
  | https://dbpedia.org/page/Crater_Lake               | dbo:maximumDepth 594.055200 | "This content was extracted from Wikipedia and  |
  |                                                    |                             | is licensed under CC BY-SA 4.0 International"   |
  | https://www.nps.gov/crla/learn/nature/crater-lake.htm | "1,943 ft (592 m)"       | INDEPENDENT — NPS's own text, not Wikipedia-derived |

  Keystone margin: the two figures differ by 6 ft / 2 m (1,949 vs 1,943; 594 vs 592) and by
  literal digit string, so no plausible extraction noise can collapse one into the other.
  The Wikipedia article does contain the string "1,943" ONCE, buried in a Britannica
  reference entry — which is why the NPS check requires 1,943 to sit near an NPS/park-service
  attribution rather than merely appearing somewhere in the text.

  Parametric-leak resistance: the proof-of-visit token is the number of dives the Deep Rover
  submersible made to the bottom of the lake — **47** — which is printed on the NPS page
  ("Deep Rover made 47 dives to the bottom") and appears NOWHERE in the Wikipedia article
  (verified: the Wikipedia article never names Deep Rover and has no dive count). It is not
  a memorable/aggregated fact, so it can only be produced by opening the NPS page.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ---------------------------------------------------------------------------------------
# The syndication family (single source of truth for statement, validators, compiled plan).
# ---------------------------------------------------------------------------------------
SYNDICATED_URLS: List[Dict[str, str]] = [
    {"label": "en.wikipedia.org",
     "url": "https://en.wikipedia.org/wiki/Crater_Lake",
     "slug": r"en\.wikipedia\.org/wiki/crater_lake|wikipedia"},
    {"label": "alchetron.com",
     "url": "https://alchetron.com/Crater-Lake",
     "slug": r"alchetron"},
    {"label": "dbpedia.org",
     "url": "https://dbpedia.org/page/Crater_Lake",
     "slug": r"dbpedia"},
]

INDEPENDENT_LABEL = "nps.gov (U.S. National Park Service)"
CLAIMED_DEPTH_FT = "1,949"     # the figure the three syndicated copies all carry
NPS_DEPTH_FT = "1,943"         # the figure the independent authority prints
DIVE_COUNT = "47"              # page-only proof-of-visit token (NPS page only)

# --- regexes ---------------------------------------------------------------------------
# Proximity windows use [^.] (newline-tolerant) so a report that puts the label and the value
# on separate lines still matches, while a sentence-ending period still bounds the window.

# Proximity filler. A plain ``[^.]`` window is newline-tolerant but breaks on the dots inside
# URLs ("nps.gov", ".htm") — and every answer to THIS task is dense with URLs, so the plain
# form would reject correct reports. ``_NEAR`` therefore admits a period only when it is
# glued to a following non-space character (URL/decimal punctuation) and still treats a
# sentence-ending ". " as a hard boundary.
_NEAR = r"(?:[^.]|\.(?=\S))"

# The claimed (syndicated) figure, in feet or metres.
_CLAIMED_FIGURE = re.compile(r"\b1,?949\b|\b594\b")

# The independent authority's figure, and it must be ATTRIBUTED to that authority.
_NPS_CUE = r"(nps\.gov|national\s+park\s+service|\bnps\b|park\s+service)"
_NPS_FIGURE_NEAR_NPS = re.compile(
    rf"{_NPS_CUE}{_NEAR}{{0,200}}\b1,?943\b|\b1,?943\b{_NEAR}{{0,200}}{_NPS_CUE}",
    re.IGNORECASE,
)

# Page-only proof of visit: 47 dives by the Deep Rover submersible.
_DIVE_PROOF = re.compile(
    rf"\b47\b{_NEAR}{{0,60}}\bdives?\b|\bdives?\b{_NEAR}{{0,40}}\b47\b"
    rf"|deep\s+rover{_NEAR}{{0,80}}\b47\b",
    re.IGNORECASE,
)

# Duplicate-recognition cue, tied to the syndication family by proximity.
_DUP_CUE = (
    r"mirror(?:s|ed|ing)?|copy|copies|copied|republish(?:ed|es|ing)?|reproduc\w*|"
    r"duplicat\w*|syndicat\w*|derived\s+from|derivative|verbatim|scrape[ds]?|"
    r"same\s+(?:underlying\s+)?(?:source|article|text|content|origin)|not\s+independent|"
    r"single\s+source|one\s+source|counts?\s+as\s+(?:one|1)|same\s+wikipedia"
)
_FAMILY_CUE = r"wikipedia|alchetron|dbpedia"
_DUP_NEAR_FAMILY = re.compile(
    rf"(?:{_DUP_CUE}){_NEAR}{{0,140}}(?:{_FAMILY_CUE})"
    rf"|(?:{_FAMILY_CUE}){_NEAR}{{0,140}}(?:{_DUP_CUE})",
    re.IGNORECASE,
)

# An explicit "only one independent source" statement (the alternative keystone phrasing).
_COUNT_ONE_INDEPENDENT = re.compile(
    rf"\b(?:one|1|a\s+single|only\s+one)\b{_NEAR}{{0,40}}independent\s+(?:source|origin)"
    rf"|independent\s+sources?\s*[:=]\s*(?:one|1)\b"
    rf"|number\s+of\s+independent\s+sources{_NEAR}{{0,20}}\b1\b",
    re.IGNORECASE,
)

# THE DEFECT, tier A (strict): explicitly calling the copies INDEPENDENT sources. Note the
# count alternation deliberately excludes "one"/"a single" — the correct answer says exactly
# that ("one independent source") and must never trip this.
_PLURAL_COUNT = r"(?:three|3|four|4|all\s+(?:three|3|four|4)|multiple|several)"
_SOURCE_NOUN = r"(?:sources?|sites?|websites?|pages?|domains?)"
_FALSE_CORROBORATION_STRICT = re.compile(
    rf"{_PLURAL_COUNT}\s+(?:different\s+|separate\s+|distinct\s+)*independent\s+{_SOURCE_NOUN}"
    rf"|(?:confirmed|corroborated|supported|verified|backed)\s+by\s+{_PLURAL_COUNT}\s+"
    rf"(?:different\s+|separate\s+|distinct\s+)*independent\s+{_SOURCE_NOUN}",
    re.IGNORECASE,
)
# Tier B (soft): "three sources confirm it" without the word "independent". Corroboration
# verbs ONLY — a neutral "all three pages print 1,949 ft" is a factual observation a correct
# answer makes and must not be punished, so state/report/print/say are excluded.
_FALSE_CORROBORATION_SOFT = re.compile(
    rf"{_PLURAL_COUNT}\s+(?:different\s+|separate\s+|distinct\s+)*{_SOURCE_NOUN}"
    rf"{_NEAR}{{0,60}}(?:confirm\w*|corroborat\w*|verif\w*|agree\w*|back\s+up|substantiat\w*)"
    rf"|(?:confirmed|corroborated|verified|backed)\s+by\s+{_PLURAL_COUNT}\s+"
    rf"(?:different\s+|separate\s+|distinct\s+)*{_SOURCE_NOUN}",
    re.IGNORECASE,
)
# Words that flip a corroboration match into a correct *denial* of it.
_NEGATOR = re.compile(
    r"\bnot\b|n't\b|\bno\b|\bnever\b|\bonly\b|\bfals\w*|\bincorrect\w*|\bwrong\b|"
    r"\billusor\w*|\bmistak\w*|\bappear\w*|\bseem\w*|\bwould\s+be\b|\bcannot\b|\bcan't\b|"
    r"\bavoid\w*|\bnaive\w*|\brather\s+than\b|\bdespite\b|\bmisleading\b|\bapparent\w*",
    re.IGNORECASE,
)
_DUP_CUE_RE = re.compile(_DUP_CUE, re.IGNORECASE)


def _claims_false_corroboration(text: str) -> bool:
    """True when the answer ASSERTS multi-source corroboration (un-negated) off the copies.

    Two tiers, because a correct answer talks about this defect in order to reject it:
      * strict ("three INDEPENDENT sources ...") — excused only by a negator in the 90
        characters preceding the match ("these are NOT three independent sources").
      * soft ("three sources confirm ...") — excused by a negator OR by a duplication cue
        anywhere in a +/-120 character window, since "all three agree because two are copies
        of the first" is a correct statement, not the defect.
    """
    for m in _FALSE_CORROBORATION_STRICT.finditer(text):
        if not _NEGATOR.search(text[max(0, m.start() - 90): m.end()]):
            return True
    for m in _FALSE_CORROBORATION_SOFT.finditer(text):
        window = text[max(0, m.start() - 120): m.end() + 120]
        if not (_NEGATOR.search(window) or _DUP_CUE_RE.search(window)):
            return True
    return False


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "302",
        "test_name": "Mechanism: syndicated duplicate URLs (illusory corroboration)",
        "difficulty_level": "9/10",
        "category": "Source Provenance & Corroboration",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['url']}" for i, e in enumerate(SYNDICATED_URLS, 1))
    return (
        "CLAIM UNDER TEST: \"Crater Lake (Oregon) has a maximum depth of 1,949 feet "
        "(594 metres).\"\n\n"
        "Open EACH of the following three pages (do not guess from memory — visit them) and "
        "record, for each, the depth figure it prints AND any credit / licence / attribution "
        "line stating where that page's text came from:\n"
        f"{listing}\n\n"
        "Then answer, using the attribution lines you actually read:\n"
        "  (a) How many INDEPENDENT sources do those three URLs amount to? Pages that "
        "republish the same underlying text count ONCE, no matter how many domains they sit "
        "on. State the number and say explicitly which pages are copies of which origin.\n"
        "  (b) Now obtain a genuinely independent source: find and open the U.S. National "
        "Park Service's own page for Crater Lake on nps.gov and report the depth figure in "
        "feet printed there. Say whether it matches the claim under test.\n"
        "  (c) As proof you opened the NPS page, report how many dives the Deep Rover "
        "submersible made to the bottom of the lake, as stated on that page.\n\n"
        "Cite the exact URL of every page you opened. Do NOT describe the claim as "
        "corroborated by multiple independent sources unless you can name sources that are "
        "genuinely independent of one another."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Count of genuinely INDEPENDENT sources among the three given URLs (and which are copies)",
        "Depth figure printed on each of the three syndicated pages",
        "Depth figure in feet printed on the NPS (nps.gov) Crater Lake page",
        "Number of Deep Rover dives stated on the NPS page (proof of visit)",
        "Exact source URL for every page opened",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (three syndicated URLs + the NPS page)",
        "KEYSTONE: recognises the three URLs are one republished source, and does NOT assert "
        "multi-source independent corroboration of the claim",
        "Reports the NPS figure (1,943 ft) attributed to the National Park Service",
        "Reports the page-only Deep Rover dive count (47)",
        "Cites all four source URLs",
    ]


# ---------------------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------------------
def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Keystone credit requires GROUNDING (visit.count > 0) plus the independence verdict.

    Grounding matters here more than usual: "those are Wikipedia mirrors" is guessable from
    the domain names alone, so an ungrounded assertion must earn nothing.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = extract_final_text(result)
    if _claims_false_corroboration(text):
        return False
    return bool(_DUP_NEAR_FAMILY.search(text) or _COUNT_ONE_INDEPENDENT.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: three syndicated pages + the NPS page; >=3 to pass)"}


def validate_keystone_source_independence(result: Dict[str, Any],
                                          observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the answer must treat the three syndicated URLs as ONE source.

    Fails to 0 if the answer asserts (un-negated) that three/multiple independent sources
    confirm the claim — that assertion IS the defect this mechanism task exists to detect —
    or if it never recognises the shared origin, or if it was never grounded in a visit.
    """
    text = extract_final_text(result)
    if _claims_false_corroboration(text):
        return {"check": "keystone_source_independence", "passed": False, "score": 0.0,
                "reason": "Asserts multi-source corroboration from syndicated copies of one "
                          "Wikipedia article (illusory corroboration)"}
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_source_independence", "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": "Recognises the three URLs as ONE republished source" if passed
                      else "Never states the three URLs share one underlying source "
                           "(or answer was ungrounded)"}


def validate_nps_independent_figure(result: Dict[str, Any],
                                    observability: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary: the independent authority's figure (1,943 ft), attributed to NPS.

    Short-circuits to 0 without the keystone — reporting 1,943 while still calling the three
    copies independent corroboration is not a solved task.
    """
    if not _keystone_ok(result, observability):
        return {"check": "nps_independent_figure", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> independent figure not credited"}
    text = extract_final_text(result)
    ok = bool(_NPS_FIGURE_NEAR_NPS.search(text))
    return {"check": "nps_independent_figure", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": "NPS depth figure 1,943 ft reported and attributed to the NPS" if ok
                      else "NPS figure (1,943 ft) missing or not attributed to nps.gov"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth/coverage diagnostic (UN-gated, by design).

    Measures how much of the evidence set was actually gathered — the syndicated figure, each
    of the three copies, the independent page and its page-only token — regardless of whether
    the final independence verdict came out right. This is the axis that separates a
    structured agent (which enumerates and provenance-tags every source) from a linear one
    even when the final answer is botched, so it is deliberately NOT short-circuited.
    """
    text = extract_final_text(result).lower()
    items = [
        ("claimed_figure_1949", bool(_CLAIMED_FIGURE.search(text))),
        ("wikipedia_page", bool(re.search(r"wikipedia", text))),
        ("alchetron_page", bool(re.search(r"alchetron", text))),
        ("dbpedia_page", bool(re.search(r"dbpedia", text))),
        ("nps_page", bool(re.search(r"nps\.gov|national\s+park\s+service", text))),
        ("deep_rover_dive_count", bool(_DIVE_PROOF.search(text))),
    ]
    covered = [name for name, ok in items if ok]
    n = len(items)
    return {"check": "coverage", "passed": len(covered) == n, "score": len(covered) / n,
            "reason": f"{len(covered)}/{n} evidence items gathered ({', '.join(covered) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary: one citation per opened page. Gated on the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for e in SYNDICATED_URLS if re.search(e["slug"], text))
    if re.search(r"nps\.gov", text):
        cited += 1
    n = len(SYNDICATED_URLS) + 1
    return {"check": "citations", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} source pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_source_independence,
        validate_nps_independent_figure,
        validate_coverage,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored scaffold for the ``graph_compiled`` variant.

    Encodes only STRUCTURE: read each given page's figure *and its attribution line*, locate
    the independent agency page, then read two facts off it. It leaks nothing — not the
    verdict ("these are mirrors"), not 1,943, not 594/1,949, not 47. The runtime model still
    does every page read, the provenance grouping and the independence count itself.
    """
    leaves: List[Dict[str, Any]] = [
        {
            # id keyed on the GIVEN domain, never on the verdict.
            "id": re.sub(r"[^a-z0-9]+", "_", e["label"].lower()).strip("_"),
            "instruction": (
                f"Open {e['url']} and report two things: (1) the maximum-depth figure the "
                "page prints for the lake, exactly as written; (2) quote verbatim any "
                "credit, licence, attribution or 'content from' line on the page that says "
                "where the page's text was obtained from (write 'none' if there is none)."
            ),
            "expect": "DEPTH FIGURE AS PRINTED — verbatim attribution/licence line (or 'none') — the URL",
            "depends_on": [],
        }
        for e in SYNDICATED_URLS
    ]
    leaves.append({
        "id": "agency_page_url",
        "instruction": (
            "Find the U.S. National Park Service's own page about Crater Lake on the nps.gov "
            "domain (the park's nature/lake page, not a third-party site). Report its exact URL."
        ),
        "expect": "EXACT nps.gov URL of the National Park Service's Crater Lake page",
        "depends_on": [],
    })
    leaves.append({
        "id": "agency_depth_figure",
        "instruction": (
            "Open {agency_page_url} and read the lake's depth figure in FEET exactly as "
            "printed on that page. Do not substitute a figure from any other site."
        ),
        "expect": "DEPTH IN FEET AS PRINTED ON THAT PAGE — the URL",
        "depends_on": ["agency_page_url"],
    })
    leaves.append({
        "id": "agency_dive_count",
        "instruction": (
            "Open {agency_page_url} and report how many dives the Deep Rover submersible "
            "made to the bottom of the lake, as stated on that page."
        ),
        "expect": "NUMBER OF DIVES — the URL",
        "depends_on": ["agency_page_url"],
    })
    return {
        "leaves": leaves,
        "aggregation": (
            "You have, for each of three given web pages, its depth figure and its verbatim "
            "attribution/licence line, plus a government agency page's own depth figure and "
            "dive count. FIRST group the three given pages by the ORIGIN their attribution "
            "lines point to: pages that republish the same underlying text are ONE source, "
            "however many domains they occupy. Report (a) the number of genuinely independent "
            "sources those three pages amount to and which pages are copies of which origin, "
            "(b) the agency page's depth figure in feet and whether it matches the claim "
            "under test, and (c) the Deep Rover dive count. Do not describe the claim as "
            "corroborated by multiple independent sources unless the sources you name are "
            "genuinely independent of one another. Cite every source URL."
        ),
    }
