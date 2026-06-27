"""
Test 074: Tier 5 (graph/navigation) — SAME-NAME DISAMBIGUATION + EXTRACT.
Level: graph   Weight: long   Difficulty: 9/10

THREE cities share the name "Hamilton" but are in entirely different countries or territories.
Each is pinned by a country/territory disambiguator. The agent must navigate to the CORRECT
Wikipedia page for each entity — NOT the most famous "Hamilton" (Hamilton, Ontario, Canada,
population ~569,000) — and read the population from the infobox.

The KEYSTONE is the population of the SMALLEST Hamilton among the three: Hamilton, Bermuda,
whose Wikipedia infobox records population 854 (2016 census). This figure is ~670× smaller
than the famous-homonym decoy (Hamilton, Ontario: 569,353), so any model that confuses the
two — or guesses from memory rather than reading the page — misses the keystone by orders of
magnitude.

A secondary built-in trap: Hamilton, Bermuda shares its name with Hamilton Parish (a separate,
larger administrative division of Bermuda). The city-proper population (854) lives only in
the infobox of the city article; the parish population is different. Correct disambiguation
therefore requires reaching the CITY article, not a general Bermuda geography article.

Ground truth (verified live against English Wikipedia, 2026-06-27):

  Entity                                 Infobox population   Wikipedia slug
  ────────────────────────────────────────────────────────────────────────────────────
  Hamilton, Bermuda   ← KEYSTONE entity        854 (2016)      Hamilton,_Bermuda
  Hamilton, New Zealand                     192,100 (Jun 2025) Hamilton,_New_Zealand
  Hamilton, South Lanarkshire, Scotland      54,480 (2020)     Hamilton,_South_Lanarkshire
  ────────────────────────────────────────────────────────────────────────────────────
  Hamilton, Ontario, Canada  [FAMOUS DECOY] 569,353 (2021)     Hamilton,_Ontario
  ────────────────────────────────────────────────────────────────────────────────────

Anti-parametric mechanism: "Hamilton" as an unqualified city name strongly primes any LLM
toward Hamilton, Ontario (the most-populated Hamilton in the English-speaking world; also
the most frequently referenced). A model that navigates to Ontario's page — or recalls
Ontario's population from parametric memory — returns 569,353 instead of 854, failing the
keystone gate by a factor of ~670. A model that knows Bermuda's capital is "small" but does
not actually read the page will estimate a plausible-sounding round number (hundreds or
thousands), not the census value 854.

Why 854 is page-only: Hamilton, Bermuda is described on Wikipedia as one of the world's
smallest national capital cities by resident population. The exact census figure (854) is not
parametrically recallable by any consumer LLM; even knowing the city is tiny, a model would
estimate a round number. Only reading the infobox on the correct city article yields "854".

Discrimination profile (per REASONING_TEST_DESIGN.md — the differential-lift target):
  parametric (no tools):  LOW  — defaults to Ontario value (~569,000) or guesses wrong round number
  graph native (cheap):   LOW  — may fail to use the disambiguator, land on Ontario's page, or
                                  guess a plausible round figure without reading the infobox
  sequential_react:       decent — explicit search + page reads can land on the correct values
  graph_compiled:         HIGH  — each leaf pins the exact entity + disambiguator; executor reads
                                  854 from the correct infobox; aggregation restates all three
                                  city→population pairs before concluding

Margin / robustness: 854 is unique among small integers in this context. The other two correct
populations (192,100 and 54,480) do not contain "854" as a word-bounded token, and Ontario's
decoy value (569,353) is an order of magnitude larger. A single noisy extraction cannot flip
the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── verified ground-truth fixtures (single source of truth for statement, validators, plan) ──
ENTITIES: List[Dict[str, Any]] = [
    {
        "key": "bermuda",
        "name": "Hamilton, Bermuda",
        "desc": "capital city of Bermuda (British Overseas Territory)",
        "population": 854,
        "slug": "Hamilton,_Bermuda",
        # name_rx: 'bermuda' is unique enough — no other entity shares this token
        "name_rx": r"\bbermuda\b",
        # pop_rx: \b854\b is word-bounded; cannot match 54,480 or 192,100 or 569,353
        "pop_rx": r"\b854\b",
        # slug_rx: lowercase substring of the canonical Wikipedia URL path
        "slug_rx": r"wiki/hamilton,_bermuda",
        "keystone": True,
    },
    {
        "key": "nz",
        "name": "Hamilton, New Zealand",
        "desc": "city in the Waikato Region of New Zealand",
        "population": 192100,
        "slug": "Hamilton,_New_Zealand",
        "name_rx": r"\bnew\s+zealand\b",
        # pop_rx: matches "192,100" or "192100" (Wikipedia may use either)
        "pop_rx": r"(?<!\d)192[,\s]?100(?!\d)",
        "slug_rx": r"wiki/hamilton,_new_zealand",
        "keystone": False,
    },
    {
        "key": "scotland",
        "name": "Hamilton, South Lanarkshire",
        "desc": "town in South Lanarkshire, Scotland",
        "population": 54480,
        "slug": "Hamilton,_South_Lanarkshire",
        "name_rx": r"\b(?:lanarkshire|hamilton.*scotland|scotland.*hamilton)\b",
        # pop_rx: matches "54,480" or "54480"
        "pop_rx": r"(?<!\d)54[,\s]?480(?!\d)",
        "slug_rx": r"wiki/hamilton,_south_lanarkshire",
        "keystone": False,
    },
]

KEYSTONE_ENTITY = next(e for e in ENTITIES if e["keystone"])   # Hamilton, Bermuda

# Famous decoy: the most well-known "Hamilton" city — NOT in the task, the primary fail mode.
# A model guessing from memory or navigating without the disambiguator lands here.
FAMOUS_DECOY_NAME = "Hamilton, Ontario, Canada"
FAMOUS_DECOY_POP = 569353    # 2021 Canadian census; fails \b854\b gate by factor ~670

# ── compiled regex constants ───────────────────────────────────────────────────────────────────
_KEYSTONE_RX = re.compile(r"\b854\b")
_BERMUDA_RX = re.compile(r"\bbermuda\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "074",
        "test_name": "Tier 5: Same-name disambiguation + extract (Hamilton cities, population argmin)",
        "difficulty_level": "9/10",
        "category": "Same-name disambiguation + page-only extraction",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} — {e['desc']}"
        for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — navigate Wikipedia yourself to find each page. Three cities "
        "share the name 'Hamilton' but are in different countries or territories:\n"
        f"{listing}\n\n"
        "IMPORTANT: do NOT use Hamilton, Ontario, Canada — that is a different, much larger city "
        "with the same name. Use ONLY the exact disambiguators above (country or territory) to "
        "locate each entity's specific Wikipedia page.\n\n"
        "For EACH of the three cities, open its Wikipedia page and read the current population "
        "from the infobox. Then compare the three populations and identify the city with the "
        "SMALLEST population.\n\n"
        "Report: (a) the population of the SMALLEST Hamilton and the name of that city — this "
        "single population figure is the keystone answer; (b) all three populations with the "
        "city name for each; (c) the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The population of the smallest Hamilton and which city that is (keystone answer)",
        "All three populations with their city names (Bermuda / New Zealand / South Lanarkshire)",
        "Wikipedia URL for each of the three pages visited",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (one per Hamilton entity)",
        "Correctly identifies Hamilton, Bermuda as the city with the smallest population",
        "Reports Hamilton, Bermuda's population as 854 (2016 census figure from the infobox)",
        "Reports all three populations correctly (854 / 192,100 / 54,480)",
        "Cites the Wikipedia URL for each entity",
        "Does NOT use Hamilton, Ontario, Canada's population (569,353) — the famous same-name decoy",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text. Prefers deliverables[0] (the keystone slot); falls back to
    output.final_deliverable via extract_final_text."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: final deliverable concatenated with all deliverable slots, so
    coverage and citation checks can see figures the agent placed outside the primary slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: deliverables[0] contains 854 — Hamilton, Bermuda's infobox population.

    854 is the 2016-census city-proper population. It cannot be produced by a model that
    navigates to Hamilton, Ontario (569,353), Hamilton, New Zealand (192,100), or Hamilton,
    South Lanarkshire (54,480), nor by any plausible round-number guess for a small capital.
    """
    return bool(_KEYSTONE_RX.search(_primary_text(result)))


# ── validation functions ───────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a three-way fan-out needs at least 3 page reads."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 3,
        "score": min(1.0, n / 3.0),
        "reason": f"{n} visit(s) (target ≥3: one page per Hamilton entity)",
    }


def validate_keystone_population(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): primary text contains Hamilton, Bermuda's population (854).

    \b854\b in deliverables[0] is the sole gate. The famous-decoy value (Ontario: 569,353) and
    both other correct values (192,100 and 54,480) cannot satisfy it. A model that guesses from
    memory, uses the wrong page, or estimates a round number for a "small capital" fails here.
    """
    passed = _keystone_ok(result)
    return {
        "check": "keystone_population",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Population 854 (Hamilton, Bermuda, 2016 census) present in primary answer"
            if passed
            else (
                "Population 854 (Hamilton, Bermuda) absent or incorrect — "
                "common failure: Ontario's 569,353 used instead (wrong disambiguated page)"
            )
        ),
    }


def validate_argmin_entity(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: Hamilton, Bermuda is named somewhere in the output.

    Short-circuits to 0 when the keystone is absent, keeping scores bimodal and preventing a
    run that incidentally mentions 'Bermuda' without reporting the correct population from banking
    partial credit. When the keystone passes (854 present), this check verifies the agent also
    named the correct entity — not merely an anonymous population figure.
    """
    if not _keystone_ok(result):
        return {
            "check": "argmin_entity",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → argmin entity not credited",
        }
    text = _all_text(result)
    has_bermuda = bool(_BERMUDA_RX.search(text))
    return {
        "check": "argmin_entity",
        "passed": has_bermuda,
        "score": 1.0 if has_bermuda else 0.0,
        "reason": (
            "Hamilton, Bermuda named in output"
            if has_bermuda
            else "Hamilton, Bermuda not named in output (population 854 present but entity unlabelled)"
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the three entities' populations were correctly read.

    A population is credited only when the entity's name token AND its population figure both
    appear in all_text. The three population values (854, 192,100, 54,480) are all distinct and
    do not overlap as word-bounded tokens, so there is no cross-crediting. Deliberately NOT gated
    on the keystone — measures fan-out completeness even when the argmin comparison is wrong, the
    axis that separates a structured multi-leaf agent from a linear one that drops an entity.
    """
    text = _all_text(result)
    hits = [
        e["name"]
        for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE)
        and re.search(e["pop_rx"], text)
    ]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} entities' populations gathered "
            f"({', '.join(hits) if hits else 'none'})"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source Wikipedia pages for the three entities.

    Short-circuits to 0 when the keystone is absent (keeps scores bimodal). At least 2 of the
    3 slugs must appear to pass. The famous-decoy slug (hamilton,_ontario) is not penalised here
    — the keystone gate already handles that failure mode.
    """
    if not _keystone_ok(result):
        return {
            "check": "citation",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → source URLs not credited",
        }
    text = _all_text(result).lower()
    cited = [e["name"] for e in ENTITIES if re.search(e["slug_rx"], text)]
    n = len(ENTITIES)
    return {
        "check": "citation",
        "passed": len(cited) >= 2,
        "score": len(cited) / n,
        "reason": (
            f"{len(cited)}/{n} entity pages cited "
            f"({', '.join(cited) if cited else 'none'}; need ≥2 to pass)"
        ),
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_population,
        validate_argmin_entity,
        validate_coverage,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None → harness applies its default structured rubric judge (gpt-5-mini), as in 059/060.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate DAG scaffold for the graph_compiled variant.

    THREE INDEPENDENT parallel leaves — one per Hamilton entity. Each leaf explicitly pins the
    disambiguator (country or territory) and warns against the famous Ontario homonym, so the
    cheap executor opens the CORRECT page and reads the CORRECT infobox value. Every leaf asks
    for ONE figure (population) from ONE page. All comparison logic — listing the three values
    and identifying the smallest — lives ONLY in the aggregation step, which is forced to write
    out all three city→population pairs before concluding, so the cheap executor never shortcuts
    to a parametric guess.

    Encodes STRUCTURE only: names the three given entities and their disambiguators but leaks NO
    population figures, no rank ordering, and not which entity has the smallest population.
    """
    return {
        "leaves": [
            {
                "id": "bermuda_pop",
                "instruction": (
                    "Navigate to the Wikipedia page for Hamilton, Bermuda — the capital city of "
                    "the British Overseas Territory of Bermuda. This is NOT Hamilton, Ontario, "
                    "Canada, nor any other city named Hamilton; use the term 'Bermuda' to reach "
                    "the correct page. Read the city's population from the infobox and report "
                    "ONLY that single population figure and the exact Wikipedia URL. "
                    "Do not guess from memory."
                ),
                "expect": "POPULATION (single integer) — exact Wikipedia URL for Hamilton, Bermuda",
                "depends_on": [],
            },
            {
                "id": "nz_pop",
                "instruction": (
                    "Navigate to the Wikipedia page for Hamilton, New Zealand — the city in the "
                    "Waikato Region of New Zealand. This is NOT Hamilton, Ontario, Canada, nor "
                    "Bermuda's Hamilton; use 'New Zealand' or 'Waikato' to reach the correct page. "
                    "Read the city's population from the infobox and report ONLY that single "
                    "population figure and the exact Wikipedia URL. Do not guess from memory."
                ),
                "expect": "POPULATION (single integer) — exact Wikipedia URL for Hamilton, New Zealand",
                "depends_on": [],
            },
            {
                "id": "scotland_pop",
                "instruction": (
                    "Navigate to the Wikipedia page for Hamilton, South Lanarkshire — the town in "
                    "South Lanarkshire, Scotland. This is NOT Hamilton, Ontario, Canada; use "
                    "'South Lanarkshire' or 'Scotland' to reach the correct page. Read the town's "
                    "population from the infobox and report ONLY that single population figure and "
                    "the exact Wikipedia URL. Do not guess from memory."
                ),
                "expect": "POPULATION (single integer) — exact Wikipedia URL for Hamilton, South Lanarkshire",
                "depends_on": [],
            },
        ],
        "aggregation": (
            "You now have the populations of three cities, each named 'Hamilton' but in different "
            "countries or territories. Before drawing any conclusion, write out all three explicitly "
            "on separate lines in the form '<city, country/territory>: <population>'. Then compare "
            "the three populations and identify which single city has the SMALLEST population. "
            "State: (a) that city's name and its population — this population figure is the "
            "keystone answer; (b) all three city→population pairs; (c) the Wikipedia URL of every "
            "page you read. Write out all three populations BEFORE concluding which is smallest."
        ),
    }
