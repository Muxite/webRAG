"""
Test 067: Tier 5 (graph) — MEDIAN-BY-DATE ordering (the middle of five, not an extreme).
Level: graph   Weight: long   Difficulty: 9/10

A hard multi-entity ORDERING task built to punish the cheap-model shortcuts of "name the famous
one" and "name an extreme (the oldest / the newest)". Among FIVE large hydroelectric dams the agent
must look up ONE date per entity — the YEAR the dam was commissioned / opened (a single unambiguous
infobox 'Opening date' field) — SORT the five chronologically, and report the MEDIAN dam: the 3rd
of the five by opening year, which is NEITHER the earliest NOR the latest.

    Bhumibol Dam          Daniel-Johnson Dam          Salto Grande Dam
    Merowe Dam          Bakun Dam

The five entities are deliberately a mix whose FAME and ENGINEERING SIGNIFICANCE are decorrelated
from chronological POSITION: Daniel-Johnson Dam (the world's largest multiple-arch-and-buttress dam)
and Merowe Dam (Africa's largest contemporary hydroelectric facility) sit at NON-median positions,
while the comparatively obscure Salto Grande Dam — a binational hydroelectric project on the Uruguay
River between Argentina and Uruguay — lands exactly in the middle.  The answer is therefore a
POSITIONAL / ORDERING fact, not a single memorizable attribute: a parametric model that never opens
the pages cannot recall "which of these five is 3rd-oldest", and every fame-driven or
extreme-driven shortcut lands on the WRONG dam.

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): drops an entity, mis-sorts, or shortcuts to an extreme / the most-notable
    name -> the earliest (Bhumibol), the latest (Bakun), or the engineering record-holder (Daniel-
    Johnson Dam), all of which are WRONG; or simply guesses a year from memory.
  * frontier sequential (ReAct): looks up all five years, sorts, picks the 3rd -> decent.
  * graph_compiled: five parallel leaves each fetch ONE opening year (one date for one dam); the
    aggregation owns the ENTIRE sort and the median selection, and is forced to WRITE OUT the full
    chronological ordering before concluding -> the cheap executor is rescued and a diverse-grounding
    reranker can re-derive the ordering and catch a slip.

The trap is engineered three ways:
  (1) the answer is the MIDDLE of the order, so BOTH parametric extremes are decoys — naming the
      earliest (Bhumibol Dam) OR the latest (Bakun Dam) fails;
  (2) the most engineering-notable dams in the set (Daniel-Johnson = world's largest arch-buttress
      dam; Merowe = Africa's largest contemporary hydroelectric facility) are NOT the median, so
      "name the record-holder" mis-picks;
  (3) the median (Salto Grande Dam) is the LEAST globally-famous member — a mid-20th-century
      binational dam on the Uruguay River between Argentina and Uruguay, a structure that hardly
      appears in trivia databases — so its 3rd-oldest position is not a recallable fact and the
      model must actually READ five pages and SORT.

Ground truth (verified against live English Wikipedia 2026-06-27 — each dam's infobox 'Opening
date' field, the year it was commissioned; every one of the five has a single unambiguous opening
year in the infobox):

  dam                       opening year   wiki slug
  Bhumibol Dam                  1964       wiki/Bhumibol_Dam            <- EARLIEST (decoy)
  Daniel-Johnson Dam            1970       wiki/Daniel-Johnson_Dam
  Salto Grande Dam              1979       wiki/Salto_Grande_Dam        <- MEDIAN (keystone)
  Merowe Dam                    2009       wiki/Merowe_Dam
  Bakun Dam                     2011       wiki/Bakun_Dam               <- LATEST (decoy)

    https://en.wikipedia.org/wiki/Bhumibol_Dam          (Opening date: 1964; on the Ping River,
                                                         Thailand; formerly the Yanhi Dam)
    https://en.wikipedia.org/wiki/Daniel-Johnson_Dam    (Opening date: 1970; on the Manicouagan
                                                         River, Quebec, Canada; also known as Manic-5;
                                                         world's largest arch-and-buttress dam)
    https://en.wikipedia.org/wiki/Salto_Grande_Dam      (Opening date: 1979; on the Uruguay River,
                                                         between Concordia, Argentina and Salto,
                                                         Uruguay; jointly owned by both countries)
    https://en.wikipedia.org/wiki/Merowe_Dam            (Opening date: March 3, 2009; on the Nile,
                                                         ~220 mi north of Khartoum, Sudan; Africa's
                                                         largest contemporary hydroelectric facility)
    https://en.wikipedia.org/wiki/Bakun_Dam             (Opening date: 2011; came online 6 August
                                                         2011; on the Balui River, Sarawak, Malaysia)

  CHRONOLOGICAL ORDER (confirmed live):
    1964 Bhumibol < 1970 Daniel-Johnson < 1979 Salto Grande < 2009 Merowe < 2011 Bakun
    => the MEDIAN (3rd of 5) is the SALTO GRANDE DAM (1979).

  DIVERGENCE FROM THE PARAMETRIC SHORTCUTS:
    earliest           = Bhumibol Dam (1964)          <- NOT the keystone
    latest             = Bakun Dam (2011)             <- NOT the keystone
    world's largest arch-buttress dam = Daniel-Johnson (1970, 2nd) <- NOT the keystone
    Africa's largest hydro facility   = Merowe (2009, 4th)        <- NOT the keystone
    median (3rd) = Salto Grande Dam (1979)            <- the keystone, distinct from all shortcuts

  MARGIN / WORST-CASE +/-1yr SLIP CHECK: the smallest gap adjacent to the median is
  Daniel-Johnson -> Salto Grande = 9 years (1970 -> 1979); the other side, Salto Grande ->
  Merowe = 30 years.  A +/-1 year (indeed +/-4 year) misread on ANY single date cannot change
  which dam is 3rd:
    * Salto Grande 1979 +/-4 -> 1975..1983, still strictly inside (1970, 2009) -> still 3rd;
    * Daniel-Johnson 1970 +8 -> 1978 < 1975 (Salto Grande's worst-low) -> stays 2nd;
    * Merowe 2009 -29 -> 1980 > 1983 (Salto Grande's worst-high) -> stays 4th (realistic +/-1
      is trivial);
    * Bhumibol (1964) and Bakun (2011) are the extremes; a +/-1 slip leaves them extreme.
  So one noisy extraction on any single value cannot flip the keystone.

  ANTI-PARAMETRIC: none of the five opening years is memorizable from training knowledge — Bhumibol
  Dam (Thailand's largest dam, named after King Bhumibol Adulyadej), Daniel-Johnson Dam (Quebec's
  Manic-5 hydroelectric complex), Salto Grande Dam (a binational Uruguay-Argentina project), Merowe
  Dam (Sudan's largest hydro facility), and Bakun Dam (Sarawak, Malaysia) are all regionally
  significant but globally obscure engineering structures.  No cheap model can recall which is
  3rd-oldest of this set; the model must read and sort all five pages.

  KEYSTONE = the MEDIAN entity (Salto Grande Dam). Secondary (gated) value = its opening year (1979).
  All five opening years are distinct, so the un-gated coverage diagnostic is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``year`` is the looked-up opening year; ``median`` flags the keystone. Nothing here is
# leaked into the task statement or the compiled plan.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "bhumibol", "name": "Bhumibol Dam", "year": 1964, "median": False,
     "name_rx": r"\bbhumibol\b", "year_rx": r"(?<!\d)1964(?!\d)", "slug_rx": r"wiki/bhumibol"},
    {"key": "daniel_johnson", "name": "Daniel-Johnson Dam", "year": 1970, "median": False,
     "name_rx": r"\bdaniel[\s\-]*johnson\b", "year_rx": r"(?<!\d)1970(?!\d)",
     "slug_rx": r"wiki/daniel[\s\-_]*johnson"},
    {"key": "salto_grande", "name": "Salto Grande Dam", "year": 1979, "median": True,
     "name_rx": r"\bsalto\s+grande\b", "year_rx": r"(?<!\d)1979(?!\d)",
     "slug_rx": r"wiki/salto[\s\-_]*grande"},
    {"key": "merowe", "name": "Merowe Dam", "year": 2009, "median": False,
     "name_rx": r"\bmerowe\b", "year_rx": r"(?<!\d)2009(?!\d)", "slug_rx": r"wiki/merowe"},
    {"key": "bakun", "name": "Bakun Dam", "year": 2011, "median": False,
     "name_rx": r"\bbakun\b", "year_rx": r"(?<!\d)2011(?!\d)", "slug_rx": r"wiki/bakun"},
]

WINNER = next(e for e in ENTITIES if e["median"])       # Salto Grande Dam — the chronological median
WINNER_YEAR = WINNER["year"]                            # 1979

# Winner / other-entity name regexes used by the keystone gate.
_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["median"])

# Median / middle-position triggers that assert the answer ("X is the median / the middle / the
# 3rd-oldest"). Distinct from the superlative sets of 059/060 because here the keystone is the
# CENTRE of the order, not an extreme.
_MED = r"median|middlemost|middle|midmost|mid-?point|3rd|third"

# Keystone winner detection: the median dam (Salto Grande) tied to a 'median / middle / 3rd'
# assertion, in either direction, with the proximity window forbidden from crossing into ANY other
# dam's name (so a sorted listing where Salto Grande is followed by Merowe/Bakun, then a rival is
# declared the median, can never satisfy it). The window is [^.;] — newline-tolerant (a header line
# then the answer below still matches) but bounded at sentence periods AND clause-separating
# semicolons, so a rival asserted as the median in one clause cannot reach a Salto Grande mention
# in the next. The "than" guard blocks "older than Salto Grande" from counting.
#   dir 1  (subject -> position):  "Salto Grande ... is the median / the 3rd"
#   dir 2  (position -> subject):  "the median (3rd) is Salto Grande"  (NOT "... than Salto Grande")
_SALTO_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _MED + r")\b"
    + r"|\b(?:" + _MED + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "067",
        "test_name": "Tier 5: Median-by-date ordering (obscure hydroelectric dams, the middle of five)",
        "difficulty_level": "9/10",
        "category": "Multi-entity ordering + median selection",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following five large hydroelectric dams, open the dam's "
        "Wikipedia page and read ONE date from the infobox: the YEAR the dam was commissioned or "
        "opened (the infobox 'Opening date' field):\n"
        f"{listing}\n\n"
        "Then SORT the five dams in chronological order by opening year and determine the MEDIAN "
        "dam — the one in the MIDDLE of the order, i.e. the 3rd of the five by opening year. "
        "Note: the answer is the MIDDLE dam, which is NEITHER the earliest-commissioned dam NOR the "
        "latest-commissioned dam, and is not necessarily the most engineering-notable one.\n\n"
        "Report (a) which dam is the median (middle, 3rd-oldest) by opening year (a single dam "
        "name — the keystone), (b) that median dam's opening year, (c) all five dams' opening "
        "years (the five dates you looked up), and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which dam is the MEDIAN (middle, 3rd-oldest) by opening year (the primary answer / "
        "keystone)",
        "That median dam's opening year",
        "All five dams' opening years (the five looked-up dates)",
        "Source URL for each dam's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per dam, a five-way fan-out)",
        "Correctly names the Salto Grande Dam as the median by opening year (NOT the earliest, "
        "Bhumibol Dam; NOT the latest, Bakun Dam; NOT the engineering record-holder, Daniel-Johnson "
        "Dam)",
        "Reports the median dam's opening year (1979)",
        "Gathers all five opening years (one per dam)",
        "Cites the source pages",
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
    """Full reported text: the final deliverable plus every deliverable slot, so coverage / value /
    citation checks can see figures the agent placed outside the primary answer slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names the Salto Grande Dam as the median by opening year.

    Word-bounded, and NOT satisfied by merely listing Salto Grande among the five: when another dam
    is also named, Salto Grande must be the one tied to a 'median / middle / 3rd' assertion
    (tempered so a rival named as the median — or '... than Salto Grande' — never counts). A terse
    primary answer that names only the winner (Salto Grande, with no rival in the slot) also passes.

    Credit requires GROUNDING: the value string alone is insufficient — the agent must have
    actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess would earn credit.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True  # names only the winner
    return bool(_SALTO_WINS.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a five-way fan-out wants one page per dam."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per dam; >=4 to pass)"}


def validate_keystone_median(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the median by opening year is the Salto Grande Dam — NOT the earliest
    (Bhumibol Dam), NOT the latest (Bakun Dam), and NOT the engineering record-holder (Daniel-Johnson
    Dam, world's largest arch-and-buttress dam). A model that shortcuts to an extreme or guesses a
    notable name mis-picks."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_median", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Salto Grande Dam named as the median by opening year" if passed
                      else "Median dam (Salto Grande Dam, 1979) missing/incorrect (beware: "
                           "Bhumibol is the earliest, Bakun Dam the latest, Daniel-Johnson Dam the "
                           "engineering record-holder)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the FIVE (dam, opening-year) pairs were
    gathered.

    A pair is credited only when BOTH that dam's name AND its opening year appear; the five years
    are all distinct, so there is no cross-crediting. Deliberately NOT gated on the keystone — it
    measures whether the agent actually fanned out to all five dams and collected each date even
    when it botches the sort or the median selection, the axis that separates a structured
    (five-leaf) agent from a linear one that drops an entity.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE) and re.search(e["year_rx"], text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} dams' opening years gathered ({', '.join(hits) or 'none'})"}


def validate_median_year(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the median dam's opening year (1979). Short-circuits to 0 when the
    keystone is absent, so a wrong/guessed median can't bank the value credit."""
    if not _keystone_ok(result, observability):
        return {"check": "median_year", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> opening-year value not credited"}
    text = _all_text(result)
    ok = bool(re.search(WINNER["year_rx"], text))
    return {"check": "median_year", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"median dam's opening year ({WINNER_YEAR}) present" if ok
                       else f"median dam's opening year ({WINNER_YEAR}) not found")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the dam pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} dam pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_median,
        validate_coverage,
        validate_median_year,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), as in 054/055.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    FIVE INDEPENDENT parallel leaves: for each of the five GIVEN dams, one leaf fetches the YEAR
    it was commissioned / opened — one atomic date per leaf. ALL ordering — the full chronological
    sort AND the median selection — lives only in the aggregation step, which is forced to WRITE OUT
    the complete sorted ordering explicitly before concluding, so the cheap executor never shortcuts
    to an extreme or a notable-sounding name and a diverse-grounding reranker can re-derive the
    order and catch a slip. Encodes STRUCTURE only: it names the five GIVEN dams, but leaks no
    opening year, no ordering, and not which dam is the median.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_opened",
            "instruction": (
                f"Open the Wikipedia page for the {e['name']} and read, from the infobox "
                "'Opening date' field, the YEAR the dam was commissioned or opened. Report ONLY "
                "that single year and the source URL. Do not guess from memory."
            ),
            "expect": "OPENING YEAR (a single year) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five dams, the year it was commissioned / opened. SORT "
            "the five dams in chronological order by opening year, writing out the full ordered "
            "list explicitly on its own line in the form '<year> <dam>' from earliest to latest — "
            "produce the complete sorted ordering BEFORE drawing any conclusion. THEN identify the "
            "dam occupying the MIDDLE position of that ordering — the 3rd of the five — that dam's "
            "name is the keystone answer (it is NEITHER the earliest nor the latest, and need not "
            "be the most engineering-notable). Show the full sorted ordering before concluding. "
            "Report (a) that median dam and its opening year, (b) all five dams' opening years, "
            "and (c) cite each dam's source URL."
        ),
    }
