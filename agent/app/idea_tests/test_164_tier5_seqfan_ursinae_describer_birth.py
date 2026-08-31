"""
Test 164: Tier 5 (graph) — SEQUENTIAL PREFIX -> 6-WAY FAN-OUT -> MERGE (natural history)
Level: graph   Weight: long   Difficulty: 9/10   Category: sequential_prefix_fanout_merge

Shape (the reason this task exists): the suite's breadth tasks (152-158) hand the agent its N
entities in the mandate, so every arm starts with a ready-made fan-out. Here the fan-out set is
UNNAMEABLE from the prompt — it has to be *earned* by a genuinely dependent 3-hop prefix — and
each of the six branches is itself a TWO-PAGE subtask, not a single infobox lookup:

  PHASE 1 — SEQUENTIAL PREFIX (each hop knowable only from the previous page):
      HOP 1  Wojtek (the given start)          -> the animal's SUBSPECIES (Ursus arctos syriacus)
      HOP 2  Syrian brown bear                 -> the SPECIES it belongs to (brown bear)
      HOP 3  Brown bear                        -> the SUBFAMILY in its taxobox (Ursinae)
                                               -> the SIX living species of that subfamily
  PHASE 2 — FAN-OUT (six mutually independent branches, 2 page-reads each):
      per species: (a) its own article -> the binomial authority (describer + year)
                   (b) THAT PERSON's article -> the describer's YEAR OF BIRTH
  PHASE 3 — MERGE: argmax over the six describers' birth years.

Ground truth (every row verified against live English Wikipedia, 2026-08-29):
  species (article)                       binomial authority        describer          born
  --------------------------------------  ------------------------  -----------------  ----
  Brown bear /wiki/Brown_bear             Ursus arctos L., 1758     Carl Linnaeus      1707
  American black bear /wiki/American_...  U. americanus Pallas 1780 P. S. Pallas       1741
  Polar bear /wiki/Polar_bear             U. maritimus Phipps 1774  Constantine Phipps 1744
  Sloth bear /wiki/Sloth_bear             Melursus ursinus (Shaw,   George Shaw        1751
                                          1791)
  Asian black bear /wiki/Asian_black_bear U. thibetanus G. Cuvier,  Georges Cuvier     1769
                                          1823
  Sun bear /wiki/Sun_bear                 Helarctos malayanus       Stamford Raffles   1781  <= KEYSTONE
                                          (Raffles, 1821)

  KEYSTONE = the sun bear (Helarctos malayanus), described by Stamford Raffles, born 1781.

Verified sources and the exact values read from them (all English Wikipedia, 2026-08-29):
  https://en.wikipedia.org/wiki/Wojtek                 "was a Syrian brown bear" (article title
                                                       is plain "Wojtek" — the species is NOT in
                                                       the title, so hop 1 is a real page read)
  https://en.wikipedia.org/wiki/Syrian_brown_bear      "subspecies of brown bear", Ursus arctos
                                                       syriacus; taxobox Ursidae / Ursinae
  https://en.wikipedia.org/wiki/Brown_bear             taxobox Subfamily: Ursinae, Genus: Ursus,
                                                       "Ursus arctos, Linnaeus, 1758"
  https://en.wikipedia.org/wiki/Bear                   "Ursinae (containing six species divided
                                                       into one to three genera, depending on the
                                                       authority)" — the SET SIZE is stated by the
                                                       source; each of the six taxoboxes below
                                                       independently lists Subfamily: Ursinae
  https://en.wikipedia.org/wiki/Polar_bear             "Ursus maritimus" Phipps, 1774; Ursinae
  https://en.wikipedia.org/wiki/American_black_bear    "Ursus americanus Pallas, 1780"; Ursinae
  https://en.wikipedia.org/wiki/Sloth_bear             "Melursus ursinus (Shaw, 1791)"; Ursinae
  https://en.wikipedia.org/wiki/Asian_black_bear       "Ursus thibetanus G. Cuvier, 1823"; Ursinae
  https://en.wikipedia.org/wiki/Sun_bear               "Helarctos malayanus (Raffles, 1821)";
                                                       Ursinae; "proposed by Stamford Raffles in
                                                       1821 when he first described a sun bear"
  https://en.wikipedia.org/wiki/Carl_Linnaeus          born 23 May 1707
  https://en.wikipedia.org/wiki/Peter_Simon_Pallas     born 22 September 1741
  https://en.wikipedia.org/wiki/Constantine_Phipps,_2nd_Baron_Mulgrave  born 30 May 1744
  https://en.wikipedia.org/wiki/George_Shaw_(biologist) 1751-1813 (plain /wiki/George_Shaw is a
                                                       disambiguation page — a real hazard, but
                                                       both routes give 1751)
  https://en.wikipedia.org/wiki/Georges_Cuvier         born 23 August 1769
  https://en.wikipedia.org/wiki/Stamford_Raffles       born 5 July 1781, died 5 July 1826

Margin analysis (why one noisy extraction cannot flip the keystone):
  * argmax margin: Raffles 1781 vs runner-up Georges Cuvier 1769 = 12 years, and BOTH are exact,
    undisputed calendar birth dates (not source-varying measurements like a dam height), so the
    only realistic error mode is mis-identifying a person, not mis-reading a number.
  * Cuvier ambiguity is safe by construction: the sibling naturalist Frederic Cuvier (b. 1773 —
    the describer of the spectacled bear, which is NOT in this subfamily) is also below 1781, so
    picking the wrong Cuvier still leaves the sun bear the argmax.
  * Phipps has a documented birth-date discrepancy (a plaque says 9 May 1744 vs the infobox's
    30 May 1744) — the YEAR is unaffected, and 1744 is 37 years from the keystone.
  * Wrong-AXIS decoy: merging on the DESCRIPTION year instead of the describer's birth year gives
    the Asian black bear (G. Cuvier, 1823). No token of that wrong answer (1823 / Cuvier /
    thibetanus) collides with 1781 / Raffles / sun bear, so the two outcomes stay separable.
  * Out-of-set decoys: giant panda (Ailuropodinae, David 1869) and spectacled bear (Tremarctinae,
    F. Cuvier 1825) are living bears but NOT Ursinae; including them would not change the argmax
    either (David b. 1826, F. Cuvier b. 1773 — both < 1781), so a slightly over-broad set still
    yields the same keystone. The set is chosen at SPECIES level precisely because the genus-level
    split of Ursinae is the one place real sources disagree ("one to three genera").

Leak resistance: "which bear's describer was born last" is not a memorized fact; producing it
requires six describer pages actually read and compared. A model that answers from parametric
memory typically returns the description-year argmax (1823) or the famous name (Linnaeus).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, waypoint_chain_coverage


START_ENTITY = "Wojtek"
SET_SIZE = 6

# The sequential prefix. Each waypoint is credited only with per-waypoint page evidence.
PREFIX: List[Dict[str, str]] = [
    {"name": "Syrian brown bear", "name_rx": r"syrian\s+brown\s+bear|arctos\s+syriacus",
     "slug_rx": r"wiki/syrian_brown_bear"},
    {"name": "Brown bear", "name_rx": r"brown\s+bear|ursus\s+arctos",
     "slug_rx": r"wiki/brown_bear"},
    {"name": "Ursinae", "name_rx": r"\bursinae\b", "slug_rx": r"wiki/ursinae|wiki/bear\b"},
]

# The fan-out set — single source of truth for the statement, validators and the compiled plan.
# ``species_rx``/``describer_rx``/``birth`` are what a resolved branch must show; ``describer_slug``
# is the SECOND page of each branch (citing it is what proves the two-hop subtask was done).
ENTRIES: List[Dict[str, str]] = [
    {"species": "Brown bear", "binomial": "Ursus arctos", "described": "1758",
     "describer": "Carl Linnaeus", "birth": "1707",
     "species_rx": r"brown\s+bear|ursus\s+arctos",
     "describer_rx": r"linn(a?eus|é)",
     "describer_slug": r"wiki/carl_linnaeus|wiki/linnaeus"},
    {"species": "American black bear", "binomial": "Ursus americanus", "described": "1780",
     "describer": "Peter Simon Pallas", "birth": "1741",
     "species_rx": r"american\s+black\s+bear|ursus\s+americanus|\bamericanus\b",
     "describer_rx": r"pallas",
     "describer_slug": r"wiki/peter_simon_pallas"},
    {"species": "Polar bear", "binomial": "Ursus maritimus", "described": "1774",
     "describer": "Constantine Phipps", "birth": "1744",
     "species_rx": r"polar\s+bear|ursus\s+maritimus|\bmaritimus\b",
     "describer_rx": r"phipps|mulgrave",
     "describer_slug": r"wiki/constantine_phipps|wiki/constantine_john_phipps"},
    {"species": "Sloth bear", "binomial": "Melursus ursinus", "described": "1791",
     "describer": "George Shaw", "birth": "1751",
     "species_rx": r"sloth\s+bear|melursus",
     "describer_rx": r"\bshaw\b",
     "describer_slug": r"wiki/george_shaw"},
    {"species": "Asian black bear", "binomial": "Ursus thibetanus", "described": "1823",
     "describer": "Georges Cuvier", "birth": "1769",
     "species_rx": r"asia(n|tic)\s+black\s+bear|thibetanus|moon\s+bear",
     "describer_rx": r"cuvier",
     "describer_slug": r"wiki/georges_cuvier|wiki/g%c3%a9orges_cuvier"},
    {"species": "Sun bear", "binomial": "Helarctos malayanus", "described": "1821",
     "describer": "Stamford Raffles", "birth": "1781",
     "species_rx": r"sun\s+bear|helarctos|malayanus",
     "describer_rx": r"raffles",
     "describer_slug": r"wiki/stamford_raffles|wiki/thomas_stamford_raffles"},
]

KEYSTONE = ENTRIES[-1]

# Keystone: the sun bear / Raffles named AS the argmax (describer born most recently), plus 1781.
# ``[^.]`` (not ``[^.\n]``) so a line break between "Born most recently:" and the name — a common
# report layout — still matches, while a sentence-ending period still bounds the window. Only TRUE
# superlatives trigger: a plain table row "Sun bear | Raffles | 1781" must not earn the gate.
_TARGET = r"(?:sun\s+bear|helarctos|malayanus|raffles)"
_SUPERLATIVE = (
    r"(?:latest[-\s]?born|born\s+(?:the\s+)?latest|born\s+last|born\s+most\s+recently|"
    r"most\s+recently\s+born|youngest|last\s+to\s+be\s+born|newest[-\s]?born)"
)
_ANSWER_NEAR_TARGET = re.compile(
    rf"{_SUPERLATIVE}[^.]{{0,80}}{_TARGET}|{_TARGET}[^.]{{0,80}}{_SUPERLATIVE}",
    re.IGNORECASE,
)
_KEYSTONE_BIRTH_YEAR = re.compile(r"\b1781\b")


def get_test_metadata() -> Dict[str, Any]:
    """
    Benchmark-runner metadata for this task.
    :return: Dict with test_id, test_name, difficulty_level, category, level and weight.
    """
    return {
        "test_id": "164",
        "test_name": "Tier 5: sequential prefix -> 6-way fan-out -> merge (latest-born describer, Ursinae)",
        "difficulty_level": "9/10",
        "category": "Sequential Prefix + Breadth Fan-out & Aggregation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    """
    The mandate handed to the agent. Names ONLY the starting entity and the set SIZE — never the
    subspecies, the species, the subfamily, the six members, the describers or the answer.
    :return: Task statement string.
    """
    return (
        "You are given NO URLs. Work from the English Wikipedia article titled 'Wojtek' — a "
        "celebrated animal that served with Polish forces during the Second World War.\n\n"
        "PART 1 (chain — each step is readable only from the previous page):\n"
        "  1. From that article, identify the animal's SUBSPECIES (full scientific name).\n"
        "  2. Open that subspecies' article and identify the SPECIES it belongs to.\n"
        "  3. Open that species' article and read the SUBFAMILY given in its taxobox, then find "
        "the SIX LIVING (extant) species placed in that subfamily.\n\n"
        "PART 2 (for EACH of those six living species, two page-reads):\n"
        "  a. Open the species' own article and read its binomial authority — the naturalist who "
        "first described it, and the year.\n"
        "  b. Then open THAT NATURALIST's own article and read their YEAR OF BIRTH.\n\n"
        "PART 3 (merge across all six): determine which of the six species was described by the "
        "naturalist with the LATEST (most recent) YEAR OF BIRTH. This is a question about when the "
        "describers were BORN, not about when the species were described.\n\n"
        "Report (a) that species and its describer, stating explicitly that this describer was "
        "born most recently, with the describer's birth year; (b) the full six-row table "
        "species -> describer -> year of description -> describer's year of birth; (c) the "
        "subspecies, species and subfamily you passed through; citing the exact source URL of "
        "every page you read."
    )


def get_required_deliverables() -> List[str]:
    """
    :return: Deliverables the agent's final answer must contain.
    """
    return [
        "The species whose describer was born most recently, with that describer and birth year",
        "All six species -> describer -> description year -> describer birth year rows",
        "The subspecies / species / subfamily chain traversed to reach the six species",
        "Source URL per species page and per describer page",
    ]


def get_success_criteria() -> List[str]:
    """
    :return: Human-readable success criteria for reports.
    """
    return [
        "At least 8 pages visited (3-hop prefix + six two-page branches)",
        "Correctly identifies the sun bear / Stamford Raffles (born 1781) as the argmax",
        "Reports all six species with describer and describer birth year",
        "Names the Syrian brown bear -> brown bear -> Ursinae prefix chain",
        "Cites the describer pages, not only the species pages",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """
    Keystone predicate, GROUNDED: the answer text must name the sun bear / Raffles as the
    latest-born describer AND contain 1781, and the agent must have visited at least one page
    (``visit.count > 0``), so an ungrounded parametric guess earns nothing.
    :param result: Test result payload.
    :param observability: Observability payload (visit counts, evidence).
    :return: True if the grounded keystone answer is present.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = extract_final_text(result)
    return bool(_ANSWER_NEAR_TARGET.search(text) and _KEYSTONE_BIRTH_YEAR.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traversal-volume gate: the golden path is 3 prefix pages + 6 species + 6 describer pages.
    :param result: Test result payload (unused).
    :param observability: Observability payload.
    :return: ``{"check","passed","score","reason"}``.
    """
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 8, "score": min(1.0, n / 12.0),
            "reason": f"{n} visit(s) (golden path ~15; >=8 to pass)"}


def validate_keystone_latest_born_describer(result: Dict[str, Any],
                                            observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    KEYSTONE (hard 0/1): the merge answer — the sun bear, described by Stamford Raffles, who was
    born 1781, is the latest-born of the six describers.
    :param result: Test result payload.
    :param observability: Observability payload.
    :return: ``{"check","passed","score","reason"}``.
    """
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_latest_born_describer",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Latest-born describer = Stamford Raffles (b. 1781), sun bear" if passed
                  else "Argmax answer (sun bear / Stamford Raffles, born 1781) missing or wrong",
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Breadth diagnostic (UN-gated): how many of the six branches were resolved END-TO-END — a
    branch counts only when the species, its describer AND that describer's birth year are all
    present, which is exactly the two-page subtask this task asks for.

    Deliberately not short-circuited on the keystone: it measures how much was actually gathered,
    the axis that separates a structured agent from a linear one even when the merge is botched.
    :param result: Test result payload.
    :param observability: Observability payload (unused).
    :return: ``{"check","passed","score","reason"}``.
    """
    text = extract_final_text(result).lower()
    hits: List[str] = []
    for e in ENTRIES:
        if (re.search(e["species_rx"], text)
                and re.search(e["describer_rx"], text)
                and re.search(r"\b" + e["birth"] + r"\b", text)):
            hits.append(e["species"])
    covered = len(hits)
    return {"check": "coverage", "passed": covered == SET_SIZE, "score": covered / SET_SIZE,
            "reason": f"{covered}/{SET_SIZE} branches resolved species+describer+birth year "
                      f"({', '.join(hits) or 'none'})"}


def validate_chain_prefix(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Secondary (gated on the keystone): the three sequential-prefix waypoints, each credited only
    with per-waypoint visited-page evidence, so a guessed chain earns nothing.
    :param result: Test result payload.
    :param observability: Observability payload (carries ``evidence.visited``).
    :return: ``{"check","passed","score","reason"}``.
    """
    if not _keystone_ok(result, observability):
        return {"check": "chain_prefix", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> prefix chain not credited"}
    cov = waypoint_chain_coverage(PREFIX, result, observability, extract_final_text(result))
    cov["check"] = "chain_prefix"
    return cov


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Secondary (gated on the keystone): how many of the six DESCRIBER pages — the second page of
    each branch — were cited by URL.
    :param result: Test result payload.
    :param observability: Observability payload.
    :return: ``{"check","passed","score","reason"}``.
    """
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for e in ENTRIES if re.search(e["describer_slug"], text))
    return {"check": "citations", "passed": cited >= 3, "score": cited / SET_SIZE,
            "reason": f"{cited}/{SET_SIZE} describer pages cited"}


def get_validation_functions() -> List[callable]:
    """
    :return: The deterministic validators, keystone first among the gated ones.
    """
    return [validate_visits, validate_keystone_latest_born_describer,
            validate_coverage, validate_chain_prefix, validate_citations]


def get_llm_validation_function() -> callable:
    """
    :return: None — this task is fully deterministic, no LLM judge.
    """
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """
    Offline-authored scaffold for the ``graph_compiled`` arm: a 3-leaf dependent PREFIX, a
    discovery leaf, then 6 x 2 dependent branch leaves (waves of width 6), then the merge.

    Leaks NOTHING: the only entity named is the GIVEN start (``Wojtek``). No subspecies, species,
    subfamily, member species, describer, year or the argmax appears anywhere in the plan text;
    every downstream leaf reaches its subject through ``{dep_id}`` templating only.
    :return: Schema-v2 compiled plan ``{"leaves": [...], "aggregation": str}``.
    """
    leaves: List[Dict[str, Any]] = [
        {
            "id": "start_subspecies",
            "instruction": (
                f"Open the English Wikipedia article titled '{START_ENTITY}' (a celebrated animal "
                "that served with Polish forces in the Second World War). Read WHICH SUBSPECIES "
                "the animal was, exactly as the article states it, with its full scientific name. "
                "Report only that subspecies and the exact source URL; do not guess from memory."
            ),
            "expect": "The animal's subspecies (common + full scientific name) — source URL",
            "depends_on": [],
        },
        {
            "id": "parent_species",
            "instruction": (
                "Open the Wikipedia article for the subspecies identified in the previous step "
                "({start_subspecies}). Read WHICH SPECIES it is a subspecies of. Report only that "
                "species (common name + binomial) and its exact Wikipedia URL."
            ),
            "expect": "The parent species (common name + binomial) — source URL",
            "depends_on": ["start_subspecies"],
        },
        {
            "id": "subfamily",
            "instruction": (
                "Open the Wikipedia article for the species identified in the previous step "
                "({parent_species}). Read the SUBFAMILY listed in its taxobox (the rank between "
                "family and genus). Report only that subfamily name and the source URL."
            ),
            "expect": "The subfamily named in the species' taxobox — source URL",
            "depends_on": ["parent_species"],
        },
        {
            "id": "members",
            "instruction": (
                "Using the subfamily identified in the previous step ({subfamily}), find the SIX "
                "LIVING (extant) species placed in that subfamily. Consult the subfamily's own "
                "article or its family article and list exactly those six living species by "
                "common name and binomial, NUMBERED 1 to 6. Exclude extinct species. Report the "
                "numbered list and the source URL; do not report any other fact."
            ),
            "expect": "A numbered list of the six living species of the subfamily — source URL",
            "depends_on": ["subfamily"],
        },
    ]
    for k in range(1, SET_SIZE + 1):
        leaves.append({
            "id": f"authority_{k}",
            "instruction": (
                f"From the numbered list of six living species ({{members}}), take species number "
                f"{k}. Open THAT species' own Wikipedia article and read its BINOMIAL AUTHORITY "
                "from the taxobox: the name of the naturalist who first described it and the year "
                "of description. Report the species, that naturalist's name, the description year "
                "and the exact source URL. Do not guess from memory."
            ),
            "expect": f"Species #{k}, its describing naturalist and the description year — source URL",
            "depends_on": ["members"],
        })
        leaves.append({
            "id": f"describer_birth_{k}",
            "instruction": (
                f"Take the naturalist named as the describer in the previous step "
                f"({{authority_{k}}}). Open THAT PERSON's own Wikipedia article and read their "
                "YEAR OF BIRTH directly from the page. If the name is ambiguous, pick the "
                "naturalist/zoologist who published the description. Report the species, the "
                "person, their year of birth and the exact source URL of the person's page. Do "
                "not report their year of death, and do not guess from memory."
            ),
            "expect": f"Describer of species #{k} and that person's YEAR OF BIRTH — source URL",
            "depends_on": [f"authority_{k}"],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six living species of the subfamily, the naturalist who "
            "first described it, the year of description, and THAT NATURALIST'S YEAR OF BIRTH, "
            "each with a source URL. First RESTATE all six rows in the format "
            "'species -> describer -> description year -> describer's birth year'. Then MERGE: "
            "compare the six DESCRIBERS' BIRTH YEARS (not the description years) and find the "
            "MAXIMUM — the describer born most recently. Report (a) the species that this "
            "describer described, naming the describer and stating explicitly that this person was "
            "born most recently, with their birth year; (b) all six rows; (c) the subspecies, "
            "species and subfamily you passed through; citing every source URL, including each "
            "describer's own page."
        ),
    }
