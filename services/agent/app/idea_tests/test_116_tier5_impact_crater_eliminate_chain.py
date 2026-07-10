r"""
Test 116: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD (impact craters).
Level: graph   Weight: long   Difficulty: 10/10

Same branch-then-chain shape as test 095: a multi-round, branching graph-of-thoughts
exploration where the entity to read in the final round is UNKNOWN until an elimination
resolves which candidate survives. A flat "one parallel burst, pick the famous one" agent
cannot solve it.

    ROUND 1  (breadth / ambiguity — four genuine terrestrial impact structures, eliminate to ONE)
      Chicxulub owns impact-crater recall — it is the famous 'dinosaur' (K-Pg) crater. But the
      disambiguator is PAGE-ONLY and categorical: exactly ONE of these structures is described
      on its own page as the LARGEST VERIFIED / CONFIRMED impact structure on Earth, and it is
      NOT Chicxulub (which its page calls the largest mostly-INTACT crater). Resolving it requires
      opening each structure's page and reading its ranking, not defaulting to the extinction crater.

    ROUND 2  (elect the survivor)
      Only the Vredefort impact structure (Free State, South Africa) is the largest verified one.

    ROUND 3  (keystone — obscure, page-only)
      Read the survivor structure's AGE in billions of years from its infobox: the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — 'largest VERIFIED impact structure on Earth'?
  ┌───────────────────────────────────────────────┬───────────────────────────────┬────────────┐
  │ Chicxulub crater (Yucatán, Mexico)             │ "largest mostly INTACT" (66 Ma)│ eliminated │
  │ Vredefort impact structure  ← SURVIVOR         │ "Largest VERIFIED ... on Earth"│ SURVIVES   │
  │ Sudbury Basin (Ontario, Canada)                │ ~130 km, ~1.85 Ga             │ eliminated │
  │ Chesapeake Bay impact crater (Virginia, USA)   │ ~85 km, ~35 Ma (Eocene)       │ eliminated │
  └───────────────────────────────────────────────┴───────────────────────────────┴────────────┘
      Vredefort's infobox reads "Confirmed" with the heading 'Largest verified impact structure on
      Earth'; Chicxulub's page explicitly calls itself the largest mostly-intact crater, NOT the
      largest verified structure. The elimination is a categorical string, not a numeric margin.

  ROUND 3 keystone:
      Vredefort impact structure — infobox age = 2.023 Ga (2,023 ± 4 Ma), Orosirian, Paleoproterozoic.
      [KEYSTONE]

Why leak-resistant: 2.023 Ga is a small, precise infobox figure no consumer LLM recalls; even
knowing the structure, a model would guess. The wrong (famous) survivor Chicxulub gives the K-Pg
age of 66 Ma instead — the exact failure the elimination is built to expose. The token \b2\.023\b
collides with none of the decoy ages (66 Ma, ~1.85 Ga, ~35 Ma).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Four genuine terrestrial impact structures. `disamb_rx` is that candidate's OWN distinctive
# page fact (proves its page was read); `name_rx` is its distinctive name token (all four pairwise
# distinct so coverage never cross-credits).
CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "chicxulub", "name": "Chicxulub crater",
        "desc": "the Chicxulub crater in Yucatán, Mexico — the famous 'dinosaur' (K-Pg) impact crater",
        "name_rx": r"chicxulub", "disamb_rx": r"\b66\b|dinosaur|k[-\s]?pg|cretaceous|mostly\s+intact",
        "slug_rx": r"wiki/chicxulub_crater", "survivor": False,
    },
    {
        "key": "vredefort", "name": "Vredefort impact structure",
        "desc": "the Vredefort impact structure (Vredefort Dome) in the Free State, South Africa",
        "name_rx": r"vredefort", "disamb_rx": r"\b300\b|largest\s+(?:verified|confirmed)|south\s+africa",
        "slug_rx": r"wiki/vredefort", "survivor": True,
    },
    {
        "key": "sudbury", "name": "Sudbury Basin",
        "desc": "the Sudbury Basin in Ontario, Canada",
        "name_rx": r"sudbury", "disamb_rx": r"\b130\b|1\.85|ontario|nickel",
        "slug_rx": r"wiki/sudbury_basin", "survivor": False,
    },
    {
        "key": "chesapeake", "name": "Chesapeake Bay impact crater",
        "desc": "the Chesapeake Bay impact crater in Virginia, USA",
        "name_rx": r"chesapeake", "disamb_rx": r"\b85\b|eocene|virginia|\b35\b",
        "slug_rx": r"wiki/chesapeake_bay_impact_crater", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Vredefort impact structure

# KEYSTONE: Vredefort's infobox age = 2.023 Ga (2,023 ± 4 Ma). \b2\.023\b matches "2.023" (Ga);
# the alt matches "2,023 Ma"/"2023 million years" but a bare "2023" (a year) needs the Ma/million unit.
KEYSTONE_RX = re.compile(r"\b2\.023\b|\b2,?023\s*(?:ma\b|million)", re.IGNORECASE)
SURVIVOR_SLUG = r"wiki/vredefort"
CRITERION = "the one described on its OWN page as the LARGEST VERIFIED / CONFIRMED impact structure on Earth"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "116",
        "test_name": "Tier 5: Branch-eliminate then chain (impact craters -> Vredefort age)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {c['name']} — {c['desc']}" for i, c in enumerate(CANDIDATES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous "
        "stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Consider these four terrestrial impact structures:\n"
        f"{listing}\n"
        f"Exactly ONE of these is {CRITERION} — and it is NOT the famous dinosaur/K-Pg crater "
        "(whose page instead calls it the largest mostly-INTACT crater). Open EACH structure's page "
        "and read its ranking/description to determine which one. Determine the status of all four; "
        "do not simply pick the most famous one.\n\n"
        "STAGE 2 — elect the survivor. Identify the single structure that is the largest verified "
        "impact structure on Earth.\n\n"
        "STAGE 3 — read the keystone. Open that structure's page and read its AGE in BILLIONS OF "
        "YEARS (Ga) directly from the infobox.\n\n"
        "Report: (a) the survivor structure's AGE in Ga (this single figure is the keystone answer); "
        "(b) which of the four was the survivor and each candidate's ranking/description; citing the "
        "exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor impact structure's age in billions of years (Ga) — the leak-resistant keystone",
        "Which structure is the largest verified impact structure on Earth (the survivor)",
        "Each of the four candidates' ranking/description",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per candidate structure)",
        "Determines the ranking/description of ALL FOUR candidates (branch-to-eliminate)",
        "Correctly elects the Vredefort impact structure (not the famous Chicxulub)",
        "Reports Vredefort's age (2.023 Ga / 2,023 Ma)",
        "Cites the survivor's page",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: one per candidate structure)"}


def validate_keystone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): Vredefort's infobox age (2.023 Ga). Leak-resistant. A memory guess or the
    famous wrong survivor (Chicxulub -> 66 Ma) cannot produce it."""
    passed = _keystone_ok(result)
    return {"check": "keystone_age", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Vredefort age 2.023 Ga (2,023 Ma) present" if passed
                      else "Keystone age (2.023 Ga, Vredefort impact structure) missing/incorrect"}


def validate_candidate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR structures the agent resolved to their own
    distinguishing page fact. Separates a structured multi-round agent (eliminates ALL four) from a
    shallow one that guesses the famous crater. NOT short-circuited on the keystone. Text presence is
    ANDed with the visit count (each resolved candidate needs a backing page read)."""
    text = _all_text(result)
    hits = [c["name"] for c in CANDIDATES
            if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["disamb_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "candidate_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} structures resolved to their own page fact from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the correct survivor (Vredefort). Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor election not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor named (Vredefort)={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages the chain had to read. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [c["slug_rx"] for c in CANDIDATES]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2, incl. the survivor Vredefort page)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone, validate_candidate_coverage, validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG for the graph_compiled variant. Three waves
    (four parallel candidate leaves -> one election leaf -> one keystone leaf). STRUCTURE only:
    names the GIVEN candidates and the GIVEN 'largest verified' criterion but leaks NO ranking
    result, NOT which structure survives, and NOT the age figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read how the page RANKS or "
                "describes this structure among Earth's impact craters (e.g. largest verified, largest "
                "mostly-intact, its diameter). Report the structure's name, its ranking/description, "
                "and the exact Wikipedia URL. Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{c['name']} — its ranking/description — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    election_leaf = {
        "id": "election",
        "instruction": (
            "You are given the four impact structures and each one's ranking/description:\n"
            "  Chicxulub crater -> {cand_chicxulub}\n"
            "  Vredefort impact structure -> {cand_vredefort}\n"
            "  Sudbury Basin -> {cand_sudbury}\n"
            "  Chesapeake Bay impact crater -> {cand_chesapeake}\n"
            "Determine which SINGLE one is described as the LARGEST VERIFIED / CONFIRMED impact "
            "structure on Earth (NOT the largest mostly-intact one). Report that surviving structure's "
            "name and its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The largest-verified impact structure (the survivor) — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    keystone_leaf = {
        "id": "keystone_age",
        "instruction": (
            "Open the Wikipedia page of the impact structure identified in the previous step "
            "({election}). Read that structure's AGE in BILLIONS OF YEARS (Ga) directly from the "
            "infobox. Report the structure's age in Ga and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving structure's age in Ga — source URL",
        "depends_on": ["election"],
    }
    return {
        "leaves": cand_leaves + [election_leaf, keystone_leaf],
        "aggregation": (
            "You now have (1) each of the four structures' ranking/description, (2) which single one is "
            "the largest verified impact structure on Earth (the survivor), and (3) that structure's age. "
            "Write out all four rankings BEFORE concluding which survives. Then report (a) the survivor's "
            "AGE in Ga — this single figure is the keystone answer; (b) which structure was the survivor "
            "and each candidate's ranking; citing every source URL."
        ),
    }
