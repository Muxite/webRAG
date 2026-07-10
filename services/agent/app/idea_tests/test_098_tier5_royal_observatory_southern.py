"""
Test 098: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Same branch-then-chain shape as test 095: a genuine multi-round graph-of-thoughts task that a
flat single-round fan-out or a linear chain cannot solve, because the entity to investigate in
round 2 is unknown until round 1's elimination resolves which candidate survives.

    ROUND 1  (breadth / ambiguity — 4 genuine candidates, eliminate to ONE)
      Four institutions are all styled 'Royal Observatory'. A memory-anchored agent reaches
      straight for GREENWICH (the prime-meridian one everyone knows). The PAGE-ONLY
      disambiguator: exactly ONE lies in the SOUTHERN HEMISPHERE — read each observatory's
      location / coordinates from its infobox. Greenwich (51°N), Edinburgh (55°N) and the
      Royal Observatory of Belgium at Uccle (50°N) are all northern; only the Royal Observatory,
      Cape of Good Hope (33°S, South Africa) is southern. Guessing the famous one breaks the chain.

    ROUND 2  (forward chain from the SURVIVOR — unknowable until round 1 resolves)
      The Cape observatory's OWN page names its principal historic photographic refractor: the
      McClean telescope (a.k.a. the Victoria telescope), built by the Grubb Telescope Company.

    ROUND 3  (keystone — obscure, page-only)
      Read that refractor's PHOTOGRAPHIC object-glass aperture: 24 inches (610 mm) — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — infobox location / latitude (southern hemisphere?):
  ┌───────────────────────────────────────────────┬───────────────────────┬────────────┐
  │ Royal Observatory, Greenwich                    │ 51°N  (London)         │ eliminated │
  │ Royal Observatory, Edinburgh                    │ 55°N  (Scotland)       │ eliminated │
  │ Royal Observatory of Belgium (Uccle)            │ 50°N  (Brussels)       │ eliminated │
  │ Royal Observatory, Cape of Good Hope ← SURVIVOR │ 33°56'05"S (S. Africa) │ SURVIVES   │
  └───────────────────────────────────────────────┴───────────────────────┴────────────┘
      The elimination is CATEGORICAL (a hemisphere string, S vs N), not a numeric margin — one
      noisy latitude read cannot flip it.

  ROUND 2 forward chain:
      Royal Observatory, Cape of Good Hope  →  the McClean (Victoria) telescope, a Grubb refractor
      with "18-inch visual, 24-inch photographic and 8-inch guide" object glasses.

  ROUND 3 keystone:
      The McClean/Victoria refractor's PHOTOGRAPHIC object glass = 24 inches (610 mm).  [KEYSTONE]

Why the keystone is leak-resistant: the aperture of the Cape observatory's historic photographic
refractor is an obscure instrument spec no consumer LLM recalls; even knowing "the Cape", a model
would guess. The token requires "24" adjacent to an inch unit (or "610 mm"), so a bare year or a
different aperture on the page (4/6/13/18/40 inch) cannot satisfy it — only reading the correct
instrument on the correct (southern) survivor page can, which is the depth the branch-then-chain
forces. Naming "the Cape" without the telescope hop still fails the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Four institutions styled 'Royal Observatory'. `disc_rx` is the PAGE-ONLY disambiguating fact
# (infobox latitude / hemisphere); only the survivor is southern. `name_rx` is that candidate's
# distinctive name token (pairwise distinct, so coverage never cross-credits).
OBSERVATORIES: List[Dict[str, Any]] = [
    {
        "key": "greenwich", "name": "Royal Observatory, Greenwich",
        "desc": "the prime-meridian observatory in London (home of GMT)",
        "name_rx": r"greenwich", "disc_rx": r"\b51\b",
        "slug_rx": r"wiki/royal_observatory,_greenwich", "survivor": False,
    },
    {
        "key": "edinburgh", "name": "Royal Observatory, Edinburgh",
        "desc": "the observatory on Blackford Hill in Scotland",
        "name_rx": r"edinburgh", "disc_rx": r"\b55\b",
        "slug_rx": r"wiki/royal_observatory,_edinburgh", "survivor": False,
    },
    {
        "key": "belgium", "name": "Royal Observatory of Belgium",
        "desc": "the national observatory at Uccle, near Brussels",
        "name_rx": r"belgium|uccle", "disc_rx": r"\b50\b",
        "slug_rx": r"wiki/royal_observatory_of_belgium", "survivor": False,
    },
    {
        "key": "cape", "name": "Royal Observatory, Cape of Good Hope",
        "desc": "the observatory near Cape Town, South Africa",
        "name_rx": r"cape\s+of\s+good\s+hope|cape\s+town|\bcape\b",
        "disc_rx": r"\b33\b|south\s+africa",
        "slug_rx": r"wiki/royal_observatory,_cape_of_good_hope", "survivor": True,
    },
]
SURVIVOR = next(o for o in OBSERVATORIES if o["survivor"])  # Royal Observatory, Cape of Good Hope

# ROUND-3 keystone: the McClean/Victoria refractor's PHOTOGRAPHIC object glass = 24 inches (610 mm).
# "24" must be adjacent to an inch unit so a bare year cannot match; "610 mm" is the metric form.
KEYSTONE_RX = re.compile(r"\b24[\s-]*(?:in\b|inch|inches|″|\")|\b610\s*mm", re.IGNORECASE)
# The forward-chain instrument the survivor page names (round 2).
CHAIN_RX = re.compile(r"mcclean|mc\s*clean|victoria\s+telescope|victoria\s+refractor", re.IGNORECASE)
CAPE_SLUG = r"wiki/royal_observatory,_cape_of_good_hope"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "098",
        "test_name": "Tier 5: Branch-to-eliminate then chain forward (Royal Observatories -> Cape refractor aperture)",
        "difficulty_level": "9/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {o['name']} — {o['desc']}" for i, o in enumerate(OBSERVATORIES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has three stages; each stage's target is unknown until the "
        "previous stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Four institutions are all styled 'Royal Observatory':\n"
        f"{listing}\n"
        "Exactly ONE of these four lies in the SOUTHERN HEMISPHERE. Open EACH observatory's page and "
        "read its location / latitude from the infobox to determine which one — the other three are "
        "in the northern hemisphere (London, Scotland and Brussels). Determine the hemisphere of all "
        "four; do not simply pick the most famous (Greenwich).\n\n"
        "STAGE 2 — follow the survivor forward. Open the surviving (southern) observatory's page. It "
        "names its principal historic photographic refractor (a great equatorial telescope built by "
        "the Grubb Telescope Company). Identify that instrument.\n\n"
        "STAGE 3 — read the keystone. Read that refractor's PHOTOGRAPHIC object-glass APERTURE "
        "(lens diameter) directly from the page, in inches or millimetres.\n\n"
        "Report: (a) the aperture of the survivor's historic photographic refractor (this single "
        "figure is the keystone answer); (b) which of the four Royal Observatories was the southern "
        "survivor and the hemisphere of each of the four; (c) the name of the refractor; citing the "
        "exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The aperture of the survivor observatory's historic photographic refractor (the keystone)",
        "Which Royal Observatory is in the southern hemisphere (survivor) + each of the four hemispheres",
        "The name of the survivor's historic refractor (the McClean / Victoria telescope)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per observatory candidate) plus the survivor's instrument",
        "Determines the hemisphere of ALL FOUR Royal Observatories (branch-to-eliminate)",
        "Correctly elects the Cape of Good Hope observatory as the southern survivor",
        "Identifies the McClean (Victoria) refractor",
        "Reports the refractor's photographic aperture (24 inches / 610 mm)",
        "Cites the survivor observatory page",
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
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: four observatory candidates + the survivor's instrument; >=4 to pass)"}


def validate_keystone_aperture(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the McClean/Victoria refractor's photographic aperture (24 in / 610 mm).
    Leak-resistant — reachable only by electing the Cape (southern) survivor, reading its historic
    refractor, and reading the aperture. A memory guess or a wrong (famous Greenwich) survivor cannot
    produce it."""
    passed = _keystone_ok(result)
    return {"check": "keystone_aperture", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Refractor aperture 24 in (610 mm) present" if passed
                      else "Keystone aperture (24 inches / 610 mm, McClean/Victoria refractor) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR Royal Observatories the agent resolved to
    their hemisphere. Separates a structured multi-round agent from a shallow one that guesses
    Greenwich. NOT short-circuited on the keystone. Each hemisphere is credited only with its
    candidate's own distinct name token, so there is no cross-crediting between the three northern
    ones. The text match is ANDed with actual visit evidence (credited count capped by visits), so a
    run that reads no pages scores 0 no matter what it writes."""
    text = _all_text(result)
    text_hits = [o["name"] for o in OBSERVATORIES
                 if re.search(o["name_rx"], text, re.IGNORECASE) and re.search(o["disc_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(OBSERVATORIES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} Royal Observatories resolved to their hemisphere from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; "
                      f"{len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor_and_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the chain was resolved correctly — names the survivor (Cape of Good Hope)
    AND the McClean/Victoria refractor. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/instrument resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(re.search(SURVIVOR["name_rx"], text, re.IGNORECASE))
    has_chain = bool(CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_chain)
    return {"check": "survivor_and_chain", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Cape of Good Hope)={has_survivor}, McClean/Victoria refractor={has_chain}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages read. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [o["slug_rx"] for o in OBSERVATORIES]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. the survivor Cape observatory + another)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_aperture,
        validate_branch_exploration,
        validate_survivor_and_chain,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (waves 4 -> 1 -> 1). STRUCTURE only: names
    the GIVEN candidate observatories and the GIVEN southern-hemisphere criterion, but leaks NO
    hemisphere, NOT which observatory survives, NOT the refractor name, and NOT the aperture."""
    obs_leaves = [
        {
            "id": f"obs_{o['key']}",
            "instruction": (
                f"Open the Wikipedia page for {o['name']} — {o['desc']}. Read its LOCATION / LATITUDE "
                "(and thus which hemisphere it is in, northern or southern) directly from the infobox. "
                f"Report the observatory's name ({o['name']}), its hemisphere, and the exact Wikipedia "
                "URL. Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{o['name']} — hemisphere: NORTHERN/SOUTHERN — source URL",
            "depends_on": [],
        }
        for o in OBSERVATORIES
    ]
    survivor_leaf = {
        "id": "survivor_instrument",
        "instruction": (
            "You are given the four Royal Observatories and the hemisphere of each:\n"
            "  Royal Observatory, Greenwich -> {obs_greenwich}\n"
            "  Royal Observatory, Edinburgh -> {obs_edinburgh}\n"
            "  Royal Observatory of Belgium -> {obs_belgium}\n"
            "  Royal Observatory, Cape of Good Hope -> {obs_cape}\n"
            "Determine which SINGLE one lies in the SOUTHERN HEMISPHERE. Open THAT surviving "
            "observatory's Wikipedia page. It names its principal historic photographic refractor (a "
            "great equatorial telescope built by the Grubb Telescope Company). Report the surviving "
            "observatory, the name of that refractor, and the observatory's exact Wikipedia URL. Do "
            "not guess from memory."
        ),
        "expect": "SURVIVING (southern-hemisphere) Royal Observatory + the name of its historic photographic refractor — source URL",
        "depends_on": [f"obs_{o['key']}" for o in OBSERVATORIES],
    }
    aperture_leaf = {
        "id": "refractor_aperture",
        "instruction": (
            "Using the historic photographic refractor identified in the previous step "
            "({survivor_instrument}) on the surviving observatory's page, read that refractor's "
            "PHOTOGRAPHIC OBJECT-GLASS APERTURE (its lens diameter) in inches or millimetres directly "
            "from the page. Report the aperture and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving refractor's photographic object-glass aperture (inches or mm) — source URL",
        "depends_on": ["survivor_instrument"],
    }
    return {
        "leaves": obs_leaves + [survivor_leaf, aperture_leaf],
        "aggregation": (
            "You now have (1) the hemisphere of each of the four Royal Observatories, (2) which single "
            "one is in the southern hemisphere (the survivor) and the name of its historic photographic "
            "refractor, and (3) that refractor's aperture. Write out all four hemispheres BEFORE "
            "concluding which survives. Then report (a) the aperture of the survivor's photographic "
            "refractor — this single figure is the keystone answer; (b) which Royal Observatory was the "
            "southern survivor and the hemisphere of each of the four; (c) the refractor's name; citing "
            "every source URL."
        ),
    }
