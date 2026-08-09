"""
Test 101: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Branch-then-chain (same shape as test 095): the round-2 body is unknown until round 1's
name-collision elimination resolves which 'Cassini' feature survives.

    ROUND 1  (breadth / ambiguity — 4 features share the name 'Cassini', eliminate to ONE)
      Several planetary features are named 'Cassini'. A memory-anchored agent pictures the well-
      documented LUNAR crater (or the famous Cassini spacecraft). The PAGE-ONLY disambiguator:
      exactly ONE is a named surface region on a MOON OF SATURN — Cassini Regio, the dark region
      on IAPETUS. The agent must open each feature's page and confirm the body it lies on rather
      than assume the lunar / famous one.

    ROUND 2 / 3  (forward chain from the SURVIVOR — the keystone)
      Cassini Regio's page states it covers the leading hemisphere of Saturn's moon Iapetus.
      Forward-hop to IAPETUS and read the moon's MEAN RADIUS from the infobox: 734.4 km
      (mean diameter ~1,469 km) — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — the body each 'Cassini' feature lies on (a moon of Saturn?):
  ┌───────────────────────────────────────────────┬────────────────────────┬────────────┐
  │ Cassini (lunar crater)                          │ the Moon               │ eliminated │
  │ Cassini (Martian crater)                        │ Mars                   │ eliminated │
  │ Cassini spacecraft / Cassini Division           │ a probe / Saturn's ring│ eliminated │
  │ Cassini Regio ← SURVIVOR                         │ Iapetus (moon of Saturn)│ SURVIVES  │
  └───────────────────────────────────────────────┴────────────────────────┴────────────┘
      The elimination is CATEGORICAL (which body), not a numeric margin — one noisy read cannot flip it.

  ROUND 2/3 keystone:
      Cassini Regio is on Iapetus  →  Iapetus infobox mean radius = 734.4 km
      (mean diameter ~1,469 km).  [KEYSTONE]

Why the keystone is leak-resistant: Iapetus's mean radius (734.4 km) is an obscure moon dimension no
consumer LLM recalls precisely, reachable only via the correct (Saturnian-moon) 'Cassini' survivor.
The tokens "734.4" and "1,469" are distinctive; a wrong survivor (lunar/Martian crater) points at a
different body entirely, and naming Cassini Regio without the Iapetus hop still fails the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Four planetary features named 'Cassini'. `disc_rx` is the PAGE-ONLY disambiguating fact (which body
# it lies on); only the survivor is on a moon of Saturn. `name_rx` is that candidate's distinctive token.
FEATURES: List[Dict[str, Any]] = [
    {
        "key": "lunar", "name": "Cassini (lunar crater)",
        "desc": "an impact crater on the Moon (in Mare Imbrium)",
        "name_rx": r"lunar|imbrium", "disc_rx": r"lunar|imbrium",
        "slug_rx": r"wiki/cassini_\(lunar_crater\)", "survivor": False,
    },
    {
        "key": "martian", "name": "Cassini (Martian crater)",
        "desc": "an impact crater on Mars (in the Arabia quadrangle)",
        "name_rx": r"martian|arabia", "disc_rx": r"\bmars\b|martian",
        "slug_rx": r"wiki/cassini_\(martian_crater\)", "survivor": False,
    },
    {
        "key": "spacecraft", "name": "Cassini spacecraft / Cassini Division",
        "desc": "the Saturn probe, and the gap in Saturn's rings",
        "name_rx": r"spacecraft|probe|division", "disc_rx": r"spacecraft|probe|huygens",
        "slug_rx": r"wiki/cassini(?:-huygens|_division)", "survivor": False,
    },
    {
        "key": "regio", "name": "Cassini Regio",
        "desc": "a dark surface region on a moon of Saturn",
        "name_rx": r"regio|iapetus", "disc_rx": r"iapetus",
        "slug_rx": r"wiki/cassini_regio", "survivor": True,
    },
]
SURVIVOR = next(f for f in FEATURES if f["survivor"])  # Cassini Regio (Iapetus)

# ROUND-2/3 keystone: Iapetus infobox mean radius = 734.4 km (mean diameter ~1,469 km).
KEYSTONE_RX = re.compile(r"\b734(?:\.4)?\b|\b1,?469\b", re.IGNORECASE)
IAPETUS_SLUG = r"wiki/iapetus_\(moon\)"
# The forward-chain body the survivor page names (round 2).
CHAIN_RX = re.compile(r"iapetus", re.IGNORECASE)
REGIO_SLUG = r"wiki/cassini_regio"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "101",
        "test_name": "Tier 5: Branch-to-eliminate then chain forward (Cassini features -> Iapetus mean radius)",
        "difficulty_level": "9/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {f['name']} — {f['desc']}" for i, f in enumerate(FEATURES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has two stages; the second is unknown until the first is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Several planetary things are named 'Cassini':\n"
        f"{listing}\n"
        "Exactly ONE of these is a named dark surface REGION lying on a MOON OF SATURN. Open EACH "
        "page and confirm which body it lies on to determine which one — do NOT default to the famous "
        "lunar crater or the Cassini spacecraft. Determine the body of all four.\n\n"
        "STAGE 2 — follow the survivor forward. The surviving region's page names the moon of Saturn "
        "it lies on. Open THAT moon's page and read its MEAN RADIUS (or mean diameter) in kilometres "
        "directly from the infobox.\n\n"
        "Report: (a) the mean radius of the moon the survivor region lies on (this single figure is the "
        "keystone answer); (b) which 'Cassini' feature was the survivor and the body each of the four "
        "lies on; (c) the name of the moon; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The mean radius (or diameter) of the moon the survivor region lies on (the keystone)",
        "Which 'Cassini' feature is on a moon of Saturn (survivor) + the body of each of the four",
        "The name of the moon (Iapetus)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per 'Cassini' candidate) plus the moon",
        "Determines the body of ALL FOUR 'Cassini' features (branch-to-eliminate)",
        "Correctly elects Cassini Regio (on Iapetus) as the moon-of-Saturn survivor",
        "Identifies Iapetus as the moon",
        "Reports Iapetus's mean radius (734.4 km / mean diameter ~1,469 km)",
        "Cites the survivor Cassini Regio page and the Iapetus page",
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
            "reason": f"{n} visit(s) (target >=5: four 'Cassini' candidates + the survivor's moon; >=4 to pass)"}


def validate_keystone_radius(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): Iapetus's mean radius (734.4 km) / mean diameter (~1,469 km). Leak-
    resistant — reachable only by electing the moon-of-Saturn survivor (Cassini Regio, not the famous
    lunar crater), reading the moon it lies on, and reading that moon's radius."""
    passed = _keystone_ok(result)
    return {"check": "keystone_radius", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Iapetus mean radius 734.4 km (diameter ~1,469 km) present" if passed
                      else "Keystone (734.4 km mean radius / 1,469 km diameter, Iapetus) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR 'Cassini' features the agent resolved to
    their body. Separates a structured agent from one that guesses the lunar crater. NOT short-
    circuited on the keystone. Credited only via each candidate's own distinct name token, and the
    text match is ANDed with visit evidence (capped by visits) so zero visits scores 0."""
    text = _all_text(result)
    text_hits = [f["name"] for f in FEATURES
                 if re.search(f["name_rx"], text, re.IGNORECASE) and re.search(f["disc_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(FEATURES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} 'Cassini' features resolved to their body from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; "
                      f"{len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor_and_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Cassini Regio) AND Iapetus. Short-circuits to 0 when the
    keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/moon resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(re.search(r"cassini\s+regio|\bregio\b", text, re.IGNORECASE))
    has_chain = bool(CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_chain)
    return {"check": "survivor_and_chain", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Cassini Regio)={has_survivor}, Iapetus={has_chain}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages read. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [f["slug_rx"] for f in FEATURES] + [IAPETUS_SLUG]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor Cassini Regio + Iapetus)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_radius,
        validate_branch_exploration,
        validate_survivor_and_chain,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (waves 4 -> 1 -> 1). STRUCTURE only: names the
    GIVEN candidate 'Cassini' features and the GIVEN moon-of-Saturn criterion, but leaks NO body, NOT
    which feature survives, NOT the moon name, and NOT the radius/diameter figure."""
    feat_leaves = [
        {
            "id": f"feat_{f['key']}",
            "instruction": (
                f"Open the Wikipedia page for {f['name']} — {f['desc']}. Determine which BODY it lies on "
                "or refers to (the Moon, Mars, a spacecraft/Saturn's rings, or a moon of Saturn) directly "
                f"from the page. Report the feature's name ({f['name']}), the body it lies on, and the "
                "exact Wikipedia URL. Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{f['name']} — body: __ — source URL",
            "depends_on": [],
        }
        for f in FEATURES
    ]
    survivor_leaf = {
        "id": "survivor_moon",
        "instruction": (
            "You are given the four things named 'Cassini' and the body each lies on:\n"
            "  Cassini (lunar crater) -> {feat_lunar}\n"
            "  Cassini (Martian crater) -> {feat_martian}\n"
            "  Cassini spacecraft / Cassini Division -> {feat_spacecraft}\n"
            "  Cassini Regio -> {feat_regio}\n"
            "Determine which SINGLE one is a named surface region on a MOON OF SATURN. Open THAT "
            "surviving region's Wikipedia page. It names the moon of Saturn it lies on. Report the "
            "surviving region, the NAME of that moon, and the moon's exact Wikipedia URL. Do not guess "
            "from memory."
        ),
        "expect": "SURVIVING (moon-of-Saturn) 'Cassini' region + the name of the moon it lies on — source URL",
        "depends_on": [f"feat_{f['key']}" for f in FEATURES],
    }
    radius_leaf = {
        "id": "moon_radius",
        "instruction": (
            "Open the Wikipedia page of the moon identified in the previous step ({survivor_moon}). Read "
            "that moon's MEAN RADIUS (or mean diameter) in kilometres directly from the infobox. Report "
            "the mean radius and the source URL. Do not guess from memory."
        ),
        "expect": "The moon's MEAN RADIUS (or diameter) in km — source URL",
        "depends_on": ["survivor_moon"],
    }
    return {
        "leaves": feat_leaves + [survivor_leaf, radius_leaf],
        "aggregation": (
            "You now have (1) the body each of the four 'Cassini' features lies on, (2) which single one "
            "is a region on a moon of Saturn (the survivor) and the moon it lies on, and (3) that moon's "
            "mean radius. Write out all four bodies BEFORE concluding which survives. Then report (a) the "
            "moon's mean radius in km — this single figure is the keystone answer; (b) which 'Cassini' "
            "feature was the survivor and the body of each of the four; (c) the moon's name; citing every "
            "source URL."
        ),
    }
