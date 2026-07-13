"""
Test 099: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Branch-then-chain (same shape as test 095): the round-2 target is unknown until round 1's
elimination resolves which candidate survives.

    ROUND 1  (breadth / ambiguity — 4 genuine candidates, eliminate to ONE)
      Four churches are dedicated to St. Stephen. A memory-anchored agent reaches for VIENNA's
      Stephansdom (the famous Gothic one). The PAGE-ONLY disambiguator: exactly ONE is described
      on its page as having (historically) the LARGEST church/cathedral pipe organ — St. Stephen's
      Cathedral, PASSAU, not Vienna. The agent must open each cathedral's page and check the organ
      claim rather than default to the famous one.

    ROUND 2 / 3  (forward chain from the SURVIVOR — the keystone)
      Passau Cathedral's page describes its grand organ and gives its total PIPE COUNT: 17,774
      pipes (across 233 registers) — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — page organ claim (largest church organ?):
  ┌──────────────────────────────────────────────────┬─────────────────────────────┬────────────┐
  │ St. Stephen's Cathedral, Vienna (Stephansdom)      │ famous Gothic; not largest  │ eliminated │
  │ St. Stephen's Basilica, Budapest                   │ not the largest organ       │ eliminated │
  │ St. Stephen's Cathedral, Passau ← SURVIVOR         │ "largest church organ       │ SURVIVES   │
  │                                                    │  outside the U.S." (once    │            │
  │                                                    │  the largest in the world)  │            │
  │ St. Stephen's Cathedral, Brisbane                  │ not the largest organ       │ eliminated │
  └──────────────────────────────────────────────────┴─────────────────────────────┴────────────┘
      The elimination is CATEGORICAL (only Passau's page claims the largest organ), not a numeric
      margin — one noisy read cannot flip it.

  ROUND 2/3 keystone:
      Passau Cathedral organ = 17,774 pipes, 233 registers.  [KEYSTONE: 17,774 pipes]

Why the keystone is leak-resistant: the exact pipe count (17,774) is an obscure five-figure page
fact no consumer LLM recalls; even knowing "Passau", a model would guess a round "10,000" or the
oft-mis-cited "17,974". The token \b17,?774\b is distinctive and collides with no other figure on
the chain — only reading the correct (largest-organ) survivor page can satisfy it. Naming Passau
without reading the organ spec, or defaulting to famous Vienna, fails the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Four churches dedicated to St. Stephen. `disc_rx` is the PAGE-ONLY disambiguating fact (does the
# page describe the largest church organ). Only the survivor's page makes the 'largest' claim.
CATHEDRALS: List[Dict[str, Any]] = [
    {
        "key": "vienna", "name": "St. Stephen's Cathedral, Vienna",
        "desc": "the Stephansdom, the famous Gothic cathedral of Vienna",
        "name_rx": r"vienna|stephansdom",
        "disc_rx": r"austria|gothic|wien",
        "slug_rx": r"wiki/st(?:\.|%27s)?[_ ]stephen'?s?[_ ]cathedral,?_?vienna|wiki/stephansdom",
        "survivor": False,
    },
    {
        "key": "budapest", "name": "St. Stephen's Basilica, Budapest",
        "desc": "the neoclassical basilica of Budapest",
        "name_rx": r"budapest|basilica",
        "disc_rx": r"hungary|neoclassical",
        "slug_rx": r"wiki/st(?:\.|%27s)?[_ ]stephen'?s?[_ ]basilica",
        "survivor": False,
    },
    {
        "key": "passau", "name": "St. Stephen's Cathedral, Passau",
        "desc": "the Baroque cathedral (Passau Cathedral) in Bavaria",
        "name_rx": r"passau",
        "disc_rx": r"bavaria|germany|largest",
        "slug_rx": r"wiki/passau_cathedral|wiki/st(?:\.|%27s)?[_ ]stephen'?s?[_ ]cathedral,?_?passau",
        "survivor": True,
    },
    {
        "key": "brisbane", "name": "St. Stephen's Cathedral, Brisbane",
        "desc": "the Gothic Revival cathedral of Brisbane, Australia",
        "name_rx": r"brisbane",
        "disc_rx": r"australia|queensland",
        "slug_rx": r"wiki/st(?:\.|%27s)?[_ ]stephen'?s?[_ ]cathedral,?_?brisbane",
        "survivor": False,
    },
]
SURVIVOR = next(c for c in CATHEDRALS if c["survivor"])  # St. Stephen's Cathedral, Passau

# ROUND-2/3 keystone: Passau Cathedral organ = 17,774 pipes. \b17,?774\b matches "17,774"/"17774"
# but NOT the mis-cited "17,974" nor a longer embedded number.
KEYSTONE_RX = re.compile(r"\b17,?774\b", re.IGNORECASE)
PASSAU_SLUG = r"wiki/passau_cathedral"
_PASSAU_RX = re.compile(r"passau", re.IGNORECASE)
# The forward-chain attribute: the survivor's organ is the largest.
CHAIN_RX = re.compile(r"largest.{0,40}organ|organ.{0,40}largest|233\s*(?:register|stop|rank)", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "099",
        "test_name": "Tier 5: Branch-to-eliminate then chain forward (St. Stephen's cathedrals -> Passau organ pipes)",
        "difficulty_level": "9/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {c['name']} — {c['desc']}" for i, c in enumerate(CATHEDRALS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has two stages; the second is unknown until the first is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Four churches are all dedicated to St. Stephen:\n"
        f"{listing}\n"
        "Exactly ONE of these four is described on its page as having (historically) the LARGEST "
        "church / cathedral pipe organ. Open EACH cathedral's page and check its organ claim to "
        "determine which one — do NOT simply pick the most famous (Vienna's Stephansdom). Determine "
        "the organ status of all four.\n\n"
        "STAGE 2 — read the keystone. On the surviving cathedral's page, read the TOTAL NUMBER OF "
        "PIPES in its grand organ directly from the page.\n\n"
        "Report: (a) the total pipe count of the survivor's organ (this single figure is the keystone "
        "answer); (b) which of the four St. Stephen's churches has the largest organ (the survivor); "
        "(c) how many registers/stops its organ has; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The total pipe count of the survivor cathedral's organ (the keystone)",
        "Which St. Stephen's church has the largest organ (Passau, the survivor)",
        "The number of registers/stops of that organ",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per cathedral candidate)",
        "Checks the organ claim of ALL FOUR St. Stephen's churches (branch-to-eliminate)",
        "Correctly elects Passau Cathedral as the largest-organ survivor",
        "Reports the organ's total pipe count (17,774)",
        "Cites the survivor Passau Cathedral page",
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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Grounding requirement: a correct-but-ungrounded parametric-memory guess (zero page
    visits) must not earn keystone credit."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    return grounded and bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: one per St. Stephen's cathedral candidate)"}


def validate_keystone_pipes(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the Passau organ's total pipe count (17,774). Leak-resistant — reachable
    only by electing the largest-organ survivor (Passau, not famous Vienna) and reading the pipe
    count. A memory guess or a wrong survivor cannot produce it."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_pipes", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Organ pipe count 17,774 present" if passed
                      else "Keystone pipe count (17,774, Passau Cathedral organ) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR St. Stephen's churches the agent resolved to
    their organ status. Separates a structured multi-round agent from one that guesses Vienna. NOT
    short-circuited on the keystone. Credited only via each candidate's own distinct name token, and
    the text match is ANDed with visit evidence (capped by visits) so zero visits scores 0."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CATHEDRALS
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["disc_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CATHEDRALS)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} St. Stephen's churches resolved to their organ status from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; "
                      f"{len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor_and_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Passau) AND states its organ is the largest (or the 233
    registers). Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "survivor_and_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/organ resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(_PASSAU_RX.search(text))
    has_chain = bool(CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_chain)
    return {"check": "survivor_and_chain", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Passau)={has_survivor}, largest-organ/233-registers={has_chain}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages read. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [c["slug_rx"] for c in CATHEDRALS]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. the survivor Passau + another)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_pipes,
        validate_branch_exploration,
        validate_survivor_and_chain,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (waves 4 -> 1 -> 1). STRUCTURE only: names the
    GIVEN candidate churches and the GIVEN largest-organ criterion, but leaks NO organ verdict, NOT
    which cathedral survives, and NOT the pipe count / register count."""
    cath_leaves = [
        {
            "id": f"cath_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read whether its page "
                "describes its pipe organ as (historically) the LARGEST church / cathedral organ, or "
                f"not. Report the cathedral's name ({c['name']}), that organ verdict (largest or not), "
                "and the exact Wikipedia URL. Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{c['name']} — organ verdict: LARGEST or NOT — source URL",
            "depends_on": [],
        }
        for c in CATHEDRALS
    ]
    survivor_leaf = {
        "id": "survivor_cathedral",
        "instruction": (
            "You are given the four St. Stephen's churches and the organ verdict of each:\n"
            "  St. Stephen's Cathedral, Vienna -> {cath_vienna}\n"
            "  St. Stephen's Basilica, Budapest -> {cath_budapest}\n"
            "  St. Stephen's Cathedral, Passau -> {cath_passau}\n"
            "  St. Stephen's Cathedral, Brisbane -> {cath_brisbane}\n"
            "Determine which SINGLE one is described as having the LARGEST church / cathedral organ. "
            "Report that surviving cathedral and its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The SURVIVING (largest-organ) St. Stephen's cathedral — source URL",
        "depends_on": [f"cath_{c['key']}" for c in CATHEDRALS],
    }
    pipes_leaf = {
        "id": "organ_pipes",
        "instruction": (
            "Open the Wikipedia page of the surviving cathedral identified in the previous step "
            "({survivor_cathedral}) — the one with the largest organ. Read the TOTAL NUMBER OF PIPES "
            "in its grand organ (and the number of registers/stops) directly from the page. Report the "
            "total pipe count and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving cathedral organ's TOTAL pipe count — source URL",
        "depends_on": ["survivor_cathedral"],
    }
    return {
        "leaves": cath_leaves + [survivor_leaf, pipes_leaf],
        "aggregation": (
            "You now have (1) the organ verdict of each of the four St. Stephen's churches, (2) which "
            "single one has the largest organ (the survivor), and (3) that organ's total pipe count. "
            "Write out all four organ verdicts BEFORE concluding which survives. Then report (a) the "
            "survivor organ's total pipe count — this single figure is the keystone answer; (b) which "
            "St. Stephen's church was the survivor; (c) its number of registers/stops; citing every "
            "source URL."
        ),
    }
