"""
Test 159: Tier 5 (graph) — GENUINE NARROW-SEQUENTIAL (4-hop chain), the deliberate PAIR to the
mechanism suite's genuine wide-fan-out task.
Level: graph   Weight: long   Difficulty: 9/10   Category: narrow_sequential_chain

Mechanism under test (DAG v3 "Ledger" master plan §8.3 / promotion gate §6 — "wins across BOTH
wide and sequential task shapes"): this task is topologically a CHAIN with a fan-out width of
exactly 1 at every step. Hop k's search target is UNKNOWABLE until hop k-1 has actually been read
off a page, so there is nothing to parallelise: an engine that speculatively expands several
branches at once can only burn tool calls on pages that cannot possibly be on the path. The task
therefore must NOT reward breadth machinery — a correct linear ReAct agent and a correct graph
agent should score identically on the keystone, while the graph pays strictly more cost. The
un-gated ``validate_path_efficiency`` diagnostic makes that overhead *measurable* (visits per
required hop) instead of invisible.

Chain (each hop's answer is read off the previous hop's page — no hop is guessable in advance):

    HOP 1 (given start)  Bell Rock Lighthouse -> WHO was the resident engineer who directed the
                         works.                                            [intermediate]
    HOP 2 (from hop 1)   That engineer's page names a GRANDSON who became a world-famous
                         novelist -> identify the grandson.                [intermediate]
    HOP 3 (from hop 2)   The novelist's page gives his RESTING PLACE: a mountain on a Pacific
                         island -> identify that mountain.                 [terminal entity]
    HOP 4 (from hop 3)   Read the mountain's ELEVATION off its own page.   [KEYSTONE]

Ground truth (verified against live English Wikipedia, 2026-08-25):
  - Bell Rock Lighthouse — built/directed by **Robert Stevenson** (civil engineer, 1772-1850);
    the page notes Stevenson's own account of the works and the long attribution dispute with
    consulting engineer John Rennie's descendants (hop-1 disambiguation is page-resolvable:
    take the engineer who directed the works on site).
  - Robert Stevenson (civil engineer): "his son Thomas was the father of the author Robert Louis
    Stevenson" -> the novelist grandson is **Robert Louis Stevenson** (1850-1894).
  - Robert Louis Stevenson: infobox "Resting place: Mount Vaea"; body: "...bearing him on their
    shoulders to nearby Mount Vaea, where they buried him on a spot overlooking the sea..."
  - Mount Vaea (Upolu, Samoa): infobox "Elevation: 472 m (1,549 ft)".   [KEYSTONE = 472 m / 1,549 ft]

Keystone margin — every number reachable by a wrong turn on this chain is far from 472/1549:
  - STOP-EARLY at hop 1: Bell Rock Lighthouse tower height 36 m (118 ft).
  - STOP-EARLY at hop 2/3: Robert Stevenson 1772/1850, R. L. Stevenson 1850/1894.
  - WRONG-BRANCH at hop 3: Samoa's highest peak Mount Silisili, 1,858 m (6,096 ft) — the answer a
    model that guesses "a mountain in Samoa" instead of reading the burial page produces.
  - ADJACENT-PAGE decoy: Vailima (Stevenson's estate) publishes no elevation at all.
  No decoy shares a token with 472 or 1,549, so one noisy extraction cannot flip the gate.

Leak resistance: the intermediates are famous enough to be *verifiable*, but Mount Vaea's
elevation is a page-only infobox figure that a cheap model cannot produce from parametric memory —
and it is unreachable without having genuinely resolved hops 1-3 in order.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, waypoint_chain_coverage


CHAIN: List[Dict[str, Any]] = [
    {"key": "start", "name": "Bell Rock Lighthouse", "role": "start",
     "name_rx": r"bell\s+rock",
     "slug_rx": r"wiki/bell_rock_lighthouse"},
    {"key": "engineer", "name": "Robert Stevenson (civil engineer)", "role": "intermediate",
     # "Robert Stevenson" only — deliberately NOT matching "Robert Louis Stevenson" (the grandson),
     # so hop 1 and hop 2 stay separately creditable.
     "name_rx": r"robert\s+stevenson\b",
     "slug_rx": r"wiki/robert_stevenson"},
    {"key": "novelist", "name": "Robert Louis Stevenson", "role": "intermediate",
     "name_rx": r"robert\s+louis\s+stevenson|\br\.?\s*l\.?\s*stevenson\b",
     "slug_rx": r"wiki/robert_louis_stevenson"},
    {"key": "terminal", "name": "Mount Vaea", "role": "terminal",
     "name_rx": r"\bvaea\b",
     "slug_rx": r"wiki/mount_vaea"},
]
START, ENGINEER, NOVELIST, TERMINAL = CHAIN

# ── keystone: Mount Vaea elevation 472 m OR 1,549 ft ──
KEYSTONE_RX = re.compile(r"\b472\b|\b1[,\s]?549\b", re.IGNORECASE)
_ENGINEER_RX = re.compile(ENGINEER["name_rx"], re.IGNORECASE)
_NOVELIST_RX = re.compile(NOVELIST["name_rx"], re.IGNORECASE)
_TERMINAL_RX = re.compile(TERMINAL["name_rx"], re.IGNORECASE)

# The chain's golden path is exactly four page reads; the efficiency diagnostic scores against it.
IDEAL_HOPS = len(CHAIN)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "159",
        "test_name": "Tier 5: Narrow-sequential 4-hop chain (Bell Rock Lighthouse -> Mount Vaea elevation)",
        "difficulty_level": "9/10",
        "category": "narrow_sequential_chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ each page (do not guess from "
        "memory). This is a strictly SEQUENTIAL chain: each hop's target is only knowable after you "
        "have read the previous hop's page, so do not try to jump ahead or search for the final "
        "answer directly — you cannot know what to search for until hop 3 is resolved.\n\n"
        "HOP 1 — Open the page for the Bell Rock Lighthouse (off the coast of Angus, Scotland) and "
        "read WHO the engineer was that directed its construction on site and afterwards published "
        "the account of the works. (If the page also credits a consulting engineer, take the one who "
        "directed the works.)\n\n"
        "HOP 2 — Open THAT engineer's own page. He founded a dynasty of Scottish engineers, and one "
        "of his GRANDSONS broke with the family profession and became a world-famous novelist. "
        "Identify that grandson.\n\n"
        "HOP 3 — Open THAT novelist's page and find his RESTING PLACE: he is buried on the summit of "
        "a mountain on the Pacific island where he spent his final years. Identify THAT mountain. Do "
        "NOT report the island nation's highest peak — report the mountain where he is actually "
        "buried, as stated on his own page.\n\n"
        "HOP 4 — Open THAT mountain's page and read its ELEVATION (in metres or feet) directly from "
        "it.\n\n"
        "Report: (a) the mountain's ELEVATION (this single figure is the keystone answer); (b) the "
        "engineer, the novelist and the mountain you resolved along the way; citing the exact "
        "Wikipedia URL of every page you read. Do NOT report the lighthouse's own tower height."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The terminal mountain's elevation (472 m / 1,549 ft) — the leak-resistant keystone",
        "Hop 1: the Bell Rock Lighthouse's directing engineer",
        "Hop 2: that engineer's novelist grandson",
        "Hop 3: the mountain on which the novelist is buried",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reads the Bell Rock Lighthouse page to identify the directing engineer",
        "Reads the engineer's page to identify the novelist grandson",
        "Reads the novelist's page to identify the burial mountain (not the island's highest peak)",
        "Reports the mountain's elevation (472 m / 1,549 ft)",
        "Does NOT stop early (lighthouse tower height 36 m) and does NOT wrong-branch (Mount Silisili 1,858 m)",
        "Cites the pages actually read",
        "Resolves the chain with few off-path page visits (a chain of width 1 rewards no fan-out)",
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


def _visit_count(observability: Dict[str, Any] = None) -> int:
    obs = observability or {}
    n = int((obs.get("visit") or {}).get("count", 0) or 0)
    if n:
        return n
    # Fall back to distinct evidence pages when a variant reports evidence but no visit counter.
    visited = ((obs.get("evidence") or {}).get("visited") or [])
    return len({str((e or {}).get("url") or "") for e in visited if isinstance(e, dict)} - {""})


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Keystone credit requires GROUNDING: the value string alone is insufficient — the agent must
    have actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess would earn credit."""
    if _visit_count(observability) <= 0:
        return False
    return bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = _visit_count(observability)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / float(IDEAL_HOPS)),
            "reason": f"{n} visit(s) (4-hop chain needs 4 page reads; >=3 to pass)"}


def validate_keystone_elevation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the terminal mountain's elevation (472 m / 1,549 ft), reachable ONLY by
    resolving all three prior hops in order. Rejects the stop-early lighthouse height (36 m / 118 ft)
    and the wrong-branch highest-peak answer (Mount Silisili, 1,858 m). Leak-resistant, page-only."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_elevation", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Terminal mountain elevation 472 m / 1,549 ft present" if passed
                      else "Keystone (472 m / 1,549 ft, the burial mountain) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated depth diagnostic (the sequential analogue of a breadth task's coverage check): how
    many of the four chain waypoints the agent both NAMED in its own answer AND has PER-WAYPOINT
    visited-page evidence for. Deliberately NOT short-circuited on the keystone — it measures how
    far down the one true path the agent actually got, which is the axis that separates a chain-
    following agent from one that stalled at hop 1, even when the final figure is botched."""
    return waypoint_chain_coverage(CHAIN, result, observability, _all_text(result))


def validate_path_efficiency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated COST diagnostic — the reason this task exists as the pair to the wide-fan-out task.

    The chain has width 1, so 4 page reads suffice; every additional visit is speculative work that
    a correct linear agent never performs. Score = IDEAL_HOPS / max(visits, IDEAL_HOPS), so a
    golden-path run scores 1.0 and a breadth-oriented engine that fans out into pages which cannot
    be on the path decays smoothly. Deliberately generous (a perfect-but-wasteful run still clears
    the 0.75 bar on the mean) — it prices needless breadth overhead rather than failing it, and is
    never gated on the keystone so the cost signal survives a botched final answer."""
    n = _visit_count(observability)
    if n <= 0:
        return {"check": "path_efficiency", "passed": False, "score": 0.0,
                "reason": "no page visits — nothing traversed"}
    score = IDEAL_HOPS / float(max(n, IDEAL_HOPS))
    return {"check": "path_efficiency", "passed": n <= IDEAL_HOPS + 2, "score": score,
            "reason": f"{n} visit(s) for a width-1 chain of {IDEAL_HOPS} hops "
                      f"(efficiency {score:.2f}; <= {IDEAL_HOPS + 2} to pass)"}


def validate_hop_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the answer carries all three resolved intermediates (engineer, novelist,
    terminal mountain). Short-circuits to 0 when the keystone is absent (bimodal, never a constant
    partial-credit trap)."""
    if not _keystone_ok(result, observability):
        return {"check": "hop_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> hop resolution not credited"}
    text = _all_text(result)
    has_engineer = bool(_ENGINEER_RX.search(text))
    has_novelist = bool(_NOVELIST_RX.search(text))
    has_terminal = bool(_TERMINAL_RX.search(text))
    hits = int(has_engineer) + int(has_novelist) + int(has_terminal)
    return {"check": "hop_resolution", "passed": hits == 3, "score": hits / 3.0,
            "reason": f"engineer={has_engineer}, novelist={has_novelist}, terminal={has_terminal}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for w in CHAIN if re.search(w["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 3, "score": min(1.0, cited / float(len(CHAIN))),
            "reason": f"{cited}/{len(CHAIN)} chain page(s) cited (need >=3)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_elevation,
        validate_chain_coverage,
        validate_path_efficiency,
        validate_hop_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored NARROW-SEQUENTIAL scaffold: 4 leaves, 3 edges, wave widths [1,1,1,1].

    This is the structural counterpart to the fan-out plans: there is nothing to parallelise, so
    the compiled plan is a pure chain in which each leaf templates its predecessor's finding
    ({engineer}, {novelist}, {resting_place}). It encodes only STRUCTURE — it names the GIVEN
    starting lighthouse and the termination condition, and leaks no engineer, no novelist, no
    mountain and no elevation figure."""
    engineer_leaf = {
        "id": "engineer",
        "instruction": (
            "Open the Wikipedia page for the Bell Rock Lighthouse (off Angus, Scotland). Read WHO "
            "the engineer was that directed its construction on site and afterwards published the "
            "account of the works; if a consulting engineer is also credited, take the one who "
            "directed the works. Report that engineer and the exact Wikipedia URL. Do not guess "
            "from memory; do not report any other fact."
        ),
        "expect": "The directing engineer of the Bell Rock Lighthouse — source URL",
        "depends_on": [],
    }
    novelist_leaf = {
        "id": "novelist",
        "instruction": (
            "Open the Wikipedia page of the engineer identified in the previous step ({engineer}). "
            "He founded a dynasty of Scottish engineers; one of his GRANDSONS broke with the family "
            "profession and became a world-famous novelist. Identify THAT grandson and report his "
            "exact Wikipedia URL. Do not guess from memory; report no other fact."
        ),
        "expect": "The engineer's novelist grandson — source URL",
        "depends_on": ["engineer"],
    }
    resting_leaf = {
        "id": "resting_place",
        "instruction": (
            "Open the Wikipedia page of the novelist identified in the previous step ({novelist}). "
            "Find his RESTING PLACE: he is buried on the summit of a mountain on the Pacific island "
            "where he spent his final years. Report WHICH mountain, exactly as named on his page "
            "(NOT the island nation's highest peak), plus that mountain's Wikipedia URL. Do not "
            "guess from memory."
        ),
        "expect": "The mountain named as the novelist's resting place — source URL",
        "depends_on": ["novelist"],
    }
    elevation_leaf = {
        "id": "elevation",
        "instruction": (
            "Open the Wikipedia page of the mountain identified in the previous step "
            "({resting_place}). Read its ELEVATION (in metres or feet) directly from the page. "
            "Report that elevation and the source URL. Do not guess from memory."
        ),
        "expect": "The terminal mountain's elevation — source URL",
        "depends_on": ["resting_place"],
    }
    return {
        "leaves": [engineer_leaf, novelist_leaf, resting_leaf, elevation_leaf],
        "aggregation": (
            "You now have (1) the Bell Rock Lighthouse's directing engineer, (2) his novelist "
            "grandson, (3) the mountain where that novelist is buried, and (4) that mountain's "
            "elevation. Report (a) the mountain's ELEVATION — this single figure is the keystone "
            "answer; (b) the engineer, the novelist and the mountain, citing every source URL. Do "
            "NOT report the lighthouse's own tower height, and do NOT report the island nation's "
            "highest peak."
        ),
    }
