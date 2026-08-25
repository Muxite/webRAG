r"""
Test 303: Mechanism suite — ENTITY COLLISION (near-duplicate names on one page).
Level: navigation   Weight: medium   Difficulty: 9/10   Category: entity_collision

Mechanism under test: a single authoritative page NAMES two near-identically-titled entities but
carries the figures of only ONE of them. An agent that does loose substring / fuzzy entity matching
("the page says 'Tay Rail Bridge' and has a length, so that length is the answer") attributes the
WRONG entity's number to the target. A correct agent must do identity-FIRST matching: pin the entity
by its distinguishing attributes (carries rail not road, opened year, engineer, span count) before
lifting any figure off a page.

The collision is threefold and entirely real:

    TARGET   "Tay Bridge" (a.k.a. Tay Rail Bridge) — the PRESENT, second structure across the Firth
             of Tay between Dundee and Wormit. Rail. Opened 20 June 1887. Engineered by William
             Henry Barlow (Barlow & Sons); built by William Arrol & Co. 85 spans.
             Total length = 10,711 ft (2.0286 mi; 3,265 m)   <-- KEYSTONE
             Body text of the same article also gives 10,780 ft for the second bridge; both readings
             are the TARGET entity and both are accepted.

    DECOY 1  "Tay Road Bridge" — the near-duplicate name (the target's name is a strict SUBSTRING of
             it). Carries the A92, opened 18 August 1966, designer William A Fairhurst, 42 spans.
             Total length = 2,250 m (1.4 mi). Its Wikipedia lead literally contains the string
             "Tay Rail Bridge" ("just downstream of the Tay Rail Bridge") while carrying only the
             ROAD bridge's length — this is the exact page that punishes substring matching.

    DECOY 2  The FIRST Tay Bridge — same name as the target, different entity: Thomas Bouch's
             lattice bridge, opened 1 June 1878, collapsed 28 December 1879. Shares the article
             with the target, so an agent that stops at "designer of the Tay Bridge" reports Bouch.

Keystone margin: 3,265 m / 10,711 ft (target) vs 2,250 m / ~7,382 ft (decoy) — 45% apart, fully
disjoint digit strings. No single noisy extraction can flip the gate.

Ground truth (verified against live English Wikipedia raw wikitext + API, 2026-08-25):
  https://en.wikipedia.org/wiki/Tay_Bridge
      infobox  | length = {{Convert|10711|ft|mi m}}   -> "10,711 feet (2.0286 mi; 3,265 m)"
      infobox  | opened = 1 June 1878 (1st) / 20 June 1887 (2nd); closed = 28 December 1879 (1st)
      body     | "It has an overall length of 10,780 ft, which is covered by a total of 85 spans."
      body     | William Henry Barlow, of Barlow & Sons, designed the second bridge;
                 "a contract for the new bridge's construction was awarded to Messrs William Arrol & Co"
      body     | "The bridge was designed by engineer Thomas Bouch" (FIRST bridge)
      hatnote  | "This article is about the rail bridge. For the road bridge, see Tay Road Bridge."
  https://en.wikipedia.org/wiki/Tay_Road_Bridge
      infobox  | length = {{Convert|2250|m|mi|1}}; opened = 18 August 1966; designer = William A Fairhurst
      lead     | "...from Newport-on-Tay in Fife to Dundee in Scotland, just downstream of the
                 [[Tay Rail Bridge]]. At around 2,250 m..."
      body     | "The bridge consists of 42 spans..."
  API: "Tay Rail Bridge" -> redirect -> "Tay Bridge" (pageid 144681).

Why leak-resistant: 10,711 ft / 3,265 m and the 85-span count are page-only Victorian engineering
figures no consumer LLM recalls; the famous parametric memory attached to "Tay Bridge" is the 1879
DISASTER (Bouch, the first bridge), which is DECOY 2. Reciting what the model "knows" therefore lands
on a decoy, not on the keystone. Keystone credit additionally requires visit.count > 0.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# KEYSTONE: the present RAIL bridge's total length, any published rendering.
#   10,711 ft (infobox/lead) | 10,780 ft (body) | 3,265 / 3,264 m | 3,285 m (=10,780 ft) | ~2.03 mi
KEYSTONE_RX = re.compile(
    r"\b10[,.\s]?7(?:11|80)\b|\b3[,.\s]?2(?:64|65|85)\b|\b2\.0[23]\d*\s*(?:mi\b|miles)",
    re.IGNORECASE,
)
# DECOY 1 length (Tay ROAD Bridge): 2,250 m / ~7,382 ft / 1.4 mi.
DECOY_LEN_RX = re.compile(r"\b2[,.\s]?250\b|\b7[,.\s]?38\d\b|\b1\.4\s*(?:mi\b|miles)", re.IGNORECASE)
# DECOY 1 identity facts, which must never be attributed to the target.
DECOY_ID_RX = re.compile(r"\b1966\b|fairhurst|\b42\s+spans\b|\bA92\b", re.IGNORECASE)
# Tokens that mean "this clause is talking about the TARGET (rail) bridge".
TARGET_TOKEN_RX = re.compile(r"tay\s+bridge|rail(?:way)?\s+bridge|railway\s+bridge", re.IGNORECASE)
# Tokens that mean "this clause is properly labelled as the ROAD bridge" (exempts the clause).
ROAD_TOKEN_RX = re.compile(r"\broad\b|\bA92\b|fairhurst", re.IGNORECASE)

# Identity/coverage markers that can only come from READING the target article.
OPENED_RX = re.compile(r"\b1887\b", re.IGNORECASE)
ENGINEER_RX = re.compile(r"barlow|arrol", re.IGNORECASE)
SPANS_RX = re.compile(
    r"\b85\b\s*(?:spans?\b|-)|spans?\s*[:=]?\s*\b85\b|\beighty[- ]five\s+spans?",
    re.IGNORECASE,
)
FIRST_BRIDGE_RX = re.compile(r"bouch|\b1878\b", re.IGNORECASE)

TARGET_SLUG_RX = re.compile(r"wiki/tay[_ ](?:rail[_ ])?bridge\b", re.IGNORECASE)
ROAD_EXCLUSION_RX = re.compile(r"tay\s+road\s+bridge|road\s+bridge", re.IGNORECASE)

_CHUNK_SPLIT_RX = re.compile(r"[.;!?\n]")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "303",
        "test_name": "Mechanism: entity collision — Tay Bridge (rail) length vs the near-identically-named Tay Road Bridge",
        "difficulty_level": "9/10",
        "category": "entity_collision",
        "level": "navigation",
        "weight": "medium",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not answer from "
        "memory). This task is about ENTITY IDENTITY, not breadth: the figure you need exists on "
        "exactly one page, and a near-identically-named structure sits a few hundred metres away "
        "with a completely different figure.\n\n"
        "Question: what is the TOTAL LENGTH of the bridge that carries RAIL traffic across the Firth "
        "of Tay between Dundee and Wormit — specifically the PRESENT structure, i.e. the second "
        "bridge on the site, the replacement built after the original bridge collapsed in a gale in "
        "1879?\n\n"
        "Two collisions will try to make you report the wrong number:\n"
        "  1. The 'Tay Road Bridge' is a DIFFERENT bridge (it carries a road, not trains). Its own "
        "page mentions the rail bridge by name while giving only the ROAD bridge's length — do not "
        "lift a length off a page merely because the rail bridge is named somewhere on it.\n"
        "  2. TWO different bridges have carried the name 'Tay Bridge': the original one that fell in "
        "1879 and the present replacement. You want the PRESENT one.\n\n"
        "Report: (a) the total length of the present rail bridge — this figure is the keystone "
        "answer; (b) the identity evidence that proves you pinned the right structure: the year the "
        "present bridge opened, the engineer/contractor responsible for the present bridge, how many "
        "spans it has, and who engineered the earlier bridge that collapsed; (c) an explicit "
        "statement that your figure is NOT the Tay Road Bridge's; citing the exact Wikipedia URL of "
        "the page that carries the length."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Total length of the PRESENT Tay rail bridge (the leak-resistant keystone; NOT the Tay Road Bridge's length)",
        "Identity evidence for the present structure: opening year, engineer/contractor, number of spans",
        "Identification of the earlier same-named bridge (the one that collapsed) and its engineer",
        "An explicit statement that the reported length is not the Tay Road Bridge's",
        "The exact Wikipedia URL of the page that carries the length",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages read (the collided pair must be told apart, not guessed)",
        "Reports the present rail bridge's total length (10,711 ft / 3,265 m, or the article's 10,780 ft reading)",
        "Never attributes the Tay Road Bridge's 2,250 m / 1.4 mi to the rail bridge",
        "Never attributes the road bridge's identity facts (1966, Fairhurst, 42 spans, A92) to the rail bridge",
        "Supplies identity evidence read off the page: 1887 opening, Barlow/Arrol, 85 spans, Bouch/1878 first bridge",
        "Cites the Tay Bridge (rail) Wikipedia page, not the Tay Road Bridge page",
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


def _misattributed_clauses(text: str, value_rx: re.Pattern) -> List[str]:
    """Clauses that pin a DECOY value onto the TARGET entity.

    A clause is a misattribution when it (a) carries a decoy figure/fact, (b) names the target
    (rail) bridge, and (c) is NOT labelled as the road bridge. Clause boundaries are sentence
    punctuation OR a newline, so a bullet/table layout is judged line-by-line and a properly
    labelled 'Tay Road Bridge: 2,250 m' row never fires."""
    out = []
    for chunk in _CHUNK_SPLIT_RX.split(text or ""):
        if not value_rx.search(chunk):
            continue
        if ROAD_TOKEN_RX.search(chunk):
            continue
        if TARGET_TOKEN_RX.search(chunk):
            out.append(chunk.strip())
    return out


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Keystone credit requires (1) GROUNDING — at least one real page visit, else a parametric
    guess would score; (2) the target length present in the headline answer; (3) NO clause anywhere
    in the report that pins the road bridge's length onto the rail bridge. (3) is the actual defect
    this task exists to catch: reporting both numbers is fine, swapping them is not."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    if not KEYSTONE_RX.search(_primary_text(result)):
        return False
    return not _misattributed_clauses(_all_text(result), DECOY_LEN_RX)


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: telling two collided entities apart takes reading both of them."""
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: the target page and the near-duplicate it collides with)"}


def validate_keystone_length(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the PRESENT rail bridge's total length, with the wrong-entity
    attribution explicitly rejected. Fails to 0 if the Tay Road Bridge's 2,250 m / 1.4 mi is
    presented as the rail bridge's length, even when the right number also appears somewhere."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    bad = _misattributed_clauses(_all_text(result), DECOY_LEN_RX)
    passed = _keystone_ok(result, observability)
    if passed:
        reason = "rail-bridge length (10,711 ft / 3,265 m, or 10,780 ft) reported for the correct entity"
    elif n_visits <= 0:
        reason = "no page visited -> ungrounded, keystone not credited"
    elif bad:
        reason = f"ENTITY COLLISION: road-bridge length attributed to the rail bridge -> {bad[0][:90]!r}"
    else:
        reason = "Keystone length of the present rail bridge missing/incorrect"
    return {"check": "keystone_rail_bridge_length", "passed": passed,
            "score": 1.0 if passed else 0.0, "reason": reason,
            "misattributed": bad}


def validate_identity_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how much identity evidence was actually GATHERED —
    4 page-only markers that pin the entity (1887 opening, Barlow/Arrol, 85 spans, Bouch/1878 for
    the earlier same-named bridge). Credit is CAPPED BY the visit count so a narrated answer with no
    reads cannot bank it. Deliberately NOT short-circuited on the keystone: an agent that gathered
    the identity evidence but botched the final number still scores its gathering here — this is the
    axis that separates a structured agent from a linear one."""
    text = _all_text(result)
    markers = [
        ("opened_1887", OPENED_RX),
        ("engineer_barlow_arrol", ENGINEER_RX),
        ("span_count", SPANS_RX),
        ("first_bridge_bouch_1878", FIRST_BRIDGE_RX),
    ]
    found = [name for name, rx in markers if rx.search(text)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(found), max(0, n_visits) * 2)
    credited = min(credited, len(markers))
    return {"check": "identity_coverage", "passed": credited == len(markers),
            "score": credited / len(markers),
            "reason": f"{credited}/{len(markers)} identity markers credited "
                      f"(found {found}, {n_visits} visit(s))",
            "found": found}


def validate_collision_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: was the collision actually RESOLVED rather than dodged? Two halves —
    (a) the near-duplicate road bridge is named as a distinct structure, (b) none of the road
    bridge's identity facts (1966 / Fairhurst / 42 spans / A92) are pinned on the rail bridge.
    Short-circuits to 0 when the keystone is absent (keeps scores bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "collision_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> collision resolution not credited"}
    text = _all_text(result)
    named = bool(ROAD_EXCLUSION_RX.search(text))
    bad_id = _misattributed_clauses(text, DECOY_ID_RX)
    clean = not bad_id
    score = (0.5 if named else 0.0) + (0.5 if clean else 0.0)
    return {"check": "collision_resolution", "passed": score == 1.0, "score": score,
            "reason": f"road bridge named as distinct={named}; no decoy identity facts "
                      f"attributed to the rail bridge={clean}"
                      + (f" (offending: {bad_id[0][:80]!r})" if bad_id else "")}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the TARGET page (Tay_Bridge / Tay_Rail_Bridge redirect), not the
    collided Tay_Road_Bridge page. Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URL not credited"}
    passed = bool(TARGET_SLUG_RX.search(_all_text(result)))
    return {"check": "citations", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"cites the Tay Bridge (rail) page={passed}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_length,
        validate_identity_coverage,
        validate_collision_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored identity-FIRST DAG for the graph_compiled variant.

      * leaf1 (id keyed on the GIVEN entity description, never on the answer) — resolve WHICH Firth
        of Tay bridge is the present rail structure and get its URL. Nothing else may proceed until
        the entity is pinned.
      * leaf2/3/4 — depend_on leaf1 and template {tay_rail_bridge}: one atomic fact each, all read
        off that one resolved page (length, opening year, engineer + span count are split so no leaf
        has to hold two facts at once).

    STRUCTURE only: restates the GIVEN cues (rail vs road, Dundee/Wormit, the earlier collapse) and
    leaks NO length, year, name or span count."""
    identity_leaf = {
        "id": "tay_rail_bridge",
        "instruction": (
            "Two similarly named bridges cross the Firth of Tay at Dundee: one carries RAIL traffic "
            "(between Dundee and Wormit) and one carries a ROAD (the A92). Identify the Wikipedia "
            "article for the RAIL bridge — the present structure, i.e. the replacement built after "
            "the original bridge of that name collapsed in a gale. Beware: the road bridge's own "
            "article names the rail bridge in its opening sentence; that does not make it the rail "
            "bridge's article. Report which page is the rail bridge and its exact Wikipedia URL. "
            "Do not report any measurement, date or engineer yet."
        ),
        "expect": "The Wikipedia URL of the present rail bridge over the Firth of Tay",
        "depends_on": [],
    }
    length_leaf = {
        "id": "bridge_length",
        "instruction": (
            "Open the page identified in the previous step ({tay_rail_bridge}) — the RAIL bridge over "
            "the Firth of Tay. Read its TOTAL LENGTH directly from that page (infobox and/or lead "
            "sentence). Report the length with its units and the source URL. Do not report the "
            "length of the road bridge, and do not answer from memory."
        ),
        "expect": "The total length of the rail bridge, with source URL",
        "depends_on": ["tay_rail_bridge"],
    }
    opened_leaf = {
        "id": "bridge_opened",
        "instruction": (
            "Open the page identified earlier ({tay_rail_bridge}). Two different bridges have carried "
            "this name on this site. Read the year the PRESENT (second) bridge opened — not the year "
            "the original one opened or the year it collapsed. Report that single year and the "
            "source URL."
        ),
        "expect": "The opening year of the present rail bridge",
        "depends_on": ["tay_rail_bridge"],
    }
    builder_leaf = {
        "id": "bridge_builder",
        "instruction": (
            "Open the page identified earlier ({tay_rail_bridge}). Read (i) who engineered/built the "
            "PRESENT bridge, (ii) how many spans the present bridge has, and (iii) who engineered "
            "the EARLIER bridge of the same name that collapsed. Report all three with the source "
            "URL. Do not confuse the two engineers."
        ),
        "expect": "Engineer/contractor and span count of the present bridge, plus the earlier bridge's engineer",
        "depends_on": ["tay_rail_bridge"],
    }
    return {
        "leaves": [identity_leaf, length_leaf, opened_leaf, builder_leaf],
        "aggregation": (
            "You now have (1) which Firth of Tay bridge is the present RAIL bridge, (2) its total "
            "length, (3) its opening year, and (4) its engineer/contractor, span count and the "
            "earlier same-named bridge's engineer. Report (a) the total length of the present rail "
            "bridge — this figure is the keystone answer; (b) the identity evidence (opening year, "
            "engineer/contractor, span count, the earlier bridge's engineer); (c) an explicit "
            "statement that the figure is NOT the road bridge's; citing the source URL of the page "
            "that carries the length. Never present the road bridge's length or its opening "
            "year/designer as the rail bridge's."
        ),
    }
