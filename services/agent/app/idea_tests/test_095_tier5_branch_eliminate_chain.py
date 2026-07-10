"""
Test 095: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 10/10

DELIBERATELY unlike every other task in the suite: it structurally requires a MULTI-ROUND,
BRANCHING graph-of-thoughts exploration — NOT a flat single-round fan-out (052/053/090/091)
and NOT a single linear chain (065/092). A shallow "one round of parallel branches, pick the
best, done" strategy cannot solve it, because the entity to investigate in round 2 is unknown
until round 1's elimination resolves which candidate survives.

    ROUND 1  (breadth / ambiguity — 4 genuine candidates, eliminate to ONE)
      Britain has four principal Rivers Avon (Bristol, Warwickshire/Shakespeare's,
      Hampshire/Salisbury, Scottish/Strathspey). They are a REAL ambiguity — a model asked
      for "a River Avon in Britain" reaches first for the famous Warwickshire (Shakespeare's)
      or Bristol (Bath) Avon. The disambiguator is PAGE-ONLY: exactly ONE of the four empties
      into the ENGLISH CHANNEL. That fact lives only in each river's own infobox 'mouth'
      field — it is not inferable from the river's name, and resolving it requires opening and
      reading each of the four candidate pages (real work per branch, not a free filter).

    ROUND 2  (forward chain from the SURVIVOR — with an embedded 2nd disambiguation)
      The survivor's OWN page reveals the next target, unknowable until round 1 resolves it:
      near its mouth the surviving Avon merges with a river called the 'Stour'. But there are
      FIVE Rivers Stour in England (Dorset, Kent, Suffolk/Essex, Warwickshire, Worcestershire),
      so the agent must disambiguate AGAIN to the specific one that joins the surviving Avon.

    ROUND 3  (keystone — obscure, page-only)
      Read that Stour's CATCHMENT / BASIN area from its infobox: the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-09):

  ROUND 1 candidates — infobox 'mouth' (English Channel?):
  ┌──────────────────────────────────────────────┬──────────────────────────────┬────────────┐
  │ River Avon, Bristol                           │ Severn Estuary (Avonmouth)   │ eliminated │
  │ River Avon, Warwickshire (Shakespeare's)      │ River Severn (Tewkesbury)    │ eliminated │
  │ River Avon, Hampshire (Salisbury)  ← SURVIVOR │ ENGLISH CHANNEL (Christchurch)│ SURVIVES  │
  │ River Avon, Strathspey (Scottish)             │ River Spey (Ballindalloch)   │ eliminated │
  └──────────────────────────────────────────────┴──────────────────────────────┴────────────┘
      Only the Hampshire (Salisbury) Avon reaches the English Channel — the other three empty
      into the Severn, the Severn Estuary, or the Spey. None of the eliminated pages mention the
      English Channel at all, so the elimination is categorical (a unique string match), not a
      numeric margin: one noisy mouth-read cannot flip it.

  ROUND 2 forward chain:
      Hampshire (Salisbury) Avon  →  merges with the River Stour at Christchurch Harbour.
      Among the FIVE English Rivers Stour, the one that joins it there is the DORSET Stour
      (River_Stour,_Dorset) — cross-confirmed on both pages ("...meeting the River Stour and
      flowing into Christchurch Harbour" / the Stour "is joined by the River Avon" at Christchurch).

  ROUND 3 keystone:
      River Stour, Dorset — infobox catchment / basin size = 1,240 km² (480 sq mi).  [KEYSTONE]

Why the keystone is leak-resistant: the Dorset Stour's catchment area (1,240 km²) is a small,
obscure infobox figure no consumer LLM recalls parametrically; even knowing the river, a model
would guess. Its token \b1,?240\b is a distinctive 4-digit value that collides with NONE of the
lengths on any page in the chain (Avon lengths 137 / 134 / 96 / 63.9 km; Dorset Stour length
98 km / 61 mi), so a single noisy extraction cannot satisfy it — only reading the correct
downstream page can, which is exactly the depth the branch-then-chain forces.

Anti-parametric / anti-shallow mechanism:
  * A flat "one parallel burst, pick the best" agent fails: the Dorset Stour is unknowable until
    round 1 elects the Hampshire Avon AND round 2 reads that it merges with a Stour.
  * A memory-anchored agent reaches for Shakespeare's (Warwickshire) or Bristol's Avon — the two
    famous decoys — which do NOT reach the English Channel, breaking the chain (the Warwickshire
    Avon joins the Severn, not any Stour), so it never reaches the keystone.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── ROUND-1 candidate set (single source of truth for statement, validators and the plan) ──
# Four genuine Rivers Avon. `mouth_rx` is the PAGE-ONLY disambiguating fact (infobox 'mouth');
# only the survivor's mouth is the English Channel. `name_rx` is that candidate's distinctive
# name token (the four are pairwise distinct, so coverage never cross-credits).
AVONS: List[Dict[str, Any]] = [
    {
        "key": "bristol", "name": "River Avon, Bristol",
        "desc": "the Bristol Avon (flows through Bath and Bristol)",
        "name_rx": r"bristol", "mouth_rx": r"severn\s+estuary|bristol\s+channel|avonmouth",
        "slug_rx": r"wiki/river_avon,_bristol", "survivor": False,
    },
    {
        "key": "warwickshire", "name": "River Avon, Warwickshire",
        "desc": "the Warwickshire or 'Shakespeare's' Avon (flows through Stratford-upon-Avon)",
        # mouth_rx keys on the UNIQUE confluence town (Tewkesbury), not the bare "Severn", so it
        # never cross-credits with the Bristol Avon's "Severn Estuary" mouth.
        "name_rx": r"warwickshire|shakespeare", "mouth_rx": r"tewkesbury",
        "slug_rx": r"wiki/river_avon,_warwickshire", "survivor": False,
    },
    {
        "key": "hampshire", "name": "River Avon, Hampshire",
        "desc": "the Hampshire or 'Salisbury' Avon (flows through Salisbury)",
        "name_rx": r"hampshire|salisbury", "mouth_rx": r"english\s+channel|christchurch",
        "slug_rx": r"wiki/river_avon,_hampshire", "survivor": True,
    },
    {
        "key": "scottish", "name": "River Avon, Strathspey",
        "desc": "the Scottish (Strathspey / Banffshire) Avon (rises in the Cairngorms, in Moray)",
        "name_rx": r"strathspey|banffshire|scottish\s+avon", "mouth_rx": r"\bspey\b|ballindalloch",
        "slug_rx": r"wiki/river_avon,_strathspey", "survivor": False,
    },
]
SURVIVOR = next(a for a in AVONS if a["survivor"])  # River Avon, Hampshire (Salisbury)

# ── ROUND-3 keystone: the Dorset Stour's infobox catchment / basin area ──
# 1,240 km² (equivalently 480 sq mi). \b1,?240\b matches "1,240" or "1240" but NOT "21,240"
# (no leading boundary) nor "1,2400" (no trailing boundary); the sq-mile alternative is guarded
# by an adjacent 'sq'/'square' so a bare "480" (e.g. an elevation) cannot satisfy it.
KEYSTONE_RX = re.compile(r"\b1,?240\b|\b480\s*(?:sq|square)", re.IGNORECASE)
DORSET_STOUR_SLUG = r"wiki/river_stour,_dorset"
_DORSET_RX = re.compile(r"\bdorset\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "095",
        "test_name": "Tier 5: Branch-to-eliminate then chain forward (Rivers Avon -> Dorset Stour catchment)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {a['name']} — {a['desc']}" for i, a in enumerate(AVONS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has three stages; each stage's target is unknown until the "
        "previous stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Britain has four principal rivers named 'Avon':\n"
        f"{listing}\n"
        "Exactly ONE of these four empties into the ENGLISH CHANNEL. Open EACH river's page and "
        "read its mouth/outflow from the infobox to determine which one — the other three empty "
        "into the River Severn, the Severn Estuary, or the River Spey. Determine the status of "
        "all four; do not simply guess the most famous one.\n\n"
        "STAGE 2 — follow the survivor forward. Open the surviving river's page. Near its mouth it "
        "merges with a river called the 'Stour'. There are FIVE different Rivers Stour in England "
        "(Dorset, Kent, Suffolk, Warwickshire, Worcestershire), so identify the SPECIFIC one that "
        "joins your surviving Avon at its mouth.\n\n"
        "STAGE 3 — read the keystone. Open that River Stour's page and read its CATCHMENT (basin) "
        "area in SQUARE KILOMETRES directly from the infobox.\n\n"
        "Report: (a) the Stour's CATCHMENT area in km² (this single figure is the keystone answer); "
        "(b) which of the four Rivers Avon was the survivor and the body of water each of the four "
        "empties into; (c) which River Stour it was; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The surviving Stour's catchment/basin area in km² (the leak-resistant keystone)",
        "Which River Avon empties into the English Channel (the survivor) + each of the four mouths",
        "Which River Stour joins the survivor (the Dorset Stour)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per Avon candidate) plus the survivor's Stour",
        "Determines the mouth/status of ALL FOUR Rivers Avon (branch-to-eliminate)",
        "Correctly elects the Hampshire (Salisbury) Avon as the English-Channel survivor",
        "Correctly identifies the Dorset Stour (not one of the other four Rivers Stour)",
        "Reports the Dorset Stour's catchment area (1,240 km² / 480 sq mi)",
        "Cites the survivor Avon page and the Dorset Stour page",
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
    """Full reported text: final deliverable concatenated with every deliverable slot, so the
    breadth and citation checks see facts placed outside the primary slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate: the Dorset Stour's catchment area (1,240 km² / 480 sq mi) in the primary text."""
    return bool(KEYSTONE_RX.search(_primary_text(result)))


# ── validation functions ─────────────────────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: the branch-to-eliminate needs the four Avon pages, and the forward
    chain needs the Dorset Stour page — at least five distinct reads for a full solve."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: four Avon candidates + the survivor's Stour; >=4 to pass)"}


def validate_keystone_catchment(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the Dorset Stour's infobox catchment area (1,240 km²). Leak-resistant.

    Reachable only by walking the full branch-then-chain: elect the Hampshire Avon (English-Channel
    survivor), read that it merges with the Stour, disambiguate to the Dorset Stour, read its
    catchment. A memory guess, a wrong survivor (famous Warwickshire/Bristol decoy), or the wrong
    Stour cannot produce 1,240 km²."""
    passed = _keystone_ok(result)
    return {"check": "keystone_catchment", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Dorset Stour catchment 1,240 km² (480 sq mi) present" if passed
                      else "Keystone catchment (1,240 km² / 480 sq mi, River Stour, Dorset) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR Rivers Avon the agent actually resolved —
    i.e. reported both the candidate's name and the mouth/outflow it empties into.

    This is the axis that separates a structured multi-round agent (which branches to eliminate ALL
    four candidates before selecting) from a shallow/linear one that guesses the famous Avon and
    never checks the others. Deliberately NOT short-circuited on the keystone: it credits the
    breadth of the elimination even when the downstream keystone is botched. A mouth is credited
    only with its candidate's own distinct name token, so there is no cross-crediting between the
    two Severn-mouthed Avons (Bristol vs Warwickshire).

    Execution cross-check: each candidate's mouth is a PAGE-ONLY infobox fact, so a text match
    alone is not proof — a weak model can narrate the four mouths from the mandate/memory with ZERO
    page visits and would otherwise bank breadth it never earned. We AND the text match with actual
    visit evidence: observability exposes only a visit COUNT (no per-URL detail), so we cap the
    credited breadth at the number of visits. A run that reads no pages scores 0 no matter what it
    writes, closing the same text-presence-as-proof loophole the candidate_coverage gate fixed."""
    text = _all_text(result)
    text_hits = [a["name"] for a in AVONS
                 if re.search(a["name_rx"], text, re.IGNORECASE) and re.search(a["mouth_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)  # a resolved candidate needs a backing page visit
    n = len(AVONS)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} Rivers Avon resolved to their mouth from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; "
                      f"{len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor_and_stour(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the chain was resolved correctly — the output names the survivor (the
    Hampshire/Salisbury Avon) AND the Dorset Stour. Short-circuits to 0 when the keystone is absent,
    keeping scores bimodal so an incidental mention cannot bank partial credit."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_stour", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/Stour resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(re.search(SURVIVOR["name_rx"], text, re.IGNORECASE))
    has_stour = bool(_DORSET_RX.search(text))
    hits = int(has_survivor) + int(has_stour)
    return {"check": "survivor_and_stour", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Hampshire/Salisbury Avon)={has_survivor}, Dorset Stour={has_stour}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages the chain had to read (the survivor Avon page and the
    Dorset Stour page count double; the other Avon pages also count). Short-circuits to 0 when the
    keystone is absent so a wrong-terminus run cannot bank citation credit."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [a["slug_rx"] for a in AVONS] + [DORSET_STOUR_SLUG]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor Avon + Dorset Stour)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_catchment,
        validate_branch_exploration,
        validate_survivor_and_stour,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    # None -> harness applies its default structured rubric judge, as in the sibling graph tasks.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold for the graph_compiled variant.

    Three waves, a genuine tree (NOT a pure fan-out, NOT a pure chain):
      * WAVE 1 — four INDEPENDENT parallel leaves, one per candidate River Avon, each reading ONLY
        that river's mouth/outflow from its infobox. Ids keyed on the GIVEN candidate names.
      * WAVE 2 — ONE dependent leaf that templates all four mouths ({avon_*}), elects the
        English-Channel survivor, and reads (from that survivor's page) the SPECIFIC River Stour it
        merges with — the second disambiguation. Depends on all four wave-1 leaves.
      * WAVE 3 — ONE dependent leaf that templates the survivor's Stour ({survivor_confluence}) and
        reads that Stour's catchment area. Depends on wave 2.

    Self-describing: every leaf restates its own subject in its answer text so the aggregator binds
    the entity after leaf ids are stripped. STRUCTURE only — it names the GIVEN candidates and the
    GIVEN English-Channel criterion but leaks NO mouth, NOT which Avon survives, NOT the Dorset
    Stour, and NOT the catchment figure; the cheap runtime model still does every page-read."""
    avon_leaves = [
        {
            "id": f"avon_{a['key']}",
            "instruction": (
                f"Open the Wikipedia page for {a['name']} — {a['desc']}. Read its MOUTH / OUTFLOW "
                "(the body of water it ultimately empties into) directly from the infobox. Report "
                f"the river's name ({a['name']}), the body of water it empties into, and the exact "
                "Wikipedia URL. Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{a['name']} — empties into: BODY OF WATER — source URL",
            "depends_on": [],
        }
        for a in AVONS
    ]
    survivor_leaf = {
        "id": "survivor_confluence",
        "instruction": (
            "You are given the four Rivers Avon and the body of water each empties into:\n"
            "  River Avon, Bristol -> {avon_bristol}\n"
            "  River Avon, Warwickshire -> {avon_warwickshire}\n"
            "  River Avon, Hampshire -> {avon_hampshire}\n"
            "  River Avon, Strathspey -> {avon_scottish}\n"
            "Determine which SINGLE one empties into the ENGLISH CHANNEL. Open THAT surviving river's "
            "Wikipedia page. Near its mouth it merges with a river named the 'Stour'; there are FIVE "
            "different Rivers Stour in England, so identify the SPECIFIC one that joins this surviving "
            "Avon at its mouth. Report the surviving Avon, the specific River Stour it joins, and that "
            "Stour's exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "SURVIVING River Avon (English-Channel one) + the specific River Stour it joins — source URL",
        "depends_on": [f"avon_{a['key']}" for a in AVONS],
    }
    catchment_leaf = {
        "id": "stour_catchment",
        "instruction": (
            "Open the Wikipedia page of the specific River Stour identified in the previous step "
            "({survivor_confluence}) — the Stour that joins the surviving Avon at its mouth. Read that "
            "River Stour's CATCHMENT / BASIN area in SQUARE KILOMETRES directly from the infobox. "
            "Report the River Stour's catchment area in km² and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving River Stour's CATCHMENT area in km² — source URL",
        "depends_on": ["survivor_confluence"],
    }
    return {
        "leaves": avon_leaves + [survivor_leaf, catchment_leaf],
        "aggregation": (
            "You now have (1) the mouth of each of the four Rivers Avon, (2) which single one empties "
            "into the English Channel (the survivor) and the specific River Stour it merges with at its "
            "mouth, and (3) that Stour's catchment area. Write out all four Avon mouths BEFORE "
            "concluding which survives. Then report (a) the surviving Stour's CATCHMENT area in km² — "
            "this single figure is the keystone answer; (b) which River Avon was the survivor and the "
            "body of water each of the four empties into; (c) which River Stour it was; citing every "
            "source URL."
        ),
    }
