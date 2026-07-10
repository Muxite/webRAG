"""
Test 104: Tier 5 (graph) — BRANCH-TO-ELIMINATE (giant titanosaurs), THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 10/10

Same proven branch-then-chain shape as test_095: a genuine multi-round, BRANCHING
graph-of-thoughts task in which the round-2 target is unknown until round-1's elimination
resolves the survivor.

    ROUND 1  (breadth / ambiguity — 4 genuine giant titanosaurs, eliminate to ONE)
      Several colossal titanosaurs are routinely called the 'largest dinosaur'. A memory-anchored
      model reaches for Argentinosaurus (the fame-anchored 'largest') or Patagotitan. The
      disambiguator is PAGE-ONLY: exactly ONE is known from the MOST COMPLETE giant-sauropod
      skeleton (a large fraction of the skeleton recovered) — Dreadnoughtus. The other three are
      known from famously FRAGMENTARY remains. Resolving this requires reading each genus's
      completeness/holotype description, not equating 'biggest' with 'the answer'.

    ROUND 2  (forward chain from the SURVIVOR — a measured osteological figure)
      The survivor's OWN holotype description names a specific MEASURED bone. Read that element's
      length from the survivor's page — a figure that exists only for the correctly elected
      most-complete taxon (the famously fragmentary decoys have only ESTIMATED, not measured, bones).

    ROUND 3  (keystone — obscure, page-only)
      Read the length of the survivor's largest measured shoulder-girdle bone (the scapula): the
      keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — skeletal completeness:
  ┌───────────────────────────────────────────────┬────────────────────────────────┬────────────┐
  │ Argentinosaurus  (fame decoy, 'largest')       │ very fragmentary (few elements)│ eliminated │
  │ Patagotitan                                    │ partial, multiple individuals  │ eliminated │
  │ Puertasaurus                                   │ four vertebrae only            │ eliminated │
  │ Dreadnoughtus            ← SURVIVOR             │ ~70.4% by bone type (most      │ SURVIVES  │
  │                                                 │ complete giant titanosaur)     │            │
  └───────────────────────────────────────────────┴────────────────────────────────┴────────────┘
      Only Dreadnoughtus is described as the most complete giant titanosaur ("116 bones... 45.3%;
      100 of ~142 bone types = 70.4% complete"). The three decoys are known from a handful of bones.

  ROUND 2/3 keystone:
      Dreadnoughtus, holotype description: "At 1.74 m, its scapula is longer than any other known
      titanosaur shoulder blade."  [KEYSTONE = 1.74 m scapula length]
      (Its ilium is 1.31 m; total body length ~26 m; neck ~11.3 m; mass ~48–49 t — none collide
      with the 1.74 m token.)

Why the keystone is leak-resistant: a specific measured titanosaur scapula length (1.74 m) is an
osteological page-oddity no consumer LLM recalls parametrically; even naming the genus, a model
would guess. \\b1\\.74\\b collides with none of the other figures on the chain (26 / 11.3 / 1.31 /
48 / 49), so a single noisy extraction cannot satisfy it — only reading the correct survivor's page
can, which is exactly what the branch-then-chain forces.

Note (author swap from plan): the original plan keyed the keystone on a holotype FEMUR/HUMERUS
length, but the live Dreadnoughtus page publishes measured scapula (1.74 m) and ilium (1.31 m)
lengths rather than femur/humerus — so the keystone was retargeted to the measured scapula, the
same class of page-only measured-osteology figure the plan intended.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── ROUND-1 candidate set (single source of truth for statement, validators, plan) ──
# Four genuine giant titanosaurs. Only Dreadnoughtus is the most-complete skeleton; the others are
# fragmentary. `name_rx` tokens are pairwise distinct so coverage never cross-credits.
TITANS: List[Dict[str, Any]] = [
    {"key": "argentinosaurus", "name": "Argentinosaurus",
     "desc": "the fame-anchored 'largest dinosaur', known from famously fragmentary remains",
     "name_rx": r"argentinosaurus", "slug_rx": r"wiki/argentinosaurus", "survivor": False},
    {"key": "patagotitan", "name": "Patagotitan",
     "desc": "a colossal titanosaur known from partial remains of several individuals",
     "name_rx": r"patagotitan", "slug_rx": r"wiki/patagotitan", "survivor": False},
    {"key": "puertasaurus", "name": "Puertasaurus",
     "desc": "a giant titanosaur known from only four vertebrae",
     "name_rx": r"puertasaurus", "slug_rx": r"wiki/puertasaurus", "survivor": False},
    {"key": "dreadnoughtus", "name": "Dreadnoughtus",
     "desc": "a giant titanosaur whose holotype is the most complete of any giant sauropod",
     "name_rx": r"dreadnoughtus", "slug_rx": r"wiki/dreadnoughtus", "survivor": True},
]
SURVIVOR = next(t for t in TITANS if t["survivor"])  # Dreadnoughtus

# ── ROUND-3 keystone: the survivor's measured scapula length, 1.74 m ──
# \b1\.74\b matches "1.74 m" but never "11.74"/"1.740"; a bare imperial form is not published.
KEYSTONE_RX = re.compile(r"\b1\.74\b", re.IGNORECASE)
SURVIVOR_SLUG = r"wiki/dreadnoughtus"
# survivor chain-target: the named measured bone
_CHAIN_RX = re.compile(r"scapula|shoulder\s+blade|shoulder-?girdle", re.IGNORECASE)
_SURV_RX = re.compile(SURVIVOR["name_rx"], re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "104",
        "test_name": "Tier 5: Branch-eliminate then chain (giant titanosaurs -> most-complete survivor -> scapula length)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {t['name']} — {t['desc']}" for i, t in enumerate(TITANS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous "
        "stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. These four are all colossal titanosaur dinosaurs:\n"
        f"{listing}\n"
        "Exactly ONE of the four is known from the MOST COMPLETE skeleton of any giant sauropod (a "
        "large fraction of the skeleton recovered); the other three are known from famously "
        "fragmentary remains (a handful of bones each). Open EACH genus's page and read its "
        "skeletal completeness / holotype description to determine which one — do NOT simply pick "
        "the most famous 'largest dinosaur'.\n\n"
        "STAGE 2 — follow the survivor forward. Open the surviving genus's page. Because its "
        "skeleton is so complete, its holotype description gives an EXACT MEASURED length for a "
        "named bone — its shoulder blade (scapula), the longest such bone known for any titanosaur.\n\n"
        "STAGE 3 — read the keystone. Read that survivor's scapula (shoulder-blade) length in metres "
        "directly from its holotype description.\n\n"
        "Report: (a) the survivor's SCAPULA length in metres (this single figure is the keystone "
        "answer); (b) which of the four titanosaurs was the survivor and the completeness of each; "
        "citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor's scapula (shoulder-blade) length in metres (the leak-resistant keystone)",
        "Which titanosaur is the most-complete survivor (Dreadnoughtus) + each candidate's completeness",
        "The named measured bone (the scapula)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per candidate) plus the survivor's holotype figure",
        "Determines the completeness of ALL FOUR titanosaurs (branch-to-eliminate)",
        "Correctly elects Dreadnoughtus as the most-complete survivor (not the famous Argentinosaurus)",
        "Reports the survivor's scapula length (1.74 m)",
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
    """UN-gated process metric: the branch needs the four candidate pages, the chain the survivor's."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: four titanosaur candidates + the survivor; >=4 to pass)"}


def validate_keystone_scapula(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the survivor's measured scapula length (1.74 m). Leak-resistant."""
    passed = _keystone_ok(result)
    return {"check": "keystone_scapula", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Scapula length 1.74 m present" if passed
                      else "Keystone scapula length (1.74 m, Dreadnoughtus) missing/incorrect"}


def validate_elimination_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the four giant titanosaurs the agent actually named
    while investigating completeness. This separates a structured multi-round agent (which checks
    ALL four before electing) from a shallow one that grabs the famous 'largest'. NOT short-circuited
    on the keystone. Each candidate's distinct name token anchors credit; the credited count is
    CAPPED BY the visit count so a model narrating the four names from the mandate with ZERO page
    reads banks nothing (completeness is a page-only fact)."""
    text = _all_text(result)
    hits = [t["name"] for t in TITANS if re.search(t["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(TITANS)
    return {"check": "elimination_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} titanosaurs investigated from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_survivor_and_bone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the chain resolved correctly — names the survivor (Dreadnoughtus) AND the
    named measured bone (scapula). Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_bone", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/bone resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(_SURV_RX.search(text))
    has_bone = bool(_CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_bone)
    return {"check": "survivor_and_bone", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Dreadnoughtus)={has_survivor}, bone(scapula)={has_bone}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages the chain read. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [t["slug_rx"] for t in TITANS]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + a decoy)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_scapula,
        validate_elimination_coverage,
        validate_survivor_and_bone,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold for the graph_compiled variant.

    Three waves (4 -> 1 -> 1), a genuine tree (NOT a pure fan-out, NOT a pure chain):
      * WAVE 1 — four INDEPENDENT parallel leaves, one per candidate titanosaur, each reading ONLY
        that genus's skeletal completeness. Ids keyed on the GIVEN candidate names.
      * WAVE 2 — ONE dependent leaf that templates all four completeness reports, elects the
        most-complete survivor, and reports which genus it is. Depends on all four wave-1 leaves.
      * WAVE 3 — ONE dependent leaf that templates the survivor ({survivor}) and reads its named
        measured shoulder-bone length. Depends on wave 2.

    STRUCTURE only — names the GIVEN candidates and the GIVEN completeness criterion but leaks NO
    completeness figure, NOT which genus survives, and NOT the scapula length; the cheap runtime
    model still does every page-read."""
    cand_leaves = [
        {
            "id": f"titan_{t['key']}",
            "instruction": (
                f"Open the Wikipedia page for the titanosaur {t['name']}. Read how COMPLETE its "
                "known skeleton / holotype is (what fraction of the skeleton has been recovered, or "
                f"whether it is known only from fragmentary remains). Report the genus ({t['name']}), "
                "its skeletal completeness, and the exact Wikipedia URL. Do not guess from memory; "
                "do not report any other fact."
            ),
            "expect": f"{t['name']} — skeletal completeness — source URL",
            "depends_on": [],
        }
        for t in TITANS
    ]
    survivor_leaf = {
        "id": "survivor",
        "instruction": (
            "You are given four giant titanosaurs and how complete each one's skeleton is:\n"
            "  Argentinosaurus -> {titan_argentinosaurus}\n"
            "  Patagotitan -> {titan_patagotitan}\n"
            "  Puertasaurus -> {titan_puertasaurus}\n"
            "  Dreadnoughtus -> {titan_dreadnoughtus}\n"
            "Determine which SINGLE one is known from the MOST COMPLETE skeleton of any giant "
            "sauropod (the other three are fragmentary). Report which titanosaur is the most-complete "
            "survivor and its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The MOST-COMPLETE surviving titanosaur genus — source URL",
        "depends_on": [f"titan_{t['key']}" for t in TITANS],
    }
    bone_leaf = {
        "id": "scapula_length",
        "instruction": (
            "Open the Wikipedia page of the most-complete surviving titanosaur identified in the "
            "previous step ({survivor}). Because its skeleton is so complete, its holotype "
            "description gives an EXACT MEASURED length for its shoulder blade (scapula) — the "
            "longest such bone known for any titanosaur. Read that scapula length in METRES directly "
            "from the holotype description. Report the scapula length in metres and the source URL. "
            "Do not guess from memory."
        ),
        "expect": "The survivor's SCAPULA length in metres — source URL",
        "depends_on": ["survivor"],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf, bone_leaf],
        "aggregation": (
            "You now have (1) the skeletal completeness of each of the four titanosaurs, (2) which "
            "single one is the most-complete survivor, and (3) that survivor's scapula length. Write "
            "out all four completeness reports BEFORE concluding which survives. Then report (a) the "
            "survivor's SCAPULA length in metres — this single figure is the keystone answer; (b) "
            "which titanosaur was the survivor and each candidate's completeness; citing every source URL."
        ),
    }
