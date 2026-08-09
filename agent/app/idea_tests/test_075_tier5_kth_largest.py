"""
Test 075: Tier 5 (graph) — 3rd-deepest (ordinal k=3, NOT the extreme) over a PAGE-ONLY depth.
Level: graph   Weight: long   Difficulty: 8/10

Six fjords and fjord-type water bodies from three countries. For each, the agent opens the
Wikipedia page and reads ONE infobox figure — the MAXIMUM DEPTH in metres. The six depths are
then sorted from deepest to shallowest and the task asks which is the THIRD DEEPEST, deliberately
NOT the maximum (deepest) and NOT the minimum (shallowest).

The shape is engineered to expose the cheap-model failure mode and demonstrate that an expensive-
authored DAG scaffold recovers premium accuracy:
  * cheap native / parametric: shortcuts to "name the deepest" (Sognefjorden — the MAX decoy,
    the famous Norwegian fjord) or "name the most famous" (Milford Sound — the MIN decoy, the
    iconic New Zealand tourist site), both wrong; or it guesses an incorrect ordinal rank because
    it cannot recall all six depths.
  * graph_compiled: six parallel leaves gather all six depths; the aggregation step owns the
    explicit sort and k=3 selection, so the cheap executor is structurally rescued.

The trap is engineered two ways:
  (1) max depth is page-only — the agent MUST open each page and read the infobox figure; none
      of the six depths is a widely memorised fact;
  (2) the depth ranking diverges from intuition — Sognefjorden (the "famous deep Norwegian
      fjord") IS the deepest but NOT the 3rd deepest; the 3rd deepest is Jervis Inlet, a
      Canadian fjord-inlet almost unknown as a "deep fjord record"; and Milford Sound (a
      celebrated New Zealand tourist landmark) is the SHALLOWEST of the six.

Ground truth (verified against live English Wikipedia 2026-06-27 — each entity's infobox
'Max. depth' row; all six are independent entities with one unambiguous figure):

  entity                    country / region         max depth (m)
  Sognefjorden              Norway                      1,308   <- MAX DECOY (deepest)
  Hardangerfjord            Norway                        860   <- 2nd deepest
  Jervis Inlet              British Columbia, Canada      670   <- 3RD DEEPEST (KEYSTONE)
  Romsdalsfjord             Norway                        550   <- 4th deepest
  Lysefjord                 Norway                        422   <- 5th deepest
  Milford Sound             New Zealand                   291   <- MIN DECOY (shallowest)
    https://en.wikipedia.org/wiki/Sognefjorden
    https://en.wikipedia.org/wiki/Hardangerfjord
    https://en.wikipedia.org/wiki/Jervis_Inlet
    https://en.wikipedia.org/wiki/Romsdalsfjord
    https://en.wikipedia.org/wiki/Lysefjord
    https://en.wikipedia.org/wiki/Milford_Sound

  3RD DEEPEST = Jervis Inlet (670 m). Neighbours: Hardangerfjord (860 m) above, Romsdalsfjord
  (550 m) below. Margin above: 860 − 670 = 190 m (22.1% of 860). Margin below: 670 − 550 =
  120 m (17.9% of 670). Both margins exceed 15%, so no plausible single noisy extraction can
  flip the 3rd-place rank.

  Worst-case slip: Jervis Inlet read at ±10% → 603–737 m. Even at 603 m it remains above
  Romsdalsfjord (550 m); even at 737 m it remains below Hardangerfjord (860 m). Rank-3 is
  preserved under any realistic extraction error.

  Full depth ranking (confirmed live):
    1st  Sognefjorden   1,308 m  (MAX DECOY — model that "picks the deepest" picks this → WRONG)
    2nd  Hardangerfjord   860 m  (parametric fallback: "2nd-most-famous Norwegian fjord" → WRONG)
    3rd  Jervis Inlet     670 m  KEYSTONE — obscure Canadian inlet, depth not recallable
    4th  Romsdalsfjord    550 m
    5th  Lysefjord        422 m
    6th  Milford Sound    291 m  (MIN DECOY — model that "picks the most famous tourist spot"
                                  picks this → WRONG; Milford Sound is famous but shallowest)

  ANTI-PARAMETRIC: no standard "3rd-deepest cross-national fjord set" exists in training data.
  Jervis Inlet's depth (670 m) is not a widely cited figure; the inlet is not associated with
  "fjord depth records" in popular knowledge. A no-browse model cannot correctly rank this
  specific cross-national set of six without reading all six pages. It will either name
  Sognefjorden (deepest → 1st → wrong for k=3), or Hardangerfjord (2nd → wrong), or guess
  randomly. Milford Sound (shallowest) is the most famous tourist entity and is the MIN decoy.

  KEYSTONE = the 3rd-deepest ENTITY (Jervis Inlet). Secondary (gated) value = its depth ~670 m.
  Coverage = all 6 (entity, depth) pairs gathered. level "graph", weight "long".
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``depth`` is shown for provenance only; ONLY entity names + countries are surfaced in the task
# statement and the compiled plan — no figure and no winner / rank are ever leaked.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "sognefjorden", "name": "Sognefjorden", "country": "Norway",
     "depth": 1308, "winner": False,
     "name_rx": r"sognefjord(?:en)?",
     "depth_rx": r"(?<!\d)1[,]?308(?!\d)",
     "slug_rx": r"wiki/sognefjord(?:en)?"},
    {"key": "hardangerfjord", "name": "Hardangerfjord", "country": "Norway",
     "depth": 860, "winner": False,
     "name_rx": r"hardangerfjord",
     "depth_rx": r"(?<!\d)860(?!\d)",
     "slug_rx": r"wiki/hardangerfjord"},
    {"key": "jervis_inlet", "name": "Jervis Inlet", "country": "British Columbia, Canada",
     "depth": 670, "winner": True,
     "name_rx": r"jervis\s+inlet",
     "depth_rx": r"(?<!\d)670(?!\d)",
     "slug_rx": r"wiki/jervis_inlet"},
    {"key": "romsdalsfjord", "name": "Romsdalsfjord", "country": "Norway",
     "depth": 550, "winner": False,
     "name_rx": r"romsdalsfjord",
     "depth_rx": r"(?<!\d)550(?!\d)",
     "slug_rx": r"wiki/romsdalsfjord"},
    {"key": "lysefjord", "name": "Lysefjord", "country": "Norway",
     "depth": 422, "winner": False,
     "name_rx": r"lysefjord",
     "depth_rx": r"(?<!\d)422(?!\d)",
     "slug_rx": r"wiki/lysefjord"},
    {"key": "milford_sound", "name": "Milford Sound", "country": "New Zealand",
     "depth": 291, "winner": False,
     "name_rx": r"milford\s+sound",
     "depth_rx": r"(?<!\d)291(?!\d)",
     "slug_rx": r"wiki/milford_sound"},
]

WINNER = next(e for e in ENTITIES if e["winner"])    # Jervis Inlet — the 3rd deepest
WINNER_DEPTH = 670                                   # metres (verified live)
DEPTH_TOL = 0.025                                    # +/- 2.5% band on the secondary value check
K = 3                                                # ordinal position (3rd deepest)

# ---- robustness invariants: checked at import so fixture drift is caught immediately ----
_BY_DEPTH = sorted(ENTITIES, key=lambda e: e["depth"], reverse=True)
assert _BY_DEPTH[K - 1] is WINNER, (
    f"3rd-deepest must be the declared winner; got {_BY_DEPTH[K - 1]['name']}"
)
assert _BY_DEPTH[0] is not WINNER, "winner must NOT be the deepest (max decoy)"
assert _BY_DEPTH[-1] is not WINNER, "winner must NOT be the shallowest (min decoy)"
_DEPTH_ABOVE = _BY_DEPTH[K - 2]["depth"]   # 2nd deepest: Hardangerfjord 860 m
_DEPTH_BELOW = _BY_DEPTH[K]["depth"]       # 4th deepest: Romsdalsfjord 550 m
assert (_DEPTH_ABOVE - WINNER_DEPTH) / _DEPTH_ABOVE > 0.15, (
    f"margin above keystone ({(_DEPTH_ABOVE - WINNER_DEPTH) / _DEPTH_ABOVE:.1%}) must exceed 15%"
)
assert (WINNER_DEPTH - _DEPTH_BELOW) / WINNER_DEPTH > 0.15, (
    f"margin below keystone ({(WINNER_DEPTH - _DEPTH_BELOW) / WINNER_DEPTH:.1%}) must exceed 15%"
)

# Winner / other-entity name regexes used by the keystone gate.
_WINNER_RX = WINNER["name_rx"]                                   # r"jervis\s+inlet"
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# Ordinal triggers that assert a "3rd" rank position.
_ORDINAL = r"3(?:rd|d)\b|third"

# Keystone rank detection: Jervis Inlet tied to a '3rd deepest / third largest' assertion, in
# either direction, with the proximity window forbidden from crossing into ANY other entity's
# name. The window is [^.;] — newline-tolerant (a rank-list header then the name on the next
# line still matches) but bounded at sentence periods AND clause-separating semicolons.
# The dir-2 guard blocks "3rd shallowest ... Jervis Inlet" (an assertion from the wrong end)
# by excluding "shallowest / smallest / from bottom" before the winner name.
#   dir 1  (subject -> ordinal):  "Jervis Inlet ... is the 3rd deepest"
#   dir 2  (ordinal -> subject):  "The 3rd deepest is Jervis Inlet"  (not "3rd shallowest")
_JERVIS_3RD = re.compile(
    r"(?:" + _WINNER_RX + r")(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _ORDINAL + r")\b"
    + r"|\b(?:" + _ORDINAL + r")\b(?:(?!\bshallowest\b|\bsmallest\b|\bfrom\s+bottom\b|"
    + _OTHERS + r")[^.;]){0,55}(?:" + _WINNER_RX + r")",
    re.IGNORECASE,
)

# Depth-value detection: any 3–4 digit integer. Candidates filtered against WINNER_DEPTH band.
_DEPTH_NUM = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "075",
        "test_name": "Tier 5: Page-only depth k=3 ordinal (3rd deepest of six fjords)",
        "difficulty_level": "8/10",
        "category": "Breadth Fan-out & Aggregation (page-only depth, k-th largest, ordinal)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['country']})"
        for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not "
        "guess from memory). For EACH of the following six fjords and fjord-type water bodies, "
        "open the entity's Wikipedia page and read ONE figure directly from the infobox: its "
        "MAXIMUM DEPTH in metres.\n"
        f"{listing}\n\n"
        "Then SORT the six maximum-depth figures from deepest to shallowest and determine which "
        "entity is the THIRD DEEPEST (rank 3 out of 6). IMPORTANT: the third-deepest entity is "
        "NOT necessarily the largest, the longest, or the most well-known water body in the set "
        "— you must read each entity's maximum depth from its page and sort the actual figures "
        "rather than reasoning from fame or prominence.\n\n"
        "Report (a) which entity is the third deepest (a single name — the keystone), "
        "(b) that entity's maximum depth in metres, (c) all six entities' maximum depth figures "
        "listed in rank order (1st deepest through 6th deepest), and (d) the exact source URL "
        "of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which entity is the third deepest (the primary answer / keystone)",
        "That entity's maximum depth in metres",
        "All six entities' maximum depth figures in rank order (deepest to shallowest)",
        "Source URL for each entity's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per entity, a six-way fan-out)",
        "Correctly names Jervis Inlet as the third deepest (NOT Sognefjorden, which is the "
        "deepest, nor Milford Sound, which is the shallowest / most famous tourist site)",
        "Reports Jervis Inlet's maximum depth near 670 m (within +/- 2.5%)",
        "Reports all six entities' maximum depth figures in the correct rank order",
        "Cites the source pages",
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text: prefer deliverables[0] when present, else fall back."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: final deliverable plus all deliverable slots."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names Jervis Inlet as the 3rd deepest entity.

    Not satisfied by merely listing Jervis Inlet among the six: when another entity is also
    named in the primary slot, Jervis Inlet must be the one tied to a '3rd / third' ordinal
    assertion (via the proximity regex). A terse primary answer that names only the winner
    (Jervis Inlet, with no rival in the slot) also passes.

    The gate rejects:
      * naming Sognefjorden (MAX decoy, 1st deepest) as the answer;
      * naming Milford Sound (MIN decoy, 6th / shallowest) as the answer;
      * naming Hardangerfjord (2nd deepest) as the answer;
      * any response that names a rival as the 3rd without placing Jervis Inlet there.

    Also requires GROUNDING: the value string alone is insufficient — the agent must have
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
        return True   # names only the winner — terse correct answer
    return bool(_JERVIS_3RD.search(text))


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_visits(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out expects one page per entity."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 4,
        "score": min(1.0, n / 6.0),
        "reason": f"{n} visit(s) (target >=6: one page per entity; >=4 to pass)",
    }


def validate_keystone_kth(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the 3rd deepest entity is Jervis Inlet — NOT the deepest
    (Sognefjorden, the MAX decoy that a shortcut model names) and NOT the shallowest
    (Milford Sound, the MIN decoy / most famous tourist site)."""
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_kth",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Jervis Inlet named as the 3rd deepest entity" if passed
            else "3rd-deepest entity (Jervis Inlet, 670 m) missing or incorrect "
                 "(beware: Sognefjorden is the DEEPEST at 1,308 m; "
                 "Milford Sound is the SHALLOWEST at 291 m; "
                 "Hardangerfjord is 2nd deepest at 860 m)"
        ),
    }


def validate_coverage(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: how many of the SIX entities' depths were gathered.

    An entity is credited only when BOTH its name AND its (distinct) depth figure appear in the
    reported text; the six depth values are all distinct, so there is no cross-crediting.
    Deliberately NOT gated on the keystone — it measures whether the agent fanned out to all
    six entities and read each depth even when it botches the rank selection.

    Credit is CAPPED BY visit count (``min(hits, n_visits)``) so a 0-visit parametric-memory
    answer that merely recalls the figures cannot bank partial credit here without ever browsing.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE)
        and re.search(e["depth_rx"], text)
    ]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": credited == n,
        "score": credited / n,
        "reason": (
            f"{credited}/{n} entities' depths gathered ({', '.join(hits[:credited]) or 'none'}; "
            f"{len(hits)} text-matched, {n_visits} visit(s))"
        ),
    }


def validate_winner_depth(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: the keystone's depth (~670 m, within +/- 2.5%).
    Short-circuits to 0 when the keystone is absent, so a wrong / guessed 3rd-deepest entity
    cannot bank the value credit."""
    if not _keystone_ok(result, observability):
        return {
            "check": "winner_depth",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> depth value not credited",
        }
    text = _all_text(result)
    lo = WINNER_DEPTH * (1.0 - DEPTH_TOL)
    hi = WINNER_DEPTH * (1.0 + DEPTH_TOL)
    found = [int(m) for m in _DEPTH_NUM.findall(text)]
    ok = any(lo <= v <= hi for v in found)
    return {
        "check": "winner_depth",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"Jervis Inlet depth within {lo:.0f}–{hi:.0f} m present" if ok
            else f"no depth in {lo:.0f}–{hi:.0f} m (expected ~{WINNER_DEPTH} m); "
                 f"saw {found or 'none'}"
        ),
    }


def validate_citation(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: cites the entity pages. Short-circuits to 0 when keystone absent."""
    if not _keystone_ok(result, observability):
        return {
            "check": "citation",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> source URLs not credited",
        }
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {
        "check": "citation",
        "passed": cited >= 3,
        "score": cited / n,
        "reason": f"{cited}/{n} entity pages cited",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_kth,
        validate_coverage,
        validate_winner_depth,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 062/064.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves: for each of the six GIVEN entities, one leaf opens that
    entity's Wikipedia page and reads ONE atomic figure — its maximum depth in metres. The
    depth figure sits in the infobox of each page, so one page-read per entity suffices.

    ALL ordering and rank selection lives ONLY in the aggregation step, which is forced to
    WRITE OUT the full sorted list (all six entities and depths, numbered 1–6 from deepest to
    shallowest) before concluding, so the cheap executor can never shortcut to fame-based
    guessing and a diverse-grounding reranker can re-derive and cross-check the 3rd-place rank.

    Encodes STRUCTURE only: names the six GIVEN entities and their countries, but leaks no
    depth figure, no rank, and no winner entity.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its MAXIMUM DEPTH in metres. "
                "Report ONLY that single depth figure (in metres) and the source URL. "
                "Do not guess from memory."
            ),
            "expect": "MAXIMUM DEPTH in metres (a single integer figure) — source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six water bodies, its maximum depth in metres "
            "(with a source URL). SORT the six depths from deepest to shallowest and write "
            "out the FULL ORDERED LIST before drawing any conclusion — list each entity and "
            "its depth, numbered 1 (deepest) through 6 (shallowest). "
            "THEN, from that ordered list, identify the entity at rank position 3 — that is "
            "the THIRD DEEPEST, which is the keystone answer. "
            "NOTE: the third deepest is NOT necessarily the largest, longest, or most famous "
            "water body — sort the actual depth figures you looked up, not by fame or size. "
            "Report (a) the third-deepest entity and its maximum depth in metres, stating "
            "explicitly that it is the third deepest, (b) the full ranked list (all six "
            "entities and depths, numbered 1 through 6 from deepest to shallowest), and "
            "(c) cite each entity's source URL."
        ),
    }
