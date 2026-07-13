"""
Test 071: Tier 5 (graph) — CLOSEST-TO a reference (argmin of |value - reference|).
Level: graph   Weight: long   Difficulty: 9/10

A hard quantitative-reasoning task built to punish the cheap-model shortcuts of "pick the deepest",
"pick the shallowest", and "pick the most familiar lake". Given ONE reference lake and FIVE candidate
lakes, the agent must look up a single comparable infobox integer — the lake's MAXIMUM DEPTH in
metres — for the reference and for each candidate, then report which CANDIDATE's depth is CLOSEST to
the reference's, i.e. the candidate that MINIMISES the absolute difference |candidate depth −
reference depth|.

    REFERENCE: Lake Matano (Sulawesi, Indonesia)
    CANDIDATES: Lake Superior   Lake Kivu   Lake Tahoe   Great Slave Lake   Lake Malawi

"Closest" is a COMPUTED RELATION, not a memorizable fact: nothing on any page states how near one
lake's depth is to another's, so the agent must read all six integers, subtract each candidate from
the reference, and take the argmin of the five absolute differences. The reference lake (Lake Matano)
is an obscure Indonesian lake whose maximum depth of 590 m is not a parametrically recallable figure,
and the closest candidate (Great Slave Lake, 614 m) wins by an absolute difference of only 24 m —
far too narrow to guess without reading the actual page — so the three classic shortcuts all mis-pick.

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): drops a lake, mis-subtracts, or shortcuts to a raw extreme (deepest →
    Lake Malawi, or shallowest → Lake Superior) or to the most recognizable name (Lake Superior /
    Lake Malawi / Lake Tahoe) and names the WRONG lake.
  * frontier sequential (ReAct): looks up all six depths, subtracts, compares → decent.
  * graph_compiled: six parallel leaves each fetch ONE depth (one integer for one lake, the
    reference included); the aggregation owns EVERY subtraction AND the argmin, and is forced to
    WRITE OUT each per-candidate absolute difference before concluding → the cheap executor is
    rescued and a diverse-grounding reranker can re-derive it.

The trap is engineered three ways:
  (1) "nearness" is page-only — no infobox states how close two lakes' depths are, so the agent
      MUST subtract and compare absolute differences;
  (2) the closest candidate (Great Slave Lake) is NEITHER the deepest (Lake Malawi, 706 m) NOR
      the shallowest (Lake Superior, 406 m) — so "pick deepest/shallowest" fails both ways;
  (3) the closest candidate is NOT the most famous candidate (Lake Superior and Lake Malawi are far
      more widely known, yet both are wrong), so a fame/most-recognizable shortcut also mis-picks.

Ground truth (verified against live English Wikipedia 2026-06-27 — each lake's infobox 'Max. depth'
row, the metres figure; every figure is a single unambiguous 'Max. depth' entry; Lake Kivu's
infobox reads 480 m while its article body mentions 475 m — the infobox figure 480 m is used here):

  role        lake                 max depth m   slug                          |depth − Matano|
  REFERENCE   Lake Matano          590           wiki/Lake_Matano                        -
  candidate   Lake Superior        406           wiki/Lake_Superior               184   <- SHALLOWEST (decoy)
  candidate   Lake Kivu            480           wiki/Lake_Kivu                   110
  candidate   Lake Tahoe           501           wiki/Lake_Tahoe                   89   <- runner-up
  candidate   Great Slave Lake     614           wiki/Great_Slave_Lake             24   <- ARGMIN (keystone)
  candidate   Lake Malawi          706           wiki/Lake_Malawi                 116   <- DEEPEST (decoy)

  CLOSEST-TO ARGMIN = Great Slave Lake (|614 − 590| = 24 m). Runner-up = Lake Tahoe (89 m), so
  the margin is +65 m absolute (runner-up is 3.7× further away than the winner).
  WORST-CASE-SLIP CHECK: for the argmin to flip from Great Slave Lake to Tahoe, the reference
  would have to be misread below the Great Slave / Tahoe boundary at (614 + 501) / 2 = 557.5 m,
  i.e. ≥33 m below the true reference of 590 m. For the flip to Kivu the boundary is
  (614 + 480) / 2 = 547.0 m, i.e. ≥43 m below. A single noisy text extraction would need to move
  one depth figure by at least 33 m to flip the winner; a ±1 or ±5 m misread cannot.

  RANKING DIVERGENCE (confirmed live):
    by raw depth: Malawi 706 > Great Slave 614 > Tahoe 501 > Kivu 480 > Superior 406
                                                         → deepest Malawi, shallowest Superior
    by |depth − Matano 590|: Great Slave 24 < Tahoe 89 < Kivu 110 < Malawi 116 < Superior 184
                                                         → closest Great Slave Lake
    => the closest candidate (Great Slave Lake, 614 m) is the SECOND deepest by raw depth:
       NEITHER the deepest (Malawi) NOR the shallowest (Superior). "Pick deepest" lands on Malawi
       (wrong) and "pick shallowest" lands on Superior (wrong).

  ANTI-PARAMETRIC: Lake Superior (the most prominent Great Lake, largest by surface area) and Lake
  Malawi (a famous African Rift Valley lake, also called Nyasa) are the obvious famous guesses, but
  both are decoys. Lake Tahoe (also well-known, "second deepest lake in the US") is the runner-up
  but still wrong. The keystone is Great Slave Lake — the deepest lake in North America at 614 m —
  whose nearness to the obscure Indonesian Lake Matano (590 m) yields a gap of only 24 m. This
  nearness is not a recallable fact: Lake Matano's depth is not commonly known (it is the deepest
  lake in Indonesia / Southeast Asia, the 11th deepest globally by some rankings), and the specific
  24 m proximity between these two geographically unrelated lakes cannot be guessed without reading
  both pages.

  KEYSTONE = the argmin CANDIDATE (Great Slave Lake). Secondary (gated) value = its depth (614 m)
  or the computed difference (~24 m). All six depths are distinct (590, 406, 480, 501, 614, 706),
  so the un-gated coverage diagnostic is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``depth`` / ``diff`` are recorded for provenance only; nothing here is leaked into the statement
# or the compiled plan beyond the lake names already given to the agent.
REFERENCE: Dict[str, Any] = {
    "name": "Lake Matano", "region": "Sulawesi, Indonesia", "depth": 590,
    "name_rx": r"\b(?:lake\s+)?matano\b",
    "depth_rx": r"(?<!\d)590(?!\d)",
    "slug_rx": r"wiki/lake_matano",
}

CANDIDATES: List[Dict[str, Any]] = [
    {"key": "superior", "name": "Lake Superior", "region": "North America",
     "depth": 406, "diff": 184, "closest": False,
     "name_rx": r"\bsuperior\b",
     "depth_rx": r"(?<!\d)406(?!\d)",
     "slug_rx": r"wiki/lake_superior"},
    {"key": "kivu", "name": "Lake Kivu", "region": "Rwanda / Democratic Republic of Congo",
     "depth": 480, "diff": 110, "closest": False,
     "name_rx": r"\bkivu\b",
     "depth_rx": r"(?<!\d)48[05](?!\d)|(?<!\d)475(?!\d)",   # infobox 480 m; article body 475 m
     "slug_rx": r"wiki/lake_kivu"},
    {"key": "tahoe", "name": "Lake Tahoe", "region": "California / Nevada, United States",
     "depth": 501, "diff": 89, "closest": False,
     "name_rx": r"\btahoe\b",
     "depth_rx": r"(?<!\d)501(?!\d)",
     "slug_rx": r"wiki/lake_tahoe"},
    {"key": "great_slave", "name": "Great Slave Lake", "region": "Northwest Territories, Canada",
     "depth": 614, "diff": 24, "closest": True,
     "name_rx": r"\bgreat\s+slave\b",
     "depth_rx": r"(?<!\d)614(?!\d)",
     "slug_rx": r"wiki/great_slave_lake"},
    {"key": "malawi", "name": "Lake Malawi", "region": "Malawi / Mozambique / Tanzania",
     "depth": 706, "diff": 116, "closest": False,
     "name_rx": r"\bmalawi\b|\bnyasa\b|\bniassa\b",
     "depth_rx": r"(?<!\d)706(?!\d)",
     "slug_rx": r"wiki/lake_malawi|wiki/lake_nyasa"},
]

ALL_ENTITIES: List[Dict[str, Any]] = [REFERENCE] + CANDIDATES   # the six depths for coverage

WINNER = next(c for c in CANDIDATES if c["closest"])   # Great Slave Lake — the closest-to argmin
WINNER_DEPTH = 614                                      # Great Slave Lake's maximum depth, m
WINNER_DIFF = 24                                        # |614 - 590|

# ---- DESIGN INVARIANTS, asserted at import so fixtures can never silently drift out of spec. ----
_BY_DEPTH = sorted(CANDIDATES, key=lambda c: c["depth"])
_DEEPEST_CAND = _BY_DEPTH[-1]
_SHALLOWEST_CAND = _BY_DEPTH[0]
_BY_DIFF = sorted(CANDIDATES, key=lambda c: c["diff"])
assert _BY_DIFF[0] is WINNER, "argmin of |diff| must be the declared winner"
assert _DEEPEST_CAND is not WINNER, "winner must NOT be the deepest candidate (decoy A)"
assert _SHALLOWEST_CAND is not WINNER, "winner must NOT be the shallowest candidate (decoy B)"
_MARGIN_RATIO = _BY_DIFF[1]["diff"] / _BY_DIFF[0]["diff"]
assert _MARGIN_RATIO > 2.0, (
    f"runner-up must be at least 2× further than winner (got {_MARGIN_RATIO:.2f}×)"
)
assert all(c["diff"] == abs(c["depth"] - REFERENCE["depth"]) for c in CANDIDATES), (
    "candidate diffs must equal |depth - reference depth|"
)

# Winner / other-candidate name regexes used by the keystone gate. ``_OTHERS`` lists only the OTHER
# CANDIDATES — the reference (Lake Matano) is deliberately NOT included, so "Great Slave Lake is
# closest to Lake Matano" is allowed inside the proximity window.
_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(c["name_rx"] for c in CANDIDATES if not c["closest"])

# Closeness triggers that assert a winner ("X is closest to / nearest to the reference"). Same
# battle-tested set as the sibling closest-to test (formerly test_071). Deliberately excludes bare
# "deepest"/"shallowest" so "the shallowest lake" (describing Lake Superior's raw depth, not its
# nearness to the reference) does NOT count.
_NEAR = (r"closest|nearest|closer|nearer|most\s+similar|best\s+match|"
         r"smallest\s+(?:absolute\s+)?difference|least\s+(?:absolute\s+)?difference|"
         r"minimum\s+(?:absolute\s+)?difference|smallest\s+gap")

# Keystone winner detection: Great Slave Lake tied to a 'closest / smallest difference' assertion,
# in either direction, with the proximity window forbidden from crossing into ANY other candidate's
# name (so "Lake Malawi is closest; Great Slave Lake is second" can never satisfy it). The window
# is [^.;] — newline-tolerant but bounded at sentence periods AND semicolons. The "than" guard
# blocks "closer than Great Slave Lake" (Great Slave Lake the LOSER) from counting.
#   dir 1  (subject → closeness):  "Great Slave Lake … is closest to Lake Matano's depth"
#   dir 2  (closeness → subject):  "the closest is Great Slave Lake"  (NOT "closer than Great Slave")
_GREAT_SLAVE_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _NEAR + r")\b"
    + r"|\b(?:" + _NEAR + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Difference-value detector: matches the exact winner difference "24" (word-bounded) for the gated
# secondary. The value is small enough that _INT_RX (3–7-digit extractor) would miss it, so we
# match it directly. Gating on the keystone prevents spurious "24"s from banking credit.
_DIFF_RX = re.compile(r"(?<!\d)24(?!\d)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "071",
        "test_name": "Tier 5: Closest-to-reference (argmin of absolute difference in max depth)",
        "difficulty_level": "9/10",
        "category": "Quantitative reasoning + closest-to (argmin |value - reference|)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {c['name']} ({c['region']})" for i, c in enumerate(CANDIDATES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). You are given ONE reference lake and FIVE candidate lakes. For the reference "
        "lake AND for EACH candidate, open the lake's Wikipedia page and read ONE integer from the "
        "infobox 'Max. depth' row: the lake's MAXIMUM DEPTH in metres. That is six depth figures "
        "in total.\n\n"
        f"REFERENCE lake: {REFERENCE['name']} ({REFERENCE['region']})\n"
        f"CANDIDATE lakes:\n{listing}\n\n"
        "Then COMPUTE, for each of the five candidates, the ABSOLUTE DIFFERENCE between that "
        f"candidate's maximum depth and {REFERENCE['name']}'s maximum depth "
        "(|candidate depth − reference depth|; this difference is NOT printed on any page — you "
        "must subtract the two figures yourself). COMPARE the five absolute differences and "
        f"determine which CANDIDATE lake's maximum depth is CLOSEST to {REFERENCE['name']}'s (the "
        "smallest absolute difference). Note: the answer is the candidate NEAREST in depth, which "
        "is NOT necessarily the candidate with the greatest depth, nor the one with the shallowest "
        "depth, nor the most recognizable lake.\n\n"
        "Report (a) which candidate lake's depth is closest to "
        f"{REFERENCE['name']}'s (a single lake name — the keystone), (b) that candidate's maximum "
        "depth and its absolute difference from the reference, (c) all six maximum-depth figures "
        "you looked up (the reference and the five candidates), and (d) the exact source URL of "
        "every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        (f"Which candidate lake's maximum depth is closest to {REFERENCE['name']}'s "
         "(the primary answer / keystone)"),
        "That closest candidate's maximum depth and its absolute difference from the reference",
        "All six maximum-depth figures (the reference and the five candidates)",
        "Source URL for each lake's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per lake: reference + five candidates, a six-way fan-out)",
        "Correctly names Great Slave Lake as the candidate closest in depth to Lake Matano (NOT "
        "Lake Malawi, the deepest candidate, NOR Lake Superior, the shallowest, NOR the famous "
        "Lake Tahoe which is the runner-up but wrong)",
        f"Reports the closest candidate's depth ({WINNER_DEPTH} m) or its difference from the "
        f"reference (~{WINNER_DIFF} m)",
        "Gathers all six maximum-depth figures (Lake Matano and the five candidates)",
        "Cites the source pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text. Prefer ``deliverables[0]`` (the contract's primary slot) when present;
    otherwise fall back to ``output.final_deliverable``."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: the final deliverable plus every deliverable slot, so coverage / value /
    citation checks can see figures the agent placed outside the primary answer slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names Great Slave Lake as the candidate closest in depth to
    Lake Matano.

    Word-bounded, and NOT satisfied by merely listing Great Slave Lake among the candidates: when
    another candidate is also named, Great Slave Lake must be the one tied to a 'closest / smallest
    difference' assertion (tempered so a rival named as the winner — or 'closer than Great Slave
    Lake' — never counts). A terse primary answer that names only the winner (Great Slave Lake,
    with no rival in the slot) also passes. Mentioning the reference (Lake Matano) is always allowed.

    Also requires GROUNDING: the value string alone is insufficient — the agent must have actually
    visited at least one page (visit.count > 0), else an ungrounded parametric-memory guess would
    earn credit."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True  # names only the winner (reference mentions are fine)
    return bool(_GREAT_SLAVE_WINS.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per lake (reference + five)."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 5, "score": min(1.0, n / 6.0),
            "reason": f"{n} visit(s) (target >=6: one page per lake; >=5 to pass)"}


def validate_keystone_closest(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the candidate closest in depth to Lake Matano is Great Slave Lake —
    NOT the deepest (Lake Malawi), NOT the shallowest (Lake Superior), and NOT the most familiar
    (Lake Superior / Lake Malawi / Lake Tahoe). A model that shortcuts to a raw extreme or guesses
    a famous name mis-picks."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_closest", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Great Slave Lake named as the candidate closest in depth to Lake Matano"
                       if passed else
                       "Closest-depth candidate (Great Slave Lake) missing/incorrect (beware: Lake "
                       "Malawi is deepest, Lake Superior is shallowest, Lake Tahoe is runner-up "
                       "but wrong)")}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the SIX depths (reference + five
    candidates) were gathered.

    A depth is credited only when BOTH that lake's name AND its depth figure appear; the six depth
    integers are all distinct (590, 406, 480, 501, 614, 706), so there is no cross-crediting.
    Deliberately NOT gated on the keystone — it measures whether the agent actually fanned out to
    all six lakes even when it botches the subtraction or the argmin, the axis that separates a
    structured (multi-leaf) agent from a linear one that drops an entity.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ALL_ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["depth_rx"], text)]
    n = len(ALL_ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} lakes' depths gathered ({', '.join(hits) or 'none'})"}


def validate_closest_value(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the closest candidate's depth (614 m) OR the computed absolute difference
    (~24 m). Short-circuits to 0 when the keystone is absent, so a wrong/guessed winner can't bank
    the value credit."""
    if not _keystone_ok(result, observability):
        return {"check": "closest_value", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> value not credited"}
    text = _all_text(result)
    has_depth = bool(re.search(WINNER["depth_rx"], text))
    has_diff = bool(_DIFF_RX.search(text))
    ok = has_depth or has_diff
    return {"check": "closest_value", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"Great Slave Lake's depth ({WINNER_DEPTH} m) or difference "
                       f"~{WINNER_DIFF} m reported"
                       if ok else
                       f"neither Great Slave Lake's depth ({WINNER_DEPTH} m) nor the "
                       f"difference ~{WINNER_DIFF} m present")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the lake pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ALL_ENTITIES if re.search(e["slug_rx"], text))
    n = len(ALL_ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} lake pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_closest,
        validate_coverage,
        validate_closest_value,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 064/065.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves: one leaf per lake (the reference Lake Matano and each of the
    five candidates) fetches that lake's MAXIMUM DEPTH in metres — one atomic integer per leaf.
    ALL arithmetic — every |candidate − reference| subtraction AND the final argmin — lives only
    in the aggregation step, which is forced to WRITE OUT each per-candidate absolute difference
    explicitly before concluding, so the cheap executor never subtracts the wrong pair or shortcuts
    to a raw extreme, and a diverse-grounding reranker can re-derive and catch a slip. Encodes
    STRUCTURE only: it names the reference and the five given candidates but leaks no depth figure,
    no difference, and not which candidate wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ALL_ENTITIES:
        key = e.get("key", "reference")
        role = "REFERENCE" if e is REFERENCE else "candidate"
        region = e.get("region", "")
        leaves.append({
            "id": f"{key}_depth",
            "instruction": (
                f"Open the Wikipedia page for {e['name']} ({region}) and read, from the infobox "
                f"'Max. depth' row, the lake's MAXIMUM DEPTH in metres. Report ONLY that single "
                "integer (the metres figure, not the feet figure) and the source URL. Do not "
                "guess from memory."
            ),
            "expect": "MAXIMUM DEPTH in metres (a single integer) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            f"You now have the maximum depth (metres) of six lakes: the REFERENCE lake "
            f"{REFERENCE['name']} and the five candidate lakes "
            f"({', '.join(c['name'] for c in CANDIDATES)}). For EACH of the five candidates, "
            f"write out the absolute difference explicitly on its own line in the form "
            f"'<candidate>: |<candidate depth> − <{REFERENCE['name']} depth>| = <difference>' — "
            "compute every one of the five absolute differences BEFORE drawing any conclusion. "
            "THEN, comparing those five differences, state which SINGLE candidate lake's maximum "
            f"depth is CLOSEST to {REFERENCE['name']}'s (the smallest absolute difference) — "
            "that candidate's name is the keystone answer. Show every subtraction before "
            "concluding. This need NOT be the candidate with the greatest depth, nor the one with "
            "the shallowest depth, nor the most familiar lake. Report (a) that candidate, its "
            "maximum depth, and its absolute difference from the reference, (b) all six "
            "maximum-depth figures, and (c) cite each lake's source URL."
        ),
    }
