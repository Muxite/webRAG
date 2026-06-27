"""
Test 080: Tier 5 (integration) — ODD-ONE-OUT / NEGATION (the shared-property exception), variant B.
Level: integration   Weight: long   Difficulty: 9/10

A negation-trap task built to punish two cheap-model failure modes at once: (1) flipping "which
does NOT drain to the Caspian Sea" into "which DOES" (the negation flip), and (2) failing to fan
out across all five rivers so the lone, non-obvious exception is missed. The agent is given FIVE
rivers of the South Caucasus / Georgia that all APPEAR to share one binary hydrographic property --
draining to the Caspian Sea (via the Kura River). Exactly FOUR genuinely do; exactly ONE does NOT.
The agent must identify the SINGLE river that does NOT satisfy the property:

    Alazani    Aragvi    Iori    Khrami    Rioni

The property is binary and UNAMBIGUOUS on Wikipedia, with TWO authoritative cross-checks:
  * each Caspian-draining river's Wikipedia infobox shows "Mouth: Kura" and basin chain "Kura ->
    Caspian Sea"; the exception's lead sentence explicitly states it flows to the Black Sea; and
  * the Wikipedia "Rioni" article's lead sentence confirms: "flows west to the Black Sea, entering
    it north of the city of Poti."
There is no judgement call -- a river either ultimately drains to the Caspian Sea or it does not.

The exception is SOLID-BUT-NOT-ICONIC and is the OPPOSITE of guessable from the geographic prior:
the Rioni is the second-longest river in Georgia and the dominant river of western Georgia, yet the
dominant drainage of the entire South Caucasus flows EASTWARD to the Caspian (via the Kura system).
All four satisfiers (Alazani, Aragvi, Iori, Khrami) are tributaries of the Kura and each Wikipedia
infobox shows the identical basin chain "Kura -> Caspian Sea." The Rioni flows WESTWARD to Poti on
the Black Sea -- the exact opposite direction -- but a model without specific Caucasian geography
knowledge defaults to "Caspian" for all five rivers, because the Caspian basin is the larger and
more prominent of the two drainage systems in the region. Two further traps are baked in:
  * SAME-COUNTRY TRAP: all five rivers flow through or border Georgia, so the geographic setting
    gives no hint that one drains to a different sea.
  * NO ALTERNATIVE SINGLE-OUT: there is no clean competing 4-1 split. Alazani, Iori, and Khrami
    have their mouths in Azerbaijan; Aragvi and Rioni have their mouths in Georgia -> 3-2 not 4-1.
    No single alternative binary property singles out exactly one river; the ONLY clean 4-1 split is
    "drains to the Caspian Sea," and it singles out the Rioni.

Why it discriminates (per REASONING_TEST_DESIGN.md -- the differential-lift target):
  * cheap native (parametric / graph): defaults "Caspian" for all five (geographic prior), or flips
    the negation and names one of the four Caspian tributaries, or drops the Rioni and never checks
    it -> wrong odd-one-out.
  * frontier sequential (ReAct): reads each article, spots Rioni's westward Black Sea drainage,
    handles the negation -> decent.
  * graph_compiled: five neutral parallel leaves each determine ONE river's drainage destination;
    the aggregation owns the negation -- it is forced to WRITE OUT each river's drainage BEFORE
    naming the single one that fails the property -> the cheap executor is rescued from both the
    flip and the dropped-entity failure.

The trap is engineered two ways:
  (1) NEGATION: the question asks for the one that does NOT drain to the Caspian Sea; a model that
      reads "Caspian" and names the most Caspian-looking river inverts the task.
  (2) NON-OBVIOUS EXCEPTION: the exception (Rioni) is the LEAST-expected non-Caspian river of the
      five -- all five are Georgian rivers in the Caucasus, a region dominated by Caspian drainage;
      fame/stereotype points AWAY from picking Rioni as the exception, so the model must read the
      page.

Ground truth (verified against live English Wikipedia 2026-06-27 -- each river's lead sentence /
infobox; the property "drains to the Caspian Sea" is binary and explicitly stated):

  river     drains to Caspian?   evidence (live lead / infobox)
  Alazani       YES              infobox: Mouth = Kura at Mingechevir, Azerbaijan; basin "Kura -> Caspian Sea"
  Aragvi        YES              infobox: Mouth = Mtkvari (Kura) at Mtskheta, Georgia; basin "Kura -> Caspian Sea"
  Iori          YES              article: "flows into the Mingachevir reservoir, which is drained by the Kura"; basin -> Caspian Sea
  Khrami        YES              infobox: "a right tributary of the Kura (Mtkvari)"; basin "Kura -> Caspian Sea"
  Rioni         NO               lead: "flows west to the Black Sea, entering it north of the city of Poti"  <- ODD ONE OUT
    https://en.wikipedia.org/wiki/Alazani_River
    https://en.wikipedia.org/wiki/Aragvi
    https://en.wikipedia.org/wiki/Iori_River
    https://en.wikipedia.org/wiki/Khrami_River
    https://en.wikipedia.org/wiki/Rioni

  ROBUSTNESS: EXACTLY ONE river (Rioni) violates the shared property "drains to the Caspian Sea";
  the other four clearly satisfy it (each is a tributary of the Kura River, whose mouth is on the
  Caspian Sea; each Wikipedia infobox shows the basin chain "Kura -> Caspian Sea"). The property is
  binary and unambiguous -- no partial drainage, no disputed outflow. A single noisy read cannot
  flip it: there is no near-Caspian ambiguity among the four satisfiers, and the Rioni's Black Sea
  drainage is explicitly stated in its lead sentence and confirmed by its mouth location (Poti, on
  the Georgian Black Sea coast).

  KEYSTONE = the odd-one-out ENTITY: Rioni, identified as the one that does NOT drain to the
  Caspian Sea (i.e. the one that drains to the Black Sea). The keystone is robust to the negation
  trap: it is NOT credited when a model names one of the four Caspian-draining rivers, and --
  critically -- it is NOT credited when a model names the Rioni but asserts it DOES drain to the
  Caspian Sea (i.e. the negation-flip decoy). Secondary (gated) = correctly states WHY it is the
  exception (it drains to the Black Sea / flows to Poti). Coverage (un-gated, collision-free) = how
  many of the five had their drainage destination determined.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``caspian`` is shown for provenance only; nothing about which river is the exception is leaked
# into the task statement or the compiled plan.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "alazani", "name": "Alazani", "caspian": True, "odd": False,
     "name_rx": r"\balazani\b", "slug_rx": r"wiki/alazani"},
    {"key": "aragvi", "name": "Aragvi", "caspian": True, "odd": False,
     "name_rx": r"\baragvi\b", "slug_rx": r"wiki/aragvi"},
    {"key": "iori", "name": "Iori", "caspian": True, "odd": False,
     "name_rx": r"\biori\b", "slug_rx": r"wiki/iori"},
    {"key": "khrami", "name": "Khrami", "caspian": True, "odd": False,
     "name_rx": r"\bkhrami\b", "slug_rx": r"wiki/khrami"},
    {"key": "rioni", "name": "Rioni", "caspian": False, "odd": True,
     "name_rx": r"\brioni\b", "slug_rx": r"wiki/rioni", "sea": "Black Sea"},
]

ODD = next(e for e in ENTITIES if e["odd"])    # Rioni -- the not-Caspian exception
ODD_SEA = ODD["sea"]                            # "Black Sea"

# ----- keystone regex machinery (negation-robust) ----------------------------------------
_RIONI = r"\brioni\b"
_OTHERS = r"\b(?:alazani|aragvi|iori|khrami)\b"   # the four Caspian-draining rivers

# Contrast / clause-pivot words: when one of these sits between two subjects, an attribute on one
# side does NOT belong to the subject on the other side.
_BREAK = r"\b(?:than|unlike|while|whilst|whereas|but|however|except|yet|though|although)\b"
# Negators: block an affirmative reading of a following property word.
_NEG = r"(?:\bnot\b|\bno\b|\bnever\b|isn'?t|aren'?t|\bn't\b)"
# Group/aggregate references: "the rest / all four drain to the Caspian" attaches the claim to
# the GROUP, not to a nearby Rioni mention.
_GROUP = r"\b(?:rest|others?|remaining|four|all|each)\b"

# VETO -- "Rioni drains to the Caspian" asserted affirmatively (the negation-flip decoy that names
# the right entity but assigns it the WRONG property value). Window forbidden from crossing a
# negator, a contrast pivot, a group reference or another river name.
_RIONI_CASP = re.compile(
    _RIONI + r"(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _GROUP + r"|" + _OTHERS
    + r")[^.;]){0,50}\bcaspian\b",
    re.IGNORECASE,
)

# Rioni tied to a "Black Sea / not-Caspian / is-the-exception" assertion. Either direction,
# window forbidden from crossing into another river's name (so a drainage attribute reported for
# one of the four cannot be borrowed by a nearby Rioni mention).
_NOTCASP = (
    r"black\s+sea|poti|not\s+caspian|isn'?t\s+caspian|"
    r"drains?\s+to\s+the\s+black|flows?\s+to\s+the\s+black|empties?\s+into\s+the\s+black|"
    r"odd\s+one|outlier|exception|doesn'?t\s+belong|does\s+not\s+belong|"
    r"breaks?\s+the\s+pattern|stands?\s+out"
)
_RIONI_IS_ODD = re.compile(
    _RIONI + r"(?:(?!" + _OTHERS + r")[^.;]){0,80}\b(?:" + _NOTCASP + r")\b"
    + r"|\b(?:" + _NOTCASP + r")\b(?:(?!" + _OTHERS + r")[^.;]){0,80}" + _RIONI,
    re.IGNORECASE,
)

# RIVAL VETO -- one of the FOUR Caspian rivers asserted as the odd-one-out / the exception (a
# wrong-entity answer). Window forbidden from crossing a negator, a contrast pivot or the Rioni
# name, so a correct contrastive phrasing ("the four drain to Caspian, whereas Rioni goes to the
# Black Sea") never trips it, but "Alazani is the exception; it drains to the Black Sea" does.
_RIVAL_MARK = (
    r"odd\s+one|outlier|exception|doesn'?t\s+belong|does\s+not\s+belong|"
    r"breaks?\s+the\s+pattern|not\s+caspian|isn'?t\s+caspian|"
    r"drains?\s+to\s+the\s+black\s+sea|black\s+sea"
)
_RIVAL_ODD = re.compile(
    _OTHERS + r"(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _RIONI + r")[^.;]){0,60}\b(?:" + _RIVAL_MARK + r")\b"
    + r"|\b(?:" + _RIVAL_MARK + r")\b(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _RIONI + r")[^.;]){0,60}" + _OTHERS,
    re.IGNORECASE,
)

# WHY (gated secondary): the property value that differs -- Rioni drains to the Black Sea (not the
# Caspian); enters the Black Sea near Poti.
_WHY = re.compile(
    r"black\s+sea|poti|drains?\s+to\s+the\s+black|flows?\s+to\s+the\s+black|"
    r"empties?\s+into\s+the\s+black|not\s+caspian|isn'?t\s+caspian",
    re.IGNORECASE,
)

# Per-entity coverage regex (collision-free): an entity's drainage destination is "determined" when
# its OWN name sits within a short window of a drainage/status word (caspian / black sea / kura /
# poti / drains to / flows into / ...) and that window does not cross ANY other river's name --
# so a grouped "Alazani, Aragvi, Iori and Khrami all drain to the Kura" credits only the name
# adjacent to the status word, never a neighbour.
_STATUS = (
    r"\b(?:caspian|black\s+sea|kura|poti|tributary|"
    r"drains?\s+to|flows?\s+into|empties?\s+into|outflow|mouth)\b"
)
for _e in ENTITIES:
    _others_rx = "|".join(o["name_rx"] for o in ENTITIES if o is not _e)
    _e["cov_rx"] = re.compile(
        _e["name_rx"] + r"(?:(?!" + _others_rx + r")[^.;]){0,55}" + _STATUS
        + r"|" + _STATUS + r"(?:(?!" + _others_rx + r")[^.;]){0,55}" + _e["name_rx"],
        re.IGNORECASE,
    )


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "080",
        "test_name": "Tier 5: Odd-one-out / negation variant B (Caucasus river drainage)",
        "difficulty_level": "9/10",
        "category": "Odd-one-out / negation (binary categorical exception)",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs -- search to find each river's Wikipedia page and READ it (do not "
        "guess from memory). Below are five rivers of the South Caucasus / Georgia that all APPEAR "
        "to share a single binary hydrographic property: draining to the Caspian Sea (most of the "
        "region's rivers drain east to the Caspian via the Kura River). In fact exactly FOUR of "
        "them DO drain to the Caspian Sea and exactly ONE does NOT.\n\n"
        f"{listing}\n\n"
        "For EACH of the five rivers, open its Wikipedia page and determine from the lead sentence "
        "and/or infobox whether the river ultimately drains to the Caspian Sea or whether it drains "
        "to a different sea. Then identify the SINGLE river that does NOT share the property -- "
        "i.e. the one river among the five that does NOT drain to the Caspian Sea (the one that "
        "drains to a different sea). Be careful with the wording: you are asked for the river that "
        "does NOT drain to the Caspian Sea, NOT one of the four that do; and do NOT pick the most "
        "Caspian-looking river -- pick the one that actually drains elsewhere.\n\n"
        "Report (a) which ONE river is the odd one out -- the river that does NOT drain to the "
        "Caspian Sea (a single river name -- the keystone); (b) WHY it is the exception (the sea "
        "it actually drains to); (c) for EACH of the five, whether it drains to the Caspian Sea "
        "or not; and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which ONE river is the odd one out -- the one that does NOT drain to the Caspian Sea "
        "(the primary answer / keystone)",
        "WHY that river is the exception (the sea it actually drains to)",
        "For each of the five rivers, whether it drains to the Caspian Sea or not",
        "Source URL for each river's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per river; a five-way fan-out is the target)",
        "Correctly names Rioni as the odd one out -- the ONE river that does NOT drain to the "
        "Caspian Sea (NOT one of the four Caspian tributaries: Alazani, Aragvi, Iori, Khrami), "
        "and does NOT claim Rioni drains to the Caspian",
        "States WHY Rioni is the exception (it drains to the Black Sea / flows to Poti / is not "
        "a Caspian tributary)",
        "Determines the drainage destination of all five rivers",
        "Cites the source pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text. Prefer ``deliverables[0]`` when present; else fall back to output."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: final deliverable plus every deliverable slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE gate (negation-robust): deliverables[0] names Rioni as the river that does NOT
    drain to the Caspian Sea.

      * Rioni must be named.
      * It must NOT be asserted to drain to the Caspian Sea (the negation-flip decoy "Rioni drains
        to the Caspian" names the right entity but assigns the WRONG property value -> no credit).
      * If only Rioni is named in the primary slot (a terse correct answer), that passes.
      * If one or more of the four Caspian rivers is also named, (a) none of them may be asserted
        as the odd-one-out / exception, and (b) Rioni must be the one tied to the Black Sea /
        not-Caspian / exception assertion (window not crossing another river's name).
    """
    text = _primary_text(result)
    if not re.search(_RIONI, text, re.IGNORECASE):
        return False
    if _RIONI_CASP.search(text):                  # asserts Rioni drains to Caspian -> flip
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True                               # only Rioni named, not called Caspian
    if _RIVAL_ODD.search(text):                   # a Caspian river named as the exception
        return False
    return bool(_RIONI_IS_ODD.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a five-way fan-out wants one page per river."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per river; >=4 to pass)"}


def validate_keystone_odd_one_out(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the odd one out is Rioni -- the ONE river that does NOT drain to the
    Caspian Sea (it flows west to the Black Sea at Poti). A model that flips the negation and names
    a Caspian river (Alazani / Aragvi / Iori / Khrami), or that names Rioni but asserts Rioni drains
    to the Caspian, gets no credit."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_odd_one_out",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Rioni named as the one that does NOT drain to the Caspian Sea (the odd one out)"
            if passed else
            "Odd-one-out (Rioni, the only non-Caspian river) missing/incorrect (beware the "
            "negation: the answer is the river that does NOT drain to the Caspian, not one of "
            "the four Caspian tributaries; and 'Rioni drains to the Caspian' is the wrong "
            "property value)"
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: how many of the FIVE rivers had their drainage destination
    DETERMINED. A river is credited only when its OWN name appears within a short window of a
    drainage/status word (caspian / black sea / kura / poti / drains to / ...) that does not cross
    another river's name -- collision-free. Deliberately NOT gated on the keystone: it measures
    whether the agent fanned out and wrote each river's drainage even when it botches the negation,
    the axis that separates a structured (multi-leaf) agent from a linear one that drops an entity
    or lumps them all as 'Caspian'."""
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES if e["cov_rx"].search(text)]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} rivers' drainage destinations determined "
            f"({', '.join(hits) or 'none'})"
        ),
    }


def validate_why_exception(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: states WHY Rioni is the exception -- the property value that differs (it
    drains to the Black Sea / flows to Poti / is not a Caspian tributary). Short-circuits to 0
    when the keystone is absent, so a wrong/guessed odd-one-out cannot bank the reason credit."""
    if not _keystone_ok(result):
        return {"check": "why_exception", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> reason (why it is the exception) not credited"}
    ok = bool(_WHY.search(_all_text(result)))
    return {
        "check": "why_exception",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"reason present (Rioni drains to the {ODD_SEA} / Poti / not Caspian)" if ok
            else f"no reason given (expected Rioni drains to the {ODD_SEA} / flows to Poti / "
                 "not a Caspian tributary)"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the river pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} river pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_odd_one_out,
        validate_coverage,
        validate_why_exception,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    FIVE INDEPENDENT parallel leaves -- one per GIVEN river -- each determining ONE river's
    drainage destination with an IDENTICAL, neutral instruction (does it drain to the Caspian, or
    to a different sea?). The negation is owned entirely by the aggregation step, which is forced
    to WRITE OUT each river's drainage destination (Caspian Sea / other) BEFORE naming the single
    one that fails the property -- so the cheap executor cannot flip "which does NOT" into "which
    DOES" and cannot drop an entity. Encodes STRUCTURE only: it names the five GIVEN rivers, but
    leaks NOTHING about which one is the exception -- every leaf is the same shape and the per-river
    yes/no determination is computed from the leaf results, not pre-supplied.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_drainage",
            "instruction": (
                f"Open the Wikipedia page for the {e['name']} River (a river of the South Caucasus "
                f"/ Georgia region) and determine, from the lead sentence and the infobox, whether "
                f"the {e['name']} ultimately drains to the CASPIAN SEA (e.g. via the Kura River) "
                f"or whether it drains to a DIFFERENT sea (e.g. the Black Sea). Answer strictly "
                f"with 'Caspian Sea' or 'not Caspian Sea (drains to <sea>)' and give the source "
                f"URL. Do not guess from memory."
            ),
            "expect": (
                f"'Caspian Sea' OR 'not Caspian Sea (drains to <sea>)' -- source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five rivers, a determination of whether it ultimately "
            "drains to the Caspian Sea or to a different sea. FIRST, write out each river's "
            "drainage destination explicitly, one per line, in the form '<river>: Caspian Sea' or "
            "'<river>: NOT Caspian Sea -- drains to <sea>'. Determine all FIVE before drawing any "
            "conclusion. The shared property is 'drains to the Caspian Sea'; exactly four rivers "
            "satisfy it and exactly ONE does NOT. Now identify the SINGLE river that does NOT "
            "satisfy the property -- the one that does NOT drain to the Caspian Sea (it drains to "
            "a different sea) -- that river's name is the keystone answer. Be careful with the "
            "NEGATION: the answer is the river that does NOT drain to the Caspian, not one of the "
            "four that do, and you must NOT describe that river as draining to the Caspian. "
            "Report (a) that river (the odd one out), (b) WHY it is the exception (the sea it "
            "actually drains to), (c) all five rivers' drainage destinations, and (d) cite each "
            "river's source URL."
        ),
    }
