"""
Test 080: Tier 5 (integration) — ODD-ONE-OUT / NEGATION (the shared-property exception), variant B.
Level: integration   Weight: long   Difficulty: 9/10

A negation-trap task built to punish two cheap-model failure modes at once: (1) flipping "which
does NOT empty into the Gulf of Papua" into "which DOES" (the negation flip), and (2) failing to
fan out across all five rivers so the lone, non-obvious exception is missed. The agent is given
FIVE rivers of the island of New Guinea (all in Papua New Guinea) that all APPEAR to share one
binary hydrographic property -- emptying into the GULF OF PAPUA (the great bay of the Coral Sea on
New Guinea's SOUTH coast, into which the region's best-known rivers drain). Exactly FOUR genuinely
do; exactly ONE does NOT. The agent must identify the SINGLE river that does NOT satisfy the
property:

    Fly    Kikori    Purari    Sepik    Turama

The property is binary and UNAMBIGUOUS on Wikipedia, with TWO authoritative cross-checks:
  * each Gulf-of-Papua river's Wikipedia infobox shows "Mouth: Gulf of Papua"; the exception's
    infobox shows "Mouth: Bismarck Sea" and its lead sentence states it flows to the Bismarck Sea
    off NORTHERN Papua New Guinea; and
  * the exception's lead ("flows ... to the Bismarck Sea off northern Papua New Guinea") confirms it
    reaches the sea on the OPPOSITE (north) coast, near Wewak -- not the Gulf of Papua at all.
There is no judgement call -- a river empties into the Gulf of Papua or it does not.

Why the exception is GENUINELY NON-PARAMETRIC (the fix over the retired Rioni->Black Sea variant):
New Guinea river hydrography is deeply obscure. Almost no reader -- and few models parametrically --
knows which specific sea each Papuan river empties into; there is no "everyone knows river X drains
to sea Y" prior to fall back on, so the odd-one-out cannot be guessed without opening the pages.
Worse for the guesser, the exception is the Sepik -- the LONGEST and most-prominent river of New
Guinea -- so any fame/prominence shortcut lumps it with the region's dominant southern outlet (the
Gulf of Papua) and lands on the WRONG answer: the prior points TOWARD the Sepik being a satisfier,
AWAY from it being the exception. The four true satisfiers (Fly, Kikori, Purari, Turama) are all
obscure south-coast rivers whose "Mouth: Gulf of Papua" is a page-only infobox fact. Two further
traps are baked in:
  * SAME-COUNTRY TRAP: all five rivers are rivers of Papua New Guinea on the same island, so the
    geographic setting gives no hint that one drains to a different sea on a different coast.
  * NO ALTERNATIVE SINGLE-OUT: there is no clean competing 4-1 split by fame or length -- the Fly
    (2nd-longest, famous) is a satisfier, so "the famous one" does not cleanly single out the Sepik.
    The ONLY clean 4-1 split is "empties into the Gulf of Papua," and it singles out the Sepik.

Why it discriminates (per REASONING_TEST_DESIGN.md -- the differential-lift target):
  * cheap native (parametric / graph): has no PNG-river prior, so it defaults "Gulf of Papua" for
    all five (the dominant outlet), or flips the negation and names one of the four Gulf rivers, or
    drops the Sepik and never checks it -> wrong odd-one-out.
  * frontier sequential (ReAct): reads each infobox, spots the Sepik's northward Bismarck-Sea mouth,
    handles the negation -> decent.
  * graph_compiled: five neutral parallel leaves each determine ONE river's mouth/outlet sea; the
    aggregation owns the negation -- it is forced to WRITE OUT each river's outlet BEFORE naming the
    single one that fails the property -> the cheap executor is rescued from both the flip and the
    dropped-entity failure.

The trap is engineered two ways:
  (1) NEGATION: the question asks for the one that does NOT empty into the Gulf of Papua; a model
      that reads "Gulf of Papua" and names the most Gulf-looking river inverts the task.
  (2) NON-OBVIOUS EXCEPTION: the exception (Sepik) is the LEAST-expected non-Gulf river of the five
      -- New Guinea's most prominent river, which prominence points TOWARD the dominant southern
      outlet; fame/stereotype points AWAY from picking the Sepik as the exception, so the model must
      read the page (its mouth is on the OPPOSITE, northern coast).

Ground truth (verified against live English Wikipedia 2026-07-07 -- each river's infobox "Mouth"
field / lead sentence; the property "empties into the Gulf of Papua" is binary and explicitly
stated):

  river     empties into Gulf of Papua?   evidence (live lead / infobox)
  Fly           YES              infobox: Mouth = Gulf of Papua, Papua New Guinea (south coast)
  Kikori        YES              infobox: Mouth = Gulf of Papua; lead: "flows southeast into the Gulf of Papua"
  Purari        YES              infobox: Mouth = Gulf of Papua, Papua New Guinea; lead: "to the Gulf of Papua"
  Turama        YES              infobox: Mouth = Gulf of Papua, Papua New Guinea (head of the Gulf)
  Sepik         NO               infobox: Mouth = Bismarck Sea; lead: "flows ... to the Bismarck Sea off
                                 northern Papua New Guinea" (north coast, near Wewak)   <- ODD ONE OUT
    https://en.wikipedia.org/wiki/Fly_River
    https://en.wikipedia.org/wiki/Kikori_River
    https://en.wikipedia.org/wiki/Purari_River
    https://en.wikipedia.org/wiki/Turama_River
    https://en.wikipedia.org/wiki/Sepik

  ROBUSTNESS: EXACTLY ONE river (Sepik) violates the shared property "empties into the Gulf of
  Papua"; the other four clearly satisfy it (each Wikipedia infobox literally lists "Mouth: Gulf of
  Papua"). The property is binary and unambiguous -- no partial drainage, no disputed outflow. A
  single noisy read cannot flip it: there is no near-Gulf ambiguity among the four satisfiers, and
  the Sepik's Bismarck-Sea mouth is stated in both its infobox and lead sentence and confirmed by
  its north-coast location (off Wewak) -- the OPPOSITE side of the island.

  KEYSTONE = the odd-one-out ENTITY: Sepik, identified as the one that does NOT empty into the Gulf
  of Papua (i.e. the one that empties into the Bismarck Sea on the north coast). The keystone is
  robust to the negation trap: it is NOT credited when a model names one of the four Gulf-draining
  rivers, and -- critically -- it is NOT credited when a model names the Sepik but asserts it DOES
  empty into the Gulf of Papua (i.e. the negation-flip decoy). Secondary (gated) = correctly states
  WHY it is the exception (it empties into the Bismarck Sea / on the northern coast). Coverage
  (un-gated, collision-free) = how many of the five had their outlet sea determined.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``gulf`` is shown for provenance only; nothing about which river is the exception is leaked into
# the task statement or the compiled plan.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "fly", "name": "Fly", "gulf": True, "odd": False,
     "name_rx": r"\bfly\b", "slug_rx": r"wiki/fly_river"},
    {"key": "kikori", "name": "Kikori", "gulf": True, "odd": False,
     "name_rx": r"\bkikori\b", "slug_rx": r"wiki/kikori"},
    {"key": "purari", "name": "Purari", "gulf": True, "odd": False,
     "name_rx": r"\bpurari\b", "slug_rx": r"wiki/purari"},
    {"key": "turama", "name": "Turama", "gulf": True, "odd": False,
     "name_rx": r"\bturama\b", "slug_rx": r"wiki/turama"},
    {"key": "sepik", "name": "Sepik", "gulf": False, "odd": True,
     "name_rx": r"\bsepik\b", "slug_rx": r"wiki/sepik", "sea": "Bismarck Sea"},
]

ODD = next(e for e in ENTITIES if e["odd"])    # Sepik -- the not-Gulf-of-Papua exception
ODD_SEA = ODD["sea"]                            # "Bismarck Sea"

# ----- keystone regex machinery (negation-robust) ----------------------------------------
_SEPIK = r"\bsepik\b"
_OTHERS = r"\b(?:fly|kikori|purari|turama)\b"   # the four Gulf-of-Papua rivers

# Contrast / clause-pivot words: when one of these sits between two subjects, an attribute on one
# side does NOT belong to the subject on the other side.
_BREAK = r"\b(?:than|unlike|while|whilst|whereas|but|however|except|yet|though|although)\b"
# Negators: block an affirmative reading of a following property word.
_NEG = r"(?:\bnot\b|\bno\b|\bnever\b|isn'?t|aren'?t|\bn't\b)"
# Group/aggregate references: "the rest / all four empty into the Gulf" attaches the claim to the
# GROUP, not to a nearby Sepik mention.
_GROUP = r"\b(?:rest|others?|remaining|four|all|each)\b"
# The shared-property phrase (the satisfier value): empties into the Gulf of Papua / Coral Sea.
_GULF = r"(?:gulf\s+of\s+papua|coral\s+sea)"

# VETO -- "Sepik empties into the Gulf of Papua" asserted affirmatively (the negation-flip decoy
# that names the right entity but assigns it the WRONG property value). Window forbidden from
# crossing a negator, a contrast pivot, a group reference or another river name.
_SEPIK_GULF = re.compile(
    _SEPIK + r"(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _GROUP + r"|" + _OTHERS
    + r")[^.;\n]){0,50}" + _GULF,
    re.IGNORECASE,
)

# Sepik tied to a "Bismarck Sea / north coast / not-Gulf / is-the-exception" assertion. Either
# direction, window forbidden from crossing into another river's name (so a mouth attribute reported
# for one of the four cannot be borrowed by a nearby Sepik mention).
_NOTGULF = (
    r"bismarck|north(?:ern)?\s+coast|wewak|flows?\s+north|drains?\s+north|"
    r"not\s+the\s+gulf|isn'?t\s+the\s+gulf|doesn'?t\s+empty\s+into\s+the\s+gulf|"
    r"odd\s+one|outlier|exception|doesn'?t\s+belong|does\s+not\s+belong|"
    r"breaks?\s+the\s+pattern|stands?\s+out"
)
_SEPIK_IS_ODD = re.compile(
    _SEPIK + r"(?:(?!" + _OTHERS + r")[^.;\n]){0,80}\b(?:" + _NOTGULF + r")\b"
    + r"|\b(?:" + _NOTGULF + r")\b(?:(?!" + _OTHERS + r")[^.;\n]){0,80}" + _SEPIK,
    re.IGNORECASE,
)

# RIVAL VETO -- one of the FOUR Gulf rivers asserted as the odd-one-out / the exception (a
# wrong-entity answer). Window forbidden from crossing a negator, a contrast pivot or the Sepik
# name, so a correct contrastive phrasing ("the four empty into the Gulf, whereas the Sepik goes to
# the Bismarck Sea") never trips it, but "the Fly is the exception; it drains to the Bismarck Sea"
# does.
_RIVAL_MARK = (
    r"odd\s+one|outlier|exception|doesn'?t\s+belong|does\s+not\s+belong|"
    r"breaks?\s+the\s+pattern|bismarck|north(?:ern)?\s+coast|wewak"
)
_RIVAL_ODD = re.compile(
    _OTHERS + r"(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _SEPIK + r")[^.;\n]){0,60}\b(?:" + _RIVAL_MARK + r")\b"
    + r"|\b(?:" + _RIVAL_MARK + r")\b(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _SEPIK + r")[^.;\n]){0,60}" + _OTHERS,
    re.IGNORECASE,
)

# WHY (gated secondary): the property value that differs -- Sepik empties into the Bismarck Sea (not
# the Gulf of Papua); on the northern coast near Wewak.
_WHY = re.compile(
    r"bismarck|north(?:ern)?\s+coast|wewak|flows?\s+north|drains?\s+north|"
    r"not\s+the\s+gulf|isn'?t\s+the\s+gulf",
    re.IGNORECASE,
)

# Per-entity coverage regex (collision-free): an entity's outlet is "determined" when its OWN name
# sits within a short window of a drainage/status word (gulf of papua / coral sea / bismarck /
# mouth / empties into / flows into / delta / ...) and that window does not cross ANY other river's
# name -- so a grouped "Fly, Kikori, Purari and Turama all empty into the Gulf of Papua" credits
# only the name adjacent to the status word, never a neighbour.
_STATUS = (
    r"\b(?:gulf\s+of\s+papua|coral\s+sea|bismarck|mouth|delta|outflow|outlet|"
    r"empties?\s+into|drains?\s+(?:in)?to|flows?\s+(?:in)?to|north\s+coast|south\s+coast)\b"
)
for _e in ENTITIES:
    _others_rx = "|".join(o["name_rx"] for o in ENTITIES if o is not _e)
    _e["cov_rx"] = re.compile(
        _e["name_rx"] + r"(?:(?!" + _others_rx + r")[^.;\n]){0,55}" + _STATUS
        + r"|" + _STATUS + r"(?:(?!" + _others_rx + r")[^.;\n]){0,55}" + _e["name_rx"],
        re.IGNORECASE,
    )


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "080",
        "test_name": "Tier 5: Odd-one-out / negation variant B (New Guinea river drainage)",
        "difficulty_level": "9/10",
        "category": "Odd-one-out / negation (binary categorical exception)",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs -- search to find each river's Wikipedia page and READ it (do not "
        "guess from memory). Below are five rivers of the island of New Guinea (all in Papua New "
        "Guinea) that all APPEAR to share a single binary hydrographic property: emptying into the "
        "Gulf of Papua (the great bay of the Coral Sea on New Guinea's SOUTH coast, into which the "
        "region's best-known rivers drain). In fact exactly FOUR of them DO empty into the Gulf of "
        "Papua and exactly ONE does NOT.\n\n"
        f"{listing}\n\n"
        "For EACH of the five rivers, open its Wikipedia page and determine from the lead sentence "
        "and/or the infobox 'Mouth' field which sea or gulf the river empties into -- specifically "
        "whether it empties into the Gulf of Papua or into a DIFFERENT sea (on a different coast). "
        "Then identify the SINGLE river that does NOT share the property -- i.e. the one river among "
        "the five that does NOT empty into the Gulf of Papua (the one that reaches a different sea). "
        "Be careful with the wording: you are asked for the river that does NOT empty into the Gulf "
        "of Papua, NOT one of the four that do; and do NOT pick the most prominent / Gulf-looking "
        "river -- pick the one that actually reaches a different sea.\n\n"
        "Report (a) which ONE river is the odd one out -- the river that does NOT empty into the "
        "Gulf of Papua (a single river name -- the keystone); (b) WHY it is the exception (the sea "
        "it actually empties into); (c) for EACH of the five, which sea/gulf it empties into; and "
        "(d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which ONE river is the odd one out -- the one that does NOT empty into the Gulf of Papua "
        "(the primary answer / keystone)",
        "WHY that river is the exception (the sea it actually empties into)",
        "For each of the five rivers, which sea/gulf it empties into",
        "Source URL for each river's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per river; a five-way fan-out is the target)",
        "Correctly names Sepik as the odd one out -- the ONE river that does NOT empty into the "
        "Gulf of Papua (NOT one of the four Gulf-of-Papua rivers: Fly, Kikori, Purari, Turama), "
        "and does NOT claim the Sepik empties into the Gulf of Papua",
        "States WHY the Sepik is the exception (it empties into the Bismarck Sea / on the northern "
        "coast / not into the Gulf of Papua)",
        "Determines the outlet sea of all five rivers",
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
    """KEYSTONE gate (negation-robust): deliverables[0] names Sepik as the river that does NOT empty
    into the Gulf of Papua.

      * Sepik must be named.
      * It must NOT be asserted to empty into the Gulf of Papua (the negation-flip decoy "Sepik
        empties into the Gulf of Papua" names the right entity but assigns the WRONG property value
        -> no credit).
      * If only Sepik is named in the primary slot (a terse correct answer), that passes.
      * If one or more of the four Gulf rivers is also named, (a) none of them may be asserted as
        the odd-one-out / exception, and (b) Sepik must be the one tied to the Bismarck Sea /
        north-coast / not-Gulf / exception assertion (window not crossing another river's name).
    """
    text = _primary_text(result)
    if not re.search(_SEPIK, text, re.IGNORECASE):
        return False
    if _SEPIK_GULF.search(text):                  # asserts Sepik empties into the Gulf -> flip
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True                               # only Sepik named, not called Gulf-of-Papua
    if _RIVAL_ODD.search(text):                   # a Gulf river named as the exception
        return False
    return bool(_SEPIK_IS_ODD.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a five-way fan-out wants one page per river."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per river; >=4 to pass)"}


def validate_keystone_odd_one_out(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the odd one out is Sepik -- the ONE river that does NOT empty into the
    Gulf of Papua (it empties into the Bismarck Sea off the northern coast, near Wewak). A model that
    flips the negation and names a Gulf-of-Papua river (Fly / Kikori / Purari / Turama), or that
    names the Sepik but asserts the Sepik empties into the Gulf of Papua, gets no credit."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_odd_one_out",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Sepik named as the one that does NOT empty into the Gulf of Papua (the odd one out)"
            if passed else
            "Odd-one-out (Sepik, the only non-Gulf-of-Papua river) missing/incorrect (beware the "
            "negation: the answer is the river that does NOT empty into the Gulf of Papua, not one "
            "of the four Gulf rivers; and 'Sepik empties into the Gulf of Papua' is the wrong "
            "property value)"
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: how many of the FIVE rivers had their outlet sea DETERMINED. A
    river is credited only when its OWN name appears within a short window of a drainage/status word
    (gulf of papua / coral sea / bismarck / mouth / empties into / ...) that does not cross another
    river's name -- collision-free. Deliberately NOT gated on the keystone: it measures whether the
    agent fanned out and wrote each river's outlet even when it botches the negation, the axis that
    separates a structured (multi-leaf) agent from a linear one that drops an entity or lumps them
    all as 'Gulf of Papua'."""
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES if e["cov_rx"].search(text)]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} rivers' outlet seas determined "
            f"({', '.join(hits) or 'none'})"
        ),
    }


def validate_why_exception(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: states WHY the Sepik is the exception -- the property value that differs (it
    empties into the Bismarck Sea / on the northern coast / near Wewak / not into the Gulf of
    Papua). Short-circuits to 0 when the keystone is absent, so a wrong/guessed odd-one-out cannot
    bank the reason credit."""
    if not _keystone_ok(result):
        return {"check": "why_exception", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> reason (why it is the exception) not credited"}
    ok = bool(_WHY.search(_all_text(result)))
    return {
        "check": "why_exception",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"reason present (Sepik empties into the {ODD_SEA} / northern coast / not the Gulf)"
            if ok else
            f"no reason given (expected Sepik empties into the {ODD_SEA} / on the northern coast / "
            "not into the Gulf of Papua)"
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

    FIVE INDEPENDENT parallel leaves -- one per GIVEN river -- each determining ONE river's outlet
    sea with an IDENTICAL, neutral instruction (which sea does it empty into?). The negation is
    owned entirely by the aggregation step, which is forced to WRITE OUT each river's outlet sea
    BEFORE naming the single one that fails the property -- so the cheap executor cannot flip "which
    does NOT" into "which DOES" and cannot drop an entity. Encodes STRUCTURE only: it names the five
    GIVEN rivers, but leaks NOTHING about which one is the exception -- every leaf is the same shape
    and the per-river yes/no determination is computed from the leaf results, not pre-supplied. Each
    leaf is SELF-DESCRIBING: it repeats the river's name at the start of its answer so the fact
    survives the aggregation's stripping of leaf IDs.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_outlet",
            "instruction": (
                f"Open the Wikipedia page for the {e['name']} River (a river of the island of New "
                f"Guinea, in Papua New Guinea) and determine, from the lead sentence and the infobox "
                f"'Mouth' field, which sea or gulf the {e['name']} empties into -- specifically "
                f"whether it empties into the GULF OF PAPUA (on New Guinea's south coast) or into a "
                f"DIFFERENT sea (on a different coast). Answer strictly with "
                f"'{e['name']}: Gulf of Papua' or '{e['name']}: <other sea> (NOT the Gulf of "
                f"Papua)' -- ALWAYS repeat the river name '{e['name']}' at the start of your answer "
                f"so the fact is self-contained even if read out of context. Give the source URL. "
                f"Do not guess from memory."
            ),
            "expect": (
                f"'{e['name']}: Gulf of Papua' OR '{e['name']}: <other sea> (NOT the Gulf of "
                f"Papua)' -- source URL"
            ),
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five rivers, a determination of which sea or gulf it "
            "empties into. FIRST, write out each river's outlet explicitly, one per line, in the "
            "form '<river>: Gulf of Papua' or '<river>: NOT the Gulf of Papua -- empties into "
            "<sea>'. Determine all FIVE before drawing any conclusion. The shared property is "
            "'empties into the Gulf of Papua'; exactly four rivers satisfy it and exactly ONE does "
            "NOT. Now identify the SINGLE river that does NOT satisfy the property -- the one that "
            "does NOT empty into the Gulf of Papua (it reaches a different sea) -- that river's name "
            "is the keystone answer. Be careful with the NEGATION: the answer is the river that does "
            "NOT empty into the Gulf of Papua, not one of the four that do, and you must NOT describe "
            "that river as emptying into the Gulf of Papua. Report (a) that river (the odd one out), "
            "(b) WHY it is the exception (the sea it actually empties into), (c) all five rivers' "
            "outlet seas, and (d) cite each river's source URL."
        ),
    }
