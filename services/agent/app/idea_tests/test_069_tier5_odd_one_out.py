"""
Test 069: Tier 5 (integration) — ODD-ONE-OUT / NEGATION (the shared-property exception).
Level: integration   Weight: long   Difficulty: 9/10

A negation-trap task built to punish two cheap-model failure modes at once: (1) flipping "which
does NOT satisfy the property" into "which DOES" (the negation flip), and (2) failing to fan out
across all five entities so the lone, non-obvious exception is missed. The agent is given FIVE
seemingly-similar Central/Southeast-European countries that all APPEAR to share one binary
geographic property -- being LANDLOCKED (no sea coastline). Exactly FOUR genuinely are landlocked;
exactly ONE is NOT. The agent must identify the SINGLE country that does NOT satisfy the property:

    Austria    Bosnia and Herzegovina    North Macedonia    Serbia    Slovakia

The property is binary and UNAMBIGUOUS on Wikipedia, with TWO authoritative cross-checks:
  * each landlocked country's lead sentence states "is a landlocked country"; the exception's lead
    states it has a sea coastline; and
  * the English Wikipedia "List of landlocked countries" INCLUDES all four satisfiers and EXCLUDES
    the exception.
There is no judgement call -- a country is landlocked or it is not.

The exception is SOLID-BUT-NOT-ICONIC and is the OPPOSITE of guessable from fame: Bosnia and
Herzegovina is the textbook "almost-landlocked" country -- overwhelmingly mountainous and
near-universally (mis)remembered as landlocked -- yet it is NOT landlocked: it has a 20 km coastline
on the Adriatic Sea at the Neum corridor (the town of Neum). The famous "landlocked" exemplars among
the five are Austria and Slovakia, so a fame/stereotype shortcut lands on a LANDLOCKED country, never
on Bosnia. Two further traps are baked in:
  * SLOV-CONFUSION: the landlocked "Slov-" country (Slovakia) is PRESENT, while the famously-coastal
    "Slov-" country (Slovenia) is deliberately ABSENT -- a model that half-remembers "a Slov country
    has a coast" mis-assigns the coast to Slovakia (which is landlocked) and mis-picks.
  * NO ALTERNATIVE SINGLE-OUT: there is no competing clean 4-1 split -- EU 2-3, eurozone 2-3,
    NATO 2-3, ex-Yugoslavia 3-2 -- so the ONLY property that singles out exactly one country is
    "is landlocked", and it singles out Bosnia. The agent cannot back into the answer via some other
    categorical accident; it must read the pages.

Why it discriminates (per REASONING_TEST_DESIGN.md -- the differential-lift target):
  * cheap native (parametric / graph): flips the negation and names a LANDLOCKED country (Austria /
    Slovakia the obvious ones), or "knows" Bosnia is mountainous and assumes it is landlocked, or
    drops an entity and never checks Bosnia -> wrong odd-one-out.
  * frontier sequential (ReAct): checks each lead, recalls/confirms Bosnia's 20 km Neum coast,
    handles the negation -> decent.
  * graph_compiled: five neutral parallel leaves each determine ONE country's landlocked status; the
    aggregation owns the negation -- it is forced to WRITE OUT each country's status (landlocked /
    not landlocked) BEFORE naming the single one that fails the property -> the cheap executor is
    rescued from both the flip and the dropped-entity failure.

The trap is engineered two ways:
  (1) NEGATION: the question asks for the one that does NOT satisfy "is landlocked"; a model that
      reads "landlocked" and answers the most-landlocked-seeming country inverts the task.
  (2) NON-OBVIOUS EXCEPTION: the exception (Bosnia) is the LEAST-expected coastal country of the
      five -- the textbook "almost-landlocked" state; fame/stereotype points AWAY from it, so the
      model must actually read the pages.

Ground truth (verified against live English Wikipedia 2026-06-27 -- each country's lead sentence /
infobox, cross-checked against the "List of landlocked countries"; the property "is landlocked" is
binary and explicitly stated):

  country                  landlocked?   evidence (live lead / infobox / list)
  Austria                     YES        "Austria ... is a landlocked country in Central Europe"
  Bosnia and Herzegovina      NO         "... a 20-kilometre-long (12-mile) coast on the Adriatic
                                          Sea in the south" (Neum); EXCLUDED from the landlocked list  <- ODD ONE OUT
  North Macedonia             YES        "North Macedonia ... is a landlocked country in Southeast Europe"
  Serbia                      YES        "Serbia ... is a landlocked country in Southeast and Central Europe"
  Slovakia                    YES        "Slovakia ... is a landlocked country in Central Europe"
    https://en.wikipedia.org/wiki/Austria
    https://en.wikipedia.org/wiki/Bosnia_and_Herzegovina
    https://en.wikipedia.org/wiki/North_Macedonia
    https://en.wikipedia.org/wiki/Serbia
    https://en.wikipedia.org/wiki/Slovakia

  ROBUSTNESS: EXACTLY ONE country (Bosnia and Herzegovina) violates the shared property "is
  landlocked"; the other four clearly satisfy it (each has zero coastline, its lead explicitly says
  "is a landlocked country", and each is listed in the "List of landlocked countries"). The property
  is binary and unambiguous -- no metro-vs-city-proper or disputed-status judgement calls. A single
  noisy read cannot flip it: there is no near-landlocked ambiguity among the four, and Bosnia's coast
  is explicitly stated (a 20 km Adriatic coast at Neum) and confirmed by its EXCLUSION from the
  landlocked list.

  KEYSTONE = the odd-one-out ENTITY: Bosnia and Herzegovina, identified as the one that does NOT
  satisfy "is landlocked" (i.e. the one that HAS a sea coast). The keystone is robust to the
  negation trap: it is NOT credited when a model names one of the four landlocked countries, and --
  critically -- it is NOT credited when a model names Bosnia but asserts Bosnia DOES satisfy the
  property (i.e. claims "Bosnia is landlocked"). Secondary (gated) = correctly states WHY it is the
  exception (its Adriatic/Neum coastline / that it is not landlocked). Coverage (un-gated,
  collision-free) = how many of the five had their landlocked status determined.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``landlocked`` is shown for provenance only; nothing about which country is the exception is
# leaked into the task statement or the compiled plan.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "austria", "name": "Austria", "landlocked": True, "odd": False,
     "name_rx": r"\baustria\b", "slug_rx": r"wiki/austria"},
    {"key": "bosnia", "name": "Bosnia and Herzegovina", "landlocked": False, "odd": True,
     "name_rx": r"\bbosnia\b", "slug_rx": r"wiki/bosnia", "sea": "Adriatic Sea"},
    {"key": "north_macedonia", "name": "North Macedonia", "landlocked": True, "odd": False,
     "name_rx": r"\bmacedonia\b", "slug_rx": r"wiki/north_macedonia"},
    {"key": "serbia", "name": "Serbia", "landlocked": True, "odd": False,
     "name_rx": r"\bserbia\b", "slug_rx": r"wiki/serbia"},
    {"key": "slovakia", "name": "Slovakia", "landlocked": True, "odd": False,
     "name_rx": r"\bslovakia\b", "slug_rx": r"wiki/slovakia"},
]

ODD = next(e for e in ENTITIES if e["odd"])              # Bosnia -- the not-landlocked exception
ODD_SEA = ODD["sea"]                                     # "Adriatic Sea"

# ----- keystone regex machinery (negation-robust) ---------------------------------------------
_BOSNIA = r"\b(?:bosnia|bih)\b"
_OTHERS = r"\b(?:austria|slovakia|serbia|macedonia)\b"     # the four landlocked countries

# Contrast / clause-pivot words: when one of these sits between two subjects, an attribute on one
# side does NOT belong to the subject on the other side ("X is landlocked, WHEREAS Bosnia ...").
_BREAK = r"\b(?:than|unlike|while|whilst|whereas|but|however|except|yet|though|although)\b"
# Negators: block an affirmative reading of a following property word ("not / never landlocked").
_NEG = r"(?:\bnot\b|\bno\b|\bnever\b|isn'?t|aren'?t|\bn't\b)"
# Group/aggregate references: "the rest / the others / all four are landlocked" -- the landlocked
# claim attaches to the GROUP, not to a nearby Bosnia mention.
_GROUP = r"\b(?:rest|others?|remaining|four|all|each)\b"

# VETO -- "Bosnia IS landlocked" asserted affirmatively (the negation-flip decoy that names the
# right entity but assigns it the WRONG property value). Single direction (Bosnia -> landlocked);
# the short window is forbidden from crossing a negator, a contrast pivot, a group reference or any
# other country name -- so "Bosnia is NOT landlocked", "Bosnia ... whereas the others are
# landlocked", and "Austria is landlocked" can never trip it, while "Bosnia is landlocked" and
# "Bosnia is the odd one out because it is landlocked" do.
_BOS_LL = re.compile(
    _BOSNIA + r"(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _GROUP + r"|" + _OTHERS
    + r")[^.;\n]){0,40}\blandlocked\b",
    re.IGNORECASE,
)

# Bosnia tied to a "not-landlocked / has-a-coast / is-the-exception" assertion. Either direction,
# window forbidden from crossing into another country's name (so a coast attribute reported for one
# of the four cannot be borrowed by a nearby Bosnia mention).
_NOTLL = (
    r"coast(?:al|line|s)?|seacoast|seaboard|adriatic|neum|\bsea\b|maritime|"
    r"not\s+landlocked|isn'?t\s+landlocked|access\s+to\s+the\s+sea|"
    r"odd\s+one|outlier|exception|doesn'?t\s+belong|does\s+not\s+belong|"
    r"breaks?\s+the\s+pattern|stands?\s+out"
)
_BOSNIA_IS_ODD = re.compile(
    _BOSNIA + r"(?:(?!" + _OTHERS + r")[^.;\n]){0,80}\b(?:" + _NOTLL + r")\b"
    + r"|\b(?:" + _NOTLL + r")\b(?:(?!" + _OTHERS + r")[^.;\n]){0,80}" + _BOSNIA,
    re.IGNORECASE,
)

# RIVAL VETO -- one of the FOUR landlocked countries asserted as the odd-one-out / the exception
# (a wrong-entity answer that also happens to mention Bosnia). Window forbidden from crossing a
# negator, "no", a contrast pivot or Bosnia, so a correct contrastive phrasing ("Austria has no
# coastline", "the four are landlocked, whereas Bosnia is the exception") never trips it.
_RIVAL_MARK = (
    r"odd\s+one|outlier|exception|doesn'?t\s+belong|does\s+not\s+belong|"
    r"breaks?\s+the\s+pattern|not\s+landlocked|isn'?t\s+landlocked|"
    r"has\s+a\s+coast|coastal|coastline"
)
_RIVAL_ODD = re.compile(
    _OTHERS + r"(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _BOSNIA + r")[^.;\n]){0,60}\b(?:" + _RIVAL_MARK + r")\b"
    + r"|\b(?:" + _RIVAL_MARK + r")\b(?:(?!" + _NEG + r"|" + _BREAK + r"|" + _BOSNIA + r")[^.;\n]){0,60}" + _OTHERS,
    re.IGNORECASE,
)

# WHY (gated secondary): the property value that differs -- Bosnia has a sea coastline / is not
# landlocked (Adriatic / Neum / 20 km).
_WHY = re.compile(
    r"adriatic|neum|coast(?:al|line|s)?|seaboard|\bsea\b|maritime|"
    r"not\s+landlocked|isn'?t\s+landlocked|20\s*-?\s*(?:km|kilomet)|access\s+to\s+the\s+sea",
    re.IGNORECASE,
)

# Per-entity coverage regex (collision-free): an entity's landlocked status is "determined" when its
# OWN name sits within a short window of a status word (landlocked / coast / sea / Adriatic / Neum)
# and that window does not cross ANY other country's name -- so a grouped "Austria, Serbia, Slovakia
# are landlocked" credits only the name adjacent to the status word, never a neighbour.
_STATUS = r"\b(?:landlocked|coast(?:al|line|s)?|adriatic|neum|sea|maritime|seaboard)\b"
for _e in ENTITIES:
    _others_rx = "|".join(o["name_rx"] for o in ENTITIES if o is not _e)
    _e["cov_rx"] = re.compile(
        _e["name_rx"] + r"(?:(?!" + _others_rx + r")[^.;\n]){0,55}" + _STATUS
        + r"|" + _STATUS + r"(?:(?!" + _others_rx + r")[^.;\n]){0,55}" + _e["name_rx"],
        re.IGNORECASE,
    )


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "069",
        "test_name": "Tier 5: Odd-one-out / negation (the shared-property exception)",
        "difficulty_level": "9/10",
        "category": "Odd-one-out / negation (binary categorical exception)",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs -- search to find each country's Wikipedia page and READ it (do not "
        "guess from memory). Below are five European countries that all APPEAR to share a single "
        "binary geographic property: being LANDLOCKED (having NO sea coastline). In fact exactly "
        "FOUR of them ARE landlocked and exactly ONE is NOT.\n\n"
        f"{listing}\n\n"
        "For EACH of the five countries, open its Wikipedia page and determine from the lead "
        "sentence / infobox whether it is a landlocked country or whether it has a sea coastline. "
        "Then identify the SINGLE country that does NOT share the property -- i.e. the one country "
        "among the five that is NOT landlocked (the one that has a sea coastline). Be careful with "
        "the wording: you are asked for the country that does NOT satisfy 'is landlocked', NOT one "
        "of the landlocked four; and do NOT pick the most landlocked-looking country -- pick the one "
        "that actually has a coastline.\n\n"
        "Report (a) which ONE country is the odd one out -- the country that is NOT landlocked / "
        "does NOT satisfy the shared property (a single country name -- the keystone); (b) WHY it is "
        "the exception (the sea / coastline it borders); (c) for EACH of the five, whether it is "
        "landlocked or not; and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which ONE country is the odd one out -- the one that is NOT landlocked / does NOT satisfy "
        "the shared property (the primary answer / keystone)",
        "WHY that country is the exception (the sea / coastline it borders, i.e. it is not landlocked)",
        "For each of the five countries, whether it is landlocked or not",
        "Source URL for each country's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per country; a five-way fan-out is the target)",
        "Correctly names Bosnia and Herzegovina as the odd one out -- the ONE country that is NOT "
        "landlocked (NOT one of the four landlocked countries: Austria, North Macedonia, Serbia, "
        "Slovakia), and does NOT claim Bosnia is landlocked",
        "States WHY Bosnia is the exception (its Adriatic Sea / Neum coastline / that it is not "
        "landlocked)",
        "Determines the landlocked status of all five countries",
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
    """Full reported text: the final deliverable plus every deliverable slot, so coverage / why /
    citation checks can see content the agent placed outside the primary answer slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate (negation-robust): deliverables[0] names Bosnia and Herzegovina as the country
    that does NOT satisfy the shared property "is landlocked".

      * Bosnia must be named.
      * It must NOT be asserted to BE landlocked (the negation-flip decoy "Bosnia is landlocked"
        names the right entity but assigns the WRONG property value -> no credit).
      * If only Bosnia is named in the primary slot (a terse correct answer), that passes.
      * If one or more of the four landlocked countries is also named, (a) none of them may be
        asserted as the odd-one-out / exception, and (b) Bosnia must be the one tied to the
        not-landlocked / has-a-coast / is-the-exception assertion (window not crossing another
        country's name).

    Credit requires GROUNDING: the value string alone is insufficient — the agent must have
    actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess would earn credit.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _primary_text(result)
    if not re.search(_BOSNIA, text, re.IGNORECASE):
        return False
    if _BOS_LL.search(text):                        # asserts Bosnia IS landlocked -> negation flip
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True                                 # only Bosnia named, not called landlocked
    if _RIVAL_ODD.search(text):                     # a landlocked country named as the exception
        return False
    return bool(_BOSNIA_IS_ODD.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a five-way fan-out wants one page per country."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per country; >=4 to pass)"}


def validate_keystone_odd_one_out(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the odd one out is Bosnia and Herzegovina -- the ONE country that does
    NOT satisfy the shared property "is landlocked" (it has a 20 km Adriatic coast at Neum). A model
    that flips the negation and names a landlocked country (Austria / North Macedonia / Serbia /
    Slovakia), or that names Bosnia but asserts Bosnia IS landlocked, gets no credit."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_odd_one_out", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Bosnia and Herzegovina named as the one that is NOT landlocked (the odd one "
                      "out)" if passed
                      else "Odd-one-out (Bosnia and Herzegovina, the only non-landlocked country) "
                           "missing/incorrect (beware the negation: the answer is the country that "
                           "is NOT landlocked, not one of the four landlocked countries; and 'Bosnia "
                           "is landlocked' is the wrong property value)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: how many of the FIVE countries had their landlocked status
    DETERMINED. A country is credited only when its OWN name appears within a short window of a
    status word (landlocked / coast / sea / Adriatic / Neum) that does not cross another country's
    name -- collision-free. Deliberately NOT gated on the keystone: it measures whether the agent
    fanned out and wrote each country's status even when it botches the negation, the axis that
    separates a structured (multi-leaf) agent from a linear one that drops an entity or lumps them
    all as 'landlocked'."""
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES if e["cov_rx"].search(text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} countries' landlocked status determined "
                      f"({', '.join(hits) or 'none'})"}


def validate_why_exception(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: states WHY Bosnia is the exception -- the property value that differs (its
    Adriatic / Neum coastline / that it is not landlocked). Short-circuits to 0 when the keystone is
    absent, so a wrong/guessed odd-one-out cannot bank the reason credit."""
    if not _keystone_ok(result, observability):
        return {"check": "why_exception", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> reason (why it is the exception) not credited"}
    ok = bool(_WHY.search(_all_text(result)))
    return {"check": "why_exception", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"reason present (Bosnia's {ODD_SEA} / Neum coastline / not landlocked)" if ok
                       else f"no reason given (expected Bosnia's {ODD_SEA} / Neum coastline / not "
                            "landlocked)")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the country pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} country pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_odd_one_out,
        validate_coverage,
        validate_why_exception,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), as in 056/059.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    FIVE INDEPENDENT parallel leaves -- one per GIVEN country -- each determining ONE country's
    landlocked status with an IDENTICAL, neutral instruction (is it landlocked, or does it have a
    sea coastline?). The negation is owned entirely by the aggregation step, which is forced to
    WRITE OUT each country's status (landlocked / not landlocked) BEFORE naming the single one that
    fails the property -- so the cheap executor cannot flip "which does NOT" into "which DOES" and
    cannot drop an entity. Encodes STRUCTURE only: it names the five GIVEN countries, but leaks
    NOTHING about which one is the exception -- every leaf is the same shape and the per-country
    yes/no determination is computed from the leaf results, not pre-supplied.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_status",
            "instruction": (
                f"Open the Wikipedia page for {e['name']} and determine, from the lead sentence and "
                f"the infobox, whether {e['name']} is a LANDLOCKED country (it has NO sea coastline) "
                f"or whether it HAS a sea coastline. Answer strictly with '{e['name']}: landlocked' or "
                f"'{e['name']}: not landlocked (coastline on <sea>)' -- ALWAYS repeat the country name "
                f"'{e['name']}' at the start of your answer so the fact is self-contained even if read "
                "out of context. Give the source URL. Do not guess from memory."
            ),
            "expect": f"'{e['name']}: landlocked' OR '{e['name']}: not landlocked (coastline on <sea>)' "
                      "-- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five countries, a determination of whether it is "
            "landlocked or has a sea coastline. FIRST, write out each country's status explicitly, "
            "one per line, in the form '<country>: landlocked' or '<country>: NOT landlocked -- "
            "coastline on <sea>'. Determine all FIVE before drawing any conclusion. The shared "
            "property is 'is landlocked'; exactly four countries satisfy it and exactly ONE does "
            "NOT. Now identify the SINGLE country that does NOT satisfy the property -- the one that "
            "is NOT landlocked (it has a sea coastline) -- that country's name is the keystone "
            "answer. Be careful with the NEGATION: the answer is the country that is NOT landlocked, "
            "not one of the landlocked four, and you must NOT describe that country as landlocked. "
            "Report (a) that country (the odd one out), (b) WHY it is the exception (the sea / "
            "coastline that makes it not landlocked), (c) all five countries' landlocked status, "
            "and (d) cite each country's source URL."
        ),
    }
