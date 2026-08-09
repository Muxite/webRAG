"""
Test 068: Tier 5 (graph) — MULTI-CONSTRAINT AND-FILTER (three-way set intersection).
Level: graph   Weight: long   Difficulty: 9/10

A hard set-intersection task built to punish the cheap-model failure mode of dropping a constraint
(or short-circuiting on a single famous attribute) instead of holding THREE independent lookups in
mind and intersecting them. Among SIX given European countries the agent must look up THREE
attributes per country and name the UNIQUE country that satisfies ALL THREE constraints at once:

    (A) is LANDLOCKED, AND
    (B) has a total population OVER 4 MILLION, AND
    (C) uses the EURO as its official currency.

The six countries are SOLID-BUT-NOT-ICONIC (no France / Germany / Italy / Spain / Switzerland) and
are engineered so that EACH single constraint is satisfied by SEVERAL of them — so dropping ANY ONE
constraint changes the answer set — yet EXACTLY ONE country satisfies all three. The keystone is the
LEAST-headline member of the set, and the conjunction "landlocked & >4M & euro" is not a memorizable
fact, so a parametric / fame shortcut cannot recover it; the agent must actually READ all three
attributes for each country and intersect.

    Slovakia        Czech Republic   Hungary
    Portugal        Greece           Kosovo

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): collapses the three-way filter to one attribute or drops a constraint and
    mis-picks (the biggest landlocked country -> Czech Republic; a big euro country -> Portugal /
    Greece; a landlocked euro user too small -> Kosovo); or guesses a more famous name from memory.
  * frontier sequential (ReAct): looks up all eighteen attribute values, applies the three filters,
    returns the single survivor -> decent.
  * graph_compiled: eighteen parallel leaves each fetch ONE attribute for ONE country; the
    aggregation owns the entire AND-filter and is forced to WRITE OUT each country's
    constraint-by-constraint check before concluding -> the cheap executor is rescued and a
    diverse-grounding reranker can re-derive the survivor and catch a slip.

The trap is engineered three ways — EACH constraint is necessary (drop one => wrong, larger set):
  (1) drop LANDLOCKED  -> {Slovakia, Portugal, Greece}  (>4M & euro)   -> attractors Portugal/Greece
  (2) drop POPULATION  -> {Slovakia, Kosovo}             (landlocked & euro) -> attractor Kosovo
  (3) drop CURRENCY    -> {Slovakia, Czech Republic, Hungary} (landlocked & >4M) -> attractor Czech
  => the ONLY country in all three sets is Slovakia. No single attribute, and no pair of attributes,
     isolates it; the full three-way conjunction is required.

Ground truth (verified against live English Wikipedia 2026-06-27 — each country's infobox + lead;
landlocked status is stated verbatim in the lead, population is the infobox figure, currency is the
infobox 'Currency' field):

  country           landlocked   population (infobox)          currency        all-three?
  Slovakia          YES          5,449,270 (2021 census)       Euro            YES  <- KEYSTONE
  Czech Republic    YES          10,915,839 (2025 est)         Czech koruna    no  (fails euro)
  Hungary           YES          9,603,634 (2022 census)       Forint          no  (fails euro)
  Portugal          NO (Atlantic) 11,424,031 (2025 est)        Euro            no  (fails landlocked)
  Greece            NO (Med.)    10,372,335 (2025 est)         Euro            no  (fails landlocked)
  Kosovo            YES          1,585,566 (2024 census)       Euro            no  (fails population)
    https://en.wikipedia.org/wiki/Slovakia
    https://en.wikipedia.org/wiki/Czech_Republic
    https://en.wikipedia.org/wiki/Hungary
    https://en.wikipedia.org/wiki/Portugal
    https://en.wikipedia.org/wiki/Greece
    https://en.wikipedia.org/wiki/Kosovo

  KEYSTONE = Slovakia (the unique country that is landlocked AND over 4 million AND uses the euro).

  EACH SINGLE CONSTRAINT IS SATISFIED BY MORE THAN ONE COUNTRY (so the conjunction is necessary):
    landlocked          : Slovakia, Czech Republic, Hungary, Kosovo        (4 of 6)
    population > 4M      : Slovakia, Czech Republic, Hungary, Portugal, Greece (5 of 6)
    currency = euro      : Slovakia, Portugal, Greece, Kosovo              (4 of 6)
  No single filter, and no two filters, leave Slovakia alone — only all three do.

  ROBUSTNESS / MARGINS (no borderline value where a small misread flips membership):
    * LANDLOCKED is categorical and stated verbatim: Slovakia/Czech/Hungary/Kosovo "landlocked";
      Portugal on the Atlantic, Greece on the Mediterranean — no ambiguity.
    * POPULATION threshold = 4,000,000. The members straddle it with a HUGE gap: Kosovo
      ~1.59M (2.41M below) vs Slovakia ~5.45M (1.45M above, ~36%); every other member is 9.6M-11.4M.
      No country lies between 1.6M and 5.4M, so no plausible misread moves anyone across 4M.
    * CURRENCY is categorical: Slovakia/Portugal/Greece/Kosovo use the Euro (Kosovo adopted it de
      facto; this is stated in its Wikipedia infobox); the Czech Republic uses the Czech koruna and
      Hungary uses the forint — distinct, infobox-stated, unambiguous.

  ANTI-PARAMETRIC: Slovakia is NOT the population extreme (Portugal 11.4M and Czech 10.9M are
  larger), NOT the only landlocked member, and NOT the only euro user; it is purely the intersection.
  The Kosovo decoy (landlocked + euro but <4M) is less parametrically obvious than Luxembourg because
  Kosovo's euro usage is informal/de facto and less widely known. More-recognizable members (Greece,
  Portugal, the Czech Republic) are decoys that each miss exactly one constraint, so a fame /
  single-attribute guess mis-picks. The keystone is recoverable only by reading and intersecting all
  three attributes.

  KEYSTONE = the unique satisfying ENTITY (Slovakia). Secondary (gated) = correctly states Slovakia's
  three attribute values (landlocked, population ~5.4M, euro). Coverage (un-gated) = how many of the
  six countries had all three attributes gathered; the six population figures are pairwise distinct,
  so the coverage diagnostic is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# Per-entity regexes are case-insensitive where applied. ``pop_rx`` matches either the full infobox
# integer (comma/space tolerant, digit-bounded) OR a rounded "X.Y million" form; the six ranges are
# pairwise disjoint so coverage attribution can never cross between countries. Constraint booleans
# (``landlocked``/``pop_over_4m``/``euro``) are shown for provenance and to DERIVE the winner; nothing
# numeric here is leaked into the task statement or the compiled plan.
POP_THRESHOLD = 4_000_000

ENTITIES: List[Dict[str, Any]] = [
    {"key": "slovakia", "name": "Slovakia",
     "landlocked": True, "population": 5_449_270, "pop_over_4m": True, "euro": True,
     "name_rx": r"\bslovak", "slug_rx": r"wiki/slovakia",
     "pop_rx": r"(?<!\d)5[,\s]?4\d{2}[,\s]?\d{3}(?!\d)|\b5\.4\d?\s*million\b",
     "currency_rx": r"\beuro\b|€|\beur\b", "geo_rx": r"landlock"},
    {"key": "czech", "name": "Czech Republic",
     "landlocked": True, "population": 10_915_839, "pop_over_4m": True, "euro": False,
     "name_rx": r"\bczech|\bczechia", "slug_rx": r"wiki/czech",
     "pop_rx": r"(?<!\d)10[,\s]?[59]\d{2}[,\s]?\d{3}(?!\d)|\b10\.[59]\d?\s*million\b",
     "currency_rx": r"\bkoruna\b|\bczk\b", "geo_rx": r"landlock"},
    {"key": "hungary", "name": "Hungary",
     "landlocked": True, "population": 9_603_634, "pop_over_4m": True, "euro": False,
     "name_rx": r"\bhungar", "slug_rx": r"wiki/hungary",
     "pop_rx": r"(?<!\d)9[,\s]?[456]\d{2}[,\s]?\d{3}(?!\d)|\b9\.[456]\d?\s*million\b",
     "currency_rx": r"\bforint\b|\bhuf\b", "geo_rx": r"landlock"},
    {"key": "portugal", "name": "Portugal",
     "landlocked": False, "population": 11_424_031, "pop_over_4m": True, "euro": True,
     "name_rx": r"\bportug", "slug_rx": r"wiki/portugal",
     "pop_rx": r"(?<!\d)11[,\s]?4\d{2}[,\s]?\d{3}(?!\d)|\b11\.4\d?\s*million\b",
     "currency_rx": r"\beuro\b|€|\beur\b",
     "geo_rx": r"coast|atlantic|ocean|\bsea\b"},
    {"key": "greece", "name": "Greece",
     "landlocked": False, "population": 10_372_335, "pop_over_4m": True, "euro": True,
     "name_rx": r"\bgreec|\bgreek", "slug_rx": r"wiki/greece",
     "pop_rx": r"(?<!\d)10[,\s]?[34]\d{2}[,\s]?\d{3}(?!\d)|\b10\.[34]\d?\s*million\b",
     "currency_rx": r"\beuro\b|€|\beur\b",
     "geo_rx": r"coast|mediterran|aegean|ionian|ocean|\bsea\b"},
    {"key": "kosovo", "name": "Kosovo",
     "landlocked": True, "population": 1_585_566, "pop_over_4m": False, "euro": True,
     "name_rx": r"\bkosov", "slug_rx": r"wiki/kosovo",
     "pop_rx": r"(?<!\d)1[,\s]?[45678]\d{2}[,\s]?\d{3}(?!\d)|\b1\.[4-7]\d?\s*million\b",
     "currency_rx": r"\beuro\b|€|\beur\b", "geo_rx": r"landlock"},
]

# Derive the keystone purely from the three constraint booleans (a self-check that exactly one
# country is landlocked AND over 4M AND euro). If the fixtures ever drift, this raises at import.
_SATISFIERS = [e for e in ENTITIES if e["landlocked"] and e["pop_over_4m"] and e["euro"]]
assert len(_SATISFIERS) == 1, f"AND-filter must have exactly one survivor, got {[e['name'] for e in _SATISFIERS]}"
WINNER = _SATISFIERS[0]                                  # Slovakia — the unique satisfier

# Winner / other-entity name regexes used by the keystone gate.
_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if e is not WINNER)

# AND-filter "winner" assertion triggers: words that assert a country is THE unique answer to the
# three-way intersection ("the only one", "satisfies all three", "the answer is ...", "qualifies").
# Comparative/superlative words are deliberately excluded (this is a set-membership task, not an
# argmax). "single" is excluded to avoid matching the EU "single market".
_SAT = (r"only|sole|solely|unique|uniquely|all three|all 3|"
        r"satisfies all|meets all|fulfil|qualif|answer|sought|is the country")

# Keystone winner detection: the winner (Slovakia) tied to an "is the only / satisfies all three /
# is the answer" assertion, in either direction, with the proximity window forbidden from crossing
# into ANY other country's name (so "the Czech Republic is the answer; Slovakia also uses the euro"
# can never satisfy it). The window is [^.;] — newline-tolerant (a header line then the answer below
# still matches) but bounded at sentence periods AND clause-separating semicolons.
#   dir 1  (subject -> assertion):  "Slovakia ... is the only one that satisfies all three"
#   dir 2  (assertion -> subject):  "the only country meeting all three is Slovakia"
_SLOVAKIA_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SAT + r")\b"
    + r"|\b(?:" + _SAT + r")\b(?:(?!" + _OTHERS + r")[^.;]){0,90}" + _WINNER_RX,
    re.IGNORECASE,
)

# Pattern that flags a non-winner country tied to an "is the answer / satisfies all" assertion.
# Used by _keystone_enumeration_ok to catch "Czech Republic is the only one ..." etc.
_OTHER_WINS = re.compile(
    r"(?:" + _OTHERS + r")(?:(?!" + _WINNER_RX + r")[^.;]){0,90}\b(?:" + _SAT + r")\b"
    + r"|\b(?:" + _SAT + r")\b(?:(?!" + _WINNER_RX + r")[^.;]){0,90}(?:" + _OTHERS + r")",
    re.IGNORECASE,
)


def _keystone_enumeration_ok(text: str) -> bool:
    """Credits enumeration-style answers where Slovakia's row shows all three constraints
    satisfied (landlocked=yes, euro present, >=2 'yes' tokens) but no explicit assertion
    word ('only', 'unique', 'satisfies all', …) appears near its name.

    This handles aggregation outputs formatted as:
        Slovakia: landlocked=yes, population=5,449,270 (yes), currency=euro (yes) — URL
        Czech Republic: landlocked=yes, population=10,700,000 (yes), currency=Czech koruna (no)
        …

    Guards:
      • Slovakia's row must show landlocked=yes AND contain 'euro' AND have >=2 'yes' tokens.
      • No other country's row may show the same three-way positive (guards against a hallucinated
        table where a decoy is also credited all-yes).
      • Rejected outright if any non-winner country is tied to an assertion word (e.g. "Czech
        Republic is the only one that satisfies all three").
    """
    # NOTE: name_rx values may contain alternation (e.g. r"\bczech|\bczechia").  When used as the
    # *prefix* of a larger regex they must be wrapped in (?:…) so the alternation is grouped and
    # doesn't swallow the suffix pattern.
    _other_pat = "|".join(e["name_rx"] for e in ENTITIES if e is not WINNER)

    # --- 1. Find Slovakia's row (from its name to the next country-name line or blank line). ---
    sk_m = re.search(
        r"(?:" + _WINNER_RX + r")[^\n]*(?:\n(?!(?:" + _other_pat + r"))[^\n]*){0,3}",
        text, re.IGNORECASE,
    )
    if not sk_m:
        return False
    sk_row = sk_m.group(0)

    # --- 2. Slovakia's row must show all three constraints as positive. ---
    if not re.search(r"landlocked[=:\s]+yes", sk_row, re.IGNORECASE):
        return False
    if not re.search(r"\beuro\b|€|\beur\b", sk_row, re.IGNORECASE):
        return False
    if len(re.findall(r"\byes\b", sk_row, re.IGNORECASE)) < 2:
        return False

    # --- 3. No other country's row may show all three as positive. ---
    for e in ENTITIES:
        if e is WINNER:
            continue
        other_m = re.search(
            r"(?:" + e["name_rx"] + r")[^\n]*(?:\n(?!(?:" + _WINNER_RX + r"|" + _other_pat + r"))[^\n]*){0,2}",
            text, re.IGNORECASE,
        )
        if not other_m:
            continue
        other_row = other_m.group(0)
        if (re.search(r"landlocked[=:\s]+yes", other_row, re.IGNORECASE)
                and re.search(r"\beuro\b|€|\beur\b", other_row, re.IGNORECASE)
                and len(re.findall(r"\byes\b", other_row, re.IGNORECASE)) >= 3):
            return False  # hallucinated table: a decoy also shows all-three positive

    # --- 4. Reject if any non-winner is explicitly asserted as "the answer / only / unique". ---
    if _OTHER_WINS.search(text):
        return False

    return True


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "068",
        "test_name": "Tier 5: Multi-constraint AND-filter (three-way set intersection)",
        "difficulty_level": "9/10",
        "category": "Multi-constraint set intersection (AND-filter)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']}" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following six countries, open the country's Wikipedia page "
        "and read THREE attributes: (A) whether the country is LANDLOCKED (has no sea coastline); "
        "(B) its total POPULATION from the infobox; and (C) its official CURRENCY from the "
        "infobox:\n"
        f"{listing}\n\n"
        "Then determine the SINGLE country that satisfies ALL THREE of the following constraints "
        "simultaneously:\n"
        "  (A) it is LANDLOCKED, AND\n"
        "  (B) its total population is OVER 4 MILLION, AND\n"
        "  (C) its official currency is the EURO.\n\n"
        "Exactly one of the six countries satisfies all three. Note: several countries satisfy each "
        "constraint on its own, so you must apply all three together — dropping any one constraint "
        "would give a different, larger answer. The winner is NOT necessarily the largest country, "
        "nor the only landlocked one, nor the only euro user.\n\n"
        "Report (a) which single country satisfies all three constraints (one country name — the "
        "keystone), (b) that country's three attribute values (landlocked status, population, and "
        "currency), (c) for all six countries the three attributes you gathered, and (d) the exact "
        "source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which single country is landlocked AND has population over 4 million AND uses the euro "
        "(the primary answer / keystone)",
        "That winning country's three attribute values (landlocked status, population, currency)",
        "For all six countries, the three attributes gathered (landlocked status, population, "
        "currency)",
        "Source URL for each country's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 6 pages visited (one per country, a six-way fan-out)",
        "Correctly names Slovakia as the unique country satisfying all three constraints (NOT the "
        "Czech Republic, the largest landlocked country; NOT Portugal or Greece, big euro countries "
        "with coastlines; NOT Kosovo, a landlocked euro user under 4 million)",
        "States the winner's three attributes: landlocked, population ~5.4 million, currency euro",
        "Gathers all three attributes for each of the six countries",
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
    citation checks can see attributes the agent placed outside the primary answer slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names Slovakia as the unique all-three satisfier.

    Three acceptance paths (any one suffices):
      1. Terse: only Slovakia is named (no other country present) — assumed to be the answer.
      2. Explicit: Slovakia is tied to an 'is the only / satisfies all three / is the answer'
         assertion within a window that never crosses another country's name (so naming a decoy
         as the answer while mentioning Slovakia cannot pass).
      3. Enumeration: Slovakia's row shows landlocked=yes, euro, and >=2 'yes' tokens while no
         other country's row shows all three as positive (handles aggregation-style output where
         the model writes a per-country attribute table without using assertion words near Slovakia).

    Rejected outright if Slovakia is absent, or if another country is explicitly asserted as the
    answer, or if Slovakia's row shows a wrong attribute.

    Credit requires GROUNDING: the value string alone is insufficient — the agent must have
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
        return True  # path 1: only the winner is named
    if _SLOVAKIA_WINS.search(text):
        return True  # path 2: explicit assertion near Slovakia
    return _keystone_enumeration_ok(text)  # path 3: enumeration table


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per country."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 6.0),
            "reason": f"{n} visit(s) (target >=6: one page per country; >=4 to pass)"}


def validate_keystone_filter(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the unique country that is landlocked AND over 4 million AND uses the
    euro is Slovakia. A model that drops a constraint mis-picks (Czech Republic = largest landlocked;
    Portugal/Greece = big euro countries with coastlines; Kosovo = landlocked euro user under
    4 million), as does a fame / single-attribute shortcut."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_filter", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Slovakia named as the unique all-three satisfier" if passed
                      else "Unique all-three satisfier (Slovakia) missing/incorrect (beware: Czech "
                           "Republic is the largest landlocked country; Portugal/Greece are big euro "
                           "countries with coastlines; Kosovo is a landlocked euro user under 4M)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: for how many of the SIX countries were all three
    attributes gathered.

    A country is credited only when its name AND its (pairwise-distinct) population figure AND its
    currency token AND a geography indicator (landlocked, or a coastline/sea term for the two
    coastal members) all appear. The population figure pins the credit to the specific country, so
    the shared 'euro'/'landlocked' tokens can never cross-credit between countries. Deliberately NOT
    gated on the keystone — it measures whether the agent actually fanned out to all six countries
    and pulled each one's three attributes even when it botches the intersection, the axis that
    separates a structured (multi-leaf) agent from a linear one that drops an entity.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["pop_rx"], text, re.IGNORECASE)
            and re.search(e["currency_rx"], text, re.IGNORECASE)
            and re.search(e["geo_rx"], text, re.IGNORECASE)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} countries' three attributes gathered ({', '.join(hits) or 'none'})"}


def validate_winner_attributes(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winner's three attribute values are stated — landlocked, a population
    figure ~5.4M (over the 4M threshold), and the euro. Short-circuits to 0 when the keystone is
    absent, so a wrong/guessed winner can't bank the value credit."""
    if not _keystone_ok(result, observability):
        return {"check": "winner_attributes", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> attribute values not credited"}
    text = _all_text(result)
    checks = {
        "landlocked": bool(re.search(WINNER["geo_rx"], text, re.IGNORECASE)),
        "population": bool(re.search(WINNER["pop_rx"], text, re.IGNORECASE)),
        "euro": bool(re.search(WINNER["currency_rx"], text, re.IGNORECASE)),
    }
    got = sum(checks.values())
    missing = [k for k, v in checks.items() if not v]
    return {"check": "winner_attributes", "passed": got == 3, "score": got / 3.0,
            "reason": (f"all three of Slovakia's attributes stated (landlocked, ~5.4M, euro)" if got == 3
                       else f"{got}/3 of Slovakia's attributes stated; missing: {', '.join(missing)}")}


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
        validate_keystone_filter,
        validate_coverage,
        validate_winner_attributes,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), as in 054/055.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    EIGHTEEN INDEPENDENT parallel leaves: for each of the six GIVEN countries, one leaf reads its
    landlocked status, one reads its infobox population, and one reads its infobox currency — one
    atomic attribute per leaf. ALL filtering — applying the three constraints and intersecting them —
    lives only in the aggregation step, which is forced to WRITE OUT each country's
    constraint-by-constraint check explicitly before concluding, so the cheap executor never drops a
    constraint or short-circuits to a single famous attribute, and a diverse-grounding reranker can
    re-derive the survivor. Encodes STRUCTURE only: it names the six GIVEN countries and the three
    GIVEN constraints (landlocked, population over 4 million, euro), but leaks no attribute value,
    no population figure, and not which country wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_landlocked",
            "instruction": (
                f"Open the Wikipedia page for {e['name']} and determine, from the lead/geography, "
                f"whether {e['name']} is LANDLOCKED (has no sea coastline) or has a coastline. "
                "Report ONLY 'landlocked' or 'has a coastline' and the source URL. Do not guess "
                "from memory."
            ),
            "expect": "LANDLOCKED or HAS A COASTLINE -- source URL",
            "depends_on": [],
        })
        leaves.append({
            "id": f"{e['key']}_population",
            "instruction": (
                f"Open the Wikipedia page for {e['name']} and read its total POPULATION from the "
                "infobox. Report ONLY that single population number and the source URL. Do not "
                "guess from memory."
            ),
            "expect": "POPULATION (a single number) -- source URL",
            "depends_on": [],
        })
        leaves.append({
            "id": f"{e['key']}_currency",
            "instruction": (
                f"Open the Wikipedia page for {e['name']} and read its official CURRENCY from the "
                "infobox. Report ONLY the currency name and the source URL. Do not guess from "
                "memory."
            ),
            "expect": "CURRENCY (the official currency name) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six countries, three attributes: whether it is "
            "landlocked, its total population, and its official currency. For EACH of the six "
            "countries, write out its check explicitly on its own line in the form "
            "'<country>: landlocked=<yes/no>, population=<value> (over 4 million? yes/no), "
            "currency=<value> (euro? yes/no)' — evaluate all six BEFORE drawing any conclusion. "
            "THEN identify the SINGLE country for which ALL THREE are yes (landlocked AND population "
            "over 4 million AND currency is the euro) — that country's name is the keystone answer. "
            "Show every country's three-way check before concluding. Exactly one country satisfies "
            "all three; it need not be the largest country, the only landlocked one, or the only "
            "euro user. Report (a) that country and its three attribute values, (b) all six "
            "countries' three attributes, and (c) cite each country's source URL."
        ),
    }
