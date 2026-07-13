"""
Test 059: Tier 5 (graph) — argmax over a COMPUTED RATIO (goals per appearance).
Level: graph   Weight: long   Difficulty: 9/10

A hard quantitative-reasoning task built to punish the cheap-model shortcut of "pick the biggest
raw number". Among FIVE retired international forwards the agent must determine which has the
highest goals-per-appearance ratio, where the ratio is NOT stated on any page and must be COMPUTED
by DIVIDING two looked-up infobox integers:

    ratio = total senior international goals  /  total senior international appearances (caps)

The five entities are deliberately SOLID-BUT-NOT-ICONIC internationals (not Pelé / Müller /
Charlton / Romário / Eusébio), each with a SINGLE senior national team (so the infobox carries one
unambiguous "caps (goals)" row — no Total-row summing of a second nation). The ratio winner is the
LEAST-famous member, so the comparison is NOT a memorized "fact": a parametric model that never
opens the pages cannot recall it and a fame/most-goals shortcut lands on the wrong man.

    Jan Koller (Czech Republic)   Miroslav Klose (Germany)   Robbie Keane (Republic of Ireland)
    Jon Dahl Tomasson (Denmark)   Henrik Larsson (Sweden)

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): drops an entity, mis-divides, or shortcuts to the biggest raw quantity
    (most goals -> Klose, or most caps -> Keane, or inverts the ratio to caps/goals -> Larsson) and
    names the WRONG player; or simply guesses the most famous name (Klose) from memory.
  * frontier sequential (ReAct): looks up all ten integers, divides, compares -> decent.
  * graph_compiled: ten parallel leaves each fetch ONE integer (one quantity for one player); the
    aggregation owns every division AND the argmax, and is forced to WRITE OUT each division before
    concluding -> the cheap executor is rescued and a diverse-grounding reranker can re-derive it.

The trap is engineered three ways:
  (1) the ratio is page-only — no infobox states goals-per-game, so the agent MUST divide;
  (2) the ratio ranking DIFFERS from BOTH raw rankings — the most-goals player (Klose) and the
      most-caps player (Keane) are BOTH wrong, so "pick the biggest number" fails either way;
  (3) inverting the ratio (caps/goals, dividing by the wrong quantity) maximizes at Larsson, the
      least efficient scorer — so a wrong-base division also mis-picks.

Ground truth (verified against live English Wikipedia 2026-06-26 — each player's infobox senior
'National team' row "caps (goals)"; every one of the five played for exactly ONE senior nation, so
the single row is unambiguous):

  player              country                goals (num)  caps (den)   ratio = goals/caps
  Jan Koller          Czech Republic              55           91          0.6044   <- ARGMAX (keystone)
  Miroslav Klose      Germany                     71          137          0.5182   <- most GOALS (decoy)
  Robbie Keane        Republic of Ireland         68          146          0.4658   <- most CAPS  (decoy)
  Jon Dahl Tomasson   Denmark                     52          112          0.4643
  Henrik Larsson      Sweden                      37          106          0.3491
    https://en.wikipedia.org/wiki/Jan_Koller
    https://en.wikipedia.org/wiki/Miroslav_Klose
    https://en.wikipedia.org/wiki/Robbie_Keane
    https://en.wikipedia.org/wiki/Jon_Dahl_Tomasson
    https://en.wikipedia.org/wiki/Henrik_Larsson

  RATIO ARGMAX = Jan Koller (0.6044). Runner-up = Klose (0.5182), so the margin is +0.086 absolute
  (~16.6% relative). A one-unit misread on either side cannot flip it: Koller's worst plausible
  slip (54/91 = 0.593 or 55/92 = 0.598) still tops Klose's best (72/137 = 0.526), so one noisy
  extraction cannot change the keystone.

  RANKING DIVERGENCE (confirmed live):
    by goals (numerator):  Klose 71 > Keane 68 > Koller 55 > Tomasson 52 > Larsson 37   -> Klose
    by caps (denominator): Keane 146 > Klose 137 > Tomasson 112 > Larsson 106 > Koller 91 -> Keane
    by ratio:              Koller 0.604 > Klose 0.518 > Keane 0.466 > Tomasson 0.464 > Larsson 0.349
    inverted (caps/goals): Larsson 2.865 > Tomasson 2.154 > Keane 2.147 > Klose 1.930 > Koller 1.655
    => the ratio winner (Koller) is NEITHER the most-goals player (Klose) NOR the most-caps player
       (Keane); he in fact has the FEWEST caps. "Pick the biggest raw quantity" is wrong on both
       axes and the caps/goals inversion mis-picks Larsson, so a wrong-base division also fails.

  ANTI-PARAMETRIC: Klose (all-time World Cup top scorer) is the most-famous member and the obvious
  guess, but he is the most-goals DECOY. The keystone is Koller — the least-famous member, whose
  goals-per-cap superiority over these four is not a recallable fact — so the cheap model must
  actually READ the pages and DIVIDE rather than pattern-match a name from memory.

  KEYSTONE = the argmax ENTITY (Jan Koller). Secondary (gated) value = his ratio ~0.60 (within
  +/- 3%). All ten integers are distinct, so the un-gated coverage diagnostic is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``ratio`` is shown for provenance only; nothing here is leaked into the task statement or plan.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "koller", "name": "Jan Koller", "country": "Czech Republic",
     "goals": 55, "caps": 91, "ratio": 0.6044, "winner": True,
     "name_rx": r"\bkoller\b", "goals_rx": r"(?<!\d)55(?!\d)",
     "caps_rx": r"(?<!\d)91(?!\d)", "slug_rx": r"wiki/jan_koller"},
    {"key": "klose", "name": "Miroslav Klose", "country": "Germany",
     "goals": 71, "caps": 137, "ratio": 0.5182, "winner": False,
     "name_rx": r"\bklose\b", "goals_rx": r"(?<!\d)71(?!\d)",
     "caps_rx": r"(?<!\d)137(?!\d)", "slug_rx": r"wiki/miroslav_klose"},
    {"key": "keane", "name": "Robbie Keane", "country": "Republic of Ireland",
     "goals": 68, "caps": 146, "ratio": 0.4658, "winner": False,
     "name_rx": r"\bkeane\b", "goals_rx": r"(?<!\d)68(?!\d)",
     "caps_rx": r"(?<!\d)146(?!\d)", "slug_rx": r"wiki/robbie_keane"},
    {"key": "tomasson", "name": "Jon Dahl Tomasson", "country": "Denmark",
     "goals": 52, "caps": 112, "ratio": 0.4643, "winner": False,
     "name_rx": r"\btomasson\b", "goals_rx": r"(?<!\d)52(?!\d)",
     "caps_rx": r"(?<!\d)112(?!\d)", "slug_rx": r"wiki/jon_dahl_tomasson"},
    {"key": "larsson", "name": "Henrik Larsson", "country": "Sweden",
     "goals": 37, "caps": 106, "ratio": 0.3491, "winner": False,
     "name_rx": r"\blarsson\b", "goals_rx": r"(?<!\d)37(?!\d)",
     "caps_rx": r"(?<!\d)106(?!\d)", "slug_rx": r"wiki/henrik_larsson"},
]

WINNER = next(e for e in ENTITIES if e["winner"])      # Jan Koller — the ratio argmax
WINNER_RATIO = 0.6044                                   # 55 / 91
RATIO_TOL = 0.03                                         # +/- 3% relative on the secondary value

# Winner / other-entity name regexes used by the keystone gate.
_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# Comparative / superlative triggers that assert a winner ("X is the highest / has the best ratio").
# Same battle-tested set as the percentage-change sibling (060).
_SUP = (r"more|larger|greater|higher|bigger|largest|greatest|highest|biggest|most|maximum|best|top")

# Keystone winner detection: the winner (Koller) tied to a 'highest / best ratio' assertion, in
# either direction, with the proximity window forbidden from crossing into ANY other player's name
# (so "Klose has the highest ratio, Koller is second" can never satisfy it). The window is [^.;] —
# newline-tolerant (a header line then the answer below still matches) but bounded at sentence
# periods AND clause-separating semicolons, so a rival asserted as the winner in one clause cannot
# reach a Koller mention in the next. The "than" guard blocks "higher than Koller" (Koller the
# LOSER) from counting.
#   dir 1  (subject -> superlative):  "Koller ... has the highest goals-per-appearance ratio"
#   dir 2  (superlative -> subject):  "the highest ratio is Koller"  (NOT "higher than Koller")
_KOLLER_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Ratio-value detection: a single-digit decimal (e.g. "0.60", "0.604", "0.6044") or the exact
# fraction "55/91". The tolerance band is computed from WINNER_RATIO +/- RATIO_TOL.
_RATIO_NUM = re.compile(r"\b(\d\.\d{1,4})\b")
_RATIO_FRAC = re.compile(r"\b55\s*/\s*91\b")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "059",
        "test_name": "Tier 5: Computed-ratio argmax (highest goals-per-appearance)",
        "difficulty_level": "9/10",
        "category": "Quantitative reasoning + computed-ratio argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']} ({e['country']})" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following five retired forwards, open the player's Wikipedia "
        "page and read TWO integers from the infobox senior 'National team' row: the player's total "
        "senior international GOALS and his total senior international APPEARANCES (caps):\n"
        f"{listing}\n\n"
        "Then COMPUTE, for each player, his GOALS-PER-APPEARANCE ratio = total goals / total caps "
        "(this ratio is NOT printed on the page — you must divide the two figures yourself). "
        "COMPARE the five ratios and determine which player has the HIGHEST goals-per-appearance "
        "ratio. Note: the answer is the largest RATIO, which is NOT necessarily the player with the "
        "most goals, nor the one with the most appearances.\n\n"
        "Report (a) which player has the highest goals-per-appearance ratio (a single player name — "
        "the keystone), (b) that player's computed ratio value, (c) all five players' goals and "
        "appearances (the ten figures you looked up), and (d) the exact source URL of every page "
        "you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which player has the highest goals-per-appearance ratio (the primary answer / keystone)",
        "That winning player's computed ratio value (approximately, goals per appearance)",
        "All five players' goals and appearances (the ten looked-up figures)",
        "Source URL for each player's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per player, a five-way fan-out)",
        "Correctly names Jan Koller as the highest goals-per-appearance ratio (NOT Miroslav Klose, "
        "who has the most goals, NOR Robbie Keane, who has the most caps)",
        "Reports the winner's ratio near 0.60 (within +/- 3%)",
        "Gathers all ten figures (goals and caps for each of the five players)",
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
    """KEYSTONE gate: deliverables[0] names Jan Koller as the highest-ratio player.

    Word-bounded, and NOT satisfied by merely listing Koller among the five: when another player is
    also named, Koller must be the one tied to a 'highest / best ratio' assertion (tempered so a
    rival named as the winner — or 'more than Koller' — never counts). A terse primary answer that
    names only the winner (Koller, with no rival in the slot) also passes.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True  # names only the winner
    return bool(_KOLLER_WINS.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a five-way fan-out wants one page per player."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: one page per player; >=4 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the highest goals-per-appearance ratio is Jan Koller — NOT the
    most-goals player (Miroslav Klose) and NOT the most-caps player (Robbie Keane). A model that
    shortcuts to a raw quantity, inverts the ratio (caps/goals -> Larsson), or guesses the most
    famous name (Klose) mis-picks."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Jan Koller named as the highest goals-per-appearance ratio" if passed
                      else "Highest-ratio player (Jan Koller) missing/incorrect (beware: Klose has "
                           "the most goals, Robbie Keane the most caps)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the FIVE ratios' inputs were gathered.

    A ratio's inputs are credited only when BOTH that player's goals AND caps appear (alongside his
    name); the ten integers are all distinct, so there is no cross-crediting. Deliberately NOT
    gated on the keystone — it measures whether the agent actually fanned out to all five players
    and collected both figures even when it botches the division or the argmax, the axis that
    separates a structured (multi-leaf) agent from a linear one that drops an entity.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["goals_rx"], text) and re.search(e["caps_rx"], text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} players' goals+caps gathered ({', '.join(hits) or 'none'})"}


def validate_winner_ratio(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winner's computed ratio (~0.60, within +/- 3%). Short-circuits to 0
    when the keystone is absent, so a wrong/guessed winner can't bank the value credit."""
    if not _keystone_ok(result, observability):
        return {"check": "winner_ratio", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> ratio value not credited"}
    text = _all_text(result)
    lo, hi = WINNER_RATIO * (1.0 - RATIO_TOL), WINNER_RATIO * (1.0 + RATIO_TOL)
    found = [float(m) for m in _RATIO_NUM.findall(text)]
    ok = any(lo <= v <= hi for v in found) or bool(_RATIO_FRAC.search(text))
    return {"check": "winner_ratio", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"winner's ratio within {lo:.3f}-{hi:.3f} present" if ok
                       else f"no ratio in {lo:.3f}-{hi:.3f} (expected ~{WINNER_RATIO:.4f}); "
                            f"saw {found or 'none'}")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the player pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} player pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_argmax,
        validate_coverage,
        validate_winner_ratio,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), as in 054/055.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    TEN INDEPENDENT parallel leaves: for each of the five GIVEN players, one leaf fetches his total
    senior international GOALS and another his total senior international APPEARANCES — one atomic
    integer per leaf. ALL arithmetic — every goals/caps division AND the final argmax — lives only
    in the aggregation step, which is forced to WRITE OUT each per-player division explicitly before
    concluding, so the cheap executor never divides by the wrong quantity or shortcuts to a raw
    maximum and a diverse-grounding reranker can re-derive and catch a slip. Encodes STRUCTURE only:
    it names the five GIVEN players and their national teams, but leaks no goals figure, no caps
    figure, no ratio, and not which player wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": f"{e['key']}_goals",
            "instruction": (
                f"Open the Wikipedia page for {e['name']} and read, from the infobox senior "
                f"'National team' row, the total number of GOALS he scored in senior internationals "
                f"for {e['country']}. Report ONLY that single integer and the source URL. Do not "
                "guess from memory."
            ),
            "expect": "GOALS (a single integer) -- source URL",
            "depends_on": [],
        })
        leaves.append({
            "id": f"{e['key']}_caps",
            "instruction": (
                f"Open the Wikipedia page for {e['name']} and read, from the infobox senior "
                f"'National team' row, the total number of senior international APPEARANCES (caps) "
                f"he made for {e['country']}. Report ONLY that single integer and the source URL. "
                "Do not guess from memory."
            ),
            "expect": "APPEARANCES / CAPS (a single integer) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the five players, two integers: his total senior "
            "international goals and his total senior international appearances (caps). For EACH of "
            "the five players, write out the division explicitly on its own line in the form "
            "'<player>: <goals> / <caps> = <ratio>' rounded to three decimal places — compute "
            "every one of the five divisions BEFORE drawing any conclusion. THEN, comparing those "
            "five computed ratios, state which SINGLE player has the HIGHEST goals-per-appearance "
            "ratio — that player's name is the keystone answer. Show every division before "
            "concluding. This need NOT be the player with the most goals, nor the one with the "
            "most appearances. Report (a) that player and his goals-per-appearance ratio, (b) all "
            "five players' goals and appearances, and (c) cite each player's source URL."
        ),
    }
