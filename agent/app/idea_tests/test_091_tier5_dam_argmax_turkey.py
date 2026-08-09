"""
Test 091: Tier 5 (graph) — argmax over a PAGE-ONLY infobox figure (structural HEIGHT, in metres)
          across SIX named dams in Turkey. Same argmax keystone shape as test 059 / 064 / 079,
          but with a single looked-up figure per entity (no cross-figure arithmetic) and a
          FAME-DECOY built into the entity set.

The agent must determine which of SIX Turkish dams is the TALLEST — i.e. which has the greatest
structural height (crest height above the streambed) as printed in its own Wikipedia infobox. No
page states the cross-dam ranking; the agent must read six page-only heights and compare them.

    Deriner Dam        (Çoruh River, Artvin)          <- TALLEST of the six (argmax)
    Atatürk Dam        (Euphrates, Şanlıurfa)          <- FAME DECOY (most famous, but only 4th)
    Altınkaya Dam      (Kızılırmak, Samsun)
    Oymapınar Dam      (Manavgat River, Antalya)
    Hasan Uğurlu Dam   (Yeşilırmak, Samsun)
    Karakaya Dam       (Euphrates, Diyarbakır)

Why it discriminates (per REASONING_TEST_DESIGN.md — the differential-lift target):
  * cheap native (graph): guesses the most salient name (Atatürk Dam, the flagship of the GAP
    project and by far the most famous dam in Turkey) or the actual national record-holder from
    memory; cannot read six page-only heights and argmax them without visiting.
  * frontier sequential (ReAct): looks up all six heights and compares -> decent.
  * graph_compiled: six parallel leaves each fetch ONE dam's height from ONE page; aggregation
    owns the argmax, forced to list every height before concluding -> cheap executor is rescued.

The trap is engineered two ways:
  (1) FAME DECOY — Atatürk Dam is the single most recognisable dam name in Turkey (largest by
      reservoir volume AND installed capacity, a national icon), but at 169 m it is only the
      FOURTH-tallest of the six; "pick the famous one" mis-picks.
  (2) ANTI-PARAMETRIC / RECORD DECOY — the actual tallest dam in Turkey is the Yusufeli Dam
      (275 m, completed 2022), which is DELIBERATELY NOT in the set. A model that recalls "the
      tallest Turkish dam" from memory answers with a dam that is not even a choice; the tallest
      OF THESE SIX (Deriner, an obscure double-curvature arch dam on the Çoruh) is knowable only
      by reading and comparing the six heights.

Ground truth (verified against live English Wikipedia 2026-07-07 — each dam's infobox
'Height' row, in metres):

  dam                region                     height (m)
  Deriner Dam        Çoruh River, Artvin           249     <- ARGMAX (tallest of the six)
  Altınkaya Dam      Kızılırmak, Samsun            195     <- 2nd (runner-up)
  Oymapınar Dam      Manavgat River, Antalya       185
  Hasan Uğurlu Dam   Yeşilırmak, Samsun            175
  Atatürk Dam        Euphrates, Şanlıurfa          169     <- FAME DECOY (most famous name)
  Karakaya Dam       Euphrates, Diyarbakır         158

    https://en.wikipedia.org/wiki/Deriner_Dam
    https://en.wikipedia.org/wiki/Atat%C3%BCrk_Dam
    https://en.wikipedia.org/wiki/Alt%C4%B1nkaya_Dam
    https://en.wikipedia.org/wiki/Oymap%C4%B1nar_Dam
    https://en.wikipedia.org/wiki/Hasan_U%C4%9Furlu_Dam
    https://en.wikipedia.org/wiki/Karakaya_Dam

  HEIGHT ARGMAX = Deriner Dam (249 m). Runner-up = Altınkaya Dam (195 m), so the margin is
  +54 m absolute (~+27.7% relative). A worst-case misread cannot flip the keystone: reading
  Deriner as 230 m still clears Altınkaya (195) by 35 m; reading Altınkaya as 210 m still
  trails Deriner (249) by 39 m. The 54 m gap absorbs any single plausible extraction error.

  FAME-DECOY / WIDE-MARGIN checks (confirmed live, asserted at import time):
    (a) height-argmax (Deriner, 249 m) != the most famous dam in the set (Atatürk, 169 m) -> True
    (b) winner height (249) exceeds 2nd place (Altınkaya, 195) by ~27.7% (>> 15%)          -> True
    by height:  Deriner 249 > Altınkaya 195 > Oymapınar 185 > Hasan Uğurlu 175 > Atatürk 169
                > Karakaya 158                                                 -> Deriner Dam
    => the single most salient name (Atatürk Dam) is the baited decoy; Deriner, the winner, is
       an obscure arch dam that only a reader — not a memoriser — can identify as the tallest.

  ANTI-PARAMETRIC: neither the winner (Deriner, 249 m) nor its rank is memorable — the famous
  Atatürk Dam is a rock-fill embankment celebrated for its reservoir and power output, not its
  height, and the national height record (Yusufeli, 275 m) is not in the set. The agent must
  READ each infobox 'Height' figure and compare the six.

  KEYSTONE = Deriner Dam (the height argmax). Secondary (gated) = its height value 249 m
  (accepted 244–254 m). All six heights are distinct, so the un-gated coverage diagnostic is
  collision-free.
"""

from typing import Dict, Any, List
import os
import re
from agent.app.idea_test_utils import extract_final_text


# ---- verified fixtures (single source of truth for statement, validators and plan) ----
ENTITIES: List[Dict[str, Any]] = [
    {
        "key": "deriner",
        "name": "Deriner Dam",
        "region": "Çoruh River, Artvin",
        "height_m": 249,            # Wikipedia infobox: "Height 249 m (817 ft)"
        "winner": True,
        "name_rx": r"\bderiner\b",
        "ht_rx":  r"(?<!\d)249(?!\d)",
        "slug_rx": r"wiki/deriner",
    },
    {
        "key": "altinkaya",
        "name": "Altınkaya Dam",
        "region": "Kızılırmak, Samsun",
        "height_m": 195,            # Wikipedia infobox: "Height 195 m (640 ft)"
        "winner": False,
        "name_rx": r"alt[ıi]nkaya",
        "ht_rx":  r"(?<!\d)195(?!\d)",
        "slug_rx": r"wiki/alt(?:%c4%b1|ı|i)nkaya",
    },
    {
        "key": "oymapinar",
        "name": "Oymapınar Dam",
        "region": "Manavgat River, Antalya",
        "height_m": 185,            # Wikipedia infobox: "Height 185 m"
        "winner": False,
        "name_rx": r"oymap[ıi]nar",
        "ht_rx":  r"(?<!\d)185(?!\d)",
        "slug_rx": r"wiki/oymap(?:%c4%b1|ı|i)nar",
    },
    {
        "key": "hasan_ugurlu",
        "name": "Hasan Uğurlu Dam",
        "region": "Yeşilırmak, Samsun",
        "height_m": 175,            # Wikipedia infobox: "Height 175 m (574 ft)"
        "winner": False,
        "name_rx": r"hasan\s+u(?:ğ|g)urlu",
        "slug_rx": r"wiki/hasan_u(?:%c4%9f|ğ|g)urlu",
        "ht_rx":  r"(?<!\d)175(?!\d)",
    },
    {
        "key": "ataturk",
        "name": "Atatürk Dam",
        "region": "Euphrates, Şanlıurfa",
        "height_m": 169,            # Wikipedia infobox: "Height 169 m (554 ft)"  <- FAME DECOY
        "winner": False,
        "name_rx": r"atat(?:ü|u)rk",
        "ht_rx":  r"(?<!\d)169(?!\d)",
        "slug_rx": r"wiki/atat(?:%c3%bc|ü|u)rk",
    },
    {
        "key": "karakaya",
        "name": "Karakaya Dam",
        "region": "Euphrates, Diyarbakır",
        "height_m": 158,            # Wikipedia infobox: "Height 158 m (518 ft)"
        "winner": False,
        "name_rx": r"\bkarakaya\b",
        "ht_rx":  r"(?<!\d)158(?!\d)",
        "slug_rx": r"wiki/karakaya",
    },
]

WINNER = next(e for e in ENTITIES if e["winner"])          # Deriner Dam — height argmax
WINNER_HEIGHT = WINNER["height_m"]                          # 249 m
HEIGHT_LO, HEIGHT_HI = 244, 254                             # accepted band for the secondary value

# ---- FAME-DECOY / WIDE-MARGIN invariants: asserted at import so the fixtures can never
# ---- silently drift out of the design (winner is tallest; winner != famous decoy; margin > 15%). --
_MOST_FAMOUS = next(e for e in ENTITIES if e["key"] == "ataturk")   # Atatürk Dam — the fame decoy
_BY_HEIGHT = sorted(ENTITIES, key=lambda e: e["height_m"], reverse=True)
assert _BY_HEIGHT[0] is WINNER,      "height argmax must be the declared winner"
assert _MOST_FAMOUS is not WINNER,   "winner must NOT be the most famous dam in the set (fame decoy)"
_W_H = WINNER["height_m"]
_RUNNER_UP_H = _BY_HEIGHT[1]["height_m"]
assert (_W_H - _RUNNER_UP_H) / _RUNNER_UP_H > 0.15, \
    "keystone margin must exceed 15% (currently ~27.7%)"

# ---- winner / others regexes used by the keystone gate ---------------------------------
_WINNER_RX = WINNER["name_rx"]   # r"\bderiner\b"
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# TRUE superlative triggers asserting a tallest / greatest-height winner (no bare "high").
_SUP = r"tallest|highest|greatest|largest|biggest|maximum|max|most|top"

# Keystone winner detection: Deriner tied to a 'tallest / greatest height' assertion, in either
# direction, with the proximity window forbidden from crossing into ANY other dam's name and
# from crossing a sentence/clause boundary. Mirrors test_079's _CHURCHILL_WINS exactly:
#   dir 1  (subject -> superlative): "Deriner Dam … is the tallest"
#   dir 2  (superlative -> subject): "the tallest is Deriner" (NOT "taller than …")
_DERINER_WINS = re.compile(
    _WINNER_RX + r"(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}" + _WINNER_RX,
    re.IGNORECASE,
)

# Winner-height detection: a 3-digit number in the 244–254 band (i.e. ~249 m).
_HT_NUM = re.compile(r"(?<!\d)(2[0-9]{2})(?!\d)")


# =====================================================================
#  CONTRACT FUNCTIONS
# =====================================================================

def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "091",
        "test_name": (
            "Tier 5: Page-only height argmax, wide-margin fame-decoy "
            "(tallest of six named dams in Turkey)"
        ),
        "difficulty_level": "8/10",
        "category": "Quantitative reasoning + page-only argmax",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['region']})"
        for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not "
        "guess from memory). For EACH of the following six dams in Turkey, open the dam's "
        "Wikipedia page and read ONE number from the infobox: the dam's structural HEIGHT "
        "(the 'Height' row, in metres):\n"
        f"{listing}\n\n"
        "Then COMPARE the six heights and determine which SINGLE dam is the TALLEST (greatest "
        "structural height in metres). No Wikipedia page prints this cross-dam comparison — you "
        "must read each dam's own infobox height and compare the six figures yourself. Note: the "
        "answer is the tallest of THESE SIX dams, which is not necessarily the most famous dam "
        "in the list, and not necessarily the tallest dam in Turkey overall.\n\n"
        "Report (a) which of the six dams is the tallest (a single dam name — the keystone), "
        "(b) that dam's height in metres, (c) all six dams' heights (the six figures you looked "
        "up), and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which of the six dams is the tallest (the primary answer / keystone)",
        "That winning dam's height in metres",
        "All six dams' heights (the six looked-up figures)",
        "Source URL for each dam's Wikipedia page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per dam, a six-way fan-out)",
        "Correctly names Deriner Dam as the tallest of the six (NOT Atatürk Dam, the most "
        "famous name in the set, which is only the fourth-tallest at 169 m)",
        "Reports the winner's height near 249 m (accepted 244–254 m)",
        "Gathers all six dams' heights",
        "Cites the six source pages",
    ]


# ---- internal helpers ---------------------------------------------------

def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text: prefer deliverables[0] when present; fall back to final_deliverable."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: final deliverable + every deliverable slot concatenated."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names Deriner Dam as the tallest of the six.

    Word-bounded, and NOT satisfied by merely listing Deriner among the six: when another dam is
    also named, Deriner must be the one tied to a 'tallest / greatest height' assertion (tempered
    so a rival named as the winner — or 'taller than Deriner' — never counts). A terse primary
    answer naming only the winner (Deriner, with no rival in the slot) also passes.

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
        return True   # names only the winner, no rival present
    return bool(_DERINER_WINS.search(text))


def _height_value_present(text: str) -> bool:
    """True when a number in the winner's height band (244–254 m, i.e. ~249) appears."""
    for raw in _HT_NUM.findall(text):
        v = int(raw)
        if HEIGHT_LO <= v <= HEIGHT_HI:
            return True
    return False


# =====================================================================
#  VALIDATION FUNCTIONS
# =====================================================================

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per dam."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 4,
        "score": min(1.0, n / 6.0),
        "reason": f"{n} visit(s) (target ≥6: one page per dam; ≥4 to pass)",
    }


def validate_keystone_argmax(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the tallest of the six is Deriner Dam (249 m) — NOT the most famous
    name (Atatürk Dam, 169 m, only 4th). A model that guesses the salient name or the national
    height record (Yusufeli, not in the set) mis-picks.
    """
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_argmax",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Deriner Dam named as the tallest of the six"
            if passed else
            "Tallest dam (Deriner Dam, 249 m) missing/incorrect — beware: Atatürk Dam is the "
            "most famous name in the set but only the fourth-tallest (169 m)"
        ),
    }


def validate_coverage(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the six dams' heights were gathered.

    A dam's height is credited only when BOTH its name AND its height figure appear. All six
    heights are distinct (no cross-crediting). Deliberately NOT gated on the keystone — it
    measures structured fan-out, not correctness.
    """
    text = _all_text(result)
    hits = [
        e["name"]
        for e in ENTITIES
        if (re.search(e["name_rx"], text, re.IGNORECASE)
            and re.search(e["ht_rx"], text))
    ]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": (
            f"{len(hits)}/{n} dams' heights gathered "
            f"({', '.join(hits) or 'none'})"
        ),
    }


def validate_winner_height(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: the winner's height (~249 m, accepted 244–254 m). Short-circuits to 0
    when the keystone is absent, so a guessed winner cannot bank the value credit.
    """
    if not _keystone_ok(result, observability):
        return {
            "check": "winner_height",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → height value not credited",
        }
    ok = _height_value_present(_all_text(result))
    return {
        "check": "winner_height",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"winner's height in {HEIGHT_LO}–{HEIGHT_HI} m present"
            if ok else
            f"no height in {HEIGHT_LO}–{HEIGHT_HI} m (expected ~{WINNER_HEIGHT} m)"
        ),
    }


def validate_citation(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: cites the dam pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {
            "check": "citation",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent → source URLs not credited",
        }
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {
        "check": "citation",
        "passed": cited >= 3,
        "score": cited / n,
        "reason": f"{cited}/{n} dam pages cited",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_argmax,
        validate_coverage,
        validate_winner_height,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None → the harness applies its default structured rubric judge, as in 064 / 079.
    return None


# =====================================================================
#  COMPILED PLAN  (offline-authored leak-free scaffold)
# =====================================================================

def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out / aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves: for each of the six GIVEN dams, one leaf opens that dam's
    Wikipedia page and reads its infobox HEIGHT (metres) — one page-read per dam. The argmax
    lives only in the aggregation step, which is forced to LIST every per-dam height explicitly
    before concluding, rescuing the cheap executor and letting a diverse-grounding reranker
    re-derive and catch any slip.

    Encodes STRUCTURE ONLY: names the six GIVEN dams and their regions, but leaks no height
    figure and not which dam wins. Each leaf's answer text self-describes its dam (name in the
    'expect' line) so aggregation binding survives leaf-ID stripping.

    Composition kill-switch (Part D Stage 1 validation, off by default): when
    ``IDEA_TEST_COMPILED_COMPOSITION_KILLSWITCH`` is set, the plan additionally declares
    ``agg_mode: "computed"`` plus an ``argmax`` ``composition`` over the six height leaves, routing
    the final comparison through ``execution_compiled._compose_argmax`` (deterministic numeric
    comparison) instead of the free-text aggregation prompt below. Unset (the default) reproduces
    today's plan dict exactly — the ``aggregation`` string stays byte-identical either way, since
    it remains the fallback whenever composition can't resolve honestly.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for {e['name']} ({e['region']}) and read, from the "
                "infobox, ONE figure: the dam's structural HEIGHT (the 'Height' row, in metres). "
                f"Report ONLY that height, stated as '{e['name']}: height = <number> m', and the "
                "source URL. Do not compare, do not rank, do not compute anything — just read and "
                "report this one dam's height."
            ),
            "expect": (
                f"{e['name']}: height = <number> m — source URL"
            ),
            "depends_on": [],
        })
    plan: Dict[str, Any] = {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six dams, its structural height in metres. "
            "FIRST, list all six on their own lines in the form "
            "'<dam>: <height> m' — write out every one of the six heights BEFORE drawing any "
            "conclusion. THEN, comparing those six heights, state which SINGLE dam is the "
            "TALLEST (greatest height in metres) — that dam name is the keystone answer. This "
            "need NOT be the most famous dam in the list, nor the tallest dam in Turkey overall. "
            "Report (a) that dam and its height in metres, (b) all six dams' heights, and "
            "(c) cite each dam's source URL."
        ),
    }
    if os.environ.get("IDEA_TEST_COMPILED_COMPOSITION_KILLSWITCH", "") not in ("", "0", "false", "False"):
        plan["agg_mode"] = "computed"
        plan["composition"] = {
            "op": "argmax",
            "items": [
                {"leaf": e["key"], "label": e["name"], "type": "number", "unit": "m"}
                for e in ENTITIES
            ],
            "answer_noun": "dam",
            "value_label": "height",
            "unit": "m",
        }
    return plan
