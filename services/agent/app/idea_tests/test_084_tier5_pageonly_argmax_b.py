"""
Test 084: Tier 5 (breadth) — 6-way Fan-out & Aggregation over a PAGE-ONLY physical quantity.
Level: graph   Weight: long   Difficulty: 8/10

A direct clone of the proven breadth winner (test 062), but the aggregated quantity is each
lake's MAXIMUM DEPTH in metres — the deepest measured point below the surface, NOT the lake's
elevation above sea level. Maximum depth is the ideal anti-parametric axis: unlike surface area
or location, it is essentially never memorised, it is printed once in the infobox of each lake's
Wikipedia page, and (critically) the depth ranking does NOT track fame, so the cheap shortcut
"pick the most famous lake" lands on the wrong answer.

NO URLs are given. For EACH of six lakes the agent must open the page and read ONE figure (its
maximum depth in metres), then AGGREGATE across all six to report which lake is the DEEPEST.
A linear ReAct agent must serialise six gather-hops into a capped step budget and hold all six
figures in one degrading scratchpad; a graph fans the six reads out in parallel and aggregates
structurally. It is the discriminator for the "expensive-model-authored scaffold lets a cheap
model recover top-tier accuracy" thesis: see ``get_compiled_plan``.

The trap is engineered two ways:
  (1) maximum depth is page-only — it is not recallable, so the agent MUST open each page;
  (2) the depth ranking DIVERGES from the fame ranking — the most celebrated lake in the set
      (Crater Lake, famous as "the deepest lake in the United States") is only SECOND, not the
      winner — while the winner (Lake O'Higgins / San Martín) is obscure enough that most
      parametric models cannot recall or compare its exact depth figure.

Ground truth (verified against live English Wikipedia 2026-06-27 — each lake's infobox
'Max. depth' row; all six figures are independent, single-valued, and unambiguous):

  lake                              country              max depth (m)
  Lake O'Higgins / San Martín       Argentina / Chile         836   <- ARGMAX (keystone)
  Crater Lake                       United States             594   <- FAME DECOY (2nd, not winner)
  Lake Matano                       Indonesia                 590
  Hornindalsvatnet                  Norway                    514
  Quesnel Lake                      Canada                    511
  Lake Como                         Italy                     425
    https://en.wikipedia.org/wiki/O%27Higgins/San_Mart%C3%ADn_Lake
    https://en.wikipedia.org/wiki/Crater_Lake
    https://en.wikipedia.org/wiki/Lake_Matano
    https://en.wikipedia.org/wiki/Hornindalsvatnet
    https://en.wikipedia.org/wiki/Quesnel_Lake
    https://en.wikipedia.org/wiki/Lake_Como

  DEPTH ARGMAX = Lake O'Higgins / San Martín (836 m). Runner-up = Crater Lake (594 m), so the
  margin is +242 m absolute (~29% relative). Source disagreement on these figures is well under
  242 m and no single noisy extraction can flip the keystone.

  RANKING DIVERGENCE (confirmed live):
    by fame:  Crater Lake (famous as "deepest US lake") > Lake Como (Italian tourist landmark)
              > Hornindalsvatnet ("Europe's deepest") > others   -> MOST FAMOUS != winner
    by depth: O'Higgins 836 > Crater Lake 594 > Matano 590 > Hornindalsvatnet 514
              > Quesnel 511 > Como 425                          -> DEEPEST = O'Higgins
    => the depth winner (O'Higgins) is less famous than Crater Lake and Lake Como;
       "pick the most famous" mis-picks the fame decoy (Crater Lake).

  ANTI-PARAMETRIC: none of the six max-depth figures is commonly memorised; the exact depth of
  O'Higgins (836 m) requires reading the page.  A parametric model that never opens the pages
  cannot correctly rank all six, and a "pick the most famous / celebrated for depth" shortcut
  lands on Crater Lake (the fame decoy at 2nd place), not on the keystone.

  KEYSTONE = the argmax ENTITY (Lake O'Higgins / San Martín). Secondary (gated) value = its
  maximum depth ~836 m. All six depth figures are distinct, so the un-gated coverage diagnostic
  is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- verified fixtures (single source of truth for statement, validators, and the plan) -----
# ``depth`` is shown for provenance; ONLY lake names + countries are surfaced in the task
# statement and the compiled plan — no figure and no winner are ever leaked.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "ohiggins", "name": "Lake O'Higgins / San Martín", "country": "Argentina / Chile",
     "depth": 836, "winner": True,
     "name_rx": r"o.higgins|san\s+mart[ií]n\s+lake|lake\s+san\s+mart[ií]n",
     "depth_rx": r"(?<!\d)836(?!\d)",
     "slug_rx": r"wiki/.*higgins"},
    {"key": "crater_lake", "name": "Crater Lake", "country": "United States",
     "depth": 594, "winner": False,
     "name_rx": r"crater\s+lake",
     "depth_rx": r"(?<!\d)594(?!\d)",
     "slug_rx": r"wiki/crater_lake"},
    {"key": "matano", "name": "Lake Matano", "country": "Indonesia",
     "depth": 590, "winner": False,
     "name_rx": r"\bmatano\b",
     "depth_rx": r"(?<!\d)590(?!\d)",
     "slug_rx": r"wiki/lake_matano|wiki/matano"},
    {"key": "hornindalsvatnet", "name": "Hornindalsvatnet", "country": "Norway",
     "depth": 514, "winner": False,
     "name_rx": r"hornindals",
     "depth_rx": r"(?<!\d)514(?!\d)",
     "slug_rx": r"wiki/hornindalsvatnet"},
    {"key": "quesnel", "name": "Quesnel Lake", "country": "Canada",
     "depth": 511, "winner": False,
     "name_rx": r"quesnel",
     "depth_rx": r"(?<!\d)511(?!\d)",
     "slug_rx": r"wiki/quesnel_lake"},
    {"key": "como", "name": "Lake Como", "country": "Italy",
     "depth": 425, "winner": False,
     "name_rx": r"\bcomo\b",
     "depth_rx": r"(?<!\d)425(?!\d)",
     "slug_rx": r"wiki/lake_como"},
]

WINNER = next(e for e in ENTITIES if e["winner"])    # Lake O'Higgins — the depth argmax
WINNER_DEPTH = 836                                    # metres (verified live)
DEPTH_TOL = 0.025                                     # +/- 2.5% band on the secondary value

# Winner / other-entity name regexes used by the keystone gate.
_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# Comparative / superlative triggers that assert a winner ("X is the deepest / greatest depth").
# The "\b...\b" guards prevent spurious partial-word matches.
_SUP = (r"more|larger|greater|higher|deeper|bigger|deepest|largest|greatest|highest|biggest|"
        r"most|maximum|best|top")

# Keystone winner detection: O'Higgins tied to a 'deepest / greatest / maximum depth' assertion,
# in either direction, with the proximity window forbidden from crossing into ANY other lake's
# name (so "Crater Lake is the deepest, and O'Higgins is second" can never satisfy it). The
# window is [^.;] — newline-tolerant but bounded at sentence periods AND semicolons. The "than"
# guard in dir 2 blocks "deeper than O'Higgins" (O'Higgins the LOSER) from counting.
#   dir 1 (subject -> superlative): "O'Higgins ... has the greatest maximum depth"
#   dir 2 (superlative -> subject): "the deepest lake is O'Higgins" (NOT "than O'Higgins")
_OHIGGINS_WINS = re.compile(
    r"(?:" + _WINNER_RX + r")(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}(?:" + _WINNER_RX + r")",
    re.IGNORECASE,
)

# Maximum-depth value detection: any 3-digit figure in the text.
# Candidates are filtered against the WINNER_DEPTH +/- DEPTH_TOL band in the validator.
_DEPTH_NUM = re.compile(r"(?<!\d)(\d{3})(?!\d)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "084",
        "test_name": "Tier 5: Page-only max-depth argmax b (deepest lake, fame decoy at 2nd)",
        "difficulty_level": "8/10",
        "category": "Breadth Fan-out & Aggregation (page-only max-depth argmax)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['country']})" for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following six lakes, open the lake's Wikipedia page and "
        "read ONE figure directly from the infobox: its MAXIMUM DEPTH in metres (the deepest "
        "measured point below the surface — this is NOT the lake's elevation above sea level):\n"
        f"{listing}\n\n"
        "Then COMPARE the six maximum-depth figures and determine which lake is the DEEPEST. "
        "IMPORTANT: the deepest lake is NOT necessarily the most famous or the largest by surface "
        "area — you must read each lake's maximum depth from its page rather than reasoning from "
        "how well-known the lake is.\n\n"
        "Report (a) which lake has the greatest maximum depth (a single lake name — the keystone), "
        "(b) that lake's maximum depth in metres, (c) all six lakes' maximum-depth figures, and "
        "(d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which lake has the greatest maximum depth (the primary answer / keystone)",
        "That winning lake's maximum depth in metres",
        "All six lakes' maximum-depth figures",
        "Source URL for each lake's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per lake, a six-way fan-out)",
        "Correctly names Lake O'Higgins / San Martín as the deepest (NOT Crater Lake, "
        "which is famous as the deepest US lake but is only second in this set)",
        "Reports the winner's maximum depth near 836 m (within +/- 2.5%)",
        "Reports all six lakes' maximum-depth figures",
        "Cites the source pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text. Prefer ``deliverables[0]`` when present; fall back to final output."""
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
    """KEYSTONE gate: deliverables[0] names Lake O'Higgins / San Martín as the deepest lake.

    NOT satisfied by merely listing O'Higgins among the six: when another lake is also named,
    O'Higgins must be the one tied to a 'deepest / greatest / maximum depth' assertion. A terse
    primary answer that names only the winner (O'Higgins, with no rival in the slot) also passes.
    """
    text = _primary_text(result)
    if not re.search(_WINNER_RX, text, re.IGNORECASE):
        return False
    if not re.search(_OTHERS, text, re.IGNORECASE):
        return True  # names only the winner — no rival to disambiguate
    return bool(_OHIGGINS_WINS.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per lake."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 4,
        "score": min(1.0, n / 6.0),
        "reason": f"{n} visit(s) (target >=6: one page per lake; >=4 to pass)",
    }


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the deepest lake is Lake O'Higgins / San Martín, NOT the fame decoy
    (Crater Lake). A model that shortcuts to fame / reputation mis-picks Crater Lake."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_argmax",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Lake O'Higgins / San Martín named as the deepest lake" if passed
            else "Deepest lake (O'Higgins / San Martín) missing/incorrect (beware: Crater Lake "
                 "is the FAME DECOY — famous as deepest US lake, but only 2nd at 594 m here)"
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the SIX lakes' max-depth figures were
    gathered.

    A lake is credited only when BOTH its name AND its (distinct) depth figure appear; the six
    figures are all distinct, so there is no cross-crediting. Deliberately NOT gated on the
    keystone — measures whether the agent fanned out to all six lakes regardless of the argmax.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE) and re.search(e["depth_rx"], text)
    ]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} lakes' max depth gathered ({', '.join(hits) or 'none'})",
    }


def validate_winner_depth(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winner's maximum depth (~836 m, within +/- 2.5%). Short-circuits to
    0 when the keystone is absent, so a guessed winner can't bank the value credit."""
    if not _keystone_ok(result):
        return {
            "check": "winner_depth",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> depth value not credited",
        }
    text = _all_text(result)
    lo, hi = WINNER_DEPTH * (1.0 - DEPTH_TOL), WINNER_DEPTH * (1.0 + DEPTH_TOL)
    found = [int(m) for m in _DEPTH_NUM.findall(text)]
    ok = any(lo <= v <= hi for v in found)
    return {
        "check": "winner_depth",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"winner's max depth within {lo:.0f}-{hi:.0f} m present" if ok
            else f"no depth in {lo:.0f}-{hi:.0f} m (expected ~{WINNER_DEPTH}); "
                 f"saw {found or 'none'}"
        ),
    }


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the lake pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
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
        "reason": f"{cited}/{n} lake pages cited",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_argmax,
        validate_coverage,
        validate_winner_depth,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 052/062.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves: for each lake, one leaf reads ONE atomic figure — that
    lake's maximum depth in metres. The argmax lives only in the aggregation step, which is
    reminded that depth is NOT the same as fame, so the cheap executor never shortcuts to the
    most celebrated name. Encodes STRUCTURE only: names the six lakes and their countries, but
    leaks no depth figure and never states which lake wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            # id keyed on the GIVEN lake (never the argmax answer), so the scaffold leaks nothing.
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its MAXIMUM DEPTH in metres — the deepest measured "
                "point below the surface, which is NOT the lake's elevation above sea level. "
                "Report ONLY that single maximum-depth figure (in metres) and the source URL. "
                "Do not guess from memory."
            ),
            "expect": "MAXIMUM DEPTH in metres (a single figure) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six lakes, its maximum depth in metres (with a source "
            "URL). COMPARE the six depth figures and determine which SINGLE lake has the GREATEST "
            "maximum depth — that lake's name is the keystone answer. NOTE: the deepest lake is "
            "NOT necessarily the most famous or the best-known; compare the depth figures "
            "themselves, not how celebrated each lake is. Report (a) the deepest lake and its "
            "maximum depth in metres, stating explicitly that it has the greatest depth, (b) the "
            "full list lake -> maximum depth in metres for all six, and (c) cite each lake's "
            "source URL."
        ),
    }
