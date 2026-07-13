"""
Test 062: Tier 5 (breadth) — 6-way Fan-out & Aggregation over a PAGE-ONLY physical quantity.
Level: graph   Weight: long   Difficulty: 8/10

A direct clone of the proven breadth winner (test 052), but the aggregated quantity is each
peak's TOPOGRAPHIC PROMINENCE in metres — the height of a summit above its highest connecting
saddle (key col), NOT its elevation above sea level. Prominence is the ideal anti-parametric
axis: unlike elevation it is essentially never memorized, it is printed once in the infobox of
each peak's page, and (critically) the prominence ranking does NOT track the elevation ranking,
so the cheap shortcut "pick the tallest / most famous mountain" lands on the wrong peak.

NO URLs are given. For EACH of six obscure ~7,400-7,650 m Central/High-Asian peaks the agent
must open the page and read ONE figure (its prominence), then AGGREGATE across all six to report
which peak is the MOST topographically prominent (argmax). A *linear* ReAct agent must serialize
six gather-hops into a capped step budget and hold all six figures in one degrading scratchpad to
compute the argmax; a graph fans the six reads out in parallel and aggregates structurally. It is
the discriminator for the "expensive-model-authored scaffold lets a cheap model recover top-tier
accuracy" thesis: see ``get_compiled_plan`` (the offline-authored fan-out/aggregate plan the
``graph_compiled`` variant executes).

The trap is engineered two ways:
  (1) prominence is page-only — it is not recallable, so the agent MUST open each page and read it;
  (2) the prominence ranking DIVERGES from the elevation ranking — the most prominent peak is in
      fact the SHORTEST of the six, and the TALLEST peak is only the third-most-prominent — so
      "pick the biggest/tallest mountain" mis-picks.

Ground truth (verified against live English Wikipedia 2026-06-26 — each peak's infobox
'Topographic prominence' row; all six are independent peaks with one unambiguous figure):

  peak                            country               elevation (m)   prominence (m)
  Jengish Chokusu (Pik Pobedy)    Kyrgyzstan / China      7,439  <-min     4,148  <- ARGMAX (keystone)
  Mount Gongga (Minya Konka)      China (Sichuan)         7,509            3,642  <- prominence runner-up
  Kongur Tagh                     China (Xinjiang)        7,649  <-max     3,585  <- TALLEST (elevation decoy)
  Ismoil Somoni Peak              Tajikistan              7,495            3,402
  Muztagh Ata                     China (Xinjiang)        7,546            2,698
  Noshaq                          Afghanistan / Pakistan  7,492            2,024
    https://en.wikipedia.org/wiki/Jengish_Chokusu
    https://en.wikipedia.org/wiki/Mount_Gongga
    https://en.wikipedia.org/wiki/Kongur_Tagh
    https://en.wikipedia.org/wiki/Ismoil_Somoni_Peak
    https://en.wikipedia.org/wiki/Muztagh_Ata
    https://en.wikipedia.org/wiki/Noshaq

  PROMINENCE ARGMAX = Jengish Chokusu (4,148 m). Runner-up = Mount Gongga (3,642 m), so the margin
  is +506 m absolute (~13.9% relative). Source disagreement on these figures is well under 506 m
  and no single noisy extraction can flip the keystone.

  RANKING DIVERGENCE (confirmed live):
    by elevation:  Kongur Tagh 7,649 > Muztagh Ata 7,546 > Mount Gongga 7,509 > Ismoil Somoni 7,495
                   > Noshaq 7,492 > Jengish Chokusu 7,439                      -> TALLEST = Kongur Tagh
    by prominence: Jengish Chokusu 4,148 > Mount Gongga 3,642 > Kongur Tagh 3,585 > Ismoil Somoni
                   3,402 > Muztagh Ata 2,698 > Noshaq 2,024                    -> MOST PROMINENT = Jengish
    => the prominence winner (Jengish Chokusu) is the SHORTEST of the six; the tallest (Kongur Tagh)
       is only third by prominence. "Pick the tallest peak" is wrong, and the two rankings are in
       fact almost reversed at the top.

  ANTI-PARAMETRIC: none of the six is an iconic eight-thousander; their prominence figures are not
  recallable. A parametric model that never opens the pages cannot rank them, and a "tallest /
  most famous" shortcut lands on Kongur Tagh (the elevation decoy), not on the keystone.

  KEYSTONE = the argmax ENTITY (Jengish Chokusu). Secondary (gated) value = its prominence ~4,148 m.
  All six prominence figures are distinct, so the un-gated coverage diagnostic is collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) -----
# ``elevation`` / ``prominence`` are shown for provenance; ONLY peak names + countries are surfaced
# in the task statement and the compiled plan — no figure and no winner are ever leaked.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "jengish_chokusu", "name": "Jengish Chokusu", "country": "Kyrgyzstan / China",
     "elevation": 7439, "prominence": 4148, "winner": True,
     "name_rx": r"jengish|chokusu|pobed[ay]|victory\s+peak|tomur|tuomuer|t[oö]m[uü]r",
     "prom_rx": r"(?<!\d)4[,]?148(?!\d)", "slug_rx": r"wiki/jengish_chokusu|wiki/pobeda_peak|wiki/tomur"},
    {"key": "gongga", "name": "Mount Gongga", "country": "China (Sichuan)",
     "elevation": 7509, "prominence": 3642, "winner": False,
     "name_rx": r"gongga|minya\s+konka",
     "prom_rx": r"(?<!\d)3[,]?642(?!\d)", "slug_rx": r"wiki/mount_gongga|wiki/gongga_shan|wiki/minya_konka"},
    {"key": "kongur_tagh", "name": "Kongur Tagh", "country": "China (Xinjiang)",
     "elevation": 7649, "prominence": 3585, "winner": False,
     "name_rx": r"kongur",
     "prom_rx": r"(?<!\d)3[,]?585(?!\d)", "slug_rx": r"wiki/kongur_tagh"},
    {"key": "ismoil_somoni", "name": "Ismoil Somoni Peak", "country": "Tajikistan",
     "elevation": 7495, "prominence": 3402, "winner": False,
     "name_rx": r"ismoil\s+somoni|\bsomoni\b|communism|kommunizma",
     "prom_rx": r"(?<!\d)3[,]?402(?!\d)", "slug_rx": r"wiki/ismoil_somoni_peak|wiki/communism_peak"},
    {"key": "muztagh_ata", "name": "Muztagh Ata", "country": "China (Xinjiang)",
     "elevation": 7546, "prominence": 2698, "winner": False,
     "name_rx": r"muztagh|muztag\s*ata",
     "prom_rx": r"(?<!\d)2[,]?698(?!\d)", "slug_rx": r"wiki/muztagh_ata|wiki/muztagata"},
    {"key": "noshaq", "name": "Noshaq", "country": "Afghanistan / Pakistan",
     "elevation": 7492, "prominence": 2024, "winner": False,
     "name_rx": r"noshaq|nowshak",
     "prom_rx": r"(?<!\d)2[,]?024(?!\d)", "slug_rx": r"wiki/noshaq|wiki/nowshak"},
]

WINNER = next(e for e in ENTITIES if e["winner"])     # Jengish Chokusu — the prominence argmax
WINNER_PROM = 4148                                     # metres (verified live)
PROM_TOL = 0.025                                        # +/- 2.5% band on the secondary value

# Winner / other-entity name regexes used by the keystone gate.
_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# Comparative / superlative triggers that assert a winner ("X is the most prominent / has the
# highest prominence"). Same battle-tested set as the computed-ratio sibling (059). The "\b...\b"
# guards keep "top" from firing inside "topographic", which saturates this domain's prose.
_SUP = (r"more|larger|greater|higher|bigger|largest|greatest|highest|biggest|most|maximum|best|top")

# Keystone winner detection: the winner (Jengish Chokusu) tied to a 'most/highest prominence'
# assertion, in either direction, with the proximity window forbidden from crossing into ANY other
# peak's name (so "Kongur Tagh is the most prominent, Jengish is second" can never satisfy it). The
# window is [^.;] — newline-tolerant (a header line then the answer below still matches) but bounded
# at sentence periods AND clause-separating semicolons. The "than" guard blocks "more prominent than
# Jengish" (Jengish the LOSER) from counting.
#   dir 1  (subject -> superlative):  "Jengish Chokusu ... has the highest prominence"
#   dir 2  (superlative -> subject):  "the most prominent peak is Jengish Chokusu"  (NOT "than Jengish")
_JENGISH_WINS = re.compile(
    r"(?:" + _WINNER_RX + r")(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}(?:" + _WINNER_RX + r")",
    re.IGNORECASE,
)

# Prominence-value detection: a 4-digit figure with an optional thousands comma ("4,148" / "4148").
# Candidates are filtered against the WINNER_PROM +/- PROM_TOL band.
_PROM_NUM = re.compile(r"(?<!\d)(\d[,]?\d{3})(?!\d)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "062",
        "test_name": "Tier 5: Page-only prominence argmax (most topographically prominent peak)",
        "difficulty_level": "8/10",
        "category": "Breadth Fan-out & Aggregation (page-only prominence argmax)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']} ({e['country']})" for i, e in enumerate(ENTITIES, 1))
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following six mountains, open the peak's Wikipedia page and "
        "read ONE figure directly from the infobox: its TOPOGRAPHIC PROMINENCE in metres (the height "
        "of the summit above its highest connecting saddle / key col — this is NOT the same as the "
        "peak's elevation above sea level):\n"
        f"{listing}\n\n"
        "Then COMPARE the six prominence figures and determine which mountain has the HIGHEST "
        "topographic prominence. IMPORTANT: the most topographically prominent peak is NOT "
        "necessarily the tallest by elevation, nor the most famous — you must read each peak's "
        "prominence from its page rather than reasoning from how tall or well-known the mountain "
        "is.\n\n"
        "Report (a) which mountain has the highest topographic prominence (a single peak name — the "
        "keystone), (b) that peak's prominence in metres, (c) all six peaks' prominence figures, and "
        "(d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which mountain has the highest topographic prominence (the primary answer / keystone)",
        "That winning peak's prominence in metres",
        "All six peaks' topographic prominence figures",
        "Source URL for each peak's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per peak, a six-way fan-out)",
        "Correctly names Jengish Chokusu as the most topographically prominent (NOT Kongur Tagh, "
        "which is the TALLEST by elevation but only third by prominence)",
        "Reports the winner's prominence near 4,148 m (within +/- 2.5%)",
        "Reports all six peaks' prominence figures",
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
    """KEYSTONE gate: deliverables[0] names Jengish Chokusu as the most prominent peak.

    Word-bounded, and NOT satisfied by merely listing Jengish among the six: when another peak is
    also named, Jengish must be the one tied to a 'most / highest prominence' assertion (tempered so
    a rival named as the winner — or 'more prominent than Jengish' — never counts). A terse primary
    answer that names only the winner (Jengish, with no rival in the slot) also passes.

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
        return True  # names only the winner
    return bool(_JENGISH_WINS.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per peak."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 6.0),
            "reason": f"{n} visit(s) (target >=6: one page per peak; >=4 to pass)"}


def validate_keystone_argmax(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the most topographically prominent peak is Jengish Chokusu — NOT the
    tallest peak (Kongur Tagh). A model that shortcuts to elevation/fame, or guesses the most
    famous name, mis-picks Kongur Tagh (the elevation decoy)."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_argmax", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Jengish Chokusu named as the most topographically prominent peak" if passed
                      else "Most-prominent peak (Jengish Chokusu) missing/incorrect (beware: Kongur "
                           "Tagh is the TALLEST by elevation but only third by prominence)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the SIX peaks' prominence figures were
    gathered.

    A peak is credited only when BOTH its name AND its (distinct) prominence figure appear; the six
    figures are all distinct, so there is no cross-crediting. Deliberately NOT gated on the keystone
    — it measures whether the agent actually fanned out to all six peaks and read each prominence
    even when it botches the argmax, the axis that separates a structured (multi-leaf) agent from a
    linear one that drops an entity.
    """
    text = _all_text(result)
    hits = [e["name"] for e in ENTITIES
            if re.search(e["name_rx"], text, re.IGNORECASE) and re.search(e["prom_rx"], text)]
    n = len(ENTITIES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} peaks' prominence gathered ({', '.join(hits) or 'none'})"}


def validate_winner_prominence(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winner's prominence (~4,148 m, within +/- 2.5%). Short-circuits to 0
    when the keystone is absent, so a wrong/guessed winner can't bank the value credit."""
    if not _keystone_ok(result, observability):
        return {"check": "winner_prominence", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> prominence value not credited"}
    text = _all_text(result)
    lo, hi = WINNER_PROM * (1.0 - PROM_TOL), WINNER_PROM * (1.0 + PROM_TOL)
    found = [int(m.replace(",", "")) for m in _PROM_NUM.findall(text)]
    ok = any(lo <= v <= hi for v in found)
    return {"check": "winner_prominence", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"winner's prominence within {lo:.0f}-{hi:.0f} m present" if ok
                       else f"no prominence in {lo:.0f}-{hi:.0f} m (expected ~{WINNER_PROM}); "
                            f"saw {found or 'none'}")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the peak pages. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for e in ENTITIES if re.search(e["slug_rx"], text))
    n = len(ENTITIES)
    return {"check": "citation", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} peak pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_argmax,
        validate_coverage,
        validate_winner_prominence,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), as in 052/059.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves: for each of the six GIVEN peaks, one leaf reads ONE atomic
    figure — that peak's topographic prominence in metres. The argmax lives only in the aggregation
    step, which is reminded that prominence is NOT elevation, so the cheap executor never shortcuts
    to the tallest peak and a diverse-grounding reranker can re-derive the comparison. Encodes
    STRUCTURE only: it names the six GIVEN peaks and their countries, but leaks no prominence figure
    and never states which peak wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            # id keyed on the GIVEN peak (never the argmax answer), so the scaffold leaks nothing.
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for the mountain {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its TOPOGRAPHIC PROMINENCE in metres — the height of the "
                "summit above its highest connecting saddle (key col), which is NOT the same as its "
                "elevation above sea level. Report ONLY that single prominence figure (in metres) and "
                "the source URL. Do not guess from memory."
            ),
            "expect": "TOPOGRAPHIC PROMINENCE in metres (a single figure) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You now have, for each of the six peaks, its topographic prominence in metres (with a "
            "source URL). COMPARE the six prominence figures and determine which SINGLE peak has the "
            "HIGHEST topographic prominence — that peak's name is the keystone answer. NOTE: the most "
            "topographically prominent peak is NOT necessarily the tallest by elevation, so compare "
            "the prominence figures themselves, not how tall or famous each mountain is. Report (a) "
            "the most prominent peak and its prominence in metres, stating explicitly that it has the "
            "highest prominence, (b) the full list peak -> prominence in metres for all six, and (c) "
            "cite each peak's source URL."
        ),
    }
