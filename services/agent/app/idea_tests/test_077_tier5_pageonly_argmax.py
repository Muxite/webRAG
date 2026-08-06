"""
Test 077: Tier 5 (breadth) — 6-way Fan-out & Aggregation over a PAGE-ONLY physical quantity.
Level: graph   Weight: long   Difficulty: 8/10

A direct application of the proven breadth-argmax pattern (test 062 / 064), applied to a new
domain: bridge longest-span lengths. For each of six obscure-but-solid bridges the agent must
open the bridge's Wikipedia page and read ONE figure — its LONGEST SPAN (main span) in metres —
then aggregate across all six to report which bridge has the LONGEST main span (argmax).

A *linear* ReAct agent must serialise six gather-hops into a capped step budget and hold all six
figures in one degrading scratchpad to compute the argmax; a graph fans the six reads out in
parallel and aggregates structurally. It is the discriminator for the
"expensive-model-authored scaffold lets a cheap model recover top-tier accuracy" thesis.

The trap is engineered two ways:
  (1) longest-span values are page-only — they are not recallable for these specific bridges, so
      the agent MUST open each page and read the infobox figure;
  (2) the span ranking DIVERGES from the fame ranking — the bridge with the longest main span
      (Russky Bridge, Russia) is the least famous in the list, while the most iconic bridge
      (Pont de Normandie, France) finishes 4th by span, so "pick the most famous bridge" lands
      on the wrong answer.

Ground truth (verified against live English Wikipedia 2026-06-27 — each bridge's infobox
'Longest span' row; all six are distinct figures):

  bridge                           country    longest span (m)
  Russky Bridge                    Russia          1,104  <- ARGMAX (keystone)
  Edong Yangtze River Bridge       China             926  <- runner-up
  Tatara Bridge                    Japan             890
  Pont de Normandie                France            856  <- MOST FAMOUS (was world's longest
                                                              cable-stayed when built 1994) — decoy
  Rion-Antirion Bridge             Greece            560
  Helgeland Bridge                 Norway            425
    https://en.wikipedia.org/wiki/Russky_Bridge
    https://en.wikipedia.org/wiki/Edong_Yangtze_River_Bridge
    https://en.wikipedia.org/wiki/Tatara_Bridge
    https://en.wikipedia.org/wiki/Pont_de_Normandie
    https://en.wikipedia.org/wiki/Rion-Antirion_Bridge
    https://en.wikipedia.org/wiki/Helgeland_Bridge

  SPAN ARGMAX = Russky Bridge (1,104 m). Runner-up = Edong (926 m), so the margin is +178 m
  absolute (+19.2% relative). Source disagreement on these infobox figures is well under 178 m
  and no single noisy extraction can flip the keystone.

  RANKING DIVERGENCE (confirmed live):
    by fame:  Pont de Normandie > Rion-Antirion > Tatara > Russky > Edong > Helgeland
              (approximate, based on recognisability / media coverage)
    by span:  Russky 1,104 > Edong 926 > Tatara 890 > Pont de Normandie 856
              > Rion-Antirion 560 > Helgeland 425       -> LONGEST = Russky Bridge
    => the longest-span winner (Russky Bridge) is among the least famous in the list; the most
       famous bridge (Pont de Normandie) is only 4th by main span. "Pick the most famous bridge"
       gives the wrong answer.

  ANTI-PARAMETRIC: Pont de Normandie (France, 1994) is the iconic cable-stayed bridge in
  western engineering consciousness and was the world's longest cable-stayed when built; a
  parametric model that pattern-matches to fame or vaguely recalls the 1994 record guesses
  Pont de Normandie (wrong). Tatara Bridge (Japan, 1999) held the record from 1999–2012;
  a model that pattern-matches to Japanese engineering fame guesses Tatara (also wrong). The
  actual winner, Russky Bridge (Russia, 2012 APEC infrastructure), is substantially less
  publicised outside infrastructure circles, and its specific span of 1,104 m is not a
  recallable trivia fact for cheap models — it must be read from the page.

  KEYSTONE = the argmax ENTITY (Russky Bridge). Secondary (gated) value = its longest span
  ~1,104 m. All six span figures are distinct, so the un-gated coverage diagnostic is
  collision-free.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- verified fixtures (single source of truth for statement, validators and the plan) -----
# ``span`` is shown for provenance; ONLY bridge names + countries are surfaced in the task
# statement and the compiled plan — no span figure and no winner are ever leaked there.
ENTITIES: List[Dict[str, Any]] = [
    {"key": "russky", "name": "Russky Bridge", "country": "Russia",
     "span": 1104, "winner": True,
     "name_rx": r"russky",
     "span_rx": r"(?<!\d)1[,]?104(?!\d)",
     "slug_rx": r"wiki/russky.bridge"},
    {"key": "edong", "name": "Edong Yangtze River Bridge", "country": "China",
     "span": 926, "winner": False,
     "name_rx": r"edong",
     "span_rx": r"(?<!\d)926(?!\d)",
     "slug_rx": r"wiki/edong"},
    {"key": "tatara", "name": "Tatara Bridge", "country": "Japan",
     "span": 890, "winner": False,
     "name_rx": r"tatara",
     "span_rx": r"(?<!\d)890(?!\d)",
     "slug_rx": r"wiki/tatara.bridge"},
    {"key": "normandie", "name": "Pont de Normandie", "country": "France",
     "span": 856, "winner": False,
     "name_rx": r"normand(?:ie|y)\b|pont\s*de\s*normand",
     "span_rx": r"(?<!\d)856(?!\d)",
     "slug_rx": r"wiki/pont.de.normand"},
    {"key": "rion_antirion", "name": "Rion-Antirion Bridge", "country": "Greece",
     "span": 560, "winner": False,
     # Real canonical English Wikipedia title is "Rio-Antirrio Bridge" (en-dash, double-r, "io"
     # not "ion") -- "Rion-Antirion_Bridge" redirects there, confirmed live. Both the name and
     # slug patterns kept their original form as one alternative alongside the real spelling.
     "name_rx": r"antirion|antirrio|trikoupis",
     "span_rx": r"(?<!\d)560(?!\d)",
     "slug_rx": r"wiki/(?:rion-antirion|rio.{0,12}antirr?io)"},
    {"key": "helgeland", "name": "Helgeland Bridge", "country": "Norway",
     "span": 425, "winner": False,
     "name_rx": r"helgeland",
     "span_rx": r"(?<!\d)425(?!\d)",
     "slug_rx": r"wiki/helgeland.bridge"},
]

WINNER = next(e for e in ENTITIES if e["winner"])      # Russky Bridge — the span argmax
WINNER_SPAN = 1104                                      # metres (verified live)
SPAN_TOL = 0.025                                        # +/- 2.5% band on the secondary value

# ---- Wide-margin / decoy invariants — asserted at import so fixtures can never silently
# ---- drift out of the design (winner is span argmax; clears runner-up by >15%). ----
_BY_SPAN = sorted(ENTITIES, key=lambda e: e["span"], reverse=True)
assert _BY_SPAN[0] is WINNER, "span argmax must be the declared winner"
_RUNNER_UP_SPAN = _BY_SPAN[1]["span"]
assert (WINNER["span"] - _RUNNER_UP_SPAN) / _RUNNER_UP_SPAN > 0.15, (
    "keystone margin must exceed 15% over runner-up"
)
_FAMOUS_DECOY = next(e for e in ENTITIES if e["key"] == "normandie")
assert _FAMOUS_DECOY["span"] < WINNER["span"], "famous decoy must not be the winner"

# Winner / other-entity name regexes used by the keystone gate.
_WINNER_RX = WINNER["name_rx"]
_OTHERS = "|".join(e["name_rx"] for e in ENTITIES if not e["winner"])

# Comparative / superlative triggers that assert a winner ("X has the longest / greatest span").
# "longest" is the natural superlative for bridge spans and is explicitly included.
_SUP = (
    r"more|larger|greater|higher|bigger|largest|greatest|highest|biggest|most|maximum|best|top"
    r"|longest|widest"
)

# Keystone winner detection: Russky Bridge tied to a 'longest / greatest span' assertion, in
# either direction, with the proximity window forbidden from crossing ANY other bridge name (so
# "Pont de Normandie has the longest span; Russky is second" can never satisfy it). The window
# is [^.;] — newline-tolerant but bounded at sentence periods and clause-separating semicolons.
# The "than" guard blocks "longer than Russky" (Russky the LOSER) from counting.
#   dir 1 (subject -> superlative):  "Russky Bridge ... has the longest main span"
#   dir 2 (superlative -> subject):  "the longest span is Russky Bridge"  (NOT "than Russky")
_RUSSKY_WINS = re.compile(
    r"(?:" + _WINNER_RX + r")(?:(?!" + _OTHERS + r")[^.;]){0,90}\b(?:" + _SUP + r")\b"
    + r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|" + _OTHERS + r")[^.;]){0,55}(?:" + _WINNER_RX + r")",
    re.IGNORECASE,
)

# Span-value detection: a 4-digit figure with optional thousands comma ("1,104" / "1104").
# Candidates are filtered against WINNER_SPAN +/- SPAN_TOL band.
_SPAN_NUM = re.compile(r"(?<!\d)(\d[,]?\d{3})(?!\d)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "077",
        "test_name": "Tier 5: Page-only bridge-span argmax (longest main span among six bridges)",
        "difficulty_level": "8/10",
        "category": "Breadth Fan-out & Aggregation (page-only span argmax)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(
        f"  {i}. {e['name']} ({e['country']})" for i, e in enumerate(ENTITIES, 1)
    )
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). For EACH of the following six bridges, open the bridge's Wikipedia page "
        "and read ONE figure directly from the infobox: its LONGEST SPAN in metres — the length "
        "of the bridge's central span between its main supports (this is NOT the total bridge "
        "length, which includes approach spans):\n"
        f"{listing}\n\n"
        "Then COMPARE the six span figures and determine which bridge has the LONGEST main span. "
        "IMPORTANT: the bridge with the longest main span is NOT necessarily the most famous "
        "bridge in the list — you must read each bridge's span figure from its page rather than "
        "reasoning from how well-known the bridge is.\n\n"
        "Report (a) which bridge has the longest main span (a single bridge name — the keystone), "
        "(b) that bridge's longest span in metres, (c) all six bridges' span figures, and "
        "(d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which bridge has the longest main span (the primary answer / keystone)",
        "That winning bridge's longest span in metres",
        "All six bridges' main-span figures",
        "Source URL for each bridge's page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (one per bridge, a six-way fan-out)",
        "Correctly names Russky Bridge as having the longest main span (NOT Pont de Normandie, "
        "which is the most famous bridge in this list but only 4th by span length at 856 m)",
        "Reports the winner's span near 1,104 m (within ±2.5%)",
        "Reports all six bridges' span figures",
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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names Russky Bridge as having the longest main span.

    Not satisfied by merely listing Russky Bridge among the six: when another bridge is also
    named, Russky must be the one tied to a 'longest / greatest span' assertion (tempered so a
    rival named as winner — or 'longer than Russky' — never counts). A terse primary answer that
    names only the winner (Russky, with no rival in the slot) also passes.

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
        return True  # names only the winner
    return bool(_RUSSKY_WINS.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: a six-way fan-out wants one page per bridge."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 4,
        "score": min(1.0, n / 6.0),
        "reason": f"{n} visit(s) (target >=6: one page per bridge; >=4 to pass)",
    }


def validate_keystone_argmax(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the bridge with the longest main span is Russky Bridge — NOT the most
    famous bridge (Pont de Normandie, which is only 4th at 856 m). A model that shortcuts to
    fame, or vaguely recalls the 1994 record-holder, mis-picks Pont de Normandie."""
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_argmax",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Russky Bridge named as having the longest main span" if passed
            else "Longest-span bridge (Russky Bridge) missing/incorrect (beware: Pont de "
                 "Normandie is the most famous bridge in this list but only 4th at 856 m)"
        ),
    }


def validate_coverage(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the SIX bridges' span figures were
    gathered.

    A bridge is credited only when BOTH its name AND its (distinct) span figure appear; the six
    figures are all distinct, so there is no cross-crediting. Deliberately NOT gated on the
    keystone — it measures whether the agent actually fanned out to all six bridges and read each
    span, the axis that separates a structured (multi-leaf) agent from a linear one that drops
    an entity.
    """
    text = _all_text(result)
    hits = [
        e["name"] for e in ENTITIES
        if re.search(e["name_rx"], text, re.IGNORECASE) and re.search(e["span_rx"], text)
    ]
    n = len(ENTITIES)
    return {
        "check": "coverage",
        "passed": len(hits) == n,
        "score": len(hits) / n,
        "reason": f"{len(hits)}/{n} bridges' spans gathered ({', '.join(hits) or 'none'})",
    }


def validate_winner_span(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: the winner's span (~1,104 m, within +/-2.5%). Short-circuits to 0 when
    the keystone is absent, so a wrong/guessed winner can't bank the value credit."""
    if not _keystone_ok(result, observability):
        return {
            "check": "winner_span",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> span value not credited",
        }
    text = _all_text(result)
    lo, hi = WINNER_SPAN * (1.0 - SPAN_TOL), WINNER_SPAN * (1.0 + SPAN_TOL)
    found = [int(m.replace(",", "")) for m in _SPAN_NUM.findall(text)]
    ok = any(lo <= v <= hi for v in found)
    return {
        "check": "winner_span",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"winner's span within {lo:.0f}–{hi:.0f} m present" if ok
            else f"no span in {lo:.0f}–{hi:.0f} m (expected ~{WINNER_SPAN} m); "
                 f"saw {found or 'none'}"
        ),
    }


def validate_citation(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: cites the bridge pages. Short-circuits to 0 when the keystone is
    absent."""
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
        "reason": f"{cited}/{n} bridge pages cited",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_argmax,
        validate_coverage,
        validate_winner_span,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, as in 062/064.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    SIX INDEPENDENT parallel leaves: for each of the six GIVEN bridges, one leaf reads ONE
    atomic figure — that bridge's longest main span in metres. The argmax lives only in the
    aggregation step, which is forced to write out every bridge's span before concluding, so the
    cheap executor never shortcuts to the most famous bridge and a diverse-grounding reranker can
    re-derive the comparison. Encodes STRUCTURE only: it names the six GIVEN bridges and their
    countries, but leaks no span figure and never states which bridge wins.
    """
    leaves: List[Dict[str, Any]] = []
    for e in ENTITIES:
        leaves.append({
            # id keyed on the GIVEN bridge (never the argmax answer), so the scaffold leaks nothing.
            "id": e["key"],
            "instruction": (
                f"Open the Wikipedia page for the bridge {e['name']} ({e['country']}) and read, "
                "directly from the infobox, its LONGEST SPAN in metres — the length of the "
                "bridge's central span between its main supports (NOT the total bridge length, "
                "which includes approach spans). Report ONLY that single span figure (in metres) "
                "and the source URL. Do not guess from memory."
            ),
            "expect": "LONGEST SPAN in metres (a single figure) -- source URL",
            "depends_on": [],
        })
    return {
        "leaves": leaves,
        # Deterministic composition: the executor compares the six gathered spans in Python and
        # renders the winner plus the full per-bridge breakdown itself (zero extra LLM calls) —
        # a live cross-model check found a weak model (llama3.2:3b) correctly extracting all six
        # real span figures (Russky 1104m plainly visible in its own list) and then still
        # concluding a SMALLER entry (Rio-Antirrio, 560m) was "the longest" -- the exact
        # compute-right/conclude-wrong failure this mechanism exists to eliminate.
        # NOTE (found via adversarial stress-testing before wiring, not assumed safe): unlike test
        # 062's identical `argmax` wiring, this task's own `value_label` MUST NOT share a word
        # with `_SUP` (this file's superlative-trigger vocabulary, e.g. "longest"/"highest") --
        # the composer's breakdown row format ("<Name>: <value_label>=<value>") would otherwise
        # trivially satisfy `_RUSSKY_WINS`'s proximity check for EVERY entity's own row
        # (including a non-winner's), regardless of which bridge the lead sentence actually names.
        # Reproduced live with "longest main span" (wrongly credited a runner-up with zero missing
        # leaves); "main span" (no `_SUP` word) verified clean on both the degraded-leaf case and
        # the full correct-render case before choosing it here.
        "agg_mode": "computed",
        "composition": {
            "op": "argmax",
            "answer_noun": "bridge",
            "value_label": "main span",
            "unit": "m",
            "items": [{"leaf": e["key"], "label": e["name"], "type": "number"} for e in ENTITIES],
        },
        "aggregation": (
            "You now have, for each of the six bridges, its longest main span in metres (with a "
            "source URL). Before drawing any conclusion, write out each bridge's span on its own "
            "line in the form '<bridge name>: <span> m'. Write all six lines FIRST, then "
            "comparing those six figures, state which SINGLE bridge has the LONGEST main span — "
            "that bridge's name is the keystone answer. NOTE: the bridge with the longest main "
            "span is NOT necessarily the most famous bridge in the list; compare the span figures "
            "themselves, not how well-known each bridge is. Report (a) the bridge with the "
            "longest main span and its span in metres, stating explicitly that it has the longest "
            "span, (b) the full list bridge -> span in metres for all six, and (c) cite each "
            "bridge's source URL."
        ),
    }
